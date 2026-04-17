"""Runtime powertrain model for FSAE EV drivetrain.

Implements speed/torque/force/power relationships for a PMSM motor
with a single-speed gear reduction.  The model handles:
- Motor RPM from vehicle speed (and inverse)
- Torque capability vs RPM (flat + field-weakening + above-max cutoff)
- Wheel torque and tractive force through gear ratio and efficiency
- Drive and regenerative braking force from throttle/brake demand
- Electrical power drawn from (or returned to) the battery pack,
  including passive back-EMF rectification above K_e*omega > V_pack
- Pack current from electrical power and instantaneous pack voltage
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from fsae_sim.vehicle.powertrain import PowertrainConfig

if TYPE_CHECKING:
    from fsae_sim.vehicle.motor_efficiency import MotorEfficiencyMap


@dataclass(frozen=True)
class LVCUCommandState:
    """Diagnostic state returned by :meth:`PowertrainModel.lvcu_torque_command`.

    Attributes:
        torque_nm: Commanded motor torque in Nm (>= 0).
        bse_latched: True if the BSE (brake+throttle interlock) is active
            and therefore torque has been forced to zero.
        apps_mismatch: True if the APPS (TPS1/TPS2) mismatch threshold was
            exceeded (diagnostic only — does not gate torque here because
            the caller decides what to do with the flag).
        startup_gate_active: True if the LVCU startup gate
            (torque_request < 5 Nm && motor_speed < 500 RPM) applies.
    """
    torque_nm: float
    bse_latched: bool
    apps_mismatch: bool
    startup_gate_active: bool


class PowertrainModel:
    """Runtime powertrain model for FSAE EV drivetrain.

    All public methods accept and return scalar floats and are designed to
    be called inside a simulation time-step loop.  Vectorised batch usage
    is intentionally delegated to callers via NumPy broadcasting over the
    scalar interface.

    When a ``MotorEfficiencyMap`` is provided, the motor+inverter efficiency
    varies with RPM and torque (from EMRAX 228 characterization data),
    combined with a fixed gearbox efficiency.  Otherwise, falls back to
    the fixed ``config.drivetrain_efficiency`` scalar.

    Args:
        config: Frozen ``PowertrainConfig`` dataclass with motor, inverter,
            LVCU, and drivetrain parameters.
        efficiency_map: Optional 2D efficiency lookup for operating-point-
            dependent motor+inverter efficiency.
    """

    TIRE_RADIUS_M: float = 0.2042  # Hoosier 16x7.5-10 LC0, UNLOADED_RADIUS from .tir file

    # Gearbox mechanical efficiency — the only loss between motor shaft
    # and wheel.  Motor+inverter efficiency is handled separately in
    # electrical_power() via the efficiency map.
    _GEARBOX_EFFICIENCY: float = 0.97

    # C3: motor-vs-regen inverter efficiency asymmetry.
    # The Cascadia CM200DX datasheet reports a small (~1-2 pp) lower
    # efficiency for regenerative (IGBT body-diode conduction with
    # synchronous rectification) vs motoring operation at the same
    # operating point.  We apply this as a small offset only — not the
    # 15% "factor" that the prior code used, which double-counted the
    # motor+inverter losses already encoded in the MotorEfficiencyMap.
    # Source: Cascadia CM200DX application note, figs. 7-9 (motoring vs
    # regen efficiency envelopes at 400 V DC, 10-200 A phase current).
    _REGEN_EFFICIENCY_OFFSET_PP: float = 0.02  # 2 percentage points

    # C2: PMSM back-EMF constant K_e sourced from
    # ``PowertrainConfig.motor_back_emf_constant_v_s_per_rad`` (default
    # 0.045 V/(rad/s) for the EMRAX 228 MV LC as used on CT-16EV).
    # Access via ``self.config`` — no class constant so operator tuning
    # propagates correctly.

    # APPS mismatch trip threshold, from firmware
    # (`tps_dist_error = fabs(tps1 - tps2) > APPS_TRIP_PERCENT`).  The
    # firmware uses 10% in LVCU Code.txt (via the
    # `APPS_TRIP_PERCENT` macro).  Kept as an explicit class constant
    # so tests and docs do not drift from firmware.
    _APPS_TRIP_FRACTION: float = 0.1

    def __init__(
        self,
        config: PowertrainConfig,
        efficiency_map: "MotorEfficiencyMap | None" = None,
    ) -> None:
        self.config = config
        self._efficiency_map = efficiency_map

        # Pre-compute constants used in every call.  The effective torque
        # ceiling is inverter ∧ LVCU ∧ (optional operational safety cap).
        hard_ceiling = min(
            config.torque_limit_inverter_nm,
            config.torque_limit_lvcu_nm,
        )
        if config.safety_torque_cap_nm is not None:
            hard_ceiling = min(hard_ceiling, config.safety_torque_cap_nm)
        self._torque_limit_nm: float = hard_ceiling
        self._rad_per_s_per_rpm: float = math.pi / 30.0  # 2*pi/60

        # Regen efficiency fallback (no motor map): use drivetrain_eff
        # minus a small motoring-vs-regen offset (see C3).
        self._regen_efficiency_fallback: float = max(
            0.0, config.drivetrain_efficiency - self._REGEN_EFFICIENCY_OFFSET_PP
        )

    def _get_efficiency(self, motor_rpm: float, motor_torque_nm: float) -> float:
        """Motor + inverter efficiency at the given operating point.

        Used by ``electrical_power()`` to convert motor shaft power to
        battery power.  Gearbox efficiency is excluded here because the
        gearbox is downstream of the motor shaft — its friction reduces
        wheel torque (handled in ``wheel_torque()``) but does not
        increase electrical demand from the battery.

        Uses the motor efficiency map when available, otherwise falls
        back to the fixed ``config.drivetrain_efficiency``.
        """
        if self._efficiency_map is not None:
            return self._efficiency_map.efficiency(motor_rpm, motor_torque_nm)
        return self.config.drivetrain_efficiency

    # ------------------------------------------------------------------
    # Speed / RPM conversion
    # ------------------------------------------------------------------

    def motor_rpm_from_speed(self, vehicle_speed_ms: float) -> float:
        """Convert vehicle speed (m/s) to motor shaft RPM.

        Derivation:
            wheel_angular_velocity [rad/s] = v / r
            wheel_rpm               [rpm]  = (v / r) * 60 / (2*pi)
            motor_rpm               [rpm]  = wheel_rpm * gear_ratio

        Args:
            vehicle_speed_ms: Vehicle longitudinal speed in m/s.

        Returns:
            Motor shaft speed in RPM.  Returns 0.0 for negative speed
            inputs (reversing is not modelled).
        """
        speed = max(0.0, vehicle_speed_ms)
        wheel_rpm = (speed / self.TIRE_RADIUS_M) * 60.0 / (2.0 * math.pi)
        return wheel_rpm * self.config.gear_ratio

    def speed_from_motor_rpm(self, motor_rpm: float) -> float:
        """Convert motor shaft RPM to vehicle speed (m/s).

        Inverse of ``motor_rpm_from_speed``.

        Args:
            motor_rpm: Motor shaft speed in RPM.

        Returns:
            Vehicle longitudinal speed in m/s.  Returns 0.0 for negative
            RPM inputs.
        """
        rpm = max(0.0, motor_rpm)
        wheel_rpm = rpm / self.config.gear_ratio
        return wheel_rpm * self.TIRE_RADIUS_M * 2.0 * math.pi / 60.0

    # ------------------------------------------------------------------
    # Torque capability
    # ------------------------------------------------------------------
    #
    # D-15: single, unified field-weakening model.
    # Previously there were two independent models — a
    # ``torque_delivery_factor(rpm)`` multiplier applied in the replay
    # branch, and a separate linear-in-RPM taper inside
    # ``max_motor_torque``.  Replay already carries the *measured*
    # delivered torque, so applying an extra derate there
    # double-counted field weakening; and the linear taper inside
    # ``max_motor_torque`` is not the right shape for a PMSM anyway —
    # above the constant-torque region the motor operates at
    # ~constant power, i.e. T(ω) = P_max / ω, a hyperbolic curve.
    #
    # New model:
    #   T_max(rpm) = T_limit                         for rpm ≤ brake_speed_rpm
    #              = min(T_limit, P_max / ω(rpm))   for brake_speed_rpm < rpm ≤ motor_speed_max_rpm
    #              = 0                                for rpm > motor_speed_max_rpm
    # where P_max = T_limit × ω(brake_speed_rpm) — the mechanical
    # power the motor delivers at the corner of the envelope.
    # ``torque_delivery_factor`` is deleted; callers in the replay
    # branch use the measured torque directly.

    def max_motor_torque(self, motor_rpm: float) -> float:
        """Maximum motor output torque at given RPM (Nm).

        PMSM with constant-torque region and constant-power
        field-weakening region:

        1. **Constant-torque** (0 ≤ rpm ≤ brake_speed_rpm):
           Full torque = min(inverter_limit, lvcu_limit).

        2. **Constant-power field-weakening**
           (brake_speed_rpm < rpm ≤ motor_speed_max_rpm):
           T(ω) = P_max / ω, clamped to never exceed the hard
           torque ceiling. P_max is the mechanical power at the
           corner of the envelope (T_limit × ω(brake_speed_rpm)).

        3. **Over-speed** (rpm > motor_speed_max_rpm):
           Zero torque — motor cannot operate above max electrical
           frequency.

        Args:
            motor_rpm: Motor shaft speed in RPM.

        Returns:
            Maximum available motor torque in Nm (≥ 0).
        """
        rpm = max(0.0, motor_rpm)

        if rpm <= self.config.brake_speed_rpm:
            return self._torque_limit_nm

        if rpm <= self.config.motor_speed_max_rpm:
            omega = rpm * self._rad_per_s_per_rpm
            omega_corner = self.config.brake_speed_rpm * self._rad_per_s_per_rpm
            p_max = self._torque_limit_nm * omega_corner
            return min(self._torque_limit_nm, p_max / omega)

        # Above maximum RPM
        return 0.0

    def lvcu_torque_command(
        self,
        pedal_pct: float,
        motor_rpm: float,
        bms_current_limit_a: float,
        *,
        brake_pressed: bool = False,
        prior_bse_latched: bool = False,
        tps1: Optional[float] = None,
        tps2: Optional[float] = None,
        return_state: bool = False,
    ) -> Union[float, LVCUCommandState]:
        """Motor torque command replicating the real LVCU firmware.

        Faithfully implements the torque command chain from LVCU Code.txt:
        pedal -> tmap_lut (dead zone remap) -> torque_lut (power-limited
        ceiling) -> inverter clamp -> BSE/APPS/startup interlocks.

        BSE (S13): if ``brake_pressed`` and pedal >= 10%, latch BSE and
        zero the torque request. Once latched, BSE clears only when pedal
        falls below 5%. Callers must thread ``prior_bse_latched`` across
        consecutive calls to preserve the latch state — the model itself
        is stateless.

        BMS safety offset (S14): ``effective_limit = max(0,
        bms_current_limit_a - lvcu_bms_current_offset_a)`` before the
        power-divide, matching firmware line 151.

        NF-41: pedal-span divide is guarded by ``max(..., 1e-6)`` and the
        config's ``__post_init__`` rejects span < 0.01 so this path cannot
        silently amplify noise.

        Args:
            pedal_pct: Raw pedal position in [0.0, 1.0] (TPS_combined).
            motor_rpm: Motor shaft speed in RPM.
            bms_current_limit_a: Raw BMS discharge current limit in A
                (before the LVCU's `-3` safety offset).
            brake_pressed: Brake pedal above BPS setpoint (S13).
            prior_bse_latched: BSE latch state carried from the previous
                sim step. Required to correctly model the hysteresis
                (latch at >= 10% with brake, clear at < 5%).
            tps1, tps2: Individual TPS sensor readings in [0, 1]. If
                either is ``None``, APPS-mismatch diagnostic is false.
            return_state: When True, return an :class:`LVCUCommandState`
                with diagnostic flags; otherwise return a bare float
                (backwards-compatible).

        Returns:
            Commanded motor torque in Nm (>= 0), or an
            :class:`LVCUCommandState` if ``return_state=True``.
        """
        cfg = self.config

        # 1. tmap_lut: dead zone remap [V_MIN, V_MAX] -> [0, 1].
        # NF-41: guard the divide so a pathological config never crashes.
        pedal_clamped = max(cfg.lvcu_pedal_deadzone_low,
                           min(pedal_pct, cfg.lvcu_pedal_deadzone_high))
        span = max(
            cfg.lvcu_pedal_deadzone_high - cfg.lvcu_pedal_deadzone_low,
            1e-6,
        )
        pedal_remapped = (pedal_clamped - cfg.lvcu_pedal_deadzone_low) / span

        # 2. S14: subtract the BMS safety offset before the power divide.
        bms_limit_effective = max(
            0.0, bms_current_limit_a - cfg.lvcu_bms_current_offset_a
        )

        # 3. torque_lut: power-limited torque ceiling.
        omega_term = max(cfg.lvcu_omega_floor, motor_rpm * cfg.lvcu_rpm_scale)
        power_ceiling_nm = cfg.lvcu_power_constant * bms_limit_effective / omega_term

        # LVCU torque limit (software cap)
        torque_ceiling_nm = min(cfg.torque_limit_lvcu_nm, power_ceiling_nm)

        # Overspeed override
        if motor_rpm >= cfg.lvcu_overspeed_rpm:
            torque_ceiling_nm = cfg.lvcu_overspeed_torque_nm

        # Inverter hardware limit (independent clamp)
        torque_ceiling_nm = min(torque_ceiling_nm, cfg.torque_limit_inverter_nm)

        # Operational safety cap (optional)
        if cfg.safety_torque_cap_nm is not None:
            torque_ceiling_nm = min(torque_ceiling_nm, cfg.safety_torque_cap_nm)

        # 4. Final command: remapped pedal * clamped ceiling.
        torque_request = pedal_remapped * torque_ceiling_nm

        # 5. S13: BSE latch. Firmware:
        #       if(!bse_error) bse_error = brake_pressed && tps_combined >= 0.1;
        #       else            bse_error = tps_combined >= 0.05;
        # We replicate the two-state hysteresis using the caller-supplied
        # `prior_bse_latched` to make the call sequence explicit.
        # Firmware clears BSE when tps_combined < 0.05 (strict), so the
        # clear condition here mirrors that with `> 0.05` on the retain
        # side.  Using `>= 0.05` left the latch stuck at exactly 5%.
        if prior_bse_latched:
            bse_latched = pedal_pct > 0.05
        else:
            bse_latched = brake_pressed and (pedal_pct >= 0.10)
        if bse_latched:
            torque_request = 0.0

        # 6. APPS mismatch — gate torque to zero per firmware.
        # LVCU Code.txt trips torque when |tps1 − tps2| > APPS_TRIP_PERCENT.
        # Previously this was "caller decides" and no caller acted, so the
        # fault behaviour was silently missing from sim.
        if tps1 is not None and tps2 is not None:
            apps_mismatch = abs(tps1 - tps2) > self._APPS_TRIP_FRACTION
            if apps_mismatch:
                torque_request = 0.0
        else:
            apps_mismatch = False

        # 7. Startup gate diagnostic.
        startup_gate = (torque_request < 5.0) and (motor_rpm < 500.0)

        if return_state:
            return LVCUCommandState(
                torque_nm=torque_request,
                bse_latched=bse_latched,
                apps_mismatch=apps_mismatch,
                startup_gate_active=startup_gate,
            )
        return torque_request

    def lvcu_torque_ceiling(
        self, motor_rpm: float, bms_current_limit_a: float,
    ) -> float:
        """LVCU power-limited torque ceiling without dead zone remap.

        Returns the maximum torque the LVCU would allow at the given RPM
        and BMS current limit. Use this with a torque fraction (0-1) that
        has already been through the real LVCU — avoids double-processing
        the dead zone remap.

        Args:
            motor_rpm: Motor shaft speed in RPM.
            bms_current_limit_a: Raw BMS discharge current limit in A
                (the LVCU `-3` offset is applied inside).

        Returns:
            Torque ceiling in Nm.
        """
        cfg = self.config
        # S14: apply the BMS safety offset here too.
        bms_limit_effective = max(
            0.0, bms_current_limit_a - cfg.lvcu_bms_current_offset_a
        )
        omega_term = max(cfg.lvcu_omega_floor, motor_rpm * cfg.lvcu_rpm_scale)
        power_ceiling_nm = cfg.lvcu_power_constant * bms_limit_effective / omega_term
        torque_ceiling_nm = min(cfg.torque_limit_lvcu_nm, power_ceiling_nm)
        if motor_rpm >= cfg.lvcu_overspeed_rpm:
            torque_ceiling_nm = cfg.lvcu_overspeed_torque_nm
        torque_ceiling_nm = min(torque_ceiling_nm, cfg.torque_limit_inverter_nm)
        if cfg.safety_torque_cap_nm is not None:
            torque_ceiling_nm = min(torque_ceiling_nm, cfg.safety_torque_cap_nm)
        return torque_ceiling_nm

    # ------------------------------------------------------------------
    # Torque and force through drivetrain
    # ------------------------------------------------------------------

    def wheel_torque(self, motor_torque_nm: float) -> float:
        """Wheel torque from motor torque through gear reduction and friction.

        Only gearbox friction is applied here.  Motor+inverter efficiency
        affects electrical power (handled in ``electrical_power()``), not
        the mechanical torque delivered to the wheels.

        Args:
            motor_torque_nm: Motor shaft torque in Nm.

        Returns:
            Wheel hub torque in Nm.  Positive = driving, negative = braking
            (regen sign is preserved).
        """
        return motor_torque_nm * self.config.gear_ratio * self._GEARBOX_EFFICIENCY

    def wheel_force(self, motor_torque_nm: float) -> float:
        """Tractive force at the tire contact patch from motor torque.

        Args:
            motor_torque_nm: Motor shaft torque in Nm.

        Returns:
            Force in N at the contact patch.  Positive = forward, negative =
            rearward (regen/braking).
        """
        return self.wheel_torque(motor_torque_nm) / self.TIRE_RADIUS_M

    # ------------------------------------------------------------------
    # Drive and regen demand
    # ------------------------------------------------------------------

    def drive_force(self, throttle_pct: float, vehicle_speed_ms: float) -> float:
        """Tractive force (N) at given throttle demand and vehicle speed.

        The commanded motor torque is ``throttle_pct * max_motor_torque(rpm)``.
        Throttle is clamped to [0, 1] and speed is clamped to >= 0.

        Args:
            throttle_pct: Throttle demand in the range [0.0, 1.0].
            vehicle_speed_ms: Vehicle longitudinal speed in m/s.

        Returns:
            Forward tractive force in N (>= 0).
        """
        throttle = max(0.0, min(1.0, throttle_pct))
        rpm = self.motor_rpm_from_speed(vehicle_speed_ms)
        max_torque = self.max_motor_torque(rpm)
        commanded_torque = throttle * max_torque
        return self.wheel_force(commanded_torque)

    def regen_force(self, brake_pct: float, vehicle_speed_ms: float) -> float:
        """Regenerative braking force (N, negative = decelerating).

        Regen torque capability is limited by the same motor torque envelope
        used for driving.  The returned force is negative (opposing motion).

        S12 note on gearbox sign: in generator (regen) mode, gearbox friction
        *adds* to the retarding torque at the wheel, because the wheel must
        drive the motor through a lossy gearbox — the friction is borne by
        the car, not the motor.  So the correct transformation is

            T_wheel = T_motor * gear_ratio / η_gearbox

        (not ``* η_gearbox`` as in the motoring direction).  This makes the
        mechanical retarding force ~3% larger than a naïve multiply.  The
        electrical-energy asymmetry (recovering less than the mechanical
        input because of motor+inverter losses) is handled separately in
        ``electrical_power()``.

        Args:
            brake_pct: Regen brake demand in the range [0.0, 1.0].
            vehicle_speed_ms: Vehicle longitudinal speed in m/s.

        Returns:
            Regen braking force in N (<= 0).  Zero if speed is zero or the
            motor is above its operating range.
        """
        brake = max(0.0, min(1.0, brake_pct))
        speed = max(0.0, vehicle_speed_ms)
        if speed == 0.0:
            return 0.0

        rpm = self.motor_rpm_from_speed(speed)
        # Generator torque capability uses the same RPM-torque envelope.
        max_regen_torque = self.max_motor_torque(rpm)
        commanded_torque = brake * max_regen_torque
        # S12: divide by η_gearbox, not multiply. Gearbox friction adds
        # to the retarding torque the car feels at the contact patch.
        regen_wheel_torque = (
            commanded_torque * self.config.gear_ratio / self._GEARBOX_EFFICIENCY
        )
        return -(regen_wheel_torque / self.TIRE_RADIUS_M)

    # ------------------------------------------------------------------
    # Electrical power
    # ------------------------------------------------------------------

    # Coast-state torque threshold: below this magnitude the motor is
    # considered "not actively commanded" and the back-EMF rectifier
    # model applies (instead of the efficiency-map divide).  0.5 Nm
    # is well below the LVCU startup gate (5 Nm) and below telemetry
    # noise floor on Torque Feedback.
    _COAST_TORQUE_THRESHOLD_NM: float = 0.5

    def electrical_power(
        self,
        motor_torque_nm: float,
        motor_rpm: float,
        pack_voltage_v: float | None = None,
    ) -> float:
        """Electrical power exchanged with the battery pack (W).

        Sign convention (battery perspective):
        - **Positive** (motoring): power drawn *from* the battery.
        - **Negative** (regen): power returned *to* the battery.

        Dispatched on motor state (torque magnitude), not driver action:

        1. **Motoring** (``motor_torque_nm > COAST_THRESHOLD``):
           ``P_elec = T·ω / η(rpm, T)``.  Efficiency map (or
           drivetrain_efficiency fallback) converts mechanical shaft
           power to electrical demand.
        2. **Commanded regen** (``motor_torque_nm < -COAST_THRESHOLD``):
           ``P_elec = T·ω × η_regen(rpm, |T|)``.  Mechanical input times
           regen efficiency (losses reduce what reaches the pack).
        3. **Coast** (``|motor_torque_nm| ≤ COAST_THRESHOLD``):
           Back-EMF rectifier model.  If ``K_e·ω > V_pack`` the inverter
           body diodes conduct and current flows into the pack; otherwise
           zero current (free-wheeling).  Requires ``pack_voltage_v``; if
           None, returns 0 (no rectification).

        Args:
            motor_torque_nm: Motor shaft torque in Nm.  Positive = motoring,
                negative = generating (commanded regen).
            motor_rpm: Motor shaft speed in RPM.
            pack_voltage_v: Instantaneous pack terminal voltage (V).
                Required for the back-EMF rectifier branch; optional
                otherwise (backwards-compat).

        Returns:
            Electrical power in W (positive = battery discharge).
        """
        if motor_rpm <= 0.0:
            return 0.0

        omega = motor_rpm * self._rad_per_s_per_rpm  # rad/s

        # --- Coast branch: passive back-EMF rectification ---
        if abs(motor_torque_nm) <= self._COAST_TORQUE_THRESHOLD_NM:
            if pack_voltage_v is None or pack_voltage_v <= 0.0:
                return 0.0
            v_bemf = self.config.motor_back_emf_constant_v_s_per_rad * omega
            if v_bemf <= pack_voltage_v:
                # Body diodes reverse-biased → no current flow.
                return 0.0
            # Simplest honest rectifier: assume negligible source
            # impedance → current limited only by the measured
            # coast operating point.  We model the overvoltage as
            # driving current through the battery's own internal
            # resistance, but without a calibrated R we conservatively
            # return the power associated with clamping V_bemf to
            # V_pack — i.e. P = V_pack * I where the inverter sinks
            # enough current to hold V_bemf = V_pack.  Without the
            # current limit we can only give an upper bound; return
            # a small pack-credit scaled by the overvoltage ratio.
            #
            # This branch is not exercised under the Michigan stint
            # (V_bemf < V_pack always at realistic RPMs; see class
            # constant comment).  Kept honest-and-simple until a
            # validation point demonstrates it fires.
            overvoltage = v_bemf - pack_voltage_v
            # Use a nominal per-phase resistance of 0.05 Ω
            # (EMRAX 228 phase resistance order of magnitude) so
            # I = overvoltage / R_phase, P = V_pack * I.
            R_phase = 0.05
            i_regen = overvoltage / R_phase
            return -pack_voltage_v * i_regen

        p_mechanical = motor_torque_nm * omega  # W

        if p_mechanical > 0.0:
            # Motoring: battery must supply more than mechanical output.
            eta = self._get_efficiency(motor_rpm, motor_torque_nm)
            if eta > 0.0:
                return p_mechanical / eta
            return 0.0

        # Commanded regen (p_mechanical < 0).
        # C3: do NOT multiply the map efficiency by _REGEN_EFFICIENCY_FACTOR
        # (the old factor double-counted motor+inverter losses already
        # encoded in the MotorEfficiencyMap).  The motor-vs-regen
        # asymmetry is a small (~1-2 pp) offset — apply it on BOTH the
        # map and fallback paths so the two are consistent and the
        # "motor map off" toggle doesn't silently change regen
        # accounting by 2 pp.
        if self._efficiency_map is not None:
            eta_motoring = self._efficiency_map.efficiency(
                motor_rpm, abs(motor_torque_nm)
            )
            eta_regen = max(0.0, eta_motoring - self._REGEN_EFFICIENCY_OFFSET_PP)
        else:
            eta_regen = self._regen_efficiency_fallback
        return p_mechanical * eta_regen

    def pack_current(self, electrical_power_w: float, pack_voltage_v: float) -> float:
        """Pack current from electrical power and instantaneous pack voltage.

        Uses P = V * I.  Sign convention matches ``electrical_power``:
        positive current = discharging the pack.

        Args:
            electrical_power_w: Electrical power in W (positive = motoring).
            pack_voltage_v: Pack terminal voltage in V.  Must be > 0.

        Returns:
            Pack current in A (positive = discharge).

        Raises:
            ValueError: If ``pack_voltage_v`` is zero or negative.
        """
        if pack_voltage_v <= 0.0:
            raise ValueError(
                f"pack_voltage_v must be positive, got {pack_voltage_v!r}"
            )
        return electrical_power_w / pack_voltage_v
