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
    from fsae_sim.vehicle.inverter_delivery import InverterDeliveryMap
    from fsae_sim.vehicle.motor_efficiency import MotorEfficiencyMap
    from fsae_sim.vehicle.tire_model import PacejkaTireModel


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
        inverter_delivery_map: "InverterDeliveryMap | None" = None,
        tire_model: "PacejkaTireModel | None" = None,
    ) -> None:
        self.config = config
        self._efficiency_map = efficiency_map
        self._inverter_delivery_map = inverter_delivery_map
        self._tire_model = tire_model
        self.rolling_radius_m = float(config.rolling_radius_m)

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
    # Rolling radius (load-dependent when tire model attached)
    # ------------------------------------------------------------------

    def rolling_radius_for(self, fz: float | None) -> float:
        """Effective rolling radius (m) for a given tire normal load.

        When a Pacejka tire model is attached, returns
        ``tire_model.loaded_radius(fz)`` — the static loaded radius from
        PAC2002 vertical stiffness (linear spring formulation).  When no
        tire model is attached, or ``fz`` is ``None``, returns the
        configured ``rolling_radius_m`` (the static / unloaded value).

        The ``fz`` argument is the mean per-tire normal load (N), not the
        total vertical force on the car.  Callers are responsible for
        averaging across the four wheels (TUMFTM ``laptime-simulation``
        precedent — see plan U4 default).

        References:
            Pacejka, *Tyre and Vehicle Dynamics* 3e (2012) §1.3, §4.3.6.
            Adams Tire 2018 PAC2002 docs (Stackpole/Hoosier export).
        """
        if fz is None or self._tire_model is None:
            return self.rolling_radius_m
        return float(self._tire_model.loaded_radius(float(fz)))

    # ------------------------------------------------------------------
    # Speed / RPM conversion
    # ------------------------------------------------------------------

    def motor_rpm_from_speed(
        self, vehicle_speed_ms: float, *, fz: float | None = None,
    ) -> float:
        """Convert vehicle speed (m/s) to motor shaft RPM.

        Derivation:
            wheel_angular_velocity [rad/s] = v / r
            wheel_rpm               [rpm]  = (v / r) * 60 / (2*pi)
            motor_rpm               [rpm]  = wheel_rpm * gear_ratio

        When ``fz`` (mean per-tire normal load, N) is provided and a
        tire model is attached, the load-dependent rolling radius from
        ``rolling_radius_for(fz)`` is used in place of the configured
        static radius.  This captures the ~3-4 % rolling-radius
        reduction under FSAE-typical tire loads (issue 18).

        Args:
            vehicle_speed_ms: Vehicle longitudinal speed in m/s.
            fz: Optional mean per-tire normal load (N).  When ``None``
                (default, for back-compat), the static config radius
                is used.

        Returns:
            Motor shaft speed in RPM.  Returns 0.0 for negative speed
            inputs (reversing is not modelled).
        """
        speed = max(0.0, vehicle_speed_ms)
        r = self.rolling_radius_for(fz)
        wheel_rpm = (speed / r) * 60.0 / (2.0 * math.pi)
        return wheel_rpm * self.config.gear_ratio

    def speed_from_motor_rpm(
        self, motor_rpm: float, *, fz: float | None = None,
    ) -> float:
        """Convert motor shaft RPM to vehicle speed (m/s).

        Inverse of ``motor_rpm_from_speed``.

        Args:
            motor_rpm: Motor shaft speed in RPM.
            fz: Optional mean per-tire normal load (N).  See
                :meth:`motor_rpm_from_speed` for usage.

        Returns:
            Vehicle longitudinal speed in m/s.  Returns 0.0 for negative
            RPM inputs.
        """
        rpm = max(0.0, motor_rpm)
        r = self.rolling_radius_for(fz)
        wheel_rpm = rpm / self.config.gear_ratio
        return wheel_rpm * r * 2.0 * math.pi / 60.0

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

    def pedal_to_torque_request(
        self, pedal_remapped: float, motor_rpm: float, bms_current_limit_a: float,
    ) -> float:
        """Firmware-faithful LVCU torque request for an already-remapped pedal.

        Mirrors the LVCU's ``torque_lut(tps)`` step exactly:
        - Power-cap from BMS current and motor speed
        - LVCU torque limit (firmware setpoint, e.g. 220 Nm)
        - Overspeed override
        - Optional safety cap

        Crucially, this does NOT apply the inverter's hardware torque cap.
        The real LVCU sends the request as computed; the inverter clips
        it downstream during delivery (modeled by
        :meth:`apply_inverter_delivery`). Capping at the LVCU layer
        produces a request that's too low whenever the LVCU was over-
        requesting on purpose — the recorded telemetry shows real LVCU
        Torque Req routinely exceeds 85 Nm at high pedal because the
        firmware doesn't know about the inverter's IQ setting.

        ``pedal_remapped`` is expected to be the firmware's
        ``tmap_lut(tps_combined)`` output — i.e. the AiM "Throttle Pos"
        channel, which is already deadzone-remapped per LVCU Code.txt
        line 499 (``TxData[7] = (int)(tmap_lut(tps_combined) * 100)``).
        Use this for pedal-replay strategies that play back recorded
        post-deadzone pedal values.
        """
        cfg = self.config
        bms_limit_effective = max(
            0.0, bms_current_limit_a - cfg.lvcu_bms_current_offset_a
        )
        omega_term = max(cfg.lvcu_omega_floor, motor_rpm * cfg.lvcu_rpm_scale)
        power_ceiling_nm = cfg.lvcu_power_constant * bms_limit_effective / omega_term
        torque_ceiling_nm = min(cfg.torque_limit_lvcu_nm, power_ceiling_nm)
        if motor_rpm >= cfg.lvcu_overspeed_rpm:
            torque_ceiling_nm = cfg.lvcu_overspeed_torque_nm
        # No inverter cap here — see docstring.
        if cfg.safety_torque_cap_nm is not None:
            torque_ceiling_nm = min(torque_ceiling_nm, cfg.safety_torque_cap_nm)
        return max(0.0, float(pedal_remapped)) * torque_ceiling_nm

    # ------------------------------------------------------------------
    # Inverter delivery
    # ------------------------------------------------------------------

    def apply_inverter_delivery(
        self,
        motor_rpm: float,
        lvcu_command_nm: float,
        *,
        fz: float | None = None,
    ) -> float:
        """Translate an LVCU torque request into delivered shaft torque.

        The Cascadia inverter does not perfectly follow the LVCU's
        command — at very low RPM (transient ramp) and above ~2800 RPM
        (field weakening), delivered torque sits below the request. The
        effect is captured by an :class:`InverterDeliveryMap` calibrated
        from telemetry.

        When no map is attached, returns the command unchanged so the
        powertrain remains backward-compatible. Negative commands
        (regen) pass through untouched; the map only models motoring.

        Note: ``fz`` is accepted on the public signature so callers can
        route a load reference through the powertrain pipeline (motor
        RPM -> command -> delivered torque -> wheel force).  The inverter
        map itself is rolling-radius-independent (it is a (rpm, torque)
        -> torque interpolation), but its caller chain becomes
        load-aware once the kwarg is in place.
        """
        del fz  # accepted for callsite uniformity; map itself is radius-independent
        if lvcu_command_nm <= 0.0 or self._inverter_delivery_map is None:
            return lvcu_command_nm
        return self._inverter_delivery_map.delivered_torque(
            motor_rpm, lvcu_command_nm,
        )

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

    def wheel_force(
        self, motor_torque_nm: float, *, fz: float | None = None,
    ) -> float:
        """Tractive force at the tire contact patch from motor torque.

        Args:
            motor_torque_nm: Motor shaft torque in Nm.
            fz: Optional mean per-tire normal load (N).  When provided
                with a tire model attached, the load-dependent rolling
                radius is used in place of the static config radius.

        Returns:
            Force in N at the contact patch.  Positive = forward, negative =
            rearward (regen/braking).
        """
        r = self.rolling_radius_for(fz)
        return self.wheel_torque(motor_torque_nm) / r

    def motor_torque_from_wheel_force(
        self, wheel_force_n: float, *, fz: float | None = None,
    ) -> float:
        """Inverse of :meth:`wheel_force` for a realized contact-patch force."""
        denom = self.config.gear_ratio * self._GEARBOX_EFFICIENCY
        if denom <= 0.0:
            return 0.0
        r = self.rolling_radius_for(fz)
        return wheel_force_n * r / denom

    # ------------------------------------------------------------------
    # Drive and regen demand
    # ------------------------------------------------------------------

    def drive_force(
        self,
        throttle_pct: float,
        vehicle_speed_ms: float,
        *,
        fz: float | None = None,
    ) -> float:
        """Tractive force (N) at given throttle demand and vehicle speed.

        The commanded motor torque is ``throttle_pct * max_motor_torque(rpm)``.
        Throttle is clamped to [0, 1] and speed is clamped to >= 0.

        Args:
            throttle_pct: Throttle demand in the range [0.0, 1.0].
            vehicle_speed_ms: Vehicle longitudinal speed in m/s.
            fz: Optional mean per-tire normal load (N).  Threads through
                ``motor_rpm_from_speed`` and ``wheel_force`` for
                load-dependent rolling radius.

        Returns:
            Forward tractive force in N (>= 0).
        """
        throttle = max(0.0, min(1.0, throttle_pct))
        rpm = self.motor_rpm_from_speed(vehicle_speed_ms, fz=fz)
        max_torque = self.max_motor_torque(rpm)
        commanded_torque = throttle * max_torque
        return self.wheel_force(commanded_torque, fz=fz)

    def regen_force(
        self,
        brake_pct: float,
        vehicle_speed_ms: float,
        *,
        fz: float | None = None,
    ) -> float:
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

        rpm = self.motor_rpm_from_speed(speed, fz=fz)
        # Generator torque capability uses the same RPM-torque envelope.
        max_regen_torque = self.max_motor_torque(rpm)
        commanded_torque = brake * max_regen_torque
        # S12: divide by η_gearbox, not multiply. Gearbox friction adds
        # to the retarding torque the car feels at the contact patch.
        regen_wheel_torque = (
            commanded_torque * self.config.gear_ratio / self._GEARBOX_EFFICIENCY
        )
        r = self.rolling_radius_for(fz)
        return -(regen_wheel_torque / r)

    # ------------------------------------------------------------------
    # Electrical power
    # ------------------------------------------------------------------

    # Coast-state torque threshold: below this magnitude the motor is
    # considered "not actively commanded" and the back-EMF rectifier
    # model applies (instead of the efficiency-map divide).  0.5 Nm
    # is well below the LVCU startup gate (5 Nm) and below telemetry
    # noise floor on Torque Feedback.
    _COAST_TORQUE_THRESHOLD_NM: float = 0.5

    # Per-phase resistance for the back-EMF rectifier guard branch.
    # EMRAX 228 phase resistance order-of-magnitude.
    _BEMF_PHASE_RESISTANCE_OHM: float = 0.05

    def _iron_loss_w(self, omega_m_rad_s: float) -> float:
        """Stator iron loss = hysteresis + eddy current vs electrical freq.

        ``P_iron = k_h * f_e + k_e * f_e^2`` with
        ``f_e = pole_pairs * omega_m / (2*pi)`` (Pyrhonen §3.6,
        Hanselman §10.5). Hysteresis dominates at low f_e, eddy at high.

        Args:
            omega_m_rad_s: Motor mechanical angular velocity (rad/s).

        Returns:
            Iron-loss power dissipated in stator (W, >= 0).
        """
        coast = self.config.coast_loss
        f_e = coast.pole_pairs * omega_m_rad_s / (2.0 * math.pi)
        return coast.k_h_w_per_hz * f_e + coast.k_e_w_per_hz2 * f_e * f_e

    def _windage_loss_w(self, omega_m_rad_s: float) -> float:
        """Windage + bearing drag, ``P = a*omega + b*omega^2``.

        Linear term covers viscous bearing drag, quadratic term
        air-windage between rotor and stator (Hanselman §10.5,
        Pyrhonen §3.7).
        """
        coast = self.config.coast_loss
        return (
            coast.a_w_w_per_rad_s * omega_m_rad_s
            + coast.b_w_w_per_rad_s2 * omega_m_rad_s * omega_m_rad_s
        )

    def _pwm_switch_loss_w(self, pack_voltage_v: float) -> float:
        """PWM gate-drive overhead at zero phase current.

        Inverter switches at f_sw (~8 kHz on CM200DX) even at zero
        commanded torque to maintain Iq=Id=0 closed-loop control;
        gate-drive energy ``E_g`` per device dominates.
        ``P_pwm ≈ pwm_overhead_w * V_pack / V_nom`` (linear V_pack
        scaling per Infineon AN2008-03; the gate-charge → P_pwm
        path scales linearly with bus voltage at constant f_sw).
        """
        coast = self.config.coast_loss
        return coast.pwm_overhead_w * pack_voltage_v / coast.pack_voltage_nominal_v

    def _bemf_rectify_guard_w(
        self, omega_m_rad_s: float, pack_voltage_v: float,
    ) -> float:
        """Passive body-diode rectification guard.

        Only fires when ``V_bemf = K_e * omega_m > V_pack``. On CT-16EV
        at any sustained operating point this branch is dormant
        (``K_e=0.045 V/(rad/s)``, ``omega < 700 rad/s`` → V_bemf < 32 V
        ≪ V_pack ~380 V). Kept as a sanity guard so a future high-K_e
        motor or low-V_pack stint exposes the physics naturally.

        Returns:
            0.0 when V_bemf <= V_pack (the normal operating regime on
            CT-16EV at any RPM with a healthy pack).

            A negative value (regen / pack-charging) when V_bemf
            exceeds V_pack: the inverter body diodes conduct and the
            pack absorbs current. Sign matches the rest of
            ``electrical_power`` (negative = battery charging).
            Magnitude follows ``P = -V_pack * I`` with
            ``I = (V_bemf - V_pack) / R_phase``.
        """
        v_bemf = self.config.motor_back_emf_constant_v_s_per_rad * omega_m_rad_s
        if v_bemf <= pack_voltage_v:
            return 0.0
        # Rectifier fires: body diodes conduct, pack charges.
        # I_rectify = overvoltage / R_phase, P = -V_pack * I (negative
        # = regen / pack-charging, matching the rest of electrical_power).
        overvoltage = v_bemf - pack_voltage_v
        i_regen = overvoltage / self._BEMF_PHASE_RESISTANCE_OHM
        return -pack_voltage_v * i_regen

    def _coast_power_w(
        self, omega_m_rad_s: float, pack_voltage_v: float | None,
    ) -> float:
        """4-term coast electrical-power model + back-EMF guard.

        ``P_coast = P_aux + P_iron(omega) + P_windage(omega)
                    + P_pwm_switch(V_pack) + P_bemf_rectify(...)``

        Args:
            omega_m_rad_s: Motor angular velocity (rad/s, > 0).
            pack_voltage_v: Pack terminal voltage (V). When ``None``,
                the PWM and rectifier-guard terms are skipped; only
                aux + iron + windage are returned. This back-compat
                path lets callers that don't carry pack voltage still
                see a non-zero coast power, biased low by exactly the
                PWM contribution.

        Returns:
            Coast electrical power in W (positive = pack discharge).
            Net value is the sum of (always positive) loss terms minus
            the (always non-positive) rectifier contribution. On CT-16EV
            the rectifier is always 0, so the net is positive in
            practice; an externally-driven motor (V_bemf > V_pack)
            could produce a net negative value.
        """
        coast = self.config.coast_loss
        p_aux = coast.p_aux_w
        p_iron = self._iron_loss_w(omega_m_rad_s)
        p_windage = self._windage_loss_w(omega_m_rad_s)
        p_pwm = 0.0
        p_bemf = 0.0
        if pack_voltage_v is not None and pack_voltage_v > 0.0:
            p_pwm = self._pwm_switch_loss_w(pack_voltage_v)
            p_bemf = self._bemf_rectify_guard_w(omega_m_rad_s, pack_voltage_v)
        return p_aux + p_iron + p_windage + p_pwm + p_bemf

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
           4-term physical no-load loss model
           ``P = P_aux + P_iron(omega) + P_windage(omega) + P_pwm(V_pack)
           + P_bemf_rectify(omega, V_pack)``. Each term is bounded by
           datasheet physics (Cascadia CM200DX, EMRAX 228 MV LC). The
           back-EMF rectifier is a guard (~always 0 at CT-16EV
           operating points). When ``pack_voltage_v`` is None, the PWM
           and rectifier terms are skipped — the result is biased low
           by the PWM contribution (~350 W at nominal V_pack). See
           :class:`fsae_sim.vehicle.powertrain.CoastLossConfig`.

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

        # --- Coast branch: 4-term physical no-load loss model ---
        # Replaces the previous 0-W coast branch (which lost ~45 Wh per
        # stint vs. telemetry — issue 22). Each term is tied to a named
        # mechanism with datasheet-bounded coefficients; calibrate the
        # full coefficient vector via ``scripts/calibrate_coast_loss.py``
        # against telemetry for production use.
        #
        # Refs: Pyrhonen §3.6 (iron), §3.7 (windage); Hanselman §10.5
        # (mechanical); Krause/Wasynczuk/Sudhoff §3.5/§6 (PMSM no-load);
        # Cascadia CM200DX rev D §5.4 (PWM); Infineon AN2008-03
        # (gate-charge); Mohan/Undeland/Robbins §27-2.
        if abs(motor_torque_nm) <= self._COAST_TORQUE_THRESHOLD_NM:
            return self._coast_power_w(omega, pack_voltage_v)

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
