"""Runtime powertrain model for FSAE EV drivetrain.

Implements speed/torque/force/power relationships for a PMSM motor
with a single-speed gear reduction.  The model handles:
- Motor RPM from vehicle speed (and inverse)
- Torque capability vs RPM (flat + field-weakening + above-max cutoff)
- Wheel torque and tractive force through gear ratio and efficiency
- Drive and regenerative braking force from throttle/brake demand
- Electrical power drawn from (or returned to) the battery pack
- Pack current from electrical power and instantaneous pack voltage
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fsae_sim.vehicle.powertrain import PowertrainConfig

if TYPE_CHECKING:
    from fsae_sim.vehicle.motor_efficiency import MotorEfficiencyMap


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

    # Regen capture efficiency relative to drivetrain efficiency.
    # The mechanical-to-electrical conversion path has the same gearbox
    # friction but the inverter regeneration efficiency is slightly lower
    # than motoring.  A conservative 85 % factor captures this.
    _REGEN_EFFICIENCY_FACTOR: float = 0.85

    def __init__(
        self,
        config: PowertrainConfig,
        efficiency_map: "MotorEfficiencyMap | None" = None,
    ) -> None:
        self.config = config
        self._efficiency_map = efficiency_map

        # Pre-compute constants used in every call
        self._torque_limit_nm: float = min(
            config.torque_limit_inverter_nm,
            config.torque_limit_lvcu_nm,
        )
        self._rad_per_s_per_rpm: float = math.pi / 30.0  # 2*pi/60

        # Regen efficiency: generator mode recovers less than motoring consumes
        self._regen_efficiency: float = (
            config.drivetrain_efficiency * self._REGEN_EFFICIENCY_FACTOR
        )

    def _get_efficiency(self, motor_rpm: float, motor_torque_nm: float) -> float:
        """Total drivetrain efficiency at the given operating point.

        Uses the motor efficiency map when available, otherwise falls
        back to the fixed ``config.drivetrain_efficiency``.
        """
        if self._efficiency_map is not None:
            return self._efficiency_map.total_efficiency(motor_rpm, motor_torque_nm)
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
    # Torque delivery
    # ------------------------------------------------------------------

    # Field-weakening onset RPM and derating slope, derived from
    # Torque Feedback / LVCU Torque Req analysis of Michigan 2025
    # telemetry.  Below _FW_ONSET the motor delivers ~100% of the
    # command; above it, back-EMF limits delivery linearly.
    _FW_ONSET_RPM: float = 2800.0
    _FW_SLOPE_PER_RPM: float = 0.75 / 250.0  # ratio drop per RPM above onset

    def torque_delivery_factor(self, motor_rpm: float) -> float:
        """Fraction of commanded torque the motor physically delivers.

        Below the field-weakening onset the motor tracks the command
        at ~100%.  Above onset, back-EMF reduces delivery linearly.
        Derived from Michigan 2025 Torque Feedback vs LVCU Torque Req.

        Args:
            motor_rpm: Motor shaft speed in RPM.

        Returns:
            Delivery factor in (0, 1].
        """
        if motor_rpm <= self._FW_ONSET_RPM:
            return 1.0
        derate = (motor_rpm - self._FW_ONSET_RPM) * self._FW_SLOPE_PER_RPM
        return max(0.0, 1.0 - derate)

    # ------------------------------------------------------------------
    # Torque capability
    # ------------------------------------------------------------------

    def max_motor_torque(self, motor_rpm: float) -> float:
        """Maximum motor output torque at given RPM (Nm).

        The PMSM operates in three distinct regions:

        1. **Constant-torque region** (0 <= rpm <= brake_speed_rpm):
           Full torque = min(inverter_limit, lvcu_limit).

        2. **Field-weakening region** (brake_speed_rpm < rpm <= motor_speed_max_rpm):
           Torque tapers linearly from full torque to zero as RPM rises from
           the brake speed to the maximum speed.  This approximates the
           hyperbolic power curve of a PMSM under field weakening.

        3. **Over-speed region** (rpm > motor_speed_max_rpm):
           Zero torque — the motor cannot operate above its maximum electrical
           frequency.

        Args:
            motor_rpm: Motor shaft speed in RPM.

        Returns:
            Maximum available motor torque in Nm (>= 0).
        """
        rpm = max(0.0, motor_rpm)

        if rpm <= self.config.brake_speed_rpm:
            return self._torque_limit_nm

        if rpm <= self.config.motor_speed_max_rpm:
            # Linear taper from full torque at brake_speed down to 0 at max_speed
            span = self.config.motor_speed_max_rpm - self.config.brake_speed_rpm
            excess = rpm - self.config.brake_speed_rpm
            taper_fraction = 1.0 - (excess / span)
            return self._torque_limit_nm * taper_fraction

        # Above maximum RPM
        return 0.0

    def lvcu_torque_command(
        self, pedal_pct: float, motor_rpm: float, bms_current_limit_a: float,
    ) -> float:
        """Motor torque command replicating the real LVCU firmware.

        Faithfully implements the torque command chain from LVCU Code.txt:
        pedal -> tmap_lut (dead zone remap) -> torque_lut (power-limited
        ceiling) -> inverter clamp.

        Args:
            pedal_pct: Raw pedal position in [0.0, 1.0].
            motor_rpm: Motor shaft speed in RPM.
            bms_current_limit_a: BMS discharge current limit in A.

        Returns:
            Commanded motor torque in Nm (>= 0).
        """
        cfg = self.config

        # 1. tmap_lut: dead zone remap [V_MIN, V_MAX] -> [0, 1]
        pedal_clamped = max(cfg.lvcu_pedal_deadzone_low,
                           min(pedal_pct, cfg.lvcu_pedal_deadzone_high))
        pedal_remapped = (
            (pedal_clamped - cfg.lvcu_pedal_deadzone_low)
            / (cfg.lvcu_pedal_deadzone_high - cfg.lvcu_pedal_deadzone_low)
        )

        # 2. torque_lut: power-limited torque ceiling
        omega_term = max(cfg.lvcu_omega_floor, motor_rpm * cfg.lvcu_rpm_scale)
        power_ceiling_nm = cfg.lvcu_power_constant * bms_current_limit_a / omega_term

        # LVCU torque limit (software cap)
        torque_ceiling_nm = min(cfg.torque_limit_lvcu_nm, power_ceiling_nm)

        # Overspeed override
        if motor_rpm >= cfg.lvcu_overspeed_rpm:
            torque_ceiling_nm = cfg.lvcu_overspeed_torque_nm

        # Inverter hardware limit (independent clamp)
        torque_ceiling_nm = min(torque_ceiling_nm, cfg.torque_limit_inverter_nm)

        # 3. Final command: remapped pedal * clamped ceiling
        return pedal_remapped * torque_ceiling_nm

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
            bms_current_limit_a: BMS discharge current limit in A.

        Returns:
            Torque ceiling in Nm.
        """
        cfg = self.config
        omega_term = max(cfg.lvcu_omega_floor, motor_rpm * cfg.lvcu_rpm_scale)
        power_ceiling_nm = cfg.lvcu_power_constant * bms_current_limit_a / omega_term
        torque_ceiling_nm = min(cfg.torque_limit_lvcu_nm, power_ceiling_nm)
        if motor_rpm >= cfg.lvcu_overspeed_rpm:
            torque_ceiling_nm = cfg.lvcu_overspeed_torque_nm
        return min(torque_ceiling_nm, cfg.torque_limit_inverter_nm)

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
        used for driving, scaled by the regen efficiency factor.  The returned
        force is negative (opposing motion).

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
        # Regen wheel force: only gearbox friction reduces shaft torque
        # to wheel torque.  Motor/inverter regen efficiency affects energy
        # recovery (electrical_power), not mechanical braking force.
        regen_wheel_torque = commanded_torque * self.config.gear_ratio * self._GEARBOX_EFFICIENCY
        return -(regen_wheel_torque / self.TIRE_RADIUS_M)

    # ------------------------------------------------------------------
    # Electrical power
    # ------------------------------------------------------------------

    def electrical_power(self, motor_torque_nm: float, motor_rpm: float) -> float:
        """Electrical power exchanged with the battery pack (W).

        Sign convention (battery perspective):
        - **Positive** (motoring): power drawn *from* the battery.
        - **Negative** (regen): power returned *to* the battery.

        For motoring the mechanical power is divided by drivetrain efficiency
        to account for friction losses.  For regen, mechanical power is
        multiplied by efficiency (losses reduce energy recovered).

        At zero speed the mechanical power is zero regardless of torque, so
        electrical power is also zero (no back-EMF, no current flows at 0 RPM
        in a speed-controlled inverter).

        Args:
            motor_torque_nm: Motor shaft torque in Nm.  Positive = motoring,
                negative = generating (regen).
            motor_rpm: Motor shaft speed in RPM.

        Returns:
            Electrical power in W (positive = battery discharge).
        """
        if motor_rpm <= 0.0:
            return 0.0

        omega = motor_rpm * self._rad_per_s_per_rpm  # rad/s
        p_mechanical = motor_torque_nm * omega  # W

        if p_mechanical >= 0.0:
            # Motoring: battery must supply more than mechanical output.
            # Use operating-point-dependent efficiency when available.
            eta = self._get_efficiency(motor_rpm, motor_torque_nm)
            if eta > 0.0:
                return p_mechanical / eta
            return 0.0
        else:
            # Regen: battery receives less than mechanical input due to losses.
            # Use operating-point efficiency for regen too.
            if self._efficiency_map is not None:
                eta_regen = (
                    self._efficiency_map.total_efficiency(motor_rpm, abs(motor_torque_nm))
                    * self._REGEN_EFFICIENCY_FACTOR
                )
            else:
                eta_regen = self._regen_efficiency
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
