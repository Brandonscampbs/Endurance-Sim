"""Powertrain configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PowertrainConfig:
    """Motor, inverter, and drivetrain parameters."""
    motor_speed_max_rpm: float
    brake_speed_rpm: float
    torque_limit_inverter_nm: float
    # LVCU software torque limit. Firmware default is 2200//x10 = 220 Nm
    # (LVCU Code.txt:109). Older configs used 150 Nm; that was an
    # operational cap for Michigan 2025, not the firmware value. See
    # C17 in docs/SIMULATOR_AUDIT_2026-04-16.md.
    torque_limit_lvcu_nm: float
    iq_limit_a: float
    id_limit_a: float
    gear_ratio: float
    drivetrain_efficiency: float
    # LVCU torque command parameters (from real LVCU Code.txt)
    lvcu_power_constant: float = 420.0        # 4200 in 0.1Nm CAN units / 10
    lvcu_rpm_scale: float = 0.1076            # RPM to angular velocity scale
    lvcu_omega_floor: float = 23.04           # 230.4 in CAN units / 10
    lvcu_pedal_deadzone_low: float = 0.1      # tmap_lut V_MIN
    lvcu_pedal_deadzone_high: float = 0.9     # tmap_lut V_MAX
    lvcu_overspeed_rpm: float = 6000.0        # hard torque override threshold
    lvcu_overspeed_torque_nm: float = 30.0    # torque at overspeed (300/10)

    # S14: BMS current-limit safety offset used inside the LVCU
    # (`current_limit = (RxData[1] << 8 | RxData[0]) - 3` — LVCU Code.txt:151).
    lvcu_bms_current_offset_a: float = 3.0

    # Optional operational safety cap layered on top of the firmware LVCU
    # limit. Used to record the 150 Nm competition cap ("run conservatively")
    # without pretending the firmware imposed it. Leave as None to disable.
    safety_torque_cap_nm: Optional[float] = None

    # Motor back-EMF constant (V per rad/s of electrical frequency at the
    # motor terminals, treated here as V per rad/s of mechanical angular
    # velocity because pole pairs are folded into the calibration).
    # EMRAX 228 MV LC: ~0.045 V/(rad/s). Used by the C2 back-EMF
    # rectification term in `PowertrainModel.electrical_power`.
    motor_back_emf_constant_v_s_per_rad: float = 0.045

    def __post_init__(self) -> None:
        # NF-41: reject configs with a pedal-deadzone span that would cause
        # divide-by-zero or catastrophic noise amplification.
        span = self.lvcu_pedal_deadzone_high - self.lvcu_pedal_deadzone_low
        if span < 0.01:
            raise ValueError(
                f"lvcu_pedal_deadzone_high ({self.lvcu_pedal_deadzone_high}) "
                f"must exceed lvcu_pedal_deadzone_low "
                f"({self.lvcu_pedal_deadzone_low}) by at least 0.01; got "
                f"span={span:.6f}."
            )
