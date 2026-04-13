"""Powertrain configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PowertrainConfig:
    """Motor, inverter, and drivetrain parameters."""
    motor_speed_max_rpm: float
    brake_speed_rpm: float
    torque_limit_inverter_nm: float
    torque_limit_lvcu_nm: float
    iq_limit_a: float
    id_limit_a: float
    gear_ratio: float
    drivetrain_efficiency: float
