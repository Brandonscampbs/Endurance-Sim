"""Top-level vehicle configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from fsae_sim.vehicle.battery import BatteryConfig
from fsae_sim.vehicle.powertrain import PowertrainConfig


@dataclass(frozen=True)
class VehicleParams:
    """Physical vehicle parameters."""
    mass_kg: float
    frontal_area_m2: float
    drag_coefficient: float
    rolling_resistance: float
    wheelbase_m: float


@dataclass(frozen=True)
class VehicleConfig:
    """Complete vehicle configuration loaded from YAML."""
    name: str
    year: int
    description: str
    vehicle: VehicleParams
    powertrain: PowertrainConfig
    battery: BatteryConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleConfig":
        """Load vehicle configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            name=data["name"],
            year=data["year"],
            description=data["description"],
            vehicle=VehicleParams(**data["vehicle"]),
            powertrain=PowertrainConfig(**data["powertrain"]),
            battery=BatteryConfig.from_dict(data["battery"]),
        )
