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
    downforce_coefficient: float = 0.0  # Cl * A (m²), 0 = no downforce
    rotor_inertia_kg_m2: float = 0.06  # EMRAX 228 default
    wheel_inertia_kg_m2: float = 0.3   # per wheel (10" Hoosier + aluminum rim)
    # D-08: brake system peak pressure (bar). Data-independent so that
    # recalibration on a short subset produces the same normalized brake_pct
    # as recalibration on the full endurance. Default is a conservative
    # FSAE-typical max; override in configs/ct16ev.yaml when the DSS Brake
    # System sheet provides a measured value.
    brake_max_pressure_bar: float = 60.0


@dataclass(frozen=True)
class TireConfig:
    """Tire model configuration."""

    tir_file: str
    static_camber_front_deg: float
    static_camber_rear_deg: float
    grip_scale: float = 1.0  # TTC-to-car grip calibration factor


@dataclass(frozen=True)
class SuspensionConfig:
    """Suspension geometry and compliance parameters (DSS values)."""

    roll_stiffness_front_nm_per_deg: float
    roll_stiffness_rear_nm_per_deg: float
    roll_center_height_front_mm: float
    roll_center_height_rear_mm: float
    roll_camber_front_deg_per_deg: float
    roll_camber_rear_deg_per_deg: float
    front_track_mm: float
    rear_track_mm: float


@dataclass(frozen=True)
class VehicleConfig:
    """Complete vehicle configuration loaded from YAML."""

    name: str
    year: int
    description: str
    vehicle: VehicleParams
    powertrain: PowertrainConfig
    battery: BatteryConfig
    tire: TireConfig | None = None
    suspension: SuspensionConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleConfig":
        """Load vehicle configuration from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tire_data = data.get("tire")
        suspension_data = data.get("suspension")

        try:
            return cls(
                name=data["name"],
                year=data["year"],
                description=data["description"],
                vehicle=VehicleParams(**data["vehicle"]),
                powertrain=PowertrainConfig(**data["powertrain"]),
                battery=BatteryConfig.from_dict(data["battery"]),
                tire=TireConfig(**tire_data) if tire_data is not None else None,
                suspension=SuspensionConfig(**suspension_data) if suspension_data is not None else None,
            )
        except TypeError as e:
            raise ValueError(f"{path}: {e}") from e
