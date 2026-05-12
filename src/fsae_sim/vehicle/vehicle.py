"""Top-level vehicle configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fsae_sim.vehicle.battery import BatteryConfig
from fsae_sim.vehicle.environment import EnvironmentConfig
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
    cg_height_m: float = 0.2794
    weight_distribution_front: float = 0.53
    downforce_distribution_front: float = 0.61
    aero_drag_is_cda: bool = True
    aero_downforce_is_cla: bool = True
    # D-08: brake system peak pressure (bar). Data-independent so that
    # recalibration on a short subset produces the same normalized brake_pct
    # as recalibration on the full endurance. Default is a conservative
    # FSAE-typical max; override in configs/ct16ev.yaml when the DSS Brake
    # System sheet provides a measured value.
    brake_max_pressure_bar: float = 60.0
    # Per-event ambient-air state (replaces the bare
    # ``physics_constants.AIR_DENSITY_KG_M3`` global).  Default block is
    # the Michigan International Speedway endurance day; override per
    # event from local METAR.  Frozen dataclasses cannot share mutable
    # defaults, so we use ``field(default_factory=EnvironmentConfig)``.
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg!r}")
        if self.wheelbase_m <= 0.0:
            raise ValueError(
                f"wheelbase_m must be positive, got {self.wheelbase_m!r}"
            )
        for name, value in (
            ("weight_distribution_front", self.weight_distribution_front),
            ("downforce_distribution_front", self.downforce_distribution_front),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.cg_height_m <= 0.0:
            raise ValueError(
                f"cg_height_m must be positive, got {self.cg_height_m!r}"
            )


@dataclass(frozen=True)
class TireConfig:
    """Tire model configuration."""

    tir_file: str
    static_camber_front_deg: float
    static_camber_rear_deg: float
    grip_scale: float = 1.0  # TTC-to-car grip calibration factor
    grip_scale_source: str = "nominal"
    grip_scale_notes: str = ""

    def __post_init__(self) -> None:
        if self.grip_scale <= 0.0:
            raise ValueError(
                f"grip_scale must be positive, got {self.grip_scale!r}"
            )


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

    def require_predictive_ready(
        self,
        *,
        allow_empirical_grip: bool = False,
    ) -> None:
        """Fail loudly if a vehicle config is incomplete for prediction."""
        missing: list[str] = []
        if self.tire is None:
            missing.append("tire")
        if self.suspension is None:
            missing.append("suspension")
        if not self.vehicle.aero_drag_is_cda:
            missing.append("vehicle.aero_drag_is_cda=true")
        if self.vehicle.downforce_coefficient > 0.0 and not self.vehicle.aero_downforce_is_cla:
            missing.append("vehicle.aero_downforce_is_cla=true")
        if missing:
            raise ValueError(
                f"{self.name} is not prediction-ready; missing or ambiguous "
                f"physics sections: {', '.join(missing)}"
            )

        assert self.tire is not None
        empirical_source = self.tire.grip_scale_source.lower()
        uses_empirical_grip = (
            self.tire.grip_scale != 1.0
            and any(
                token in empirical_source
                for token in ("telemetry", "endurance", "validation", "calibration")
            )
        )
        if uses_empirical_grip and not allow_empirical_grip:
            raise ValueError(
                f"{self.name} tire grip_scale={self.tire.grip_scale} comes from "
                f"{self.tire.grip_scale_source!r}; prediction mode requires an "
                "independent tire assumption or allow_empirical_grip=True."
            )

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
