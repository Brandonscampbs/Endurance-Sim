"""Battery pack configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DischargeLimitPoint:
    """Temperature-dependent discharge current limit."""
    temp_c: float
    max_current_a: float


@dataclass(frozen=True)
class BatteryConfig:
    """Battery pack configuration parameters."""
    cell_type: str
    series: int
    parallel: int
    cell_voltage_min_v: float
    cell_voltage_max_v: float
    discharged_soc_pct: float
    soc_taper_threshold_pct: float
    soc_taper_rate_a_per_pct: float
    discharge_limits: tuple[DischargeLimitPoint, ...]
    cell_capacity_ah: float = 4.5  # Molicel P45B default; P50B = 5.0
    pack_structural_thermal_mass_kj_per_k: float = 7.5  # ≈25% of cell thermal mass (busbars, plates, enclosure)
    # Passive cooling: Newton's law h·(T_cell − T_ambient).
    # CT-16EV (2025) has no active cooling — enclosure is effectively
    # adiabatic on endurance-lap timescales.  Default is 0 (no cooling)
    # to match the real car; override in the config when the 2026 car
    # gets a cooling system.  With h > 0 the model has a proper
    # equilibrium instead of ramping until the BMS kills power.
    thermal_conductance_w_per_k: float = 0.0
    ambient_temperature_c: float = 25.0

    @property
    def pack_voltage_min_v(self) -> float:
        return self.cell_voltage_min_v * self.series

    @property
    def pack_voltage_max_v(self) -> float:
        return self.cell_voltage_max_v * self.series

    @property
    def pack_capacity_ah(self) -> float:
        return self.cell_capacity_ah * self.parallel

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryConfig":
        """Build from parsed YAML dict."""
        # cell_capacity_ah is required for new configs but optional for
        # backward compatibility (defaults to P45B 4.5 Ah).
        kwargs = dict(
            cell_type=data["cell_type"],
            series=data["topology"]["series"],
            parallel=data["topology"]["parallel"],
            cell_voltage_min_v=data["cell_voltage_min_v"],
            cell_voltage_max_v=data["cell_voltage_max_v"],
            discharged_soc_pct=data["discharged_soc_pct"],
            soc_taper_threshold_pct=data["soc_taper"]["threshold_pct"],
            soc_taper_rate_a_per_pct=data["soc_taper"]["rate_a_per_pct"],
            discharge_limits=tuple(
                DischargeLimitPoint(**dl) for dl in data["discharge_limits"]
            ),
        )
        if "cell_capacity_ah" in data:
            kwargs["cell_capacity_ah"] = float(data["cell_capacity_ah"])
        if "pack_structural_thermal_mass_kj_per_k" in data:
            kwargs["pack_structural_thermal_mass_kj_per_k"] = float(
                data["pack_structural_thermal_mass_kj_per_k"]
            )
        if "thermal_conductance_w_per_k" in data:
            kwargs["thermal_conductance_w_per_k"] = float(
                data["thermal_conductance_w_per_k"]
            )
        if "ambient_temperature_c" in data:
            kwargs["ambient_temperature_c"] = float(
                data["ambient_temperature_c"]
            )
        return cls(**kwargs)
