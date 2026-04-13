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

    @property
    def pack_voltage_min_v(self) -> float:
        return self.cell_voltage_min_v * self.series

    @property
    def pack_voltage_max_v(self) -> float:
        return self.cell_voltage_max_v * self.series

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryConfig":
        """Build from parsed YAML dict."""
        return cls(
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
