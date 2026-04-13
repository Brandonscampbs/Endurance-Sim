"""FSAE endurance and efficiency scoring formulas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnduranceScore:
    """Combined endurance + efficiency score."""
    endurance_points: float
    efficiency_points: float
    total_points: float


def calculate_endurance_points(
    team_time_s: float,
    fastest_time_s: float,
    max_points: float = 300.0,
) -> float:
    """Calculate FSAE endurance event points.

    Implemented in Phase 4.
    """
    raise NotImplementedError


def calculate_efficiency_points(
    team_energy_kwh: float,
    team_time_s: float,
    min_energy_kwh: float,
    fastest_time_s: float,
    max_points: float = 100.0,
) -> float:
    """Calculate FSAE efficiency event points.

    Implemented in Phase 4.
    """
    raise NotImplementedError
