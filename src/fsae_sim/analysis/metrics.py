"""Post-processing metrics computed from simulation results."""

import pandas as pd


def compute_lap_times(states: pd.DataFrame) -> list[float]:
    """Extract per-lap times from simulation state time series.

    Implemented in Phase 2.
    """
    raise NotImplementedError


def compute_energy_per_lap(states: pd.DataFrame) -> list[float]:
    """Compute energy consumed per lap in kWh.

    Implemented in Phase 2.
    """
    raise NotImplementedError


def compute_pareto_frontier(
    results: pd.DataFrame,
    time_col: str = "total_time_s",
    energy_col: str = "total_energy_kwh",
) -> pd.DataFrame:
    """Find Pareto-optimal points minimizing both time and energy.

    Implemented in Phase 3.
    """
    raise NotImplementedError
