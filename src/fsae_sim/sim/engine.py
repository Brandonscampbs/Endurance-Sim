"""Quasi-static endurance simulation engine."""

from dataclasses import dataclass

import pandas as pd

from fsae_sim.driver.strategy import DriverStrategy
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig


@dataclass
class SimResult:
    """Output of a single simulation run."""
    config_name: str
    strategy_name: str
    track_name: str
    states: pd.DataFrame  # time series of SimState fields
    total_time_s: float
    total_energy_kwh: float
    final_soc: float
    laps_completed: int


class SimulationEngine:
    """Quasi-static endurance simulation.

    For each track segment, resolves speed from force balance and
    driver strategy, steps battery state, and records results.
    """

    def __init__(
        self,
        vehicle: VehicleConfig,
        track: Track,
        strategy: DriverStrategy,
    ):
        self.vehicle = vehicle
        self.track = track
        self.strategy = strategy

    def run(self, num_laps: int = 1) -> SimResult:
        """Run the endurance simulation.

        Implemented in Phase 2.
        """
        raise NotImplementedError("Simulation engine not yet implemented")
