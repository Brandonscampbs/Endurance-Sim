"""Tests that the dynamics_backend switch in VehicleConfig routes engine
construction correctly between legacy and dynamics6dof backends.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fsae_sim.driver.strategy import ControlAction, ControlCommand, DriverStrategy, SimState
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.vehicle import VehicleConfig


class _FullThrottleStrategy(DriverStrategy):
    name = "full_throttle_test"

    def decide(self, state: SimState, upcoming):
        return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)


@pytest.fixture
def legacy_config_path() -> Path:
    return Path(__file__).parents[1].parent / "configs" / "ct16ev.yaml"


@pytest.fixture
def battery_for_backend_tests(legacy_config_path):
    cfg = VehicleConfig.from_yaml(legacy_config_path)
    return BatteryModel(cfg.battery)


def test_default_backend_is_legacy(legacy_config_path):
    cfg = VehicleConfig.from_yaml(legacy_config_path)
    assert cfg.dynamics_backend == "legacy"


def test_legacy_backend_still_constructs_vehicle_dynamics(legacy_config_path):
    from fsae_sim.vehicle.dynamics import VehicleDynamics
    cfg = VehicleConfig.from_yaml(legacy_config_path)

    # Engine needs a track; provide a minimal one from the existing fixture pattern.
    from fsae_sim.track.track import Segment, Track
    track = Track(
        name="test",
        segments=[Segment(index=0, distance_start_m=0.0, length_m=10.0, curvature=0.0, grade=0.0)],
    )
    strategy = _FullThrottleStrategy()

    engine = SimulationEngine(cfg, track, strategy, BatteryModel(cfg.battery))
    assert isinstance(engine.dynamics, VehicleDynamics)


def test_dynamics6dof_backend_constructs_fastest_lap_backend(legacy_config_path):
    from fsae_sim.dynamics6dof.backend import FastestLapDynamicsBackend

    cfg = VehicleConfig.from_yaml(legacy_config_path)
    cfg_6dof = replace(cfg, dynamics_backend="dynamics6dof")

    from fsae_sim.track.track import Segment, Track
    track = Track(
        name="test",
        segments=[Segment(index=0, distance_start_m=0.0, length_m=10.0, curvature=0.0, grade=0.0)],
    )
    strategy = _FullThrottleStrategy()

    engine = SimulationEngine(cfg_6dof, track, strategy, BatteryModel(cfg_6dof.battery))
    assert isinstance(engine.dynamics, FastestLapDynamicsBackend)


def test_dynamics6dof_backend_runs_short_sim(legacy_config_path, voltt_cell_path):
    """Full end-to-end: engine runs a 1-lap sim under dynamics6dof backend."""
    from fsae_sim.data.loader import load_voltt_csv
    from fsae_sim.track.track import Segment, Track

    cfg = VehicleConfig.from_yaml(legacy_config_path)
    cfg_6dof = replace(cfg, dynamics_backend="dynamics6dof")

    segments = [
        Segment(
            index=i,
            distance_start_m=i * 20.0,
            length_m=20.0,
            curvature=0.02 if i % 3 == 0 else 0.0,
            grade=0.0,
        )
        for i in range(10)
    ]
    track = Track(name="test", segments=segments)

    battery = BatteryModel(cfg_6dof.battery)
    battery.calibrate_from_voltt(load_voltt_csv(voltt_cell_path))

    engine = SimulationEngine(cfg_6dof, track, strategy=_FullThrottleStrategy(),
                              battery_model=battery)
    result = engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=25.0)

    assert result.total_time_s > 0.0
    assert result.final_soc < 95.0
