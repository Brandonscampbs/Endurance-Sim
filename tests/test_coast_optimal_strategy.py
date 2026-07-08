"""Tests for the idealized no-brake coast-planning driver."""

from __future__ import annotations

import numpy as np
import pytest

from fsae_sim.driver.strategy import ControlAction, SimState
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel


def _track_with_corner(num_segments: int = 120, seg_len: float = 1.0) -> Track:
    segs = []
    for i in range(num_segments):
        if i < 3 or i >= num_segments - 3:
            curvature = 0.02
        elif 70 <= i < 90:
            curvature = 0.06
        else:
            curvature = 0.0
        segs.append(Segment(
            index=i,
            distance_start_m=i * seg_len,
            length_m=seg_len,
            curvature=curvature,
            grade=0.0,
        ))
    return Track(name="coast_corner", segments=segs, source="synthetic")


def _state(speed_ms: float, seg_idx: int) -> SimState:
    return SimState(
        time=0.0,
        distance=float(seg_idx),
        speed=float(speed_ms),
        soc=0.95,
        pack_voltage=400.0,
        pack_current=0.0,
        cell_temp=30.0,
        lap=0,
        segment_idx=seg_idx,
    )


@pytest.fixture
def vehicle_config(ct16ev_config_path) -> VehicleConfig:
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def battery(vehicle_config, voltt_cell_path) -> BatteryModel:
    from fsae_sim.data.loader import load_voltt_csv

    df = load_voltt_csv(voltt_cell_path)
    model = BatteryModel(vehicle_config.battery)
    model.calibrate_from_voltt(df)
    return model


def test_coast_optimal_strategy_name_and_prediction_safety(vehicle_config):
    from fsae_sim.driver.strategies import CoastOptimalStrategy

    strategy = CoastOptimalStrategy.from_config(vehicle_config, _track_with_corner())
    assert strategy.name == "ideal_coast"
    assert strategy.uses_observed_speed_caps is False


def test_coast_optimal_strategy_never_commands_brake_or_regen(vehicle_config):
    from fsae_sim.driver.strategies import CoastOptimalStrategy

    track = _track_with_corner()
    strategy = CoastOptimalStrategy.from_config(vehicle_config, track)
    strategy.set_envelope(np.full(track.num_segments, 18.0))

    cmd = strategy.decide(_state(30.0, 10), track.segments[10:15])
    assert cmd.action == ControlAction.COAST
    assert cmd.throttle_pct == 0.0
    assert cmd.brake_pct == 0.0
    assert cmd.regen_request_pct == 0.0


def test_coast_optimal_releases_before_corner(vehicle_config):
    from fsae_sim.driver.strategies import CoastOptimalStrategy

    track = _track_with_corner()
    strategy = CoastOptimalStrategy.from_config(vehicle_config, track)
    # Hard physical envelope high enough that the coast envelope is set by
    # corner limits and passive coast-down, not by this artificial ceiling.
    strategy.set_envelope(np.full(track.num_segments, 40.0))
    envelope = strategy.coast_envelope

    straight_before_corner = 60
    corner = 75
    assert envelope[straight_before_corner] > envelope[corner]

    cmd_fast = strategy.decide(
        _state(envelope[straight_before_corner] + 1.0, straight_before_corner),
        track.segments[straight_before_corner:straight_before_corner + 5],
    )
    assert cmd_fast.action == ControlAction.COAST
    assert cmd_fast.brake_pct == 0.0

    cmd_slow = strategy.decide(
        _state(max(1.0, envelope[straight_before_corner] - 5.0), straight_before_corner),
        track.segments[straight_before_corner:straight_before_corner + 5],
    )
    assert cmd_slow.brake_pct == 0.0
    assert cmd_slow.regen_request_pct == 0.0
    assert 0.0 <= cmd_slow.throttle_pct <= 1.0


def test_coast_optimal_does_not_request_torque_above_motor_speed_cap(
    vehicle_config,
):
    from fsae_sim.driver.strategies import CoastOptimalStrategy
    from fsae_sim.vehicle.powertrain_model import PowertrainModel

    track = _track_with_corner()
    strategy = CoastOptimalStrategy.from_config(vehicle_config, track)
    strategy.set_envelope(np.full(track.num_segments, 80.0))
    pt = PowertrainModel(vehicle_config.powertrain)
    cap_speed_ms = pt.speed_from_motor_rpm(
        vehicle_config.powertrain.motor_speed_max_rpm,
    )

    cmd = strategy.decide(
        _state(cap_speed_ms + 1.0, 20),
        track.segments[20:25],
    )
    assert cmd.action == ControlAction.COAST
    assert cmd.throttle_pct == 0.0
    assert cmd.brake_pct == 0.0


def test_coast_optimal_engine_run_uses_no_brake_force(
    vehicle_config,
    battery,
):
    from fsae_sim.driver.strategies import CoastOptimalStrategy
    from fsae_sim.sim.engine import SimulationEngine, SimulationMode

    track = _track_with_corner()
    strategy = CoastOptimalStrategy.from_config(vehicle_config, track)
    engine = SimulationEngine(
        vehicle_config,
        track,
        strategy,
        battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    )
    result = engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=30.0)

    states = result.states
    assert result.laps_completed == 1
    assert (states["brake_pct"] == 0.0).all()
    assert (states["regen_force_n"] == 0.0).all()
    assert states["brake_force_n"].max() == pytest.approx(0.0, abs=1e-9)


def test_existing_strategies_still_import_with_coast_optimal():
    from fsae_sim.driver.strategies import (
        AdaptiveStrategy,
        CalibratedStrategy,
        CoastOptimalStrategy,
        ReplayStrategy,
    )

    assert AdaptiveStrategy.name == "adaptive"
    assert CalibratedStrategy.name == "driver"
    assert ReplayStrategy.name == "replay"
    assert CoastOptimalStrategy.name == "ideal_coast"
