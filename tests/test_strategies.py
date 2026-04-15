"""Tests for driver strategy implementations."""

import numpy as np
import pytest

from fsae_sim.driver.strategy import ControlAction, SimState
from fsae_sim.driver.strategies import (
    CoastOnlyStrategy,
    ReplayStrategy,
    ThresholdBrakingStrategy,
)
from fsae_sim.track.track import Segment
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.vehicle import VehicleParams


@pytest.fixture
def vehicle_params():
    return VehicleParams(
        mass_kg=278.0,
        frontal_area_m2=1.0,
        drag_coefficient=1.502,
        rolling_resistance=0.015,
        wheelbase_m=1.549,
        downforce_coefficient=2.18,
    )


@pytest.fixture
def dynamics(vehicle_params):
    return VehicleDynamics(vehicle_params)


@pytest.fixture
def straight_segment():
    return Segment(index=0, distance_start_m=0.0, length_m=50.0, curvature=0.0, grade=0.0)


@pytest.fixture
def corner_segment():
    """Tight 15m radius corner."""
    return Segment(index=1, distance_start_m=50.0, length_m=20.0, curvature=1.0 / 15.0, grade=0.0)


@pytest.fixture
def sim_state_fast():
    """State at 15 m/s (~54 km/h) - fast for a corner."""
    return SimState(
        time=10.0, distance=100.0, speed=15.0, soc=0.90,
        pack_voltage=440.0, pack_current=30.0, cell_temp=30.0,
        lap=0, segment_idx=0,
    )


@pytest.fixture
def sim_state_slow():
    """State at 5 m/s (~18 km/h) - slow."""
    return SimState(
        time=10.0, distance=50.0, speed=5.0, soc=0.90,
        pack_voltage=440.0, pack_current=10.0, cell_temp=30.0,
        lap=0, segment_idx=0,
    )


def _make_replay_strategy():
    """Build a simple replay strategy with known data."""
    distances = np.array([0.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
    throttle = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5])
    brake = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    torque = np.array([60.0, 50.0, 0.0, 0.0, 40.0, 70.0, 30.0])
    return ReplayStrategy(distances, throttle, brake, torque, lap_distance_m=1000.0)


# ---------------------------------------------------------------------------
# ReplayStrategy
# ---------------------------------------------------------------------------

class TestReplayStrategy:

    def test_throttle_at_known_distance(self):
        strategy = _make_replay_strategy()
        state = SimState(0, 50.0, 10.0, 0.9, 440, 20, 30, 0, 0)
        cmd = strategy.decide(state, [])
        assert cmd.action == ControlAction.THROTTLE
        assert cmd.throttle_pct > 0.5

    def test_brake_at_known_distance(self):
        strategy = _make_replay_strategy()
        state = SimState(0, 400.0, 10.0, 0.9, 440, 20, 30, 0, 0)
        cmd = strategy.decide(state, [])
        assert cmd.action == ControlAction.BRAKE
        assert cmd.brake_pct > 0.3

    def test_coast_at_known_distance(self):
        strategy = _make_replay_strategy()
        state = SimState(0, 200.0, 10.0, 0.9, 440, 20, 30, 0, 0)
        cmd = strategy.decide(state, [])
        assert cmd.action == ControlAction.COAST

    def test_distance_wraps_around_lap(self):
        strategy = _make_replay_strategy()
        # distance 1050 should wrap to 50 (lap=1000m)
        state = SimState(0, 1050.0, 10.0, 0.9, 440, 20, 30, 1, 0)
        cmd = strategy.decide(state, [])
        assert cmd.action == ControlAction.THROTTLE

    def test_name(self):
        strategy = _make_replay_strategy()
        assert strategy.name == "replay"


# ---------------------------------------------------------------------------
# CoastOnlyStrategy
# ---------------------------------------------------------------------------

class TestCoastOnlyStrategy:

    def test_throttle_when_slow(self, dynamics, sim_state_slow, corner_segment):
        strategy = CoastOnlyStrategy(dynamics)
        # corner speed limit ~13.6 m/s, car at 5 m/s → throttle
        cmd = strategy.decide(sim_state_slow, [corner_segment])
        assert cmd.action == ControlAction.THROTTLE
        assert cmd.throttle_pct == 1.0

    def test_coast_when_near_limit(self, dynamics, corner_segment):
        strategy = CoastOnlyStrategy(dynamics, coast_margin_ms=2.0)
        # With downforce, corner speed limit is higher (~15+ m/s)
        limit = dynamics.max_cornering_speed(corner_segment.curvature)
        state = SimState(0, 50.0, limit - 1.0, 0.9, 440, 20, 30, 0, 0)
        cmd = strategy.decide(state, [corner_segment])
        assert cmd.action == ControlAction.COAST

    def test_never_brakes(self, dynamics, sim_state_fast, corner_segment):
        strategy = CoastOnlyStrategy(dynamics)
        # even when speed exceeds limit, should only coast
        state = SimState(0, 50.0, 20.0, 0.9, 440, 30, 30, 0, 0)
        cmd = strategy.decide(state, [corner_segment])
        assert cmd.action != ControlAction.BRAKE

    def test_throttle_with_no_upcoming(self, dynamics, sim_state_fast):
        strategy = CoastOnlyStrategy(dynamics)
        cmd = strategy.decide(sim_state_fast, [])
        assert cmd.action == ControlAction.THROTTLE

    def test_name(self, dynamics):
        assert CoastOnlyStrategy(dynamics).name == "coast_only"


# ---------------------------------------------------------------------------
# ThresholdBrakingStrategy
# ---------------------------------------------------------------------------

class TestThresholdBrakingStrategy:

    def test_throttle_when_slow(self, dynamics, sim_state_slow, corner_segment):
        strategy = ThresholdBrakingStrategy(dynamics)
        cmd = strategy.decide(sim_state_slow, [corner_segment])
        assert cmd.action == ControlAction.THROTTLE

    def test_coast_when_near_limit(self, dynamics, corner_segment):
        strategy = ThresholdBrakingStrategy(dynamics, coast_margin_ms=3.0)
        state = SimState(0, 50.0, 12.0, 0.9, 440, 20, 30, 0, 0)
        cmd = strategy.decide(state, [corner_segment])
        assert cmd.action == ControlAction.COAST

    def test_brake_when_over_limit(self, dynamics, corner_segment):
        strategy = ThresholdBrakingStrategy(dynamics, brake_threshold_ms=1.0)
        # corner limit ~13.6, set speed to 16 → well over limit
        state = SimState(0, 50.0, 16.0, 0.9, 440, 30, 30, 0, 0)
        cmd = strategy.decide(state, [corner_segment])
        assert cmd.action == ControlAction.BRAKE
        assert cmd.brake_pct > 0.0

    def test_brake_intensity_configurable(self, dynamics, corner_segment):
        strategy = ThresholdBrakingStrategy(dynamics, brake_intensity=0.8)
        state = SimState(0, 50.0, 18.0, 0.9, 440, 30, 30, 0, 0)
        cmd = strategy.decide(state, [corner_segment])
        assert cmd.brake_pct == 0.8

    def test_name(self, dynamics):
        assert ThresholdBrakingStrategy(dynamics).name == "threshold_braking"
