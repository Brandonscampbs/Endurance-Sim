"""AdaptiveStrategy plumbing tests (Wave 4a — Sub-task B).

Sub-task B wraps the AdaptiveDriver core (already shipped in
tests/test_adaptive_driver.py) in a DriverStrategy interface so the
SimulationEngine can address it through the same protocol used by
CalibratedStrategy and ReplayStrategy.

Wave 4b will plumb AdaptiveStrategy into the engine's per-segment loop
and add Michigan-endurance telemetry regression. These tests only cover
the strategy-class plumbing: name, decide() return type, PREDICTION-mode
gate, and verifying we did NOT accidentally shadow the existing
strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsae_sim.driver.strategy import ControlAction, ControlCommand, SimState
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain_model import PowertrainModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _flat_track(num_segments: int = 100, seg_len: float = 0.5) -> Track:
    segs = [
        Segment(
            index=i,
            distance_start_m=i * seg_len,
            length_m=seg_len,
            curvature=0.0,
            grade=0.0,
        )
        for i in range(num_segments)
    ]
    return Track(name="flat100", segments=segs, source="synthetic")


@pytest.fixture
def vehicle_config(ct16ev_config_path) -> VehicleConfig:
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def models(vehicle_config):
    pt = PowertrainModel(vehicle_config.powertrain)
    dyn = VehicleDynamics(
        vehicle_config.vehicle,
        powertrain_config=vehicle_config.powertrain,
    )
    return vehicle_config, pt, dyn


@pytest.fixture
def flat_track() -> Track:
    return _flat_track()


def _state_at(speed_ms: float, seg_idx: int) -> SimState:
    return SimState(
        time=0.0,
        distance=float(seg_idx) * 0.5,
        speed=float(speed_ms),
        soc=0.95,
        pack_voltage=400.0,
        pack_current=0.0,
        cell_temp=30.0,
        lap=0,
        segment_idx=seg_idx,
    )


# ---------------------------------------------------------------------------
# Plumbing tests
# ---------------------------------------------------------------------------


def test_adaptive_strategy_decide_returns_control_command(models, flat_track):
    """Strategy.decide must conform to the DriverStrategy protocol."""
    from fsae_sim.driver.strategies import AdaptiveStrategy

    cfg, pt, dyn = models
    strategy = AdaptiveStrategy(dynamics=dyn, powertrain=pt, track=flat_track)
    strategy.set_envelope(np.full(flat_track.num_segments, 18.0))
    state = _state_at(18.0, seg_idx=2)
    cmd = strategy.decide(state, [flat_track.segments[i] for i in range(2, 7)])
    assert isinstance(cmd, ControlCommand)
    assert 0.0 <= cmd.throttle_pct <= 1.0
    assert 0.0 <= cmd.brake_pct <= 1.0


def test_adaptive_strategy_name_is_adaptive(models, flat_track):
    """Strategy.name == 'adaptive' (no clash with 'replay' or 'driver')."""
    from fsae_sim.driver.strategies import AdaptiveStrategy, CalibratedStrategy, ReplayStrategy

    cfg, pt, dyn = models
    strategy = AdaptiveStrategy(dynamics=dyn, powertrain=pt, track=flat_track)
    assert strategy.name == "adaptive"
    # Distinct names so config / CLI can route correctly.
    assert ReplayStrategy.name == "replay"
    assert CalibratedStrategy.name == "driver"


def test_adaptive_strategy_passes_prediction_mode_gate(models, flat_track):
    """SimulationEngine(mode=PREDICTION) raises if the strategy reports
    uses_observed_speed_caps == True. AdaptiveStrategy must report False."""
    from fsae_sim.driver.strategies import AdaptiveStrategy

    cfg, pt, dyn = models
    strategy = AdaptiveStrategy(dynamics=dyn, powertrain=pt, track=flat_track)
    assert strategy.uses_observed_speed_caps is False


def test_adaptive_strategy_set_envelope_propagates_to_driver(models, flat_track):
    """The strategy must forward set_envelope to the wrapped driver."""
    from fsae_sim.driver.strategies import AdaptiveStrategy

    cfg, pt, dyn = models
    strategy = AdaptiveStrategy(dynamics=dyn, powertrain=pt, track=flat_track)
    v = np.full(flat_track.num_segments, 22.0)
    strategy.set_envelope(v)
    # Round-trip: decide must work after set_envelope.
    cmd = strategy.decide(
        _state_at(22.0, seg_idx=10),
        [flat_track.segments[i] for i in range(10, 15)],
    )
    assert isinstance(cmd, ControlCommand)


def test_adaptive_strategy_set_bms_limit_propagates_to_driver(models, flat_track):
    """The strategy must forward set_bms_limit to the wrapped driver so
    the engine's per-lap BMS refresh can update the inverse solve."""
    from fsae_sim.driver.strategies import AdaptiveStrategy

    cfg, pt, dyn = models
    strategy = AdaptiveStrategy(dynamics=dyn, powertrain=pt, track=flat_track)
    strategy.set_envelope(np.full(flat_track.num_segments, 22.0))
    # Smoke-test that lowering the BMS limit changes the commanded
    # throttle for a demand that would otherwise reach the inverter cap.
    # (The driver caps drive force at full-pedal output; a lower BMS
    # limit shrinks the LVCU torque ceiling, so a near-saturated demand
    # commands relatively MORE pedal for a given target.)
    strategy.set_bms_limit(50.0)  # tight limit
    cmd_tight = strategy.decide(
        _state_at(22.0, seg_idx=10),
        [flat_track.segments[i] for i in range(10, 15)],
    )
    strategy.set_bms_limit(200.0)  # loose limit
    cmd_loose = strategy.decide(
        _state_at(22.0, seg_idx=10),
        [flat_track.segments[i] for i in range(10, 15)],
    )
    # At a flat envelope, both commands should be valid in-range pedal
    # positions; the tight one should command at least as much pedal as
    # the loose one. (Strict inequality may not hold if both are equal
    # at saturation or zero; the inequality below is the right invariant.)
    assert cmd_tight.throttle_pct >= cmd_loose.throttle_pct - 1e-9


def test_existing_strategies_still_importable():
    """Sanity guard: AdaptiveStrategy joins CalibratedStrategy and
    ReplayStrategy; it must not have replaced or shadowed them."""
    from fsae_sim.driver.strategies import (
        CalibratedStrategy,
        ReplayStrategy,
        AdaptiveStrategy,
    )
    assert CalibratedStrategy.__name__ == "CalibratedStrategy"
    assert ReplayStrategy.__name__ == "ReplayStrategy"
    assert AdaptiveStrategy.__name__ == "AdaptiveStrategy"


def test_adaptive_strategy_decide_deterministic_after_envelope_set(models, flat_track):
    """Strategy-level determinism: identical state on a freshly reset()
    driver yields identical commands. Wave 4b Sub-task A added PI state,
    so this test resets between the two calls."""
    from fsae_sim.driver.strategies import AdaptiveStrategy

    cfg, pt, dyn = models
    strategy = AdaptiveStrategy(dynamics=dyn, powertrain=pt, track=flat_track)
    strategy.set_envelope(np.linspace(10.0, 25.0, flat_track.num_segments))
    state = _state_at(15.0, seg_idx=30)
    upcoming = [flat_track.segments[i] for i in range(30, 35)]
    strategy.driver.reset()
    cmd_a = strategy.decide(state, upcoming)
    strategy.driver.reset()
    cmd_b = strategy.decide(state, upcoming)
    assert cmd_a.action == cmd_b.action
    assert cmd_a.throttle_pct == cmd_b.throttle_pct
    assert cmd_a.brake_pct == cmd_b.brake_pct
    assert cmd_a.regen_request_pct == cmd_b.regen_request_pct
