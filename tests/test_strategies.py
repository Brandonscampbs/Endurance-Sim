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

    def test_d06_preserves_negative_lvcu_torque(self):
        """D-06: LVCU Torque Req < 0 (regen) must be preserved, not clipped to 0.

        Prior to D-06, from_aim_data clipped LVCU Torque Req to [0, 85],
        deleting every regen command. That broke replay energy accounting
        and validation of regen events.
        """
        import pandas as pd
        # Build synthetic AiM DataFrame with negative LVCU Torque Req values.
        n = 50
        dist = np.linspace(0, 100, n)
        aim_df = pd.DataFrame({
            "Distance on GPS Speed": dist,
            "Throttle Pos": np.zeros(n),
            "FBrakePressure": np.zeros(n),
            "RBrakePressure": np.zeros(n),
            # Mix of positive and negative torque commands (regen + drive).
            "LVCU Torque Req": np.linspace(-40.0, 40.0, n),
        })
        strategy = ReplayStrategy.from_aim_data(aim_df, 0, n, lap_distance_m=200.0)
        # Torque at start of window should be ~-40 Nm (regen).
        assert strategy.target_torque(0.0) < -30.0
        # Torque near end (before wrap) should be ~+40 Nm (drive).
        assert strategy.target_torque(99.0) > 30.0

    def test_d06_preserves_negative_lvcu_torque_full_endurance(self):
        """D-06: from_full_endurance must also preserve negative torque."""
        import pandas as pd
        n = 100
        dist = np.linspace(0, 1000, n)
        # Keep GPS Speed > 5 km/h so moving filter keeps everything.
        aim_df = pd.DataFrame({
            "Distance on GPS Speed": dist,
            "GPS Speed": np.full(n, 30.0),
            "Throttle Pos": np.zeros(n),
            "FBrakePressure": np.zeros(n),
            "RBrakePressure": np.zeros(n),
            "LVCU Torque Req": np.linspace(-50.0, 50.0, n),
            "Pack Voltage": np.full(n, 400.0),
            "Pack Current": np.zeros(n),
        })
        strategy = ReplayStrategy.from_full_endurance(aim_df, lap_distance_m=100.0)
        assert strategy.target_torque(0.0) < -30.0


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


class TestControlCommandMetadata:
    """D-27: ControlCommand supports an optional metadata dict."""

    def test_default_metadata_is_none(self):
        from fsae_sim.driver.strategy import ControlAction, ControlCommand
        cmd = ControlCommand(ControlAction.COAST)
        assert cmd.metadata is None

    def test_metadata_roundtrip(self):
        from fsae_sim.driver.strategy import ControlAction, ControlCommand
        meta = {"max_speed_ms": 12.5}
        cmd = ControlCommand(
            ControlAction.THROTTLE,
            throttle_pct=0.8,
            brake_pct=0.0,
            metadata=meta,
        )
        assert cmd.metadata is not None
        assert cmd.metadata["max_speed_ms"] == 12.5
        assert cmd.throttle_pct == 0.8


class TestReplayThresholdConstants:
    """D-24: ReplayStrategy uses module-level constants for action thresholds."""

    def test_constants_exported(self):
        from fsae_sim.driver import strategies as _s
        assert _s._THROTTLE_ACTION_THRESHOLD == pytest.approx(0.05)
        assert _s._BRAKE_ACTION_THRESHOLD == pytest.approx(0.05)

    def test_decide_uses_constants(self, monkeypatch):
        from fsae_sim.driver import strategies as _s
        from fsae_sim.driver.strategy import ControlAction, SimState

        dist = np.array([0.0, 100.0])
        throttle = np.array([0.0, 1.0])
        brake = np.array([0.0, 0.0])
        torque = np.array([0.0, 80.0])

        replay = _s.ReplayStrategy(dist, throttle, brake, torque, 100.0, wrap=False)
        # Raising the threshold should demote a 0.1 throttle to COAST.
        monkeypatch.setattr(_s, "_THROTTLE_ACTION_THRESHOLD", 0.2)
        state = SimState(0, 10.0, 5.0, 0.9, 400, 0, 25, 0, 0)
        cmd = replay.decide(state, [])
        assert cmd.action == ControlAction.COAST


class TestSegmentToZoneIndex:
    """D-25: zone_for_segment uses an O(1) index array, matches linear scan."""

    def test_matches_linear_scan(self):
        from fsae_sim.analysis.telemetry_analysis import DriverZone
        from fsae_sim.driver.strategies import CalibratedStrategy
        from fsae_sim.driver.strategy import ControlAction

        zones = [
            DriverZone(
                zone_id=0, segment_start=0, segment_end=3,
                action=ControlAction.THROTTLE, intensity=0.8,
                distance_start_m=0.0, distance_end_m=200.0, label="a",
            ),
            DriverZone(
                zone_id=1, segment_start=4, segment_end=7,
                action=ControlAction.COAST, intensity=0.0,
                distance_start_m=200.0, distance_end_m=400.0, label="b",
            ),
            DriverZone(
                zone_id=2, segment_start=8, segment_end=9,
                action=ControlAction.BRAKE, intensity=0.5,
                distance_start_m=400.0, distance_end_m=500.0, label="c",
            ),
        ]
        strat = CalibratedStrategy(zones, num_segments=10)

        for seg in range(10):
            fast = strat.zone_for_segment(seg)
            expected = next(z for z in zones if z.segment_start <= seg <= z.segment_end)
            assert fast is expected, f"seg {seg}: fast={fast.zone_id} expected={expected.zone_id}"


class TestEnvelopeAwareStrategies:
    """D-20: CoastOnly/ThresholdBraking honor an injected SpeedEnvelope."""

    def test_coast_only_sees_downstream_corner_via_envelope(self, dynamics, corner_segment):
        from fsae_sim.driver.strategies import CoastOnlyStrategy
        from fsae_sim.driver.strategy import SimState
        from fsae_sim.track.track import Segment

        straight = Segment(index=0, distance_start_m=0.0, length_m=5.0, curvature=0.0, grade=0.0)
        # Synthetic envelope: segment 0 ceiling forced low by a downstream corner.
        envelope = np.array([5.0, 20.0, 20.0])
        strat_no_env = CoastOnlyStrategy(dynamics)
        strat_env = CoastOnlyStrategy(dynamics, envelope=envelope)

        state = SimState(0, 0.0, 8.0, 0.9, 400, 0, 25, 0, 0)
        # With envelope, segment 0 ceiling is 5 m/s so 8 m/s ⇒ coast.
        cmd_env = strat_env.decide(state, [straight])
        assert cmd_env.action == ControlAction.COAST

        # Without envelope, the straight has no corner limit ⇒ throttle.
        cmd_raw = strat_no_env.decide(state, [straight])
        assert cmd_raw.action == ControlAction.THROTTLE

    def test_threshold_braking_envelope_brakes_earlier(self, dynamics):
        from fsae_sim.driver.strategies import ThresholdBrakingStrategy
        from fsae_sim.driver.strategy import SimState
        from fsae_sim.track.track import Segment

        # Two "straights" but envelope marks both as pre-corner slow zones.
        upcoming = [
            Segment(index=0, distance_start_m=0.0, length_m=5.0, curvature=0.0, grade=0.0),
            Segment(index=1, distance_start_m=5.0, length_m=5.0, curvature=0.0, grade=0.0),
        ]
        envelope = np.array([8.0, 8.0])  # envelope says 8 m/s max here
        state = SimState(0, 0.0, 20.0, 0.9, 400, 30, 25, 0, 0)

        strat_env = ThresholdBrakingStrategy(
            dynamics, brake_threshold_ms=1.0, brake_intensity=0.5, envelope=envelope,
        )
        strat_raw = ThresholdBrakingStrategy(
            dynamics, brake_threshold_ms=1.0, brake_intensity=0.5,
        )

        assert strat_env.decide(state, upcoming).action == ControlAction.BRAKE
        # Raw strategy sees inf corner speed on straights ⇒ no brake.
        assert strat_raw.decide(state, upcoming).action != ControlAction.BRAKE


class TestCalibratedStrategyWithParams:
    """D-28: CalibratedStrategy.with_params applies DriverParams multipliers."""

    def _zones(self):
        from fsae_sim.analysis.telemetry_analysis import DriverZone
        return [
            DriverZone(
                zone_id=0, segment_start=0, segment_end=9,
                action=ControlAction.THROTTLE, intensity=0.8,
                distance_start_m=0.0, distance_end_m=500.0, label="all",
            ),
        ]

    def test_throttle_scale_halves_intensity(self):
        from fsae_sim.driver.strategies import CalibratedStrategy, DriverParams
        from fsae_sim.driver.strategy import SimState

        baseline = CalibratedStrategy(self._zones(), num_segments=10)
        scaled = baseline.with_params(DriverParams(throttle_scale=0.5))

        state = SimState(0, 0.0, 10.0, 0.9, 400, 0, 25, 0, 3)
        base_cmd = baseline.decide(state, [])
        scaled_cmd = scaled.decide(state, [])

        assert base_cmd.throttle_pct == pytest.approx(0.8)
        assert scaled_cmd.throttle_pct == pytest.approx(0.4)

    def test_max_throttle_caps_value(self):
        from fsae_sim.driver.strategies import CalibratedStrategy, DriverParams
        from fsae_sim.driver.strategy import SimState

        baseline = CalibratedStrategy(self._zones(), num_segments=10)
        capped = baseline.with_params(DriverParams(max_throttle=0.3))

        state = SimState(0, 0.0, 10.0, 0.9, 400, 0, 25, 0, 3)
        cmd = capped.decide(state, [])
        assert cmd.throttle_pct == pytest.approx(0.3)
