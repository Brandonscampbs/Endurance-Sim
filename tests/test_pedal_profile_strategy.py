"""Tests for PedalProfileStrategy pedal-profile driver model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsae_sim.driver.strategy import ControlAction, ControlCommand, SimState
from fsae_sim.track.track import Segment, Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(n_segments: int = 10, length_m: float = 5.0) -> Track:
    segments = []
    for i in range(n_segments):
        curv = 0.0 if i < 7 else 0.02
        segments.append(Segment(
            index=i, distance_start_m=i * length_m,
            length_m=length_m, curvature=curv, grade=0.0,
        ))
    return Track(name="test", segments=segments)


def make_state(segment_idx: int = 0, speed: float = 10.0, lap: int = 0) -> SimState:
    return SimState(
        time=0.0, distance=0.0, speed=speed,
        soc=0.9, pack_voltage=350.0, pack_current=0.0,
        cell_temp=30.0, lap=lap, segment_idx=segment_idx,
    )


# ---------------------------------------------------------------------------
# DriverParams
# ---------------------------------------------------------------------------

class TestDriverParams:
    def test_defaults(self):
        from fsae_sim.driver.strategies import DriverParams
        p = DriverParams()
        assert p.throttle_scale == 1.0
        assert p.brake_scale == 1.0
        assert p.coast_throttle == 0.0
        assert p.max_throttle == 1.0
        assert p.max_brake == 1.0

    def test_custom_values(self):
        from fsae_sim.driver.strategies import DriverParams
        p = DriverParams(throttle_scale=0.8, brake_scale=1.2)
        assert p.throttle_scale == 0.8
        assert p.brake_scale == 1.2


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_construction(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy, DriverParams
        n = 10
        throttle = np.array([0.5, 0.6, 0.7, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4])
        brake = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0])
        actions = np.array([1, 1, 1, 1, 1, 0, 0, 2, 2, 1])
        ref_speed = np.ones(n) * 12.0

        strategy = PedalProfileStrategy(
            throttle_pct=throttle, brake_pct=brake,
            actions=actions, ref_speed_ms=ref_speed,
            num_segments=n,
        )
        assert strategy.name == "pedal_profile"
        assert strategy.num_segments == n

    def test_rejects_mismatched_lengths(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy
        with pytest.raises(ValueError, match="same length"):
            PedalProfileStrategy(
                throttle_pct=np.array([0.5, 0.6]),
                brake_pct=np.array([0.0]),
                actions=np.array([1, 1]),
                ref_speed_ms=np.array([10.0, 10.0]),
                num_segments=2,
            )


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------

class TestDecide:
    def _make_strategy(self, **kwargs):
        from fsae_sim.driver.strategies import PedalProfileStrategy, DriverParams
        n = 10
        throttle = np.array([0.5, 0.6, 0.7, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4])
        brake = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0])
        actions = np.array([1, 1, 1, 1, 1, 0, 0, 2, 2, 1])
        ref_speed = np.ones(n) * 12.0
        params = DriverParams(**kwargs) if kwargs else DriverParams()
        return PedalProfileStrategy(
            throttle_pct=throttle, brake_pct=brake,
            actions=actions, ref_speed_ms=ref_speed,
            num_segments=n, params=params,
        )

    def test_throttle_segment(self):
        strategy = self._make_strategy()
        cmd = strategy.decide(make_state(segment_idx=2), [])
        assert cmd.action == ControlAction.THROTTLE
        assert cmd.throttle_pct == pytest.approx(0.7)
        assert cmd.brake_pct == 0.0

    def test_coast_segment(self):
        strategy = self._make_strategy()
        cmd = strategy.decide(make_state(segment_idx=5), [])
        assert cmd.action == ControlAction.COAST
        assert cmd.throttle_pct == 0.0
        assert cmd.brake_pct == 0.0

    def test_brake_segment(self):
        strategy = self._make_strategy()
        cmd = strategy.decide(make_state(segment_idx=7), [])
        assert cmd.action == ControlAction.BRAKE
        assert cmd.brake_pct == pytest.approx(0.3)
        assert cmd.throttle_pct == 0.0

    def test_lap_wrapping(self):
        strategy = self._make_strategy()
        # Segment 12 % 10 = 2 -> throttle at 0.7
        cmd = strategy.decide(make_state(segment_idx=12), [])
        assert cmd.action == ControlAction.THROTTLE
        assert cmd.throttle_pct == pytest.approx(0.7)

    def test_throttle_scale(self):
        strategy = self._make_strategy(throttle_scale=0.5)
        cmd = strategy.decide(make_state(segment_idx=2), [])
        assert cmd.action == ControlAction.THROTTLE
        assert cmd.throttle_pct == pytest.approx(0.35)  # 0.7 * 0.5

    def test_brake_scale(self):
        strategy = self._make_strategy(brake_scale=2.0)
        cmd = strategy.decide(make_state(segment_idx=7), [])
        assert cmd.action == ControlAction.BRAKE
        assert cmd.brake_pct == pytest.approx(0.6)  # 0.3 * 2.0

    def test_max_throttle_cap(self):
        strategy = self._make_strategy(max_throttle=0.5)
        # Segment 3 has throttle 0.8, should be capped to 0.5
        cmd = strategy.decide(make_state(segment_idx=3), [])
        assert cmd.throttle_pct == pytest.approx(0.5)

    def test_max_brake_cap(self):
        strategy = self._make_strategy(max_brake=0.4)
        # Segment 8 has brake 0.5, should be capped to 0.4
        cmd = strategy.decide(make_state(segment_idx=8), [])
        assert cmd.brake_pct == pytest.approx(0.4)

    def test_coast_throttle(self):
        strategy = self._make_strategy(coast_throttle=0.05)
        cmd = strategy.decide(make_state(segment_idx=5), [])
        assert cmd.action == ControlAction.COAST
        assert cmd.throttle_pct == pytest.approx(0.05)

    def test_throttle_clamped_to_one(self):
        strategy = self._make_strategy(throttle_scale=2.0)
        # Segment 3 has 0.8 * 2.0 = 1.6, should clamp to 1.0
        cmd = strategy.decide(make_state(segment_idx=3), [])
        assert cmd.throttle_pct == pytest.approx(1.0)

    def test_brake_clamped_to_one(self):
        strategy = self._make_strategy(brake_scale=3.0)
        # Segment 8 has 0.5 * 3.0 = 1.5, should clamp to 1.0
        cmd = strategy.decide(make_state(segment_idx=8), [])
        assert cmd.brake_pct == pytest.approx(1.0)
