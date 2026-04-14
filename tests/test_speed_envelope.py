"""Tests for forward-backward speed envelope solver."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from fsae_sim.track.track import Segment, Track
from fsae_sim.sim.speed_envelope import SpeedEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_segment(index: int, curvature: float = 0.0, length: float = 5.0,
                 grade: float = 0.0, grip_factor: float = 1.0) -> Segment:
    return Segment(
        index=index,
        distance_start_m=index * length,
        length_m=length,
        curvature=curvature,
        grade=grade,
        grip_factor=grip_factor,
    )


def make_track(segments: list[Segment], name: str = "test") -> Track:
    return Track(name=name, segments=segments)


def make_dynamics(mass_kg: float = 288.0, corner_speed_fn=None,
                  resistance_fn=None, max_traction_fn=None,
                  max_braking_fn=None):
    """Mock VehicleDynamics with configurable behavior."""
    dyn = MagicMock()
    dyn.vehicle = MagicMock()
    dyn.vehicle.mass_kg = mass_kg
    dyn.m_effective = mass_kg + 37.0  # ~325 kg effective

    if corner_speed_fn is None:
        def corner_speed_fn(curvature, grip_factor=1.0):
            kappa = abs(curvature)
            if kappa < 1e-6:
                return float("inf")
            return math.sqrt(1.3 * 9.81 / kappa) * grip_factor
    dyn.max_cornering_speed.side_effect = corner_speed_fn

    if resistance_fn is None:
        resistance_fn = lambda v, grade=0.0: 50.0 + 0.5 * v * v
    dyn.total_resistance.side_effect = resistance_fn

    if max_traction_fn is None:
        max_traction_fn = lambda v: 3000.0
    dyn.max_traction_force.side_effect = max_traction_fn

    if max_braking_fn is None:
        max_braking_fn = lambda v: 4000.0
    dyn.max_braking_force.side_effect = max_braking_fn

    return dyn


def make_powertrain(max_drive_force: float = 2000.0, max_regen_force: float = -800.0):
    """Mock PowertrainModel."""
    pt = MagicMock()
    pt.drive_force.side_effect = lambda throttle, v: throttle * max_drive_force
    pt.regen_force.side_effect = lambda brake, v: brake * max_regen_force
    return pt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStraightOnlyTrack:
    """All-straight track: envelope limited by acceleration only."""

    def test_speeds_increase_monotonically_from_rest(self):
        segs = [make_segment(i) for i in range(20)]
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v_max = env.compute(initial_speed=0.5)

        assert len(v_max) == 20
        assert v_max[-1] > v_max[0]
        assert np.all(v_max > 0)

    def test_envelope_respects_initial_speed(self):
        segs = [make_segment(i) for i in range(10)]
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v_slow = env.compute(initial_speed=0.5)
        v_fast = env.compute(initial_speed=15.0)

        assert v_fast[0] >= v_slow[0]


class TestSingleCorner:
    """Straight-corner-straight: braking zone must appear before corner."""

    def test_braking_zone_before_corner(self):
        segs = (
            [make_segment(i) for i in range(10)]
            + [make_segment(i + 10, curvature=0.1) for i in range(5)]
            + [make_segment(i + 15) for i in range(10)]
        )
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v_max = env.compute(initial_speed=0.5)

        corner_speed = math.sqrt(1.3 * 9.81 / 0.1)

        for i in range(10, 15):
            assert v_max[i] <= corner_speed + 0.5

        assert v_max[9] < v_max[5]

    def test_acceleration_zone_after_corner(self):
        segs = (
            [make_segment(i) for i in range(5)]
            + [make_segment(i + 5, curvature=0.1) for i in range(5)]
            + [make_segment(i + 10) for i in range(10)]
        )
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v_max = env.compute(initial_speed=0.5)

        assert v_max[15] > v_max[10]


class TestEnvelopeIsPhysicallyValid:
    """The envelope must not require impossible deceleration."""

    def test_no_segment_exceeds_max_deceleration(self):
        segs = (
            [make_segment(i) for i in range(15)]
            + [make_segment(i + 15, curvature=0.15) for i in range(3)]
            + [make_segment(i + 18) for i in range(10)]
        )
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v_max = env.compute(initial_speed=0.5)

        m_eff = dyn.m_effective
        for i in range(len(v_max) - 1):
            if v_max[i + 1] < v_max[i]:
                dv_sq = v_max[i] ** 2 - v_max[i + 1] ** 2
                a_required = dv_sq / (2 * segs[i].length_m)
                max_decel = (4000.0 + 50.0 + 0.5 * v_max[i] ** 2) / m_eff
                assert a_required <= max_decel * 1.1


class TestEnvelopeAlwaysLECornerSpeed:
    """Envelope speed must never exceed corner speed limit at any segment."""

    def test_envelope_bounded_by_corner_speed(self):
        segs = (
            [make_segment(i) for i in range(5)]
            + [make_segment(i + 5, curvature=0.08) for i in range(5)]
            + [make_segment(i + 10) for i in range(5)]
        )
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v_max = env.compute(initial_speed=0.5)

        for i, seg in enumerate(segs):
            corner_limit = dyn.max_cornering_speed(seg.curvature, seg.grip_factor)
            assert v_max[i] <= corner_limit + 0.5


class TestCornerSpeedCache:
    """Corner speeds should be cached and reused across compute() calls."""

    def test_cache_hit_returns_same_result(self):
        segs = [make_segment(i, curvature=0.05 if i % 5 == 0 else 0.0) for i in range(20)]
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        v1 = env.compute(initial_speed=0.5)
        call_count_after_first = dyn.max_cornering_speed.call_count

        v2 = env.compute(initial_speed=0.5)
        call_count_after_second = dyn.max_cornering_speed.call_count

        assert call_count_after_second == call_count_after_first
        np.testing.assert_array_equal(v1, v2)

    def test_different_initial_speed_reuses_cache(self):
        segs = [make_segment(i, curvature=0.05 if i % 3 == 0 else 0.0) for i in range(15)]
        track = make_track(segs)
        dyn = make_dynamics()
        pt = make_powertrain()

        env = SpeedEnvelope(dyn, pt, track)
        env.compute(initial_speed=0.5)
        count1 = dyn.max_cornering_speed.call_count

        env.compute(initial_speed=10.0)
        count2 = dyn.max_cornering_speed.call_count

        assert count2 == count1

    def test_different_track_invalidates_cache(self):
        segs1 = [make_segment(i) for i in range(10)]
        segs2 = [make_segment(i, curvature=0.05) for i in range(10)]
        track1 = make_track(segs1, name="track1")
        track2 = make_track(segs2, name="track2")
        dyn = make_dynamics()
        pt = make_powertrain()

        env1 = SpeedEnvelope(dyn, pt, track1)
        env1.compute(initial_speed=0.5)
        count1 = dyn.max_cornering_speed.call_count

        env2 = SpeedEnvelope(dyn, pt, track2)
        env2.compute(initial_speed=0.5)
        count2 = dyn.max_cornering_speed.call_count

        assert count2 > count1
