"""Tests for CalibratedStrategy zone-based driver model."""

from __future__ import annotations

import pandas as pd
import pytest

from fsae_sim.driver.strategy import ControlAction, ControlCommand, SimState
from fsae_sim.track.track import Segment, Track
from fsae_sim.analysis.telemetry_analysis import DriverZone
from fsae_sim.driver.strategies import CalibratedStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(n_segments: int = 10, length_m: float = 50.0) -> Track:
    segments = []
    for i in range(n_segments):
        curv = 0.0 if i % 4 != 3 else 0.02
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


def make_zones() -> list[DriverZone]:
    """3 zones: throttle (seg 0-3), coast (seg 4-6), brake (seg 7-9)."""
    return [
        DriverZone(0, 0, 3, ControlAction.THROTTLE, 0.8, 0.0, 200.0, "Straight"),
        DriverZone(1, 4, 6, ControlAction.COAST, 0.0, 200.0, 350.0, "Turn 1 entry"),
        DriverZone(2, 7, 9, ControlAction.BRAKE, 0.3, 350.0, 500.0, "Turn 1 apex"),
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_zones_direct(self):
        track = make_track()
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)
        assert strategy.name == "calibrated"
        assert len(strategy.zones) == 3

    def test_from_zone_list(self):
        track = make_track()
        zone_list = [
            {"segments": (0, 4), "action": "throttle", "intensity": 1.0, "label": "Straight"},
            {"segments": (5, 7), "action": "coast", "intensity": 0.0, "label": "Turn 1"},
            {"segments": (8, 9), "action": "brake", "intensity": 0.5, "label": "Turn 1 apex"},
        ]
        strategy = CalibratedStrategy.from_zone_list(zone_list, track)
        assert len(strategy.zones) == 3
        assert strategy.zones[0].action == ControlAction.THROTTLE


# ---------------------------------------------------------------------------
# decide() method
# ---------------------------------------------------------------------------

class TestDecide:
    def test_throttle_zone(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        cmd = strategy.decide(make_state(segment_idx=0), [])
        assert cmd.action == ControlAction.THROTTLE
        assert cmd.throttle_pct == pytest.approx(0.8)
        assert cmd.brake_pct == 0.0

    def test_coast_zone(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        cmd = strategy.decide(make_state(segment_idx=5), [])
        assert cmd.action == ControlAction.COAST
        assert cmd.throttle_pct == 0.0
        assert cmd.brake_pct == 0.0

    def test_brake_zone(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        cmd = strategy.decide(make_state(segment_idx=8), [])
        assert cmd.action == ControlAction.BRAKE
        assert cmd.brake_pct == pytest.approx(0.3)
        assert cmd.throttle_pct == 0.0

    def test_lap_wrapping(self):
        """segment_idx in second lap wraps to first-lap zones."""
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        # Segment 12 % 10 = 2 -> throttle zone
        cmd = strategy.decide(make_state(segment_idx=12), [])
        assert cmd.action == ControlAction.THROTTLE

        # Segment 15 % 10 = 5 -> coast zone
        cmd = strategy.decide(make_state(segment_idx=15), [])
        assert cmd.action == ControlAction.COAST


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_zone_for_segment(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        z = strategy.zone_for_segment(0)
        assert z.zone_id == 0
        assert z.action == ControlAction.THROTTLE

        z = strategy.zone_for_segment(5)
        assert z.zone_id == 1
        assert z.action == ControlAction.COAST

    def test_to_dataframe(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        df = strategy.to_dataframe()
        assert len(df) == 3
        assert "zone_id" in df.columns
        assert "action" in df.columns
        assert "label" in df.columns

    def test_to_driver_brief(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        brief = strategy.to_driver_brief()
        assert isinstance(brief, str)
        assert "Straight" in brief
        assert "Turn 1" in brief

    def test_with_zone_override(self):
        zones = make_zones()
        strategy = CalibratedStrategy(zones, num_segments=10)

        new_strategy = strategy.with_zone_override(
            zone_id=0, action=ControlAction.COAST, intensity=0.0,
        )
        # Original unchanged
        assert strategy.zones[0].action == ControlAction.THROTTLE
        # New strategy has override
        assert new_strategy.zones[0].action == ControlAction.COAST

        # Other zones unchanged
        assert new_strategy.zones[1].action == ControlAction.COAST
        assert new_strategy.zones[2].action == ControlAction.BRAKE
