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


# ---------------------------------------------------------------------------
# C13: Zone intensity is load-bearing for runtime decide()
# ---------------------------------------------------------------------------

class TestZoneIntensityIsAuthoritative:
    """Zones are the unit of control. Per-segment intensities are calibration
    artifacts; runtime decide() must read zone.intensity, not _segment_actions.
    Otherwise to_driver_brief() and the simulated behavior diverge.
    """

    def test_decide_returns_zone_intensity_not_segment(self):
        """Build zones with intensity=0.8, pass segment_intensities=[0.2]*10.
        decide() must return 0.8 (zone-level), NOT 0.2 (per-segment).
        """
        import numpy as np
        zones = [
            DriverZone(
                zone_id=0, segment_start=0, segment_end=9,
                action=ControlAction.THROTTLE, intensity=0.8,
                distance_start_m=0.0, distance_end_m=500.0,
                label="Full Straight",
            )
        ]
        per_segment = np.full(10, 0.2)
        strategy = CalibratedStrategy(
            zones, num_segments=10, segment_intensities=per_segment,
        )

        for seg_idx in range(10):
            cmd = strategy.decide(make_state(segment_idx=seg_idx), [])
            assert cmd.action == ControlAction.THROTTLE
            assert cmd.throttle_pct == pytest.approx(0.8), (
                f"segment {seg_idx} returned {cmd.throttle_pct}, expected 0.8"
            )

    def test_brief_matches_decide(self):
        """to_driver_brief() and decide() must agree on intensity."""
        import numpy as np
        zones = [
            DriverZone(
                zone_id=0, segment_start=0, segment_end=4,
                action=ControlAction.THROTTLE, intensity=0.75,
                distance_start_m=0.0, distance_end_m=250.0, label="Straight",
            ),
            DriverZone(
                zone_id=1, segment_start=5, segment_end=9,
                action=ControlAction.BRAKE, intensity=0.40,
                distance_start_m=250.0, distance_end_m=500.0, label="Turn 1",
            ),
        ]
        per_segment = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        strategy = CalibratedStrategy(
            zones, num_segments=10, segment_intensities=per_segment,
        )

        brief = strategy.to_driver_brief()
        # Brief reports 75% throttle in zone 0
        assert "75% throttle" in brief
        # And decide() must also use 75% throttle
        cmd = strategy.decide(make_state(segment_idx=2), [])
        assert cmd.throttle_pct == pytest.approx(0.75)
