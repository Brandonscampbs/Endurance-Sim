"""Tests for telemetry extraction and zone collapsing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from fsae_sim.driver.strategy import ControlAction
from fsae_sim.track.track import Segment, Track

from fsae_sim.analysis.telemetry_analysis import (
    extract_per_segment_actions,
    collapse_to_zones,
    detect_laps,
    extract_tire_grip_scale,
    DriverZone,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(n_segments: int = 10, length_m: float = 50.0) -> Track:
    """Build a simple track with alternating straight/curved segments."""
    segments = []
    for i in range(n_segments):
        curv = 0.0 if i % 3 != 2 else 0.02  # every 3rd segment is a corner
        segments.append(Segment(
            index=i,
            distance_start_m=i * length_m,
            length_m=length_m,
            curvature=curv,
            grade=0.0,
        ))
    return Track(name="test_track", segments=segments)


def make_telemetry(
    track: Track,
    actions: list[str],
    intensities: list[float],
    speed_kmh: float = 40.0,
    num_laps: int = 1,
) -> pd.DataFrame:
    """Build synthetic AiM-like DataFrame matching a track.

    actions: list of "throttle", "coast", "brake" per segment.
    intensities: throttle_pct (0-100) or brake pressure per segment.
    """
    rows = []
    lap_dist = track.total_distance_m
    t = 0.0
    for lap in range(num_laps):
        for i, seg in enumerate(track.segments):
            mid_dist = seg.distance_start_m + seg.length_m / 2 + lap * lap_dist
            dt = seg.length_m / (speed_kmh / 3.6)
            action = actions[i % len(actions)]
            intensity = intensities[i % len(intensities)]
            rows.append({
                "Distance on GPS Speed": mid_dist,
                "GPS Speed": speed_kmh,
                "Throttle Pos": intensity if action == "throttle" else 0.0,
                "FBrakePressure": intensity if action == "brake" else 0.0,
                "RBrakePressure": 0.0,
                "Time": t,
            })
            t += dt
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# extract_per_segment_actions
# ---------------------------------------------------------------------------

class TestExtractPerSegmentActions:
    def test_all_throttle(self):
        track = make_track(5, 50.0)
        actions = ["throttle"] * 5
        intensities = [80.0] * 5
        df = make_telemetry(track, actions, intensities)

        result = extract_per_segment_actions(df, track)

        assert len(result) == 5
        assert all(result["action"] == ControlAction.THROTTLE)
        assert result["intensity"].iloc[0] == pytest.approx(0.8, abs=0.05)

    def test_mixed_actions(self):
        track = make_track(6, 50.0)
        actions = ["throttle", "throttle", "coast", "brake", "throttle", "coast"]
        intensities = [100.0, 80.0, 0.0, 30.0, 100.0, 0.0]
        df = make_telemetry(track, actions, intensities)

        result = extract_per_segment_actions(df, track)

        assert result["action"].iloc[0] == ControlAction.THROTTLE
        assert result["action"].iloc[2] == ControlAction.COAST
        assert result["action"].iloc[3] == ControlAction.BRAKE

    def test_correct_column_set(self):
        track = make_track(3)
        df = make_telemetry(track, ["throttle"] * 3, [50.0] * 3)
        result = extract_per_segment_actions(df, track)

        expected_cols = {
            "segment_idx", "distance_m", "curvature",
            "mean_throttle_pct", "mean_brake_bar",
            "mean_speed_kmh", "action", "intensity",
        }
        assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# collapse_to_zones
# ---------------------------------------------------------------------------

class TestCollapseToZones:
    def test_same_action_merges(self):
        track = make_track(6, 50.0)
        # All throttle at similar intensity -> should merge to 1-2 zones
        actions = ["throttle"] * 6
        intensities = [80.0] * 6
        df = make_telemetry(track, actions, intensities)

        seg_df = extract_per_segment_actions(df, track)
        zones = collapse_to_zones(seg_df, track)

        # All same action+intensity should collapse to 1 zone
        assert len(zones) == 1
        assert zones[0].action == ControlAction.THROTTLE

    def test_different_actions_separate(self):
        track = make_track(6, 50.0)
        actions = ["throttle", "throttle", "coast", "coast", "throttle", "throttle"]
        intensities = [80.0, 80.0, 0.0, 0.0, 80.0, 80.0]
        df = make_telemetry(track, actions, intensities)

        seg_df = extract_per_segment_actions(df, track)
        zones = collapse_to_zones(seg_df, track)

        assert len(zones) == 3
        assert zones[0].action == ControlAction.THROTTLE
        assert zones[1].action == ControlAction.COAST
        assert zones[2].action == ControlAction.THROTTLE

    def test_zone_distances_cover_track(self):
        track = make_track(10, 50.0)
        actions = ["throttle", "throttle", "coast", "brake",
                   "throttle", "throttle", "throttle", "coast",
                   "brake", "throttle"]
        intensities = [80.0, 80.0, 0.0, 20.0,
                      100.0, 100.0, 100.0, 0.0,
                      30.0, 90.0]
        df = make_telemetry(track, actions, intensities)

        seg_df = extract_per_segment_actions(df, track)
        zones = collapse_to_zones(seg_df, track)

        # First zone starts at segment 0
        assert zones[0].segment_start == 0
        # Last zone ends at last segment
        assert zones[-1].segment_end == 9
        # No gaps
        for i in range(len(zones) - 1):
            assert zones[i].segment_end + 1 == zones[i + 1].segment_start

    def test_zone_has_label(self):
        track = make_track(6, 50.0)
        actions = ["throttle"] * 6
        intensities = [80.0] * 6
        df = make_telemetry(track, actions, intensities)

        seg_df = extract_per_segment_actions(df, track)
        zones = collapse_to_zones(seg_df, track)

        assert all(isinstance(z.label, str) for z in zones)
        assert all(len(z.label) > 0 for z in zones)

    def test_merge_tolerance(self):
        """Slightly different intensities within tolerance should merge."""
        track = make_track(4, 50.0)
        actions = ["throttle"] * 4
        # Small intensity variation within default tolerance (0.05)
        intensities = [80.0, 82.0, 79.0, 81.0]
        df = make_telemetry(track, actions, intensities)

        seg_df = extract_per_segment_actions(df, track)
        zones = collapse_to_zones(seg_df, track, merge_tolerance=0.05)

        # All within tolerance => single zone
        assert len(zones) == 1

    def test_max_speed_ms_tracks_peak_not_mean(self):
        """D-10: max_speed_ms must reflect near-peak speed, not the mean.

        Feed a zone with increasing per-segment speeds (synthetic ramp).
        ``np.mean`` would yield ~30 kph; the post-fix 95th-percentile
        must be ≥ 49 kph (converted to m/s, ~13.6).
        """
        track = make_track(5, 50.0)
        # Single-zone throttle with the speeds ramping segment-by-segment.
        seg_df = pd.DataFrame({
            "segment_idx": list(range(5)),
            "distance_m": [seg.distance_start_m + seg.length_m / 2 for seg in track.segments],
            "curvature": [seg.curvature for seg in track.segments],
            "mean_throttle_pct": [80.0] * 5,
            "mean_brake_bar": [0.0] * 5,
            "mean_speed_kmh": [10.0, 20.0, 30.0, 40.0, 50.0],
            "action": [ControlAction.THROTTLE] * 5,
            "intensity": [0.8] * 5,
        })

        zones = collapse_to_zones(seg_df, track)

        assert len(zones) == 1
        peak_kmh = zones[0].max_speed_ms * 3.6
        # 95th percentile of [10,20,30,40,50] = 48; mean would be 30.
        # Guard against a regression to the mean: must exceed 45.
        assert peak_kmh > 45.0, (
            f"D-10 regression: max_speed_ms {peak_kmh:.2f} kph ≤ 45; "
            "collapse_to_zones is averaging instead of peaking."
        )


# ---------------------------------------------------------------------------
# detect_laps
# ---------------------------------------------------------------------------

class TestDetectLaps:
    def test_basic_detection(self):
        """Synthetic data with clear distance rollover pattern."""
        # Simulate 3 laps of 1000m each
        n_per_lap = 200
        rows = []
        for lap in range(3):
            for i in range(n_per_lap):
                dist = lap * 1000.0 + (i / n_per_lap) * 1000.0
                rows.append({"Distance on GPS Speed": dist, "GPS Speed": 40.0})

        df = pd.DataFrame(rows)
        laps = detect_laps(df, lap_distance_m=1000.0)

        assert len(laps) >= 2  # at least 2 laps detected
        for start, end, lap_time in laps:
            assert start < end


# ---------------------------------------------------------------------------
# D-07: per-lap distance rescale
# ---------------------------------------------------------------------------


class TestD18AutoSelectLaps:
    """D-18: the auto-lap-select filter rejects the driver-change lap
    (normal distance, extended time, low mean speed) — old filter only
    checked distance and let it through."""

    def test_driver_change_lap_rejected(self):
        from fsae_sim.analysis.telemetry_analysis import _auto_select_laps

        # 4 laps: normal, normal, driver-change, normal. Driver-change
        # has normal distance but 50% longer time and 50% slower mean speed.
        base_time = 60.0  # 60s per normal lap
        base_speed = 40.0  # km/h
        n_per_lap = 100
        rows = []
        boundaries = []
        row_idx = 0
        t = 0.0
        for lap in range(4):
            is_driver_change = (lap == 2)
            lap_time = base_time * 1.5 if is_driver_change else base_time
            lap_speed = base_speed * 0.5 if is_driver_change else base_speed
            start_idx = row_idx
            dt = lap_time / n_per_lap
            for i in range(n_per_lap):
                rows.append({
                    "Time": t,
                    "Distance on GPS Speed": row_idx * 5.0,  # unused
                    "GPS Speed": lap_speed,
                })
                t += dt
                row_idx += 1
            # All laps have the same "distance" (sanity — filter needs
            # distance matching median).
            boundaries.append((start_idx, row_idx, 800.0))

        df = pd.DataFrame(rows)
        selected = _auto_select_laps(df, boundaries)
        # Lap 0 skipped (warmup); lap 2 (driver change) rejected.
        # Expect only laps 1 and 3.
        start_indices = [s for s, _, _ in selected]
        expected_starts = {boundaries[1][0], boundaries[3][0]}
        assert set(start_indices) == expected_starts, (
            f"Expected laps 1 & 3, got start_indices={start_indices!r}"
        )


class TestD07PerLapDistanceRescale:
    """Per-lap arc-length is rescaled to track.total_distance_m so that
    segment-midpoint lookups land at the right physical location across
    all laps, even when the GPS arc-length drifts lap-to-lap."""

    def _build_two_lap_df(self, track: Track, lap2_scale: float) -> pd.DataFrame:
        """Build a 2-lap df where lap 1 spans the true track distance and
        lap 2 spans `lap2_scale * track.total_distance_m` (simulating GPS
        drift). Throttle is 80% on lap 1 segment 0 and 20% on all other
        segments; lap 2 has the *same physical layout* (throttle 80% at
        the front of the lap)."""
        rows = []
        total = track.total_distance_m
        n_per_lap = 200
        for lap_idx in range(2):
            lap_span = total if lap_idx == 0 else total * lap2_scale
            lap_offset = total if lap_idx == 1 else 0.0  # raw distance is cumulative
            for i in range(n_per_lap):
                frac = i / n_per_lap
                dist_in_lap = frac * lap_span
                phys_frac = frac  # physical position along the track
                # Throttle high on first segment only, low elsewhere.
                if phys_frac < 1.0 / track.num_segments:
                    throttle = 80.0
                else:
                    throttle = 20.0
                rows.append({
                    "Distance on GPS Speed": lap_offset + dist_in_lap,
                    "GPS Speed": 40.0,
                    "Throttle Pos": throttle,
                    "LVCU Torque Req": 0.6 * 85.0 if throttle > 50.0 else 0.1 * 85.0,
                    "FBrakePressure": 0.0,
                    "RBrakePressure": 0.0,
                    "GPS Latitude": 42.0 + frac * 0.001,
                    "GPS Longitude": -83.5,
                    "GPS LatAcc": 0.0,
                    "GPS Slope": 0.0,
                })
        return pd.DataFrame(rows)

    def test_rescale_routes_samples_to_correct_segment(self):
        """Without rescale, lap 2 with 0.8x arc-length leaves the last 20%
        of segments empty for that lap. After rescale, segment 0 still
        gets 80% throttle samples from both laps."""
        track = make_track(n_segments=5, length_m=100.0)  # 500m total
        df = self._build_two_lap_df(track, lap2_scale=0.8)

        result = extract_per_segment_actions(df, track)
        # Segment 0 is the high-throttle segment across both laps.
        seg0 = result.iloc[0]
        assert seg0["action"] == ControlAction.THROTTLE, (
            f"segment 0 should be THROTTLE, got {seg0['action']!r}"
        )
        # And the far segment (4) should be low-throttle, routed there
        # by the rescale.
        seg_last = result.iloc[-1]
        assert seg_last["action"] != ControlAction.THROTTLE or \
            seg_last["intensity"] < 0.3


# ---------------------------------------------------------------------------
# DriverZone dataclass
# ---------------------------------------------------------------------------

class TestDriverZone:
    def test_frozen(self):
        zone = DriverZone(
            zone_id=0, segment_start=0, segment_end=5,
            action=ControlAction.THROTTLE, intensity=0.8,
            distance_start_m=0.0, distance_end_m=250.0,
            label="Straight",
        )
        with pytest.raises(AttributeError):
            zone.intensity = 0.5  # type: ignore

    def test_fields(self):
        zone = DriverZone(
            zone_id=1, segment_start=3, segment_end=7,
            action=ControlAction.COAST, intensity=0.0,
            distance_start_m=150.0, distance_end_m=400.0,
            label="Turn 1 entry",
        )
        assert zone.zone_id == 1
        assert zone.action == ControlAction.COAST
        assert zone.label == "Turn 1 entry"


# ---------------------------------------------------------------------------
# extract_tire_grip_scale
# ---------------------------------------------------------------------------


class TestExtractTireGripScale:
    def test_known_lateral_g(self):
        """Synthetic data with known peak lateral G should produce expected scale."""
        n = 1000
        rng = np.random.default_rng(42)
        lat_acc_g = rng.uniform(0.0, 1.3, size=n)
        speed_kmh = rng.uniform(25.0, 60.0, size=n)

        df = pd.DataFrame({
            "GPS LatAcc": lat_acc_g,
            "GPS Speed": speed_kmh,
        })

        tire_model = MagicMock()
        tire_model.peak_lateral_force.return_value = 2.66 * 657.0

        result = extract_tire_grip_scale(
            aim_df=df,
            mass_kg=288.0,
            cla=2.18,
            tire_model=tire_model,
            fz_representative=657.0,
        )

        assert 0.3 < result["grip_scale"] < 0.6
        assert result["effective_mu_95"] > 0
        assert result["pacejka_mu"] == pytest.approx(2.66, rel=0.01)

    def test_filters_low_speed_samples(self):
        """Samples below 15 km/h should be excluded."""
        # 5 low-speed high-G samples (should be excluded) + 15 high-speed low-G samples
        df = pd.DataFrame({
            "GPS LatAcc": [2.0] * 5 + [0.5] * 15,
            "GPS Speed": [5.0] * 5 + [40.0] * 15,
        })

        tire_model = MagicMock()
        tire_model.peak_lateral_force.return_value = 2.66 * 657.0

        result = extract_tire_grip_scale(
            aim_df=df,
            mass_kg=288.0,
            cla=2.18,
            tire_model=tire_model,
            fz_representative=657.0,
        )

        assert result["effective_mu_95"] < 1.0
