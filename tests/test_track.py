"""Tests for track extraction from AiM GPS telemetry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fsae_sim.track import Segment, Track
from tests.conftest import requires_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_track(aim_csv_path) -> Track:
    """Call from_telemetry once and cache via pytest fixture indirection."""
    return Track.from_telemetry(aim_csv_path)


# ---------------------------------------------------------------------------
# Unit tests (no data required)
# ---------------------------------------------------------------------------

class TestSegmentDataclass:
    """Segment dataclass behaves correctly regardless of telemetry."""

    def test_segment_fields(self):
        seg = Segment(
            index=0,
            distance_start_m=0.0,
            length_m=5.0,
            curvature=0.02,
            grade=0.01,
        )
        assert seg.index == 0
        assert seg.distance_start_m == 0.0
        assert seg.length_m == 5.0
        assert seg.curvature == pytest.approx(0.02)
        assert seg.grade == pytest.approx(0.01)
        assert seg.grip_factor == pytest.approx(1.0)

    def test_segment_is_frozen(self):
        seg = Segment(
            index=0,
            distance_start_m=0.0,
            length_m=5.0,
            curvature=0.0,
            grade=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            seg.curvature = 0.5  # type: ignore[misc]

    def test_track_lap_distance_equals_total_distance(self):
        segs = [
            Segment(index=i, distance_start_m=float(i * 5), length_m=5.0, curvature=0.0, grade=0.0)
            for i in range(10)
        ]
        track = Track(name="test", segments=segs)
        assert track.lap_distance_m == pytest.approx(track.total_distance_m)
        assert track.lap_distance_m == pytest.approx(50.0)

    def test_track_num_segments(self):
        segs = [
            Segment(index=i, distance_start_m=float(i * 5), length_m=5.0, curvature=0.0, grade=0.0)
            for i in range(7)
        ]
        track = Track(name="test", segments=segs)
        assert track.num_segments == 7


# ---------------------------------------------------------------------------
# Integration tests (require real telemetry data)
# ---------------------------------------------------------------------------

@requires_data
class TestTrackExtraction:
    """Validate Track.from_telemetry against the Michigan 2025 endurance data."""

    @pytest.fixture()
    def track(self, aim_csv_path) -> Track:
        """Extract track from telemetry for each test."""
        return Track.from_telemetry(aim_csv_path)

    # ------------------------------------------------------------------
    # 1. Track extraction produces segments (non-empty)
    # ------------------------------------------------------------------

    def test_segments_non_empty(self, track: Track):
        """Track must contain at least one segment."""
        assert len(track.segments) > 0, "Track has no segments"

    def test_track_name(self, track: Track):
        """Default name should be 'Michigan Endurance'."""
        assert track.name == "Michigan Endurance"

    # ------------------------------------------------------------------
    # 2. Segment distances are contiguous
    # ------------------------------------------------------------------

    def test_segment_distances_contiguous(self, track: Track):
        """For every consecutive pair, start + length == next start."""
        segs = track.segments
        for i in range(len(segs) - 1):
            expected_next = segs[i].distance_start_m + segs[i].length_m
            actual_next = segs[i + 1].distance_start_m
            assert expected_next == pytest.approx(actual_next, abs=1e-9), (
                f"Gap between segment {i} and {i+1}: "
                f"expected {expected_next}, got {actual_next}"
            )

    def test_segment_indices_sequential(self, track: Track):
        """Segment index field must match position in list."""
        for pos, seg in enumerate(track.segments):
            assert seg.index == pos, f"Segment at position {pos} has index {seg.index}"

    # ------------------------------------------------------------------
    # 3. Total track distance is reasonable for Michigan (800–1200 m)
    # ------------------------------------------------------------------

    def test_lap_distance_reasonable(self, track: Track):
        """Michigan FSAE endurance lap is approximately 1 km."""
        dist = track.lap_distance_m
        assert 800.0 <= dist <= 1200.0, (
            f"Lap distance {dist:.1f} m is outside expected 800–1200 m range"
        )

    def test_total_distance_equals_lap_distance(self, track: Track):
        """lap_distance_m and total_distance_m must agree."""
        assert track.lap_distance_m == pytest.approx(track.total_distance_m)

    # ------------------------------------------------------------------
    # 4. Segments have finite curvature values (no NaN / inf)
    # ------------------------------------------------------------------

    def test_curvature_finite(self, track: Track):
        """All curvature values must be finite (no NaN or ±inf)."""
        curvatures = [s.curvature for s in track.segments]
        non_finite = [(i, v) for i, v in enumerate(curvatures) if not math.isfinite(v)]
        assert non_finite == [], (
            f"Non-finite curvature at segment indices: "
            f"{[i for i, _ in non_finite]}"
        )

    def test_grade_finite(self, track: Track):
        """All grade values must be finite."""
        grades = [s.grade for s in track.segments]
        non_finite = [(i, v) for i, v in enumerate(grades) if not math.isfinite(v)]
        assert non_finite == [], (
            f"Non-finite grade at segment indices: {[i for i, _ in non_finite]}"
        )

    # ------------------------------------------------------------------
    # 5. Curvature magnitude is reasonable for FSAE (|κ| < 0.1 → r > 10 m)
    # ------------------------------------------------------------------

    def test_curvature_magnitude_reasonable(self, track: Track):
        """Smoothed curvature magnitude must stay below 0.1 /m (radius > 10 m).

        FSAE courses must fit inside a 9 m wide lane so minimum radius is
        well above 10 m.  The raw GPS can produce transient spikes; the
        rolling-median smoother should suppress them.
        """
        violations = [
            (s.index, s.curvature)
            for s in track.segments
            if abs(s.curvature) > 0.1
        ]
        assert violations == [], (
            f"Curvature > 0.1 /m (radius < 10 m) in segments: "
            f"{[(i, f'{v:.4f}') for i, v in violations]}"
        )

    def test_curvature_non_zero_somewhere(self, track: Track):
        """Michigan has many corners; track should not be all zeros."""
        nonzero = sum(1 for s in track.segments if abs(s.curvature) > 0.005)
        assert nonzero > 10, (
            f"Only {nonzero} segments have |curvature| > 0.005 /m; "
            "track may be incorrectly flat"
        )

    # ------------------------------------------------------------------
    # 6. Grade values are small (Michigan is very flat)
    # ------------------------------------------------------------------

    def test_grade_small(self, track: Track):
        """Michigan is essentially flat; grade magnitude should stay < 0.05."""
        violations = [
            (s.index, s.grade)
            for s in track.segments
            if abs(s.grade) >= 0.05
        ]
        # Allow a handful of outlier segments from GPS noise
        assert len(violations) <= 5, (
            f"{len(violations)} segments have |grade| >= 0.05 "
            f"(expected Michigan to be flat): "
            f"{[(i, f'{v:.4f}') for i, v in violations[:5]]}"
        )

    def test_grade_mean_near_zero(self, track: Track):
        """Michigan is a closed loop; mean grade must be near zero."""
        grades = np.array([s.grade for s in track.segments])
        assert abs(grades.mean()) < 0.01, (
            f"Mean grade {grades.mean():.4f} is too large for a flat circuit"
        )

    # ------------------------------------------------------------------
    # 7. Number of segments is reasonable (100–300 for 5 m bins on ~1 km)
    # ------------------------------------------------------------------

    def test_segment_count_reasonable(self, track: Track):
        """For 5 m bins on a ~1 km lap, expect 100–300 segments."""
        n = track.num_segments
        assert 100 <= n <= 300, (
            f"Got {n} segments; expected 100–300 for 5 m bins on a ~1 km lap"
        )

    def test_all_segment_lengths_equal_bin_size(self, track: Track):
        """All segments should have the default 5 m length."""
        wrong = [s for s in track.segments if s.length_m != pytest.approx(5.0, abs=1e-9)]
        assert wrong == [], (
            f"{len(wrong)} segments do not have length_m == 5.0"
        )

    # ------------------------------------------------------------------
    # Bonus: custom bin size propagates correctly
    # ------------------------------------------------------------------

    def test_custom_bin_size(self, aim_csv_path):
        """from_telemetry respects a custom bin_size_m argument."""
        track_10m = Track.from_telemetry(aim_csv_path, bin_size_m=10.0)
        assert track_10m.num_segments > 0
        assert all(s.length_m == pytest.approx(10.0) for s in track_10m.segments)
        # Fewer segments than default 5 m
        track_5m = Track.from_telemetry(aim_csv_path, bin_size_m=5.0)
        assert track_10m.num_segments < track_5m.num_segments
