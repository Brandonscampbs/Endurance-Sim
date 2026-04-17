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
        # D-03: coast_throttle field deleted.
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

    def test_coast_always_zero_throttle(self):
        """D-03: coast_throttle removed; COAST always returns throttle=0."""
        strategy = self._make_strategy()
        cmd = strategy.decide(make_state(segment_idx=5), [])
        assert cmd.action == ControlAction.COAST
        assert cmd.throttle_pct == 0.0

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


# ---------------------------------------------------------------------------
# with_params()
# ---------------------------------------------------------------------------

class TestWithParams:
    def _make_strategy(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy
        n = 10
        throttle = np.array([0.5, 0.6, 0.7, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4])
        brake = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0])
        actions = np.array([1, 1, 1, 1, 1, 0, 0, 2, 2, 1])
        ref_speed = np.ones(n) * 12.0
        return PedalProfileStrategy(
            throttle_pct=throttle, brake_pct=brake,
            actions=actions, ref_speed_ms=ref_speed,
            num_segments=n,
        )

    def test_returns_new_instance(self):
        original = self._make_strategy()
        modified = original.with_params(throttle_scale=0.5)
        assert modified is not original
        assert modified.params.throttle_scale == 0.5
        assert original.params.throttle_scale == 1.0

    def test_preserves_profile_data(self):
        original = self._make_strategy()
        modified = original.with_params(brake_scale=2.0)
        np.testing.assert_array_equal(modified._throttle_pct, original._throttle_pct)
        np.testing.assert_array_equal(modified._brake_pct, original._brake_pct)
        np.testing.assert_array_equal(modified._actions, original._actions)

    def test_multiple_params(self):
        original = self._make_strategy()
        modified = original.with_params(throttle_scale=0.8, max_throttle=0.6, max_brake=0.9)
        assert modified.params.throttle_scale == 0.8
        assert modified.params.max_throttle == 0.6
        assert modified.params.max_brake == 0.9
        assert modified.params.brake_scale == 1.0

    def test_chained_with_params(self):
        original = self._make_strategy()
        step1 = original.with_params(throttle_scale=0.9)
        step2 = step1.with_params(brake_scale=1.5)
        assert step2.params.throttle_scale == 0.9
        assert step2.params.brake_scale == 1.5


# ---------------------------------------------------------------------------
# from_telemetry()
# ---------------------------------------------------------------------------

class TestFromTelemetry:
    """Test calibration from synthetic telemetry data."""

    def _make_telemetry(self, n_samples: int = 500, lap_distance: float = 50.0) -> pd.DataFrame:
        """Synthetic telemetry: 2 laps, throttle/coast/brake pattern."""
        dist_per_lap = np.linspace(0, lap_distance, n_samples // 2, endpoint=False)
        dist = np.concatenate([dist_per_lap, dist_per_lap + lap_distance])
        speed = np.full(len(dist), 40.0)  # 40 km/h constant

        # Pattern: first 70% throttle, next 15% coast, last 15% brake
        n_half = n_samples // 2
        throttle = np.zeros(len(dist))
        brake_f = np.full(len(dist), -18.5)  # bad front sensor (like real data)
        brake_r = np.zeros(len(dist))

        for offset in [0, n_half]:
            seg_size = n_half
            t_end = int(seg_size * 0.7)
            c_end = int(seg_size * 0.85)
            throttle[offset:offset + t_end] = 60.0  # 60% pedal
            brake_r[offset + c_end:offset + seg_size] = 15.0  # 15 bar

        # Need GPS columns for lap detection
        lat = np.linspace(42.0, 42.001, n_half)
        lat = np.concatenate([lat, lat])
        lon = np.full(len(dist), -83.5)

        return pd.DataFrame({
            "Distance on GPS Speed": dist,
            "GPS Speed": speed,
            "Throttle Pos": throttle,
            "FBrakePressure": brake_f,
            "RBrakePressure": brake_r,
            "GPS Latitude": lat,
            "GPS Longitude": lon,
            "GPS LatAcc": np.zeros(len(dist)),
            "GPS Slope": np.zeros(len(dist)),
        })

    def test_basic_calibration(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy
        aim_df = self._make_telemetry()
        track = make_track(n_segments=10, length_m=5.0)

        strategy = PedalProfileStrategy.from_telemetry(aim_df, track)
        assert strategy.num_segments == 10
        assert strategy.name == "pedal_profile"
        assert strategy.params.throttle_scale == 1.0

    def test_throttle_segments_have_pedal_values(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy
        aim_df = self._make_telemetry()
        track = make_track(n_segments=10, length_m=5.0)

        strategy = PedalProfileStrategy.from_telemetry(aim_df, track)
        throttle_mask = strategy._actions == 1
        assert throttle_mask.sum() > 0
        throttle_vals = strategy._throttle_pct[throttle_mask]
        assert np.all(throttle_vals > 0.0)
        assert np.all(throttle_vals <= 1.0)

    def test_brake_segments_have_brake_values(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy
        aim_df = self._make_telemetry()
        track = make_track(n_segments=10, length_m=5.0)

        strategy = PedalProfileStrategy.from_telemetry(aim_df, track)
        brake_mask = strategy._actions == 2
        if brake_mask.sum() > 0:
            brake_vals = strategy._brake_pct[brake_mask]
            assert np.all(brake_vals > 0.0)
            assert np.all(brake_vals <= 1.0)

    def test_ref_speed_populated(self):
        from fsae_sim.driver.strategies import PedalProfileStrategy
        aim_df = self._make_telemetry()
        track = make_track(n_segments=10, length_m=5.0)

        strategy = PedalProfileStrategy.from_telemetry(aim_df, track)
        assert np.all(strategy._ref_speed_ms > 0.0)

    def test_d04_calibration_respects_lap_subset(self):
        """D-04: `from_telemetry(..., laps=[0..4])` must produce different
        per-segment arrays than `laps=[5..9]` when the two subsets differ
        in driving behavior. Proves the held-out subset genuinely matters
        — otherwise the train/test split is cosmetic."""
        from fsae_sim.driver.strategies import PedalProfileStrategy

        n_laps = 10
        n_samples_per_lap = 200
        # Lap distance must be in [500, 2000] to pass detect_lap_boundaries
        # sanity check.
        lap_distance = 800.0
        track_seg_length = lap_distance / 10
        track = make_track(n_segments=10, length_m=track_seg_length)

        rows = []
        for lap in range(n_laps):
            dist_per_lap = np.linspace(
                0.0, lap_distance, n_samples_per_lap, endpoint=False,
            )
            for i, d in enumerate(dist_per_lap):
                # Laps 0-4 drive at 70% pedal; laps 5-9 drive at 20%.
                pedal = 70.0 if lap < 5 else 20.0
                # Build GPS latitude as a sawtooth crossing the median
                # once per lap, which is how detect_lap_boundaries finds
                # laps.
                frac = i / n_samples_per_lap
                # Triangular lat that starts below median, rises above,
                # then falls: crossing once per lap.
                if frac < 0.5:
                    lat = 42.0 - 0.001 + frac * 2 * 0.002
                else:
                    lat = 42.0 + 0.001 - (frac - 0.5) * 2 * 0.002
                rows.append({
                    "Distance on GPS Speed": lap * lap_distance + d,
                    "GPS Speed": 40.0,
                    "Throttle Pos": pedal,
                    "LVCU Torque Req": pedal * 85.0 / 100.0,
                    "FBrakePressure": 0.0,
                    "RBrakePressure": 0.0,
                    "GPS Latitude": lat,
                    "GPS Longitude": -83.5,
                    "GPS LatAcc": 0.0,
                    "GPS Slope": 0.0,
                })
        aim_df = pd.DataFrame(rows)

        # Confirm lap detection works so `laps=` is actually respected.
        from fsae_sim.analysis.telemetry_analysis import _detect_lap_boundaries_safe
        boundaries = _detect_lap_boundaries_safe(aim_df)
        assert len(boundaries) >= 5, (
            f"Test setup: lap detection produced {len(boundaries)} laps, "
            "need at least 5 for the subset test to be meaningful."
        )

        strat_first = PedalProfileStrategy.from_telemetry(
            aim_df, track, laps=[0, 1, 2],
        )
        strat_second = PedalProfileStrategy.from_telemetry(
            aim_df, track, laps=[3, 4],
        )
        # Strategy built on the 70%-pedal laps should have a very different
        # throttle profile than the one built on 20%-pedal laps (which
        # may even fall below the 3% torque-fraction threshold for THROTTLE).
        assert not np.allclose(
            strat_first._throttle_pct, strat_second._throttle_pct, atol=1e-3,
        ), (
            "D-04: per-segment throttle_pct should differ between calibration "
            "subsets that drive at different torque levels; "
            f"got first={strat_first._throttle_pct}, second={strat_second._throttle_pct}"
        )

    def test_d01_classify_on_torque_fraction(self):
        """D-01: a segment with low raw pedal (3%) but high torque request
        (15% of inverter cap) should classify as THROTTLE with the torque
        fraction as intensity — not COAST based on raw pedal alone."""
        from fsae_sim.driver.strategies import PedalProfileStrategy

        n_samples = 400
        n_half = n_samples // 2
        lap_distance = 50.0
        dist_per_lap = np.linspace(0, lap_distance, n_half, endpoint=False)
        dist = np.concatenate([dist_per_lap, dist_per_lap + lap_distance])
        speed = np.full(len(dist), 40.0)

        # Raw pedal pinned at 3% throughout (below the old 5% threshold).
        throttle = np.full(len(dist), 3.0)
        # LVCU Torque Req at 15% of inverter cap (85 Nm) = 12.75 Nm.
        torque_req = np.full(len(dist), 0.15 * 85.0)
        brake_f = np.zeros(len(dist))
        brake_r = np.zeros(len(dist))

        lat = np.linspace(42.0, 42.001, n_half)
        lat = np.concatenate([lat, lat])
        lon = np.full(len(dist), -83.5)

        aim_df = pd.DataFrame({
            "Distance on GPS Speed": dist,
            "GPS Speed": speed,
            "Throttle Pos": throttle,
            "LVCU Torque Req": torque_req,
            "FBrakePressure": brake_f,
            "RBrakePressure": brake_r,
            "GPS Latitude": lat,
            "GPS Longitude": lon,
            "GPS LatAcc": np.zeros(len(dist)),
            "GPS Slope": np.zeros(len(dist)),
        })
        track = make_track(n_segments=10, length_m=5.0)
        strategy = PedalProfileStrategy.from_telemetry(aim_df, track)

        # All segments should classify as THROTTLE (action==1).
        assert np.any(strategy._actions == 1), (
            "D-01: low pedal + high torque must classify as THROTTLE, "
            f"got actions={strategy._actions!r}"
        )
        throttle_mask = strategy._actions == 1
        # Intensity should be the torque fraction, ~0.15.
        np.testing.assert_allclose(
            strategy._throttle_pct[throttle_mask], 0.15, atol=0.01,
        )
