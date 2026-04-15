# PedalProfileStrategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 50-zone CalibratedStrategy (10.5% time error) with a per-segment pedal-profile driver model that feeds raw throttle/brake through the LVCU firmware chain, enabling independent driver and car-tune sweeps.

**Architecture:** `PedalProfileStrategy` stores per-segment throttle position and brake pressure arrays extracted from telemetry. At runtime, `decide()` outputs raw pedal values that the engine routes through `lvcu_torque_command()` — the existing LVCU firmware model. `DriverParams` dataclass holds sweep multipliers (throttle_scale, brake_scale, etc.). No engine changes needed.

**Tech Stack:** Python, NumPy, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-04-15-pedal-profile-driver-model-design.md`

---

### Task 1: DriverParams dataclass + PedalProfileStrategy constructor and decide()

**Files:**
- Create: `tests/test_pedal_profile_strategy.py`
- Modify: `src/fsae_sim/driver/strategies.py`

- [ ] **Step 1: Write failing tests for DriverParams and PedalProfileStrategy.decide()**

Create `tests/test_pedal_profile_strategy.py`:

```python
"""Tests for PedalProfileStrategy pedal-profile driver model."""

from __future__ import annotations

import numpy as np
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
        actions = np.array([1, 1, 1, 1, 1, 0, 0, 2, 2, 1])  # 0=coast, 1=throttle, 2=brake
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pedal_profile_strategy.py -v`
Expected: FAIL — `ImportError: cannot import name 'DriverParams'` and `'PedalProfileStrategy'`

- [ ] **Step 3: Implement DriverParams and PedalProfileStrategy in strategies.py**

Add to `src/fsae_sim/driver/strategies.py` after the existing imports, before `ReplayStrategy`:

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DriverParams:
    """Tunable driver behavior parameters for sweeps.

    All multipliers default to 1.0 (baseline = telemetry behavior).
    """
    throttle_scale: float = 1.0
    brake_scale: float = 1.0
    coast_throttle: float = 0.0
    max_throttle: float = 1.0
    max_brake: float = 1.0
```

Then add the class at the end of the file (after `CalibratedStrategy`):

```python
class PedalProfileStrategy(DriverStrategy):
    """Per-segment pedal-profile driver model.

    Stores raw throttle position and brake pressure per track segment,
    extracted from telemetry.  At runtime, outputs pedal values that
    the engine routes through ``lvcu_torque_command()`` — the same
    firmware chain the real car uses.

    For sweeps, ``DriverParams`` multipliers scale pedal inputs (driver
    behavior) while ``PowertrainConfig`` changes affect LVCU processing
    (car tune).  Both sweep independently.
    """

    name = "pedal_profile"

    def __init__(
        self,
        throttle_pct: np.ndarray,
        brake_pct: np.ndarray,
        actions: np.ndarray,
        ref_speed_ms: np.ndarray,
        num_segments: int,
        *,
        params: DriverParams | None = None,
    ) -> None:
        if not (len(throttle_pct) == len(brake_pct) == len(actions) == len(ref_speed_ms) == num_segments):
            raise ValueError(
                f"All arrays must have the same length as num_segments ({num_segments}), "
                f"got throttle={len(throttle_pct)}, brake={len(brake_pct)}, "
                f"actions={len(actions)}, ref_speed={len(ref_speed_ms)}"
            )
        self._throttle_pct = np.asarray(throttle_pct, dtype=np.float64)
        self._brake_pct = np.asarray(brake_pct, dtype=np.float64)
        self._actions = np.asarray(actions, dtype=np.int32)
        self._ref_speed_ms = np.asarray(ref_speed_ms, dtype=np.float64)
        self._num_segments = num_segments
        self.params = params or DriverParams()

    @property
    def num_segments(self) -> int:
        return self._num_segments

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        seg_idx = state.segment_idx % self._num_segments
        action_code = int(self._actions[seg_idx])

        if action_code == 1:  # THROTTLE
            throttle = float(self._throttle_pct[seg_idx]) * self.params.throttle_scale
            throttle = min(throttle, self.params.max_throttle)
            throttle = max(0.0, min(1.0, throttle))
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=throttle, brake_pct=0.0)

        elif action_code == 2:  # BRAKE
            brake = float(self._brake_pct[seg_idx]) * self.params.brake_scale
            brake = min(brake, self.params.max_brake)
            brake = max(0.0, min(1.0, brake))
            return ControlCommand(ControlAction.BRAKE, throttle_pct=0.0, brake_pct=brake)

        else:  # COAST (0)
            return ControlCommand(
                ControlAction.COAST,
                throttle_pct=max(0.0, min(1.0, self.params.coast_throttle)),
                brake_pct=0.0,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pedal_profile_strategy.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pedal_profile_strategy.py src/fsae_sim/driver/strategies.py
git commit -m "feat(driver): add PedalProfileStrategy with DriverParams for sweep support"
```

---

### Task 2: with_params() method

**Files:**
- Modify: `tests/test_pedal_profile_strategy.py`
- Modify: `src/fsae_sim/driver/strategies.py`

- [ ] **Step 1: Write failing tests for with_params()**

Append to `tests/test_pedal_profile_strategy.py`:

```python
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
        # Same underlying profile data
        np.testing.assert_array_equal(modified._throttle_pct, original._throttle_pct)
        np.testing.assert_array_equal(modified._brake_pct, original._brake_pct)
        np.testing.assert_array_equal(modified._actions, original._actions)

    def test_multiple_params(self):
        original = self._make_strategy()
        modified = original.with_params(throttle_scale=0.8, max_throttle=0.6, coast_throttle=0.02)
        assert modified.params.throttle_scale == 0.8
        assert modified.params.max_throttle == 0.6
        assert modified.params.coast_throttle == 0.02
        # Unmodified params keep defaults
        assert modified.params.brake_scale == 1.0

    def test_chained_with_params(self):
        original = self._make_strategy()
        step1 = original.with_params(throttle_scale=0.9)
        step2 = step1.with_params(brake_scale=1.5)
        assert step2.params.throttle_scale == 0.9
        assert step2.params.brake_scale == 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pedal_profile_strategy.py::TestWithParams -v`
Expected: FAIL — `AttributeError: 'PedalProfileStrategy' object has no attribute 'with_params'`

- [ ] **Step 3: Add with_params() to PedalProfileStrategy**

Add this method to `PedalProfileStrategy` in `src/fsae_sim/driver/strategies.py`:

```python
    def with_params(self, **kwargs) -> PedalProfileStrategy:
        """Return a new strategy with modified DriverParams.

        Shares the underlying profile arrays (immutable numpy views).
        Only the DriverParams are replaced.

        Args:
            **kwargs: Fields of DriverParams to override.

        Returns:
            New PedalProfileStrategy with updated params.
        """
        new_params = replace(self.params, **kwargs)
        return PedalProfileStrategy(
            throttle_pct=self._throttle_pct,
            brake_pct=self._brake_pct,
            actions=self._actions,
            ref_speed_ms=self._ref_speed_ms,
            num_segments=self._num_segments,
            params=new_params,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pedal_profile_strategy.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pedal_profile_strategy.py src/fsae_sim/driver/strategies.py
git commit -m "feat(driver): add with_params() for PedalProfileStrategy sweep support"
```

---

### Task 3: from_telemetry() classmethod

**Files:**
- Modify: `tests/test_pedal_profile_strategy.py`
- Modify: `src/fsae_sim/driver/strategies.py`

- [ ] **Step 1: Write failing tests for from_telemetry()**

Append to `tests/test_pedal_profile_strategy.py`:

```python
class TestFromTelemetry:
    """Test calibration from synthetic telemetry data."""

    def _make_telemetry(self, n_samples: int = 500, lap_distance: float = 50.0) -> pd.DataFrame:
        """Synthetic telemetry: 2 laps, throttle/coast/brake pattern."""
        import pandas as pd
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
        # First 70% of segments should be throttle with nonzero pedal
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
        # All ref speeds should be positive (we set 40 km/h)
        assert np.all(strategy._ref_speed_ms > 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pedal_profile_strategy.py::TestFromTelemetry -v`
Expected: FAIL — `AttributeError: type object 'PedalProfileStrategy' has no attribute 'from_telemetry'`

- [ ] **Step 3: Implement from_telemetry()**

Add this classmethod to `PedalProfileStrategy` in `src/fsae_sim/driver/strategies.py`:

```python
    @classmethod
    def from_telemetry(
        cls,
        aim_df: pd.DataFrame,
        track: Track,
        *,
        laps: list[int] | None = None,
        throttle_threshold: float = 5.0,
        brake_threshold: float = 2.0,
        name: str = "pedal_profile",
    ) -> PedalProfileStrategy:
        """Calibrate from AiM telemetry.

        Samples raw throttle position and brake pressure at each track
        segment midpoint, classifies actions per-lap, then aggregates
        across representative laps using majority vote for action and
        median for pedal/brake values.

        Args:
            aim_df: AiM telemetry DataFrame.
            track: Track geometry with segments.
            laps: Which lap indices to use (None = auto-select).
            throttle_threshold: Throttle % above which = THROTTLE.
            brake_threshold: Brake pressure (bar) above which = BRAKE.
            name: Strategy name.

        Returns:
            Calibrated PedalProfileStrategy.
        """
        from fsae_sim.analysis.telemetry_analysis import _detect_lap_boundaries_safe

        num_segments = track.num_segments
        lap_boundaries = _detect_lap_boundaries_safe(aim_df)

        # Select representative laps
        if lap_boundaries and len(lap_boundaries) >= 2:
            if laps is not None:
                selected = [lap_boundaries[i] for i in laps if i < len(lap_boundaries)]
            else:
                selected = []
                median_dist = float(np.median([d for _, _, d in lap_boundaries]))
                for i, (s, e, d) in enumerate(lap_boundaries):
                    if i == 0:
                        continue
                    if abs(d - median_dist) > median_dist * 0.15:
                        continue
                    selected.append((s, e, d))
            if not selected:
                selected = lap_boundaries
        else:
            # Single-pass fallback: treat whole dataset as one lap
            total_dist = aim_df["Distance on GPS Speed"].values
            selected = [(0, len(aim_df), float(total_dist[-1] - total_dist[0]))]

        # Brake normalization across all moving data
        speed_all = aim_df["GPS Speed"].values
        moving = speed_all > 5.0
        brake_all = np.maximum(
            aim_df["FBrakePressure"].values,
            aim_df["RBrakePressure"].values,
        )
        nonzero_brake = brake_all[moving & (brake_all > 0)]
        brake_norm = float(np.percentile(nonzero_brake, 99)) if len(nonzero_brake) > 0 else 1.0
        brake_norm = max(brake_norm, 1.0)

        # Per-lap, per-segment extraction
        action_matrix = []  # (n_laps, n_segments) int
        throttle_matrix = []  # (n_laps, n_segments) float, raw pedal 0-1
        brake_matrix = []  # (n_laps, n_segments) float, normalized 0-1
        speed_matrix = []  # (n_laps, n_segments) float, m/s

        for start_idx, end_idx, _ in selected:
            lap_df = aim_df.iloc[start_idx:end_idx]
            lap_dist_raw = lap_df["Distance on GPS Speed"].values
            lap_d = lap_dist_raw - lap_dist_raw[0]
            lap_throttle = lap_df["Throttle Pos"].values
            lap_speed = lap_df["GPS Speed"].values
            lap_brake = np.maximum(
                lap_df["FBrakePressure"].values,
                lap_df["RBrakePressure"].values,
            )

            lap_actions = np.zeros(num_segments, dtype=int)
            lap_throttles = np.zeros(num_segments)
            lap_brakes = np.zeros(num_segments)
            lap_speeds = np.zeros(num_segments)

            for seg in track.segments:
                mid = seg.distance_start_m + seg.length_m / 2.0
                half_bin = seg.length_m / 2.0

                mask = (lap_d >= mid - half_bin) & (lap_d < mid + half_bin)
                if not np.any(mask):
                    nearest_idx = np.argmin(np.abs(lap_d - mid))
                    mask = np.zeros(len(lap_d), dtype=bool)
                    mask[nearest_idx] = True

                seg_throttle = float(np.median(lap_throttle[mask]))
                seg_brake = float(np.median(lap_brake[mask]))
                seg_speed = float(np.mean(lap_speed[mask]))

                if seg_brake > brake_threshold:
                    lap_actions[seg.index] = 2  # BRAKE
                    lap_brakes[seg.index] = float(np.clip(seg_brake / brake_norm, 0.0, 1.0))
                    lap_throttles[seg.index] = 0.0
                elif seg_throttle > throttle_threshold:
                    lap_actions[seg.index] = 1  # THROTTLE
                    lap_throttles[seg.index] = float(np.clip(seg_throttle / 100.0, 0.0, 1.0))
                    lap_brakes[seg.index] = 0.0
                else:
                    lap_actions[seg.index] = 0  # COAST
                    lap_throttles[seg.index] = 0.0
                    lap_brakes[seg.index] = 0.0

                lap_speeds[seg.index] = seg_speed / 3.6  # km/h -> m/s

            action_matrix.append(lap_actions)
            throttle_matrix.append(lap_throttles)
            brake_matrix.append(lap_brakes)
            speed_matrix.append(lap_speeds)

        # Aggregate across laps
        action_arr = np.array(action_matrix)  # (n_laps, n_segments)
        throttle_arr = np.array(throttle_matrix)
        brake_arr = np.array(brake_matrix)
        speed_arr = np.array(speed_matrix)

        final_actions = np.zeros(num_segments, dtype=int)
        final_throttle = np.zeros(num_segments)
        final_brake = np.zeros(num_segments)
        final_speed = np.zeros(num_segments)

        for i in range(num_segments):
            # Majority vote for action
            counts = np.bincount(action_arr[:, i], minlength=3)
            winner = int(np.argmax(counts))
            final_actions[i] = winner

            # Median of pedal/brake values across laps where the
            # winning action was chosen
            winner_mask = action_arr[:, i] == winner
            if np.any(winner_mask):
                final_throttle[i] = float(np.median(throttle_arr[:, i][winner_mask]))
                final_brake[i] = float(np.median(brake_arr[:, i][winner_mask]))
            else:
                final_throttle[i] = float(np.median(throttle_arr[:, i]))
                final_brake[i] = float(np.median(brake_arr[:, i]))

            final_speed[i] = float(np.mean(speed_arr[:, i]))

        return cls(
            throttle_pct=final_throttle,
            brake_pct=final_brake,
            actions=final_actions,
            ref_speed_ms=final_speed,
            num_segments=num_segments,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pedal_profile_strategy.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pedal_profile_strategy.py src/fsae_sim/driver/strategies.py
git commit -m "feat(driver): add from_telemetry() calibration for PedalProfileStrategy"
```

---

### Task 4: Update validation script and run full validation

**Files:**
- Modify: `scripts/validate_driver_model.py`

- [ ] **Step 1: Run existing tests to confirm nothing is broken**

Run: `pytest tests/ -v --tb=short`
Expected: All existing tests PASS

- [ ] **Step 2: Update validate_driver_model.py to use PedalProfileStrategy**

Replace the import and strategy construction in `scripts/validate_driver_model.py`. Change:

```python
from fsae_sim.driver.strategies import CalibratedStrategy
```
to:
```python
from fsae_sim.driver.strategies import PedalProfileStrategy
```

Change the strategy calibration section (around line 49):

```python
    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    zones = strategy.zones

    print(f"  Created {len(zones)} zones")
    print(f"\n  Zone summary:")
    for z in zones:
        action_str = z.action.value.upper()
        if z.action.value == "throttle":
            detail = f"{z.intensity * 100:.0f}%"
        elif z.action.value == "brake":
            detail = f"{z.intensity * 100:.0f}%"
        else:
            detail = ""
        print(f"    Zone {z.zone_id:2d}: {z.label:20s} "
              f"({z.distance_start_m:6.0f}-{z.distance_end_m:6.0f}m) "
              f"{action_str:8s} {detail}")

    # ── Zone quality checks ──
    print("\n[4/6] Zone quality checks...")
    all_ok = True
    for z in zones:
        span = z.distance_end_m - z.distance_start_m
        if span > 200:
            print(f"  WARNING: Zone {z.zone_id} ({z.label}) spans {span:.0f}m > 200m")
            all_ok = False
        if span < 5:
            print(f"  WARNING: Zone {z.zone_id} ({z.label}) spans {span:.0f}m < 5m")
            all_ok = False
    if not (25 <= len(zones) <= 50):
        print(f"  WARNING: {len(zones)} zones (expected 25-50)")
    if all_ok and 25 <= len(zones) <= 50:
        print("  All zone quality checks passed")
```

to:

```python
    strategy = PedalProfileStrategy.from_telemetry(aim_df, track)

    # Profile summary
    n_throttle = int(np.sum(strategy._actions == 1))
    n_coast = int(np.sum(strategy._actions == 0))
    n_brake = int(np.sum(strategy._actions == 2))
    print(f"  {strategy.num_segments} segments: "
          f"{n_throttle} throttle, {n_coast} coast, {n_brake} brake")
    throttle_segs = strategy._throttle_pct[strategy._actions == 1]
    if len(throttle_segs) > 0:
        print(f"  Throttle pedal: mean={np.mean(throttle_segs)*100:.1f}%, "
              f"median={np.median(throttle_segs)*100:.1f}%, "
              f"max={np.max(throttle_segs)*100:.1f}%")
    brake_segs = strategy._brake_pct[strategy._actions == 2]
    if len(brake_segs) > 0:
        print(f"  Brake pressure: mean={np.mean(brake_segs)*100:.1f}%, "
              f"median={np.median(brake_segs)*100:.1f}%, "
              f"max={np.max(brake_segs)*100:.1f}%")

    # ── Profile quality checks ──
    print("\n[4/6] Profile quality checks...")
    all_ok = True
    if n_throttle == 0:
        print("  WARNING: No throttle segments found")
        all_ok = False
    if n_coast == 0:
        print("  WARNING: No coast segments found")
        all_ok = False
    total_pct = 100.0 * (n_throttle + n_coast + n_brake) / strategy.num_segments
    if total_pct < 99.9:
        print(f"  WARNING: Only {total_pct:.1f}% of segments classified")
        all_ok = False
    if all_ok:
        print("  All profile quality checks passed")
```

- [ ] **Step 3: Run the updated validation script**

Run: `python scripts/validate_driver_model.py`
Expected: 22 laps complete. Check validation metrics — target is improvement over CalibratedStrategy's 5/8 pass rate. Ideal: 8/8 pass matching replay performance.

- [ ] **Step 4: Run all tests to confirm nothing is broken**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_driver_model.py
git commit -m "feat(driver): switch validation to PedalProfileStrategy"
```
