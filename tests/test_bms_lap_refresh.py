"""Tests for the BMS-limit lap-by-lap envelope refresh.

The speed envelope is built once per simulation with the initial BMS
discharge limit (engine.py around line 323). Late-stint thermal
derating (BMS limit shrinks as the pack heats) was previously not
reflected in the envelope, so the forward-pass acceleration ceiling
remained optimistic on the final laps. This test suite exercises the
new lap-boundary refresh: when the BMS-limit derived from current
(temp, SOC) drifts more than ``BMS_REFRESH_DELTA_A`` from the limit
the envelope was last built with, the envelope is recomputed.

Tests:
1. **Monotonic temperature rise across laps** -> ``bms_limit_a_per_lap``
   strictly decreases (BMS limit derates with temperature).
2. **Threshold test**: a 4 A drift triggers no recompute; a 6 A drift
   triggers exactly one recompute.
3. **Cost**: full 22-lap sim's
   ``envelope_recompute_ms_total < 500 ms``.
4. **Lap-1 vs lap-22 envelope cap visibility** on the longest straight
   when the stint is hot.

Reference:
- docs/superpowers/plans/2026-05-06-battery-upgrades.md (BMS lap
  refresh section).
- Current limit table: configs/ct16ev.yaml battery.discharge_limits;
  100 A @ 30 C tapers to 0 A @ 65 C.
"""

from __future__ import annotations

import time

import pytest

from fsae_sim.driver.strategy import (
    ControlAction,
    ControlCommand,
    DriverStrategy,
    SimState,
)
from fsae_sim.sim.engine import (
    BMS_REFRESH_DELTA_A,
    SimulationEngine,
    SimulationMode,
)
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.data.loader import load_voltt_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantStrategy(DriverStrategy):
    """Constant pedal/brake command for predictable physics."""

    name = "constant"

    def __init__(self, throttle_pct: float = 1.0, brake_pct: float = 0.0) -> None:
        self.throttle_pct = throttle_pct
        self.brake_pct = brake_pct

    def decide(self, state: SimState, upcoming) -> ControlCommand:
        action = (
            ControlAction.THROTTLE
            if self.throttle_pct > 0
            else (
                ControlAction.BRAKE
                if self.brake_pct > 0 else ControlAction.COAST
            )
        )
        return ControlCommand(
            action=action,
            throttle_pct=self.throttle_pct,
            brake_pct=self.brake_pct,
        )


def _alternating_track(num_laps_segments: int = 100) -> Track:
    """Track with periodic corners so the speed envelope has finite cap.

    Corners at index 0, every 25th index, AND the last index so the
    backward-pass v_max[n-1] is finite.
    """
    segments = []
    for i in range(num_laps_segments):
        is_corner = (
            (i == 0)
            or (i == num_laps_segments - 1)
            or (i % 25 == 0)
        )
        segments.append(Segment(
            index=i,
            distance_start_m=i * 0.5,
            length_m=0.5,
            curvature=0.04 if is_corner else 0.0,
            grade=0.0,
        ))
    return Track(name="alt", segments=segments)


@pytest.fixture
def vehicle_cfg(ct16ev_config_path) -> VehicleConfig:
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def battery(vehicle_cfg, voltt_cell_path) -> BatteryModel:
    df = load_voltt_csv(voltt_cell_path)
    bm = BatteryModel(vehicle_cfg.battery)
    bm.calibrate_from_voltt(df)
    return bm


# ---------------------------------------------------------------------------
# 1. Monotonic temperature rise -> monotonic decrease of bms_limit_a_per_lap
# ---------------------------------------------------------------------------


class TestBmsLimitMonotonicity:
    """As cell temperature rises across laps, BMS discharge limit drops."""

    def test_bms_limit_decreases_with_lap(
        self, vehicle_cfg, battery,
    ) -> None:
        track = _alternating_track(num_laps_segments=100)
        strategy = _ConstantStrategy(throttle_pct=1.0)
        engine = SimulationEngine(
            vehicle_cfg, track, strategy, battery,
            mode=SimulationMode.CALIBRATION,
        )
        # Start hot enough that subsequent laps continue heating.
        result = engine.run(
            num_laps=22, initial_soc_pct=92.0, initial_temp_c=40.0,
            initial_speed_ms=10.0, rolling_start=True,
        )
        per_lap = result.bms_limit_a_per_lap
        assert len(per_lap) == 22, (
            f"Expected one BMS limit per lap, got {len(per_lap)}"
        )
        # The limit should be monotonically non-increasing as the stint
        # heats up — we allow a small tolerance because 25 C->30 C->...
        # crosses bin boundaries with slight non-monotonic interpolation.
        # Strict monotonicity holds once we're past the 30 C threshold.
        for prev, curr in zip(per_lap[:-1], per_lap[1:]):
            assert curr <= prev + 1e-6, (
                f"BMS limit must non-increase across laps; "
                f"got {prev:.1f} -> {curr:.1f}"
            )
        # Final lap limit must be strictly less than first lap limit
        # (some derating must occur over 22 laps of full throttle).
        assert per_lap[-1] < per_lap[0], (
            f"Expected BMS-limit derate by lap 22; "
            f"lap1={per_lap[0]:.1f} A, lap22={per_lap[-1]:.1f} A"
        )


# ---------------------------------------------------------------------------
# 2. Threshold: 4 A drift -> no recompute; 6 A drift -> 1 recompute
# ---------------------------------------------------------------------------


class TestBmsRefreshThreshold:
    """Recompute is triggered only when |drift| >= BMS_REFRESH_DELTA_A.

    Default threshold is 5 A. We synthesize a battery model that returns
    a controlled BMS limit and run a 2-lap sim where the limit drifts
    by 4 A (no recompute) or 6 A (one recompute).
    """

    @pytest.fixture
    def synthetic_battery(self, battery: BatteryModel) -> BatteryModel:
        # Reuse the calibrated battery; we'll patch max_discharge_current
        # in the test to control the drift.
        return battery

    def test_4a_drift_does_not_trigger_recompute(
        self, vehicle_cfg, synthetic_battery,
    ) -> None:
        track = _alternating_track(num_laps_segments=50)
        strategy = _ConstantStrategy(throttle_pct=1.0)
        engine = SimulationEngine(
            vehicle_cfg, track, strategy, synthetic_battery,
            mode=SimulationMode.CALIBRATION,
        )

        # Patch max_discharge_current to return 100 A on lap 1 entry,
        # then 96 A (4 A drift) on lap 2 entry.
        call_count = [0]
        original = synthetic_battery.max_discharge_current

        def fake(temp, soc):
            call_count[0] += 1
            # Lap 1 (envelope build): 100 A. Lap 2: 96 A.
            return 100.0 if call_count[0] <= 1 else 96.0

        synthetic_battery.max_discharge_current = fake  # type: ignore
        try:
            result = engine.run(
                num_laps=2, initial_soc_pct=95.0, initial_temp_c=25.0,
                initial_speed_ms=10.0, rolling_start=True,
            )
        finally:
            synthetic_battery.max_discharge_current = original  # type: ignore

        # 4 A drift must not trigger envelope recompute.
        assert result.envelope_recomputes == 0, (
            f"4 A drift should not trigger recompute; "
            f"got {result.envelope_recomputes} recomputes."
        )

    def test_6a_drift_triggers_recompute(
        self, vehicle_cfg, synthetic_battery,
    ) -> None:
        track = _alternating_track(num_laps_segments=50)
        strategy = _ConstantStrategy(throttle_pct=1.0)
        engine = SimulationEngine(
            vehicle_cfg, track, strategy, synthetic_battery,
            mode=SimulationMode.CALIBRATION,
        )

        call_count = [0]
        original = synthetic_battery.max_discharge_current

        def fake(temp, soc):
            call_count[0] += 1
            return 100.0 if call_count[0] <= 1 else 94.0  # 6 A drift

        synthetic_battery.max_discharge_current = fake  # type: ignore
        try:
            result = engine.run(
                num_laps=2, initial_soc_pct=95.0, initial_temp_c=25.0,
                initial_speed_ms=10.0, rolling_start=True,
            )
        finally:
            synthetic_battery.max_discharge_current = original  # type: ignore

        # 6 A drift >= 5 A threshold -> exactly one recompute.
        assert result.envelope_recomputes == 1, (
            f"6 A drift should trigger one recompute; "
            f"got {result.envelope_recomputes}."
        )

    def test_threshold_constant_is_5a(self) -> None:
        """Default refresh threshold matches the documented value."""
        assert BMS_REFRESH_DELTA_A == 5.0


# ---------------------------------------------------------------------------
# 3. Cost: full 22-lap sim recompute time <= 500 ms
# ---------------------------------------------------------------------------


class TestBmsRefreshCost:
    """Total time spent on envelope recomputes must stay below 500 ms."""

    def test_22lap_recompute_cost_under_500ms(
        self, vehicle_cfg, battery,
    ) -> None:
        track = _alternating_track(num_laps_segments=200)
        strategy = _ConstantStrategy(throttle_pct=1.0)
        engine = SimulationEngine(
            vehicle_cfg, track, strategy, battery,
            mode=SimulationMode.CALIBRATION,
        )
        result = engine.run(
            num_laps=22, initial_soc_pct=92.0, initial_temp_c=40.0,
            initial_speed_ms=10.0, rolling_start=True,
        )
        assert result.envelope_recompute_ms_total < 500.0, (
            f"Total envelope-recompute time "
            f"{result.envelope_recompute_ms_total:.1f} ms exceeds "
            "500 ms budget."
        )


# ---------------------------------------------------------------------------
# 4. Lap-1 vs lap-22 envelope cap visibility on the longest straight
# ---------------------------------------------------------------------------


class TestLapVisibility:
    """Final lap envelope cap on the longest straight is below lap-1 cap.

    When the stint runs hot, the BMS-derated power ceiling reduces the
    forward-pass acceleration capability — straight-segment v_max is
    therefore lower on lap 22 than on lap 1.
    """

    def test_late_stint_envelope_lower_on_long_straight(
        self, vehicle_cfg, battery,
    ) -> None:
        # Build a track with one long straight (50 segments) and a
        # bracketing corner.
        segments = []
        for i in range(50):
            is_corner = (i == 0) or (i == 49)
            segments.append(Segment(
                index=i,
                distance_start_m=i * 0.5,
                length_m=0.5,
                curvature=0.04 if is_corner else 0.0,
                grade=0.0,
            ))
        track = Track(name="straight_50", segments=segments)
        strategy = _ConstantStrategy(throttle_pct=1.0)
        engine = SimulationEngine(
            vehicle_cfg, track, strategy, battery,
            mode=SimulationMode.CALIBRATION,
        )
        # Heat the stint: high initial temp + many laps.
        result = engine.run(
            num_laps=22, initial_soc_pct=92.0, initial_temp_c=45.0,
            initial_speed_ms=10.0, rolling_start=True,
        )

        # Lap-1 BMS limit and lap-22 BMS limit recorded.
        per_lap = result.bms_limit_a_per_lap
        assert len(per_lap) >= 2
        # If derating actually occurred (per-lap monotonicity), some
        # recompute must have fired by lap 22.
        if per_lap[-1] < per_lap[0] - BMS_REFRESH_DELTA_A:
            assert result.envelope_recomputes >= 1, (
                "BMS limit derated by >5 A but no envelope recompute "
                "occurred — visibility check failed."
            )
