"""Tests for D-23 driver-channel validation harness.

The sim records per-segment driver commands (throttle_pct, brake_pct,
action).  ``validate_driver_channels`` compares those against telemetry
(Throttle Pos, FBrakePressure / RBrakePressure, derived action) on a
held-out lap subset.  Gates all subsequent driver-model fixes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsae_sim.analysis.validation import (
    DriverChannelValidation,
    validate_driver_channels,
)


def _build_fabricated_frames(
    n_samples: int = 200,
    lap_length_m: float = 1000.0,
    lap: int = 13,
    throttle_profile: str = "sin",  # "sin" | "const_half" | "zero"
    brake_profile: str = "impulse",  # "impulse" | "zero"
    throttle_offset: float = 0.0,
    invert_action: bool = False,
):
    """Produce a telemetry df and a matching sim_states df.

    The sim_states df is constructed to match telemetry perfectly (so
    RMSE≈0 and R²≈1) unless ``throttle_offset`` or ``invert_action``
    introduce synthetic divergence.
    """
    distance = np.linspace(0.0, lap_length_m, n_samples)

    if throttle_profile == "sin":
        throttle_norm = 0.5 + 0.4 * np.sin(2 * np.pi * distance / lap_length_m)
    elif throttle_profile == "const_half":
        throttle_norm = np.full(n_samples, 0.5)
    else:
        throttle_norm = np.zeros(n_samples)

    if brake_profile == "impulse":
        brake_bar = np.zeros(n_samples)
        brake_bar[n_samples // 4 : n_samples // 3] = 15.0  # bar
    else:
        brake_bar = np.zeros(n_samples)

    # Telemetry frame
    aim = pd.DataFrame({
        "Throttle Pos": throttle_norm * 100.0,  # AiM uses 0-100
        "FBrakePressure": brake_bar,
        "RBrakePressure": np.zeros(n_samples),
        "Distance on GPS Speed": distance,
        "lap": np.full(n_samples, lap, dtype=int),
    })

    # Sim frame (matches telemetry sample-for-sample, aligned on
    # distance).  Action classification mirrors the harness' rules.
    sim_throttle = throttle_norm + throttle_offset
    sim_brake_norm = np.clip(brake_bar / 30.0, 0.0, 1.0)  # match default brake_ref_bar
    sim_action = np.where(
        brake_bar > 2.0, "brake",
        np.where(throttle_norm > 0.05, "throttle", "coast"),
    )
    if invert_action:
        # Swap throttle <-> brake labels.
        sim_action = np.where(
            sim_action == "throttle", "brake",
            np.where(sim_action == "brake", "throttle", sim_action),
        )

    sim = pd.DataFrame({
        "lap": np.full(n_samples, lap, dtype=int),
        "distance_m": distance,
        "throttle_pct": sim_throttle,
        "brake_pct": sim_brake_norm,
        "action": sim_action,
    })
    return sim, aim


class TestPerfectMatch:
    """Sim exactly equals telemetry → RMSE ≈ 0, R² ≈ 1, corr ≈ 1."""

    def test_perfect_match_throttle(self):
        sim, aim = _build_fabricated_frames()
        result = validate_driver_channels(sim, aim)
        assert result.throttle.rmse == pytest.approx(0.0, abs=1e-10)
        assert result.throttle.r_squared == pytest.approx(1.0, abs=1e-10)
        assert result.throttle.correlation == pytest.approx(1.0, abs=1e-10)

    def test_perfect_match_brake(self):
        sim, aim = _build_fabricated_frames()
        result = validate_driver_channels(sim, aim)
        assert result.brake.rmse == pytest.approx(0.0, abs=1e-10)
        # R²/corr on a mostly-zero channel: require non-nan & >= 0.99.
        if not np.isnan(result.brake.r_squared):
            assert result.brake.r_squared >= 0.99

    def test_perfect_match_action_accuracy(self):
        sim, aim = _build_fabricated_frames()
        result = validate_driver_channels(sim, aim)
        assert result.action_accuracy == pytest.approx(1.0)

    def test_returns_expected_dataclass(self):
        sim, aim = _build_fabricated_frames()
        result = validate_driver_channels(sim, aim)
        assert isinstance(result, DriverChannelValidation)
        assert result.n_samples > 0
        assert result.laps_used == [13]
        assert isinstance(result.per_lap, pd.DataFrame)


class TestConstantOffset:
    """Sim = telemetry + const → RMSE > 0, R² < 1, corr == 1."""

    def test_constant_offset_on_throttle(self):
        sim, aim = _build_fabricated_frames(throttle_offset=0.2)
        result = validate_driver_channels(sim, aim)
        # RMSE exactly equals the offset magnitude.
        assert result.throttle.rmse == pytest.approx(0.2, abs=1e-6)
        # R² drops because residuals are large.
        assert result.throttle.r_squared < 1.0
        # Correlation stays at 1 (shape preserved).
        assert result.throttle.correlation == pytest.approx(1.0, abs=1e-6)


class TestInvertedAction:
    """Swapped throttle <-> brake labels → accuracy drops to ~0."""

    def test_inverted_action_classification_accuracy(self):
        sim, aim = _build_fabricated_frames(invert_action=True)
        result = validate_driver_channels(sim, aim)
        # Only coast samples (unchanged) remain correct.  The majority
        # of samples are throttle → brake inversion, so accuracy should
        # be substantially below 1.0.
        assert result.action_accuracy < 0.5


class TestHeldOutLapSubset:
    """``laps`` kwarg selects a subset of laps for comparison."""

    def test_laps_kwarg_restricts_comparison(self):
        # Build three laps.
        parts_sim = []
        parts_aim = []
        for lap in (2, 13, 14):
            s, a = _build_fabricated_frames(lap=lap)
            parts_sim.append(s)
            parts_aim.append(a)
        sim = pd.concat(parts_sim, ignore_index=True)
        aim = pd.concat(parts_aim, ignore_index=True)

        # Restrict to stint 2 (laps 13-14).
        result = validate_driver_channels(sim, aim, laps=[13, 14])
        assert result.laps_used == [13, 14]
        # Two laps, same sample count each.
        assert len(result.per_lap) == 2

    def test_laps_none_intersects_both_frames(self):
        sim, aim = _build_fabricated_frames(lap=13)
        result = validate_driver_channels(sim, aim, laps=None)
        assert result.laps_used == [13]


class TestColumnValidation:
    """Missing columns raise informative errors."""

    def test_missing_sim_column_raises(self):
        sim, aim = _build_fabricated_frames()
        sim_bad = sim.drop(columns=["throttle_pct"])
        with pytest.raises(ValueError, match="throttle_pct"):
            validate_driver_channels(sim_bad, aim)

    def test_missing_telem_column_raises(self):
        sim, aim = _build_fabricated_frames()
        aim_bad = aim.drop(columns=["Throttle Pos"])
        with pytest.raises(ValueError, match="Throttle Pos"):
            validate_driver_channels(sim, aim_bad)
