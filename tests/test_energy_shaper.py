"""Energy-budget shaper tests (Wave 4b Sub-task B).

The shaper takes a per-segment target speed and attenuates it when the
driver is over (or projected to be over) the stint energy budget. Two
strategies:

- FCFB (First-Come-First-Brake / Full-Capability-Full-Boost default):
  no shaping until the entire budget is exceeded; at exhaustion drop
  the target proportionally to the overage.
- LBP (Lap-Budget Proportional): at lap k of N, expect roughly k/N of
  the budget to be spent. If the driver is over the pro-rata share
  for the current lap, attenuate.

Both strategies are pure functions: no side effects on the speed
envelope; no randomness; same inputs -> same output.

Citation: Heilmeier et al. "A Quasi-Steady-State Lap Time Simulation
for Electrified Race Cars" (IEEE EVER 2019,
https://ieeexplore.ieee.org/document/8813646/). TUMFTM's open-source
laptime-simulation companion codebase ships LBP and a Least-Squares
allocator; LBP is the relevant taxonomy entry for this shaper.
"""

from __future__ import annotations

import pytest

from fsae_sim.driver.energy_shaper import EnergyShaper, EnergyShaperConfig


# ---------------------------------------------------------------------------
# FCFB
# ---------------------------------------------------------------------------


def test_fcfb_no_attenuation_when_under_budget():
    """With FCFB at 50% budget used and 10/22 laps in, the target speed
    must pass through unchanged (no shaping until total budget exceeded)."""
    cfg = EnergyShaperConfig(strategy="fcfb", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=0.5 * 6.4,
        lap_index=10,
        segment_progress=0.5,
    )
    assert v == pytest.approx(25.0, abs=1e-12)


def test_fcfb_attenuates_when_over_budget():
    """When energy used exceeds the total budget, FCFB must reduce target."""
    cfg = EnergyShaperConfig(strategy="fcfb", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=7.0,  # over 6.4 budget
        lap_index=20,
        segment_progress=0.5,
    )
    # Target reduced strictly below v_max; cannot go below 0.
    assert v < 25.0
    assert v >= 0.0


def test_fcfb_attenuation_increases_with_overage():
    """The further over budget, the more attenuation."""
    cfg = EnergyShaperConfig(strategy="fcfb", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v_small_over = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=6.5,
        lap_index=21,
        segment_progress=0.5,
    )
    v_big_over = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=8.0,
        lap_index=21,
        segment_progress=0.5,
    )
    assert v_big_over <= v_small_over - 1e-9


def test_fcfb_at_exact_budget_no_attenuation():
    """Boundary case: at exactly the total budget, no shaping yet."""
    cfg = EnergyShaperConfig(strategy="fcfb", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=6.4,
        lap_index=21,
        segment_progress=0.5,
    )
    assert v == pytest.approx(25.0, abs=1e-12)


# ---------------------------------------------------------------------------
# LBP (Lap-Budget Proportional)
# ---------------------------------------------------------------------------


def test_lbp_attenuates_when_over_lap_pro_rata():
    """At lap 10/22 with energy_used=5.0 kWh and budget 6.4 kWh, the
    pro-rata budget for lap 10 is (10+1)/22 * 6.4 = 3.20 kWh. The
    driver used 5.0 kWh, so they are above pro-rata by 1.8 kWh and
    LBP should attenuate the target speed.
    """
    cfg = EnergyShaperConfig(strategy="lbp", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=5.0,
        lap_index=10,
        segment_progress=0.5,
    )
    assert v < 25.0


def test_lbp_no_attenuation_when_under_lap_pro_rata():
    """At lap 10/22 with energy_used=2.5 kWh and budget 6.4 kWh, the
    pro-rata budget for the current lap is (10+1)/22 * 6.4 ~ 3.2 kWh.
    Driver used 2.5 — under, so no attenuation.
    """
    cfg = EnergyShaperConfig(strategy="lbp", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=2.5,
        lap_index=10,
        segment_progress=0.5,
    )
    assert v == pytest.approx(25.0, abs=1e-12)


def test_lbp_first_lap_pro_rata_uses_lap_one_share():
    """On the first lap (index 0), pro-rata budget is 1/22 * budget. If
    the driver has already used more, attenuate."""
    cfg = EnergyShaperConfig(strategy="lbp", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    pro_rata = (0 + 1) / 22 * 6.4  # ~0.291 kWh
    v_over = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=pro_rata + 0.5,  # 0.5 kWh over pro-rata
        lap_index=0,
        segment_progress=0.5,
    )
    v_under = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=pro_rata - 0.1,
        lap_index=0,
        segment_progress=0.5,
    )
    assert v_over < 25.0
    assert v_under == pytest.approx(25.0, abs=1e-12)


def test_lbp_last_lap_full_budget_no_attenuation_under():
    """At the last lap, the full budget is available; only attenuate
    if the driver has overspent the whole thing."""
    cfg = EnergyShaperConfig(strategy="lbp", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=25.0,
        energy_used_kwh=6.0,
        lap_index=21,  # last lap (0-indexed)
        segment_progress=0.5,
    )
    assert v == pytest.approx(25.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Determinism (same inputs -> same outputs)
# ---------------------------------------------------------------------------


def test_determinism_fcfb():
    cfg = EnergyShaperConfig(strategy="fcfb", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    args = dict(v_max=25.0, energy_used_kwh=7.5, lap_index=18, segment_progress=0.3)
    a = shaper.shape_target_speed(**args)
    b = shaper.shape_target_speed(**args)
    assert a == b


def test_determinism_lbp():
    cfg = EnergyShaperConfig(strategy="lbp", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    args = dict(v_max=25.0, energy_used_kwh=4.0, lap_index=10, segment_progress=0.5)
    a = shaper.shape_target_speed(**args)
    b = shaper.shape_target_speed(**args)
    assert a == b


# ---------------------------------------------------------------------------
# Sanity guards
# ---------------------------------------------------------------------------


def test_zero_v_max_passes_through_zero():
    """If the envelope target is already zero, the shaper cannot push
    it below zero. Idempotent floor at 0."""
    cfg = EnergyShaperConfig(strategy="fcfb", total_budget_kwh=6.4, laps_total=22)
    shaper = EnergyShaper(cfg)
    v = shaper.shape_target_speed(
        v_max=0.0,
        energy_used_kwh=10.0,  # massive overspend
        lap_index=21,
        segment_progress=0.9,
    )
    assert v == pytest.approx(0.0, abs=1e-12)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="Unknown energy shaper strategy"):
        EnergyShaperConfig(strategy="nope", total_budget_kwh=6.4, laps_total=22)
