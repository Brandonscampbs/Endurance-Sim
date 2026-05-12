"""Tests for speed-dependent pack convection h_eff(v) = h_static + k_v * v.

Replaces the constant `thermal_conductance_w_per_k` in
`BatteryModel.heat_out_w` with a linear forced-convection model.  The
``vehicle_speed_ms`` parameter is plumbed through ``step`` and
``step_power``; legacy callers (no speed) collapse to the static term
and reproduce the pre-change conductance behaviour bit-identically.

References:
- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer* 7e
  (Wiley 2011), Ch. 7 (External flow) §7.2 (flat plate) — justifies
  the linear ``h ∝ v`` chord through the Re range we operate in
  (5-28 m/s).
- Plan ``docs/superpowers/plans/2026-05-06-battery-upgrades.md``
  Part 2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fsae_sim.vehicle import BatteryConfig, DischargeLimitPoint
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.data.loader import load_voltt_csv

VOLTT_CELL_PATH = Path(
    "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/"
    "2025_Pack_cell.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_battery_config(
    *,
    h_static_w_per_k: float = 0.0,
    k_v_w_per_k_per_ms: float = 0.0,
    ambient_temperature_c: float = 25.0,
) -> BatteryConfig:
    return BatteryConfig(
        cell_type="P45B",
        series=110,
        parallel=4,
        cell_voltage_min_v=2.55,
        cell_voltage_max_v=4.195,
        discharged_soc_pct=2.0,
        soc_taper_threshold_pct=85.0,
        soc_taper_rate_a_per_pct=1.0,
        discharge_limits=tuple([
            DischargeLimitPoint(30.0, 100.0),
            DischargeLimitPoint(65.0, 0.0),
        ]),
        h_static_w_per_k=h_static_w_per_k,
        k_v_w_per_k_per_ms=k_v_w_per_k_per_ms,
        ambient_temperature_c=ambient_temperature_c,
    )


@pytest.fixture
def calibrated_model():
    """Battery model calibrated against 2025 Voltt cell data with
    h_static + k_v default to 0 (adiabatic baseline)."""
    cfg = _make_battery_config(
        h_static_w_per_k=0.0, k_v_w_per_k_per_ms=0.0, ambient_temperature_c=25.0,
    )
    df = load_voltt_csv(VOLTT_CELL_PATH)
    model = BatteryModel(cfg)
    model.calibrate_from_voltt(df)
    return model


# ---------------------------------------------------------------------------
# heat_out_w unit tests
# ---------------------------------------------------------------------------

class TestHeatOutSpeedDependence:
    def test_h_eff_at_zero_speed_equals_h_static(
        self, calibrated_model: BatteryModel,
    ) -> None:
        """heat_out_w at v=0 must match the static-only formula exactly."""
        cfg = _make_battery_config(
            h_static_w_per_k=5.0,
            k_v_w_per_k_per_ms=0.5,
            ambient_temperature_c=25.0,
        )
        model = BatteryModel(cfg)
        # Bring calibration over by re-using cell data
        df = load_voltt_csv(VOLTT_CELL_PATH)
        model.calibrate_from_voltt(df)
        q_out = model.heat_out_w(temp_c=40.0, vehicle_speed_ms=0.0)
        expected = 5.0 * (40.0 - 25.0)  # 75 W
        assert q_out == pytest.approx(expected, rel=1e-9)

    def test_h_eff_at_20ms_scales_linearly(
        self, calibrated_model: BatteryModel,
    ) -> None:
        """heat_out_w at v=20 m/s uses (h_static + 20*k_v) * dT."""
        cfg = _make_battery_config(
            h_static_w_per_k=5.0,
            k_v_w_per_k_per_ms=0.5,
            ambient_temperature_c=25.0,
        )
        model = BatteryModel(cfg)
        df = load_voltt_csv(VOLTT_CELL_PATH)
        model.calibrate_from_voltt(df)
        q_out = model.heat_out_w(temp_c=40.0, vehicle_speed_ms=20.0)
        expected = (5.0 + 20.0 * 0.5) * (40.0 - 25.0)  # 15.0 * 15 = 225 W
        assert q_out == pytest.approx(expected, rel=1e-9)

    def test_h_eff_zero_dT_zero_q(
        self, calibrated_model: BatteryModel,
    ) -> None:
        """At T_pack == T_ambient the cooling is zero at any speed."""
        cfg = _make_battery_config(
            h_static_w_per_k=10.0,
            k_v_w_per_k_per_ms=1.0,
            ambient_temperature_c=25.0,
        )
        model = BatteryModel(cfg)
        df = load_voltt_csv(VOLTT_CELL_PATH)
        model.calibrate_from_voltt(df)
        assert model.heat_out_w(25.0, 0.0) == 0.0
        assert model.heat_out_w(25.0, 30.0) == 0.0

    def test_backward_compat_thermal_conductance_alias(self) -> None:
        """Existing ``thermal_conductance_w_per_k`` constructor kwarg still
        works (maps to ``h_static_w_per_k`` with ``k_v_w_per_k_per_ms=0``)
        so legacy configs and tests continue to pass bit-identically."""
        cfg = BatteryConfig(
            cell_type="P45B",
            series=110,
            parallel=4,
            cell_voltage_min_v=2.55,
            cell_voltage_max_v=4.195,
            discharged_soc_pct=2.0,
            soc_taper_threshold_pct=85.0,
            soc_taper_rate_a_per_pct=1.0,
            discharge_limits=tuple([DischargeLimitPoint(30.0, 100.0)]),
            thermal_conductance_w_per_k=8.0,  # legacy alias
            ambient_temperature_c=25.0,
        )
        assert cfg.h_static_w_per_k == pytest.approx(8.0)
        assert cfg.k_v_w_per_k_per_ms == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# step / step_power back-compat
# ---------------------------------------------------------------------------

class TestStepSignatureBackwardCompat:
    def test_step_without_vehicle_speed_uses_static_only(
        self, calibrated_model: BatteryModel,
    ) -> None:
        """Calling step without ``vehicle_speed_ms`` defaults to 0, which
        collapses to the static-only conductance — reproducing the
        pre-change behavior of `thermal_conductance_w_per_k * dT`."""
        cfg = _make_battery_config(
            h_static_w_per_k=10.0,
            k_v_w_per_k_per_ms=2.0,
            ambient_temperature_c=25.0,
        )
        model = BatteryModel(cfg)
        df = load_voltt_csv(VOLTT_CELL_PATH)
        model.calibrate_from_voltt(df)

        # With v=0 default, heat-out term uses h_static only.
        # No current (no heat in), pack starts at 40 C, ambient 25 C.
        # heat_out = 10 * (40 - 25) = 150 W
        # dT = -heat_out * dt / thermal_mass
        _, new_temp_no_v, _ = model.step(
            pack_current_a=0.0, dt_s=10.0, soc_pct=80.0, temp_c=40.0,
        )
        _, new_temp_explicit_zero, _ = model.step(
            pack_current_a=0.0, dt_s=10.0, soc_pct=80.0, temp_c=40.0,
            vehicle_speed_ms=0.0,
        )
        # Two calls must produce identical state.
        assert new_temp_no_v == pytest.approx(new_temp_explicit_zero, rel=1e-12)

    def test_step_with_high_speed_cools_faster(
        self, calibrated_model: BatteryModel,
    ) -> None:
        """At v>0 the convective coefficient is larger, so the pack
        cools faster (lower temperature after the step)."""
        cfg = _make_battery_config(
            h_static_w_per_k=5.0,
            k_v_w_per_k_per_ms=1.0,
            ambient_temperature_c=25.0,
        )
        model = BatteryModel(cfg)
        df = load_voltt_csv(VOLTT_CELL_PATH)
        model.calibrate_from_voltt(df)

        _, t_slow, _ = model.step(
            0.0, 10.0, 80.0, 40.0, vehicle_speed_ms=0.0,
        )
        _, t_fast, _ = model.step(
            0.0, 10.0, 80.0, 40.0, vehicle_speed_ms=20.0,
        )
        assert t_fast < t_slow

    def test_step_power_threads_vehicle_speed(
        self, calibrated_model: BatteryModel,
    ) -> None:
        """step_power forwards vehicle_speed_ms to step."""
        cfg = _make_battery_config(
            h_static_w_per_k=5.0,
            k_v_w_per_k_per_ms=1.0,
            ambient_temperature_c=25.0,
        )
        model = BatteryModel(cfg)
        df = load_voltt_csv(VOLTT_CELL_PATH)
        model.calibrate_from_voltt(df)

        _, t_slow, _, _ = model.step_power(
            terminal_power_w=0.0, dt_s=10.0, soc_pct=80.0, temp_c=40.0,
            vehicle_speed_ms=0.0,
        )
        _, t_fast, _, _ = model.step_power(
            terminal_power_w=0.0, dt_s=10.0, soc_pct=80.0, temp_c=40.0,
            vehicle_speed_ms=20.0,
        )
        assert t_fast < t_slow


# ---------------------------------------------------------------------------
# Calibration script smoke test
# ---------------------------------------------------------------------------

TELEMETRY_PATH = Path("Real-Car-Data-And-Stats/CleanedEndurance.csv")


@pytest.mark.skipif(
    not TELEMETRY_PATH.exists(),
    reason="endurance telemetry not available; calibration smoke test requires it",
)
class TestCalibrationScript:
    """Smoke test the calibration script.

    Verifies that the calibrator runs end-to-end against
    `CleanedEndurance.csv`, fits non-negative ``(h_static, k_v)`` within
    physical bounds, and writes a YAML snippet.  Residual targets per
    plan:
        - mean |T_residual| <= 1.0 K
        - peak |T_residual| <= 5.0 K
        - |final-temp residual| <= 1.0 K
    """

    def test_calibrate_thermal_script_runs(self, tmp_path: Path) -> None:
        from scripts.calibrate_thermal import calibrate_thermal_against_telemetry

        result = calibrate_thermal_against_telemetry(
            telemetry_csv=TELEMETRY_PATH,
            output_yaml=tmp_path / "thermal.yaml",
            voltt_cell_csv=VOLTT_CELL_PATH,
        )
        # Fitted params within physical bounds.
        assert 0.5 <= result.h_static_w_per_k <= 50.0
        assert 0.0 <= result.k_v_w_per_k_per_ms <= 10.0
        # Residual targets.
        assert result.mean_residual_k <= 1.0, (
            f"mean residual {result.mean_residual_k:.3f} K exceeds 1.0 K target"
        )
        assert result.peak_residual_k <= 5.0, (
            f"peak residual {result.peak_residual_k:.3f} K exceeds 5.0 K target"
        )
        assert abs(result.final_residual_k) <= 1.0, (
            f"final-temp residual {result.final_residual_k:+.3f} K "
            "exceeds ±1.0 K target"
        )
        # YAML output was written.
        assert (tmp_path / "thermal.yaml").exists()
