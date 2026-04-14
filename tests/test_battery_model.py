"""Tests for the runtime battery model."""

import numpy as np
import pytest

from fsae_sim.vehicle import BatteryConfig, DischargeLimitPoint
from fsae_sim.vehicle.battery_model import BatteryModel, BatteryState
from fsae_sim.data.loader import load_voltt_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ct16ev_battery_config():
    """CT-16EV battery config (110S4P P45B)."""
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
            DischargeLimitPoint(35.0, 85.0),
            DischargeLimitPoint(40.0, 65.0),
            DischargeLimitPoint(45.0, 55.0),
            DischargeLimitPoint(50.0, 45.0),
            DischargeLimitPoint(55.0, 40.0),
            DischargeLimitPoint(60.0, 35.0),
            DischargeLimitPoint(65.0, 0.0),
        ]),
    )


@pytest.fixture
def calibrated_model(ct16ev_battery_config, voltt_cell_path):
    """Battery model calibrated against 2025 Voltt cell data."""
    df = load_voltt_csv(voltt_cell_path)
    model = BatteryModel(ct16ev_battery_config, cell_capacity_ah=4.5)
    model.calibrate(df)
    return model


# ---------------------------------------------------------------------------
# Config and initialization
# ---------------------------------------------------------------------------

class TestBatteryModelInit:

    def test_uncalibrated_model_raises(self, ct16ev_battery_config):
        model = BatteryModel(ct16ev_battery_config)
        assert not model.calibrated
        with pytest.raises(RuntimeError, match="calibrated"):
            model.ocv(50.0)

    def test_calibration_flag(self, calibrated_model):
        assert calibrated_model.calibrated

    def test_pack_capacity(self, ct16ev_battery_config):
        model = BatteryModel(ct16ev_battery_config, cell_capacity_ah=4.5)
        assert model.pack_capacity_ah == 18.0


# ---------------------------------------------------------------------------
# OCV-SOC curve validation against Voltt data
# ---------------------------------------------------------------------------

class TestOCVCurve:

    def test_ocv_at_full_charge(self, calibrated_model):
        """OCV at 100% SOC should be ~4.19V (P45B fully charged)."""
        ocv = calibrated_model.ocv(100.0)
        assert 4.15 < ocv < 4.22

    def test_ocv_at_mid_soc(self, calibrated_model):
        """OCV at ~80% SOC should be ~4.00V."""
        ocv = calibrated_model.ocv(80.0)
        assert 3.90 < ocv < 4.10

    def test_ocv_at_low_soc(self, calibrated_model):
        """OCV at ~60% SOC should be ~3.78V."""
        ocv = calibrated_model.ocv(60.0)
        assert 3.65 < ocv < 3.90

    def test_ocv_monotonically_increasing(self, calibrated_model):
        """OCV should increase with SOC."""
        soc_points = np.linspace(10, 100, 50)
        ocv_points = [calibrated_model.ocv(s) for s in soc_points]
        for i in range(1, len(ocv_points)):
            assert ocv_points[i] >= ocv_points[i - 1], (
                f"OCV not monotonic at SOC={soc_points[i]:.1f}%"
            )

    def test_ocv_matches_voltt_data(self, calibrated_model, voltt_cell_path):
        """OCV interpolation should match Voltt OCV column within 1%."""
        df = load_voltt_csv(voltt_cell_path)
        # Sample every 1000 rows to avoid noise
        sample_idx = np.arange(0, len(df), 1000)
        for idx in sample_idx:
            soc = df["SOC [%]"].iloc[idx]
            expected_ocv = df["OCV [V]"].iloc[idx]
            model_ocv = calibrated_model.ocv(soc)
            rel_err = abs(model_ocv - expected_ocv) / expected_ocv
            assert rel_err < 0.01, (
                f"OCV mismatch at SOC={soc:.1f}%: model={model_ocv:.4f}, "
                f"expected={expected_ocv:.4f}, err={rel_err:.4f}"
            )


# ---------------------------------------------------------------------------
# Internal resistance
# ---------------------------------------------------------------------------

class TestInternalResistance:

    def test_resistance_positive(self, calibrated_model):
        """Resistance should always be positive."""
        for soc in [20, 40, 60, 80, 100]:
            r = calibrated_model.internal_resistance(soc)
            assert r > 0, f"Negative resistance at SOC={soc}%"

    def test_resistance_reasonable_range(self, calibrated_model):
        """P45B internal resistance should be 10-100 mOhm."""
        for soc in [30, 50, 70, 90]:
            r = calibrated_model.internal_resistance(soc)
            assert 0.005 < r < 0.15, (
                f"Unreasonable resistance {r:.4f} Ohm at SOC={soc}%"
            )


# ---------------------------------------------------------------------------
# Pack voltage under load
# ---------------------------------------------------------------------------

class TestPackVoltage:

    def test_no_load_voltage(self, calibrated_model):
        """At zero current, pack voltage should equal OCV * series."""
        soc = 90.0
        v_pack = calibrated_model.pack_voltage(soc, 0.0)
        expected = calibrated_model.ocv(soc) * 110
        assert abs(v_pack - expected) < 0.1

    def test_voltage_drops_under_load(self, calibrated_model):
        """Pack voltage should decrease with discharge current."""
        soc = 80.0
        v_no_load = calibrated_model.pack_voltage(soc, 0.0)
        v_30a = calibrated_model.pack_voltage(soc, 30.0)
        v_60a = calibrated_model.pack_voltage(soc, 60.0)
        assert v_no_load > v_30a > v_60a

    def test_pack_voltage_matches_voltt(self, calibrated_model, voltt_cell_path, voltt_pack_path):
        """Pack voltage under load should match Voltt pack data within 2%."""
        cell_df = load_voltt_csv(voltt_cell_path)
        pack_df = load_voltt_csv(voltt_pack_path)

        # Sample during active discharge periods
        sample_idx = np.arange(100, len(pack_df), 2000)
        errors = []
        for idx in sample_idx:
            soc = pack_df["SOC [%]"].iloc[idx]
            # Voltt convention: negative current = discharge
            pack_current = -pack_df["Current [A]"].iloc[idx]  # flip to positive=discharge
            expected_v = pack_df["Voltage [V]"].iloc[idx]

            if abs(pack_current) < 0.5:
                continue  # skip near-zero current points

            model_v = calibrated_model.pack_voltage(soc, pack_current)
            rel_err = abs(model_v - expected_v) / expected_v
            errors.append(rel_err)

        if errors:
            mean_err = np.mean(errors)
            assert mean_err < 0.02, f"Mean pack voltage error {mean_err:.4f} > 2%"


# ---------------------------------------------------------------------------
# Discharge limits
# ---------------------------------------------------------------------------

class TestDischargeLimits:

    def test_limit_at_30c(self, ct16ev_battery_config):
        model = BatteryModel(ct16ev_battery_config)
        assert model.max_discharge_current(30.0, 90.0) == 100.0

    def test_limit_at_65c(self, ct16ev_battery_config):
        model = BatteryModel(ct16ev_battery_config)
        assert model.max_discharge_current(65.0, 90.0) == 0.0

    def test_limit_interpolated(self, ct16ev_battery_config):
        """Between 30C and 35C, limit should interpolate between 100A and 85A."""
        model = BatteryModel(ct16ev_battery_config)
        limit = model.max_discharge_current(32.5, 90.0)
        assert 91.0 < limit < 94.0  # ~92.5A expected

    def test_soc_taper_below_threshold(self, ct16ev_battery_config):
        """At SOC=75% (10% below 85% threshold), limit should reduce by 10A."""
        model = BatteryModel(ct16ev_battery_config)
        full_limit = model.max_discharge_current(30.0, 90.0)  # 100A
        tapered = model.max_discharge_current(30.0, 75.0)  # 100 - 10 = 90A
        assert abs(tapered - 90.0) < 0.1

    def test_soc_taper_extreme(self, ct16ev_battery_config):
        """At very low SOC, taper should drive limit to zero."""
        model = BatteryModel(ct16ev_battery_config)
        limit = model.max_discharge_current(30.0, 2.0)
        # 100 - (85-2)*1 = 100 - 83 = 17A
        assert abs(limit - 17.0) < 0.1

    def test_no_taper_above_threshold(self, ct16ev_battery_config):
        model = BatteryModel(ct16ev_battery_config)
        limit_85 = model.max_discharge_current(30.0, 85.0)
        limit_95 = model.max_discharge_current(30.0, 95.0)
        assert limit_85 == limit_95 == 100.0


# ---------------------------------------------------------------------------
# SOC stepping (coulomb counting)
# ---------------------------------------------------------------------------

class TestSOCStepping:

    def test_discharge_reduces_soc(self, calibrated_model):
        new_soc, _, _ = calibrated_model.step(50.0, 1.0, 80.0, 30.0)
        assert new_soc < 80.0

    def test_zero_current_no_soc_change(self, calibrated_model):
        new_soc, _, _ = calibrated_model.step(0.0, 10.0, 80.0, 30.0)
        assert abs(new_soc - 80.0) < 1e-6

    def test_regen_increases_soc(self, calibrated_model):
        """Negative current (regen) should increase SOC."""
        new_soc, _, _ = calibrated_model.step(-10.0, 1.0, 50.0, 30.0)
        assert new_soc > 50.0

    def test_soc_clamped_at_zero(self, calibrated_model):
        """SOC should not go below 0."""
        new_soc, _, _ = calibrated_model.step(100.0, 100000.0, 1.0, 30.0)
        assert new_soc >= 0.0

    def test_soc_clamped_at_100(self, calibrated_model):
        """SOC should not go above 100."""
        new_soc, _, _ = calibrated_model.step(-100.0, 100000.0, 99.0, 30.0)
        assert new_soc <= 100.0

    def test_coulomb_counting_accuracy(self, calibrated_model):
        """Verify coulomb counting math: 1A for 1 hour on a 4.5Ah cell pack."""
        # 1A pack current for 3600s on 4P pack
        # Cell current = 1/4 = 0.25A
        # dSOC = 0.25 * 3600 / (4.5 * 3600) * 100 = 5.56%
        soc = 80.0
        new_soc, _, _ = calibrated_model.step(1.0, 3600.0, soc, 30.0)
        expected_dsoc = (1.0 / 4) / 4.5 * 100  # 5.556%
        actual_dsoc = soc - new_soc
        assert abs(actual_dsoc - expected_dsoc) < 0.01


# ---------------------------------------------------------------------------
# Thermal model
# ---------------------------------------------------------------------------

class TestThermalModel:

    def test_discharge_heats_up(self, calibrated_model):
        """Cell should warm up during discharge."""
        _, new_temp, _ = calibrated_model.step(50.0, 60.0, 80.0, 25.0)
        assert new_temp > 25.0

    def test_zero_current_no_heating(self, calibrated_model):
        """No current should produce no temperature change."""
        _, new_temp, _ = calibrated_model.step(0.0, 60.0, 80.0, 25.0)
        assert abs(new_temp - 25.0) < 1e-6

    def test_thermal_rise_reasonable(self, calibrated_model):
        """Sustained 15A avg for 30 min (realistic endurance load) from 25C."""
        soc = 95.0
        temp = 25.0
        dt = 1.0
        for _ in range(1800):
            # Alternate between 30A (accel) and 0A (coast) for ~15A avg
            current = 30.0 if _ % 2 == 0 else 0.0
            soc, temp, _ = calibrated_model.step(current, dt, soc, temp)
        # Voltt shows ~33C after full 30min run starting at 25C
        assert 27.0 < temp < 45.0


# ---------------------------------------------------------------------------
# Integration: full run validation against Voltt data
# ---------------------------------------------------------------------------

class TestVolttIntegration:

    def test_soc_trajectory_matches_voltt(self, calibrated_model, voltt_cell_path):
        """Step through the Voltt current profile and check SOC tracks."""
        df = load_voltt_csv(voltt_cell_path)

        soc = 100.0
        temp = 25.0
        dt = 0.1  # Voltt timestep

        # Sample every 5000 rows (spread across the run)
        checkpoints = [5000, 10000, 15000, len(df) - 1]
        checkpoint_idx = 0

        for i in range(1, len(df)):
            # Voltt current: negative = discharge, flip for our convention
            cell_current = df["Current [A]"].iloc[i]
            pack_current = -cell_current * calibrated_model.config.parallel

            soc, temp, _ = calibrated_model.step(pack_current, dt, soc, temp)

            if checkpoint_idx < len(checkpoints) and i == checkpoints[checkpoint_idx]:
                expected_soc = df["SOC [%]"].iloc[i]
                soc_err = abs(soc - expected_soc)
                assert soc_err < 2.0, (
                    f"SOC mismatch at row {i}: model={soc:.2f}%, "
                    f"Voltt={expected_soc:.2f}%, err={soc_err:.2f}%"
                )
                checkpoint_idx += 1
