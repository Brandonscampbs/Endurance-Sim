# Tire Grip Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the Pacejka tire model's grip scaling factor from endurance telemetry so corner speeds, speed profile, and energy consumption match reality.

**Architecture:** Add `grip_scale` to TireConfig/YAML, add `apply_grip_scale()` to PacejkaTireModel that scales LMUY/LMUX, add extraction function to compute the scale from telemetry lateral G. The Pacejka Magic Formula shape is preserved; only peak force magnitude changes.

**Tech Stack:** Python, NumPy, SciPy (existing Pacejka model), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/fsae_sim/vehicle/tire_model.py` | Modify | Add `apply_grip_scale()` method |
| `src/fsae_sim/vehicle/vehicle.py` | Modify | Add `grip_scale` field to `TireConfig` |
| `src/fsae_sim/sim/engine.py` | Modify | Apply grip scale during construction |
| `src/fsae_sim/analysis/telemetry_analysis.py` | Modify | Add `extract_tire_grip_scale()` function |
| `configs/ct16ev.yaml` | Modify | Add calibrated `grip_scale` value |
| `tests/test_tire_model.py` | Modify | Add grip scale tests |
| `tests/test_vehicle.py` | Modify | Add grip_scale config loading test |

---

### Task 1: Add `apply_grip_scale()` to PacejkaTireModel

**Files:**
- Modify: `src/fsae_sim/vehicle/tire_model.py:25-50`
- Test: `tests/test_tire_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tire_model.py`:

```python
class TestGripScale:
    """Verify grip scaling reduces peak force while preserving stiffness."""

    def test_apply_grip_scale_reduces_peak_lateral(self, tire_10psi: PacejkaTireModel) -> None:
        fz = 657.0  # nominal load
        peak_before = tire_10psi.peak_lateral_force(fz)
        tire_10psi.apply_grip_scale(0.5)
        peak_after = tire_10psi.peak_lateral_force(fz)
        assert peak_after == pytest.approx(peak_before * 0.5, rel=0.05)

    def test_apply_grip_scale_reduces_peak_longitudinal(self, tire_10psi: PacejkaTireModel) -> None:
        fz = 657.0
        peak_before = tire_10psi.peak_longitudinal_force(fz)
        tire_10psi.apply_grip_scale(0.5)
        peak_after = tire_10psi.peak_longitudinal_force(fz)
        assert peak_after == pytest.approx(peak_before * 0.5, rel=0.05)

    def test_apply_grip_scale_preserves_cornering_stiffness(self, tire_10psi: PacejkaTireModel) -> None:
        """Cornering stiffness Kya = B*C*D should be preserved because B compensates."""
        fz = 657.0
        small_alpha = 0.01  # rad, linear region
        fy_before = tire_10psi.lateral_force(small_alpha, fz)
        tire_10psi.apply_grip_scale(0.5)
        fy_after = tire_10psi.lateral_force(small_alpha, fz)
        # At small slip, Fy ~ Kya * alpha. Kya should be unchanged.
        assert fy_after == pytest.approx(fy_before, rel=0.10)

    def test_apply_grip_scale_1_is_noop(self, tire_10psi: PacejkaTireModel) -> None:
        fz = 657.0
        peak_before = tire_10psi.peak_lateral_force(fz)
        tire_10psi.apply_grip_scale(1.0)
        peak_after = tire_10psi.peak_lateral_force(fz)
        assert peak_after == pytest.approx(peak_before, rel=0.001)

    def test_apply_grip_scale_stacks(self, tire_10psi: PacejkaTireModel) -> None:
        """Calling twice should multiply scales."""
        fz = 657.0
        peak_original = tire_10psi.peak_lateral_force(fz)
        tire_10psi.apply_grip_scale(0.5)
        tire_10psi.apply_grip_scale(0.5)
        peak_after = tire_10psi.peak_lateral_force(fz)
        assert peak_after == pytest.approx(peak_original * 0.25, rel=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tire_model.py::TestGripScale -v`
Expected: FAIL with `AttributeError: 'PacejkaTireModel' object has no attribute 'apply_grip_scale'`

- [ ] **Step 3: Implement `apply_grip_scale()`**

Add method to `PacejkaTireModel` in `src/fsae_sim/vehicle/tire_model.py`, after the `_parse` method (around line 150):

```python
def apply_grip_scale(self, scale: float) -> None:
    """Scale tire grip by multiplying LMUY and LMUX scaling factors.

    This is the standard Pacejka mechanism for calibrating TTC rig data
    to on-car grip. Scales peak force (D = mu * Fz) while preserving
    cornering stiffness (B compensates since B = Kya / (C * D)).

    Args:
        scale: Grip multiplier. 1.0 = no change, 0.5 = halve peak grip.
            Values should be positive.
    """
    self.scaling["LMUY"] = self.scaling.get("LMUY", 1.0) * scale
    self.scaling["LMUX"] = self.scaling.get("LMUX", 1.0) * scale
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tire_model.py::TestGripScale -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/vehicle/tire_model.py tests/test_tire_model.py
git commit -m "feat: add apply_grip_scale() to PacejkaTireModel"
```

---

### Task 2: Add `grip_scale` to TireConfig and YAML loading

**Files:**
- Modify: `src/fsae_sim/vehicle/vehicle.py:26-32`
- Test: `tests/test_vehicle.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_vehicle.py`:

```python
class TestTireConfigGripScale:
    def test_grip_scale_default(self):
        tc = TireConfig(
            tir_file="path/to/file.tir",
            static_camber_front_deg=-1.25,
            static_camber_rear_deg=-1.25,
        )
        assert tc.grip_scale == 1.0

    def test_grip_scale_explicit(self):
        tc = TireConfig(
            tir_file="path/to/file.tir",
            static_camber_front_deg=-1.25,
            static_camber_rear_deg=-1.25,
            grip_scale=0.46,
        )
        assert tc.grip_scale == 0.46

    def test_grip_scale_from_yaml(self, tmp_path):
        yaml_content = (
            "name: test\nyear: 2025\ndescription: test\n"
            "vehicle:\n  mass_kg: 278.0\n  frontal_area_m2: 1.0\n  drag_coefficient: 1.5\n"
            "  rolling_resistance: 0.015\n  wheelbase_m: 1.549\n"
            "powertrain:\n  motor_speed_max_rpm: 2900\n  brake_speed_rpm: 2400\n"
            "  torque_limit_inverter_nm: 85.0\n  torque_limit_lvcu_nm: 150.0\n"
            "  iq_limit_a: 170.0\n  id_limit_a: 30.0\n  gear_ratio: 3.818\n"
            "  drivetrain_efficiency: 0.92\n"
            "battery:\n  cell_type: P45B\n  topology: {series: 110, parallel: 4}\n"
            "  cell_voltage_min_v: 2.55\n  cell_voltage_max_v: 4.20\n  discharged_soc_pct: 2.0\n"
            "  soc_taper: {threshold_pct: 85.0, rate_a_per_pct: 1.0}\n"
            "  discharge_limits:\n    - {temp_c: 30.0, max_current_a: 100.0}\n"
            "    - {temp_c: 65.0, max_current_a: 0.0}\n"
            "tire:\n  tir_file: path/to/tire.tir\n  static_camber_front_deg: -1.5\n"
            "  static_camber_rear_deg: -2.0\n  grip_scale: 0.46\n"
            "suspension:\n  roll_stiffness_front_nm_per_deg: 238.0\n"
            "  roll_stiffness_rear_nm_per_deg: 258.0\n  roll_center_height_front_mm: 88.9\n"
            "  roll_center_height_rear_mm: 63.5\n  roll_camber_front_deg_per_deg: -0.5\n"
            "  roll_camber_rear_deg_per_deg: -0.554\n  front_track_mm: 1194.0\n"
            "  rear_track_mm: 1168.0\n"
        )
        (tmp_path / "cfg.yaml").write_text(yaml_content)
        config = VehicleConfig.from_yaml(tmp_path / "cfg.yaml")
        assert config.tire.grip_scale == 0.46
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vehicle.py::TestTireConfigGripScale -v`
Expected: FAIL with `TypeError: TireConfig.__init__() got an unexpected keyword argument 'grip_scale'`

- [ ] **Step 3: Add `grip_scale` to TireConfig**

In `src/fsae_sim/vehicle/vehicle.py`, modify the `TireConfig` dataclass:

```python
@dataclass(frozen=True)
class TireConfig:
    """Tire model configuration."""

    tir_file: str
    static_camber_front_deg: float
    static_camber_rear_deg: float
    grip_scale: float = 1.0  # TTC-to-car grip calibration factor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vehicle.py::TestTireConfigGripScale -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all vehicle tests to check nothing broke**

Run: `pytest tests/test_vehicle.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/fsae_sim/vehicle/vehicle.py tests/test_vehicle.py
git commit -m "feat: add grip_scale field to TireConfig"
```

---

### Task 3: Apply grip_scale in SimulationEngine construction

**Files:**
- Modify: `src/fsae_sim/sim/engine.py:71-89`

- [ ] **Step 1: Apply grip_scale after tire model construction**

In `src/fsae_sim/sim/engine.py`, in the `__init__` method, add `apply_grip_scale` call after `PacejkaTireModel` construction. Modify the block starting at line 74:

```python
        if _HAS_TIRE_MODELS and tire_cfg is not None and susp_cfg is not None:
            tire_model = PacejkaTireModel(tire_cfg.tir_file)
            if tire_cfg.grip_scale != 1.0:
                tire_model.apply_grip_scale(tire_cfg.grip_scale)
            load_transfer = LoadTransferModel(vehicle.vehicle, susp_cfg)
```

The rest of the block stays the same.

- [ ] **Step 2: Run existing engine tests to verify nothing broke**

Run: `pytest tests/test_engine.py tests/test_engine_envelope.py -v`
Expected: All existing tests PASS (grip_scale defaults to 1.0)

- [ ] **Step 3: Commit**

```bash
git add src/fsae_sim/sim/engine.py
git commit -m "feat: apply tire grip_scale in SimulationEngine construction"
```

---

### Task 4: Add `extract_tire_grip_scale()` extraction function

**Files:**
- Modify: `src/fsae_sim/analysis/telemetry_analysis.py`
- Test: `tests/test_telemetry_analysis.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_telemetry_analysis.py`:

```python
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from fsae_sim.analysis.telemetry_analysis import extract_tire_grip_scale


class TestExtractTireGripScale:
    def test_known_lateral_g(self):
        """Synthetic data with known peak lateral G should produce expected scale."""
        n = 1000
        # Simulate driving with peak lateral G around 1.2
        rng = np.random.default_rng(42)
        lat_acc_g = rng.uniform(0.0, 1.3, size=n)
        speed_kmh = rng.uniform(25.0, 60.0, size=n)

        df = pd.DataFrame({
            "GPS LatAcc": lat_acc_g,
            "GPS Speed": speed_kmh,
        })

        # Mock tire model with peak mu = 2.66
        tire_model = MagicMock()
        tire_model.peak_lateral_force.return_value = 2.66 * 657.0  # mu * Fz

        result = extract_tire_grip_scale(
            aim_df=df,
            mass_kg=288.0,
            cla=2.18,
            tire_model=tire_model,
            fz_representative=657.0,
        )

        # 95th percentile of uniform(0, 1.3) ~ 1.235
        # Effective mu ~ 1.235 * 288 * 9.81 / (288 * 9.81 + downforce)
        # At ~42.5 km/h avg for high-G samples: downforce ~ 190 N
        # effective_mu ~ 1.235 * 2825 / (2825 + 190) ~ 1.158
        # scale = 1.158 / 2.66 ~ 0.435
        assert 0.3 < result["grip_scale"] < 0.6
        assert result["effective_mu_95"] > 0
        assert result["pacejka_mu"] == pytest.approx(2.66, rel=0.01)

    def test_filters_low_speed_samples(self):
        """Samples below 15 km/h should be excluded."""
        df = pd.DataFrame({
            "GPS LatAcc": [2.0, 2.0, 0.5, 0.5],
            "GPS Speed": [5.0, 10.0, 40.0, 50.0],  # first two below threshold
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

        # Only the 0.5g samples at 40/50 km/h should be used
        # (the 2.0g samples at 5/10 km/h are excluded)
        assert result["effective_mu_95"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_telemetry_analysis.py::TestExtractTireGripScale -v`
Expected: FAIL with `ImportError: cannot import name 'extract_tire_grip_scale'`

- [ ] **Step 3: Implement `extract_tire_grip_scale()`**

Add to `src/fsae_sim/analysis/telemetry_analysis.py`, after the imports:

```python
def extract_tire_grip_scale(
    aim_df: pd.DataFrame,
    mass_kg: float,
    cla: float,
    tire_model,
    fz_representative: float,
    *,
    min_speed_kmh: float = 15.0,
    min_lat_g: float = 0.3,
    percentile: float = 95.0,
    rho: float = 1.225,
) -> dict:
    """Extract tire grip scale factor from endurance telemetry.

    Computes the car's real effective friction coefficient from lateral
    acceleration data and compares to the Pacejka model's peak mu.
    The ratio is the LMUY scaling factor needed to calibrate TTC rig
    data to on-car grip.

    Args:
        aim_df: AiM telemetry DataFrame with GPS LatAcc (g) and GPS Speed (km/h).
        mass_kg: Total vehicle mass including driver (kg).
        cla: Downforce coefficient * area (ClA, m^2).
        tire_model: PacejkaTireModel instance (uncalibrated).
        fz_representative: Representative per-tire normal load (N) for
            computing Pacejka peak mu.
        min_speed_kmh: Minimum speed to include (filters parking/pit).
        min_lat_g: Minimum lateral G to include (filters straights).
        percentile: Percentile for peak grip extraction (default 95th).
        rho: Air density (kg/m^3).

    Returns:
        Dict with keys: grip_scale, effective_mu_95, pacejka_mu,
        n_samples, peak_lat_g.
    """
    g = 9.81
    speed_kmh = aim_df["GPS Speed"].values
    lat_g = np.abs(aim_df["GPS LatAcc"].values)

    # Filter: moving and cornering
    mask = (speed_kmh > min_speed_kmh) & (lat_g > min_lat_g)
    if np.sum(mask) < 10:
        raise ValueError(
            f"Not enough cornering samples: {np.sum(mask)} "
            f"(need >= 10 with speed > {min_speed_kmh} km/h and |lat_g| > {min_lat_g})"
        )

    speed_ms = speed_kmh[mask] * (1000.0 / 3600.0)
    lat_g_filtered = lat_g[mask]

    # Effective mu: accounts for downforce augmenting normal force
    lateral_force = mass_kg * lat_g_filtered * g
    downforce = 0.5 * rho * cla * speed_ms ** 2
    total_normal = mass_kg * g + downforce
    effective_mu = lateral_force / total_normal

    mu_at_percentile = float(np.percentile(effective_mu, percentile))

    # Pacejka peak mu at representative load
    pacejka_peak_fy = tire_model.peak_lateral_force(fz_representative)
    pacejka_mu = pacejka_peak_fy / fz_representative

    grip_scale = mu_at_percentile / pacejka_mu

    return {
        "grip_scale": float(grip_scale),
        "effective_mu_95": float(mu_at_percentile),
        "pacejka_mu": float(pacejka_mu),
        "n_samples": int(np.sum(mask)),
        "peak_lat_g": float(np.percentile(lat_g_filtered, percentile)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_telemetry_analysis.py::TestExtractTireGripScale -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/analysis/telemetry_analysis.py tests/test_telemetry_analysis.py
git commit -m "feat: add extract_tire_grip_scale() for telemetry-based calibration"
```

---

### Task 5: Compute calibrated value and update ct16ev.yaml

**Files:**
- Modify: `configs/ct16ev.yaml`

- [ ] **Step 1: Run extraction on real telemetry to get the value**

Run this script to compute the calibrated grip_scale:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from fsae_sim.data.loader import load_aim_csv
from fsae_sim.vehicle.vehicle import VehicleConfig
from fsae_sim.vehicle.tire_model import PacejkaTireModel
from fsae_sim.analysis.telemetry_analysis import extract_tire_grip_scale

_, aim_df = load_aim_csv('Real-Car-Data-And-Stats/2025 Endurance Data.csv')
config = VehicleConfig.from_yaml('configs/ct16ev.yaml')
tire = PacejkaTireModel(config.tire.tir_file)

# Representative Fz: static weight per tire + some downforce
fz_static = config.vehicle.mass_kg * 9.81 / 4  # ~706 N
result = extract_tire_grip_scale(
    aim_df=aim_df,
    mass_kg=config.vehicle.mass_kg,
    cla=config.vehicle.downforce_coefficient,
    tire_model=tire,
    fz_representative=fz_static,
)

print(f'Grip scale:       {result[\"grip_scale\"]:.4f}')
print(f'Effective mu 95%: {result[\"effective_mu_95\"]:.3f}')
print(f'Pacejka mu:       {result[\"pacejka_mu\"]:.3f}')
print(f'Peak lat G 95%:   {result[\"peak_lat_g\"]:.3f}')
print(f'Cornering samples: {result[\"n_samples\"]}')
"
```

Expected: `grip_scale` approximately 0.40-0.50. Record the exact value.

- [ ] **Step 2: Update ct16ev.yaml with the calibrated value**

In `configs/ct16ev.yaml`, add `grip_scale` to the tire section with the value from step 1:

```yaml
tire:
  tir_file: "Real-Car-Data-And-Stats/Tire Models from TTC/Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
  static_camber_front_deg: -1.25
  static_camber_rear_deg: -1.25
  grip_scale: <VALUE_FROM_STEP_1>  # TTC-to-car calibration from Michigan 2025 telemetry
```

- [ ] **Step 3: Run config loading tests**

Run: `pytest tests/test_vehicle.py -v`
Expected: All PASS. The new test `test_grip_scale_default` confirms backward compat; the ct16ev tests pass because `from_yaml` passes `grip_scale` through via `**tire_data`.

- [ ] **Step 4: Commit**

```bash
git add configs/ct16ev.yaml
git commit -m "feat: add calibrated grip_scale to ct16ev.yaml from Michigan 2025 telemetry"
```

---

### Task 6: Validate — run diagnostic and tier3 validation

**Files:**
- None modified (validation only)

- [ ] **Step 1: Run the tier3 validation to check corner speeds**

Run: `python scripts/validate_tier3.py 2>&1 | head -40`

Check that:
- Corner speed predictions for tight corners are now in the 30-50 km/h range (not 70+)
- Pacejka peak mu reports the scaled value (~1.1-1.3 instead of 2.66)

- [ ] **Step 2: Run the driver model validation**

Run: `python scripts/validate_driver_model.py 2>&1`

Check that:
- Driving time error improves from 10.3%
- Energy error improves from 37%
- Speed profile should show variation (not flat at 60 km/h)

- [ ] **Step 3: Run the segment-by-segment diagnostic**

Run: `python scripts/diagnose_accuracy.py 2>&1`

Check that:
- Speed errors in corners are reduced (the +20-29 km/h errors at tight corners should shrink)
- Speed profile has corners below 45 km/h where telemetry shows 30-40 km/h

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS. No regressions.

- [ ] **Step 5: Commit any validation script updates if needed**

If validation scripts need minor adjustments to report the new grip info, commit those.
