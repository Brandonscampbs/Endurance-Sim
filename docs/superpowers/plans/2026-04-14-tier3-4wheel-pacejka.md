# Tier 3: 4-Wheel Model with Pacejka Tires — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the point-mass quasi-static sim with a 4-wheel model using real Pacejka tire data, 4-wheel load transfer with roll stiffness distribution, and a steady-state cornering solver.

**Architecture:** Bottom-up component replacement. Each layer is validated before the next depends on it: config dataclasses → tire model → load transfer model → cornering solver → VehicleDynamics wiring → SimEngine wiring. All new components are optional at the dynamics/engine level so existing tests pass unchanged until full integration.

**Tech Stack:** Python 3.12, pytest, NumPy/SciPy. PAC2002 .tir file format. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-13-tier3-simulation-upgrade-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `src/fsae_sim/vehicle/vehicle.py` | Add `TireConfig`, `SuspensionConfig` dataclasses; update `VehicleConfig` |
| Modify | `configs/ct16ev.yaml` | Add `tire:` and `suspension:` sections |
| Create | `src/fsae_sim/vehicle/tire_model.py` | `PacejkaTireModel` — .tir parser + Fy/Fx/combined/peak/radius |
| Create | `src/fsae_sim/vehicle/load_transfer.py` | `LoadTransferModel` — static/aero/lon/lat/combined loads |
| Create | `src/fsae_sim/vehicle/cornering_solver.py` | `CorneringSolver` — bisection over 4-wheel grip envelope |
| Modify | `src/fsae_sim/vehicle/dynamics.py` | Accept optional tire components; delegate cornering; add traction/braking limits |
| Modify | `src/fsae_sim/sim/engine.py` | Wire tire components from config; clamp drive/regen to traction limits |
| Modify | `src/fsae_sim/vehicle/__init__.py` | Export new classes |
| Create | `tests/test_tire_model.py` | Tire model tests |
| Create | `tests/test_load_transfer.py` | Load transfer tests |
| Create | `tests/test_cornering_solver.py` | Cornering solver tests |
| Modify | `tests/test_dynamics.py` | Add delegation, traction, braking tests |
| Modify | `tests/test_engine.py` | Add tire integration and clamping tests |

---

## Section A: Config & Dataclasses (Tasks 1–3)

### Task 1: TireConfig and SuspensionConfig dataclasses

**Files:**
- Modify: `src/fsae_sim/vehicle/vehicle.py`
- Test: `tests/test_vehicle.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_vehicle.py

from fsae_sim.vehicle.vehicle import TireConfig, SuspensionConfig


class TestTireConfig:

    def test_tire_config_construction(self):
        tc = TireConfig(
            tir_file="path/to/file.tir",
            static_camber_front_deg=-1.25,
            static_camber_rear_deg=-1.25,
        )
        assert tc.tir_file == "path/to/file.tir"
        assert tc.static_camber_front_deg == -1.25
        assert tc.static_camber_rear_deg == -1.25

    def test_tire_config_is_frozen(self):
        tc = TireConfig(
            tir_file="path/to/file.tir",
            static_camber_front_deg=-1.25,
            static_camber_rear_deg=-1.25,
        )
        with pytest.raises(AttributeError):
            tc.tir_file = "other.tir"

    def test_tire_config_requires_all_fields(self):
        with pytest.raises(TypeError):
            TireConfig(tir_file="path/to/file.tir")


class TestSuspensionConfig:

    def test_suspension_config_construction(self):
        sc = SuspensionConfig(
            roll_stiffness_front_nm_per_deg=238.0,
            roll_stiffness_rear_nm_per_deg=258.0,
            roll_center_height_front_mm=88.9,
            roll_center_height_rear_mm=63.5,
            roll_camber_front_deg_per_deg=-0.5,
            roll_camber_rear_deg_per_deg=-0.554,
            front_track_mm=1194.0,
            rear_track_mm=1168.0,
        )
        assert sc.roll_stiffness_front_nm_per_deg == 238.0
        assert sc.rear_track_mm == 1168.0

    def test_suspension_config_is_frozen(self):
        sc = SuspensionConfig(
            roll_stiffness_front_nm_per_deg=238.0,
            roll_stiffness_rear_nm_per_deg=258.0,
            roll_center_height_front_mm=88.9,
            roll_center_height_rear_mm=63.5,
            roll_camber_front_deg_per_deg=-0.5,
            roll_camber_rear_deg_per_deg=-0.554,
            front_track_mm=1194.0,
            rear_track_mm=1168.0,
        )
        with pytest.raises(AttributeError):
            sc.front_track_mm = 1200.0

    def test_suspension_config_requires_all_fields(self):
        with pytest.raises(TypeError):
            SuspensionConfig(
                roll_stiffness_front_nm_per_deg=238.0,
                roll_stiffness_rear_nm_per_deg=258.0,
            )
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_vehicle.py::TestTireConfig tests/test_vehicle.py::TestSuspensionConfig -v`
Expected: FAIL with `ImportError: cannot import name 'TireConfig'`

- [ ] **Step 3: Write implementation — add both dataclasses to vehicle.py**

Insert after the `VehicleParams` class, before `VehicleConfig`:
```python
@dataclass(frozen=True)
class TireConfig:
    """Tire model configuration."""
    tir_file: str
    static_camber_front_deg: float
    static_camber_rear_deg: float


@dataclass(frozen=True)
class SuspensionConfig:
    """Suspension geometry and compliance parameters (DSS values)."""
    roll_stiffness_front_nm_per_deg: float
    roll_stiffness_rear_nm_per_deg: float
    roll_center_height_front_mm: float
    roll_center_height_rear_mm: float
    roll_camber_front_deg_per_deg: float
    roll_camber_rear_deg_per_deg: float
    front_track_mm: float
    rear_track_mm: float
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_vehicle.py -v`
Expected: All PASS (9 existing + 6 new)

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/vehicle.py tests/test_vehicle.py
git commit -m "feat: add TireConfig and SuspensionConfig frozen dataclasses"
```

---

### Task 2: VehicleConfig with optional tire/suspension fields

**Files:**
- Modify: `src/fsae_sim/vehicle/vehicle.py`
- Test: `tests/test_vehicle.py`

- [ ] **Step 1: Write failing tests for optional fields and YAML parsing**
```python
# Append to tests/test_vehicle.py

class TestVehicleConfigOptionalFields:

    def test_existing_configs_load_without_tire_suspension(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.tire is None
        assert config.suspension is None

    def test_vehicle_config_with_tire_and_suspension(self):
        from fsae_sim.vehicle.powertrain import PowertrainConfig
        from fsae_sim.vehicle.battery import BatteryConfig

        vp = VehicleParams(
            mass_kg=278.0, frontal_area_m2=1.0, drag_coefficient=1.5,
            rolling_resistance=0.015, wheelbase_m=1.549,
        )
        pt = PowertrainConfig(
            motor_speed_max_rpm=2900, brake_speed_rpm=2400,
            torque_limit_inverter_nm=85.0, torque_limit_lvcu_nm=150.0,
            iq_limit_a=170.0, id_limit_a=30.0,
            gear_ratio=3.818, drivetrain_efficiency=0.92,
        )
        bt = BatteryConfig.from_dict({
            "cell_type": "P45B",
            "topology": {"series": 110, "parallel": 4},
            "cell_voltage_min_v": 2.55, "cell_voltage_max_v": 4.20,
            "discharged_soc_pct": 2.0,
            "soc_taper": {"threshold_pct": 85.0, "rate_a_per_pct": 1.0},
            "discharge_limits": [
                {"temp_c": 30.0, "max_current_a": 100.0},
                {"temp_c": 65.0, "max_current_a": 0.0},
            ],
        })
        tc = TireConfig(tir_file="path/to/file.tir",
                        static_camber_front_deg=-1.25,
                        static_camber_rear_deg=-1.25)
        sc = SuspensionConfig(
            roll_stiffness_front_nm_per_deg=238.0,
            roll_stiffness_rear_nm_per_deg=258.0,
            roll_center_height_front_mm=88.9,
            roll_center_height_rear_mm=63.5,
            roll_camber_front_deg_per_deg=-0.5,
            roll_camber_rear_deg_per_deg=-0.554,
            front_track_mm=1194.0, rear_track_mm=1168.0,
        )
        config = VehicleConfig(
            name="test", year=2025, description="test config",
            vehicle=vp, powertrain=pt, battery=bt, tire=tc, suspension=sc,
        )
        assert config.tire is tc
        assert config.suspension is sc

    def test_from_yaml_parses_tire_and_suspension(self, tmp_path):
        yaml_content = """\
name: test-car
year: 2025
description: "test"
vehicle:
  mass_kg: 278.0
  frontal_area_m2: 1.0
  drag_coefficient: 1.5
  rolling_resistance: 0.015
  wheelbase_m: 1.549
powertrain:
  motor_speed_max_rpm: 2900
  brake_speed_rpm: 2400
  torque_limit_inverter_nm: 85.0
  torque_limit_lvcu_nm: 150.0
  iq_limit_a: 170.0
  id_limit_a: 30.0
  gear_ratio: 3.818
  drivetrain_efficiency: 0.92
battery:
  cell_type: "P45B"
  topology: {series: 110, parallel: 4}
  cell_voltage_min_v: 2.55
  cell_voltage_max_v: 4.20
  discharged_soc_pct: 2.0
  soc_taper: {threshold_pct: 85.0, rate_a_per_pct: 1.0}
  discharge_limits:
    - {temp_c: 30.0, max_current_a: 100.0}
    - {temp_c: 65.0, max_current_a: 0.0}
tire:
  tir_file: "path/to/tire.tir"
  static_camber_front_deg: -1.5
  static_camber_rear_deg: -2.0
suspension:
  roll_stiffness_front_nm_per_deg: 238.0
  roll_stiffness_rear_nm_per_deg: 258.0
  roll_center_height_front_mm: 88.9
  roll_center_height_rear_mm: 63.5
  roll_camber_front_deg_per_deg: -0.5
  roll_camber_rear_deg_per_deg: -0.554
  front_track_mm: 1194.0
  rear_track_mm: 1168.0
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        config = VehicleConfig.from_yaml(config_file)

        assert config.tire is not None
        assert config.tire.tir_file == "path/to/tire.tir"
        assert config.tire.static_camber_front_deg == -1.5
        assert config.suspension is not None
        assert config.suspension.roll_stiffness_front_nm_per_deg == 238.0
        assert config.suspension.front_track_mm == 1194.0

    def test_from_yaml_tire_only_no_suspension(self, tmp_path):
        yaml_content = """\
name: test-car
year: 2025
description: "test"
vehicle:
  mass_kg: 278.0
  frontal_area_m2: 1.0
  drag_coefficient: 1.5
  rolling_resistance: 0.015
  wheelbase_m: 1.549
powertrain:
  motor_speed_max_rpm: 2900
  brake_speed_rpm: 2400
  torque_limit_inverter_nm: 85.0
  torque_limit_lvcu_nm: 150.0
  iq_limit_a: 170.0
  id_limit_a: 30.0
  gear_ratio: 3.818
  drivetrain_efficiency: 0.92
battery:
  cell_type: "P45B"
  topology: {series: 110, parallel: 4}
  cell_voltage_min_v: 2.55
  cell_voltage_max_v: 4.20
  discharged_soc_pct: 2.0
  soc_taper: {threshold_pct: 85.0, rate_a_per_pct: 1.0}
  discharge_limits:
    - {temp_c: 30.0, max_current_a: 100.0}
    - {temp_c: 65.0, max_current_a: 0.0}
tire:
  tir_file: "path/to/tire.tir"
  static_camber_front_deg: -1.25
  static_camber_rear_deg: -1.25
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        config = VehicleConfig.from_yaml(config_file)
        assert config.tire is not None
        assert config.suspension is None
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_vehicle.py::TestVehicleConfigOptionalFields -v`
Expected: FAIL with `TypeError: ... got an unexpected keyword argument 'tire'`

- [ ] **Step 3: Update VehicleConfig in vehicle.py**

Replace the `VehicleConfig` class:
```python
@dataclass(frozen=True)
class VehicleConfig:
    """Complete vehicle configuration loaded from YAML."""
    name: str
    year: int
    description: str
    vehicle: VehicleParams
    powertrain: PowertrainConfig
    battery: BatteryConfig
    tire: TireConfig | None = None
    suspension: SuspensionConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleConfig":
        """Load vehicle configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        tire_data = data.get("tire")
        suspension_data = data.get("suspension")

        return cls(
            name=data["name"],
            year=data["year"],
            description=data["description"],
            vehicle=VehicleParams(**data["vehicle"]),
            powertrain=PowertrainConfig(**data["powertrain"]),
            battery=BatteryConfig.from_dict(data["battery"]),
            tire=TireConfig(**tire_data) if tire_data is not None else None,
            suspension=SuspensionConfig(**suspension_data) if suspension_data is not None else None,
        )
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_vehicle.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/vehicle.py tests/test_vehicle.py
git commit -m "feat: add optional tire/suspension fields to VehicleConfig with YAML parsing"
```

---

### Task 3: YAML config + __init__.py exports

**Files:**
- Modify: `configs/ct16ev.yaml`
- Modify: `src/fsae_sim/vehicle/__init__.py`
- Test: `tests/test_vehicle.py`

- [ ] **Step 1: Write failing integration tests**
```python
# Append to tests/test_vehicle.py

class TestCT16EVTireSuspensionLoading:

    def test_ct16ev_tire_config_loaded(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.tire is not None
        assert config.tire.tir_file == (
            "Real-Car-Data-And-Stats/Tire Models from TTC/"
            "Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
        )
        assert config.tire.static_camber_front_deg == -1.25

    def test_ct16ev_suspension_config_loaded(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.suspension is not None
        assert config.suspension.roll_stiffness_front_nm_per_deg == 238.0
        assert config.suspension.front_track_mm == 1194.0


class TestInitExports:

    def test_tire_config_importable(self):
        from fsae_sim.vehicle import TireConfig
        assert TireConfig is not None

    def test_suspension_config_importable(self):
        from fsae_sim.vehicle import SuspensionConfig
        assert SuspensionConfig is not None
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_vehicle.py::TestCT16EVTireSuspensionLoading tests/test_vehicle.py::TestInitExports -v`
Expected: FAIL — YAML lacks tire section; import fails

- [ ] **Step 3a: Append to configs/ct16ev.yaml**
```yaml
tire:
  tir_file: "Real-Car-Data-And-Stats/Tire Models from TTC/Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
  static_camber_front_deg: -1.25
  static_camber_rear_deg: -1.25

suspension:
  roll_stiffness_front_nm_per_deg: 238.0
  roll_stiffness_rear_nm_per_deg: 258.0
  roll_center_height_front_mm: 88.9
  roll_center_height_rear_mm: 63.5
  roll_camber_front_deg_per_deg: -0.5
  roll_camber_rear_deg_per_deg: -0.554
  front_track_mm: 1194.0
  rear_track_mm: 1168.0
```

- [ ] **Step 3b: Update src/fsae_sim/vehicle/__init__.py**
```python
from fsae_sim.vehicle.vehicle import VehicleConfig, VehicleParams, TireConfig, SuspensionConfig
from fsae_sim.vehicle.powertrain import PowertrainConfig
from fsae_sim.vehicle.powertrain_model import PowertrainModel
from fsae_sim.vehicle.battery import BatteryConfig, DischargeLimitPoint

__all__ = [
    "VehicleConfig", "VehicleParams", "TireConfig", "SuspensionConfig",
    "PowertrainConfig", "PowertrainModel", "BatteryConfig", "DischargeLimitPoint",
]
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_vehicle.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add configs/ct16ev.yaml src/fsae_sim/vehicle/__init__.py tests/test_vehicle.py
git commit -m "feat: add tire/suspension config to ct16ev.yaml, export from vehicle package"
```

---

## Section B: Pacejka Tire Model (Tasks 4–8)

### Task 4: .tir file parser

**Files:**
- Create: `src/fsae_sim/vehicle/tire_model.py`
- Create: `tests/test_tire_model.py`

- [ ] **Step 1: Write failing tests**
```python
"""Tests for PAC2002 Pacejka tire model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from fsae_sim.vehicle.tire_model import PacejkaTireModel

PROJECT_ROOT = Path(__file__).parent.parent
TIR_DIR = PROJECT_ROOT / "Real-Car-Data-And-Stats" / "Tire Models from TTC"
TIR_10PSI = TIR_DIR / "Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
TIR_12PSI = TIR_DIR / "Round_8_Hoosier_LC0_16x7p5_10_on_8in_12psi_PAC02_UM2.tir"


@pytest.fixture
def tir_10psi_path() -> Path:
    if not TIR_10PSI.exists():
        pytest.skip(".tir file not available")
    return TIR_10PSI

@pytest.fixture
def tir_12psi_path() -> Path:
    if not TIR_12PSI.exists():
        pytest.skip(".tir file not available")
    return TIR_12PSI

@pytest.fixture
def tire(tir_10psi_path: Path) -> PacejkaTireModel:
    return PacejkaTireModel(tir_10psi_path)


class TestTirParser:

    def test_loads_without_error(self, tir_10psi_path):
        PacejkaTireModel(tir_10psi_path)

    def test_fnomin(self, tire):
        assert tire.fnomin == 657.0

    def test_unloaded_radius(self, tire):
        assert tire.unloaded_radius == pytest.approx(0.2042)

    def test_vertical_stiffness(self, tire):
        assert tire.vertical_stiffness == pytest.approx(87914.0)

    def test_lateral_coefficients_parsed(self, tire):
        assert tire.lateral["PCY1"] == pytest.approx(1.15122)
        assert tire.lateral["PDY1"] == pytest.approx(-2.66031)
        assert tire.lateral["PKY1"] == pytest.approx(-56.7924)

    def test_longitudinal_all_zero(self, tire):
        assert tire.longitudinal["PCX1"] == pytest.approx(0.0)
        assert tire.longitudinal["PDX1"] == pytest.approx(0.0)

    def test_scaling_coefficients(self, tire):
        assert tire.scaling["LFZO"] == pytest.approx(1.0)
        assert tire.scaling["LMUY"] == pytest.approx(1.0)

    def test_loaded_radius_coefficients(self, tire):
        assert tire.loaded_radius_coeffs["QV1"] == pytest.approx(403.112)

    def test_different_pressure_different_coefficients(self, tir_10psi_path, tir_12psi_path):
        t10 = PacejkaTireModel(tir_10psi_path)
        t12 = PacejkaTireModel(tir_12psi_path)
        assert t10.lateral["PCY1"] != pytest.approx(t12.lateral["PCY1"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PacejkaTireModel(tmp_path / "nonexistent.tir")
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_tire_model.py::TestTirParser -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**
```python
"""PAC2002 Pacejka tire model.

Parses .tir coefficient files (PAC2002 format from Stackpole / TTC) and
computes lateral (Fy), longitudinal (Fx), and combined forces.
"""

from __future__ import annotations

import math
import re
from pathlib import Path


class PacejkaTireModel:
    """PAC2002 Magic Formula tire model loaded from a .tir file."""

    _SECTION_MAP: dict[str, str] = {
        "LATERAL_COEFFICIENTS": "lateral",
        "LONGITUDINAL_COEFFICIENTS": "longitudinal",
        "SCALING_COEFFICIENTS": "scaling",
        "LOADED_RADIUS_COEFFICIENTS": "loaded_radius_coeffs",
        "VERTICAL": "_vertical",
        "DIMENSION": "_dimension",
    }

    def __init__(self, tir_path: str | Path) -> None:
        tir_path = Path(tir_path)
        if not tir_path.exists():
            raise FileNotFoundError(f"Tire file not found: {tir_path}")

        self.lateral: dict[str, float] = {}
        self.longitudinal: dict[str, float] = {}
        self.scaling: dict[str, float] = {}
        self.loaded_radius_coeffs: dict[str, float] = {}
        self._vertical: dict[str, float] = {}
        self._dimension: dict[str, float] = {}

        self._parse(tir_path)

        self.fnomin: float = self._vertical.get("FNOMIN", 657.0)
        self.unloaded_radius: float = self._dimension.get("UNLOADED_RADIUS", 0.2042)
        self.vertical_stiffness: float = self._vertical.get("VERTICAL_STIFFNESS", 87914.0)

    def _parse(self, tir_path: Path) -> None:
        current_attr: str | None = None
        section_re = re.compile(r"^\[(\w+)\]")
        kv_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([^\s$]+)")

        with open(tir_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("!") or line.startswith("$"):
                    continue
                sec_match = section_re.match(line)
                if sec_match:
                    current_attr = self._SECTION_MAP.get(sec_match.group(1))
                    continue
                if current_attr is None:
                    continue
                kv_match = kv_re.match(line)
                if kv_match:
                    key = kv_match.group(1)
                    value_str = kv_match.group(2).strip("'\"")
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue
                    getattr(self, current_attr)[key] = value
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_tire_model.py::TestTirParser -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/tire_model.py tests/test_tire_model.py
git commit -m "feat: PAC2002 .tir file parser for Pacejka tire model"
```

---

### Task 5: Lateral force (Fy)

**Files:**
- Modify: `src/fsae_sim/vehicle/tire_model.py`
- Modify: `tests/test_tire_model.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_tire_model.py

class TestLateralForce:

    def test_fy_at_5deg_657N(self, tire):
        fy = tire.lateral_force(math.radians(5), 657.0)
        assert fy == pytest.approx(147.40, abs=1.0)

    def test_fy_negative_at_negative_alpha(self, tire):
        fy = tire.lateral_force(math.radians(-5), 657.0)
        assert fy < 0

    def test_fy_antisymmetry(self, tire):
        fy_pos = tire.lateral_force(math.radians(5), 657.0)
        fy_neg = tire.lateral_force(math.radians(-5), 657.0)
        # Sum = 2 * SVy at Fz=657N ≈ 33.95N
        assert fy_pos + fy_neg == pytest.approx(33.95, abs=1.0)

    def test_fy_increases_with_load(self, tire):
        fy_200 = abs(tire.lateral_force(math.radians(5), 200.0))
        fy_657 = abs(tire.lateral_force(math.radians(5), 657.0))
        fy_1000 = abs(tire.lateral_force(math.radians(5), 1000.0))
        assert fy_200 < fy_657 < fy_1000

    def test_fy_load_sensitivity(self, tire):
        mu_200 = abs(tire.lateral_force(math.radians(10), 200.0)) / 200.0
        mu_657 = abs(tire.lateral_force(math.radians(10), 657.0)) / 657.0
        mu_1000 = abs(tire.lateral_force(math.radians(10), 1000.0)) / 1000.0
        assert mu_200 > mu_657 > mu_1000

    def test_fy_zero_slip_near_zero(self, tire):
        fy = tire.lateral_force(0.0, 657.0)
        assert abs(fy) < 20.0  # only SVy offset

    def test_fy_with_camber(self, tire):
        fy_no = tire.lateral_force(math.radians(5), 657.0)
        fy_cam = tire.lateral_force(math.radians(5), 657.0, math.radians(2))
        assert fy_cam > fy_no
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_tire_model.py::TestLateralForce -v`
Expected: FAIL with `AttributeError: ... 'lateral_force'`

- [ ] **Step 3: Write implementation**

Add to `PacejkaTireModel`:
```python
    def lateral_force(
        self, slip_angle_rad: float, normal_load_n: float, camber_rad: float = 0.0,
    ) -> float:
        """PAC2002 lateral force Fy."""
        lat = self.lateral
        sc = self.scaling

        fz = normal_load_n
        fz0 = self.fnomin * sc.get("LFZO", 1.0)
        dfz = (fz - fz0) / fz0

        muy = (lat["PDY1"] + lat["PDY2"] * dfz) * (1.0 - lat["PDY3"] * camber_rad ** 2) * sc.get("LMUY", 1.0)

        kya = (
            lat["PKY1"] * fz0
            * math.sin(lat["PKY2"] * math.atan(fz / (lat["PKY1"] * fz0)))
            * (1.0 - lat["PKY3"] * abs(camber_rad))
            * sc.get("LFZO", 1.0) * sc.get("LKY", 1.0)
        )

        cy = lat["PCY1"] * sc.get("LCY", 1.0)
        by = kya / (cy * muy * fz + 1e-6)

        shy = (lat["PHY1"] + lat["PHY2"] * dfz) * sc.get("LHY", 1.0) + lat["PHY3"] * camber_rad * sc.get("LKYG", 1.0)

        svy = fz * (
            (lat["PVY1"] + lat["PVY2"] * dfz) * sc.get("LVY", 1.0)
            + (lat["PVY3"] + lat["PVY4"] * dfz) * camber_rad
        ) * sc.get("LMUY", 1.0)

        alpha_star = slip_angle_rad + shy
        sign_a = 1.0 if alpha_star >= 0 else -1.0
        ey = (lat["PEY1"] + lat["PEY2"] * dfz) * (1.0 - (lat["PEY3"] + lat["PEY4"] * camber_rad) * sign_a) * sc.get("LEY", 1.0)
        ey = min(ey, 1.0)

        bx = by * alpha_star
        inner = bx - ey * (bx - math.atan(bx))
        return muy * fz * math.sin(cy * math.atan(inner)) + svy
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_tire_model.py::TestLateralForce -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/tire_model.py tests/test_tire_model.py
git commit -m "feat: PAC2002 lateral force (Fy) with Magic Formula"
```

---

### Task 6: Symmetric longitudinal force (Fx)

**Files:**
- Modify: `src/fsae_sim/vehicle/tire_model.py`
- Modify: `tests/test_tire_model.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_tire_model.py
import numpy as np

class TestLongitudinalForce:

    def test_fx_positive_at_positive_slip(self, tire):
        fx = tire.longitudinal_force(0.05, 657.0)
        assert fx > 50.0

    def test_fx_antisymmetric(self, tire):
        fx_pos = tire.longitudinal_force(0.1, 657.0)
        fx_neg = tire.longitudinal_force(-0.1, 657.0)
        assert fx_pos == pytest.approx(-fx_neg, abs=0.1)

    def test_fx_zero_at_zero_slip(self, tire):
        assert abs(tire.longitudinal_force(0.0, 657.0)) < 0.1

    def test_fx_increases_with_load(self, tire):
        fx_200 = abs(tire.longitudinal_force(0.1, 200.0))
        fx_657 = abs(tire.longitudinal_force(0.1, 657.0))
        fx_1000 = abs(tire.longitudinal_force(0.1, 1000.0))
        assert fx_200 < fx_657 < fx_1000

    def test_fx_peak_mu_matches_lateral(self, tire):
        fz = 657.0
        sr = np.linspace(-1.0, 1.0, 10000)
        peak_fx = max(abs(tire.longitudinal_force(float(s), fz)) for s in sr)
        alphas = np.linspace(-math.pi / 2, math.pi / 2, 10000)
        peak_fy = max(abs(tire.lateral_force(float(a), fz)) for a in alphas)
        assert peak_fx / fz == pytest.approx(peak_fy / fz, rel=0.05)

    def test_fx_load_sensitivity(self, tire):
        mu_200 = abs(tire.longitudinal_force(0.3, 200.0)) / 200.0
        mu_657 = abs(tire.longitudinal_force(0.3, 657.0)) / 657.0
        mu_1000 = abs(tire.longitudinal_force(0.3, 1000.0)) / 1000.0
        assert mu_200 > mu_657 > mu_1000
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_tire_model.py::TestLongitudinalForce -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write implementation**

Add to `PacejkaTireModel`:
```python
    def longitudinal_force(
        self, slip_ratio: float, normal_load_n: float, camber_rad: float = 0.0,
    ) -> float:
        """Symmetric Fx model mirroring lateral Pacejka structure.

        Since .tir Fx coefficients are all zero (USE_MODE=2), uses lateral
        coefficients mapped to longitudinal roles with |PDY1| for positive mu.
        """
        lat = self.lateral
        sc = self.scaling

        fz = normal_load_n
        fz0 = self.fnomin * sc.get("LFZO", 1.0)
        dfz = (fz - fz0) / fz0

        mux = (abs(lat["PDY1"]) + lat["PDY2"] * dfz) * (1.0 - lat["PDY3"] * camber_rad ** 2) * sc.get("LMUX", 1.0)

        kx = abs(
            lat["PKY1"] * fz0
            * math.sin(lat["PKY2"] * math.atan(fz / (lat["PKY1"] * fz0)))
            * (1.0 - lat["PKY3"] * abs(camber_rad))
            * sc.get("LFZO", 1.0) * sc.get("LKX", 1.0)
        )

        cx = lat["PCY1"] * sc.get("LCX", 1.0)
        bx = kx / (cx * mux * fz + 1e-6)

        ex = (lat["PEY1"] + lat["PEY2"] * dfz) * sc.get("LEX", 1.0)
        ex = min(ex, 1.0)

        bs = bx * slip_ratio
        inner = bs - ex * (bs - math.atan(bs))
        return mux * fz * math.sin(cx * math.atan(inner))
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_tire_model.py::TestLongitudinalForce -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/tire_model.py tests/test_tire_model.py
git commit -m "feat: symmetric longitudinal force (Fx) from lateral coefficients"
```

---

### Task 7: Combined forces, peak finders, loaded radius

**Files:**
- Modify: `src/fsae_sim/vehicle/tire_model.py`
- Modify: `tests/test_tire_model.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_tire_model.py

class TestCombinedForces:

    def test_pure_lateral_unchanged(self, tire):
        fy_pure = tire.lateral_force(math.radians(5), 657.0)
        fx, fy = tire.combined_forces(math.radians(5), 0.0, 657.0)
        assert abs(fx) < 0.1
        assert fy == pytest.approx(fy_pure, abs=0.1)

    def test_pure_longitudinal_unchanged(self, tire):
        fx_pure = tire.longitudinal_force(0.1, 657.0)
        fx, fy = tire.combined_forces(0.0, 0.1, 657.0)
        assert fx == pytest.approx(fx_pure, abs=0.1)

    def test_combined_within_friction_circle(self, tire):
        peak = max(tire.peak_lateral_force(657.0), tire.peak_longitudinal_force(657.0))
        fx, fy = tire.combined_forces(math.radians(10), 0.3, 657.0)
        assert math.sqrt(fx ** 2 + fy ** 2) <= peak * 1.01

    def test_returns_tuple_of_two(self, tire):
        result = tire.combined_forces(math.radians(5), 0.05, 657.0)
        assert isinstance(result, tuple) and len(result) == 2


class TestPeakForces:

    def test_peak_lateral_at_657N(self, tire):
        assert tire.peak_lateral_force(657.0) == pytest.approx(1481.8, abs=5.0)

    def test_peak_lateral_increases_with_load(self, tire):
        assert tire.peak_lateral_force(200.0) < tire.peak_lateral_force(657.0) < tire.peak_lateral_force(1000.0)

    def test_peak_longitudinal_matches_lateral(self, tire):
        peak_x = tire.peak_longitudinal_force(657.0)
        peak_y = tire.peak_lateral_force(657.0)
        assert peak_x == pytest.approx(peak_y, rel=0.05)

    def test_peak_longitudinal_positive(self, tire):
        assert tire.peak_longitudinal_force(657.0) > 0


class TestLoadedRadius:

    def test_less_than_unloaded(self, tire):
        assert tire.loaded_radius(657.0) < tire.unloaded_radius

    def test_at_nominal_load(self, tire):
        assert tire.loaded_radius(657.0) == pytest.approx(0.1967, abs=0.002)

    def test_decreases_with_load(self, tire):
        assert tire.loaded_radius(200.0) > tire.loaded_radius(657.0) > tire.loaded_radius(1000.0)

    def test_zero_load_equals_unloaded(self, tire):
        assert tire.loaded_radius(0.0) == pytest.approx(tire.unloaded_radius, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_tire_model.py::TestCombinedForces tests/test_tire_model.py::TestPeakForces tests/test_tire_model.py::TestLoadedRadius -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write implementation**

Add to `PacejkaTireModel`:
```python
    def peak_lateral_force(self, normal_load_n: float, camber_rad: float = 0.0) -> float:
        """Max |Fy| by sweeping slip angle from -pi/2 to pi/2."""
        n = 1000
        step = math.pi / n
        peak = 0.0
        for i in range(n + 1):
            alpha = -math.pi / 2 + i * step
            peak = max(peak, abs(self.lateral_force(alpha, normal_load_n, camber_rad)))
        return peak

    def peak_longitudinal_force(self, normal_load_n: float, camber_rad: float = 0.0) -> float:
        """Max |Fx| by sweeping slip ratio from -1.0 to 1.0."""
        n = 1000
        step = 2.0 / n
        peak = 0.0
        for i in range(n + 1):
            sr = -1.0 + i * step
            peak = max(peak, abs(self.longitudinal_force(sr, normal_load_n, camber_rad)))
        return peak

    def combined_forces(
        self, slip_angle_rad: float, slip_ratio: float, normal_load_n: float, camber_rad: float = 0.0,
    ) -> tuple[float, float]:
        """Friction-circle-scaled combined Fx and Fy."""
        fx0 = self.longitudinal_force(slip_ratio, normal_load_n, camber_rad)
        fy0 = self.lateral_force(slip_angle_rad, normal_load_n, camber_rad)
        magnitude = math.sqrt(fx0 ** 2 + fy0 ** 2)
        if magnitude < 1e-6:
            return (fx0, fy0)
        peak = max(abs(fx0), abs(fy0), self.peak_lateral_force(normal_load_n, camber_rad))
        if magnitude > peak:
            scale = peak / magnitude
            return (fx0 * scale, fy0 * scale)
        return (fx0, fy0)

    def loaded_radius(self, normal_load_n: float, speed_ms: float = 0.0) -> float:
        """Loaded tire radius from vertical stiffness."""
        r0 = self.unloaded_radius
        kz = self.vertical_stiffness
        if kz <= 0 or normal_load_n <= 0:
            return r0
        return max(r0 - normal_load_n / kz, 0.01)
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_tire_model.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/tire_model.py tests/test_tire_model.py
git commit -m "feat: combined forces, peak finders, and loaded radius"
```

---

### Task 8: Tire model package registration + integration tests

**Files:**
- Modify: `src/fsae_sim/vehicle/__init__.py`
- Modify: `tests/test_tire_model.py`

- [ ] **Step 1: Write failing integration tests**
```python
# Append to tests/test_tire_model.py

class TestIntegration:

    def test_import_from_package(self):
        from fsae_sim.vehicle import PacejkaTireModel
        assert PacejkaTireModel is not None

    def test_full_slip_sweep_no_nan(self, tire):
        for fz in [100.0, 400.0, 657.0, 1000.0]:
            for alpha_deg in range(-90, 91, 5):
                assert math.isfinite(tire.lateral_force(math.radians(alpha_deg), fz))
            for sr_pct in range(-100, 101, 5):
                assert math.isfinite(tire.longitudinal_force(sr_pct / 100.0, fz))

    def test_force_within_physical_bounds(self, tire):
        fz = 657.0
        mu_bound = 3.0
        alphas = np.linspace(-math.pi / 2, math.pi / 2, 500)
        for a in alphas:
            assert abs(tire.lateral_force(float(a), fz)) < mu_bound * fz
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_tire_model.py::TestIntegration::test_import_from_package -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add to __init__.py**

Add to `src/fsae_sim/vehicle/__init__.py`:
```python
from fsae_sim.vehicle.tire_model import PacejkaTireModel
```
Add `"PacejkaTireModel"` to `__all__`.

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_tire_model.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/__init__.py tests/test_tire_model.py
git commit -m "feat: register PacejkaTireModel in vehicle package, add integration tests"
```

---

## Section C: Load Transfer Model (Tasks 9–12)

> **Note:** `SuspensionConfig` is defined in `vehicle.py` (Task 1). `load_transfer.py` imports it from there.

### Task 9: LoadTransferModel constructor, static_loads, aero_loads

**Files:**
- Create: `src/fsae_sim/vehicle/load_transfer.py`
- Create: `tests/test_load_transfer.py`

- [ ] **Step 1: Write failing tests**
```python
"""Tests for load transfer model."""

import math
import pytest

from fsae_sim.vehicle.vehicle import VehicleParams, SuspensionConfig
from fsae_sim.vehicle.load_transfer import LoadTransferModel


@pytest.fixture
def ct16ev_vehicle():
    return VehicleParams(
        mass_kg=278.0, frontal_area_m2=1.0, drag_coefficient=1.502,
        rolling_resistance=0.015, wheelbase_m=1.549, downforce_coefficient=2.18,
    )

@pytest.fixture
def ct16ev_suspension():
    return SuspensionConfig(
        roll_stiffness_front_nm_per_deg=238.0, roll_stiffness_rear_nm_per_deg=258.0,
        roll_center_height_front_mm=88.9, roll_center_height_rear_mm=63.5,
        roll_camber_front_deg_per_deg=-0.5, roll_camber_rear_deg_per_deg=-0.554,
        front_track_mm=1194.0, rear_track_mm=1168.0,
    )

@pytest.fixture
def load_model(ct16ev_vehicle, ct16ev_suspension):
    return LoadTransferModel(ct16ev_vehicle, ct16ev_suspension)


class TestStaticLoads:

    def test_values(self, load_model):
        fl, fr, rl, rr = load_model.static_loads()
        assert fl == pytest.approx(613.62, abs=0.1)
        assert fr == pytest.approx(613.62, abs=0.1)
        assert rl == pytest.approx(749.97, abs=0.1)
        assert rr == pytest.approx(749.97, abs=0.1)

    def test_sum_to_weight(self, load_model):
        fl, fr, rl, rr = load_model.static_loads()
        assert fl + fr + rl + rr == pytest.approx(278.0 * 9.81, abs=0.01)

    def test_left_right_symmetry(self, load_model):
        fl, fr, rl, rr = load_model.static_loads()
        assert fl == pytest.approx(fr)
        assert rl == pytest.approx(rr)

    def test_rear_heavier(self, load_model):
        fl, _, rl, _ = load_model.static_loads()
        assert rl > fl


class TestAeroLoads:

    def test_zero_speed(self, load_model):
        df, dr = load_model.aero_loads(0.0)
        assert df == 0.0 and dr == 0.0

    def test_at_80kph(self, load_model):
        df, dr = load_model.aero_loads(80.0 / 3.6)
        assert df == pytest.approx(402.22, abs=1.0)
        assert dr == pytest.approx(257.16, abs=1.0)

    def test_sum_equals_total(self, load_model):
        v = 80.0 / 3.6
        df, dr = load_model.aero_loads(v)
        total = 0.5 * 1.225 * 2.18 * v ** 2
        assert df + dr == pytest.approx(total, abs=0.01)

    def test_scales_with_v_squared(self, load_model):
        df1, dr1 = load_model.aero_loads(10.0)
        df2, dr2 = load_model.aero_loads(20.0)
        assert (df2 + dr2) / (df1 + dr1) == pytest.approx(4.0, abs=0.01)

    def test_front_bias(self, load_model):
        df, dr = load_model.aero_loads(20.0)
        assert df / (df + dr) == pytest.approx(0.61, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_load_transfer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**
```python
"""Four-wheel load transfer model.

Computes vertical tire loads under combined longitudinal acceleration,
lateral acceleration, and aerodynamic downforce for an FSAE EV.
"""

from __future__ import annotations

import math

from fsae_sim.vehicle.vehicle import VehicleParams, SuspensionConfig

GRAVITY: float = 9.81
AIR_DENSITY: float = 1.225


class LoadTransferModel:
    """Four-wheel load transfer model.

    Tire order convention: (front-left, front-right, rear-left, rear-right).
    Positive lateral_g = right turn (load shifts to left tires).
    Positive longitudinal_g = accelerating (load shifts rearward).
    """

    def __init__(
        self,
        vehicle: VehicleParams,
        suspension: SuspensionConfig,
        cg_height_m: float = 0.2794,
        weight_dist_front: float = 0.45,
        downforce_dist_front: float = 0.61,
    ) -> None:
        self.mass = vehicle.mass_kg
        self.cg_height = cg_height_m
        self.wheelbase = vehicle.wheelbase_m
        self.weight_dist_front = weight_dist_front
        self.downforce_dist_front = downforce_dist_front
        self.cl_a = vehicle.downforce_coefficient

        self.front_track = suspension.front_track_mm / 1000.0
        self.rear_track = suspension.rear_track_mm / 1000.0
        self.rc_front = suspension.roll_center_height_front_mm / 1000.0
        self.rc_rear = suspension.roll_center_height_rear_mm / 1000.0

        self.roll_stiffness_front = suspension.roll_stiffness_front_nm_per_deg * 180.0 / math.pi
        self.roll_stiffness_rear = suspension.roll_stiffness_rear_nm_per_deg * 180.0 / math.pi
        self.k_roll_total = self.roll_stiffness_front + self.roll_stiffness_rear

        cg_from_front = self.wheelbase * (1.0 - self.weight_dist_front)
        self.rc_at_cg = self.rc_front + (self.rc_rear - self.rc_front) * cg_from_front / self.wheelbase

    def static_loads(self) -> tuple[float, float, float, float]:
        weight = self.mass * GRAVITY
        front_axle = weight * self.weight_dist_front
        rear_axle = weight * (1.0 - self.weight_dist_front)
        return front_axle / 2.0, front_axle / 2.0, rear_axle / 2.0, rear_axle / 2.0

    def aero_loads(self, speed_ms: float) -> tuple[float, float]:
        v = abs(speed_ms)
        total = 0.5 * AIR_DENSITY * self.cl_a * v * v
        return total * self.downforce_dist_front, total * (1.0 - self.downforce_dist_front)
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_load_transfer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/load_transfer.py tests/test_load_transfer.py
git commit -m "feat(load-transfer): constructor, static_loads, aero_loads"
```

---

### Task 10: Longitudinal and lateral transfer

**Files:**
- Modify: `src/fsae_sim/vehicle/load_transfer.py`
- Modify: `tests/test_load_transfer.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_load_transfer.py
from fsae_sim.vehicle.load_transfer import GRAVITY


class TestLongitudinalTransfer:

    def test_zero_accel(self, load_model):
        assert load_model.longitudinal_transfer(0.0) == 0.0

    def test_1g_value(self, load_model):
        # 278 * 1.0 * 9.81 * 0.2794 / 1.549 = 491.91 N
        assert load_model.longitudinal_transfer(1.0) == pytest.approx(491.91, abs=0.1)

    def test_1_5g_braking(self, load_model):
        assert load_model.longitudinal_transfer(-1.5) == pytest.approx(-737.87, abs=0.1)

    def test_linearity(self, load_model):
        assert load_model.longitudinal_transfer(1.0) == pytest.approx(2.0 * load_model.longitudinal_transfer(0.5), abs=0.01)


class TestLateralTransfer:

    def test_zero_lateral(self, load_model):
        df, dr = load_model.lateral_transfer(0.0, 0.0)
        assert df == 0.0 and dr == 0.0

    def test_1g_front(self, load_model):
        df, _ = load_model.lateral_transfer(1.0, 0.0)
        assert df == pytest.approx(315.47, abs=1.0)

    def test_1g_rear(self, load_model):
        _, dr = load_model.lateral_transfer(1.0, 0.0)
        assert dr == pytest.approx(329.88, abs=1.0)

    def test_moment_balance(self, load_model):
        df, dr = load_model.lateral_transfer(1.0, 0.0)
        reconstructed = df * load_model.front_track + dr * load_model.rear_track
        expected = load_model.mass * 1.0 * GRAVITY * load_model.cg_height
        assert reconstructed == pytest.approx(expected, abs=0.1)

    def test_rear_biased_elastic(self, load_model):
        df, dr = load_model.lateral_transfer(1.0, 0.0)
        assert dr > df
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_load_transfer.py::TestLongitudinalTransfer tests/test_load_transfer.py::TestLateralTransfer -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write implementation**

Add to `LoadTransferModel`:
```python
    def longitudinal_transfer(self, accel_g: float) -> float:
        """Longitudinal load transfer (N). Positive = rearward shift."""
        return self.mass * accel_g * GRAVITY * self.cg_height / self.wheelbase

    def lateral_transfer(self, lateral_g: float, speed_ms: float) -> tuple[float, float]:
        """Lateral load transfer per axle (N): geometric + elastic."""
        lat_g_abs = abs(lateral_g)
        f_lat = self.mass * lat_g_abs * GRAVITY

        m_front = self.mass * self.weight_dist_front
        m_rear = self.mass * (1.0 - self.weight_dist_front)

        delta_geo_front = m_front * lat_g_abs * GRAVITY * self.rc_front / self.front_track
        delta_geo_rear = m_rear * lat_g_abs * GRAVITY * self.rc_rear / self.rear_track

        roll_moment = f_lat * (self.cg_height - self.rc_at_cg)

        if self.k_roll_total > 0:
            delta_elastic_front = roll_moment * self.roll_stiffness_front / self.k_roll_total / self.front_track
            delta_elastic_rear = roll_moment * self.roll_stiffness_rear / self.k_roll_total / self.rear_track
        else:
            delta_elastic_front = delta_elastic_rear = 0.0

        return delta_geo_front + delta_elastic_front, delta_geo_rear + delta_elastic_rear
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_load_transfer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/load_transfer.py tests/test_load_transfer.py
git commit -m "feat(load-transfer): longitudinal and lateral transfer with hand-calc validation"
```

---

### Task 11: Combined tire_loads method

**Files:**
- Modify: `src/fsae_sim/vehicle/load_transfer.py`
- Modify: `tests/test_load_transfer.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_load_transfer.py

class TestTireLoads:

    def test_stationary_equals_static(self, load_model):
        fl, fr, rl, rr = load_model.tire_loads(0.0, 0.0, 0.0)
        sfl, sfr, srl, srr = load_model.static_loads()
        assert fl == pytest.approx(sfl) and fr == pytest.approx(sfr)
        assert rl == pytest.approx(srl) and rr == pytest.approx(srr)

    def test_sum_equals_weight_plus_downforce(self, load_model):
        v = 80.0 / 3.6
        fl, fr, rl, rr = load_model.tire_loads(v, 0.0, 0.0)
        expected = 278.0 * 9.81 + 0.5 * 1.225 * 2.18 * v ** 2
        assert fl + fr + rl + rr == pytest.approx(expected, abs=0.1)

    def test_sum_conserved_under_combined(self, load_model):
        v = 80.0 / 3.6
        fl, fr, rl, rr = load_model.tire_loads(v, 1.0, 0.5)
        expected = 278.0 * 9.81 + 0.5 * 1.225 * 2.18 * v ** 2
        assert fl + fr + rl + rr == pytest.approx(expected, abs=0.1)

    def test_1_5g_braking_no_negative(self, load_model):
        fl, fr, rl, rr = load_model.tire_loads(0.0, 0.0, -1.5)
        assert fl >= 0 and fr >= 0 and rl >= 0 and rr >= 0

    def test_right_turn_loads_left(self, load_model):
        fl, fr, rl, rr = load_model.tire_loads(0.0, 1.0, 0.0)
        assert fl > fr and rl > rr

    def test_left_right_symmetry(self, load_model):
        fl_r, fr_r, rl_r, rr_r = load_model.tire_loads(10.0, 1.0, 0.0)
        fl_l, fr_l, rl_l, rr_l = load_model.tire_loads(10.0, -1.0, 0.0)
        assert fl_r == pytest.approx(fr_l, abs=0.01)
        assert rl_r == pytest.approx(rr_l, abs=0.01)

    def test_extreme_decel_clamps(self, load_model):
        fl, fr, rl, rr = load_model.tire_loads(0.0, 0.0, -3.0)
        assert rl >= 0 and rr >= 0
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_load_transfer.py::TestTireLoads -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write implementation**

Add to `LoadTransferModel`:
```python
    def tire_loads(
        self, speed_ms: float, lateral_g: float, longitudinal_g: float,
    ) -> tuple[float, float, float, float]:
        """Combined vertical loads (FL, FR, RL, RR) in Newtons, clamped >= 0."""
        fl, fr, rl, rr = self.static_loads()

        aero_f, aero_r = self.aero_loads(speed_ms)
        fl += aero_f / 2.0
        fr += aero_f / 2.0
        rl += aero_r / 2.0
        rr += aero_r / 2.0

        delta_long = self.longitudinal_transfer(longitudinal_g)
        fl -= delta_long / 2.0
        fr -= delta_long / 2.0
        rl += delta_long / 2.0
        rr += delta_long / 2.0

        delta_lat_f, delta_lat_r = self.lateral_transfer(lateral_g, speed_ms)
        if lateral_g >= 0:
            fl += delta_lat_f;  fr -= delta_lat_f
            rl += delta_lat_r;  rr -= delta_lat_r
        else:
            fl -= delta_lat_f;  fr += delta_lat_f
            rl -= delta_lat_r;  rr += delta_lat_r

        return max(fl, 0.0), max(fr, 0.0), max(rl, 0.0), max(rr, 0.0)
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_load_transfer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/load_transfer.py tests/test_load_transfer.py
git commit -m "feat(load-transfer): combined tire_loads with conservation and clamping"
```

---

### Task 12: Load transfer package export

**Files:**
- Modify: `src/fsae_sim/vehicle/__init__.py`
- Modify: `tests/test_load_transfer.py`

- [ ] **Step 1: Write failing test**
```python
# Append to tests/test_load_transfer.py

class TestModuleExports:

    def test_importable_from_package(self):
        from fsae_sim.vehicle import LoadTransferModel
        assert LoadTransferModel is not None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_load_transfer.py::TestModuleExports -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add export to __init__.py**

Add to `src/fsae_sim/vehicle/__init__.py`:
```python
from fsae_sim.vehicle.load_transfer import LoadTransferModel
```
Add `"LoadTransferModel"` to `__all__`.

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_load_transfer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/__init__.py tests/test_load_transfer.py
git commit -m "feat(load-transfer): export from vehicle package"
```

---

## Section D: Cornering Solver (Tasks 13–15)

### Task 13: CorneringSolver with bisection — analytical verification

**Files:**
- Create: `src/fsae_sim/vehicle/cornering_solver.py`
- Create: `tests/test_cornering_solver.py`

- [ ] **Step 1: Write failing tests**
```python
"""Tests for steady-state cornering speed solver."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from fsae_sim.vehicle.cornering_solver import CorneringSolver


def make_linear_tire(mu: float = 1.3) -> MagicMock:
    tire = MagicMock()
    tire.peak_lateral_force.side_effect = lambda fz, camber=0.0: mu * fz
    return tire


def make_static_load_transfer(mass_kg: float) -> MagicMock:
    g = 9.81
    quarter = mass_kg * g / 4.0
    lt = MagicMock()
    lt.tire_loads.return_value = (quarter, quarter, quarter, quarter)
    lt.roll_stiffness_front = 1e6
    lt.roll_stiffness_rear = 1e6
    return lt


@pytest.fixture
def simple_solver() -> CorneringSolver:
    mass = 278.0
    return CorneringSolver(
        tire_model=make_linear_tire(1.3),
        load_transfer=make_static_load_transfer(mass),
        mass_kg=mass,
        static_camber_front_rad=0.0, static_camber_rear_rad=0.0,
        roll_camber_front=0.0, roll_camber_rear=0.0,
    )


class TestStraightSegment:

    def test_zero_curvature_infinity(self, simple_solver):
        assert simple_solver.max_cornering_speed(0.0) == float("inf")

    def test_near_zero_curvature_infinity(self, simple_solver):
        assert simple_solver.max_cornering_speed(1e-7) == float("inf")


class TestAnalyticalMatch:
    """Linear tire, no downforce: v = sqrt(mu * g / kappa)."""

    @pytest.mark.parametrize("curvature", [0.2, 1.0/15.0, 0.01, 0.05])
    def test_matches_formula(self, simple_solver, curvature):
        v_analytical = math.sqrt(1.3 * 9.81 / curvature)
        v_solver = simple_solver.max_cornering_speed(curvature)
        assert abs(v_solver - v_analytical) < 0.15

    def test_tighter_corner_slower(self, simple_solver):
        assert simple_solver.max_cornering_speed(0.01) > simple_solver.max_cornering_speed(0.1)

    def test_higher_mu_faster(self):
        mass = 278.0
        lt = make_static_load_transfer(mass)
        s_low = CorneringSolver(make_linear_tire(1.0), lt, mass, 0, 0, 0, 0)
        s_high = CorneringSolver(make_linear_tire(1.5), make_static_load_transfer(mass), mass, 0, 0, 0, 0)
        assert s_high.max_cornering_speed(0.05) > s_low.max_cornering_speed(0.05)

    def test_negative_curvature_same_as_positive(self, simple_solver):
        assert abs(simple_solver.max_cornering_speed(0.05) - simple_solver.max_cornering_speed(-0.05)) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_cornering_solver.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**
```python
"""Steady-state cornering speed solver.

Finds max speed through a corner using 4-wheel tire grip and load transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsae_sim.vehicle.tire_model import PacejkaTireModel
    from fsae_sim.vehicle.load_transfer import LoadTransferModel


class CorneringSolver:
    """Bisection search for maximum cornering speed."""

    GRAVITY: float = 9.81
    _V_LOW: float = 0.5
    _V_HIGH: float = 50.0
    _ITERATIONS: int = 30
    _CURVATURE_THRESHOLD: float = 1e-6

    def __init__(
        self,
        tire_model: PacejkaTireModel,
        load_transfer: LoadTransferModel,
        mass_kg: float,
        static_camber_front_rad: float,
        static_camber_rear_rad: float,
        roll_camber_front: float,
        roll_camber_rear: float,
    ) -> None:
        self.tire_model = tire_model
        self.load_transfer = load_transfer
        self.mass_kg = mass_kg
        self.static_camber_front_rad = static_camber_front_rad
        self.static_camber_rear_rad = static_camber_rear_rad
        self.roll_camber_front = roll_camber_front
        self.roll_camber_rear = roll_camber_rear

    def max_cornering_speed(self, curvature: float, mu_scale: float = 1.0) -> float:
        """Maximum speed (m/s) through a corner. Returns inf for straights."""
        kappa = abs(curvature)
        if kappa < self._CURVATURE_THRESHOLD:
            return float("inf")

        v_low, v_high = self._V_LOW, self._V_HIGH
        for _ in range(self._ITERATIONS):
            v_mid = (v_low + v_high) / 2.0
            if self._can_sustain(v_mid, kappa, mu_scale):
                v_low = v_mid
            else:
                v_high = v_mid
        return v_low

    def _can_sustain(self, speed: float, curvature: float, mu_scale: float) -> bool:
        a_lat = speed * speed * curvature
        a_lat_g = a_lat / self.GRAVITY
        required = self.mass_kg * a_lat

        fl_fz, fr_fz, rl_fz, rr_fz = self.load_transfer.tire_loads(speed, a_lat_g, 0.0)

        total_lat_force = self.mass_kg * a_lat
        k_total = self.load_transfer.roll_stiffness_front + self.load_transfer.roll_stiffness_rear
        roll = total_lat_force / k_total if k_total > 0 else 0.0

        # Camber per wheel: roll_camber is deg/deg, roll is rad — conversions cancel
        cam_fl = self.static_camber_front_rad + roll * self.roll_camber_front
        cam_fr = self.static_camber_front_rad - roll * self.roll_camber_front
        cam_rl = self.static_camber_rear_rad + roll * self.roll_camber_rear
        cam_rr = self.static_camber_rear_rad - roll * self.roll_camber_rear

        capacity = (
            self.tire_model.peak_lateral_force(fl_fz, cam_fl) * mu_scale
            + self.tire_model.peak_lateral_force(fr_fz, cam_fr) * mu_scale
            + self.tire_model.peak_lateral_force(rl_fz, cam_rl) * mu_scale
            + self.tire_model.peak_lateral_force(rr_fz, cam_rr) * mu_scale
        )
        return capacity >= required
```

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_cornering_solver.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/cornering_solver.py tests/test_cornering_solver.py
git commit -m "feat: steady-state cornering solver with bisection search"
```

---

### Task 14: Downforce, mu_scale, camber, edge case tests

**Files:**
- Modify: `tests/test_cornering_solver.py`

- [ ] **Step 1: Write and run new tests**

```python
# Append to tests/test_cornering_solver.py

class TestDownforceBenefit:

    @pytest.fixture
    def solver_with_downforce(self):
        mass, mu, g, rho, cl_a = 278.0, 1.3, 9.81, 1.225, 2.18
        tire = make_linear_tire(mu)
        lt = MagicMock()
        def loads(speed, lat_g, lon_g):
            total = mass * g + 0.5 * rho * cl_a * speed ** 2
            q = total / 4.0
            return (q, q, q, q)
        lt.tire_loads.side_effect = loads
        lt.roll_stiffness_front = 1e6
        lt.roll_stiffness_rear = 1e6
        return CorneringSolver(tire, lt, mass, 0, 0, 0, 0)

    def test_downforce_increases_speed(self, simple_solver, solver_with_downforce):
        v_no = simple_solver.max_cornering_speed(1.0/15.0)
        v_df = solver_with_downforce.max_cornering_speed(1.0/15.0)
        assert v_df > v_no

    def test_benefit_grows_at_higher_speed(self, simple_solver, solver_with_downforce):
        delta_tight = solver_with_downforce.max_cornering_speed(0.1) - simple_solver.max_cornering_speed(0.1)
        delta_wide = solver_with_downforce.max_cornering_speed(0.02) - simple_solver.max_cornering_speed(0.02)
        assert delta_wide > delta_tight


class TestMuScale:

    def test_below_one_reduces(self, simple_solver):
        v_nom = simple_solver.max_cornering_speed(0.05)
        v_red = simple_solver.max_cornering_speed(0.05, mu_scale=0.7)
        assert v_red < v_nom

    def test_above_one_increases(self, simple_solver):
        v_nom = simple_solver.max_cornering_speed(0.05)
        v_hi = simple_solver.max_cornering_speed(0.05, mu_scale=1.2)
        assert v_hi > v_nom

    def test_half_matches_analytical(self, simple_solver):
        v = simple_solver.max_cornering_speed(0.05, mu_scale=0.5)
        v_expected = math.sqrt(1.3 * 0.5 * 9.81 / 0.05)
        assert abs(v - v_expected) < 0.15


class TestEdgeCases:

    def test_very_tight_corner(self, simple_solver):
        assert 0.0 < simple_solver.max_cornering_speed(1.0) < 5.0

    def test_monotonically_decreasing(self, simple_solver):
        curvatures = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        speeds = [simple_solver.max_cornering_speed(k) for k in curvatures]
        for i in range(len(speeds) - 1):
            assert speeds[i] > speeds[i + 1]

    def test_always_positive_and_finite(self, simple_solver):
        for k in [0.001, 0.01, 0.1, 0.5, 1.0]:
            v = simple_solver.max_cornering_speed(k)
            assert v > 0 and math.isfinite(v)
```

- [ ] **Step 2: Run tests**
Run: `pytest tests/test_cornering_solver.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**
```bash
git add tests/test_cornering_solver.py
git commit -m "test: downforce, mu_scale, and edge case coverage for cornering solver"
```

---

### Task 15: Load-transfer sensitivity tests + package export

**Files:**
- Modify: `tests/test_cornering_solver.py`
- Modify: `src/fsae_sim/vehicle/__init__.py`

- [ ] **Step 1: Write and run tests for degressive tire interaction**

```python
# Append to tests/test_cornering_solver.py

class TestLoadTransferSensitivity:

    @staticmethod
    def _degressive_tire(mu=1.3, degression=3e-4):
        tire = MagicMock()
        tire.peak_lateral_force.side_effect = lambda fz, cam=0.0: max(0.0, mu * fz - degression * fz ** 2)
        return tire

    @staticmethod
    def _transferring_lt(mass, fraction=0.3):
        g, quarter = 9.81, mass * 9.81 / 4.0
        lt = MagicMock()
        def loads(speed, lat_g, lon_g):
            t = quarter * fraction * abs(lat_g)
            return (quarter + t, quarter - t, quarter + t, quarter - t)
        lt.tire_loads.side_effect = loads
        lt.roll_stiffness_front = 1e6
        lt.roll_stiffness_rear = 1e6
        return lt

    def test_degressive_tire_slower_with_transfer(self):
        mass, k = 278.0, 0.05
        s_static = CorneringSolver(self._degressive_tire(), make_static_load_transfer(mass), mass, 0, 0, 0, 0)
        s_transfer = CorneringSolver(self._degressive_tire(), self._transferring_lt(mass), mass, 0, 0, 0, 0)
        assert s_transfer.max_cornering_speed(k) < s_static.max_cornering_speed(k)

    def test_linear_tire_unaffected(self):
        mass, k = 278.0, 0.05
        s_static = CorneringSolver(make_linear_tire(1.3), make_static_load_transfer(mass), mass, 0, 0, 0, 0)
        s_transfer = CorneringSolver(make_linear_tire(1.3), self._transferring_lt(mass, 0.4), mass, 0, 0, 0, 0)
        assert abs(s_static.max_cornering_speed(k) - s_transfer.max_cornering_speed(k)) < 0.2
```

- [ ] **Step 2: Run tests**
Run: `pytest tests/test_cornering_solver.py -v`
Expected: All PASS

- [ ] **Step 3: Add CorneringSolver to __init__.py**

Add to `src/fsae_sim/vehicle/__init__.py`:
```python
from fsae_sim.vehicle.cornering_solver import CorneringSolver
```
Add `"CorneringSolver"` to `__all__`.

- [ ] **Step 4: Run tests**
Run: `pytest tests/test_cornering_solver.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**
```bash
git add tests/test_cornering_solver.py src/fsae_sim/vehicle/__init__.py
git commit -m "test: load-transfer sensitivity; export CorneringSolver from vehicle package"
```

---

## Section E: Integration (Tasks 16–19)

### Task 16: VehicleDynamics — optional constructor + cornering delegation

**Files:**
- Modify: `src/fsae_sim/vehicle/dynamics.py`
- Test: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_dynamics.py
from unittest.mock import MagicMock


class TestLegacyBackwardCompat:

    def test_legacy_constructor(self, ct16ev_params):
        dyn = VehicleDynamics(ct16ev_params)
        assert dyn.vehicle is ct16ev_params

    def test_legacy_cornering(self, ct16ev_params):
        dyn = VehicleDynamics(ct16ev_params)
        assert 13.0 < dyn.max_cornering_speed(1.0 / 15.0) < 20.0


class TestCorneringSolverDelegation:

    def test_delegates_to_solver(self, ct16ev_params):
        solver = MagicMock()
        solver.max_cornering_speed.return_value = 12.5
        dyn = VehicleDynamics(ct16ev_params, cornering_solver=solver)
        assert dyn.max_cornering_speed(0.05, grip_factor=0.9) == 12.5
        solver.max_cornering_speed.assert_called_once_with(0.05, mu_scale=0.9)

    def test_straight_skips_solver(self, ct16ev_params):
        solver = MagicMock()
        dyn = VehicleDynamics(ct16ev_params, cornering_solver=solver)
        assert dyn.max_cornering_speed(0.0) == float("inf")
        solver.max_cornering_speed.assert_not_called()

    def test_none_solver_uses_legacy(self, ct16ev_params):
        dyn = VehicleDynamics(ct16ev_params, cornering_solver=None)
        assert 13.0 < dyn.max_cornering_speed(1.0 / 15.0) < 20.0

    def test_stores_optional_components(self, ct16ev_params):
        tire, lt, solver = MagicMock(), MagicMock(), MagicMock()
        dyn = VehicleDynamics(ct16ev_params, tire_model=tire, load_transfer=lt, cornering_solver=solver)
        assert dyn.tire_model is tire
        assert dyn.load_transfer is lt
        assert dyn.cornering_solver is solver
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_dynamics.py::TestCorneringSolverDelegation -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'cornering_solver'`

- [ ] **Step 3: Write implementation**

Replace the import, class-level constant, `__init__`, and `max_cornering_speed` in `src/fsae_sim/vehicle/dynamics.py`:

```python
"""Vehicle dynamics force-balance model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fsae_sim.vehicle.vehicle import VehicleParams

if TYPE_CHECKING:
    from fsae_sim.vehicle.cornering_solver import CorneringSolver
    from fsae_sim.vehicle.load_transfer import LoadTransferModel
    from fsae_sim.vehicle.tire_model import PacejkaTireModel


class VehicleDynamics:
    """Longitudinal and lateral force-balance model.

    When constructed with optional tire_model, load_transfer, and
    cornering_solver, cornering speed and traction limits use physics-based
    tire models. Without them, the legacy constant-mu model is used.
    """

    AIR_DENSITY_KG_M3: float = 1.225
    GRAVITY_M_S2: float = 9.81
    _LEGACY_MAX_LATERAL_G: float = 1.3

    def __init__(
        self,
        vehicle: VehicleParams,
        tire_model: PacejkaTireModel | None = None,
        load_transfer: LoadTransferModel | None = None,
        cornering_solver: CorneringSolver | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.tire_model = tire_model
        self.load_transfer = load_transfer
        self.cornering_solver = cornering_solver

    # ... drag_force, downforce, rolling_resistance_force, grade_force,
    #     total_resistance remain UNCHANGED ...

    def max_cornering_speed(
        self, curvature: float, grip_factor: float = 1.0,
    ) -> float:
        """Maximum speed through a corner. Delegates to solver if available."""
        kappa = abs(curvature)
        if kappa < 1e-6:
            return float("inf")

        if self.cornering_solver is not None:
            return self.cornering_solver.max_cornering_speed(kappa, mu_scale=grip_factor)

        # Legacy fallback
        mu = self._LEGACY_MAX_LATERAL_G * grip_factor
        m = self.vehicle.mass_kg
        g = self.GRAVITY_M_S2
        cl_a = self.vehicle.downforce_coefficient

        if cl_a < 1e-6:
            return math.sqrt(mu * g / kappa)

        rho = self.AIR_DENSITY_KG_M3
        denom = m * kappa - 0.5 * rho * cl_a * mu
        if denom <= 0:
            return float("inf")
        return math.sqrt(m * g * mu / denom)

    # ... acceleration, resolve_exit_speed remain UNCHANGED ...
```

- [ ] **Step 4: Run ALL dynamics tests**
Run: `pytest tests/test_dynamics.py -v`
Expected: ALL PASS (existing + new)

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/dynamics.py tests/test_dynamics.py
git commit -m "feat(dynamics): accept optional tire components, delegate cornering speed"
```

---

### Task 17: max_traction_force + max_braking_force

**Files:**
- Modify: `src/fsae_sim/vehicle/dynamics.py`
- Test: `tests/test_dynamics.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_dynamics.py

class TestMaxTractionForce:

    def test_legacy_returns_inf(self, ct16ev_params):
        assert VehicleDynamics(ct16ev_params).max_traction_force(10.0) == float("inf")

    def test_with_models_returns_finite(self, ct16ev_params):
        tire = MagicMock()
        tire.peak_longitudinal_force.return_value = 1200.0
        lt = MagicMock()
        lt.tire_loads.return_value = (500.0, 500.0, 900.0, 900.0)
        dyn = VehicleDynamics(ct16ev_params, tire_model=tire, load_transfer=lt)
        result = dyn.max_traction_force(10.0)
        assert result == pytest.approx(2400.0)
        assert tire.peak_longitudinal_force.call_count == 2

    def test_uses_rear_loads(self, ct16ev_params):
        tire = MagicMock()
        tire.peak_longitudinal_force.side_effect = lambda n: n * 1.0
        lt = MagicMock()
        lt.tire_loads.return_value = (400.0, 400.0, 700.0, 800.0)
        dyn = VehicleDynamics(ct16ev_params, tire_model=tire, load_transfer=lt)
        assert dyn.max_traction_force(15.0) == pytest.approx(1500.0)


class TestMaxBrakingForce:

    def test_legacy_returns_inf(self, ct16ev_params):
        assert VehicleDynamics(ct16ev_params).max_braking_force(10.0) == float("inf")

    def test_with_models_uses_all_four(self, ct16ev_params):
        tire = MagicMock()
        tire.peak_longitudinal_force.side_effect = lambda n: n * 1.2
        lt = MagicMock()
        lt.tire_loads.return_value = (800.0, 750.0, 500.0, 550.0)
        dyn = VehicleDynamics(ct16ev_params, tire_model=tire, load_transfer=lt)
        # (800+750+500+550) * 1.2 = 3120
        assert dyn.max_braking_force(20.0) == pytest.approx(3120.0)
        assert tire.peak_longitudinal_force.call_count == 4

    def test_requires_both_models(self, ct16ev_params):
        assert VehicleDynamics(ct16ev_params, tire_model=MagicMock()).max_braking_force(10.0) == float("inf")
        assert VehicleDynamics(ct16ev_params, load_transfer=MagicMock()).max_braking_force(10.0) == float("inf")
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_dynamics.py::TestMaxTractionForce tests/test_dynamics.py::TestMaxBrakingForce -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write implementation**

Add to `VehicleDynamics` after `max_cornering_speed`, before `acceleration`:
```python
    def max_traction_force(self, speed_ms: float) -> float:
        """Max drive force from rear tires (N). Inf in legacy mode."""
        if self.tire_model is None or self.load_transfer is None:
            return float("inf")
        _, _, rl, rr = self.load_transfer.tire_loads(speed_ms, 0.0, 0.3)
        return self.tire_model.peak_longitudinal_force(rl) + self.tire_model.peak_longitudinal_force(rr)

    def max_braking_force(self, speed_ms: float) -> float:
        """Max braking force from all four tires (N). Inf in legacy mode."""
        if self.tire_model is None or self.load_transfer is None:
            return float("inf")
        fl, fr, rl, rr = self.load_transfer.tire_loads(speed_ms, 0.0, -1.0)
        return (
            self.tire_model.peak_longitudinal_force(fl)
            + self.tire_model.peak_longitudinal_force(fr)
            + self.tire_model.peak_longitudinal_force(rl)
            + self.tire_model.peak_longitudinal_force(rr)
        )
```

- [ ] **Step 4: Run ALL dynamics tests**
Run: `pytest tests/test_dynamics.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/vehicle/dynamics.py tests/test_dynamics.py
git commit -m "feat(dynamics): add max_traction_force and max_braking_force"
```

---

### Task 18: SimulationEngine wiring + traction/braking clamping

**Files:**
- Modify: `src/fsae_sim/sim/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**
```python
# Append to tests/test_engine.py
import math
from unittest.mock import MagicMock, patch


class TestTireModelIntegration:

    @patch("fsae_sim.sim.engine.CorneringSolver")
    @patch("fsae_sim.sim.engine.LoadTransferModel")
    @patch("fsae_sim.sim.engine.PacejkaTireModel")
    def test_constructs_components_from_config(
        self, mock_tire_cls, mock_lt_cls, mock_solver_cls,
        vehicle_config, battery_model, simple_track,
    ):
        """Config with tire+suspension triggers component construction."""
        engine = SimulationEngine(vehicle_config, simple_track, FullThrottleStrategy(), battery_model)
        mock_tire_cls.assert_called_once()
        mock_lt_cls.assert_called_once()
        mock_solver_cls.assert_called_once()
        assert engine.dynamics.cornering_solver is not None

    def test_legacy_mode_without_tire_config(self, vehicle_config, battery_model, simple_track):
        """Force legacy by mocking away the tire config."""
        mock_cfg = MagicMock(wraps=vehicle_config)
        mock_cfg.tire = None
        mock_cfg.suspension = None
        mock_cfg.vehicle = vehicle_config.vehicle
        mock_cfg.powertrain = vehicle_config.powertrain
        mock_cfg.battery = vehicle_config.battery
        mock_cfg.name = vehicle_config.name
        engine = SimulationEngine(mock_cfg, simple_track, FullThrottleStrategy(), battery_model)
        assert engine.dynamics.cornering_solver is None


class TestTractionClamping:

    def test_drive_force_clamped(self, vehicle_config, battery_model, simple_track):
        engine = SimulationEngine(vehicle_config, simple_track, FullThrottleStrategy(), battery_model)
        engine.dynamics.max_traction_force = lambda speed: 200.0
        result = engine.run(num_laps=1)
        throttle_mask = result.states["action"] == "throttle"
        if throttle_mask.any():
            assert result.states.loc[throttle_mask, "drive_force_n"].max() <= 200.01
```

- [ ] **Step 2: Run tests to verify they fail**
Run: `pytest tests/test_engine.py::TestTireModelIntegration tests/test_engine.py::TestTractionClamping -v`
Expected: FAIL (engine doesn't import tire components yet)

- [ ] **Step 3: Write implementation**

Update `src/fsae_sim/sim/engine.py` imports:
```python
import math

# ... existing imports ...

try:
    from fsae_sim.vehicle.tire_model import PacejkaTireModel
    from fsae_sim.vehicle.load_transfer import LoadTransferModel
    from fsae_sim.vehicle.cornering_solver import CorneringSolver
    _HAS_TIRE_MODELS = True
except ImportError:
    _HAS_TIRE_MODELS = False
```

Replace `SimulationEngine.__init__`:
```python
    def __init__(self, vehicle, track, strategy, battery_model):
        self.vehicle = vehicle
        self.track = track
        self.strategy = strategy
        self.battery_model = battery_model
        self.powertrain = PowertrainModel(vehicle.powertrain)

        tire_cfg = getattr(vehicle, "tire", None)
        susp_cfg = getattr(vehicle, "suspension", None)

        if _HAS_TIRE_MODELS and tire_cfg is not None and susp_cfg is not None:
            tire_model = PacejkaTireModel(tire_cfg.tir_file)
            load_transfer = LoadTransferModel(vehicle.vehicle, susp_cfg)
            cornering_solver = CorneringSolver(
                tire_model, load_transfer, vehicle.vehicle.mass_kg,
                math.radians(tire_cfg.static_camber_front_deg),
                math.radians(tire_cfg.static_camber_rear_deg),
                susp_cfg.roll_camber_front_deg_per_deg,
                susp_cfg.roll_camber_rear_deg_per_deg,
            )
            self.dynamics = VehicleDynamics(vehicle.vehicle, tire_model, load_transfer, cornering_solver)
        else:
            self.dynamics = VehicleDynamics(vehicle.vehicle)
```

In the `run` method force-based block, replace the force computation:
```python
                    # 3. Compute forces based on driver action
                    if cmd.action == ControlAction.THROTTLE:
                        drive_f = self.powertrain.drive_force(cmd.throttle_pct, speed)
                        drive_f = min(drive_f, self.dynamics.max_traction_force(speed))
                        regen_f = 0.0
                    elif cmd.action == ControlAction.BRAKE:
                        drive_f = 0.0
                        regen_f = self.powertrain.regen_force(cmd.brake_pct, speed)
                        max_brake = self.dynamics.max_braking_force(speed)
                        if abs(regen_f) > max_brake:
                            regen_f = -max_brake
                    else:  # COAST
                        drive_f = 0.0
                        regen_f = 0.0
```

- [ ] **Step 4: Run all engine tests**
Run: `pytest tests/test_engine.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**
```bash
git add src/fsae_sim/sim/engine.py tests/test_engine.py
git commit -m "feat(engine): wire tire model components, clamp drive/regen to traction limits"
```

---

### Task 19: Full regression — verify all existing tests pass

**Files:** None modified — verification only.

- [ ] **Step 1: Run the complete test suite**
Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS — every existing test across all modules.

- [ ] **Step 2: Verify dynamics backward compatibility**
Run: `pytest tests/test_dynamics.py::TestCorneringSpeed tests/test_dynamics.py::TestDragForce tests/test_dynamics.py::TestResolveExitSpeed -v`
Expected: ALL PASS unchanged (legacy `VehicleDynamics(params)` constructor still works).

- [ ] **Step 3: Verify engine backward compatibility**
Run: `pytest tests/test_engine.py::TestEngineBasics tests/test_engine.py::TestPhysicalSanity -v`
Expected: ALL PASS. With tire model enabled, physics changes slightly but range-based assertions should hold.

- [ ] **Step 4: Commit only if fixups were needed**
```bash
git add -A && git commit -m "fix: address any regression issues from tier 3 integration"
```

---

## Not In This Plan (Future Phases)

Per the spec, these are post-integration follow-ups:

- **Telemetry validation** (Phase 2): Compare sim predictions against AiM Michigan 2025 data — corner speeds, Fx validation, full 22-lap endurance.
- **Parameter sweep confidence** (Phase 3): Inverter torque 60–120 Nm sweep, verify no discontinuities.
- **MMM diagrams**: Milliken Moment Method for vehicle setup/balance — deferred to a separate spec.
