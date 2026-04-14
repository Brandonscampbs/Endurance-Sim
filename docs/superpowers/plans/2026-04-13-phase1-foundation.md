# Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the complete repository scaffold, Docker dev environment, working Dash dashboard on port 3000, vehicle config system, data loaders, and module interface stubs for the FSAE EV endurance simulation.

**Architecture:** Python package (`fsae_sim`) with modular domain subpackages (vehicle, track, driver, sim, scoring, optimization, analysis). Dash web app in separate `dashboard/` directory reads from `results/`. Docker single-container setup with live code reload.

**Tech Stack:** Python 3.12, NumPy, SciPy, pandas, PyYAML, Dash, Plotly, dash-bootstrap-components, pyarrow, pytest, Docker

---

### Task 1: Repository scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/fsae_sim/__init__.py`
- Create: `src/fsae_sim/vehicle/__init__.py`
- Create: `src/fsae_sim/track/__init__.py`
- Create: `src/fsae_sim/driver/__init__.py`
- Create: `src/fsae_sim/sim/__init__.py`
- Create: `src/fsae_sim/optimization/__init__.py`
- Create: `src/fsae_sim/scoring/__init__.py`
- Create: `src/fsae_sim/data/__init__.py`
- Create: `src/fsae_sim/analysis/__init__.py`
- Create: `tests/__init__.py`
- Create: `dashboard/__init__.py`
- Create: `dashboard/pages/__init__.py`
- Create: `configs/` (directory)
- Create: `results/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/fsae_sim/vehicle src/fsae_sim/track src/fsae_sim/driver \
         src/fsae_sim/sim src/fsae_sim/optimization src/fsae_sim/scoring \
         src/fsae_sim/data src/fsae_sim/analysis \
         dashboard/pages configs results tests
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "fsae-sim"
version = "0.1.0"
description = "FSAE EV endurance simulation and optimization for UConn Formula SAE Electric"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.2",
    "pyyaml>=6.0",
    "dash>=2.16",
    "plotly>=5.19",
    "dash-bootstrap-components>=1.6",
    "pyarrow>=15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
results/*
!results/.gitkeep
*.parquet
.pytest_cache/
.venv/
.env
.coverage
htmlcov/
```

- [ ] **Step 4: Create all __init__.py files and .gitkeep**

All `__init__.py` files are empty. `results/.gitkeep` is empty.

```
src/fsae_sim/__init__.py
src/fsae_sim/vehicle/__init__.py
src/fsae_sim/track/__init__.py
src/fsae_sim/driver/__init__.py
src/fsae_sim/sim/__init__.py
src/fsae_sim/optimization/__init__.py
src/fsae_sim/scoring/__init__.py
src/fsae_sim/data/__init__.py
src/fsae_sim/analysis/__init__.py
tests/__init__.py
dashboard/__init__.py
dashboard/pages/__init__.py
results/.gitkeep
```

- [ ] **Step 5: Install package in editable mode and verify**

Run: `pip install -e ".[dev]"`
Expected: successful install, `import fsae_sim` works

```bash
pip install -e ".[dev]" && python -c "import fsae_sim; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/__init__.py dashboard/__init__.py \
        dashboard/pages/__init__.py configs/ results/.gitkeep
git commit -m "feat: initialize project scaffold and package structure"
```

---

### Task 2: Vehicle configuration YAML files

**Files:**
- Create: `configs/ct16ev.yaml`
- Create: `configs/ct17ev.yaml`

- [ ] **Step 1: Create CT-16EV config**

Create `configs/ct16ev.yaml`:

```yaml
name: CT-16EV
year: 2025
description: "2025 competition car - Michigan Endurance"

vehicle:
  mass_kg: 270.0
  frontal_area_m2: 1.2
  drag_coefficient: 0.5
  rolling_resistance: 0.015
  wheelbase_m: 1.53

powertrain:
  motor_speed_max_rpm: 2900
  brake_speed_rpm: 2400
  torque_limit_inverter_nm: 85.0
  torque_limit_lvcu_nm: 150.0
  iq_limit_a: 170.0
  id_limit_a: 30.0
  gear_ratio: 3.5
  drivetrain_efficiency: 0.92

battery:
  cell_type: "P45B"
  topology:
    series: 110
    parallel: 4
  cell_voltage_min_v: 2.55
  cell_voltage_max_v: 4.195
  discharged_soc_pct: 2.0
  soc_taper:
    threshold_pct: 85.0
    rate_a_per_pct: 1.0
  discharge_limits:
    - temp_c: 30.0
      max_current_a: 100.0
    - temp_c: 35.0
      max_current_a: 85.0
    - temp_c: 40.0
      max_current_a: 65.0
    - temp_c: 45.0
      max_current_a: 55.0
    - temp_c: 50.0
      max_current_a: 45.0
    - temp_c: 55.0
      max_current_a: 40.0
    - temp_c: 60.0
      max_current_a: 35.0
    - temp_c: 65.0
      max_current_a: 0.0
```

- [ ] **Step 2: Create CT-17EV config**

Create `configs/ct17ev.yaml`:

```yaml
name: CT-17EV
year: 2026
description: "2026 competition car - 100S4P P50B, 20 lbs lighter"

vehicle:
  mass_kg: 261.0
  frontal_area_m2: 1.2
  drag_coefficient: 0.5
  rolling_resistance: 0.015
  wheelbase_m: 1.53

powertrain:
  motor_speed_max_rpm: 2900
  brake_speed_rpm: 2400
  torque_limit_inverter_nm: 85.0
  torque_limit_lvcu_nm: 150.0
  iq_limit_a: 170.0
  id_limit_a: 30.0
  gear_ratio: 3.5
  drivetrain_efficiency: 0.92

battery:
  cell_type: "P50B"
  topology:
    series: 100
    parallel: 4
  cell_voltage_min_v: 2.50
  cell_voltage_max_v: 4.20
  discharged_soc_pct: 2.0
  soc_taper:
    threshold_pct: 85.0
    rate_a_per_pct: 1.0
  discharge_limits:
    - temp_c: 30.0
      max_current_a: 100.0
    - temp_c: 35.0
      max_current_a: 85.0
    - temp_c: 40.0
      max_current_a: 65.0
    - temp_c: 45.0
      max_current_a: 55.0
    - temp_c: 50.0
      max_current_a: 45.0
    - temp_c: 55.0
      max_current_a: 40.0
    - temp_c: 60.0
      max_current_a: 35.0
    - temp_c: 65.0
      max_current_a: 0.0
```

Note: P50B cell voltage bounds and discharge limits are copied from CT-16EV as placeholders. These must be updated with actual P50B datasheet values before running CT-17EV simulations.

- [ ] **Step 3: Commit**

```bash
git add configs/ct16ev.yaml configs/ct17ev.yaml
git commit -m "feat: add vehicle configuration files for CT-16EV and CT-17EV"
```

---

### Task 3: Test infrastructure

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create conftest.py with shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for FSAE simulation tests."""

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "Real-Car-Data-And-Stats"
AIM_CSV = DATA_DIR / "2025 Endurance Data.csv"
VOLTT_CELL_CSV = DATA_DIR / "About-Energy-Volt-Simulations" / "2025_Pack_cell.csv"
VOLTT_PACK_CSV = DATA_DIR / "About-Energy-Volt-Simulations" / "2025_Pack_pack.csv"

# Config paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
CT16EV_CONFIG = CONFIGS_DIR / "ct16ev.yaml"
CT17EV_CONFIG = CONFIGS_DIR / "ct17ev.yaml"

requires_data = pytest.mark.skipif(
    not AIM_CSV.exists(),
    reason="Telemetry data files not available",
)


@pytest.fixture
def aim_csv_path():
    """Path to AiM endurance telemetry CSV."""
    if not AIM_CSV.exists():
        pytest.skip("AiM telemetry CSV not available")
    return AIM_CSV


@pytest.fixture
def voltt_cell_path():
    """Path to Voltt cell-level simulation CSV."""
    if not VOLTT_CELL_CSV.exists():
        pytest.skip("Voltt cell CSV not available")
    return VOLTT_CELL_CSV


@pytest.fixture
def voltt_pack_path():
    """Path to Voltt pack-level simulation CSV."""
    if not VOLTT_PACK_CSV.exists():
        pytest.skip("Voltt pack CSV not available")
    return VOLTT_PACK_CSV


@pytest.fixture
def ct16ev_config_path():
    """Path to CT-16EV vehicle config."""
    return CT16EV_CONFIG


@pytest.fixture
def ct17ev_config_path():
    """Path to CT-17EV vehicle config."""
    return CT17EV_CONFIG
```

- [ ] **Step 2: Verify pytest discovers fixtures**

Run: `pytest --co -q`
Expected: `no tests ran` (no test files yet, but no errors)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add test infrastructure with shared fixtures"
```

---

### Task 4: Vehicle configuration system (TDD)

**Files:**
- Create: `tests/test_vehicle.py`
- Create: `src/fsae_sim/vehicle/vehicle.py`
- Create: `src/fsae_sim/vehicle/powertrain.py`
- Create: `src/fsae_sim/vehicle/battery.py`
- Modify: `src/fsae_sim/vehicle/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_vehicle.py`:

```python
"""Tests for vehicle configuration loading."""

from fsae_sim.vehicle import VehicleConfig


class TestVehicleConfigLoading:
    """Test YAML config loading into dataclasses."""

    def test_load_ct16ev(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.name == "CT-16EV"
        assert config.year == 2025

    def test_vehicle_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.vehicle.mass_kg == 270.0
        assert config.vehicle.drag_coefficient == 0.5
        assert config.vehicle.rolling_resistance == 0.015

    def test_powertrain_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.powertrain.motor_speed_max_rpm == 2900
        assert config.powertrain.torque_limit_inverter_nm == 85.0
        assert config.powertrain.iq_limit_a == 170.0
        assert config.powertrain.gear_ratio == 3.5

    def test_battery_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.battery.cell_type == "P45B"
        assert config.battery.series == 110
        assert config.battery.parallel == 4
        assert config.battery.cell_voltage_min_v == 2.55
        assert config.battery.cell_voltage_max_v == 4.195

    def test_discharge_limits(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        limits = config.battery.discharge_limits
        assert len(limits) == 8
        assert limits[0].temp_c == 30.0
        assert limits[0].max_current_a == 100.0
        assert limits[-1].temp_c == 65.0
        assert limits[-1].max_current_a == 0.0

    def test_soc_taper(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.battery.soc_taper_threshold_pct == 85.0
        assert config.battery.soc_taper_rate_a_per_pct == 1.0

    def test_load_ct17ev(self, ct17ev_config_path):
        config = VehicleConfig.from_yaml(ct17ev_config_path)
        assert config.name == "CT-17EV"
        assert config.year == 2026
        assert config.vehicle.mass_kg == 261.0
        assert config.battery.cell_type == "P50B"
        assert config.battery.series == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vehicle.py -v`
Expected: FAIL — `ImportError: cannot import name 'VehicleConfig'`

- [ ] **Step 3: Implement battery config**

Create `src/fsae_sim/vehicle/battery.py`:

```python
"""Battery pack configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DischargeLimitPoint:
    """Temperature-dependent discharge current limit."""
    temp_c: float
    max_current_a: float


@dataclass(frozen=True)
class BatteryConfig:
    """Battery pack configuration parameters."""
    cell_type: str
    series: int
    parallel: int
    cell_voltage_min_v: float
    cell_voltage_max_v: float
    discharged_soc_pct: float
    soc_taper_threshold_pct: float
    soc_taper_rate_a_per_pct: float
    discharge_limits: tuple[DischargeLimitPoint, ...]

    @property
    def pack_voltage_min_v(self) -> float:
        return self.cell_voltage_min_v * self.series

    @property
    def pack_voltage_max_v(self) -> float:
        return self.cell_voltage_max_v * self.series

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryConfig":
        """Build from parsed YAML dict."""
        return cls(
            cell_type=data["cell_type"],
            series=data["topology"]["series"],
            parallel=data["topology"]["parallel"],
            cell_voltage_min_v=data["cell_voltage_min_v"],
            cell_voltage_max_v=data["cell_voltage_max_v"],
            discharged_soc_pct=data["discharged_soc_pct"],
            soc_taper_threshold_pct=data["soc_taper"]["threshold_pct"],
            soc_taper_rate_a_per_pct=data["soc_taper"]["rate_a_per_pct"],
            discharge_limits=tuple(
                DischargeLimitPoint(**dl) for dl in data["discharge_limits"]
            ),
        )
```

- [ ] **Step 4: Implement powertrain config**

Create `src/fsae_sim/vehicle/powertrain.py`:

```python
"""Powertrain configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PowertrainConfig:
    """Motor, inverter, and drivetrain parameters."""
    motor_speed_max_rpm: float
    brake_speed_rpm: float
    torque_limit_inverter_nm: float
    torque_limit_lvcu_nm: float
    iq_limit_a: float
    id_limit_a: float
    gear_ratio: float
    drivetrain_efficiency: float
```

- [ ] **Step 5: Implement vehicle config with YAML loading**

Create `src/fsae_sim/vehicle/vehicle.py`:

```python
"""Top-level vehicle configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from fsae_sim.vehicle.battery import BatteryConfig
from fsae_sim.vehicle.powertrain import PowertrainConfig


@dataclass(frozen=True)
class VehicleParams:
    """Physical vehicle parameters."""
    mass_kg: float
    frontal_area_m2: float
    drag_coefficient: float
    rolling_resistance: float
    wheelbase_m: float


@dataclass(frozen=True)
class VehicleConfig:
    """Complete vehicle configuration loaded from YAML."""
    name: str
    year: int
    description: str
    vehicle: VehicleParams
    powertrain: PowertrainConfig
    battery: BatteryConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleConfig":
        """Load vehicle configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            name=data["name"],
            year=data["year"],
            description=data["description"],
            vehicle=VehicleParams(**data["vehicle"]),
            powertrain=PowertrainConfig(**data["powertrain"]),
            battery=BatteryConfig.from_dict(data["battery"]),
        )
```

- [ ] **Step 6: Update vehicle __init__.py**

Write `src/fsae_sim/vehicle/__init__.py`:

```python
from fsae_sim.vehicle.vehicle import VehicleConfig, VehicleParams
from fsae_sim.vehicle.powertrain import PowertrainConfig
from fsae_sim.vehicle.battery import BatteryConfig, DischargeLimitPoint

__all__ = [
    "VehicleConfig",
    "VehicleParams",
    "PowertrainConfig",
    "BatteryConfig",
    "DischargeLimitPoint",
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_vehicle.py -v`
Expected: all 8 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/fsae_sim/vehicle/ tests/test_vehicle.py
git commit -m "feat: vehicle configuration system with YAML loading"
```

---

### Task 5: Data loaders (TDD)

**Files:**
- Create: `tests/test_loader.py`
- Create: `src/fsae_sim/data/loader.py`
- Modify: `src/fsae_sim/data/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loader.py`:

```python
"""Tests for telemetry and simulation data loaders."""

import pandas as pd
from tests.conftest import requires_data
from fsae_sim.data import load_aim_csv, load_voltt_csv


@requires_data
class TestAiMLoader:
    """Test AiM Race Studio CSV loader."""

    def test_returns_metadata_and_dataframe(self, aim_csv_path):
        metadata, df = load_aim_csv(aim_csv_path)
        assert isinstance(metadata, dict)
        assert isinstance(df, pd.DataFrame)

    def test_metadata_fields(self, aim_csv_path):
        metadata, _ = load_aim_csv(aim_csv_path)
        assert metadata["Vehicle"] == "CT-16EV"
        assert metadata["Format"] == "AiM CSV File"
        assert metadata["Sample Rate"] == "20"

    def test_dataframe_has_key_columns(self, aim_csv_path):
        _, df = load_aim_csv(aim_csv_path)
        expected_cols = [
            "Time", "GPS Speed", "RPM", "Pack Voltage",
            "Pack Current", "State of Charge", "Throttle Pos",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataframe_is_numeric(self, aim_csv_path):
        _, df = load_aim_csv(aim_csv_path)
        assert df["Time"].dtype in ("float64", "float32")
        assert df["GPS Speed"].dtype in ("float64", "float32")

    def test_dataframe_shape(self, aim_csv_path):
        _, df = load_aim_csv(aim_csv_path)
        assert len(df) > 30000  # ~37k rows at 20Hz over 31 min
        assert len(df.columns) > 90  # ~100+ channels

    def test_units_stored(self, aim_csv_path):
        _, df = load_aim_csv(aim_csv_path)
        assert "units" in df.attrs
        assert df.attrs["units"]["GPS Speed"] == "km/h"
        assert df.attrs["units"]["Time"] == "s"


@requires_data
class TestVolttLoader:
    """Test Voltt battery simulation CSV loader."""

    def test_loads_cell_csv(self, voltt_cell_path):
        df = load_voltt_csv(voltt_cell_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 10000

    def test_loads_pack_csv(self, voltt_pack_path):
        df = load_voltt_csv(voltt_pack_path)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, voltt_cell_path):
        df = load_voltt_csv(voltt_cell_path)
        expected = ["Time [s]", "Voltage [V]", "SOC [%]", "Current [A]", "Temperature [°C]"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_numeric_data(self, voltt_cell_path):
        df = load_voltt_csv(voltt_cell_path)
        assert df["Time [s]"].dtype in ("float64", "float32")
        assert df["SOC [%]"].iloc[0] == 100.0  # starts at full charge
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_aim_csv'`

- [ ] **Step 3: Implement data loaders**

Create `src/fsae_sim/data/loader.py`:

```python
"""Data loaders for AiM telemetry and Voltt battery simulation exports."""

import csv
import io
from pathlib import Path

import pandas as pd


def load_aim_csv(path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Load an AiM Race Studio CSV export.

    AiM CSV format:
    - Metadata lines: "key","value" pairs until first blank line
    - Column headers line
    - Units line
    - Blank line
    - Data lines (quoted numeric values)

    Returns:
        metadata: dict of session metadata (Vehicle, Date, Sample Rate, etc.)
        df: DataFrame with numeric columns. Units stored in df.attrs['units'].
    """
    path = Path(path)
    metadata: dict[str, str] = {}

    with open(path, "r") as f:
        lines = f.readlines()

    idx = 0

    # Parse metadata (until first blank line)
    while idx < len(lines) and lines[idx].strip():
        row = next(csv.reader(io.StringIO(lines[idx].strip())))
        metadata[row[0]] = row[1] if len(row) > 1 else ""
        idx += 1

    # Skip blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Column headers
    headers = next(csv.reader(io.StringIO(lines[idx].strip())))
    idx += 1

    # Units
    unit_list = next(csv.reader(io.StringIO(lines[idx].strip())))
    units = dict(zip(headers, unit_list))
    idx += 1

    # Skip blank lines before data
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Parse data
    data_text = "".join(lines[idx:])
    df = pd.read_csv(io.StringIO(data_text), header=None, names=headers)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.attrs["units"] = units
    df.attrs["metadata"] = metadata

    return metadata, df


def load_voltt_csv(path: str | Path) -> pd.DataFrame:
    """Load a Voltt battery simulation CSV export.

    Voltt CSV format:
    - Comment lines starting with #
    - Standard CSV with headers

    Returns:
        df: DataFrame with numeric columns.
    """
    path = Path(path)
    df = pd.read_csv(path, comment="#")
    return df
```

- [ ] **Step 4: Update data __init__.py**

Write `src/fsae_sim/data/__init__.py`:

```python
from fsae_sim.data.loader import load_aim_csv, load_voltt_csv

__all__ = ["load_aim_csv", "load_voltt_csv"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: all 10 tests PASS (or SKIP if data files not present)

- [ ] **Step 6: Commit**

```bash
git add src/fsae_sim/data/ tests/test_loader.py
git commit -m "feat: data loaders for AiM telemetry and Voltt battery CSVs"
```

---

### Task 6: Module interface stubs

**Files:**
- Create: `src/fsae_sim/track/track.py`
- Create: `src/fsae_sim/driver/strategy.py`
- Create: `src/fsae_sim/sim/engine.py`
- Create: `src/fsae_sim/scoring/scoring.py`
- Create: `src/fsae_sim/optimization/sweep.py`
- Create: `src/fsae_sim/analysis/metrics.py`
- Modify: `src/fsae_sim/track/__init__.py`
- Modify: `src/fsae_sim/driver/__init__.py`
- Modify: `src/fsae_sim/sim/__init__.py`
- Modify: `src/fsae_sim/scoring/__init__.py`
- Modify: `src/fsae_sim/optimization/__init__.py`
- Modify: `src/fsae_sim/analysis/__init__.py`

These are interface definitions only — no implementation logic. They define the contracts between modules so that Phase 2 work can be done independently per module.

- [ ] **Step 1: Create track module**

Create `src/fsae_sim/track/track.py`:

```python
"""Track representation as an ordered sequence of segments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A discrete track segment with geometric properties."""
    index: int
    distance_start_m: float
    length_m: float
    curvature: float  # 1/radius in 1/m, 0 for straight, signed for direction
    grade: float  # rise/run, positive = uphill
    grip_factor: float = 1.0  # multiplier on baseline grip, 1.0 = nominal


@dataclass
class Track:
    """Ordered sequence of segments representing a circuit."""
    name: str
    segments: list[Segment]

    @property
    def total_distance_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @classmethod
    def from_telemetry(cls, aim_csv_path: str) -> "Track":
        """Extract track segments from AiM GPS telemetry.

        Implemented in Phase 2.
        """
        raise NotImplementedError("Track extraction from telemetry not yet implemented")
```

Write `src/fsae_sim/track/__init__.py`:

```python
from fsae_sim.track.track import Track, Segment

__all__ = ["Track", "Segment"]
```

- [ ] **Step 2: Create driver/strategy module**

Create `src/fsae_sim/driver/strategy.py`:

```python
"""Driver strategy and simulation state definitions."""

from dataclasses import dataclass
from enum import Enum

from fsae_sim.track.track import Segment


class ControlAction(Enum):
    """Discrete driver action."""
    THROTTLE = "throttle"
    COAST = "coast"
    BRAKE = "brake"


@dataclass(frozen=True)
class ControlCommand:
    """Output of a driver strategy decision."""
    action: ControlAction
    throttle_pct: float = 0.0  # 0 to 1
    brake_pct: float = 0.0  # 0 to 1


@dataclass
class SimState:
    """Instantaneous simulation state passed through the time-step loop."""
    time: float  # seconds
    distance: float  # meters along track
    speed: float  # m/s
    soc: float  # 0 to 1
    pack_voltage: float  # V
    pack_current: float  # A
    cell_temp: float  # degrees C
    lap: int
    segment_idx: int


class DriverStrategy:
    """Base class for driver control strategies. Subclass to implement."""
    name: str = "base"

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        """Given current state and upcoming track, choose an action.

        Implemented in Phase 2/3 by subclasses.
        """
        raise NotImplementedError
```

Write `src/fsae_sim/driver/__init__.py`:

```python
from fsae_sim.driver.strategy import (
    ControlAction,
    ControlCommand,
    DriverStrategy,
    SimState,
)

__all__ = ["ControlAction", "ControlCommand", "DriverStrategy", "SimState"]
```

- [ ] **Step 3: Create simulation engine stub**

Create `src/fsae_sim/sim/engine.py`:

```python
"""Quasi-static endurance simulation engine."""

from dataclasses import dataclass

import pandas as pd

from fsae_sim.driver.strategy import DriverStrategy
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig


@dataclass
class SimResult:
    """Output of a single simulation run."""
    config_name: str
    strategy_name: str
    track_name: str
    states: pd.DataFrame  # time series of SimState fields
    total_time_s: float
    total_energy_kwh: float
    final_soc: float
    laps_completed: int


class SimulationEngine:
    """Quasi-static endurance simulation.

    For each track segment, resolves speed from force balance and
    driver strategy, steps battery state, and records results.
    """

    def __init__(
        self,
        vehicle: VehicleConfig,
        track: Track,
        strategy: DriverStrategy,
    ):
        self.vehicle = vehicle
        self.track = track
        self.strategy = strategy

    def run(self, num_laps: int = 1) -> SimResult:
        """Run the endurance simulation.

        Implemented in Phase 2.
        """
        raise NotImplementedError("Simulation engine not yet implemented")
```

Write `src/fsae_sim/sim/__init__.py`:

```python
from fsae_sim.sim.engine import SimulationEngine, SimResult

__all__ = ["SimulationEngine", "SimResult"]
```

- [ ] **Step 4: Create scoring stub**

Create `src/fsae_sim/scoring/scoring.py`:

```python
"""FSAE endurance and efficiency scoring formulas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnduranceScore:
    """Combined endurance + efficiency score."""
    endurance_points: float
    efficiency_points: float
    total_points: float


def calculate_endurance_points(
    team_time_s: float,
    fastest_time_s: float,
    max_points: float = 300.0,
) -> float:
    """Calculate FSAE endurance event points.

    Implemented in Phase 4.
    """
    raise NotImplementedError


def calculate_efficiency_points(
    team_energy_kwh: float,
    team_time_s: float,
    min_energy_kwh: float,
    fastest_time_s: float,
    max_points: float = 100.0,
) -> float:
    """Calculate FSAE efficiency event points.

    Implemented in Phase 4.
    """
    raise NotImplementedError
```

Write `src/fsae_sim/scoring/__init__.py`:

```python
from fsae_sim.scoring.scoring import (
    EnduranceScore,
    calculate_endurance_points,
    calculate_efficiency_points,
)

__all__ = [
    "EnduranceScore",
    "calculate_endurance_points",
    "calculate_efficiency_points",
]
```

- [ ] **Step 5: Create analysis/metrics stub**

Create `src/fsae_sim/analysis/metrics.py`:

```python
"""Post-processing metrics computed from simulation results."""

import pandas as pd


def compute_lap_times(states: pd.DataFrame) -> list[float]:
    """Extract per-lap times from simulation state time series.

    Implemented in Phase 2.
    """
    raise NotImplementedError


def compute_energy_per_lap(states: pd.DataFrame) -> list[float]:
    """Compute energy consumed per lap in kWh.

    Implemented in Phase 2.
    """
    raise NotImplementedError


def compute_pareto_frontier(
    results: pd.DataFrame,
    time_col: str = "total_time_s",
    energy_col: str = "total_energy_kwh",
) -> pd.DataFrame:
    """Find Pareto-optimal points minimizing both time and energy.

    Implemented in Phase 3.
    """
    raise NotImplementedError
```

Write `src/fsae_sim/analysis/__init__.py`:

```python
from fsae_sim.analysis.metrics import (
    compute_lap_times,
    compute_energy_per_lap,
    compute_pareto_frontier,
)

__all__ = ["compute_lap_times", "compute_energy_per_lap", "compute_pareto_frontier"]
```

- [ ] **Step 6: Create optimization/sweep stub**

Create `src/fsae_sim/optimization/sweep.py`:

```python
"""Parameter sweep runner for exploring configuration spaces."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SweepConfig:
    """Definition of a parameter sweep."""
    parameter_name: str
    values: list[float]
    base_config_path: str
    output_dir: str
    description: str = ""


def run_sweep(config: SweepConfig) -> Path:
    """Run a parameter sweep and store results.

    Implemented in Phase 3.
    """
    raise NotImplementedError
```

Write `src/fsae_sim/optimization/__init__.py`:

```python
from fsae_sim.optimization.sweep import SweepConfig, run_sweep

__all__ = ["SweepConfig", "run_sweep"]
```

- [ ] **Step 7: Verify all imports work**

Run:

```bash
python -c "
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.track import Track, Segment
from fsae_sim.driver import DriverStrategy, SimState, ControlCommand
from fsae_sim.sim import SimulationEngine, SimResult
from fsae_sim.scoring import EnduranceScore
from fsae_sim.analysis import compute_pareto_frontier
from fsae_sim.optimization import SweepConfig
from fsae_sim.data import load_aim_csv, load_voltt_csv
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 8: Run full test suite**

Run: `pytest -v`
Expected: all tests from Task 4 and Task 5 still pass, no import errors

- [ ] **Step 9: Commit**

```bash
git add src/fsae_sim/track/ src/fsae_sim/driver/ src/fsae_sim/sim/ \
        src/fsae_sim/scoring/ src/fsae_sim/analysis/ src/fsae_sim/optimization/
git commit -m "feat: module interface stubs for track, driver, sim, scoring, analysis, optimization"
```

---

### Task 7: Docker environment

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yaml`

- [ ] **Step 1: Create Dockerfile**

Create `docker/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy project
COPY . .

# Re-install in editable mode with full source
RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 3000

CMD ["python", "-m", "dashboard"]
```

- [ ] **Step 2: Create docker-compose.yaml**

Create `docker/docker-compose.yaml`:

```yaml
services:
  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ../src:/app/src
      - ../dashboard:/app/dashboard
      - ../configs:/app/configs
      - ../tests:/app/tests
      - ../results:/app/results
      - ../Real-Car-Data-And-Stats:/app/Real-Car-Data-And-Stats
    environment:
      - DASH_DEBUG=true
```

- [ ] **Step 3: Commit**

Note: Do NOT build yet — the dashboard doesn't exist. This gets verified after Task 8.

```bash
git add docker/Dockerfile docker/docker-compose.yaml
git commit -m "feat: Docker environment with live code reload"
```

---

### Task 8: Dashboard skeleton

**Files:**
- Create: `dashboard/__main__.py`
- Create: `dashboard/app.py`
- Create: `dashboard/pages/overview.py`
- Create: `dashboard/pages/strategy.py`
- Create: `dashboard/pages/cars.py`
- Create: `dashboard/pages/sweep.py`
- Create: `dashboard/pages/pareto.py`
- Create: `dashboard/pages/lap_detail.py`

- [ ] **Step 1: Create dashboard entry point**

Create `dashboard/__main__.py`:

```python
"""Entry point: python -m dashboard"""

from dashboard.app import app

if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")
```

- [ ] **Step 2: Create main Dash app**

Create `dashboard/app.py`:

```python
"""FSAE EV Simulation Dashboard."""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="FSAE EV Sim",
)

sidebar = html.Div(
    [
        html.H4("FSAE EV Sim", className="text-light mb-2"),
        html.Hr(),
        html.P("CT-16EV / CT-17EV", className="text-muted small"),
        dbc.Nav(
            [
                dbc.NavLink("Overview", href="/", active="exact"),
                dbc.NavLink("Strategy Comparison", href="/strategy", active="exact"),
                dbc.NavLink("Car Comparison", href="/cars", active="exact"),
                dbc.NavLink("Parameter Sweep", href="/sweep", active="exact"),
                dbc.NavLink("Pareto Frontier", href="/pareto", active="exact"),
                dbc.NavLink("Lap Detail", href="/lap-detail", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="bg-dark p-3",
    style={"height": "100vh", "position": "fixed", "width": "220px"},
)

content = html.Div(
    dash.page_container,
    style={"marginLeft": "240px", "padding": "20px"},
)

app.layout = html.Div([sidebar, content])

if __name__ == "__main__":
    app.run(debug=True, port=3000, host="0.0.0.0")
```

- [ ] **Step 3: Create Overview page**

Create `dashboard/pages/overview.py`:

```python
"""Overview page — simulation summary dashboard."""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

dash.register_page(__name__, path="/", name="Overview")


def metric_card(title: str, value: str, subtitle: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1 small"),
            html.H3(value, className="mb-1"),
            html.Small(subtitle, className="text-muted"),
        ]),
        className="mb-3",
    )


# Placeholder chart
placeholder_fig = go.Figure()
placeholder_fig.add_annotation(
    text="Run a simulation to see results",
    xref="paper", yref="paper",
    x=0.5, y=0.5, showarrow=False,
    font=dict(size=18, color="gray"),
)
placeholder_fig.update_layout(
    template="plotly_dark",
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=0, r=0, t=40, b=0),
    title="SOC Depletion Over Endurance Run",
)

layout = dbc.Container(
    [
        html.H2("Simulation Overview", className="mb-4"),
        dbc.Row(
            [
                dbc.Col(metric_card("Best Lap Time", "--", "No data"), md=3),
                dbc.Col(metric_card("Total Energy", "--", "No data"), md=3),
                dbc.Col(metric_card("Final SOC", "--", "No data"), md=3),
                dbc.Col(metric_card("Predicted Points", "--", "No data"), md=3),
            ]
        ),
        dbc.Row([dbc.Col(dcc.Graph(figure=placeholder_fig), md=12)]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Alert(
                        [
                            html.H5("Getting Started", className="alert-heading"),
                            html.P(
                                "Run a simulation to populate this dashboard:"
                            ),
                            html.Code(
                                "python -m fsae_sim.sim.engine --config configs/ct17ev.yaml",
                                className="d-block mb-2",
                            ),
                            html.P(
                                "Or run a parameter sweep:",
                                className="mb-1",
                            ),
                            html.Code(
                                "python -m fsae_sim.optimization.sweep "
                                "--config ct17ev.yaml --sweep max_rpm"
                            ),
                        ],
                        color="info",
                        className="mt-3",
                    ),
                    md=12,
                ),
            ]
        ),
    ],
    fluid=True,
)
```

- [ ] **Step 4: Create placeholder pages**

Create `dashboard/pages/strategy.py`:

```python
"""Strategy Comparison page — compare driver strategies side-by-side."""

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/strategy", name="Strategy Comparison")

layout = dbc.Container([
    html.H2("Strategy Comparison", className="mb-4"),
    dbc.Alert(
        "Compare driver strategies (coasting, threshold braking, etc.) "
        "on the same car and track. Available after simulation implementation.",
        color="info",
    ),
], fluid=True)
```

Create `dashboard/pages/cars.py`:

```python
"""Car Comparison page — CT-16EV vs CT-17EV."""

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/cars", name="Car Comparison")

layout = dbc.Container([
    html.H2("Car Comparison", className="mb-4"),
    dbc.Alert(
        "Compare CT-16EV (2025, 110S4P P45B) vs CT-17EV (2026, 100S4P P50B) "
        "on the same strategy and track. Available after simulation implementation.",
        color="info",
    ),
], fluid=True)
```

Create `dashboard/pages/sweep.py`:

```python
"""Parameter Sweep page — visualize sweep results."""

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/sweep", name="Parameter Sweep")

layout = dbc.Container([
    html.H2("Parameter Sweep", className="mb-4"),
    dbc.Alert(
        "Visualize parameter sweep results: heatmaps, sensitivity plots, and "
        "optimal configurations. Available after sweep infrastructure is built.",
        color="info",
    ),
], fluid=True)
```

Create `dashboard/pages/pareto.py`:

```python
"""Pareto Frontier page — time vs energy tradeoff."""

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/pareto", name="Pareto Frontier")

layout = dbc.Container([
    html.H2("Pareto Frontier", className="mb-4"),
    dbc.Alert(
        "Interactive Pareto frontier showing the lap-time vs energy tradeoff. "
        "Click points to inspect strategy and config details. "
        "Available after optimization infrastructure is built.",
        color="info",
    ),
], fluid=True)
```

Create `dashboard/pages/lap_detail.py`:

```python
"""Lap Detail page — deep dive into a single simulation run."""

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/lap-detail", name="Lap Detail")

layout = dbc.Container([
    html.H2("Lap Detail", className="mb-4"),
    dbc.Alert(
        "Segment-by-segment view of a simulation run: speed, torque, current, "
        "SOC, and thermal state over distance. "
        "Available after simulation implementation.",
        color="info",
    ),
], fluid=True)
```

- [ ] **Step 5: Verify dashboard starts locally**

Run: `python -m dashboard`

Expected: Dash prints `Dash is running on http://0.0.0.0:3000/` and the browser at localhost:3000 shows the Overview page with metric cards and sidebar navigation.

Press Ctrl+C to stop.

- [ ] **Step 6: Verify Docker build and run**

Run:

```bash
docker compose -f docker/docker-compose.yaml build
docker compose -f docker/docker-compose.yaml up -d
```

Expected: container starts, localhost:3000 shows the dashboard.

```bash
docker compose -f docker/docker-compose.yaml down
```

- [ ] **Step 7: Commit**

```bash
git add dashboard/
git commit -m "feat: Dash dashboard skeleton with 6 pages on port 3000"
```

---

### Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

Create `README.md`:

````markdown
# FSAE EV Endurance Simulation

Endurance simulation and optimization for UConn Formula SAE Electric. Predicts lap times, energy consumption, and competition points to find optimal vehicle configuration and driver strategy.

## Cars

| | CT-16EV (2025) | CT-17EV (2026) |
|---|---|---|
| Pack | 110S4P Molicel P45B | 100S4P Molicel P50B |
| Mass | ~270 kg | ~261 kg |
| Motor/Inverter | Shared | Shared |
| Controls | Shared | Shared |

## Architecture

```
Vehicle Config (YAML)  →  Simulation Engine  →  Results  →  Dashboard
                              ↑
               Track (from GPS telemetry)
               Driver Strategy (swappable)
```

**Simulation method:** Quasi-static point-mass, calibrated against real AiM telemetry from the 2025 Michigan endurance event. For each track segment, resolves speed from force balance and driver strategy, steps battery state, enforces BMS limits.

**Modules:**

| Module | Purpose |
|---|---|
| `fsae_sim.vehicle` | Vehicle, powertrain, and battery configuration |
| `fsae_sim.track` | Track representation from GPS segments |
| `fsae_sim.driver` | Driver strategy / control policy (swappable) |
| `fsae_sim.sim` | Simulation engine |
| `fsae_sim.scoring` | FSAE endurance + efficiency point formulas |
| `fsae_sim.optimization` | Parameter sweep runner |
| `fsae_sim.analysis` | Post-processing metrics, Pareto computation |
| `fsae_sim.data` | Telemetry and simulation data loaders |
| `dashboard` | Dash web app for viewing results |

## Quick Start

### With Docker

```bash
docker compose -f docker/docker-compose.yaml up
# Browser → http://localhost:3000
```

### Without Docker

```bash
pip install -e ".[dev]"
python -m dashboard
# Browser → http://localhost:3000
```

### Run tests

```bash
pytest -v
```

## Project Structure

```
├── src/fsae_sim/          # Simulation Python package
│   ├── vehicle/           # Vehicle, powertrain, battery models
│   ├── track/             # Track representation
│   ├── driver/            # Driver strategy
│   ├── sim/               # Simulation engine
│   ├── scoring/           # FSAE scoring formulas
│   ├── optimization/      # Parameter sweeps
│   ├── analysis/          # Metrics and post-processing
│   └── data/              # Data loaders
├── dashboard/             # Dash web app (port 3000)
│   └── pages/             # Dashboard pages
├── configs/               # Vehicle config YAML files
├── Real-Car-Data-And-Stats/  # Telemetry and battery data
├── results/               # Simulation outputs (gitignored)
├── tests/                 # pytest test suite
└── docker/                # Dockerfile and compose
```

## Data

- **AiM telemetry:** `2025 Endurance Data.csv` — 20Hz, ~37k samples, ~100 channels from Michigan endurance
- **Voltt battery sim:** Cell and pack level CSVs — voltage, SOC, current, temperature, heat generation for 110S4P P45B
- **BMS tune:** `Endurance Tune2.txt` — discharge limits, SOC taper, inverter/motor parameters

## Roadmap

### Phase 1 — Foundation ✅
Repository scaffold, Docker, dashboard skeleton, vehicle configs, data loaders

### Phase 2 — Core Simulation
Battery model, powertrain model, vehicle dynamics, driver behavior extraction, simulation engine, validation against real data (5% target)

### Phase 3 — Optimization & Comparison
Swappable strategies, parameter sweeps, car comparison, Pareto frontier, dashboard buildout

### Phase 4 — Scoring & Decision Support
FSAE scoring model, field estimation, points maximization, final decision dashboard
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README with architecture and roadmap"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: all tests pass

- [ ] **Step 2: Verify Docker end-to-end**

Run:

```bash
docker compose -f docker/docker-compose.yaml up --build -d
```

Open browser to http://localhost:3000. Verify:
- Sidebar shows all 6 page links
- Overview page shows metric cards
- Each page link navigates correctly
- No errors in browser console

```bash
docker compose -f docker/docker-compose.yaml down
```

- [ ] **Step 3: Verify local dev workflow**

Run:

```bash
python -m dashboard
```

Verify same as above at http://localhost:3000. Ctrl+C to stop.

- [ ] **Step 4: Run import smoke test**

```bash
python -c "
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.track import Track, Segment
from fsae_sim.driver import DriverStrategy, SimState
from fsae_sim.sim import SimulationEngine
from fsae_sim.scoring import EnduranceScore
from fsae_sim.analysis import compute_pareto_frontier
from fsae_sim.optimization import SweepConfig
from fsae_sim.data import load_aim_csv, load_voltt_csv

cfg = VehicleConfig.from_yaml('configs/ct16ev.yaml')
print(f'Loaded: {cfg.name} ({cfg.year})')
print(f'  Pack: {cfg.battery.series}S{cfg.battery.parallel}P {cfg.battery.cell_type}')
print(f'  Mass: {cfg.vehicle.mass_kg} kg')
print(f'  Motor: {cfg.powertrain.motor_speed_max_rpm} RPM max')
print('All modules OK')
"
```

Expected:
```
Loaded: CT-16EV (2025)
  Pack: 110S4P P45B
  Mass: 270.0 kg
  Motor: 2900 RPM max
All modules OK
```
