"""Shared test fixtures for FSAE simulation tests."""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# Project root needed so scripts.calibrate_coast_loss et al. resolve
# as module paths in tests that exercise the calibration scripts.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data paths
DATA_DIR = PROJECT_ROOT / "Real-Car-Data-And-Stats"
AIM_CSV = DATA_DIR / "2025 Endurance Data.csv"
VOLTT_CELL_CSV = DATA_DIR / "About-Energy-Volt-Simulations-2025-Pack" / "2025_Pack_cell.csv"
VOLTT_PACK_CSV = DATA_DIR / "About-Energy-Volt-Simulations-2025-Pack" / "2025_Pack_pack.csv"

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
