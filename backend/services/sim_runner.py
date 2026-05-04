from functools import lru_cache
from pathlib import Path

from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimResult, SimulationEngine, SimulationMode
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel

from backend.services.telemetry_service import get_telemetry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "ct16ev.yaml"
_VOLTT_CELL_PATH = (
    _PROJECT_ROOT
    / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)
_VALIDATION_HOLDOUT_LAPS_ZERO_BASED = tuple(range(12, 22))
_VALIDATION_HOLDOUT_LAPS_ONE_BASED = tuple(i + 1 for i in _VALIDATION_HOLDOUT_LAPS_ZERO_BASED)


@lru_cache(maxsize=1)
def get_vehicle_config() -> VehicleConfig:
    return VehicleConfig.from_yaml(str(_CONFIG_PATH))


@lru_cache(maxsize=1)
def get_track() -> Track:
    """Single-lap track (legacy) used by calibrated/baseline backends.

    Calibrated zones are lap-relative; per-lap stitched tracks need
    cumulative-distance-aware zones, which the strategy doesn't have
    yet. Replay paths that want lap-to-lap curvature variation should
    call ``Track.from_telemetry_per_lap`` directly.
    """
    aim_df = get_telemetry()
    return Track.from_telemetry(df=aim_df)


@lru_cache(maxsize=1)
def get_battery_model() -> BatteryModel:
    vehicle = get_vehicle_config()
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(
        get_telemetry(),
        holdout_laps=_VALIDATION_HOLDOUT_LAPS_ONE_BASED,
    )
    return battery


@lru_cache(maxsize=1)
def get_baseline_result() -> SimResult:
    """Run baseline 22-lap simulation with CalibratedStrategy."""
    vehicle = get_vehicle_config()
    track = get_track()
    battery = get_battery_model()
    aim_df = get_telemetry()

    strategy = CalibratedStrategy.from_telemetry(
        aim_df,
        track,
        holdout_laps=list(_VALIDATION_HOLDOUT_LAPS_ZERO_BASED),
    )
    engine = SimulationEngine(
        vehicle,
        track,
        strategy,
        battery,
        mode=SimulationMode.CALIBRATION,
    )
    return engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=29.0)


def run_single_lap_sim(lap_number: int = 1) -> SimResult:
    """Run a single-lap simulation for visualization."""
    vehicle = get_vehicle_config()
    aim_df = get_telemetry()
    # Single-lap viz uses the legacy single-lap reconstruction, not the
    # stitched per-lap track — one lap's geometry is exactly what we
    # want here, repeated however many times the caller asks for.
    track = Track.from_telemetry(df=aim_df)

    # Use fresh battery (not calibrated-pack) for single-lap
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(
        aim_df,
        holdout_laps=_VALIDATION_HOLDOUT_LAPS_ONE_BASED,
    )

    strategy = CalibratedStrategy.from_telemetry(
        aim_df,
        track,
        holdout_laps=list(_VALIDATION_HOLDOUT_LAPS_ZERO_BASED),
    )
    engine = SimulationEngine(
        vehicle,
        track,
        strategy,
        battery,
        mode=SimulationMode.CALIBRATION,
    )
    return engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=29.0)
