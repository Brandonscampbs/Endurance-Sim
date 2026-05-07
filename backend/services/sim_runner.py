from functools import lru_cache
from pathlib import Path

from fsae_sim.analysis.validation import detect_lap_boundaries
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


def get_validation_lap_split(num_laps: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return zero-based calibration and validation lap indices."""
    validation_laps = tuple(
        lap for lap in _VALIDATION_HOLDOUT_LAPS_ZERO_BASED
        if 0 <= lap < num_laps
    )
    validation_set = set(validation_laps)
    calibration_laps = tuple(
        lap for lap in range(num_laps) if lap not in validation_set
    )
    return calibration_laps, validation_laps


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
    aim_df = get_telemetry()
    _, validation_laps = get_validation_lap_split(_detected_lap_count(aim_df))
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(
        aim_df,
        holdout_laps=tuple(lap + 1 for lap in validation_laps),
    )
    return battery


def _detected_lap_count(aim_df) -> int:
    boundaries = detect_lap_boundaries(aim_df)
    if not boundaries:
        raise ValueError(
            "Could not detect telemetry lap boundaries; refusing to run "
            "validation with a hardcoded endurance lap count."
        )
    return len(boundaries)


# FSAE Michigan endurance is a fixed-distance event (~22 km). With this
# car's ~1005 m laps that's 22 timed laps. ``detect_lap_boundaries``
# only finds 21 because the recording stops 14.9 s into lap 22 before
# the final start/finish crossing — the real run was 22 laps but lap 22
# wasn't fully captured by lap-detection, just by total distance.
ENDURANCE_LAP_COUNT = 22


@lru_cache(maxsize=1)
def get_baseline_result() -> SimResult:
    """Run the held-out validation baseline with CalibratedStrategy."""
    vehicle = get_vehicle_config()
    track = get_track()
    battery = get_battery_model()
    aim_df = get_telemetry()
    detected = _detected_lap_count(aim_df)
    _, validation_laps = get_validation_lap_split(detected)

    strategy = CalibratedStrategy.from_telemetry(
        aim_df,
        track,
        holdout_laps=list(validation_laps),
        use_observed_speed_caps=False,
    )
    engine = SimulationEngine(
        vehicle,
        track,
        strategy,
        battery,
        mode=SimulationMode.VALIDATION,
    )
    # Run the FSAE-rule lap count (22), not the detected-boundary count
    # (21) — see ENDURANCE_LAP_COUNT note above.
    return engine.run(
        num_laps=ENDURANCE_LAP_COUNT,
        initial_soc_pct=95.0,
        initial_temp_c=29.0,
    )


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
    _, validation_laps = get_validation_lap_split(_detected_lap_count(aim_df))
    battery.calibrate_pack_from_telemetry(
        aim_df,
        holdout_laps=tuple(lap + 1 for lap in validation_laps),
    )

    strategy = CalibratedStrategy.from_telemetry(
        aim_df,
        track,
        holdout_laps=list(validation_laps),
        use_observed_speed_caps=False,
    )
    engine = SimulationEngine(
        vehicle,
        track,
        strategy,
        battery,
        mode=SimulationMode.VALIDATION,
    )
    return engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=29.0)
