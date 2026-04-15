from functools import lru_cache
from pathlib import Path

from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimResult, SimulationEngine
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


@lru_cache(maxsize=1)
def get_vehicle_config() -> VehicleConfig:
    return VehicleConfig.from_yaml(str(_CONFIG_PATH))


@lru_cache(maxsize=1)
def get_track() -> Track:
    aim_df = get_telemetry()
    return Track.from_telemetry(df=aim_df)


@lru_cache(maxsize=1)
def get_battery_model() -> BatteryModel:
    vehicle = get_vehicle_config()
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(get_telemetry())
    return battery


@lru_cache(maxsize=1)
def get_baseline_result() -> SimResult:
    """Run baseline 22-lap simulation with CalibratedStrategy."""
    vehicle = get_vehicle_config()
    track = get_track()
    battery = get_battery_model()
    aim_df = get_telemetry()

    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    engine = SimulationEngine(vehicle, track, strategy, battery)
    return engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=29.0)


def run_single_lap_sim(lap_number: int = 1) -> SimResult:
    """Run a single-lap simulation for visualization."""
    vehicle = get_vehicle_config()
    track = get_track()
    aim_df = get_telemetry()

    # Use fresh battery (not calibrated-pack) for single-lap
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(aim_df)

    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    engine = SimulationEngine(vehicle, track, strategy, battery)
    return engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=29.0)
