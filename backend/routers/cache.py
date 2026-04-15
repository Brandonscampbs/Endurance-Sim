from fastapi import APIRouter

from backend.services.sim_runner import (
    get_baseline_result,
    get_battery_model,
    get_track,
    get_vehicle_config,
)
from backend.services.telemetry_service import get_lap_boundaries, get_telemetry

router = APIRouter(prefix="/api", tags=["cache"])


@router.post("/cache/clear")
def clear_cache():
    """Flush all backend lru_caches so the next request re-runs the simulation."""
    get_telemetry.cache_clear()
    get_lap_boundaries.cache_clear()
    get_vehicle_config.cache_clear()
    get_track.cache_clear()
    get_battery_model.cache_clear()
    get_baseline_result.cache_clear()
    return {"status": "cleared"}
