from fastapi import APIRouter

from backend.services.telemetry_service import get_lap_gps_quality

router = APIRouter(prefix="/api", tags=["laps"])


@router.get("/laps")
def list_laps():
    return {"laps": get_lap_gps_quality()}
