from fastapi import APIRouter, Path

from backend.services.validation_export import get_all_laps_summary, get_validation_data

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/validation/{lap}")
def validation_for_lap(lap: int = Path(ge=1, le=30)):
    return get_validation_data(lap)


@router.get("/validation")
def validation_all_laps():
    return get_all_laps_summary()
