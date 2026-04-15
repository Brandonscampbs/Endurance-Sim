from fastapi import APIRouter

from backend.services.track_service import get_track_data

router = APIRouter(prefix="/api", tags=["track"])


@router.get("/track")
def get_track():
    return get_track_data()
