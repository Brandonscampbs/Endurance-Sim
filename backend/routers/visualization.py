from fastapi import APIRouter, Query

from backend.services.visualization_export import get_visualization_data

router = APIRouter(prefix="/api", tags=["visualization"])


@router.get("/visualization")
def visualization(source: str = Query(default="sim", pattern="^(sim|real)$")):
    return get_visualization_data(source=source)
