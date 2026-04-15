from pydantic import BaseModel


class TrackPoint(BaseModel):
    x: float
    y: float
    distance_m: float


class Sector(BaseModel):
    name: str
    sector_type: str  # "straight" or "corner"
    start_m: float
    end_m: float


class TrackData(BaseModel):
    centerline: list[TrackPoint]
    sectors: list[Sector]
    curvature: list[float]
    total_distance_m: float
