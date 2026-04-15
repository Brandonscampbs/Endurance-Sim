from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from backend.models.track import Sector, TrackData, TrackPoint
from backend.services.telemetry_service import get_lap_boundaries, get_telemetry
from fsae_sim.track.track import Track

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_CURVATURE_CORNER_THRESHOLD = 0.01  # 1/m -- above this is a corner
_MIN_SPEED_KMH = 5.0


def build_track_xy(
    lats: np.ndarray,
    lons: np.ndarray,
    distances: np.ndarray,
    bin_size_m: float = 1.0,
) -> list[TrackPoint]:
    """Convert GPS lat/lon to local XY meters via equirectangular projection."""
    lat_ref = lats[0]
    lon_ref = lons[0]
    cos_lat = np.cos(np.radians(lat_ref))

    # Degrees to meters (equirectangular)
    x_raw = (lons - lon_ref) * cos_lat * 111_320.0
    y_raw = (lats - lat_ref) * 110_540.0

    # Remove duplicate distances for spline
    mask = np.diff(distances, prepend=-1) > 0.01
    d_clean = distances[mask]
    x_clean = x_raw[mask]
    y_clean = y_raw[mask]

    if len(d_clean) < 4:
        return [TrackPoint(x=0.0, y=0.0, distance_m=0.0)]

    # Cubic spline interpolation to uniform spacing
    cs_x = CubicSpline(d_clean, x_clean)
    cs_y = CubicSpline(d_clean, y_clean)

    d_uniform = np.arange(0, d_clean[-1], bin_size_m)
    points = [
        TrackPoint(x=float(cs_x(d)), y=float(cs_y(d)), distance_m=float(d))
        for d in d_uniform
    ]
    return points


def detect_sectors(
    curvatures: list[float],
    distances: list[float],
    threshold: float = _CURVATURE_CORNER_THRESHOLD,
) -> list[Sector]:
    """Segment the track into corner and straight sectors."""
    sectors: list[Sector] = []
    corner_count = 0
    straight_count = 0

    i = 0
    while i < len(curvatures):
        is_corner = abs(curvatures[i]) > threshold
        start_idx = i
        while i < len(curvatures) and (abs(curvatures[i]) > threshold) == is_corner:
            i += 1
        end_idx = i - 1

        if is_corner:
            corner_count += 1
            name = f"Turn {corner_count}"
            sector_type = "corner"
        else:
            straight_count += 1
            name = f"Straight {straight_count}"
            sector_type = "straight"

        sectors.append(Sector(
            name=name,
            sector_type=sector_type,
            start_m=distances[start_idx],
            end_m=distances[min(end_idx, len(distances) - 1)],
        ))

    return sectors


def _load_best_lap_gps(aim_df: pd.DataFrame, track: Track) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract GPS data for the lap with the best quality."""
    boundaries = get_lap_boundaries()
    if not boundaries:
        raise ValueError("No laps detected in telemetry")

    # Pick lap with the smoothest GPS heading (proxy for quality without PosAccuracy)
    best_score = float("inf")
    best_lap_idx = 0
    for idx, (start, end, _) in enumerate(boundaries):
        lap_slice = aim_df.iloc[start:end]
        # Filter low speed samples
        fast = lap_slice[lap_slice["GPS Speed"] > _MIN_SPEED_KMH]
        if len(fast) < 20:
            continue
        if "GPS Heading" in fast.columns:
            score = float(fast["GPS Heading"].diff().abs().std())
            if not np.isnan(score) and score < best_score:
                best_score = score
                best_lap_idx = idx
        else:
            best_lap_idx = 0
            break

    start, end, _ = boundaries[best_lap_idx]
    lap_df = aim_df.iloc[start:end].copy()

    # Filter low speed
    mask = lap_df["GPS Speed"] > _MIN_SPEED_KMH
    lap_df = lap_df[mask]

    lats = lap_df["GPS Latitude"].values
    lons = lap_df["GPS Longitude"].values
    dists = lap_df["Distance on GPS Speed"].values
    # Normalize distance to start of lap
    dists = dists - dists[0]

    return lats, lons, dists


def get_track_data() -> TrackData:
    """Build complete track data with XY coordinates and sectors."""
    aim_df = get_telemetry()
    track = Track.from_telemetry(df=aim_df)

    lats, lons, dists = _load_best_lap_gps(aim_df, track)

    centerline = build_track_xy(lats, lons, dists, bin_size_m=1.0)

    curvatures = [float(s.curvature) for s in track.segments]
    seg_distances = [float(s.distance_start_m) for s in track.segments]
    sectors = detect_sectors(curvatures, seg_distances)

    return TrackData(
        centerline=centerline,
        sectors=sectors,
        curvature=curvatures,
        total_distance_m=track.total_distance_m,
    )
