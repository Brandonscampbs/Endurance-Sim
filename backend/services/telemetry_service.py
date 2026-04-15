from functools import lru_cache
from pathlib import Path

import pandas as pd

from fsae_sim.analysis.validation import detect_lap_boundaries
from fsae_sim.data.loader import load_aim_csv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AIM_CSV = _PROJECT_ROOT / "Real-Car-Data-And-Stats" / "2025 Endurance Data.csv"


@lru_cache(maxsize=1)
def get_telemetry() -> pd.DataFrame:
    """Load and cache the AiM telemetry CSV."""
    _, df = load_aim_csv(str(_AIM_CSV))
    return df


@lru_cache(maxsize=1)
def get_lap_boundaries() -> list[tuple[int, int, float]]:
    """Detect lap boundaries. Returns list of (start_idx, end_idx, lap_distance_m)."""
    df = get_telemetry()
    return detect_lap_boundaries(df)


def get_lap_data(lap_number: int) -> pd.DataFrame:
    """Extract telemetry for a single lap (1-indexed)."""
    boundaries = get_lap_boundaries()
    if lap_number < 1 or lap_number > len(boundaries):
        raise ValueError(f"Lap {lap_number} not found. Available: 1-{len(boundaries)}")
    start, end, _ = boundaries[lap_number - 1]
    df = get_telemetry()
    lap_df = df.iloc[start:end].copy()
    # Normalize distance to start of lap
    d0 = lap_df["Distance on GPS Speed"].iloc[0]
    lap_df["lap_distance_m"] = lap_df["Distance on GPS Speed"] - d0
    return lap_df


def get_lap_gps_quality() -> list[dict]:
    """Score each lap's GPS quality. Returns list of {lap_number, quality_score, time_s}."""
    df = get_telemetry()
    boundaries = get_lap_boundaries()
    results = []
    for idx, (start, end, _) in enumerate(boundaries):
        lap_slice = df.iloc[start:end]
        acc = lap_slice["GPS PosAccuracy"]
        valid = acc[acc < 200.0]
        quality = float(valid.mean()) if len(valid) > 0 else 999.0
        time_s = float(lap_slice["Time"].iloc[-1] - lap_slice["Time"].iloc[0])
        results.append({
            "lap_number": idx + 1,
            "gps_quality_score": round(quality, 1),
            "time_s": round(time_s, 2),
            "valid_gps_pct": round(100 * len(valid) / len(acc), 1),
        })
    return results
