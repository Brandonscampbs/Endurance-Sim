from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fsae_sim.analysis.validation import detect_lap_boundaries
from fsae_sim.data.loader import load_cleaned_csv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CLEANED_CSV = _PROJECT_ROOT / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"


@lru_cache(maxsize=1)
def get_telemetry() -> pd.DataFrame:
    """Load and cache the cleaned AiM telemetry CSV."""
    _, df = load_cleaned_csv(str(_CLEANED_CSV))
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
    """Score each lap by GPS signal quality.

    The cleaned CSV lacks GPS PosAccuracy, so we score by GPS heading
    variance (lower = smoother = better quality GPS signal).
    """
    df = get_telemetry()
    boundaries = get_lap_boundaries()
    results = []
    for idx, (start, end, _) in enumerate(boundaries):
        lap_slice = df.iloc[start:end]
        time_s = float(lap_slice["Time"].iloc[-1] - lap_slice["Time"].iloc[0])

        # Score by GPS heading smoothness — lower std = better quality
        if "GPS Heading" in lap_slice.columns:
            heading_std = float(lap_slice["GPS Heading"].diff().abs().std())
            quality = round(heading_std, 1) if not np.isnan(heading_std) else 999.0
        else:
            quality = 0.0  # no GPS heading column — treat all laps equally

        results.append({
            "lap_number": idx + 1,
            "gps_quality_score": quality,
            "time_s": round(time_s, 2),
            "valid_gps_pct": 100.0,
        })
    return results
