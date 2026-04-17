#!/usr/bin/env python3
"""Clean FSAE endurance telemetry data.

Removes pre-start, driver change stop, and post-finish periods from the raw
AiM telemetry CSV. Concatenates the two driving stints with continuous time.

Boundaries were determined by GPS lap analysis:
- Start (green flag launch): t=21.0s  (first hard throttle, RPM ramp)
- Stint 1 end (DC entry):    t=859.0s (car stops at DC zone)
- Stint 2 start (DC exit):   t=1069.0s (car at full racing speed after DC)
- Finish (last S/F crossing): t=1845.0s (predicted from GPS crossings at ~71s intervals)

The driver change gap (859.0 to 1069.0 = 210s) is removed. This includes the
stationary period, two brief test pulls, and the acceleration out of the DC zone.

Total driving time: 838.0 + 776.0 = 1614.0s (matches official result).

GPS notes: GPS Speed is stuck at 33.286 km/h for the first ~162s due to
poor satellite fix (only 3 sats). Distance is recalculated from GPS Speed.

Usage:
    python scripts/clean_endurance_data.py
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "Real-Car-Data-And-Stats" / "2025 Endurance Data.csv"
OUT_CSV = REPO_ROOT / "Real-Car-Data-And-Stats" / "2025 Endurance Data - Cleaned.csv"

# Boundaries (raw timestamps in seconds) determined by GPS lap analysis
STINT1_START = 21.0    # Green flag launch (hard throttle)
STINT1_END = 859.0     # Car stops at DC zone
STINT2_START = 1069.0  # Car at full racing speed after DC
STINT2_END = 1845.0    # Last S/F crossing (predicted from GPS crossing intervals)


def load_raw_data(path: Path) -> tuple[list[str], str, str, pd.DataFrame]:
    """Load the raw AiM CSV, returning metadata lines, header row, units row, and data."""
    with open(path, "r", encoding="latin-1") as f:
        all_lines = f.readlines()

    metadata_lines = all_lines[:14]
    header_line = all_lines[14]
    units_line = all_lines[15]

    # Parse header, deduplicating repeated column names
    reader = csv.reader(io.StringIO(header_line))
    raw_columns = next(reader)
    columns: list[str] = []
    seen: dict[str, int] = {}
    for col in raw_columns:
        if col in seen:
            seen[col] += 1
            columns.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            columns.append(col)

    # Data starts at line 18 (0-indexed: 17) -- line 17 is blank
    data_text = "".join(all_lines[17:])
    df = pd.read_csv(
        io.StringIO(data_text),
        header=None,
        names=columns,
        na_values=[""],
    )
    # Coerce numerics where possible; leave non-numeric strings (e.g. flags) intact
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return metadata_lines, header_line, units_line, df


def recalculate_distance(df: pd.DataFrame) -> np.ndarray:
    """Recalculate cumulative distance from GPS Speed using trapezoidal integration."""
    speed_mps = df["GPS Speed"].values / 3.6  # km/h -> m/s
    dt = np.diff(df["Time"].values, prepend=df["Time"].values[0])
    dt[0] = 0.0
    incremental = speed_mps * dt
    return np.cumsum(incremental)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract driving stints and concatenate with continuous time."""
    time = df["Time"].values

    # Find indices for boundaries
    s1_start = np.searchsorted(time, STINT1_START)
    s1_end = np.searchsorted(time, STINT1_END, side="right") - 1
    s2_start = np.searchsorted(time, STINT2_START)
    s2_end = np.searchsorted(time, STINT2_END, side="right") - 1

    s1 = df.iloc[s1_start:s1_end + 1].copy()
    s2 = df.iloc[s2_start:s2_end + 1].copy()

    # Reset time: stint 1 starts at 0, stint 2 continues from stint 1 end
    dt = 0.05  # 20 Hz sample period
    s1_duration = s1["Time"].iloc[-1] - s1["Time"].iloc[0]

    s1["Time"] = s1["Time"] - s1["Time"].iloc[0]
    s2["Time"] = s2["Time"] - s2["Time"].iloc[0] + s1_duration + dt

    cleaned = pd.concat([s1, s2], ignore_index=True)

    # Recalculate distance
    dist_col = "Distance on GPS Speed"
    if dist_col in cleaned.columns:
        cleaned[dist_col] = recalculate_distance(cleaned)

    return cleaned


def write_cleaned_csv(
    out_path: Path,
    metadata_lines: list[str],
    header_line: str,
    units_line: str,
    df: pd.DataFrame,
) -> None:
    """Write the cleaned CSV preserving AiM metadata format."""
    total_duration = df["Time"].iloc[-1]

    updated_metadata = []
    for line in metadata_lines:
        if line.startswith('"Duration"'):
            updated_metadata.append(f'"Duration","{total_duration:.0f}"\n')
        elif line.startswith('"Beacon Markers"'):
            updated_metadata.append(f'"Beacon Markers","{total_duration:.0f}"\n')
        elif line.startswith('"Segment Times"'):
            mins = int(total_duration // 60)
            secs = total_duration % 60
            updated_metadata.append(f'"Segment Times","{mins}:{secs:06.3f}"\n')
        else:
            updated_metadata.append(line)

    # Identify columns needing high precision (GPS coordinates)
    col_names = list(df.columns)
    high_prec_cols = {
        i for i, c in enumerate(col_names)
        if "Latitude" in c or "Longitude" in c
    }

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for line in updated_metadata:
            f.write(line)
        f.write(header_line)
        f.write(units_line)
        f.write("\n")

        for _, row in df.iterrows():
            values = []
            for i, val in enumerate(row):
                if pd.isna(val):
                    values.append("")
                elif i in high_prec_cols:
                    values.append(f"{val:.9f}")
                else:
                    values.append(f"{val:.4f}")
            f.write('"' + '","'.join(values) + '"\n')


def main() -> None:
    """Main entry point."""
    print(f"Loading raw telemetry from: {RAW_CSV}")
    metadata_lines, header_line, units_line, df = load_raw_data(RAW_CSV)

    time = df["Time"].values
    print(f"Raw data: {len(df)} samples, {time[-1]:.1f}s total")

    # Print boundary info
    print(f"\n--- GPS-Validated Boundaries ---")
    print(f"Stint 1: t={STINT1_START:.1f}s to {STINT1_END:.1f}s "
          f"({STINT1_END - STINT1_START:.1f}s)")
    print(f"DC gap:  t={STINT1_END:.1f}s to {STINT2_START:.1f}s "
          f"({STINT2_START - STINT1_END:.1f}s removed)")
    print(f"Stint 2: t={STINT2_START:.1f}s to {STINT2_END:.1f}s "
          f"({STINT2_END - STINT2_START:.1f}s)")

    pre_start = STINT1_START
    dc_gap = STINT2_START - STINT1_END
    post_finish = time[-1] - STINT2_END
    total_removed = pre_start + dc_gap + post_finish
    print(f"\nRemoved periods:")
    print(f"  Pre-start:       {pre_start:.1f}s")
    print(f"  Driver change:   {dc_gap:.1f}s")
    print(f"  Post-finish:     {post_finish:.1f}s")
    print(f"  Total removed:   {total_removed:.1f}s")

    # Clean
    cleaned = clean_data(df)

    total_time = cleaned["Time"].iloc[-1]
    total_points = len(cleaned)

    dist_col = "Distance on GPS Speed"
    total_dist = cleaned[dist_col].iloc[-1] if dist_col in cleaned.columns else 0.0

    stint1_samples = int((STINT1_END - STINT1_START) / 0.05) + 1
    stint1_time = STINT1_END - STINT1_START
    stint2_time = STINT2_END - STINT2_START

    print(f"\n--- Cleaned Data Summary ---")
    print(f"Total driving time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Total distance:     {total_dist:.0f}m ({total_dist / 1000:.2f} km)")
    print(f"Data points:        {total_points}")
    print(f"Stint 1 duration:   {stint1_time:.1f}s ({stint1_time / 60:.1f} min)")
    print(f"Stint 2 duration:   {stint2_time:.1f}s ({stint2_time / 60:.1f} min)")

    if "State of Charge" in cleaned.columns:
        soc_start = cleaned["State of Charge"].iloc[0]
        soc_end = cleaned["State of Charge"].iloc[-1]
        print(f"SOC:                {soc_start:.1f}% -> {soc_end:.1f}% "
              f"(used {soc_start - soc_end:.1f}%)")

    # Validation
    print(f"\n--- Validation ---")
    target_time = 1614.0
    time_err = abs(total_time - target_time)
    if time_err < 2.0:
        print(f"  Total time {total_time:.1f}s vs target {target_time:.0f}s "
              f"(err={time_err:.1f}s) -- PASS")
    else:
        print(f"  WARNING: Total time {total_time:.1f}s vs target {target_time:.0f}s "
              f"(err={time_err:.1f}s)")

    if 20000 < total_dist < 23000:
        print(f"  Distance {total_dist:.0f}m in range (20-23 km) -- PASS")
    else:
        print(f"  WARNING: Distance {total_dist:.0f}m outside range (20-23 km)")

    # GPS data quality note
    gps_speed = cleaned["GPS Speed"].values
    stuck_val = 33.286
    stuck_mask = np.abs(gps_speed - stuck_val) < 0.01
    n_stuck = np.sum(stuck_mask)
    if n_stuck > 0:
        stuck_end_idx = 0
        for i in range(len(stuck_mask)):
            if not stuck_mask[i]:
                stuck_end_idx = i
                break
        stuck_duration = cleaned["Time"].iloc[stuck_end_idx]
        print(f"\n  NOTE: GPS Speed stuck at {stuck_val:.1f} km/h for first "
              f"{stuck_duration:.1f}s ({stuck_end_idx} samples) due to poor "
              f"satellite fix. RPM-based speed available via motor RPM / "
              f"gear ratio (3.6363) / tire radius.")

    # Write output
    print(f"\nWriting cleaned CSV to: {OUT_CSV}")
    write_cleaned_csv(OUT_CSV, metadata_lines, header_line, units_line, cleaned)
    print(f"Output file size: {OUT_CSV.stat().st_size / 1_000_000:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
