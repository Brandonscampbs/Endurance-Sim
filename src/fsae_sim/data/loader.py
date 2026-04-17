"""Data loaders for AiM telemetry and Voltt battery simulation exports."""

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


def load_aim_csv(path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Load an AiM Race Studio CSV export.

    AiM CSV format:
    - Metadata lines: "key","value" pairs until first blank line
    - Column headers line
    - Units line
    - Blank line (may be absent)
    - Data lines (quoted numeric values)

    Returns:
        metadata: dict of session metadata (Vehicle, Date, Sample Rate, etc.)
        df: DataFrame with numeric columns. Units stored in df.attrs['units'].
    """
    path = Path(path)
    metadata: dict[str, str] = {}

    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    idx = 0

    # Parse metadata (until first blank line)
    while idx < len(lines) and lines[idx].strip():
        row = next(csv.reader(io.StringIO(lines[idx].strip())))
        if len(row) >= 2:
            metadata[row[0]] = row[1]
        elif len(row) == 1:
            metadata[row[0]] = ""
        idx += 1

    # Skip blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Column headers
    headers = next(csv.reader(io.StringIO(lines[idx].strip())))
    idx += 1

    # Units line
    unit_list = next(csv.reader(io.StringIO(lines[idx].strip())))
    idx += 1

    # Skip blank lines before data
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Build units dict using first occurrence of each header name
    # (pandas will rename duplicates with .1, .2 suffixes — mirror that logic)
    seen: dict[str, int] = {}
    unique_headers: list[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}.{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)

    units: dict[str, str] = {}
    for header, unit in zip(unique_headers, unit_list):
        units[header] = unit

    # Parse data — join remaining lines and read as CSV
    data_text = "".join(lines[idx:])
    df = pd.read_csv(
        io.StringIO(data_text),
        header=None,
        names=unique_headers,
    )

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.attrs["units"] = units
    df.attrs["metadata"] = metadata

    return metadata, df


def load_cleaned_csv(path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Load a cleaned AiM telemetry CSV (no AiM metadata headers).

    Expected format:
    - Row 1: column headers
    - Row 2: units (skipped)
    - Row 3+: data

    Key differences from the raw AiM export:
    - Speed is in ``LFspeed`` (front wheel speed sensor, km/h), not ``GPS Speed``
    - No ``Distance on GPS Speed`` column — computed via trapezoidal integration
    - No ``GPS PosAccuracy`` or ``GPS Radius`` columns
    - Encoding is latin-1 (°C symbol)

    Adds compatibility columns so downstream code works unchanged:
    - ``GPS Speed`` = ``LFspeed``
    - ``Distance on GPS Speed`` = cumulative integral of speed

    Returns:
        metadata: empty dict (no metadata in cleaned format)
        df: DataFrame with numeric columns and compatibility aliases.
    """
    path = Path(path)
    df = pd.read_csv(path, skiprows=[1], encoding="latin-1")

    required_columns = ("Time", "LFspeed", "GPS Latitude", "GPS Longitude")
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column {col}")

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill the rare NaN in LFspeed by interpolation
    if df["LFspeed"].isna().any():
        df["LFspeed"] = df["LFspeed"].interpolate().bfill().ffill()

    # Add compatibility columns
    df["GPS Speed"] = df["LFspeed"]

    # Compute cumulative distance from speed (trapezoidal integration)
    time = df["Time"].values
    speed_ms = df["LFspeed"].values / 3.6
    dt = np.diff(time, prepend=time[0])
    df["Distance on GPS Speed"] = np.cumsum(speed_ms * dt)

    return {}, df


def load_voltt_csv(path: str | Path) -> pd.DataFrame:
    """Load a Voltt battery simulation CSV export.

    Voltt CSV format:
    - Comment lines starting with #
    - Standard CSV with headers

    Returns:
        df: DataFrame with numeric columns.
    """
    path = Path(path)
    df = pd.read_csv(path, comment="#", encoding="utf-8")

    if len(df) == 0:
        raise ValueError(f"{path}: file contains no data rows")

    required_columns = ("SOC [%]", "OCV [V]", "Voltage [V]", "Current [A]")
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column {col}")

    return df
