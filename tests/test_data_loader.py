"""Tests for data loaders and YAML vehicle config validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from fsae_sim.data.loader import load_aim_csv, load_cleaned_csv, load_voltt_csv
from fsae_sim.vehicle.vehicle import VehicleConfig


# ---------------------------------------------------------------------------
# NF-1: load_aim_csv must use latin-1 (handles °C, µ, etc.)
# ---------------------------------------------------------------------------
def test_load_aim_csv_decodes_latin1(tmp_path: Path) -> None:
    csv_bytes = (
        b'"Vehicle","CT-16EV"\n'
        b'"Pack Temp Units","\xb0C"\n'
        b'\n'
        b'"Time","GPS Speed"\n'
        b'"s","km/h"\n'
        b'\n'
        b'"0.0","10.0"\n'
        b'"0.1","11.0"\n'
    )
    path = tmp_path / "aim.csv"
    path.write_bytes(csv_bytes)

    metadata, df = load_aim_csv(path)
    assert metadata["Pack Temp Units"] == "\u00b0C"
    assert len(df) == 2
    assert df["GPS Speed"].iloc[1] == 11.0


# ---------------------------------------------------------------------------
# NF-27: load_cleaned_csv validates required columns
# ---------------------------------------------------------------------------
def _write_cleaned_csv(path: Path, header: str) -> None:
    # row 1 header, row 2 units (skipped), row 3+ data
    path.write_text(
        f"{header}\n"
        f"{','.join(['-'] * len(header.split(',')))}\n"
        + "\n".join(",".join(["0"] * len(header.split(","))) for _ in range(3))
        + "\n",
        encoding="latin-1",
    )


def test_load_cleaned_csv_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "cleaned.csv"
    _write_cleaned_csv(path, "Time,LFspeed,GPS Latitude")  # missing GPS Longitude

    with pytest.raises(ValueError, match="missing required column GPS Longitude"):
        load_cleaned_csv(path)


def test_load_cleaned_csv_ok_with_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "cleaned.csv"
    _write_cleaned_csv(path, "Time,LFspeed,GPS Latitude,GPS Longitude")
    _, df = load_cleaned_csv(path)
    assert "GPS Speed" in df.columns  # compatibility alias


# ---------------------------------------------------------------------------
# NF-28: load_voltt_csv validates non-empty and required columns
# ---------------------------------------------------------------------------
def test_load_voltt_csv_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "voltt.csv"
    path.write_text("# comment\nSOC [%],OCV [V],Voltage [V],Current [A]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        load_voltt_csv(path)


def test_load_voltt_csv_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "voltt.csv"
    path.write_text(
        "# comment\nSOC [%],OCV [V],Voltage [V]\n100,4.2,4.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required column Current"):
        load_voltt_csv(path)


def test_load_voltt_csv_ok(tmp_path: Path) -> None:
    path = tmp_path / "voltt.csv"
    path.write_text(
        "# comment\nSOC [%],OCV [V],Voltage [V],Current [A]\n100,4.2,4.18,-0.24\n",
        encoding="utf-8",
    )
    df = load_voltt_csv(path)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# NF-12: VehicleConfig.from_yaml wraps TypeError with file context
# ---------------------------------------------------------------------------
def test_vehicle_config_unknown_field_raises_value_error(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        name: test
        year: 2026
        description: bad config
        vehicle:
          mass_kg: 288.0
          frontal_area_m2: 1.0
          drag_coefficient: 1.5
          rolling_resistance: 0.02
          wheelbase_m: 1.549
          this_field_does_not_exist: 42
        powertrain: {}
        battery: {}
        """
    ).strip()
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        VehicleConfig.from_yaml(path)
    assert str(path) in str(exc_info.value)
