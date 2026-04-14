"""Validation of simulation output against real AiM telemetry.

Provides functions to align simulation and telemetry data, compute error
metrics, and generate a structured validation report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fsae_sim.data.loader import load_aim_csv


@dataclass
class ValidationMetric:
    """Result of one validation comparison."""
    name: str
    unit: str
    telemetry_value: float
    simulation_value: float
    absolute_error: float
    relative_error_pct: float
    target_pct: float
    passed: bool


@dataclass
class ValidationReport:
    """Summary of all validation metrics."""
    metrics: list[ValidationMetric]

    @property
    def all_passed(self) -> bool:
        return all(m.passed for m in self.metrics)

    @property
    def num_passed(self) -> int:
        return sum(1 for m in self.metrics if m.passed)

    @property
    def num_total(self) -> int:
        return len(self.metrics)

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.num_passed}/{self.num_total} metrics passed",
            "-" * 70,
        ]
        for m in self.metrics:
            status = "PASS" if m.passed else "FAIL"
            lines.append(
                f"  [{status}] {m.name}: "
                f"telemetry={m.telemetry_value:.2f}{m.unit}, "
                f"sim={m.simulation_value:.2f}{m.unit}, "
                f"err={m.relative_error_pct:.1f}% "
                f"(target <{m.target_pct:.0f}%)"
            )
        return "\n".join(lines)


def detect_lap_boundaries(
    aim_df: pd.DataFrame,
    min_speed_kmh: float = 5.0,
) -> list[tuple[int, int, float]]:
    """Detect lap start/end row indices from AiM telemetry.

    Uses the same latitude-crossing method as Track.from_telemetry.

    Returns:
        List of (start_row, end_row, lap_distance_m) tuples.
        Row indices refer to the ORIGINAL aim_df (not filtered).
    """
    dist = aim_df["Distance on GPS Speed"].values
    lat = aim_df["GPS Latitude"].values
    speed = aim_df["GPS Speed"].values

    # Filter to moving samples
    moving = speed > min_speed_kmh
    center_lat = float(np.median(lat[moving]))

    # Detect upward crossings of median latitude
    crossings = []
    for i in range(1, len(lat)):
        if moving[i] and lat[i - 1] < center_lat <= lat[i]:
            crossings.append(i)

    if len(crossings) < 2:
        return []

    # Filter crossings to consistent longitude band
    lon = aim_df["GPS Longitude"].values
    lons_at_cross = [lon[c] for c in crossings]
    median_lon = float(np.median(lons_at_cross))
    consistent = [c for c in crossings if abs(lon[c] - median_lon) < 0.001]

    laps = []
    for j in range(len(consistent) - 1):
        start_idx = consistent[j]
        end_idx = consistent[j + 1]
        lap_dist = dist[end_idx] - dist[start_idx]
        if 500 < lap_dist < 2000:  # sanity check: FSAE laps are 800-1200m
            laps.append((start_idx, end_idx, lap_dist))

    return laps


def extract_lap_telemetry(
    aim_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    """Extract and normalize one lap of telemetry data.

    Returns a DataFrame with distance normalized to start at 0.
    """
    lap = aim_df.iloc[start_idx:end_idx].copy()
    base_dist = lap["Distance on GPS Speed"].iloc[0]
    lap["lap_distance_m"] = lap["Distance on GPS Speed"] - base_dist
    return lap.reset_index(drop=True)


def validate_simulation(
    sim_states: pd.DataFrame,
    aim_df: pd.DataFrame,
    lap_start_idx: int,
    lap_end_idx: int,
    target_pct: float = 5.0,
) -> ValidationReport:
    """Compare simulation output against one lap of real telemetry.

    Args:
        sim_states: DataFrame from SimResult.states (one lap).
        aim_df: Full AiM DataFrame.
        lap_start_idx: Row index of lap start in aim_df.
        lap_end_idx: Row index of lap end in aim_df.
        target_pct: Target relative error percentage for pass/fail.

    Returns:
        ValidationReport with all comparison metrics.
    """
    lap_telem = extract_lap_telemetry(aim_df, lap_start_idx, lap_end_idx)

    metrics = []

    # --- Lap time ---
    telem_time = float(
        aim_df["Time"].iloc[lap_end_idx] - aim_df["Time"].iloc[lap_start_idx]
    )
    sim_time = float(sim_states["segment_time_s"].sum())
    metrics.append(_metric("Lap time", "s", telem_time, sim_time, target_pct))

    # --- Mean speed (distance / time, not arithmetic mean of speed samples) ---
    telem_dist = float(
        aim_df["Distance on GPS Speed"].iloc[lap_end_idx]
        - aim_df["Distance on GPS Speed"].iloc[lap_start_idx]
    )
    telem_speed_kmh = (telem_dist / telem_time) * 3.6 if telem_time > 0 else 0.0

    sim_dist = float(sim_states["distance_m"].iloc[-1] - sim_states["distance_m"].iloc[0]
                      + sim_states.iloc[-1]["segment_time_s"] * sim_states.iloc[-1]["speed_ms"])
    # Simpler: use total track distance / total time
    sim_total_dist = float(sim_states["segment_time_s"].values @ sim_states["speed_ms"].values)
    sim_speed_kmh = (sim_total_dist / sim_time) * 3.6 if sim_time > 0 else 0.0
    metrics.append(_metric("Mean speed", "km/h", telem_speed_kmh, sim_speed_kmh, target_pct))

    # --- Peak speed ---
    telem_peak = float(lap_telem["GPS Speed"].max())
    sim_peak = float(sim_states["speed_kmh"].max())
    metrics.append(_metric("Peak speed", "km/h", telem_peak, sim_peak, 10.0))

    # --- SOC change ---
    telem_soc_start = float(aim_df["State of Charge"].iloc[lap_start_idx])
    telem_soc_end = float(aim_df["State of Charge"].iloc[lap_end_idx])
    telem_dsoc = telem_soc_start - telem_soc_end

    sim_soc_start = float(sim_states["soc_pct"].iloc[0])
    sim_soc_end = float(sim_states["soc_pct"].iloc[-1])
    sim_dsoc = sim_soc_start - sim_soc_end

    if telem_dsoc > 0.5:  # only compare if meaningful SOC change
        metrics.append(_metric("SOC consumed", "%", telem_dsoc, sim_dsoc, 15.0))

    # --- Pack voltage (mean during lap) ---
    telem_v = float(lap_telem["Pack Voltage"].mean())
    sim_v = float(sim_states["pack_voltage_v"].mean())
    metrics.append(_metric("Mean pack voltage", "V", telem_v, sim_v, target_pct))

    # --- Pack current (mean of nonzero) ---
    telem_i_values = lap_telem["Pack Current"].values
    telem_i_mean = float(np.mean(np.abs(telem_i_values[np.abs(telem_i_values) > 0.5]))) if np.any(np.abs(telem_i_values) > 0.5) else 0.0
    sim_i_mean = float(np.mean(np.abs(sim_states["pack_current_a"].values)))
    if telem_i_mean > 1.0:
        metrics.append(_metric("Mean |pack current|", "A", telem_i_mean, sim_i_mean, 20.0))

    return ValidationReport(metrics=metrics)


def validate_full_endurance(
    sim_states: pd.DataFrame,
    aim_df: pd.DataFrame,
    sim_total_time_s: float,
    sim_final_soc: float,
    sim_total_energy_kwh: float,
    sim_laps: int,
    target_pct: float = 5.0,
) -> ValidationReport:
    """Compare full-endurance simulation against complete AiM recording.

    Uses the final state of the AiM data as the reference for the full run.
    """
    metrics = []

    # --- Total driving time (excluding driver change / stopped periods) ---
    speed = aim_df["GPS Speed"].values
    dt_arr = np.diff(aim_df["Time"].values, prepend=aim_df["Time"].values[0])
    telem_driving_time = float(np.sum(dt_arr[speed > 5]))
    metrics.append(_metric("Driving time", "s", telem_driving_time, sim_total_time_s, target_pct))

    # --- Total distance ---
    telem_dist = float(aim_df["Distance on GPS Speed"].iloc[-1])
    sim_dist = float(sim_states["distance_m"].iloc[-1])
    metrics.append(_metric("Total distance", "m", telem_dist, sim_dist, target_pct))

    # --- Final SOC ---
    telem_soc_start = float(aim_df["State of Charge"].iloc[0])
    telem_soc_end = float(aim_df["State of Charge"].iloc[-1])
    telem_soc_consumed = telem_soc_start - telem_soc_end
    sim_soc_consumed = float(sim_states["soc_pct"].iloc[0]) - sim_final_soc
    metrics.append(_metric("SOC consumed", "%", telem_soc_consumed, sim_soc_consumed, 10.0))

    # --- Final temperature ---
    telem_temp_end = float(aim_df["Pack Temp"].iloc[-1])
    sim_temp_end = float(sim_states["cell_temp_c"].iloc[-1])
    metrics.append(_metric("Final cell temp", "C", telem_temp_end, sim_temp_end, 15.0))

    # --- Mean pack voltage ---
    moving = aim_df["GPS Speed"].values > 5
    telem_v = float(aim_df["Pack Voltage"][moving].mean())
    sim_v = float(sim_states["pack_voltage_v"].mean())
    metrics.append(_metric("Mean pack voltage", "V", telem_v, sim_v, target_pct))

    # --- Final pack voltage ---
    telem_v_end = float(aim_df["Pack Voltage"].iloc[-1])
    sim_v_end = float(sim_states["pack_voltage_v"].iloc[-1])
    metrics.append(_metric("Final pack voltage", "V", telem_v_end, sim_v_end, target_pct))

    # --- Mean pack current ---
    telem_i = float(np.mean(np.abs(aim_df["Pack Current"][moving].values)))
    sim_i = float(np.mean(np.abs(sim_states["pack_current_a"].values)))
    metrics.append(_metric("Mean |pack current|", "A", telem_i, sim_i, 20.0))

    # --- Energy consumed ---
    # Telemetry: integrate V*I over time
    telem_power = aim_df["Pack Voltage"].values * aim_df["Pack Current"].values
    dt = np.diff(aim_df["Time"].values, prepend=aim_df["Time"].values[0])
    telem_energy_j = float(np.sum(telem_power[telem_power > 0] * dt[telem_power > 0]))
    telem_energy_kwh = telem_energy_j / 3.6e6
    if telem_energy_kwh > 0.1:
        metrics.append(_metric("Energy consumed", "kWh", telem_energy_kwh, sim_total_energy_kwh, 15.0))

    return ValidationReport(metrics=metrics)


def _metric(
    name: str, unit: str, telem: float, sim: float, target_pct: float,
) -> ValidationMetric:
    abs_err = abs(sim - telem)
    rel_err = (abs_err / abs(telem) * 100) if abs(telem) > 1e-6 else 0.0
    return ValidationMetric(
        name=name,
        unit=unit,
        telemetry_value=telem,
        simulation_value=sim,
        absolute_error=abs_err,
        relative_error_pct=rel_err,
        target_pct=target_pct,
        passed=rel_err <= target_pct,
    )
