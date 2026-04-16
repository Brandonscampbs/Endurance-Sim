import numpy as np
import pandas as pd

from backend.models.track import Sector
from backend.models.validation import (
    AllLapsResponse,
    LapSummary,
    SectorComparison,
    TraceData,
    ValidationMetricResult,
    ValidationResponse,
)
from backend.services.sim_runner import get_baseline_result, get_track
from backend.services.telemetry_service import get_lap_boundaries, get_lap_data, get_telemetry
from backend.services.track_service import detect_sectors, get_track_data
from fsae_sim.analysis.validation import validate_full_endurance

_GRAVITY = 9.81


def align_traces(
    sim_dist: np.ndarray,
    sim_vals: np.ndarray,
    real_dist: np.ndarray,
    real_vals: np.ndarray,
    spacing_m: float = 1.0,
) -> TraceData:
    """Interpolate sim and real onto a shared uniform distance axis."""
    d_min = max(sim_dist[0], real_dist[0])
    d_max = min(sim_dist[-1], real_dist[-1])
    d_uniform = np.arange(d_min, d_max, spacing_m)

    sim_interp = np.interp(d_uniform, sim_dist, sim_vals)
    real_interp = np.interp(d_uniform, real_dist, real_vals)

    return TraceData(
        distance_m=[round(float(d), 1) for d in d_uniform],
        sim=[round(float(v), 3) for v in sim_interp],
        real=[round(float(v), 3) for v in real_interp],
    )


def _lap_metric(
    name: str,
    unit: str,
    real_value: float,
    sim_value: float,
    target_pct: float,
) -> ValidationMetricResult:
    """Build a ValidationMetricResult with relative error and pass/fail flag."""
    abs_err = abs(sim_value - real_value)
    rel_err = (abs_err / abs(real_value) * 100.0) if abs(real_value) > 1e-6 else 0.0
    return ValidationMetricResult(
        name=name,
        unit=unit,
        sim_value=round(sim_value, 3),
        real_value=round(real_value, 3),
        error_pct=round(rel_err, 2),
        threshold_pct=target_pct,
        passed=rel_err <= target_pct,
    )


def _compute_per_lap_metrics(
    sim_lap: pd.DataFrame,
    real_lap: pd.DataFrame,
    real_speed_col: str = "LFspeed",
) -> list[ValidationMetricResult]:
    """Compute per-lap accuracy metrics against a single lap of telemetry.

    Mirrors the spirit of ``validate_full_endurance`` but scoped to one lap so
    that the UI can show lap-specific errors instead of identical aggregates
    (fix C1). Energy uses positive-power-only integration to match the
    aggregate convention (fix C3).

    TODO: promote into ``fsae_sim.analysis.validation.validate_lap`` once the
    per-lap metric set stabilizes.
    """
    metrics: list[ValidationMetricResult] = []

    # --- Lap time ---
    sim_time = float(sim_lap["segment_time_s"].sum()) if len(sim_lap) > 0 else 0.0
    if len(real_lap) > 1:
        real_time = float(real_lap["Time"].iloc[-1] - real_lap["Time"].iloc[0])
    else:
        real_time = 0.0
    metrics.append(_lap_metric("Lap time", "s", real_time, sim_time, target_pct=5.0))

    # --- Mean speed (distance / time gives the proper time-weighted mean) ---
    if real_time > 0 and "lap_distance_m" in real_lap.columns and len(real_lap) > 1:
        real_dist = float(
            real_lap["lap_distance_m"].iloc[-1] - real_lap["lap_distance_m"].iloc[0]
        )
        real_mean_speed = (real_dist / real_time) * 3.6
    else:
        real_mean_speed = (
            float(real_lap[real_speed_col].mean()) if len(real_lap) > 0 else 0.0
        )
    sim_mean_speed = float(sim_lap["speed_kmh"].mean()) if len(sim_lap) > 0 else 0.0
    metrics.append(
        _lap_metric("Mean speed", "km/h", real_mean_speed, sim_mean_speed, target_pct=5.0)
    )

    # --- Peak speed ---
    sim_peak = float(sim_lap["speed_kmh"].max()) if len(sim_lap) > 0 else 0.0
    real_peak = float(real_lap[real_speed_col].max()) if len(real_lap) > 0 else 0.0
    metrics.append(_lap_metric("Peak speed", "km/h", real_peak, sim_peak, target_pct=10.0))

    # --- SOC consumed ---
    if len(sim_lap) > 0 and len(real_lap) > 0:
        sim_dsoc = float(sim_lap["soc_pct"].iloc[0] - sim_lap["soc_pct"].iloc[-1])
        real_dsoc = float(
            real_lap["State of Charge"].iloc[0] - real_lap["State of Charge"].iloc[-1]
        )
        if real_dsoc > 0.1:
            metrics.append(
                _lap_metric("SOC consumed", "%", real_dsoc, sim_dsoc, target_pct=15.0)
            )

    # --- Mean pack voltage ---
    if len(sim_lap) > 0 and len(real_lap) > 0:
        sim_v = float(sim_lap["pack_voltage_v"].mean())
        real_v = float(real_lap["Pack Voltage"].mean())
        metrics.append(
            _lap_metric("Mean pack voltage", "V", real_v, sim_v, target_pct=5.0)
        )

    # --- Energy consumed (positive-power-only, fix C3) ---
    if len(sim_lap) > 0 and len(real_lap) > 1:
        sim_power = sim_lap["electrical_power_w"].values
        sim_dt = sim_lap["segment_time_s"].values
        sim_energy_kwh = float(np.sum(np.maximum(sim_power, 0.0) * sim_dt)) / 3_600_000.0

        real_power = (
            real_lap["Pack Voltage"].values * real_lap["Pack Current"].values
        )
        real_dt = real_lap["Time"].diff().fillna(0).values
        real_energy_kwh = (
            float(np.sum(np.maximum(real_power, 0.0) * real_dt)) / 3_600_000.0
        )
        if real_energy_kwh > 0.01:
            metrics.append(
                _lap_metric(
                    "Energy consumed", "kWh", real_energy_kwh, sim_energy_kwh,
                    target_pct=15.0,
                )
            )

    return metrics


def _compute_sector_comparison(
    sim_states: pd.DataFrame,
    real_df: pd.DataFrame,
    sectors: list[Sector],
    lap_number: int,
) -> list[SectorComparison]:
    """Compare sim vs real timing and speed for each sector."""
    sim_lap = sim_states[sim_states["lap"] == lap_number - 1]  # 0-indexed
    comparisons = []

    # (C4) Zero-based lap distance: subtracting the first distance preserves the
    # final sample (was previously `distance_m % distance_m.max()` which dropped
    # the last point of every sector).
    if len(sim_lap) > 0:
        sim_lap_dist = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]
    else:
        sim_lap_dist = np.array([])

    num_sectors = len(sectors)
    for idx, sector in enumerate(sectors):
        is_last = idx == num_sectors - 1
        # [start, end) for all but the final sector (inclusive of end) so no
        # sample is double-counted or dropped.
        if is_last:
            sim_mask = (sim_lap_dist >= sector.start_m) & (sim_lap_dist <= sector.end_m)
            real_mask = (
                (real_df["lap_distance_m"] >= sector.start_m)
                & (real_df["lap_distance_m"] <= sector.end_m)
            )
        else:
            sim_mask = (sim_lap_dist >= sector.start_m) & (sim_lap_dist < sector.end_m)
            real_mask = (
                (real_df["lap_distance_m"] >= sector.start_m)
                & (real_df["lap_distance_m"] < sector.end_m)
            )
        sim_sec = sim_lap[sim_mask]
        real_sec = real_df[real_mask]

        sim_time = float(sim_sec["segment_time_s"].sum()) if len(sim_sec) > 0 else 0.0
        real_time = float(real_sec["Time"].diff().sum()) if len(real_sec) > 1 else 0.0
        sim_speed = float(sim_sec["speed_kmh"].mean()) if len(sim_sec) > 0 else 0.0
        # (C2) Prefer LFspeed (left-front wheel speed) over GPS Speed per CLAUDE.md.
        # Falls back to GPS Speed if LFspeed column is missing.
        real_speed_col = "LFspeed" if "LFspeed" in real_sec.columns else "GPS Speed"
        real_speed = float(real_sec[real_speed_col].mean()) if len(real_sec) > 0 else 0.0

        delta_s = sim_time - real_time
        delta_pct = (delta_s / real_time * 100) if real_time > 0 else 0.0
        speed_delta_pct = ((sim_speed - real_speed) / real_speed * 100) if real_speed > 0 else 0.0

        comparisons.append(SectorComparison(
            name=sector.name,
            sector_type=sector.sector_type,
            sim_time_s=round(sim_time, 3),
            real_time_s=round(real_time, 3),
            delta_s=round(delta_s, 3),
            delta_pct=round(delta_pct, 1),
            sim_avg_speed_kmh=round(sim_speed, 1),
            real_avg_speed_kmh=round(real_speed, 1),
            speed_delta_pct=round(speed_delta_pct, 1),
        ))

    return comparisons


def get_validation_data(lap_number: int) -> ValidationResponse:
    """Produce all validation comparison data for a single lap."""
    result = get_baseline_result()
    sim_states = result.states
    track_data = get_track_data()
    real_df = get_lap_data(lap_number)

    # Sim data for this lap (0-indexed in sim)
    sim_lap = sim_states[sim_states["lap"] == lap_number - 1].copy()
    sim_dist_in_lap = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]
    real_dist = real_df["lap_distance_m"].values

    # (C2) LFspeed (left-front wheel speed) is the preferred channel per
    # CLAUDE.md and clean_endurance_data.py. Falls back to GPS Speed if the
    # cleaned CSV does not expose LFspeed (older cleaning pipelines).
    real_speed_col = "LFspeed" if "LFspeed" in real_df.columns else "GPS Speed"

    # Build overlay traces
    speed = align_traces(
        sim_dist_in_lap, sim_lap["speed_kmh"].values,
        real_dist, real_df[real_speed_col].values,
    )
    throttle = align_traces(
        sim_dist_in_lap, sim_lap["throttle_pct"].values * 100,
        real_dist, real_df["Throttle Pos"].values,
    )

    # Normalize brake: sim is 0-1, real is bar pressure -- normalize both to 0-100
    real_brake = real_df["FBrakePressure"].values
    real_brake_max = real_brake.max() if real_brake.max() > 0 else 1.0
    brake = align_traces(
        sim_dist_in_lap, sim_lap["brake_pct"].values * 100,
        real_dist, real_brake / real_brake_max * 100,
    )

    power = align_traces(
        sim_dist_in_lap, sim_lap["electrical_power_w"].values,
        real_dist, (real_df["Pack Voltage"].values * real_df["Pack Current"].values),
    )
    soc = align_traces(
        sim_dist_in_lap, sim_lap["soc_pct"].values,
        real_dist, real_df["State of Charge"].values,
    )

    # Lateral acceleration: sim = v^2 * curvature / g, real = GPS LatAcc
    sim_lat_g = sim_lap["speed_ms"].values ** 2 * np.abs(sim_lap["curvature"].values) / _GRAVITY
    lat_accel = align_traces(
        sim_dist_in_lap, sim_lat_g,
        real_dist, np.abs(real_df["GPS LatAcc"].values),
    )

    # Track map speed coloring (interpolate speeds onto centerline points)
    cl_dists = np.array([p.distance_m for p in track_data.centerline])
    track_sim_speed = np.interp(cl_dists, sim_dist_in_lap, sim_lap["speed_kmh"].values)
    track_real_speed = np.interp(cl_dists, real_dist, real_df[real_speed_col].values)

    # Sector comparison
    sectors = _compute_sector_comparison(sim_states, real_df, track_data.sectors, lap_number)

    # (C1) Per-lap accuracy metrics. Previously this called
    # validate_full_endurance(sim_states, aim_df, ...) which produced the SAME
    # metrics for every lap (aggregate over the entire endurance). We now
    # compute per-lap metrics against sim_lap and real_df slices.
    # TODO: promote this into src/fsae_sim/analysis/validation.py as
    # validate_lap(sim_lap, real_lap, ...) once the per-lap schema is stable.
    metrics = _compute_per_lap_metrics(sim_lap, real_df, real_speed_col)

    return ValidationResponse(
        lap_number=lap_number,
        speed=speed,
        throttle=throttle,
        brake=brake,
        power=power,
        soc=soc,
        lat_accel=lat_accel,
        track_sim_speed=[round(float(v), 1) for v in track_sim_speed],
        track_real_speed=[round(float(v), 1) for v in track_real_speed],
        sectors=sectors,
        metrics=metrics,
    )


def get_all_laps_summary() -> AllLapsResponse:
    """Produce per-lap summary table and aggregate metrics."""
    result = get_baseline_result()
    sim_states = result.states
    boundaries = get_lap_boundaries()
    aim_df = get_telemetry()

    # (C2) LFspeed is the preferred wheel-speed channel per CLAUDE.md; fall back
    # to GPS Speed if the cleaned CSV has not populated it.
    real_speed_col = "LFspeed" if "LFspeed" in aim_df.columns else "GPS Speed"

    laps: list[LapSummary] = []
    for lap_idx in range(min(result.laps_completed, len(boundaries))):
        sim_lap = sim_states[sim_states["lap"] == lap_idx]
        start, end, _ = boundaries[lap_idx]
        real_lap = aim_df.iloc[start:end]

        sim_time = float(sim_lap["segment_time_s"].sum())
        real_time = float(real_lap["Time"].iloc[-1] - real_lap["Time"].iloc[0])
        time_err = abs(sim_time - real_time) / real_time * 100 if real_time > 0 else 0

        # (C3) Energy = positive-power-only V*I*dt on both sim and real to match
        # the full-endurance convention in validate_full_endurance (regen is
        # excluded from "energy consumed"). Previous signed integral made the
        # per-lap number incomparable to the all-laps aggregate.
        sim_power = sim_lap["electrical_power_w"].values
        sim_dt = sim_lap["segment_time_s"].values
        sim_energy = float(np.sum(np.maximum(sim_power, 0.0) * sim_dt)) / 3_600_000
        real_dt = real_lap["Time"].diff().fillna(0).values
        real_power = real_lap["Pack Voltage"].values * real_lap["Pack Current"].values
        real_energy = float(np.sum(np.maximum(real_power, 0.0) * real_dt)) / 3_600_000
        energy_err = abs(sim_energy - real_energy) / abs(real_energy) * 100 if real_energy != 0 else 0

        sim_mean_speed = float(sim_lap["speed_kmh"].mean())
        # (C2) Use LFspeed when available.
        real_mean_speed = float(real_lap[real_speed_col].mean())
        speed_err = abs(sim_mean_speed - real_mean_speed) / real_mean_speed * 100 if real_mean_speed > 0 else 0

        laps.append(LapSummary(
            lap_number=lap_idx + 1,
            sim_time_s=round(sim_time, 2),
            real_time_s=round(real_time, 2),
            time_error_pct=round(time_err, 1),
            sim_energy_kwh=round(sim_energy, 4),
            real_energy_kwh=round(real_energy, 4),
            energy_error_pct=round(energy_err, 1),
            mean_speed_error_pct=round(speed_err, 1),
        ))

    # Aggregate metrics
    report = validate_full_endurance(
        sim_states, aim_df,
        result.total_time_s, result.final_soc,
        result.total_energy_kwh, result.laps_completed,
    )
    metrics = [
        ValidationMetricResult(
            name=m.name, unit=m.unit,
            sim_value=round(m.simulation_value, 3),
            real_value=round(m.telemetry_value, 3),
            error_pct=round(m.relative_error_pct, 2),
            threshold_pct=m.target_pct,
            passed=m.passed,
        )
        for m in report.metrics
    ]

    return AllLapsResponse(laps=laps, metrics=metrics)
