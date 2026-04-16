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
    """Summary of all validation metrics.

    D-05 fields (``telem_discharge_j`` / ``telem_regen_j`` /
    ``telem_net_j`` and the matching ``sim_*`` counterparts) expose the
    gross/regen/net energy split on both sides of the comparison.
    Earlier code only summed positive telemetry power, silently
    discarding every regen segment.
    """
    metrics: list[ValidationMetric]
    telem_discharge_j: float = 0.0
    telem_regen_j: float = 0.0
    telem_net_j: float = 0.0
    sim_discharge_j: float = 0.0
    sim_regen_j: float = 0.0
    sim_net_j: float = 0.0

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

    # --- Pack current (time-weighted mean) ---
    telem_i_values = lap_telem["Pack Current"].values
    telem_i_mean = float(np.mean(np.abs(telem_i_values[np.abs(telem_i_values) > 0.5]))) if np.any(np.abs(telem_i_values) > 0.5) else 0.0
    sim_seg_time = sim_states["segment_time_s"].values
    sim_i_mean = float(np.average(np.abs(sim_states["pack_current_a"].values),
                                  weights=sim_seg_time))
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
    *,
    sim_total_discharge_j: float | None = None,
    sim_total_regen_j: float | None = None,
    sim_total_net_j: float | None = None,
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

    # --- Mean pack current (time-weighted) ---
    # Telemetry samples are uniform in time (20Hz), so unweighted mean is
    # already time-weighted.  Sim segments are uniform in distance, so we
    # must weight by segment_time_s to get a proper time-weighted average.
    telem_i = float(np.mean(np.abs(aim_df["Pack Current"][moving].values)))
    sim_seg_time = sim_states["segment_time_s"].values
    sim_i = float(np.average(np.abs(sim_states["pack_current_a"].values),
                             weights=sim_seg_time))
    metrics.append(_metric("Mean |pack current|", "A", telem_i, sim_i, 20.0))

    # --- Energy consumed (D-05: discharge / regen / net tracked) ---
    # Telemetry: integrate V*I over time.  Previously only positive
    # power was summed, which discarded every regen segment — biasing
    # the telemetry total high vs any sim that tracks regen.  We now
    # track gross discharge, gross regen, and net, and compare net.
    telem_power = aim_df["Pack Voltage"].values * aim_df["Pack Current"].values
    dt = np.diff(aim_df["Time"].values, prepend=aim_df["Time"].values[0])
    telem_discharge_j = float(np.sum(np.maximum(telem_power, 0.0) * dt))
    telem_regen_j = float(np.sum(np.maximum(-telem_power, 0.0) * dt))
    telem_net_j = telem_discharge_j - telem_regen_j
    telem_net_kwh = telem_net_j / 3.6e6
    if telem_net_kwh > 0.1:
        metrics.append(
            _metric("Energy consumed (net)", "kWh",
                    telem_net_kwh, sim_total_energy_kwh, 15.0)
        )

    # Sim-side counters: use explicit values when the caller passes them
    # (preferred — the caller has exact bookkeeping), otherwise derive
    # from the net scalar so the field is non-zero for downstream code.
    if sim_total_net_j is None:
        sim_net_j_val = sim_total_energy_kwh * 3.6e6
    else:
        sim_net_j_val = float(sim_total_net_j)
    sim_disch_j_val = float(sim_total_discharge_j) if sim_total_discharge_j is not None else max(sim_net_j_val, 0.0)
    sim_regen_j_val = float(sim_total_regen_j) if sim_total_regen_j is not None else 0.0

    return ValidationReport(
        metrics=metrics,
        telem_discharge_j=telem_discharge_j,
        telem_regen_j=telem_regen_j,
        telem_net_j=telem_net_j,
        sim_discharge_j=sim_disch_j_val,
        sim_regen_j=sim_regen_j_val,
        sim_net_j=sim_net_j_val,
    )


@dataclass
class DriverChannelMetric:
    """Per-channel statistics from driver-channel validation."""
    name: str            # "throttle", "brake", "action"
    rmse: float          # root-mean-squared error (same units as channel)
    r_squared: float     # coefficient of determination; 1.0 = perfect
    correlation: float   # Pearson correlation; 1.0 = perfectly linear
    n_samples: int


@dataclass
class DriverChannelValidation:
    """Result of :func:`validate_driver_channels`.

    D-23 (driver-model fix campaign): compares per-sample driver
    commands from the simulation against telemetry-derived inputs.

    Channels: ``throttle`` (0-1), ``brake`` (0-1 of a fixed physical
    reference), ``action`` (THROTTLE / BRAKE / COAST classification).
    """
    throttle: DriverChannelMetric
    brake: DriverChannelMetric
    action_accuracy: float      # fraction of samples with matching action
    per_lap: pd.DataFrame       # per-lap RMSE / R² / corr per channel
    laps_used: list[int]
    n_samples: int

    def summary(self) -> str:
        lines = [
            "Driver-Channel Validation",
            "-" * 70,
            f"  Laps used: {self.laps_used}  (n={self.n_samples} samples)",
            f"  Throttle:   RMSE={self.throttle.rmse:.4f}  "
            f"R²={self.throttle.r_squared:.3f}  "
            f"corr={self.throttle.correlation:.3f}",
            f"  Brake:      RMSE={self.brake.rmse:.4f}  "
            f"R²={self.brake.r_squared:.3f}  "
            f"corr={self.brake.correlation:.3f}",
            f"  Action classification accuracy: "
            f"{self.action_accuracy * 100:.1f}%",
        ]
        return "\n".join(lines)


def _channel_stats(sim_ch: np.ndarray, telem_ch: np.ndarray,
                   name: str) -> DriverChannelMetric:
    """Compute RMSE, R², Pearson corr on two aligned 1-D arrays."""
    mask = np.isfinite(sim_ch) & np.isfinite(telem_ch)
    s = sim_ch[mask]
    t = telem_ch[mask]
    n = len(s)
    if n == 0:
        return DriverChannelMetric(name=name, rmse=float("nan"),
                                   r_squared=float("nan"),
                                   correlation=float("nan"), n_samples=0)
    rmse = float(np.sqrt(np.mean((s - t) ** 2)))
    ss_res = float(np.sum((t - s) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    # Pearson correlation; guard zero-variance input.
    if np.std(s) < 1e-12 or np.std(t) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(s, t)[0, 1])
    return DriverChannelMetric(
        name=name, rmse=rmse, r_squared=r2, correlation=corr, n_samples=n,
    )


def validate_driver_channels(
    sim_states: pd.DataFrame,
    aim_df: pd.DataFrame,
    *,
    laps: list[int] | None = None,
    throttle_threshold: float = 0.05,   # 5% of normalized scale
    brake_threshold_bar: float = 2.0,   # 2 bar on physical pressure
    brake_ref_bar: float = 30.0,        # normalize telemetry brake to this max
    throttle_col: str = "Throttle Pos",
    front_brake_col: str = "FBrakePressure",
    rear_brake_col: str = "RBrakePressure",
    distance_col: str = "Distance on GPS Speed",
    lap_col: str = "lap",
) -> DriverChannelValidation:
    """Compare sim driver commands to telemetry driver inputs per sample.

    The sim records per-segment driver commands (throttle_pct,
    brake_pct, action).  Telemetry carries pedal position and brake
    pressure at 20 Hz.  We align on per-lap distance — sim channels
    are linearly interpolated onto the telemetry distance grid of
    each lap — then compute RMSE, R², and Pearson correlation per
    channel across the aligned samples.

    Args:
        sim_states: Per-segment sim state DataFrame (must include
            ``lap``, ``distance_m``, ``throttle_pct``, ``brake_pct``,
            ``action``).
        aim_df: Telemetry DataFrame (must include a ``lap`` column
            and the pedal/brake/distance columns).
        laps: Optional lap subset (1-based).  ``None`` = all laps
            present in both frames.  Use this to measure against a
            held-out stint (e.g., laps 13-21 for Michigan 2025).
        throttle_threshold: Normalized throttle above which a sample
            is classified as THROTTLE action (matches the
            telemetry_analysis calibration threshold of 5%).
        brake_threshold_bar: Brake pressure (bar) above which a
            sample is classified as BRAKE action.
        brake_ref_bar: Physical reference used to normalize telemetry
            brake to 0-1 for RMSE comparison against sim brake_pct.
            Default 30 bar is DSS-plausible max line pressure; the
            exact value is documented as a free parameter pending
            D-08 normalization unification.
        throttle_col / front_brake_col / rear_brake_col / distance_col /
        lap_col: Column names.

    Returns:
        :class:`DriverChannelValidation` with per-channel RMSE / R²
        / Pearson corr, action classification accuracy, and per-lap
        breakdown.
    """
    required_sim = {"lap", "distance_m", "throttle_pct", "brake_pct", "action"}
    missing_sim = required_sim - set(sim_states.columns)
    if missing_sim:
        raise ValueError(f"sim_states missing columns: {missing_sim}")
    required_telem = {lap_col, distance_col, throttle_col,
                      front_brake_col, rear_brake_col}
    missing_telem = required_telem - set(aim_df.columns)
    if missing_telem:
        raise ValueError(f"aim_df missing columns: {missing_telem}")

    if laps is None:
        # Intersect laps present in both frames.
        sim_laps = set(int(x) for x in sim_states["lap"].unique())
        telem_laps = set(int(x) for x in aim_df[lap_col].unique() if x > 0)
        laps_used = sorted(sim_laps & telem_laps)
    else:
        laps_used = sorted(int(x) for x in laps)

    sim_t_all: list[np.ndarray] = []
    sim_b_all: list[np.ndarray] = []
    sim_a_all: list[np.ndarray] = []
    telem_t_all: list[np.ndarray] = []
    telem_b_all: list[np.ndarray] = []
    telem_a_all: list[np.ndarray] = []
    per_lap_rows: list[dict] = []

    for lap_num in laps_used:
        sim_lap = sim_states[sim_states["lap"] == lap_num].sort_values("distance_m")
        telem_lap = aim_df[aim_df[lap_col] == lap_num].sort_values(distance_col)
        if len(sim_lap) < 2 or len(telem_lap) < 2:
            continue

        # Normalize both distance axes to [0, lap_length) so the
        # interpolation is robust to different absolute offsets.
        sim_d = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]
        telem_d = telem_lap[distance_col].values - telem_lap[distance_col].values[0]

        sim_throttle = np.interp(telem_d, sim_d, sim_lap["throttle_pct"].values)
        sim_brake = np.interp(telem_d, sim_d, sim_lap["brake_pct"].values)
        # Action: use nearest-neighbour (categorical), not linear interp.
        sim_actions_arr = sim_lap["action"].values
        nn_idx = np.clip(np.searchsorted(sim_d, telem_d), 0, len(sim_d) - 1)
        sim_action = sim_actions_arr[nn_idx]

        telem_throttle = telem_lap[throttle_col].values / 100.0  # 0-100 → 0-1
        telem_brake_bar = np.maximum(
            telem_lap[front_brake_col].values,
            telem_lap[rear_brake_col].values,
        )
        telem_brake_norm = np.clip(telem_brake_bar / brake_ref_bar, 0.0, 1.0)

        # Telemetry action classification (same thresholds as calibration).
        telem_action = np.where(
            telem_brake_bar > brake_threshold_bar, "brake",
            np.where(telem_throttle > throttle_threshold, "throttle", "coast"),
        )

        sim_t_all.append(sim_throttle)
        sim_b_all.append(sim_brake)
        sim_a_all.append(sim_action)
        telem_t_all.append(telem_throttle)
        telem_b_all.append(telem_brake_norm)
        telem_a_all.append(telem_action)

        # Per-lap stats
        t_stat = _channel_stats(sim_throttle, telem_throttle, "throttle")
        b_stat = _channel_stats(sim_brake, telem_brake_norm, "brake")
        a_acc = float(np.mean(sim_action == telem_action)) if len(sim_action) else float("nan")
        per_lap_rows.append({
            "lap": lap_num,
            "n_samples": len(telem_d),
            "throttle_rmse": t_stat.rmse,
            "throttle_r2": t_stat.r_squared,
            "throttle_corr": t_stat.correlation,
            "brake_rmse": b_stat.rmse,
            "brake_r2": b_stat.r_squared,
            "brake_corr": b_stat.correlation,
            "action_accuracy": a_acc,
        })

    if not sim_t_all:
        empty = DriverChannelMetric(name="", rmse=float("nan"),
                                    r_squared=float("nan"),
                                    correlation=float("nan"), n_samples=0)
        return DriverChannelValidation(
            throttle=empty, brake=empty, action_accuracy=float("nan"),
            per_lap=pd.DataFrame(per_lap_rows), laps_used=laps_used,
            n_samples=0,
        )

    sim_t = np.concatenate(sim_t_all)
    sim_b = np.concatenate(sim_b_all)
    sim_a = np.concatenate(sim_a_all)
    telem_t = np.concatenate(telem_t_all)
    telem_b = np.concatenate(telem_b_all)
    telem_a = np.concatenate(telem_a_all)

    throttle_stat = _channel_stats(sim_t, telem_t, "throttle")
    brake_stat = _channel_stats(sim_b, telem_b, "brake")
    action_acc = float(np.mean(sim_a == telem_a))

    return DriverChannelValidation(
        throttle=throttle_stat,
        brake=brake_stat,
        action_accuracy=action_acc,
        per_lap=pd.DataFrame(per_lap_rows),
        laps_used=laps_used,
        n_samples=len(sim_t),
    )


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
