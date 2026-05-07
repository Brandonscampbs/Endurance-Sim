"""Iterative sim accuracy harness.

Usage:
    python scripts/sim_compare.py [--strategy replay|calibrated] [--lap N]

Outputs to results/:
- {strategy}_sim.parquet
- {strategy}_speed_lap{N}.png   single lap overlay + residual
- {strategy}_speed_full.png     full endurance overlay
- {strategy}_summary.json       totals and per-lap RMSE
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.analysis.validation import (  # noqa: E402
    detect_lap_boundaries,
    telemetry_speed_col,
    validate_full_endurance,
)
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.driver.strategies import (  # noqa: E402
    CalibratedStrategy,
    ReplayStrategy,
)
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402
from fsae_sim.vehicle.dynamics import VehicleDynamics  # noqa: E402

CONFIG = REPO / "configs" / "ct16ev.yaml"
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = (
    REPO / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)
OUT = REPO / "results"
OUT.mkdir(exist_ok=True)
SPEED_TOLERANCE_KMH = 2.0 * 3.6
TIME_TOLERANCE_S = 5.0
VALIDATION_HOLDOUT_LAPS_ZERO_BASED = tuple(range(12, 22))


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_strategy(
    name,
    aim_df,
    track,
    speed_col: str,
    *,
    dynamics: VehicleDynamics | None = None,
    trim_replay_to_lap_start: bool = True,
    calibration_laps: list[int] | None = None,
    holdout_laps: list[int] | None = None,
    use_observed_speed_caps: bool = True,
):
    if name == "driver":
        return CalibratedStrategy.from_telemetry(
            aim_df,
            track,
            speed_col=speed_col,
            holdout_laps=holdout_laps,
            use_observed_speed_caps=use_observed_speed_caps,
        )
    if name == "replay":
        return ReplayStrategy.from_full_endurance(
            aim_df,
            track.total_distance_m,
            trim_to_lap_start=trim_replay_to_lap_start,
        )
    raise ValueError(f"Unknown strategy {name!r}")


def residual_samples(states, aim_df, laps, speed_col: str):
    """Return telemetry-sampled sim residuals by lap and distance."""
    rows = []
    for i, (s_idx, e_idx, _) in enumerate(laps):
        telem_lap = aim_df.iloc[s_idx:e_idx]
        td = (
            telem_lap["Distance on GPS Speed"].values
            - telem_lap["Distance on GPS Speed"].iloc[0]
        )
        sim_lap = states[states["lap"] == i]
        if len(sim_lap) < 2:
            continue
        sd = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]
        st = sim_lap["time_s"].values - sim_lap["time_s"].iloc[0]
        tt = telem_lap["Time"].values - telem_lap["Time"].iloc[0]
        sim_on_telem = np.interp(td, sd, sim_lap["speed_kmh"].values)
        sim_time_on_telem = np.interp(td, sd, st)
        data = {
            "lap": i + 1,
            "lap_distance_m": td,
            "telem_elapsed_s": tt,
            "sim_elapsed_s": sim_time_on_telem,
            "telem_speed_kmh": telem_lap[speed_col].values,
            "sim_speed_kmh": sim_on_telem,
            "speed_residual_kmh": (
                sim_on_telem - telem_lap[speed_col].values
            ),
            "time_residual_s": sim_time_on_telem - tt,
        }
        if (
            "electrical_power_w" in sim_lap.columns
            and {"Pack Voltage", "Pack Current"}.issubset(telem_lap.columns)
        ):
            real_power = (
                telem_lap["Pack Voltage"].values
                * telem_lap["Pack Current"].values
            )
            sim_power = np.interp(td, sd, sim_lap["electrical_power_w"].values)
            data["telem_power_w"] = real_power
            data["sim_power_w"] = sim_power
            data["power_residual_w"] = sim_power - real_power
        rows.append(pd.DataFrame(data))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def per_lap_residuals(states, aim_df, laps, speed_col: str):
    """Return lap-level speed and elapsed-time residual metrics."""
    samples = residual_samples(states, aim_df, laps, speed_col)
    rows = []
    for lap_num, lap_samples in samples.groupby("lap", sort=True):
        residual = lap_samples["speed_residual_kmh"].values
        time_residual = lap_samples["time_residual_s"].values
        s_idx, e_idx, _ = laps[int(lap_num) - 1]
        telem_lap = aim_df.iloc[s_idx:e_idx]
        sim_lap = states[states["lap"] == int(lap_num) - 1]
        rows.append({
            "lap": int(lap_num),
            "rmse_kmh": float(np.sqrt(np.mean(residual ** 2))),
            "bias_kmh": float(np.mean(residual)),
            "p95_kmh": float(np.percentile(np.abs(residual), 95)),
            "max_abs_kmh": float(np.max(np.abs(residual))),
            "within_2ms_pct": float(
                100.0 * np.mean(np.abs(residual) <= SPEED_TOLERANCE_KMH)
            ),
            "time_abs_p95_s": float(
                np.percentile(np.abs(time_residual), 95)
            ),
            "time_max_abs_s": float(np.max(np.abs(time_residual))),
            "telem_time_s": float(telem_lap["Time"].iloc[-1] - telem_lap["Time"].iloc[0]),
            "sim_time_s": float(sim_lap["segment_time_s"].sum()),
            "time_delta_s": float(
                sim_lap["segment_time_s"].sum()
                - (telem_lap["Time"].iloc[-1] - telem_lap["Time"].iloc[0])
            ),
        })
    return pd.DataFrame(rows)


def overall_residual_metrics(samples: pd.DataFrame) -> dict[str, float]:
    speed_resid = samples["speed_residual_kmh"].values
    time_resid = samples["time_residual_s"].values
    metrics = {
        "speed_rmse_kmh": float(np.sqrt(np.mean(speed_resid ** 2))),
        "speed_bias_kmh": float(np.mean(speed_resid)),
        "speed_p95_kmh": float(np.percentile(np.abs(speed_resid), 95)),
        "speed_max_abs_kmh": float(np.max(np.abs(speed_resid))),
        "speed_within_2ms_pct": float(
            100.0 * np.mean(np.abs(speed_resid) <= SPEED_TOLERANCE_KMH)
        ),
        "time_abs_p95_s": float(np.percentile(np.abs(time_resid), 95)),
        "time_max_abs_s": float(np.max(np.abs(time_resid))),
        "time_within_5s_pct": float(
            100.0 * np.mean(np.abs(time_resid) <= TIME_TOLERANCE_S)
        ),
    }
    if "power_residual_w" in samples.columns:
        power_resid = samples["power_residual_w"].values
        metrics.update({
            "power_rmse_w": float(np.sqrt(np.mean(power_resid ** 2))),
            "power_mae_w": float(np.mean(np.abs(power_resid))),
            "power_bias_w": float(np.mean(power_resid)),
            "power_p95_w": float(np.percentile(np.abs(power_resid), 95)),
        })
    return metrics


def plot_lap_overlay(
    states, aim_df, laps, lap_num, strategy_name, speed_col: str, outpath,
):
    plt = _pyplot()
    s_idx, e_idx, _ = laps[lap_num - 1]
    telem_lap = aim_df.iloc[s_idx:e_idx]
    td = (
        telem_lap["Distance on GPS Speed"].values
        - telem_lap["Distance on GPS Speed"].iloc[0]
    )
    sim_lap = states[states["lap"] == lap_num - 1]
    sd = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]
    sim_on_telem = np.interp(td, sd, sim_lap["speed_kmh"].values)
    residual = sim_on_telem - telem_lap[speed_col].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.plot(
        td, telem_lap[speed_col].values,
        color="#1f77b4", lw=1.2, label=f"Telemetry ({speed_col})",
    )
    ax1.plot(td, sim_on_telem, color="#d62728", lw=1.2, label=f"Sim ({strategy_name})")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title(f"Lap {lap_num}: Speed Overlay  (strategy: {strategy_name})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(td, residual, color="#444", lw=0.9)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.fill_between(td, residual, 0, where=residual > 0, color="#d62728", alpha=0.3, label="Sim faster")
    ax2.fill_between(td, residual, 0, where=residual < 0, color="#1f77b4", alpha=0.3, label="Sim slower")
    ax2.set_xlabel("Lap distance (m)")
    ax2.set_ylabel("Sim - Telem (km/h)")
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    bias = float(np.mean(residual))
    p95 = float(np.percentile(np.abs(residual), 95))
    ax2.set_title(f"Residual  RMSE={rmse:.2f}  bias={bias:+.2f}  p95={p95:.2f} km/h")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--strategy",
        default="driver",
        choices=["driver", "replay"],
    )
    p.add_argument("--lap", type=int, default=5)
    p.add_argument("--label", default=None,
                   help="optional file prefix override (default = strategy name)")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="skip PNG plot generation for faster iterative validation runs",
    )
    p.add_argument(
        "--track-smooth-m",
        type=float,
        default=None,
        help="override per-lap GPS-coordinate smoothing distance for replay",
    )
    p.add_argument(
        "--grip-scale",
        type=float,
        default=None,
        help="override tire grip scale for replay/calibration sweeps",
    )
    p.add_argument(
        "--mode",
        choices=[m.value for m in SimulationMode],
        default=None,
        help=(
            "simulation mode. Default is replay for --strategy replay and "
            "validation for --strategy driver."
        ),
    )
    p.add_argument(
        "--use-observed-speed-caps",
        action="store_true",
        help=(
            "allow driver-strategy telemetry p95 speed caps. Intended for "
            "calibration diagnostics; validation/prediction modes reject this."
        ),
    )
    args = p.parse_args()

    label = args.label or args.strategy
    # ``replay`` defaults to REPLAY mode (engine bypasses LVCU and uses
    # recorded torque). ``driver`` is tune-aware so it runs in
    # VALIDATION mode by default.
    mode = SimulationMode.coerce(
        args.mode or (
            SimulationMode.REPLAY.value
            if args.strategy == "replay" else SimulationMode.VALIDATION.value
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quiet calibration / clamp warnings
        vehicle = VehicleConfig.from_yaml(str(CONFIG))
        if args.grip_scale is not None:
            if vehicle.tire is None:
                raise ValueError("--grip-scale requires a tire config")
            vehicle = replace(
                vehicle,
                tire=replace(vehicle.tire, grip_scale=args.grip_scale),
            )
        _, aim_df = load_cleaned_csv(str(TELEM))
        speed_col = telemetry_speed_col(aim_df)
        from fsae_sim.analysis.validation import detect_lap_boundaries
        _laps = detect_lap_boundaries(aim_df)
        validation_laps = [
            lap for lap in VALIDATION_HOLDOUT_LAPS_ZERO_BASED
            if lap < len(_laps)
        ]
        calibration_laps = [
            lap for lap in range(len(_laps)) if lap not in set(validation_laps)
        ]
        strategy_holdout_laps = (
            validation_laps if mode == SimulationMode.VALIDATION else None
        )
        battery_holdout_laps = (
            [lap + 1 for lap in validation_laps]
            if mode == SimulationMode.VALIDATION else None
        )
        use_observed_speed_caps = (
            args.use_observed_speed_caps
            if mode == SimulationMode.CALIBRATION else False
        )
        # Build the map from GPS latitude/longitude geometry. Replay uses
        # the full cleaned recording so total distance/time compare against
        # CleanedEndurance.csv rather than only the detected lap stitch.
        # Calibrated strategies remain lap-relative and use the averaged
        # GPS-coordinate centerline.
        if args.strategy == "replay":
            track_kwargs = {}
            if args.track_smooth_m is not None:
                track_kwargs["smooth_distance_m"] = args.track_smooth_m
            track = Track.from_telemetry_full_recording(
                df=aim_df, **track_kwargs,
            )
        else:
            track = Track.from_telemetry(df=aim_df)
        battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
        if battery_holdout_laps is not None:
            battery.calibrate_pack_from_telemetry(
                aim_df,
                holdout_laps=battery_holdout_laps,
            )
        else:
            battery.calibrate_pack_from_telemetry(
                aim_df,
                allow_same_run_validation=True,
            )
        strategy = build_strategy(
            args.strategy,
            aim_df,
            track,
            speed_col,
            dynamics=VehicleDynamics(vehicle.vehicle),
            # Both replay variants use the full-recording track and need
            # their distance origin to match the track's (sample 0 of
            # the recording, including pre-pit-out tail).
            trim_replay_to_lap_start=(args.strategy != "replay"),
            calibration_laps=calibration_laps if mode == SimulationMode.VALIDATION else None,
            holdout_laps=strategy_holdout_laps,
            use_observed_speed_caps=use_observed_speed_caps,
        )
        engine = SimulationEngine(vehicle, track, strategy, battery, mode=mode)

        # Use the real driver's lap-1 entry speed instead of 0 so the sim
        # doesn't ramp from 0.5 m/s through the first 50 m of the lap. Real
        # endurance starts at the start/finish gate while the car is already
        # moving (rolling start). Match that.
        initial_speed_ms = (
            float(aim_df[speed_col].iloc[_laps[0][0]]) / 3.6
            if _laps else 0.0
        )
        # Replay tracks already contain the full cleaned recording; single-lap
        # centerline tracks are repeated for the FSAE Michigan endurance lap
        # count (22 laps of ~1005 m each = ~22 km).
        #
        # ``detect_lap_boundaries`` only finds 21 intervals because the
        # recording stops 14.9 s into lap 22 before the final crossing —
        # the real event was 22 laps but lap 22 wasn't fully captured.
        # Running 21 instead of 22 caused a ~1 km / ~70 s structural
        # under-bid in canonical-lap strategies.
        is_stitched = track.source.startswith("telemetry_per_lap")
        is_full_recording = track.source.startswith("telemetry_full_recording")
        if is_stitched or is_full_recording:
            num_laps = 1  # the full-recording track encodes all laps in one pass
        else:
            num_laps = (len(_laps) + 1) if _laps else 22

        result = engine.run(
            num_laps=num_laps, initial_soc_pct=95.0, initial_temp_c=29.0,
            initial_speed_ms=initial_speed_ms,
        )

    states = result.states
    print(f"[{label}] sim laps={result.laps_completed} "
          f"time={result.total_time_s:.1f}s "
          f"net_ah={result.net_charge_ah:.2f}Ah "
          f"net_kwh={result.net_energy_kwh:.3f}")
    print(f"track_source={track.source} speed_truth={speed_col}")
    print(f"mode={mode.value} observed_speed_caps={use_observed_speed_caps}")

    states.to_parquet(OUT / f"{label}_sim.parquet")
    aim_df.to_parquet(OUT / "telemetry.parquet")

    laps = detect_lap_boundaries(aim_df)
    samples = residual_samples(states, aim_df, laps, speed_col)
    per_lap = per_lap_residuals(states, aim_df, laps, speed_col)
    overall = overall_residual_metrics(samples)
    print(per_lap.to_string(index=False))
    print(f"\nMean RMSE across laps: {per_lap['rmse_kmh'].mean():.2f} km/h")
    print(f"Mean bias across laps:  {per_lap['bias_kmh'].mean():+.2f} km/h")
    print(
        "Overall speed: "
        f"RMSE={overall['speed_rmse_kmh']:.2f} km/h, "
        f"p95={overall['speed_p95_kmh']:.2f} km/h, "
        f"within 2 m/s={overall['speed_within_2ms_pct']:.1f}%"
    )
    print(
        "Overall elapsed time by distance: "
        f"p95={overall['time_abs_p95_s']:.2f}s, "
        f"within 5s={overall['time_within_5s_pct']:.1f}%"
    )
    if "power_rmse_w" in overall:
        print(
            "Overall terminal power by distance: "
            f"RMSE={overall['power_rmse_w'] / 1000.0:.2f} kW, "
            f"MAE={overall['power_mae_w'] / 1000.0:.2f} kW, "
            f"bias={overall['power_bias_w'] / 1000.0:+.2f} kW"
        )

    # Validation
    report = validate_full_endurance(
        sim_states=states,
        aim_df=aim_df,
        sim_total_time_s=result.total_time_s,
        sim_final_soc=result.final_soc,
        sim_total_energy_kwh=result.net_energy_kwh,
        sim_laps=result.laps_completed,
        calibration_laps=calibration_laps if mode == SimulationMode.VALIDATION else None,
        validation_laps=validation_laps if mode == SimulationMode.VALIDATION else None,
    )
    print()
    print(report.summary())

    if not args.no_plots:
        plt = _pyplot()
        # Full overlay
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            aim_df["Distance on GPS Speed"].values, aim_df[speed_col].values,
            color="#1f77b4", lw=0.7, alpha=0.6,
            label=f"Telemetry ({speed_col})",
        )
        ax.plot(states["distance_m"].values, states["speed_kmh"].values,
                color="#d62728", lw=0.7, alpha=0.85, label=f"Sim ({label})")
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Speed (km/h)")
        ax.set_title(f"Full endurance speed overlay  ({label})")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"{label}_speed_full.png", dpi=140)
        plt.close(fig)

        # Generate per-lap overlay+residual plots for a representative spread.
        for lap_n in [1, 5, 10, 15, 20]:
            if lap_n <= len(laps):
                plot_lap_overlay(
                    states, aim_df, laps, lap_n, label, speed_col,
                    OUT / f"{label}_lap{lap_n}.png",
                )

    summary = {
        "strategy": label,
        "mode": mode.value,
        "uses_observed_speed_caps": use_observed_speed_caps,
        "calibration_laps_zero_based": (
            calibration_laps if mode == SimulationMode.VALIDATION else None
        ),
        "validation_laps_zero_based": (
            validation_laps if mode == SimulationMode.VALIDATION else None
        ),
        "track_source": track.source,
        "track_smooth_distance_m": (
            args.track_smooth_m
            if args.track_smooth_m is not None
            else track.provenance.get("smooth_distance_m")
        ),
        "track_gps_outlier_replacements": int(sum(
            lap.get("gps_outlier_replacements", 0)
            for lap in track.provenance.get("per_lap", [])
            if isinstance(lap, dict)
        )),
        "grip_scale": (
            float(vehicle.tire.grip_scale)
            if vehicle.tire is not None else None
        ),
        "telemetry_speed_col": speed_col,
        "speed_tolerance_kmh": SPEED_TOLERANCE_KMH,
        "time_tolerance_s": TIME_TOLERANCE_S,
        "sim_total_time_s": result.total_time_s,
        "sim_final_soc_pct": result.final_soc,
        "sim_net_ah": result.net_charge_ah,
        "telem_net_ah": report.telem_net_ah,
        "sim_net_kwh": result.net_energy_kwh,
        "telem_total_time_s": float(aim_df["Time"].iloc[-1] - aim_df["Time"].iloc[0]),
        "detected_telem_lap_time_s": float(per_lap["telem_time_s"].sum()),
        "detected_time_delta_s": float(
            result.total_time_s - per_lap["telem_time_s"].sum()
        ),
        "telem_net_kwh": report.telem_net_j / 3.6e6,
        "sim_power_alignment_max_abs_w": float(np.max(np.abs(
            states["electrical_power_w"].values
            - states["pack_voltage_v"].values * states["pack_current_a"].values
        ))),
        "mean_rmse_kmh": float(per_lap["rmse_kmh"].mean()),
        "mean_bias_kmh": float(per_lap["bias_kmh"].mean()),
        "overall": overall,
        "per_lap": per_lap.to_dict(orient="records"),
        "validation_metrics": [
            {"name": m.name, "telem": m.telemetry_value, "sim": m.simulation_value,
             "rel_err_pct": m.relative_error_pct, "passed": bool(m.passed)}
            for m in report.metrics
        ],
    }
    with (OUT / f"{label}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
