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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from fsae_sim.sim.engine import SimulationEngine  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402

CONFIG = REPO / "configs" / "ct16ev.yaml"
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = (
    REPO / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)
OUT = REPO / "results"
OUT.mkdir(exist_ok=True)


def build_strategy(name, aim_df, track, speed_col: str):
    if name == "calibrated":
        return CalibratedStrategy.from_telemetry(
            aim_df, track, speed_col=speed_col,
        )
    if name == "replay":
        return ReplayStrategy.from_full_endurance(aim_df, track.total_distance_m)
    raise ValueError(f"Unknown strategy {name!r}")


def per_lap_residuals(states, aim_df, laps, speed_col: str):
    """Return DataFrame: lap, rmse_kmh, bias_kmh, p95_kmh."""
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
        sim_on_telem = np.interp(td, sd, sim_lap["speed_kmh"].values)
        residual = sim_on_telem - telem_lap[speed_col].values
        rows.append({
            "lap": i + 1,
            "rmse_kmh": float(np.sqrt(np.mean(residual ** 2))),
            "bias_kmh": float(np.mean(residual)),
            "p95_kmh": float(np.percentile(np.abs(residual), 95)),
            "telem_time_s": float(telem_lap["Time"].iloc[-1] - telem_lap["Time"].iloc[0]),
            "sim_time_s": float(sim_lap["segment_time_s"].sum()),
        })
    return pd.DataFrame(rows)


def plot_lap_overlay(
    states, aim_df, laps, lap_num, strategy_name, speed_col: str, outpath,
):
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
    p.add_argument("--strategy", default="calibrated", choices=["calibrated", "replay"])
    p.add_argument("--lap", type=int, default=5)
    p.add_argument("--label", default=None,
                   help="optional file prefix override (default = strategy name)")
    args = p.parse_args()

    label = args.label or args.strategy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quiet calibration / clamp warnings
        vehicle = VehicleConfig.from_yaml(str(CONFIG))
        _, aim_df = load_cleaned_csv(str(TELEM))
        speed_col = telemetry_speed_col(aim_df)
        # Build the map from GPS latitude/longitude geometry. Replay uses
        # the stitched per-lap map so each LFspeed distance trace is
        # compared against that lap's own GPS-coordinate trajectory.
        # Calibrated strategies remain lap-relative and use the averaged
        # GPS-coordinate centerline.
        if args.strategy == "replay":
            track = Track.from_telemetry_per_lap(df=aim_df)
        else:
            track = Track.from_telemetry(df=aim_df)
        battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
        battery.calibrate_pack_from_telemetry(aim_df)
        strategy = build_strategy(args.strategy, aim_df, track, speed_col)
        engine = SimulationEngine(vehicle, track, strategy, battery)

        # Use the real driver's lap-1 entry speed instead of 0 so the sim
        # doesn't ramp from 0.5 m/s through the first 50 m of the lap. Real
        # endurance starts at the start/finish gate while the car is already
        # moving (rolling start). Match that.
        from fsae_sim.analysis.validation import detect_lap_boundaries
        _laps = detect_lap_boundaries(aim_df)
        initial_speed_ms = (
            float(aim_df[speed_col].iloc[_laps[0][0]]) / 3.6
            if _laps else 0.0
        )
        # Stitched replay tracks already contain every detected telemetry
        # lap; single-lap centerline tracks are repeated for each lap.
        is_stitched = track.source.startswith("telemetry_per_lap")
        num_laps = 1 if is_stitched else (len(_laps) if _laps else 22)

        result = engine.run(
            num_laps=num_laps, initial_soc_pct=95.0, initial_temp_c=29.0,
            initial_speed_ms=initial_speed_ms,
        )

    states = result.states
    print(f"[{label}] sim laps={result.laps_completed} "
          f"time={result.total_time_s:.1f}s "
          f"final_soc={result.final_soc:.1f}% "
          f"net_kwh={result.net_energy_kwh:.3f}")
    print(f"track_source={track.source} speed_truth={speed_col}")

    states.to_parquet(OUT / f"{label}_sim.parquet")
    aim_df.to_parquet(OUT / "telemetry.parquet")

    laps = detect_lap_boundaries(aim_df)
    per_lap = per_lap_residuals(states, aim_df, laps, speed_col)
    print(per_lap.to_string(index=False))
    print(f"\nMean RMSE across laps: {per_lap['rmse_kmh'].mean():.2f} km/h")
    print(f"Mean bias across laps:  {per_lap['bias_kmh'].mean():+.2f} km/h")

    # Validation
    report = validate_full_endurance(
        sim_states=states,
        aim_df=aim_df,
        sim_total_time_s=result.total_time_s,
        sim_final_soc=result.final_soc,
        sim_total_energy_kwh=result.net_energy_kwh,
        sim_laps=result.laps_completed,
    )
    print()
    print(report.summary())

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
        "track_source": track.source,
        "telemetry_speed_col": speed_col,
        "sim_total_time_s": result.total_time_s,
        "sim_final_soc_pct": result.final_soc,
        "sim_net_kwh": result.net_energy_kwh,
        "telem_total_time_s": float(aim_df["Time"].iloc[-1] - aim_df["Time"].iloc[0]),
        "telem_net_kwh": report.telem_net_j / 3.6e6,
        "mean_rmse_kmh": float(per_lap["rmse_kmh"].mean()),
        "mean_bias_kmh": float(per_lap["bias_kmh"].mean()),
        "per_lap": per_lap.to_dict(orient="records"),
        "validation_metrics": [
            {"name": m.name, "telem": m.telemetry_value, "sim": m.simulation_value,
             "rel_err_pct": m.relative_error_pct, "passed": m.passed}
            for m in report.metrics
        ],
    }
    with (OUT / f"{label}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
