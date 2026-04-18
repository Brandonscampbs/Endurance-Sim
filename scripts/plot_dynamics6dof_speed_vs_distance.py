"""Plot speed vs distance: sim (both backends) overlaid on telemetry.

Runs one lap under each backend using CalibratedStrategy.from_telemetry, then
integrates telemetry LFspeed over time to get a reference distance axis.
Saves PNG to docs/validation/2026-04-17-dynamics6dof/.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsae_sim.analysis.validation import detect_lap_boundaries
from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.driver.strategies import CalibratedStrategy, ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.vehicle import VehicleConfig

REPO = Path(__file__).parents[1]
CONFIG = REPO / "configs" / "ct16ev.yaml"
AIM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = REPO / "Real-Car-Data-And-Stats" / "About-Energy-Volt-Simulations-2025-Pack" / "2025_Pack_cell.csv"
OUT = REPO / "docs" / "validation" / "2026-04-17-dynamics6dof"


def _pick_best_lap(aim_df: pd.DataFrame, track: Track) -> tuple[int, int, float]:
    """Return (start_idx, end_idx, distance_m) for the lap closest to track
    distance with a reasonable lap time (50-120 s)."""
    laps = detect_lap_boundaries(aim_df)
    td = track.total_distance_m
    good = []
    for start, end, dist in laps:
        t = aim_df["Time"].iloc[end] - aim_df["Time"].iloc[start]
        if abs(dist - td) / td < 0.05 and 50 < t < 120:
            good.append((start, end, dist))
    if not good:
        raise RuntimeError("No clean lap in telemetry matches the extracted track")
    good.sort(key=lambda x: abs(x[2] - td))
    return good[0]


def _run(
    backend: str,
    strategy_kind: str,
    cfg: VehicleConfig,
    track: Track,
    aim_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame:
    """Run the sim for one lap under the given backend + strategy.

    strategy_kind: 'calibrated' uses CalibratedStrategy.from_telemetry;
                    'replay' uses ReplayStrategy.from_aim_data with the picked lap.
    """
    cfg_sel = replace(cfg, dynamics_backend=backend)
    battery = BatteryModel(cfg_sel.battery)
    battery.calibrate_from_voltt(load_voltt_csv(VOLTT))

    if strategy_kind == "calibrated":
        strategy = CalibratedStrategy.from_telemetry(aim_df, track)
        init_soc = 95.0
        init_temp = 25.0
        init_speed = 0.0
    elif strategy_kind == "replay":
        strategy = ReplayStrategy.from_aim_data(
            aim_df, start_idx, end_idx, track.total_distance_m,
        )
        init_soc = float(aim_df["State of Charge"].iloc[start_idx])
        init_temp = float(aim_df["Pack Temp"].iloc[start_idx])
        init_speed = float(aim_df["LFspeed"].iloc[start_idx]) / 3.6
    else:
        raise ValueError(f"Unknown strategy_kind={strategy_kind}")

    engine = SimulationEngine(cfg_sel, track, strategy, battery)
    return engine.run(
        num_laps=1, initial_soc_pct=init_soc,
        initial_temp_c=init_temp, initial_speed_ms=init_speed,
    ).states


def _telemetry_lap_speed_distance(
    aim_df: pd.DataFrame, start_idx: int, end_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the picked lap from telemetry, indexed by distance from lap start."""
    lap = aim_df.iloc[start_idx:end_idx]
    d = lap["Distance on GPS Speed"].to_numpy(dtype=float)
    d = d - d[0]
    speed_ms = lap["LFspeed"].to_numpy(dtype=float) / 3.6
    return d, speed_ms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    cfg = VehicleConfig.from_yaml(CONFIG)
    _, aim_df = load_cleaned_csv(AIM)
    track = Track.from_telemetry(df=aim_df)

    print("Picking the best clean lap from telemetry...")
    start_idx, end_idx, lap_dist = _pick_best_lap(aim_df, track)
    lap_time = float(aim_df["Time"].iloc[end_idx] - aim_df["Time"].iloc[start_idx])
    print(f"  lap: idx {start_idx}..{end_idx}  dist={lap_dist:.1f} m  t={lap_time:.2f} s")

    runs = {}
    for kind in ("calibrated", "replay"):
        for backend in ("legacy", "dynamics6dof"):
            print(f"Running {kind} / {backend}...")
            states = _run(backend, kind, cfg, track, aim_df, start_idx, end_idx)
            runs[(kind, backend)] = states

    print("Extracting telemetry lap...")
    telem_d, telem_v = _telemetry_lap_speed_distance(aim_df, start_idx, end_idx)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for ax, kind, title in (
        (axes[0], "calibrated", "CalibratedStrategy (synthesized driver model)"),
        (axes[1], "replay", "ReplayStrategy (recorded torque + brake)"),
    ):
        ax.plot(telem_d, telem_v, color="black", linewidth=1.2, alpha=0.7,
                label="Telemetry (LFspeed)")
        s_legacy = runs[(kind, "legacy")]
        s_6dof = runs[(kind, "dynamics6dof")]
        ax.plot(s_legacy["distance_m"], s_legacy["speed_ms"],
                color="tab:blue", linewidth=1.4, label="Sim: legacy")
        ax.plot(s_6dof["distance_m"], s_6dof["speed_ms"],
                color="tab:orange", linewidth=1.4, linestyle="--",
                label="Sim: dynamics6dof")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[1].set_xlabel("Distance (m)")
    fig.suptitle("Michigan 2025 — sim vs telemetry speed, two driver models")
    fig.tight_layout()

    out_path = OUT / "speed_vs_distance.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Error summary — interpolate telemetry onto sim distance axis
    from scipy.interpolate import interp1d
    telem_interp = interp1d(telem_d, telem_v, bounds_error=False, fill_value=np.nan)

    print("\nResiduals (sim - telemetry):")
    for (kind, backend), states in runs.items():
        sim_d = states["distance_m"].to_numpy()
        sim_v = states["speed_ms"].to_numpy()
        telem_v_at_sim_d = telem_interp(sim_d)
        mask = ~np.isnan(telem_v_at_sim_d)
        err = sim_v[mask] - telem_v_at_sim_d[mask]
        rms = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        print(f"  {kind:>10s} / {backend:>12s}: "
              f"RMS={rms:.3f}  MAE={mae:.3f}  mean={np.mean(err):+.3f} m/s")


if __name__ == "__main__":
    main()
