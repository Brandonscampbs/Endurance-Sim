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

from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.vehicle import VehicleConfig

REPO = Path(__file__).parents[1]
CONFIG = REPO / "configs" / "ct16ev.yaml"
AIM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = REPO / "Real-Car-Data-And-Stats" / "About-Energy-Volt-Simulations-2025-Pack" / "2025_Pack_cell.csv"
OUT = REPO / "docs" / "validation" / "2026-04-17-dynamics6dof"


def _run(backend: str, cfg: VehicleConfig, track: Track, aim_df: pd.DataFrame) -> pd.DataFrame:
    cfg_sel = replace(cfg, dynamics_backend=backend)
    battery = BatteryModel(cfg_sel.battery)
    battery.calibrate_from_voltt(load_voltt_csv(VOLTT))
    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    engine = SimulationEngine(cfg_sel, track, strategy, battery)
    return engine.run(num_laps=1, initial_soc_pct=95.0).states


def _telemetry_lap_speed_distance(aim_df: pd.DataFrame, lap_distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract one canonical lap of (distance_m, speed_ms) from telemetry.

    Integrates LFspeed (km/h -> m/s) across telemetry time. Trims to one lap
    worth of distance so the plot x-axis matches the sim's single-lap range.
    """
    t = aim_df["Time"].to_numpy(dtype=float)
    speed_kmh = aim_df["LFspeed"].to_numpy(dtype=float)
    speed_ms = speed_kmh / 3.6
    dt = np.diff(t, prepend=t[0])
    distance = np.cumsum(speed_ms * dt)
    mask = distance <= lap_distance_m
    return distance[mask], speed_ms[mask]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    cfg = VehicleConfig.from_yaml(CONFIG)
    _, aim_df = load_cleaned_csv(AIM)
    track = Track.from_telemetry(df=aim_df)

    print("Running legacy...")
    states_legacy = _run("legacy", cfg, track, aim_df)
    print("Running dynamics6dof...")
    states_6dof = _run("dynamics6dof", cfg, track, aim_df)

    print("Extracting telemetry lap...")
    telem_d, telem_v = _telemetry_lap_speed_distance(aim_df, track.total_distance_m)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(telem_d, telem_v, color="black", linewidth=1.2, alpha=0.7,
            label="Telemetry (LFspeed)")
    ax.plot(states_legacy["distance_m"], states_legacy["speed_ms"],
            color="tab:blue", linewidth=1.4, label="Sim: legacy")
    ax.plot(states_6dof["distance_m"], states_6dof["speed_ms"],
            color="tab:orange", linewidth=1.4, linestyle="--",
            label="Sim: dynamics6dof")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Michigan 2025 lap — sim vs telemetry, speed vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = OUT / "speed_vs_distance.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Error summary — interpolate telemetry onto sim distance axis
    from scipy.interpolate import interp1d
    telem_interp = interp1d(telem_d, telem_v, bounds_error=False, fill_value=np.nan)

    for name, states in (("legacy", states_legacy), ("dynamics6dof", states_6dof)):
        sim_d = states["distance_m"].to_numpy()
        sim_v = states["speed_ms"].to_numpy()
        telem_v_at_sim_d = telem_interp(sim_d)
        mask = ~np.isnan(telem_v_at_sim_d)
        err = sim_v[mask] - telem_v_at_sim_d[mask]
        rms = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        print(f"  {name}: RMS={rms:.3f} m/s  MAE={mae:.3f} m/s  mean_err={np.mean(err):+.3f} m/s")


if __name__ == "__main__":
    main()
