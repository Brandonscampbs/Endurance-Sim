"""Overlay telemetry vs sim drive torque and brake pressure vs distance.

Answers: is the replay strategy faithfully forwarding the driver inputs,
and does the sim's action / brake force match what was recorded?
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsae_sim.analysis.telemetry_analysis import _detect_lap_boundaries_safe as detect_lap_boundaries
from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel


def main():
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )

    track = Track.from_telemetry(df=aim_df)

    battery = BatteryModel(config.battery)
    battery.calibrate_from_voltt(voltt_df)

    replay = ReplayStrategy.from_full_endurance(aim_df, track.lap_distance_m)
    engine = SimulationEngine(config, track, replay, battery)

    num_laps = round(
        aim_df["Distance on GPS Speed"].iloc[-1] / track.total_distance_m
    )
    result = engine.run(
        num_laps=num_laps,
        initial_soc_pct=float(aim_df["State of Charge"].iloc[0]),
        initial_temp_c=float(aim_df["Pack Temp"].iloc[0]),
        initial_speed_ms=float(aim_df["GPS Speed"].iloc[0]) / 3.6,
    )

    sim = result.states
    sim_dist = sim["distance_m"].values

    # --- Telemetry: same channels, interpolated to sim distance ---
    telem_dist = aim_df["Distance on GPS Speed"].values
    telem_torque = aim_df["LVCU Torque Req"].values
    # Brake pressure: use max(front, rear) bar, normalized to 99th pct
    brake_raw = np.maximum(
        aim_df["FBrakePressure"].values, aim_df["RBrakePressure"].values,
    )
    brake_positive = brake_raw[brake_raw > 0]
    bmax = float(np.percentile(brake_positive, 99)) if brake_positive.size else 1.0
    bmax = max(bmax, 1.0)
    telem_brake_norm = np.clip(brake_raw / bmax, 0.0, 1.0)
    telem_speed = aim_df["GPS Speed"].values  # km/h

    # --- Sim command record ---
    # Engine records drive_force_n, regen_force_n and action; back out a
    # torque command to compare apples-to-apples with LVCU Torque Req.
    wheel_radius = 0.2042
    gear = config.powertrain.gear_ratio
    gearbox_eta = 0.97
    # Motor torque from drive_force: drive_force = motor_torque * gear * eta / r
    sim_drive_f = sim["drive_force_n"].values if "drive_force_n" in sim.columns else np.zeros(len(sim))
    sim_motor_torque = sim_drive_f * wheel_radius / (gear * gearbox_eta)
    # Sim brake fraction: regen path: regen_force_n < 0 indicates brake.
    # ControlCommand.brake_pct is not in states; approximate from regen_force
    # normalized by max regen at speed.  For visualization, show regen_force.
    sim_regen_f = sim["regen_force_n"].values if "regen_force_n" in sim.columns else np.zeros(len(sim))
    sim_speed_kmh = sim["speed_ms"].values * 3.6

    # --- Plot ---
    fig, axs = plt.subplots(
        4, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.3},
    )

    # Panel 1: speed overlay
    ax = axs[0]
    ax.plot(telem_dist, telem_speed, lw=0.5, alpha=0.7, label="Telemetry")
    ax.plot(sim_dist, sim_speed_kmh, lw=0.5, alpha=0.7, label="Sim")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed vs Distance")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: commanded motor torque overlay
    ax = axs[1]
    ax.plot(telem_dist, telem_torque, lw=0.5, alpha=0.7, label="Telemetry (LVCU Torque Req)")
    ax.plot(sim_dist, sim_motor_torque, lw=0.5, alpha=0.7, label="Sim (back-solved from drive_force)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Motor torque (Nm)")
    ax.set_title("Drive Torque vs Distance — is the replay command reaching the model?")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 3: brake pressure (normalized) vs regen force
    ax = axs[2]
    ax.plot(
        telem_dist, telem_brake_norm,
        lw=0.5, alpha=0.7, color="tab:blue", label="Telemetry brake_pct (norm to 99th pct)",
    )
    ax.set_ylabel("Brake pct [0..1]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(
        sim_dist, -sim_regen_f,
        lw=0.5, alpha=0.7, color="tab:red", label="Sim regen force (N, positive = decel)",
    )
    ax2.set_ylabel("Sim regen force (N)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("Brake input vs Sim regen response")
    ax.grid(alpha=0.3)

    # Panel 4: sim action distribution
    ax = axs[3]
    if "action" in sim.columns:
        action_colors = {"throttle": "tab:green", "brake": "tab:red", "coast": "tab:gray"}
        for act, color in action_colors.items():
            mask = sim["action"].values == act
            ax.scatter(
                sim_dist[mask], np.full(mask.sum(), 1.0),
                c=color, s=2, label=act, alpha=0.6,
            )
        ax.set_yticks([])
        ax.set_title("Sim action (throttle/brake/coast) by distance")
        ax.legend(loc="upper right", ncols=3)
    ax.set_xlabel("Cumulative distance (m)")
    ax.grid(alpha=0.3)

    out = "results/commands_overlay.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")

    # --- One-lap zoom (e.g. lap 6) to see detail ---
    boundaries = detect_lap_boundaries(aim_df)
    if len(boundaries) >= 6:
        start, end, _ = boundaries[5]
        lap_start_dist = float(aim_df["Distance on GPS Speed"].iloc[start])
        lap_end_dist = float(aim_df["Distance on GPS Speed"].iloc[end - 1])
        fig2, axs2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"hspace": 0.3})

        t_mask = (telem_dist >= lap_start_dist) & (telem_dist <= lap_end_dist)
        s_mask = (sim_dist >= lap_start_dist) & (sim_dist <= lap_end_dist)

        axs2[0].plot(telem_dist[t_mask], telem_speed[t_mask], lw=1.0, label="Telemetry")
        axs2[0].plot(sim_dist[s_mask], sim_speed_kmh[s_mask], lw=1.0, label="Sim")
        axs2[0].set_ylabel("Speed (km/h)")
        axs2[0].set_title("Lap 6 — Speed")
        axs2[0].legend()
        axs2[0].grid(alpha=0.3)

        axs2[1].plot(telem_dist[t_mask], telem_torque[t_mask], lw=1.0, label="Telem LVCU Torque")
        axs2[1].plot(sim_dist[s_mask], sim_motor_torque[s_mask], lw=1.0, label="Sim motor torque")
        axs2[1].axhline(0, color="k", lw=0.5)
        axs2[1].set_ylabel("Torque (Nm)")
        axs2[1].set_title("Lap 6 — Drive Torque")
        axs2[1].legend()
        axs2[1].grid(alpha=0.3)

        axs2[2].plot(telem_dist[t_mask], telem_brake_norm[t_mask], lw=1.0, label="Telem brake pct")
        axs2[2].plot(sim_dist[s_mask], -sim_regen_f[s_mask] / 2000.0, lw=1.0, label="Sim regen force / 2000N")
        axs2[2].set_ylabel("Brake (normalized)")
        axs2[2].set_title("Lap 6 — Brake")
        axs2[2].legend()
        axs2[2].grid(alpha=0.3)
        axs2[2].set_xlabel("Cumulative distance (m)")

        out2 = "results/commands_overlay_lap6.png"
        plt.savefig(out2, dpi=110, bbox_inches="tight")
        print(f"Saved {out2}")


if __name__ == "__main__":
    main()
