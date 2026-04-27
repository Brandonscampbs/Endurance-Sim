"""Time-step diagnostic for replay strategy on lap 1.

Aligns sim segment outputs to telemetry samples by lap distance and
shows per-channel comparison: speed, longitudinal G, lateral G, motor
torque, drive force vs resistance. Identifies where the sim diverges.

Output:
- results/diag_replay_overlay.png  five-channel stacked plot with residuals
- results/diag_replay_table.csv    per-segment side-by-side data
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.analysis.validation import detect_lap_boundaries  # noqa: E402

OUT = REPO / "results"
SIM_PARQUET = OUT / "brake_calib_replay_sim.parquet"
TELEM_PARQUET = OUT / "telemetry.parquet"


def main() -> None:
    sim_all = pd.read_parquet(SIM_PARQUET)
    aim = pd.read_parquet(TELEM_PARQUET)
    laps = detect_lap_boundaries(aim)

    # --- Lap 1 telem slice ---
    s_idx, e_idx, _ = laps[0]
    telem = aim.iloc[s_idx:e_idx].copy()
    d0 = telem["Distance on GPS Speed"].iloc[0]
    telem["lap_d"] = telem["Distance on GPS Speed"] - d0
    telem["v_ms"] = telem["GPS Speed"] / 3.6
    # Time relative to lap start
    telem["t_rel"] = telem["Time"] - telem["Time"].iloc[0]
    # Long-G: prefer GPS LonAcc when valid; fall back to dv/dt of GPS Speed
    telem["lon_g_telem"] = telem["GPS LonAcc"].fillna(method="ffill").fillna(0.0)
    # Lateral G: GPS LatAcc when valid, else YawRate*v/g
    yaw_rate_rad = np.deg2rad(telem["YawRate"].fillna(0.0).values)
    lat_g_yaw = (yaw_rate_rad * telem["v_ms"].values) / 9.81
    lat_g_gps = telem["GPS LatAcc"].values
    telem["lat_g_telem"] = np.where(np.isfinite(lat_g_gps), lat_g_gps, lat_g_yaw)

    # --- Lap 1 sim slice ---
    sim = sim_all[sim_all["lap"] == 0].copy()
    sim["lap_d"] = sim["distance_m"] - sim["distance_m"].iloc[0]
    # Long-G in sim from speed delta over segment time
    speed_ms = sim["speed_kmh"].values / 3.6
    seg_t = sim["segment_time_s"].values
    # Forward difference: a = dv/dt; pad first sample with 0
    dv = np.diff(speed_ms, prepend=speed_ms[0])
    dt = np.where(seg_t > 1e-6, seg_t, 1e-6)
    sim["lon_g_sim"] = (dv / dt) / 9.81
    # Lateral G from sim curvature
    sim["lat_g_sim"] = (speed_ms ** 2 * sim["curvature"].values) / 9.81

    # --- Interpolate sim onto telem lap_d for direct overlay ---
    td = telem["lap_d"].values
    sd = sim["lap_d"].values

    sim_on_telem = pd.DataFrame({
        "lap_d": td,
        "v_telem_kmh": telem["GPS Speed"].values,
        "v_sim_kmh": np.interp(td, sd, sim["speed_kmh"].values),
        "lon_g_telem": telem["lon_g_telem"].values,
        "lon_g_sim": np.interp(td, sd, sim["lon_g_sim"].values),
        "lat_g_telem": telem["lat_g_telem"].values,
        "lat_g_sim": np.interp(td, sd, sim["lat_g_sim"].values),
        "torque_telem": telem["Torque Feedback"].values,
        "torque_sim": np.interp(td, sd, sim["motor_torque_nm"].values),
        "throttle_telem_pct": telem["Throttle Pos"].values,
        "throttle_sim_frac": np.interp(td, sd, sim["throttle_pct"].values),
        "drive_force_sim_N": np.interp(td, sd, sim["drive_force_n"].values),
        "resist_force_sim_N": np.interp(td, sd, sim["resistance_force_n"].values),
        "regen_force_sim_N": np.interp(td, sd, sim["regen_force_n"].values),
        "curvature_sim": np.interp(td, sd, sim["curvature"].values),
    })
    sim_on_telem["v_residual_kmh"] = sim_on_telem["v_sim_kmh"] - sim_on_telem["v_telem_kmh"]
    sim_on_telem["lon_g_residual"] = sim_on_telem["lon_g_sim"] - sim_on_telem["lon_g_telem"]
    sim_on_telem["lat_g_residual"] = sim_on_telem["lat_g_sim"] - sim_on_telem["lat_g_telem"]
    sim_on_telem["torque_residual_Nm"] = (
        sim_on_telem["torque_sim"] - sim_on_telem["torque_telem"]
    )

    # Dump CSV
    sim_on_telem.to_csv(OUT / "diag_replay_table.csv", index=False)
    print(f"Saved per-segment table -> {OUT / 'diag_replay_table.csv'}")

    # --- Find biggest divergence regions ---
    print("\n=== Top 10 largest |speed residual| points ===")
    top = sim_on_telem.iloc[
        np.argsort(np.abs(sim_on_telem["v_residual_kmh"].values))[-10:][::-1]
    ]
    for _, row in top.iterrows():
        print(
            f"  d={row['lap_d']:5.1f}  "
            f"v_telem={row['v_telem_kmh']:5.1f}  v_sim={row['v_sim_kmh']:5.1f}  "
            f"dv={row['v_residual_kmh']:+5.1f}  "
            f"lon_g(telem={row['lon_g_telem']:+.2f}, sim={row['lon_g_sim']:+.2f})  "
            f"lat_g(telem={row['lat_g_telem']:+.2f}, sim={row['lat_g_sim']:+.2f})  "
            f"torque(telem={row['torque_telem']:5.1f}, sim={row['torque_sim']:5.1f})"
        )

    # --- Identify CONTIGUOUS error bands (residual > 5 km/h for >10 m) ---
    print("\n=== Contiguous error bands (|dv| > 5 km/h spans of >5 m) ===")
    err = sim_on_telem["v_residual_kmh"].values
    sig = np.sign(np.where(np.abs(err) > 5.0, err, 0))
    bands = []
    cur_sign = 0
    cur_start = 0
    for i in range(len(sig)):
        if sig[i] != cur_sign:
            if cur_sign != 0:
                bands.append((cur_start, i - 1, cur_sign))
            cur_sign = sig[i]
            cur_start = i
    if cur_sign != 0:
        bands.append((cur_start, len(sig) - 1, cur_sign))
    for start, end, s in bands:
        d_start = float(sim_on_telem["lap_d"].iloc[start])
        d_end = float(sim_on_telem["lap_d"].iloc[end])
        if d_end - d_start < 5:
            continue
        mean_resid = float(sim_on_telem["v_residual_kmh"].iloc[start:end + 1].mean())
        peak_resid = float(sim_on_telem["v_residual_kmh"].iloc[start:end + 1].abs().max())
        mean_lon_telem = float(sim_on_telem["lon_g_telem"].iloc[start:end + 1].mean())
        mean_lon_sim = float(sim_on_telem["lon_g_sim"].iloc[start:end + 1].mean())
        mean_lat_telem = float(sim_on_telem["lat_g_telem"].iloc[start:end + 1].mean())
        mean_lat_sim = float(sim_on_telem["lat_g_sim"].iloc[start:end + 1].mean())
        sign_label = "SIM FAST" if s > 0 else "SIM SLOW"
        print(
            f"  {sign_label:8s}  d={d_start:6.1f}-{d_end:6.1f} "
            f"({d_end-d_start:5.1f} m)  "
            f"mean dv={mean_resid:+5.1f}  peak={peak_resid:5.1f}  "
            f"lon_g telem/sim={mean_lon_telem:+.2f}/{mean_lon_sim:+.2f}  "
            f"lat_g telem/sim={mean_lat_telem:+.2f}/{mean_lat_sim:+.2f}"
        )

    # --- Plot 5-channel stacked ---
    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
    d = sim_on_telem["lap_d"].values

    axes[0].plot(d, sim_on_telem["v_telem_kmh"], "b", lw=1.0, label="Telem")
    axes[0].plot(d, sim_on_telem["v_sim_kmh"], "r", lw=1.0, label="Sim")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title(
        "Lap 1 channel-by-channel comparison  (replay)  "
        + f"RMSE={np.sqrt(np.mean(sim_on_telem['v_residual_kmh']**2)):.2f} km/h"
    )
    axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)

    axes[1].plot(d, sim_on_telem["v_residual_kmh"], "k", lw=0.8)
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].fill_between(
        d, sim_on_telem["v_residual_kmh"], 0,
        where=sim_on_telem["v_residual_kmh"] > 0, color="r", alpha=0.3,
    )
    axes[1].fill_between(
        d, sim_on_telem["v_residual_kmh"], 0,
        where=sim_on_telem["v_residual_kmh"] < 0, color="b", alpha=0.3,
    )
    axes[1].set_ylabel("dv (km/h)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(d, sim_on_telem["lon_g_telem"], "b", lw=0.7, label="Telem")
    axes[2].plot(d, sim_on_telem["lon_g_sim"], "r", lw=0.7, alpha=0.7, label="Sim")
    axes[2].axhline(0, color="grey", lw=0.5)
    axes[2].set_ylabel("Long-G (g)")
    axes[2].legend(loc="upper right"); axes[2].grid(alpha=0.3)

    axes[3].plot(d, sim_on_telem["lat_g_telem"], "b", lw=0.7, label="Telem")
    axes[3].plot(d, sim_on_telem["lat_g_sim"], "r", lw=0.7, alpha=0.7, label="Sim")
    axes[3].axhline(0, color="grey", lw=0.5)
    axes[3].set_ylabel("Lat-G (g)")
    axes[3].legend(loc="upper right"); axes[3].grid(alpha=0.3)

    axes[4].plot(d, sim_on_telem["torque_telem"], "b", lw=0.7, label="Telem TorqueFB")
    axes[4].plot(d, sim_on_telem["torque_sim"], "r", lw=0.7, alpha=0.7, label="Sim motor_torque")
    axes[4].axhline(0, color="grey", lw=0.5)
    axes[4].set_ylabel("Motor torque (Nm)")
    axes[4].legend(loc="upper right"); axes[4].grid(alpha=0.3)

    axes[5].plot(d, sim_on_telem["drive_force_sim_N"], "g", lw=0.6, label="Sim drive F")
    axes[5].plot(d, sim_on_telem["resist_force_sim_N"], "orange", lw=0.6, label="Sim resist F")
    axes[5].plot(d, sim_on_telem["regen_force_sim_N"], "purple", lw=0.6, label="Sim brake F")
    axes[5].set_ylabel("Force (N)")
    axes[5].set_xlabel("Lap distance (m)")
    axes[5].legend(loc="upper right"); axes[5].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "diag_replay_channels.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved channel plot -> {OUT / 'diag_replay_channels.png'}")


if __name__ == "__main__":
    main()
