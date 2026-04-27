"""Compare sim track curvature to telemetry pointwise curvature.

The sim builds Track from the SAME telemetry, so the question is:
how much does smoothing/binning lose vs the raw signal?
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

from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402

TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
OUT = REPO / "results"


def main() -> None:
    _, aim_df = load_cleaned_csv(str(TELEM))
    track = Track.from_telemetry(df=aim_df)

    # Sim track: distance vs curvature
    sim_dist = np.array([s.distance_start_m + s.length_m / 2.0 for s in track.segments])
    sim_kappa = np.array([abs(s.curvature) for s in track.segments])
    sim_R = np.where(sim_kappa > 1e-6, 1.0 / sim_kappa, 1e6)

    # Telemetry pointwise curvature: kappa = a_lat / v^2
    # Use first lap (0-1005 m) for comparison.
    aim_df_lap = aim_df[aim_df["Distance on GPS Speed"] < track.total_distance_m].copy()
    v_ms = aim_df_lap["GPS Speed"].values * (1000 / 3600)
    a_lat_raw = np.nan_to_num(aim_df_lap["GPS LatAcc"].values, nan=0.0)
    a_lat = a_lat_raw * 9.81
    valid = v_ms > 2.0
    kappa_telem = np.zeros(len(v_ms))
    kappa_telem[valid] = np.abs(a_lat[valid]) / (v_ms[valid] ** 2)
    R_telem = np.where(kappa_telem > 1e-6, 1.0 / kappa_telem, 1e6)
    d_telem = aim_df_lap["Distance on GPS Speed"].values

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(d_telem, kappa_telem, "-", color="#1f77b4", lw=0.8, alpha=0.6,
             label="Telemetry pointwise (a_lat/v²)")
    ax1.plot(sim_dist, sim_kappa, "-", color="#d62728", lw=1.2,
             label=f"Sim track (smooth={2.0}m)")
    ax1.set_ylabel("|curvature| (1/m)")
    ax1.set_title("Track curvature: sim vs telemetry pointwise")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(kappa_telem.max(), sim_kappa.max()) * 1.05)

    ax2.plot(d_telem, np.minimum(R_telem, 200), "-", color="#1f77b4", lw=0.8, alpha=0.6,
             label="Telemetry pointwise R (clipped 200m)")
    ax2.plot(sim_dist, np.minimum(sim_R, 200), "-", color="#d62728", lw=1.2,
             label="Sim track R")
    ax2.set_xlabel("Distance (m)"); ax2.set_ylabel("Radius (m)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "track_curvature_compare.png", dpi=140)
    plt.close(fig)

    print(f"Sim track: {len(track.segments)} segments, "
          f"max kappa={sim_kappa.max():.4f} (R_min={1/sim_kappa.max():.2f}m)")
    finite_telem = kappa_telem[kappa_telem > 0]
    print(f"Telem pointwise: max kappa={finite_telem.max():.4f} "
          f"(R_min={1/finite_telem.max():.2f}m)")
    print(f"Telem p99 kappa={np.percentile(finite_telem, 99):.4f} "
          f"(R={1/np.percentile(finite_telem, 99):.2f}m)")
    print(f"Telem p95 kappa={np.percentile(finite_telem, 95):.4f} "
          f"(R={1/np.percentile(finite_telem, 95):.2f}m)")

    # Around d=234m: find sim kappa
    near_idx = np.argmin(np.abs(sim_dist - 234.0))
    print(f"\nNear d=234m: sim segment at d={sim_dist[near_idx]:.1f}, "
          f"kappa={sim_kappa[near_idx]:.4f} (R={1/max(sim_kappa[near_idx],1e-9):.2f}m)")
    near_telem_mask = (d_telem > 230) & (d_telem < 240)
    if near_telem_mask.any():
        telem_kappa_near = kappa_telem[near_telem_mask]
        telem_kappa_near = telem_kappa_near[telem_kappa_near > 0]
        if len(telem_kappa_near):
            print(f"Near d=234m telem: max kappa={telem_kappa_near.max():.4f} "
                  f"(R={1/telem_kappa_near.max():.2f}m)")


if __name__ == "__main__":
    main()
