"""Derive physics-based parameter values from telemetry.

Parameters derived:
- parasitic_drag_n: from clean coast events (no throttle, no brake, low curvature).
  Solves m*a = -(F_aero + F_rolling + F_grade + F_parasitic) for F_parasitic.
- grip_scale: ratio of demonstrated peak effective mu to Pacejka peak mu at the
  representative tire load.

Outputs to results/calibration.json so other tools can pick up the numbers.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.analysis.telemetry_analysis import extract_tire_grip_scale  # noqa: E402
from fsae_sim.analysis.validation import (  # noqa: E402
    detect_lap_boundaries,
    telemetry_speed_col,
)
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.physics_constants import AIR_DENSITY_KG_M3, GRAVITY_M_S2  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.tire_model import PacejkaTireModel  # noqa: E402


REPO_ROOT = REPO
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
CONFIG = REPO / "configs" / "ct16ev.yaml"


def closed_loop_grade_series(aim_df: pd.DataFrame) -> np.ndarray:
    """Return simulator-grade samples interpolated onto telemetry rows."""
    grade = np.zeros(len(aim_df), dtype=float)
    laps = detect_lap_boundaries(aim_df)
    if not laps:
        return grade

    track = Track.from_telemetry(df=aim_df)
    seg_dist = np.array(
        [seg.distance_start_m for seg in track.segments],
        dtype=float,
    )
    seg_grade = np.array([seg.grade for seg in track.segments], dtype=float)
    for _, (s_idx, e_idx, _) in enumerate(laps):
        lap = aim_df.iloc[s_idx:e_idx]
        lap_dist = (
            lap["Distance on GPS Speed"].values
            - lap["Distance on GPS Speed"].iloc[0]
        )
        grade[s_idx:e_idx] = np.interp(lap_dist, seg_dist, seg_grade)
    return grade


def derive_parasitic_drag(aim_df: pd.DataFrame, vehicle: VehicleConfig) -> dict:
    """Find clean coast events and back out parasitic drag from F = m*a.

    Coast event = consecutive samples with:
        Throttle Pos < 1 %, brake pressure ~ 0, |GPS LatAcc| < 0.05 g,
        speed > 25 km/h (so aero is dominant and physics is well-conditioned).

    For each coast sample we solve:
        m * a_long = -(F_aero(v) + F_rolling(v) + F_grade + F_parasitic)
    so
        F_parasitic = -m*a_long - F_aero(v) - F_rolling(v) - F_grade

    a_long is the GPS-derived longitudinal acceleration (already
    in the AiM channel ``GPS LonAcc``, in g). Use median across all
    qualifying samples — robust to one-off transients.
    """
    speed_kmh = aim_df[telemetry_speed_col(aim_df)].values
    speed_ms = speed_kmh * (1000.0 / 3600.0)
    throttle = aim_df["Throttle Pos"].values
    brake = np.maximum(
        aim_df["FBrakePressure"].values,
        aim_df["RBrakePressure"].values,
    )
    lat_g = np.abs(aim_df["GPS LatAcc"].values)
    lon_g = aim_df["GPS LonAcc"].values  # signed; coast = negative
    grade = closed_loop_grade_series(aim_df)

    # Strict coast filter
    mask = (
        (throttle < 1.0)
        & (brake < 0.5)
        & (lat_g < 0.05)
        & (speed_kmh > 25.0)
        & (lon_g < -0.02)  # actually decelerating
    )
    n = int(mask.sum())
    if n < 30:
        return {
            "n_samples": n,
            "parasitic_drag_n": float("nan"),
            "note": "not enough clean coast samples",
        }

    v = speed_ms[mask]
    a = lon_g[mask] * GRAVITY_M_S2  # m/s^2
    g_local = grade[mask]

    m = vehicle.vehicle.mass_kg
    cda = vehicle.vehicle.drag_coefficient * vehicle.vehicle.frontal_area_m2
    cla = vehicle.vehicle.downforce_coefficient
    crr = vehicle.vehicle.rolling_resistance
    rho = AIR_DENSITY_KG_M3
    g = GRAVITY_M_S2

    f_aero = 0.5 * rho * cda * v ** 2
    downforce = 0.5 * rho * cla * v ** 2
    normal = m * g + downforce
    f_rolling = normal * crr
    angle = np.arctan(g_local)
    f_grade = m * g * np.sin(angle)

    # m*a = -(f_aero + f_rolling + f_grade + f_par)
    # f_par = -m*a - f_aero - f_rolling - f_grade
    f_par = -m * a - f_aero - f_rolling - f_grade

    # Drop pathological samples (could be coast-down with regen contribution).
    f_par = f_par[np.isfinite(f_par)]
    winsor_lo = float(np.percentile(f_par, 5))
    winsor_hi = float(np.percentile(f_par, 95))
    f_par_winsor = np.clip(f_par, winsor_lo, winsor_hi)

    return {
        "n_samples": int(len(f_par)),
        "median_n": float(np.median(f_par)),
        "mean_n": float(np.mean(f_par)),
        "winsor95_mean_n": float(np.mean(f_par_winsor)),
        "winsor95_low_n": winsor_lo,
        "winsor95_high_n": winsor_hi,
        "p25_n": float(np.percentile(f_par, 25)),
        "p75_n": float(np.percentile(f_par, 75)),
        "min_speed_ms": float(v.min()),
        "max_speed_ms": float(v.max()),
    }


def derive_grip_scale(aim_df: pd.DataFrame, vehicle: VehicleConfig) -> dict:
    """Use existing helper to derive grip scale at 99 th percentile lat-g."""
    tire = PacejkaTireModel(REPO_ROOT / vehicle.tire.tir_file)
    static_load_per_tire = vehicle.vehicle.mass_kg * GRAVITY_M_S2 / 4.0

    out = {}
    for p in [50.0, 75.0, 85.0, 90.0, 95.0, 99.0]:
        out[f"p{int(p)}"] = extract_tire_grip_scale(
            aim_df,
            mass_kg=vehicle.vehicle.mass_kg,
            cla=vehicle.vehicle.downforce_coefficient,
            tire_model=tire,
            fz_representative=static_load_per_tire,
            percentile=p,
        )
    return out


def main() -> None:
    vehicle = VehicleConfig.from_yaml(str(CONFIG))
    _, aim_df = load_cleaned_csv(str(TELEM))

    print(f"Loaded telemetry: {len(aim_df)} rows")

    print("\n--- Parasitic drag derivation ---")
    parasitic = derive_parasitic_drag(aim_df, vehicle)
    for k, v in parasitic.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Grip scale derivation ---")
    grip = derive_grip_scale(aim_df, vehicle)
    for k, v in grip.items():
        print(f"  {k:5s}: grip_scale={v['grip_scale']:.4f} "
              f"effective_mu={v['effective_mu_95']:.3f} "
              f"peak_lat_g={v['peak_lat_g']:.3f}")

    out = REPO / "results" / "calibration.json"
    with out.open("w") as f:
        json.dump({"parasitic": parasitic, "grip": grip}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
