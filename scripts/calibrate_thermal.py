"""Calibrate the pack thermal forced-convection model against telemetry.

Fits ``h_static`` and ``k_v`` in ``h_eff(v) = h_static + k_v · v`` against
``CleanedEndurance.csv`` Pack Temp.  Heat-in is reconstructed from the
measured Pack Current and the calibrated pack internal resistance
``R_pack(SOC)`` (Voltt-derived, AiM-refined); heat-out is the speed-
dependent convection chord.  The fit is L2 on Pack Temp directly (not on
``dT/dt``) to avoid amplifying telemetry noise.

The fitted coefficients close audit P2 issue 4 (speed-dependent pack
convection).  Mean / peak / final-temp residual targets are taken from the
plan ``docs/superpowers/plans/2026-05-06-battery-upgrades.md``.

References:
    - Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer* 7e
      (Wiley 2011), Ch. 7 (external flow), §7.2 (flat plate).
    - Plan: ``docs/superpowers/plans/2026-05-06-battery-upgrades.md``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.data.loader import load_voltt_csv  # noqa: E402
from fsae_sim.vehicle import BatteryConfig, DischargeLimitPoint  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402

DEFAULT_TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
DEFAULT_VOLTT = (
    REPO
    / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)
DEFAULT_OUT_YAML = REPO / "configs" / "thermal.yaml"

# Bounds on the fitted parameters (W/K for h_static, W/K/(m/s) for k_v).
# Lower bound on h_static is 0.5 W/K so the optimizer doesn't collapse
# toward a degenerate adiabatic solution that the linear-in-v term then
# absorbs entirely; upper bounds are loose physical envelopes for a
# ~2 m² wetted-surface pack enclosure.
_BOUNDS_H_STATIC = (0.5, 50.0)
_BOUNDS_K_V = (0.0, 10.0)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ThermalCalibrationResult:
    """Outputs from :func:`calibrate_thermal_against_telemetry`."""

    h_static_w_per_k: float
    k_v_w_per_k_per_ms: float
    ambient_temperature_c: float
    mean_residual_k: float
    peak_residual_k: float
    final_residual_k: float
    n_samples: int


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------


def _gps_speed_ms(df: pd.DataFrame) -> np.ndarray:
    """Vehicle speed (m/s) from GPS lat/lon — independent of wheel-speed."""
    t = df["Time"].to_numpy()
    lat = df["GPS Latitude"].to_numpy()
    lon = df["GPS Longitude"].to_numpy()
    lat_m = lat * 111_111.0
    lon_m = lon * 111_111.0 * float(np.cos(np.radians(np.nanmean(lat))))
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    v_lat = np.gradient(lat_m) / dt
    v_lon = np.gradient(lon_m) / dt
    v = np.sqrt(v_lat * v_lat + v_lon * v_lon)
    return v


def _load_telemetry(csv_path: Path) -> pd.DataFrame:
    """Load CleanedEndurance.csv with AiM second-row units stripped."""
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False, skiprows=[1])
    return df.apply(pd.to_numeric, errors="coerce")


def _voltt_calibrated_model() -> BatteryModel:
    """Build a Voltt-calibrated BatteryModel for the pack resistance lookup."""
    cfg = BatteryConfig(
        cell_type="P45B",
        series=110,
        parallel=4,
        cell_voltage_min_v=2.55,
        cell_voltage_max_v=4.195,
        discharged_soc_pct=2.0,
        soc_taper_threshold_pct=85.0,
        soc_taper_rate_a_per_pct=1.0,
        discharge_limits=tuple([
            DischargeLimitPoint(30.0, 100.0),
            DischargeLimitPoint(65.0, 0.0),
        ]),
    )
    model = BatteryModel(cfg)
    model.calibrate_from_voltt(load_voltt_csv(DEFAULT_VOLTT))
    return model


# ---------------------------------------------------------------------------
# Forward thermal integrator
# ---------------------------------------------------------------------------


def _integrate_pack_temp(
    *,
    t: np.ndarray,
    v_ms: np.ndarray,
    i_pack: np.ndarray,
    soc: np.ndarray,
    pack_r_w: np.ndarray,
    thermal_mass_j_per_k: float,
    h_static: float,
    k_v: float,
    ambient_c: float,
    t0_c: float,
) -> np.ndarray:
    """Forward-Euler integrate pack temperature.

    Heat-in is ``I² · R_pack(SOC)`` (pack-level); heat-out is
    ``h_eff(v) · (T_pack − T_ambient)`` with ``h_eff(v) = h_static + k_v · |v|``.
    """
    n = len(t)
    temp = np.empty(n, dtype=float)
    temp[0] = t0_c
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0.0:
            temp[i] = temp[i - 1]
            continue
        heat_in = i_pack[i - 1] ** 2 * pack_r_w[i - 1]
        h_eff = h_static + k_v * abs(v_ms[i - 1])
        heat_out = h_eff * (temp[i - 1] - ambient_c)
        d_temp = (heat_in - heat_out) * dt / thermal_mass_j_per_k
        temp[i] = temp[i - 1] + d_temp
    return temp


def _residuals(
    params: np.ndarray,
    *,
    t: np.ndarray,
    v_ms: np.ndarray,
    i_pack: np.ndarray,
    soc: np.ndarray,
    pack_r_w: np.ndarray,
    pack_temp_meas: np.ndarray,
    thermal_mass_j_per_k: float,
    ambient_c: float,
) -> np.ndarray:
    h_static, k_v = params
    t_sim = _integrate_pack_temp(
        t=t,
        v_ms=v_ms,
        i_pack=i_pack,
        soc=soc,
        pack_r_w=pack_r_w,
        thermal_mass_j_per_k=thermal_mass_j_per_k,
        h_static=h_static,
        k_v=k_v,
        ambient_c=ambient_c,
        t0_c=float(pack_temp_meas[0]),
    )
    return t_sim - pack_temp_meas


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def calibrate_thermal_against_telemetry(
    *,
    telemetry_csv: Path = DEFAULT_TELEM,
    voltt_cell_csv: Path = DEFAULT_VOLTT,
    output_yaml: Path | None = DEFAULT_OUT_YAML,
    ambient_temperature_c: float | None = None,
) -> ThermalCalibrationResult:
    """Fit ``h_static`` and ``k_v`` against ``CleanedEndurance`` Pack Temp.

    The pack resistance ``R_pack(SOC)`` is borrowed from a Voltt-calibrated
    ``BatteryModel`` (the same model the engine uses), so the calibration
    is self-consistent with how the runtime sim computes ``I² R``.

    Ambient temperature defaults to the median Pack Temp in the first
    30 s of the recording (effectively the soak temperature before
    significant duty-cycle loading).  Override via
    ``ambient_temperature_c`` for venues where the soak window is not
    representative.

    Args:
        telemetry_csv: Path to CleanedEndurance.csv.
        voltt_cell_csv: Path to the Voltt cell simulation CSV.
        output_yaml: Optional output path for the fitted-params YAML
            snippet.  ``None`` skips writing.
        ambient_temperature_c: Optional ambient temperature override (°C).

    Returns:
        ``ThermalCalibrationResult`` with fitted parameters and residual
        diagnostics.
    """
    df = _load_telemetry(telemetry_csv)

    # Required columns; drop rows missing any of them.
    cols = ["Time", "Pack Temp", "Pack Current", "State of Charge"]
    sub = df[cols + ["GPS Latitude", "GPS Longitude"]].dropna().reset_index(drop=True)
    if len(sub) < 100:
        raise RuntimeError(
            f"Insufficient telemetry rows after dropna ({len(sub)} < 100)"
        )

    t = sub["Time"].to_numpy(dtype=float)
    pack_temp_meas = sub["Pack Temp"].to_numpy(dtype=float)
    i_pack = sub["Pack Current"].to_numpy(dtype=float)
    soc = sub["State of Charge"].to_numpy(dtype=float)
    v_ms = _gps_speed_ms(sub)

    # Pack resistance R(SOC) from the Voltt-calibrated model.
    model = _voltt_calibrated_model()
    pack_r = np.array([model.pack_resistance(float(s)) for s in soc])

    # Thermal mass = cell mass·Cp + structural mass.  Match the runtime
    # battery model so the calibrated h's flow straight into the sim.
    thermal_mass_j_per_k = float(model._thermal_mass_j_per_k)

    # Ambient: median of the first 30 s if not overridden.
    if ambient_temperature_c is None:
        soak_mask = t < (t[0] + 30.0)
        ambient_temperature_c = float(np.median(pack_temp_meas[soak_mask]))

    # Fit.
    x0 = np.array([3.0, 0.5])  # mid-range initial guess
    lo = np.array([_BOUNDS_H_STATIC[0], _BOUNDS_K_V[0]])
    hi = np.array([_BOUNDS_H_STATIC[1], _BOUNDS_K_V[1]])
    result = least_squares(
        _residuals,
        x0=x0,
        bounds=(lo, hi),
        kwargs=dict(
            t=t,
            v_ms=v_ms,
            i_pack=i_pack,
            soc=soc,
            pack_r_w=pack_r,
            pack_temp_meas=pack_temp_meas,
            thermal_mass_j_per_k=thermal_mass_j_per_k,
            ambient_c=ambient_temperature_c,
        ),
        method="trf",
        max_nfev=2000,
    )
    h_static_fit, k_v_fit = result.x

    # Forward-integrate with the fitted params for residual diagnostics.
    t_sim = _integrate_pack_temp(
        t=t,
        v_ms=v_ms,
        i_pack=i_pack,
        soc=soc,
        pack_r_w=pack_r,
        thermal_mass_j_per_k=thermal_mass_j_per_k,
        h_static=h_static_fit,
        k_v=k_v_fit,
        ambient_c=ambient_temperature_c,
        t0_c=float(pack_temp_meas[0]),
    )
    residuals = t_sim - pack_temp_meas
    mean_res = float(np.mean(np.abs(residuals)))
    peak_res = float(np.max(np.abs(residuals)))
    final_res = float(t_sim[-1] - pack_temp_meas[-1])

    out = ThermalCalibrationResult(
        h_static_w_per_k=float(h_static_fit),
        k_v_w_per_k_per_ms=float(k_v_fit),
        ambient_temperature_c=float(ambient_temperature_c),
        mean_residual_k=mean_res,
        peak_residual_k=peak_res,
        final_residual_k=final_res,
        n_samples=int(len(sub)),
    )

    if output_yaml is not None:
        _write_yaml(output_yaml, out)

    return out


def _write_yaml(path: Path, r: ThermalCalibrationResult) -> None:
    """Persist the fitted parameters as a tiny YAML snippet (no deps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Pack thermal forced-convection calibration\n"
        "# h_eff(v) = h_static + k_v * v   [W/K, m/s -> W/K]\n"
        "# Source: scripts/calibrate_thermal.py against CleanedEndurance.csv\n"
        "# Closes audit P2 issue 4.\n"
        f"h_static_w_per_k: {r.h_static_w_per_k:.6f}\n"
        f"k_v_w_per_k_per_ms: {r.k_v_w_per_k_per_ms:.6f}\n"
        f"ambient_temperature_c: {r.ambient_temperature_c:.3f}\n"
        f"# n_samples: {r.n_samples}\n"
        f"# mean_residual_k: {r.mean_residual_k:.4f}\n"
        f"# peak_residual_k: {r.peak_residual_k:.4f}\n"
        f"# final_residual_k: {r.final_residual_k:+.4f}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    result = calibrate_thermal_against_telemetry()
    print("Pack thermal forced-convection calibration")
    print(f"  telemetry rows fitted: {result.n_samples}")
    print(f"  h_static          = {result.h_static_w_per_k:.4f} W/K")
    print(f"  k_v               = {result.k_v_w_per_k_per_ms:.4f} W/K/(m/s)")
    print(f"  ambient (assumed) = {result.ambient_temperature_c:.2f} C")
    print(f"  mean |T_residual|  = {result.mean_residual_k:.3f} K (target <= 1.0)")
    print(f"  peak |T_residual|  = {result.peak_residual_k:.3f} K (target <= 5.0)")
    print(f"  final  T_residual  = {result.final_residual_k:+.3f} K (target |.| <= 1.0)")
    print(f"  output: {DEFAULT_OUT_YAML}")


if __name__ == "__main__":
    main()
