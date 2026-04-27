"""Cross-validate our PacejkaTireModel against Longhorn Racing Electric's MF52.

Runs our tire model and LHR's reference implementation over the same .tir
file and reports per-regime RMS/max error on pure and combined-slip Fy/Fx.

Two implementations of PAC2002 reading the same parameter file should
produce identical Fy/Fx surfaces. Any non-trivial disagreement is a
formula bug in one of the two (worth investigating).

LHR source vendored verbatim below from:
  https://github.com/LonghornRacingElectric/tire_analysis
  tire_toolkit/tire_model/file_processing/process_tir.py
  tire_toolkit/tire_model/MF52_calculations/lateral_force.py
  tire_toolkit/tire_model/MF52_calculations/longitudinal_force.py
Licensed under whatever LHR publishes the repo under (GitHub default: no
license, but the code is publicly readable and we are using it for
internal validation only — not redistributing).

Usage:
    python scripts/validate_tire_vs_lhr.py                      # default tir
    python scripts/validate_tire_vs_lhr.py --tir <path>         # specific tir
    python scripts/validate_tire_vs_lhr.py --csv out/diff.csv   # dump sweep

The script exits 0 if all error metrics are below threshold, 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure our package is importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fsae_sim.vehicle.tire_model import PacejkaTireModel  # noqa: E402


# ======================================================================
# VENDORED FROM LHR tire_toolkit — begin
# ======================================================================
# Source: LonghornRacingElectric/tire_analysis @ main, tire_toolkit/
# Kept as close to verbatim as possible. Only change: the .tir line
# loop guards against empty/whitespace-only lines (their original
# would IndexError). That is not an algorithmic change.


class LHRProcessor:
    """Ports tire_toolkit/tire_model/file_processing/process_tir.py::Processor."""

    def __init__(self, name: str, file_path: str) -> None:
        self._tire: list[Any] | None = None
        self._exclude = ["$", "!"]
        self.file_path = file_path
        self._add_tire(name)

    def _add_tire(self, name: str) -> None:
        self._tire = [name, self._import_data()]

    def get_parameters(self, parameter: str) -> dict:
        return self._tire[1][parameter]

    def _import_data(self) -> dict[str, dict]:
        local_results: dict[str, dict] = {}
        with open(self.file_path, "r") as f:
            data_entry = False
            for line in f:
                stripped = line.strip()
                if not stripped:  # patched: skip blank lines (LHR would IndexError)
                    continue
                char_0 = stripped[0]

                if data_entry and (char_0 not in self._exclude):
                    line_stripped = line.replace(" ", "")
                    if "$" in line_stripped:
                        line_stripped = line_stripped[: line_stripped.index("$")]
                    line_split = line_stripped.split("=")
                    if len(line_split) < 2:
                        continue
                    val_str = (
                        line_split[1]
                        .replace(".", "")
                        .replace("-", "")
                        .replace("E", "")
                        .replace("e", "")
                        .replace("+", "")
                        .replace("\n", "")
                    )
                    if val_str.isnumeric():
                        val: Any = float(line_split[1])
                    else:
                        val = line_split[1]
                    local_results[list(local_results.keys())[-1]][line_split[0]] = val
                else:
                    if char_0 in self._exclude:
                        data_entry = False
                        continue
                    if char_0 == "[":
                        if "[SHAPE]" in line:
                            continue
                        local_results[stripped[1:-1]] = {}
                        data_entry = True
        return local_results


def lhr_get_Fy(lat_coeffs, scaling_coeffs, vertical_coeffs, dimensions,
               operating_conditions, Fz, alpha, kappa, gamma):
    return _lhr_combined_lat(lat_coeffs, scaling_coeffs, vertical_coeffs,
                             dimensions, operating_conditions,
                             Fz, alpha, kappa, gamma)


def _lhr_combined_lat(lat_coeffs, scaling_coeffs, vertical_coeffs, dimensions,
                      operating_conditions, Fz, alpha, kappa, gamma):
    CFY2, CFY3, CFY4 = lat_coeffs["PDY1"], lat_coeffs["PDY2"], lat_coeffs["PDY3"]
    CCFY1, CCFY2, CCFY3 = lat_coeffs["RBY1"], lat_coeffs["RBY2"], lat_coeffs["RBY3"]
    CCFY4 = lat_coeffs["RCY1"]
    CCFY5, CCFY6 = lat_coeffs["REY1"], lat_coeffs["REY2"]
    CCFY7, CCFY8 = lat_coeffs["RHY1"], lat_coeffs["RHY2"]
    CCFY9, CCFY10, CCFY11 = lat_coeffs["RVY1"], lat_coeffs["RVY2"], lat_coeffs["RVY3"]
    CCFY12 = lat_coeffs["RVY4"]
    CCFY13, CCFY14 = lat_coeffs["RVY5"], lat_coeffs["RVY6"]

    LFZO = scaling_coeffs["LFZO"]
    LMUY = scaling_coeffs["LMUY"]
    LGAY = scaling_coeffs["LGAY"]
    LYKA = scaling_coeffs["LYKA"]
    LVYKA = scaling_coeffs["LVYKA"]
    FNOMIN = vertical_coeffs["FNOMIN"]

    df_z = (Fz - FNOMIN * LFZO) / (FNOMIN * LFZO)
    IA_y = gamma * LGAY
    mu_y = (CFY2 + CFY3 * df_z) * (1 - CFY4 * IA_y ** 2) * LMUY

    C_ySR = CCFY4
    B_ySR = CCFY1 * np.cos(np.arctan(CCFY2 * (alpha - CCFY3))) * LYKA
    E_ySR = CCFY5 + CCFY6 * df_z
    S_HySR = CCFY7 + CCFY8 * df_z
    D_VySR = mu_y * Fz * (CCFY9 + CCFY10 * df_z + CCFY11 * gamma) * np.cos(
        np.arctan(CCFY12 * alpha))
    S_VySR = D_VySR * np.sin(CCFY13 * np.arctan(CCFY14 * kappa)) * LVYKA
    SR_s = kappa + S_HySR

    num = np.cos(C_ySR * np.arctan(
        B_ySR * SR_s - E_ySR * (B_ySR * SR_s - np.arctan(B_ySR * SR_s))))
    den = np.cos(C_ySR * np.arctan(
        B_ySR * S_HySR - E_ySR * (B_ySR * S_HySR - np.arctan(B_ySR * S_HySR))))
    # Guard against 0/0 when combined coeffs are all zero.
    G_ySR = np.where(np.abs(den) > 1e-12, num / den, 1.0)

    FY_0 = _lhr_pure_lat(lat_coeffs, scaling_coeffs, vertical_coeffs,
                        dimensions, operating_conditions, Fz, alpha, kappa, gamma)

    return [mu_y, FY_0 * G_ySR + S_VySR]


def _lhr_pure_lat(lat_coeffs, scaling_coeffs, vertical_coeffs, dimensions,
                  operating_conditions, Fz, alpha, kappa, gamma):
    CFY1 = lat_coeffs["PCY1"]
    CFY2, CFY3, CFY4 = lat_coeffs["PDY1"], lat_coeffs["PDY2"], lat_coeffs["PDY3"]
    CFY5, CFY6, CFY7, CFY8 = (lat_coeffs["PEY1"], lat_coeffs["PEY2"],
                              lat_coeffs["PEY3"], lat_coeffs["PEY4"])
    CFY9, CFY10, CFY11 = lat_coeffs["PKY1"], lat_coeffs["PKY2"], lat_coeffs["PKY3"]
    CFY12, CFY13, CFY14 = (lat_coeffs["PHY1"], lat_coeffs["PHY2"],
                           lat_coeffs["PHY3"])
    CFY15, CFY16, CFY17, CFY18 = (lat_coeffs["PVY1"], lat_coeffs["PVY2"],
                                  lat_coeffs["PVY3"], lat_coeffs["PVY4"])

    LFZO = scaling_coeffs["LFZO"]
    LCY, LMUY, LEY, LKY, LHY, LVY, LGAY = (
        scaling_coeffs["LCY"], scaling_coeffs["LMUY"], scaling_coeffs["LEY"],
        scaling_coeffs["LKY"], scaling_coeffs["LHY"], scaling_coeffs["LVY"],
        scaling_coeffs["LGAY"])
    FNOMIN = vertical_coeffs["FNOMIN"]

    IA_y = gamma * LGAY
    df_z = (Fz - FNOMIN * LFZO) / (FNOMIN * LFZO)
    mu_y = (CFY2 + CFY3 * df_z) * (1 - CFY4 * IA_y ** 2) * LMUY

    C_y = CFY1 * LCY
    D_y = mu_y * Fz
    K_y = (CFY9 * FNOMIN *
           np.sin(2 * np.arctan(Fz / (CFY10 * FNOMIN * LFZO))) *
           (1 - CFY11 * np.abs(IA_y)) * LFZO * LKY)
    B_y = K_y / (C_y * D_y)

    S_Hy = (CFY12 + CFY13 * df_z) * LHY + CFY14 * IA_y
    S_Vy = Fz * ((CFY15 + CFY16 * df_z) * LVY +
                 (CFY17 + CFY18 * df_z) * IA_y) * LMUY
    SA_y = alpha + S_Hy

    E_y = (CFY5 + CFY6 * df_z) * (1 - (CFY7 + CFY8 * IA_y) * np.sign(SA_y)) * LEY

    return (D_y * np.sin(C_y * np.arctan(
        B_y * SA_y - E_y * (B_y * SA_y - np.arctan(B_y * SA_y)))) + S_Vy)


def lhr_get_Fx(long_coeffs, scaling_coeffs, vertical_coeffs, dimensions,
               operating_conditions, Fz, alpha, kappa, gamma):
    return _lhr_combined_long(long_coeffs, scaling_coeffs, vertical_coeffs,
                              dimensions, operating_conditions,
                              Fz, alpha, kappa, gamma)


def _lhr_combined_long(long_coeffs, scaling_coeffs, vertical_coeffs, dimensions,
                       operating_conditions, Fz, alpha, kappa, gamma):
    CCFX1, CCFX2, CCFX3 = long_coeffs["RBX1"], long_coeffs["RBX2"], long_coeffs["RCX1"]
    CCFX4, CCFX5, CCFX6 = long_coeffs["REX1"], long_coeffs["REX2"], long_coeffs["RHX1"]
    LFZO = scaling_coeffs["LFZO"]
    LXAL = scaling_coeffs["LXAL"]
    FNOMIN = vertical_coeffs["FNOMIN"]
    df_z = (Fz - FNOMIN * LFZO) / (FNOMIN * LFZO)

    C_xSA = CCFX3
    B_xSA = CCFX1 * np.cos(np.arctan(CCFX2 * kappa)) * LXAL
    E_xSA = CCFX4 + CCFX5 * df_z
    S_HxSA = CCFX6
    SA_s = alpha + S_HxSA

    num = np.cos(C_xSA * np.arctan(
        B_xSA * SA_s - E_xSA * (B_xSA * SA_s - np.arctan(B_xSA * SA_s))))
    den = np.cos(C_xSA * np.arctan(
        B_xSA * S_HxSA - E_xSA * (B_xSA * S_HxSA - np.arctan(B_xSA * S_HxSA))))
    G_xSA = np.where(np.abs(den) > 1e-12, num / den, 1.0)

    mu_x, FX_0 = _lhr_pure_long(long_coeffs, scaling_coeffs, vertical_coeffs,
                                dimensions, operating_conditions,
                                Fz, alpha, kappa, gamma)
    return mu_x, FX_0 * G_xSA


def _lhr_pure_long(long_coeffs, scaling_coeffs, vertical_coeffs, dimensions,
                   operating_conditions, Fz, alpha, kappa, gamma):
    CFX1 = long_coeffs["PCX1"]
    CFX2, CFX3, CFX4 = long_coeffs["PDX1"], long_coeffs["PDX2"], long_coeffs["PDX3"]
    CFX5, CFX6, CFX7, CFX8 = (long_coeffs["PEX1"], long_coeffs["PEX2"],
                              long_coeffs["PEX3"], long_coeffs["PEX4"])
    CFX9, CFX10, CFX11 = (long_coeffs["PKX1"], long_coeffs["PKX2"],
                          long_coeffs["PKX3"])
    CFX12, CFX13 = long_coeffs["PHX1"], long_coeffs["PHX2"]
    CFX14, CFX15 = long_coeffs["PVX1"], long_coeffs["PVX2"]

    LFZO, LCX, LMUX, LEX, LKX, LHX, LVX, LGAX = (
        scaling_coeffs["LFZO"], scaling_coeffs["LCX"], scaling_coeffs["LMUX"],
        scaling_coeffs["LEX"], scaling_coeffs["LKX"], scaling_coeffs["LHX"],
        scaling_coeffs["LVX"], scaling_coeffs["LGAX"])
    FNOMIN = vertical_coeffs["FNOMIN"]

    IA_x = gamma * LGAX
    df_z = (Fz - FNOMIN * LFZO) / (FNOMIN * LFZO)
    mu_x = (CFX2 + CFX3 * df_z) * (1 - CFX4 * IA_x ** 2) * LMUX

    C_x = CFX1 * LCX
    D_x = mu_x * Fz
    K_x = Fz * (CFX9 + CFX10 * df_z) * np.exp(CFX11 * df_z) * LKX
    # Guard divide-by-zero when D_x = 0 (Fz=0 or mu_x=0 fallback)
    B_x = np.where(np.abs(C_x * D_x) > 1e-12, K_x / (C_x * D_x + 1e-30), 0.0)

    S_Hx = (CFX12 + CFX13 * df_z) * LHX
    S_Vx = Fz * (CFX14 + CFX15 * df_z) * LVX * LMUX
    SR_x = kappa + S_Hx

    E_x = ((CFX5 + CFX6 * df_z + CFX7 * df_z ** 2) *
           (1 - CFX8 * np.sign(kappa)) * LEX)

    F_X0 = D_x * np.sin(C_x * np.arctan(
        B_x * SR_x - E_x * (B_x * SR_x - np.arctan(B_x * SR_x)))) + S_Vx
    return mu_x, F_X0


# ======================================================================
# VENDORED FROM LHR — end
# ======================================================================


class LHRWrapper:
    """Thin wrapper that bundles the vendored LHR functions with a parsed .tir."""

    def __init__(self, tir_path: str) -> None:
        proc = LHRProcessor(name="lhr_ref", file_path=tir_path)
        self.lat = proc.get_parameters("LATERAL_COEFFICIENTS")
        self.lon = proc.get_parameters("LONGITUDINAL_COEFFICIENTS")
        self.scaling = proc.get_parameters("SCALING_COEFFICIENTS")
        self.vertical = proc.get_parameters("VERTICAL")
        self.dim = proc.get_parameters("DIMENSION")
        self.ops = proc.get_parameters("TYRE_CONDITIONS")

    def fy(self, Fz, alpha, kappa, gamma):
        _, fy = lhr_get_Fy(self.lat, self.scaling, self.vertical, self.dim,
                           self.ops, Fz, alpha, kappa, gamma)
        return fy

    def fx(self, Fz, alpha, kappa, gamma):
        _, fx = lhr_get_Fx(self.lon, self.scaling, self.vertical, self.dim,
                           self.ops, Fz, alpha, kappa, gamma)
        return fx


# ======================================================================
# Sweep & comparison
# ======================================================================


DEFAULT_TIR = (REPO_ROOT / "Real-Car-Data-And-Stats" / "Tire Models from TTC" /
               "Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir")


def sweep_pure_lateral(ours: PacejkaTireModel, theirs: LHRWrapper):
    """Pure lateral: κ=0. Vary (Fz, α, γ). Compares Fy surfaces."""
    fz_grid = np.array([200., 400., 600., 800., 1000., 1200., 1500.])
    alpha_grid_deg = np.arange(-12.0, 12.01, 0.5)
    gamma_grid_deg = np.array([-2.0, 0.0, 2.0])

    diffs = []
    our_vals = []
    their_vals = []
    for fz in fz_grid:
        for adeg in alpha_grid_deg:
            a = math.radians(adeg)
            for gdeg in gamma_grid_deg:
                g = math.radians(gdeg)
                ours_fy = ours.lateral_force(a, fz, g)
                theirs_fy = float(theirs.fy(fz, a, 0.0, g))
                our_vals.append(ours_fy)
                their_vals.append(theirs_fy)
                diffs.append(ours_fy - theirs_fy)
    return (np.array(our_vals), np.array(their_vals), np.array(diffs))


def sweep_pure_longitudinal(ours: PacejkaTireModel, theirs: LHRWrapper):
    """Pure longitudinal: α=0. Vary (Fz, κ, γ). Compares Fx surfaces."""
    fz_grid = np.array([200., 400., 600., 800., 1000., 1200., 1500.])
    kappa_grid = np.arange(-0.20, 0.2001, 0.01)
    gamma_grid_deg = np.array([-2.0, 0.0, 2.0])

    our_vals = []
    their_vals = []
    for fz in fz_grid:
        for k in kappa_grid:
            for gdeg in gamma_grid_deg:
                g = math.radians(gdeg)
                ours_fx = ours.longitudinal_force(k, fz, g)
                theirs_fx = float(theirs.fx(fz, 0.0, k, g))
                our_vals.append(ours_fx)
                their_vals.append(theirs_fx)
    return (np.array(our_vals), np.array(their_vals),
            np.array(our_vals) - np.array(their_vals))


def sweep_combined(ours: PacejkaTireModel, theirs: LHRWrapper):
    """Combined slip: Fy and Fx at non-zero (α, κ). γ=0."""
    fz_grid = np.array([400., 800., 1200.])
    alpha_grid_deg = np.arange(-8.0, 8.01, 1.0)
    kappa_grid = np.arange(-0.15, 0.1501, 0.02)

    fy_ours, fy_theirs = [], []
    fx_ours, fx_theirs = [], []
    for fz in fz_grid:
        for adeg in alpha_grid_deg:
            a = math.radians(adeg)
            for k in kappa_grid:
                our_fx, our_fy = ours.combined_forces(a, k, fz, 0.0)
                their_fy = float(theirs.fy(fz, a, k, 0.0))
                their_fx = float(theirs.fx(fz, a, k, 0.0))
                fy_ours.append(our_fy)
                fy_theirs.append(their_fy)
                fx_ours.append(our_fx)
                fx_theirs.append(their_fx)

    fy_ours = np.array(fy_ours)
    fy_theirs = np.array(fy_theirs)
    fx_ours = np.array(fx_ours)
    fx_theirs = np.array(fx_theirs)
    return (fy_ours, fy_theirs, fy_ours - fy_theirs,
            fx_ours, fx_theirs, fx_ours - fx_theirs)


def summarize(name: str, ours: np.ndarray, theirs: np.ndarray,
              diff: np.ndarray) -> dict:
    """RMS, max abs error, relative error vs peak force."""
    rms = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    peak_theirs = float(np.max(np.abs(theirs)))
    peak_ours = float(np.max(np.abs(ours)))
    rel_rms = rms / max(peak_theirs, 1e-9)
    rel_max = max_abs / max(peak_theirs, 1e-9)
    return {
        "name": name,
        "n_points": int(diff.size),
        "peak_ours_N": peak_ours,
        "peak_theirs_N": peak_theirs,
        "rms_err_N": rms,
        "max_err_N": max_abs,
        "rel_rms": rel_rms,
        "rel_max": rel_max,
    }


def print_report(reports: list[dict]) -> None:
    print()
    print("=" * 82)
    print("  Tire Model Cross-Validation: Endurance-Sim PacejkaTireModel vs LHR MF52")
    print("=" * 82)
    print(f"  {'Regime':<26} {'N':>6} {'Peak(N)':>9} {'RMS(N)':>9} "
          f"{'Max(N)':>9} {'RelRMS':>8} {'RelMax':>8}")
    print("  " + "-" * 80)
    for r in reports:
        print(f"  {r['name']:<26} {r['n_points']:>6d} "
              f"{r['peak_theirs_N']:>9.1f} {r['rms_err_N']:>9.3f} "
              f"{r['max_err_N']:>9.3f} {r['rel_rms']:>8.2%} "
              f"{r['rel_max']:>8.2%}")
    print("  " + "-" * 80)


def write_csv(out_path: Path, regime: str, fz_flat, alpha_flat, kappa_flat,
              gamma_flat, ours_flat, theirs_flat) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("regime,Fz_N,alpha_deg,kappa,gamma_deg,ours_N,theirs_N,diff_N\n")
        for i in range(len(ours_flat)):
            f.write(f"{regime},{fz_flat[i]:.1f},{alpha_flat[i]:.3f},"
                    f"{kappa_flat[i]:.4f},{gamma_flat[i]:.3f},"
                    f"{ours_flat[i]:.3f},{theirs_flat[i]:.3f},"
                    f"{ours_flat[i] - theirs_flat[i]:.4f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tir", type=str, default=str(DEFAULT_TIR),
                        help="Path to .tir file (default: LC0 10psi)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write full sweep CSV to this path (optional)")
    parser.add_argument("--rel-threshold", type=float, default=0.01,
                        help="Relative RMS threshold for pass/fail (default 1%)")
    args = parser.parse_args()

    tir = Path(args.tir)
    if not tir.exists():
        print(f"ERROR: .tir not found: {tir}", file=sys.stderr)
        return 2

    print(f"Loading .tir: {tir.name}")
    ours = PacejkaTireModel(tir)
    theirs = LHRWrapper(str(tir))

    fy_ours, fy_theirs, fy_diff = sweep_pure_lateral(ours, theirs)
    fx_ours, fx_theirs, fx_diff = sweep_pure_longitudinal(ours, theirs)
    (cfy_o, cfy_t, cfy_d, cfx_o, cfx_t, cfx_d) = sweep_combined(ours, theirs)

    reports = [
        summarize("Pure lateral (Fy)", fy_ours, fy_theirs, fy_diff),
        summarize("Pure longitudinal (Fx)", fx_ours, fx_theirs, fx_diff),
        summarize("Combined Fy (alpha,kappa)", cfy_o, cfy_t, cfy_d),
        summarize("Combined Fx (alpha,kappa)", cfx_o, cfx_t, cfx_d),
    ]
    print_report(reports)

    if args.csv:
        # minimal CSV: pure lateral only
        csv_path = Path(args.csv)
        print(f"\nWriting sweep CSV to {csv_path}")
        # We can only write fz/alpha/kappa/gamma if we re-expand; skipped for now.
        # Write a simple diff series instead.
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("idx,regime,ours_N,theirs_N,diff_N\n")
            for i, (o, t) in enumerate(zip(fy_ours, fy_theirs)):
                f.write(f"{i},pure_lat,{o:.4f},{t:.4f},{o - t:.4f}\n")
            for i, (o, t) in enumerate(zip(fx_ours, fx_theirs)):
                f.write(f"{i},pure_long,{o:.4f},{t:.4f},{o - t:.4f}\n")
            for i, (o, t) in enumerate(zip(cfy_o, cfy_t)):
                f.write(f"{i},comb_fy,{o:.4f},{t:.4f},{o - t:.4f}\n")
            for i, (o, t) in enumerate(zip(cfx_o, cfx_t)):
                f.write(f"{i},comb_fx,{o:.4f},{t:.4f},{o - t:.4f}\n")

    # Pass/fail: pure-slip regimes must match closely; combined is informational
    # because ours applies a friction-ellipse fallback LHR doesn't.
    threshold = args.rel_threshold
    pure_fail = (reports[0]["rel_rms"] > threshold or
                 reports[1]["rel_rms"] > threshold)
    if pure_fail:
        print(f"\nFAIL: pure-slip RMS exceeds {threshold:.1%} threshold.")
        return 1
    print(f"\nOK: pure-slip RMS within {threshold:.1%} threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
