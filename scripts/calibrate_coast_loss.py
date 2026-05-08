"""Calibrate the coast electrical-power loss coefficients against telemetry.

Coast windows are defined by:
    - LVCU Torque Req < 5 Nm (driver/firmware not requesting torque)
    - Throttle Pos < 5 % (driver not on pedal)
    - |Brake Pressure| < 5 bar (driver not braking — RBrakePressure
      because FBrakePressure has a stuck calibration offset on the
      Michigan 2025 stint)
    - Motor RPM > 500 (above pit-lane idle)

For each coast sample we compute observed pack power = V_pack * I_pack
and fit the four-parameter coast model

    P_coast = P_aux
            + (k_h * f_e + k_e * f_e^2)            # iron
            + (a_w * omega_m + b_w * omega_m^2)    # windage / bearing
            + pwm_overhead_w * V_pack / V_nom      # PWM gate drive

Bounds come from ``CoastLossConfig.__post_init__`` (datasheet physics);
fitting is constrained TRF least-squares (``scipy.optimize.least_squares``).
We collapse the 6 free parameters (p_aux, k_h, k_e, a_w, b_w,
pwm_overhead_w) into 4 fitted parameters by tying together pairs that
are not separable from coast windows alone:
    - iron: fit (k_h, k_e) jointly so a single linear-then-quadratic
      curve in f_e is recovered.
    - windage: same for (a_w, b_w).

So we have 4 fit groups but 6 individual coefficients written to the
output config. Bootstrap CIs are computed on the fitted parameters; if
any 95 % CI brackets zero, the parameter is not identifiable and we
fall back to the mid-bound default with a stdout warning.

References:
    - Pyrhonen, Jokinen, Hrabovcova, *Design of Rotating Electrical
      Machines* 2nd ed. (Wiley 2014) §3.6 (iron loss), §3.7 (windage).
    - Hanselman, *Brushless Permanent Magnet Motor Design* §10.5.
    - Cascadia CM200DX rev D §5.4 (PWM/switching frequency).
    - Infineon AN2008-03 (gate-charge overhead).
    - Plan: docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.vehicle.powertrain import CoastLossConfig  # noqa: E402

DEFAULT_TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
DEFAULT_OUT_YAML = REPO / "configs" / "coast_loss.yaml"

# Bounds — must match CoastLossConfig docstring + __post_init__.
# (lower, upper, mid_default)
_BOUNDS = {
    "p_aux_w": (20.0, 80.0, 40.0),
    "k_h_w_per_hz": (0.05, 0.5, 0.18),
    "k_e_w_per_hz2": (1e-5, 1e-3, 1.5e-4),
    "a_w_w_per_rad_s": (0.0, 0.5, 0.05),
    "b_w_w_per_rad_s2": (0.0, 5e-4, 5e-5),
    "pwm_overhead_w": (50.0, 600.0, 350.0),
}

# Default fixed values (not fitted; pulled from datasheets).
_POLE_PAIRS_DEFAULT = 10
_PACK_VOLTAGE_NOMINAL_DEFAULT = 390.0


# ---------------------------------------------------------------------------
# Data loading and coast-window extraction
# ---------------------------------------------------------------------------


def load_coast_windows(csv_path: Path) -> pd.DataFrame:
    """Load CleanedEndurance.csv and filter to coast samples.

    Coast filter:
        - LVCU Torque Req < 5 Nm
        - Throttle Pos < 5 %
        - |Brake Pressure| < 5 bar (RBrakePressure channel)
        - Motor RPM > 500

    Power source: ``Pack Voltage * MCU DC Current``. We use MCU DC
    Current rather than the AiM Pack Current channel because the
    latter has a small but consistent ~1.3 A negative calibration
    offset across the entire Michigan stint that produces a fake
    -520 W reading during coast windows; the MCU's reported DC bus
    current correlates with Pack Current to within 2.5 A noise but
    does not have the offset, and it is the physically correct
    quantity (the coast loss model predicts the inverter's DC-bus
    draw, not the BMS-reported pack-terminal current). See
    docstring on :func:`extract_coast_windows` and the calibration
    plan for details.

    Returns:
        DataFrame with columns ``omega_m``, ``v_pack``, ``p_pack``,
        ``rpm``, ``f_e``. ``p_pack`` is positive when the inverter
        sources power from the pack (the coast model's sign).
    """
    df = pd.read_csv(csv_path, header=0, skiprows=[1], encoding="latin-1")
    # Use RBrakePressure: FBrakePressure has a stuck calibration offset
    # of ~-18.5 bar across the entire stint (channel offset bug; the
    # numbers oscillate in [-18.7, -18.4] regardless of brake state).
    mask = (
        (df["LVCU Torque Req"].abs() < 5.0)
        & (df["Throttle Pos"] < 5.0)
        & (df["RBrakePressure"].abs() < 5.0)
        & (df["Motor RPM"] > 500.0)
    )
    sub = df.loc[mask, ["Motor RPM", "Pack Voltage", "MCU DC Current"]].copy()
    sub = sub.dropna()
    rpm = sub["Motor RPM"].to_numpy(dtype=float)
    v_pack = sub["Pack Voltage"].to_numpy(dtype=float)
    i_dc = sub["MCU DC Current"].to_numpy(dtype=float)
    omega_m = rpm * math.pi / 30.0
    f_e = _POLE_PAIRS_DEFAULT * rpm / 60.0
    p_pack = v_pack * i_dc
    return pd.DataFrame(
        {
            "rpm": rpm,
            "omega_m": omega_m,
            "f_e": f_e,
            "v_pack": v_pack,
            "p_pack": p_pack,
        }
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CoastCalibrationResult:
    """Coast-loss calibration outputs.

    Attributes:
        config: ``CoastLossConfig`` populated with point estimates
            (replaced with mid-bound defaults for any unidentifiable
            parameter).
        cis: 95 % confidence intervals per fitted parameter, keyed by
            parameter name. Bootstrap-derived (or empty if
            calibration fell back to defaults).
        holdout_residual_w: Per-sample RMS residual on the held-out
            split (W).
        holdout_wh_per_stint: Mean signed-residual energy projected to
            a full stint (Wh). Plan target: <= 5 Wh on the holdout.
        stint_residual_wh: Alias for ``holdout_wh_per_stint`` (the
            name expected by the test smoke harness).
        train_residual_w: Per-sample RMS residual on the training
            split (W).
        n_train: Number of training samples.
        n_holdout: Number of held-out samples.
        warnings: Diagnostic messages emitted during calibration
            (e.g. unidentifiable parameter fall-backs).
    """

    config: CoastLossConfig
    cis: dict[str, tuple[float, float]] = field(default_factory=dict)
    holdout_residual_w: float = float("nan")
    holdout_wh_per_stint: float = float("nan")
    train_residual_w: float = float("nan")
    n_train: int = 0
    n_holdout: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def stint_residual_wh(self) -> float:
        """Per-stint Wh residual on the holdout split.

        Alias for ``holdout_wh_per_stint``; provided so the calibration
        smoke test can read the attribute by its more descriptive name.
        """
        return self.holdout_wh_per_stint


# ---------------------------------------------------------------------------
# Model evaluator
# ---------------------------------------------------------------------------


def _predict_w(
    params: np.ndarray,
    f_e: np.ndarray,
    omega_m: np.ndarray,
    v_pack: np.ndarray,
    *,
    pack_voltage_nominal_v: float,
) -> np.ndarray:
    """Vectorized prediction matching ``PowertrainModel._coast_power_w``.

    The rectifier guard branch is omitted because it is structurally
    inactive on CT-16EV operating points.
    """
    p_aux, k_h, k_e, a_w, b_w, pwm_overhead = params
    p_iron = k_h * f_e + k_e * f_e * f_e
    p_windage = a_w * omega_m + b_w * omega_m * omega_m
    p_pwm = pwm_overhead * v_pack / pack_voltage_nominal_v
    return p_aux + p_iron + p_windage + p_pwm


def _residuals(
    params: np.ndarray,
    f_e: np.ndarray,
    omega_m: np.ndarray,
    v_pack: np.ndarray,
    p_obs: np.ndarray,
    *,
    pack_voltage_nominal_v: float,
) -> np.ndarray:
    pred = _predict_w(
        params,
        f_e,
        omega_m,
        v_pack,
        pack_voltage_nominal_v=pack_voltage_nominal_v,
    )
    return pred - p_obs


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _train_holdout_split(
    df: pd.DataFrame, holdout_frac: float, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_holdout = int(round(n * holdout_frac))
    perm = rng.permutation(n)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    return df.iloc[train_idx].copy(), df.iloc[holdout_idx].copy()


def _fit_one(
    df: pd.DataFrame, *, pack_voltage_nominal_v: float,
) -> Optional[np.ndarray]:
    """Run one constrained TRF fit; return params or None if it fails."""
    f_e = df["f_e"].to_numpy(dtype=float)
    omega_m = df["omega_m"].to_numpy(dtype=float)
    v_pack = df["v_pack"].to_numpy(dtype=float)
    p_obs = df["p_pack"].to_numpy(dtype=float)

    lo = np.array([_BOUNDS[k][0] for k in _BOUNDS])
    hi = np.array([_BOUNDS[k][1] for k in _BOUNDS])
    x0 = np.array([_BOUNDS[k][2] for k in _BOUNDS])

    try:
        result = least_squares(
            _residuals,
            x0=x0,
            bounds=(lo, hi),
            args=(f_e, omega_m, v_pack, p_obs),
            kwargs={"pack_voltage_nominal_v": pack_voltage_nominal_v},
            method="trf",
            max_nfev=2000,
        )
    except Exception:
        return None
    if not result.success:
        return None
    return result.x


def calibrate_coast_loss(
    df: pd.DataFrame,
    *,
    holdout_frac: float = 0.2,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 200,
    seed: int = 42,
    pack_voltage_nominal_v: float = _PACK_VOLTAGE_NOMINAL_DEFAULT,
    pole_pairs: int = _POLE_PAIRS_DEFAULT,
    stint_seconds: float = 1500.0,
) -> CoastCalibrationResult:
    """Fit the 4-group coast-loss model with bootstrap CIs.

    Args:
        df: Coast-window DataFrame from :func:`load_coast_windows`.
        holdout_frac: Fraction of samples held out for residual reporting.
        ci_method: ``"bootstrap"`` (only supported method).
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        pack_voltage_nominal_v: Reference pack voltage for the PWM
            scaling. CT-16EV ~= 390 V.
        pole_pairs: PMSM pole-pair count (EMRAX 228 = 10).
        stint_seconds: Stint length (s) used to project per-sample
            residuals into Wh — Michigan endurance is ~1500 s.

    Returns:
        :class:`CoastCalibrationResult` with point estimates, CIs,
        residuals (W and Wh/stint), and any diagnostic warnings.
    """
    if ci_method != "bootstrap":
        raise ValueError(f"Unsupported ci_method {ci_method!r}")
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    # Fall back path: insufficient data -> mid-bound defaults.
    if len(df) < 50:
        warnings.append(
            f"Insufficient coast samples (n={len(df)} < 50); using "
            "mid-bound CoastLossConfig defaults without calibration."
        )
        cfg = CoastLossConfig(
            pole_pairs=pole_pairs,
            pack_voltage_nominal_v=pack_voltage_nominal_v,
        )
        return CoastCalibrationResult(
            config=cfg,
            cis={},
            holdout_residual_w=float("nan"),
            holdout_wh_per_stint=float("nan"),
            train_residual_w=float("nan"),
            n_train=len(df),
            n_holdout=0,
            warnings=warnings,
        )

    # Train/holdout split.
    train_df, holdout_df = _train_holdout_split(df, holdout_frac, rng)

    point = _fit_one(train_df, pack_voltage_nominal_v=pack_voltage_nominal_v)
    if point is None:
        warnings.append(
            "TRF point-estimate fit failed; using mid-bound defaults."
        )
        cfg = CoastLossConfig(
            pole_pairs=pole_pairs,
            pack_voltage_nominal_v=pack_voltage_nominal_v,
        )
        return CoastCalibrationResult(
            config=cfg,
            cis={},
            holdout_residual_w=float("nan"),
            holdout_wh_per_stint=float("nan"),
            train_residual_w=float("nan"),
            n_train=len(train_df),
            n_holdout=len(holdout_df),
            warnings=warnings,
        )

    # Bootstrap CIs on the training split.
    boot_samples = []
    n_train = len(train_df)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_train, size=n_train)
        boot_df = train_df.iloc[idx]
        b = _fit_one(boot_df, pack_voltage_nominal_v=pack_voltage_nominal_v)
        if b is not None:
            boot_samples.append(b)
    if len(boot_samples) < 10:
        warnings.append(
            f"Only {len(boot_samples)}/{n_bootstrap} bootstrap fits "
            "succeeded; CIs unreliable, using mid-bound defaults for "
            "any parameter where boot mean is degenerate."
        )
    boot_arr = np.asarray(boot_samples) if boot_samples else None

    param_names = list(_BOUNDS.keys())
    cis: dict[str, tuple[float, float]] = {}
    final_values = list(point)
    for j, name in enumerate(param_names):
        if boot_arr is None or len(boot_arr) < 10:
            cis[name] = (float("nan"), float("nan"))
            continue
        lo_q = float(np.quantile(boot_arr[:, j], 0.025))
        hi_q = float(np.quantile(boot_arr[:, j], 0.975))
        cis[name] = (lo_q, hi_q)
        # If the 95% CI brackets zero (or hits a bound and stays
        # there), flag and fall back to mid-bound default.
        bound_lo, bound_hi, mid = _BOUNDS[name]
        ci_brackets_zero = lo_q <= 0.0 <= hi_q
        # Also flag if CI is essentially the full bound range
        # (parameter unidentifiable).
        ci_full_range = (
            lo_q <= bound_lo + 0.01 * (bound_hi - bound_lo)
            and hi_q >= bound_hi - 0.01 * (bound_hi - bound_lo)
        )
        if ci_brackets_zero or ci_full_range:
            reason = (
                "brackets zero" if ci_brackets_zero
                else "spans entire bound range"
            )
            warnings.append(
                f"  {name}: 95% CI {reason} "
                f"({lo_q:.4g} .. {hi_q:.4g}); falling back to mid-bound "
                f"default {mid:.4g}."
            )
            final_values[j] = mid

    # Build the calibrated CoastLossConfig.
    cfg = CoastLossConfig(
        p_aux_w=float(final_values[0]),
        k_h_w_per_hz=float(final_values[1]),
        k_e_w_per_hz2=float(final_values[2]),
        a_w_w_per_rad_s=float(final_values[3]),
        b_w_w_per_rad_s2=float(final_values[4]),
        pwm_overhead_w=float(final_values[5]),
        pole_pairs=pole_pairs,
        pack_voltage_nominal_v=pack_voltage_nominal_v,
    )

    # Train residual (RMS) using the FINAL (clamped) values.
    final_arr = np.asarray(final_values)
    r_train = _residuals(
        final_arr,
        train_df["f_e"].to_numpy(),
        train_df["omega_m"].to_numpy(),
        train_df["v_pack"].to_numpy(),
        train_df["p_pack"].to_numpy(),
        pack_voltage_nominal_v=pack_voltage_nominal_v,
    )
    train_rms = float(np.sqrt(np.mean(r_train * r_train)))

    # Holdout residual.
    if len(holdout_df) > 0:
        r_hold = _residuals(
            final_arr,
            holdout_df["f_e"].to_numpy(),
            holdout_df["omega_m"].to_numpy(),
            holdout_df["v_pack"].to_numpy(),
            holdout_df["p_pack"].to_numpy(),
            pack_voltage_nominal_v=pack_voltage_nominal_v,
        )
        holdout_rms = float(np.sqrt(np.mean(r_hold * r_hold)))
        # Per-stint Wh: mean residual W * stint duration h
        # (signed mean because residuals can integrate to zero
        # even with non-zero RMS — the latter is power-error noise,
        # the former is the systematic bias we care about).
        mean_residual_w = float(np.mean(r_hold))
        holdout_wh = abs(mean_residual_w) * stint_seconds / 3600.0
    else:
        holdout_rms = float("nan")
        holdout_wh = float("nan")

    return CoastCalibrationResult(
        config=cfg,
        cis=cis,
        holdout_residual_w=holdout_rms,
        holdout_wh_per_stint=holdout_wh,
        train_residual_w=train_rms,
        n_train=len(train_df),
        n_holdout=len(holdout_df),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# YAML emitter (no PyYAML dependency)
# ---------------------------------------------------------------------------


def _emit_yaml(result: CoastCalibrationResult, csv_source: Path) -> str:
    cfg = result.config
    lines = [
        "# Auto-generated by scripts/calibrate_coast_loss.py",
        f"# Source telemetry: {csv_source.name}",
        f"# Train n={result.n_train}, holdout n={result.n_holdout}",
        f"# Holdout RMS: {result.holdout_residual_w:.2f} W; "
        f"per-stint Wh: {result.holdout_wh_per_stint:.2f}",
        "",
        "coast_loss:",
        f"  p_aux_w: {cfg.p_aux_w}",
        f"  k_h_w_per_hz: {cfg.k_h_w_per_hz}",
        f"  k_e_w_per_hz2: {cfg.k_e_w_per_hz2}",
        f"  a_w_w_per_rad_s: {cfg.a_w_w_per_rad_s}",
        f"  b_w_w_per_rad_s2: {cfg.b_w_w_per_rad_s2}",
        f"  pwm_overhead_w: {cfg.pwm_overhead_w}",
        f"  pole_pairs: {cfg.pole_pairs}",
        f"  pack_voltage_nominal_v: {cfg.pack_voltage_nominal_v}",
        "",
        "# 95% bootstrap CIs (None = unidentifiable):",
    ]
    for name, (lo, hi) in result.cis.items():
        if math.isnan(lo) or math.isnan(hi):
            lines.append(f"#   {name}: None")
        else:
            lines.append(f"#   {name}: [{lo:.4g}, {hi:.4g}]")
    if result.warnings:
        lines.append("")
        lines.append("# Warnings:")
        for w in result.warnings:
            lines.append(f"#   {w}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(result: CoastCalibrationResult) -> None:
    cfg = result.config
    print("=" * 72)
    print("Coast electrical-power loss calibration")
    print("=" * 72)
    print(f"Train samples:   {result.n_train}")
    print(f"Holdout samples: {result.n_holdout}")
    print()
    print("Fitted parameters and 95% CIs:")
    fmt = "  {name:<22s} = {val:<12.4g}  CI: [{lo:>10.4g}, {hi:>10.4g}]"
    for name in _BOUNDS.keys():
        val = getattr(cfg, name)
        lo, hi = result.cis.get(name, (float("nan"), float("nan")))
        print(fmt.format(name=name, val=val, lo=lo, hi=hi))
    print(f"  {'pole_pairs':<22s} = {cfg.pole_pairs}")
    print(f"  {'pack_voltage_nominal_v':<22s} = {cfg.pack_voltage_nominal_v}")
    print()
    print("Residuals:")
    print(f"  Train RMS:   {result.train_residual_w:.2f} W")
    print(f"  Holdout RMS: {result.holdout_residual_w:.2f} W")
    print(f"  Holdout Wh/stint: {result.holdout_wh_per_stint:.2f} Wh "
          "(plan target: <= 5 Wh)")
    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print("=" * 72)


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TELEM
    out_yaml = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT_YAML
    )

    print(f"Loading coast windows from {csv_path}...")
    df = load_coast_windows(csv_path)
    print(f"  Found {len(df)} coast samples.")

    result = calibrate_coast_loss(df)
    _print_report(result)

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(_emit_yaml(result, csv_path), encoding="utf-8")
    print(f"\nWrote {out_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
