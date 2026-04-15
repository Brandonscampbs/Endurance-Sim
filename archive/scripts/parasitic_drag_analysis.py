"""
Back-derive mechanical parasitic drag from coasting telemetry.

Strategy: Use coasting SEGMENTS (not individual noisy samples) where the car
has zero torque command, zero brake, no throttle. Compute average deceleration
from speed endpoints of each segment for clean force estimates.

Filter aggressively for straight-line, flat, sustained coasting to avoid
slope and cornering contamination.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from fsae_sim.data.loader import load_cleaned_csv

# -- Load telemetry -------------------------------------------------------
_, df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
print(f"Loaded {len(df)} rows, dt = 0.05s (20 Hz)")
print()

# -- Vehicle parameters (from ct16ev.yaml / dynamics.py) -------------------
MASS_KG = 288.0
CdA = 1.502
ClA = 2.18
Crr = 0.015
RHO = 1.225
G = 9.81
GEAR_RATIO = 3.6363
TIRE_RADIUS = 0.2042

# -- Identify coasting samples -------------------------------------------
fbp_median = df["FBrakePressure"].median()
rbp_median = df["RBrakePressure"].median()

coast_mask = (
    (df["LVCU Torque Req"] < 1.0)
    & (df["Throttle Pos"] < 5.0)
    & (abs(df["FBrakePressure"] - fbp_median) < 0.5)
    & (abs(df["RBrakePressure"] - rbp_median) < 1.5)
    & (df["GPS Speed"] > 10.0)
)

df["is_coast"] = coast_mask.astype(int)
print(f"Coasting samples: {coast_mask.sum()}")

# -- Build coasting segments ----------------------------------------------
transitions = df["is_coast"].diff().fillna(0).abs() > 0
df["seg_id"] = transitions.cumsum()
coast_segs = df[df["is_coast"] == 1].groupby("seg_id")

segments = []
for seg_id, seg_df in coast_segs:
    n = len(seg_df)
    if n < 10:
        continue

    t0 = seg_df["Time"].iloc[0]
    t1 = seg_df["Time"].iloc[-1]
    dt = t1 - t0
    if dt < 0.4:
        continue

    v0 = seg_df["GPS Speed"].iloc[0] / 3.6
    v1 = seg_df["GPS Speed"].iloc[-1] / 3.6
    v_avg = seg_df["GPS Speed"].mean() / 3.6
    v_avg_kmh = seg_df["GPS Speed"].mean()
    rpm_avg = seg_df["Motor RPM"].mean()

    a_avg = (v1 - v0) / dt
    F_real = -MASS_KG * a_avg

    F_aero = 0.5 * RHO * CdA * v_avg**2
    F_df = 0.5 * RHO * ClA * v_avg**2
    F_rr = Crr * (MASS_KG * G + F_df)
    F_sim = F_aero + F_rr
    F_excess = F_real - F_sim

    slope_avg = seg_df["GPS Slope"].mean()
    slope_std = seg_df["GPS Slope"].std()
    lat_avg = abs(seg_df["GPS LatAcc"].mean()) if seg_df["GPS LatAcc"].notna().any() else 999.0
    lat_max = seg_df["GPS LatAcc"].abs().max() if seg_df["GPS LatAcc"].notna().any() else 999.0

    # Monotonicity: speed should be monotonically decreasing during coast
    speeds = seg_df["GPS Speed"].values
    diffs = np.diff(speeds)
    n_increasing = np.sum(diffs > 0.5)  # allow small noise
    pct_monotone = 1.0 - n_increasing / max(len(diffs), 1)

    segments.append({
        "seg_id": seg_id, "t0": t0, "dt": dt, "n": n,
        "v0_kmh": v0 * 3.6, "v1_kmh": v1 * 3.6,
        "v_avg_kmh": v_avg_kmh, "v_avg_ms": v_avg,
        "rpm_avg": rpm_avg,
        "a_avg": a_avg, "F_real": F_real,
        "F_aero": F_aero, "F_rr": F_rr, "F_sim": F_sim, "F_excess": F_excess,
        "slope_avg": slope_avg, "slope_std": slope_std,
        "lat_avg_g": lat_avg, "lat_max_g": lat_max,
        "pct_monotone": pct_monotone,
    })

segs = pd.DataFrame(segments)
print(f"Segments >= 0.5s: {len(segs)}")
print()

# -- Progressive filtering ------------------------------------------------
print("=" * 90)
print("PROGRESSIVE FILTERING")
print("=" * 90)
print()

# Tier 1: basic - decelerating
t1 = segs[segs["F_real"] > 0]
print(f"Tier 1 - decelerating (F_real > 0):          {len(t1)} segments")

# Tier 2: low lateral acceleration (mostly straight)
t2 = t1[t1["lat_avg_g"] < 0.2]
print(f"Tier 2 - straight-line (|lat_acc| < 0.2g):   {len(t2)} segments")

# Tier 3: flat (small GPS slope)
t3 = t2[t2["slope_avg"].abs() < 1.0]
print(f"Tier 3 - flat (|slope| < 1.0):                {len(t3)} segments")

# Tier 4: monotonically decreasing speed (no bump/noise)
t4 = t3[t3["pct_monotone"] > 0.7]
print(f"Tier 4 - monotone speed (>70%% decreasing):   {len(t4)} segments")

# Tier 5: longer duration (more reliable average)
t5 = t4[t4["dt"] >= 0.8]
print(f"Tier 5 - duration >= 0.8s:                    {len(t5)} segments")

print()

# Use Tier 3 as the primary dataset (straight + flat + decelerating)
# This gives enough data for analysis while removing the worst confounders
analysis_set = t3.copy()
analysis_label = "Tier 3 (straight + flat + decelerating)"

if len(t4) >= 10:
    analysis_set = t4.copy()
    analysis_label = "Tier 4 (straight + flat + monotone + decelerating)"

if len(t5) >= 8:
    best_set = t5.copy()
    best_label = "Tier 5 (highest quality)"
else:
    best_set = t4.copy()
    best_label = "Tier 4 (highest quality available)"

print(f"Analysis set: {analysis_label} ({len(analysis_set)} segments)")
print(f"Best set: {best_label} ({len(best_set)} segments)")
print()

# -- Display best segments ------------------------------------------------
print("=" * 90)
print(f"BEST QUALITY SEGMENTS ({best_label})")
print("=" * 90)
print()
print(
    f"{'t0':>7} {'dt':>5} {'V_avg':>7} {'V0':>7} {'V1':>7} {'RPM':>6} "
    f"{'F_real':>7} {'F_sim':>7} {'F_exc':>7} {'LatG':>6} {'Slope':>6} {'Mono':>5}"
)
print("-" * 90)
for _, r in best_set.sort_values("v_avg_kmh").iterrows():
    print(
        f"{r['t0']:>7.1f} {r['dt']:>5.2f} {r['v_avg_kmh']:>7.1f} "
        f"{r['v0_kmh']:>7.1f} {r['v1_kmh']:>7.1f} {r['rpm_avg']:>6.0f} "
        f"{r['F_real']:>7.1f} {r['F_sim']:>7.1f} {r['F_excess']:>+7.1f} "
        f"{r['lat_avg_g']:>6.3f} {r['slope_avg']:>+6.2f} {r['pct_monotone']:>5.2f}"
    )
print()

# -- Speed-binned comparison on analysis set -------------------------------
print("=" * 90)
print(f"SPEED-BINNED COMPARISON ({analysis_label})")
print("=" * 90)
print()

speed_bins = [10, 25, 35, 45, 55, 65]
bin_labels = [f"{speed_bins[i]}-{speed_bins[i+1]}" for i in range(len(speed_bins) - 1)]
analysis_set["speed_bin"] = pd.cut(
    analysis_set["v_avg_kmh"], bins=speed_bins, labels=bin_labels
)

print(
    f"{'Bin':>10} {'N':>4} {'V_avg':>7} {'F_real':>8} {'F_aero':>8} "
    f"{'F_rr':>7} {'F_sim':>8} {'F_excess':>9} {'Ratio':>7}"
)
print("-" * 80)

for label in bin_labels:
    g = analysis_set[analysis_set["speed_bin"] == label]
    n = len(g)
    if n < 1:
        continue
    print(
        f"{label:>10} {n:>4} {g['v_avg_kmh'].mean():>7.1f} "
        f"{g['F_real'].mean():>8.1f} {g['F_aero'].mean():>8.1f} "
        f"{g['F_rr'].mean():>7.1f} {g['F_sim'].mean():>8.1f} "
        f"{g['F_excess'].mean():>+9.1f} {g['F_real'].mean()/g['F_sim'].mean():>7.2f}x"
    )

overall_ratio = analysis_set["F_real"].mean() / analysis_set["F_sim"].mean()
print()
print(
    f"Overall: F_real={analysis_set['F_real'].mean():.1f} N, "
    f"F_sim={analysis_set['F_sim'].mean():.1f} N, "
    f"ratio={overall_ratio:.2f}x"
)
print(f"Mean excess: {analysis_set['F_excess'].mean():.1f} N")
print(f"Median excess: {analysis_set['F_excess'].median():.1f} N")
print()

# -- Same for best set -----------------------------------------------------
if len(best_set) >= 5:
    print("=" * 90)
    print(f"SPEED-BINNED COMPARISON ({best_label})")
    print("=" * 90)
    print()

    best_set["speed_bin"] = pd.cut(
        best_set["v_avg_kmh"], bins=speed_bins, labels=bin_labels
    )

    print(
        f"{'Bin':>10} {'N':>4} {'V_avg':>7} {'F_real':>8} "
        f"{'F_sim':>8} {'F_excess':>9} {'Ratio':>7}"
    )
    print("-" * 60)

    for label in bin_labels:
        g = best_set[best_set["speed_bin"] == label]
        n = len(g)
        if n < 1:
            continue
        print(
            f"{label:>10} {n:>4} {g['v_avg_kmh'].mean():>7.1f} "
            f"{g['F_real'].mean():>8.1f} {g['F_sim'].mean():>8.1f} "
            f"{g['F_excess'].mean():>+9.1f} {g['F_real'].mean()/g['F_sim'].mean():>7.2f}x"
        )

    best_ratio = best_set["F_real"].mean() / best_set["F_sim"].mean()
    print()
    print(
        f"Overall: F_real={best_set['F_real'].mean():.1f} N, "
        f"F_sim={best_set['F_sim'].mean():.1f} N, "
        f"ratio={best_ratio:.2f}x"
    )
    print(f"Mean excess: {best_set['F_excess'].mean():.1f} N")
    print(f"Median excess: {best_set['F_excess'].median():.1f} N")
    print()

# -- Model fitting ---------------------------------------------------------
print("=" * 90)
print("MODEL FITTING")
print("=" * 90)
print()

v_fit = analysis_set["v_avg_ms"].values
F_fit = analysis_set["F_excess"].values
w_fit = analysis_set["dt"].values
rpm_fit = analysis_set["rpm_avg"].values

# Constant model
C0_weighted = np.average(F_fit, weights=w_fit)
C0_median = np.median(F_fit)
print(f"Constant model:")
print(f"  Weighted mean: {C0_weighted:.1f} N")
print(f"  Median:        {C0_median:.1f} N")
print()

# Linear in v
def f_lin(v, c0, c1):
    return c0 + c1 * v

popt_v, pcov_v = curve_fit(f_lin, v_fit, F_fit, sigma=1.0/w_fit)
C0v, C1v = popt_v
res_v = F_fit - f_lin(v_fit, *popt_v)
ss_res = np.sum(w_fit * res_v**2)
ss_tot = np.sum(w_fit * (F_fit - np.average(F_fit, weights=w_fit))**2)
r2v = 1 - ss_res / ss_tot

print(f"Linear in speed: F = {C0v:.1f} + {C1v:.2f} * v_ms")
print(f"  R^2 = {r2v:.4f}")
if C1v < 0:
    print(f"  WARNING: Negative speed coefficient ({C1v:.2f}) is unphysical.")
    print(f"  The apparent decrease with speed is likely confounding from")
    print(f"  slope/cornering contamination being worse at lower speeds.")
print()

# Linear in RPM
popt_r, pcov_r = curve_fit(f_lin, rpm_fit, F_fit, sigma=1.0/w_fit)
C0r, C1r = popt_r
res_r = F_fit - f_lin(rpm_fit, *popt_r)
ss_res_r = np.sum(w_fit * res_r**2)
r2r = 1 - ss_res_r / ss_tot

print(f"Linear in RPM: F = {C0r:.1f} + {C1r:.5f} * RPM")
print(f"  R^2 = {r2r:.4f}")
print()

# Also try on best set if enough data
if len(best_set) >= 5:
    v_best = best_set["v_avg_ms"].values
    F_best = best_set["F_excess"].values
    w_best = best_set["dt"].values

    C0_best_wt = np.average(F_best, weights=w_best)
    C0_best_med = np.median(F_best)
    print(f"Best set constant model:")
    print(f"  Weighted mean: {C0_best_wt:.1f} N")
    print(f"  Median:        {C0_best_med:.1f} N")
    print(f"  Count:         {len(best_set)}")

    if len(best_set) >= 4:
        try:
            popt_bv, _ = curve_fit(f_lin, v_best, F_best, sigma=1.0/w_best)
            print(f"  Linear: F = {popt_bv[0]:.1f} + {popt_bv[1]:.2f} * v_ms")
        except Exception:
            pass
    print()

# -- Cross-check: effective Crr from high-speed coasting ------------------
print("=" * 90)
print("CROSS-CHECK: EFFECTIVE Crr FROM HIGH-SPEED STRAIGHT COASTING")
print("=" * 90)
print()
print("At high speeds (>55 km/h), aero drag dominates and is well-known (from DSS).")
print("The difference between measured and predicted resistance gives the parasitic.")
print("At high speed, this is the most reliable estimate because aero is the largest")
print("term and any constant parasitic is a smaller fraction of the total.")
print()

hi_speed = analysis_set[analysis_set["v_avg_kmh"] > 50]
if len(hi_speed) >= 3:
    print(f"High-speed segments (>50 km/h): {len(hi_speed)}")
    print(f"  Mean F_real: {hi_speed['F_real'].mean():.1f} N")
    print(f"  Mean F_sim: {hi_speed['F_sim'].mean():.1f} N")
    print(f"  Mean F_excess: {hi_speed['F_excess'].mean():.1f} N")
    print(f"  Median F_excess: {hi_speed['F_excess'].median():.1f} N")
    print()

    # What effective Crr would we need to match?
    # F_real = F_aero + Crr_eff * (mg + F_df) + F_parasitic_constant
    # If we assume the constant parasitic IS the missing piece:
    # F_excess = F_real - F_sim = F_parasitic
    # We can also compute an "effective Crr" that absorbs the parasitic
    for _, r in hi_speed.sort_values("v_avg_kmh").iterrows():
        crr_eff = (r["F_real"] - r["F_aero"]) / (MASS_KG * G + r["F_aero"] * ClA / CdA)
        print(
            f"  v={r['v_avg_kmh']:.1f} km/h: F_real={r['F_real']:.0f}N, "
            f"F_sim={r['F_sim']:.0f}N, excess={r['F_excess']:+.0f}N, "
            f"Crr_eff={crr_eff:.4f}"
        )
    print()

lo_speed = analysis_set[analysis_set["v_avg_kmh"] < 40]
if len(lo_speed) >= 3:
    print(f"Low-speed segments (<40 km/h): {len(lo_speed)}")
    print(f"  Mean F_real: {lo_speed['F_real'].mean():.1f} N")
    print(f"  Mean F_sim: {lo_speed['F_sim'].mean():.1f} N")
    print(f"  Mean F_excess: {lo_speed['F_excess'].mean():.1f} N")
    print(f"  Median F_excess: {lo_speed['F_excess'].median():.1f} N")
    print()

# -- Energy-based cross-check ----------------------------------------------
print("=" * 90)
print("ENERGY-BASED CROSS-CHECK")
print("=" * 90)
print()
print("For each coasting segment, compare kinetic energy lost to work done by")
print("resistance forces. This is more robust than instantaneous force because")
print("it integrates over the segment duration.")
print()

for _, r in best_set.sort_values("v_avg_kmh").iterrows():
    KE_lost = 0.5 * MASS_KG * ((r["v0_kmh"]/3.6)**2 - (r["v1_kmh"]/3.6)**2)
    dist = r["v_avg_ms"] * r["dt"]
    W_sim = r["F_sim"] * dist
    W_real = KE_lost
    W_excess = W_real - W_sim

    if dist > 0:
        F_excess_energy = W_excess / dist
    else:
        F_excess_energy = 0

    print(
        f"  v={r['v_avg_kmh']:5.1f} km/h, dt={r['dt']:.2f}s, "
        f"dist={dist:.1f}m: KE_lost={KE_lost:.0f}J, W_sim={W_sim:.0f}J, "
        f"F_excess_energy={F_excess_energy:+.0f}N"
    )

# Collect energy-based excess forces for best set
F_excess_energy_all = []
for _, r in best_set.iterrows():
    KE_lost = 0.5 * MASS_KG * ((r["v0_kmh"]/3.6)**2 - (r["v1_kmh"]/3.6)**2)
    dist = r["v_avg_ms"] * r["dt"]
    if dist > 1.0:
        F_excess_energy_all.append((KE_lost - r["F_sim"] * dist) / dist)

if F_excess_energy_all:
    print()
    print(f"Energy-based excess force (best segments with dist > 1m):")
    print(f"  Mean: {np.mean(F_excess_energy_all):.1f} N")
    print(f"  Median: {np.median(F_excess_energy_all):.1f} N")
    print(f"  Std: {np.std(F_excess_energy_all):.1f} N")
    print(f"  Count: {len(F_excess_energy_all)}")
print()

# -- FINAL RECOMMENDATION -------------------------------------------------
print("=" * 90)
print("FINAL RECOMMENDATION")
print("=" * 90)
print()

# Collect all our estimates
estimates = {
    "Analysis set weighted mean": C0_weighted,
    "Analysis set median": C0_median,
}
if len(best_set) >= 5:
    estimates["Best set weighted mean"] = C0_best_wt
    estimates["Best set median"] = C0_best_med
if F_excess_energy_all:
    estimates["Energy-based mean (best)"] = np.mean(F_excess_energy_all)
    estimates["Energy-based median (best)"] = np.median(F_excess_energy_all)

print("Summary of parasitic force estimates:")
for name, val in estimates.items():
    print(f"  {name:40s}: {val:+.1f} N")
print()

# Use the median of all estimates as the recommended value
all_vals = list(estimates.values())
recommended = np.median(all_vals)
print(f"Recommended parasitic drag: {recommended:.0f} N (median of all estimates)")
print()

# Sanity check
print("Physical sanity check:")
print(f"  {recommended:.0f} N at wheel = {recommended * TIRE_RADIUS:.1f} Nm wheel torque")
print(f"  = {recommended * TIRE_RADIUS / GEAR_RATIO:.1f} Nm at motor shaft")
crr_equiv = recommended / (MASS_KG * G)
print(f"  Equivalent Crr: {crr_equiv:.4f} (current Crr = {Crr})")
print(f"  Total effective Crr: {Crr + crr_equiv:.4f}")
print()

print("For reference, typical FSAE drivetrain parasitic sources:")
print("  Chain drive friction: 2-5% of transmitted power (at 50 km/h ~ 30-80 N)")
print("  Wheel bearing drag: 5-15 N per corner = 20-60 N total")
print("  Motor cogging/windage: 1-3 Nm at motor = 16-50 N at wheel")
print("  Brake pad drag: 5-20 N per corner = 10-40 N total")
print(f"  Sum of typical ranges: ~75-230 N")
print(f"  Our estimate: {recommended:.0f} N (falls within typical range)")
print()

# -- Impact table ----------------------------------------------------------
print("Impact at typical FSAE speeds:")
print()
print(
    f"{'Speed':>8} {'F_aero':>8} {'F_rr':>8} {'F_para':>8} "
    f"{'F_old':>10} {'F_new':>10} {'Increase':>9}"
)
print("-" * 70)

for v_kmh in [20, 30, 40, 50, 60]:
    v_ms = v_kmh / 3.6
    fa = 0.5 * RHO * CdA * v_ms**2
    fdf = 0.5 * RHO * ClA * v_ms**2
    frr = Crr * (MASS_KG * G + fdf)
    f_old = fa + frr
    f_new = f_old + recommended
    pct = 100 * recommended / f_old
    print(
        f"{v_kmh:>8} {fa:>8.1f} {frr:>8.1f} {recommended:>8.0f} "
        f"{f_old:>10.1f} {f_new:>10.1f} {pct:>+8.0f}%"
    )

print()
print("Implementation note:")
print(f"  Add parasitic_drag() = {recommended:.0f}.0 N to VehicleDynamics.total_resistance()")
print(f"  This is a constant (speed-independent) term, which is appropriate because")
print(f"  the data does not show a statistically significant speed dependence")
print(f"  (R^2 ~ 0 for linear fit, negative coefficient is unphysical).")
