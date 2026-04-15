"""Root-cause v3: Verify GPS LonAcc sign convention and deep-dive
into the massive resistance gap.

The v2 analysis shows:
- Straight-line coasting: real car decelerates 2-4x harder than sim predicts
- Sim resistance at 0.23-0.75x of what GPS shows
- Implied Crr = 0.03-0.06 vs sim's 0.015

This is HUGE. Either:
(a) GPS LonAcc has wrong sign convention
(b) There's a real resistance source we're missing (engine braking? regen?)
(c) The car has much higher rolling resistance than assumed
(d) Grade/slope effects are significant

Let's verify with first principles.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from fsae_sim.data.loader import load_cleaned_csv

MASS_KG = 288.0
GEAR_RATIO = 3.6363
TIRE_RADIUS_M = 0.228
CDA = 1.502
CLA = 2.18
CRR = 0.015
RHO = 1.225
G = 9.81

_, df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")

print("=" * 90)
print("FORCE ROOT-CAUSE v3: Sign Convention & Resistance Deep-Dive")
print("=" * 90)

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Verify GPS LonAcc sign convention using speed derivative
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 1: GPS LonAcc sign convention check ---")

d = df.copy()
d["speed_ms"] = d["GPS Speed"] / 3.6
d["dt"] = d["Time"].diff().fillna(0.05)
d["dv_dt"] = d["speed_ms"].diff() / d["dt"]  # numerical derivative of speed

# Compare dv/dt to GPS LonAcc
mask_valid = (d["GPS Speed"] > 10) & d["GPS LonAcc"].notna() & (d["dt"] > 0.01) & (d["dt"] < 0.2)
dv = d[mask_valid]

# Smooth both for comparison
window = 5
dv_smooth = dv["dv_dt"].rolling(window, center=True).mean()
gps_lon = dv["GPS LonAcc"] * G  # convert to m/s^2

corr_positive = np.corrcoef(dv_smooth.dropna(), gps_lon.iloc[window//2:-(window//2)].values[:len(dv_smooth.dropna())])[0, 1]

print(f"  Correlation of dv/dt vs GPS_LonAcc: {corr_positive:.4f}")
if corr_positive > 0.5:
    print("  -> GPS LonAcc is POSITIVE for forward acceleration (confirmed)")
elif corr_positive < -0.5:
    print("  -> GPS LonAcc is NEGATIVE for forward acceleration (INVERTED!)")
else:
    print("  -> Weak correlation -- noisy or phase-shifted")

# Also check: during known acceleration (high throttle), is GPS LonAcc positive?
mask_hard_accel = (d["LVCU Torque Req"] > 50) & (d["GPS Speed"] > 10) & (d["GPS Speed"] < 40)
if mask_hard_accel.sum() > 0:
    mean_gps_lon_accel = d.loc[mask_hard_accel, "GPS LonAcc"].mean()
    print(f"  During hard acceleration (torque > 50 Nm): mean GPS LonAcc = {mean_gps_lon_accel:+.4f} g")
    print(f"  -> {'POSITIVE (correct)' if mean_gps_lon_accel > 0 else 'NEGATIVE (inverted!)'}")

# During known coasting/deceleration
mask_coast_check = (
    (d["LVCU Torque Req"].abs() < 2)
    & (d["Throttle Pos"] < 5)
    & (d["GPS Speed"] > 20)
)
if mask_coast_check.sum() > 0:
    mean_gps_lon_coast = d.loc[mask_coast_check, "GPS LonAcc"].mean()
    print(f"  During coasting (zero throttle): mean GPS LonAcc = {mean_gps_lon_coast:+.4f} g")
    print(f"  -> {'NEGATIVE (correct - decelerating)' if mean_gps_lon_coast < 0 else 'POSITIVE (inverted!)'}")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Check for regen during "coasting" -- maybe the car regens
# even when throttle is zero
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 2: Is there regen/engine-braking during coasting? ---")

mask_coast2 = (
    (d["Throttle Pos"] < 5.0)
    & (d["GPS Speed"] > 10.0)
    & d["Torque Feedback"].notna()
    & d["Pack Current"].notna()
)
dc = d[mask_coast2]

print(f"  Coasting samples (throttle < 5%): {len(dc)}")
print(f"  Mean Torque Feedback:  {dc['Torque Feedback'].mean():+.2f} Nm")
print(f"  Mean Torque Command:   {dc['Torque Command'].mean():+.2f} Nm")
print(f"  Mean LVCU Torque Req:  {dc['LVCU Torque Req'].mean():+.2f} Nm")
print(f"  Mean Pack Current:     {dc['Pack Current'].mean():+.2f} A")
print(f"  Mean MCU DC Current:   {dc['MCU DC Current'].mean():+.2f} A")

# Check if Torque Feedback is negative during coasting (regen)
n_neg_torque = (dc["Torque Feedback"] < -2).sum()
n_neg_current = (dc["Pack Current"] < -1).sum()
print(f"  Rows with Torque Feedback < -2 Nm: {n_neg_torque} ({n_neg_torque/len(dc)*100:.1f}%)")
print(f"  Rows with Pack Current < -1 A: {n_neg_current} ({n_neg_current/len(dc)*100:.1f}%)")

if n_neg_torque > len(dc) * 0.1:
    mean_neg_torque = dc.loc[dc["Torque Feedback"] < -2, "Torque Feedback"].mean()
    print(f"  -> REGEN IS ACTIVE during coasting! Mean regen torque: {mean_neg_torque:.1f} Nm")
    regen_force = abs(mean_neg_torque) * GEAR_RATIO * 0.97 / TIRE_RADIUS_M
    print(f"  -> This adds ~{regen_force:.0f} N of braking force the sim doesn't account for!")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: What is the ACTUAL motor torque during coasting?
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 3: Motor behavior during coast/low-throttle ---")

# The inverter might apply drag torque even at zero command
# Check Torque Feedback vs LVCU Torque Req
for speed_lo, speed_hi in [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]:
    mask_spd = (
        (d["Throttle Pos"] < 5.0)
        & (d["GPS Speed"] >= speed_lo)
        & (d["GPS Speed"] < speed_hi)
        & d["Torque Feedback"].notna()
    )
    grp = d[mask_spd]
    if len(grp) < 5:
        continue
    print(f"  {speed_lo}-{speed_hi} km/h coast: N={len(grp):>4}  "
          f"TorqueCmd={grp['Torque Command'].mean():>+6.1f} Nm  "
          f"TorqueFb={grp['Torque Feedback'].mean():>+6.1f} Nm  "
          f"LVCU={grp['LVCU Torque Req'].mean():>+5.1f} Nm  "
          f"PackI={grp['Pack Current'].mean():>+5.1f} A  "
          f"IdFb={grp['Id Feeback'].mean():>+5.2f} A  "
          f"IqFb={grp['Iq Feedback'].mean():>+5.2f} A")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Torque Feedback as actual motor torque (includes drag torque)
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 4: Using Torque Feedback as actual motor torque ---")
print("  If the inverter applies negative torque during coast, this is")
print("  an additional resistance the sim must model.")

d2 = d[d["GPS Speed"] > 5.0].copy()
d2["torque_fb"] = d2["Torque Feedback"].fillna(0)

# Re-derive resistance using Torque Feedback as actual motor output
d2["F_drive_actual"] = d2["torque_fb"].clip(lower=0) * GEAR_RATIO * 0.97 / TIRE_RADIUS_M
d2["F_regen_actual"] = d2["torque_fb"].clip(upper=0) * GEAR_RATIO * 0.97 / TIRE_RADIUS_M  # negative

# Speed derivative for accel
d2["a_lon"] = d2["GPS LonAcc"] * G

# F_net = F_drive + F_regen - F_resist
# => F_resist = F_drive + F_regen - m*a
d2["F_total_motor"] = d2["torque_fb"] * GEAR_RATIO * 0.97 / TIRE_RADIUS_M  # can be negative (regen)
d2["F_resist_actual"] = d2["F_total_motor"] - MASS_KG * d2["a_lon"]

# Sim resistance
d2["speed_ms"] = d2["GPS Speed"] / 3.6
d2["F_drag"] = 0.5 * RHO * CDA * d2["speed_ms"]**2
d2["F_downforce"] = 0.5 * RHO * CLA * d2["speed_ms"]**2
d2["F_rr"] = (MASS_KG * G + d2["F_downforce"]) * CRR
d2["a_lat_abs"] = d2["GPS LatAcc"].abs() * G
d2["F_lat"] = MASS_KG * d2["a_lat_abs"]
MU_PEAK = 1.5
ALPHA_PEAK = 0.15
C_ALPHA_TOTAL = MASS_KG * G * MU_PEAK / ALPHA_PEAK
d2["F_corner"] = d2["F_lat"]**2 / C_ALPHA_TOTAL
d2["F_resist_sim"] = d2["F_drag"] + d2["F_rr"] + d2["F_corner"]

# Compare using Torque Feedback
speed_bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
d2["speed_bin"] = pd.cut(
    d2["GPS Speed"],
    bins=speed_bins,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins[:-1], speed_bins[1:])],
)

# Straight sections (low lat accel) for clean comparison
mask_straight = d2["GPS LatAcc"].abs() < 0.2
ds = d2[mask_straight]

print(f"\n  Straight-line data (|lat_g| < 0.2): {len(ds)}")
print(f"\n{'Speed':>8} | {'N':>5} | {'F_resist_real':>13} | {'F_resist_sim':>12} | {'Ratio':>6} | {'TorqueFb':>9} | {'Accel':>8}")
print("-" * 80)

for name, grp in ds.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    fr = grp["F_resist_actual"].mean()
    fs = grp["F_resist_sim"].mean()
    ratio = fs / fr if abs(fr) > 1 else 0
    tfb = grp["torque_fb"].mean()
    acc = grp["a_lon"].mean() / G
    print(f"{name:>8} | {len(grp):>5} | {fr:>10.1f} N  | {fs:>9.1f} N  | {ratio:>5.2f} | {tfb:>+7.1f} Nm | {acc:>+6.4f}g")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Isolate just the coasting drag torque
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 5: Inverter drag torque during coasting ---")

mask_coast_clean = (
    (d["Throttle Pos"] < 3.0)
    & (d["GPS Speed"] > 15.0)
    & d["Torque Feedback"].notna()
    & d["Torque Command"].notna()
)
dc_clean = d[mask_coast_clean]

print(f"\n  Clean coast samples (throttle < 3%, speed > 15): {len(dc_clean)}")

# Histogramming the torque feedback
torque_fb_vals = dc_clean["Torque Feedback"]
print(f"\n  Torque Feedback distribution during coasting:")
print(f"    Mean:   {torque_fb_vals.mean():+.2f} Nm")
print(f"    Median: {torque_fb_vals.median():+.2f} Nm")
print(f"    Min:    {torque_fb_vals.min():+.2f} Nm")
print(f"    Max:    {torque_fb_vals.max():+.2f} Nm")
print(f"    Std:    {torque_fb_vals.std():.2f} Nm")

# Breakdown by percentile
for pct in [5, 25, 50, 75, 95]:
    val = torque_fb_vals.quantile(pct / 100)
    force = val * GEAR_RATIO * 0.97 / TIRE_RADIUS_M
    print(f"    P{pct:02d}:    {val:+.1f} Nm  -> {force:+.1f} N at wheel")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: What does LVCU Torque Req look like during coast?
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 6: LVCU Torque Req during 'coasting' ---")

# Maybe LVCU sends non-zero torque even at zero throttle?
mask_zero_throttle = (d["Throttle Pos"] < 2.0) & (d["GPS Speed"] > 10.0)
dzt = d[mask_zero_throttle]

print(f"  Zero-throttle samples (throttle < 2%): {len(dzt)}")
print(f"  Mean LVCU Torque Req:  {dzt['LVCU Torque Req'].mean():+.2f} Nm")
print(f"  Mean Torque Command:   {dzt['Torque Command'].mean():+.2f} Nm")
print(f"  Mean Torque Feedback:  {dzt['Torque Feedback'].mean():+.2f} Nm")
print(f"  Mean MCU Torque Limit: {dzt['MCU Torque Limit'].mean():+.2f} Nm")

# Check Torque Command distribution
tc_vals = dzt["Torque Command"]
print(f"\n  Torque Command distribution (zero throttle):")
for pct in [5, 25, 50, 75, 95]:
    print(f"    P{pct:02d}: {tc_vals.quantile(pct/100):+.1f} Nm")

# ══════════════════════════════════════════════════════════════════════
# STEP 7: THE COMPLETE FORCE BUDGET
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STEP 7: COMPLETE FORCE BUDGET (using Torque Feedback, all data)")
print("=" * 90)

# For ALL moving samples, compute:
# F_motor = TorqueFeedback * G * eta_gearbox / r
# F_resist_implied = F_motor - m * a_GPS
# F_resist_sim = drag + rr + cornering

d3 = d[(d["GPS Speed"] > 5) & d["Torque Feedback"].notna() & d["GPS LonAcc"].notna()].copy()
d3["speed_ms"] = d3["GPS Speed"] / 3.6
d3["a_lon"] = d3["GPS LonAcc"] * G
d3["F_motor"] = d3["Torque Feedback"] * GEAR_RATIO * 0.97 / TIRE_RADIUS_M
d3["F_resist_implied"] = d3["F_motor"] - MASS_KG * d3["a_lon"]

d3["F_drag"] = 0.5 * RHO * CDA * d3["speed_ms"]**2
d3["F_downforce"] = 0.5 * RHO * CLA * d3["speed_ms"]**2
d3["F_rr"] = (MASS_KG * G + d3["F_downforce"]) * CRR
d3["a_lat_abs"] = d3["GPS LatAcc"].fillna(0).abs() * G
d3["F_lat"] = MASS_KG * d3["a_lat_abs"]
d3["F_corner"] = d3["F_lat"]**2 / C_ALPHA_TOTAL
d3["F_resist_sim"] = d3["F_drag"] + d3["F_rr"] + d3["F_corner"]

# Unexplained resistance
d3["F_unexplained"] = d3["F_resist_implied"] - d3["F_resist_sim"]

# Speed bins
d3["speed_bin"] = pd.cut(
    d3["GPS Speed"],
    bins=speed_bins,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins[:-1], speed_bins[1:])],
)

print(f"\nTotal samples: {len(d3)}")
print(f"\n{'Speed':>8} | {'N':>5} | {'F_real':>8} | {'F_sim':>8} | {'Unexlnd':>8} | {'Drag':>6} | {'RR':>6} | {'Corner':>7} | {'TorqFb':>7}")
print("-" * 90)

for name, grp in d3.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    fr = grp["F_resist_implied"].mean()
    fs = grp["F_resist_sim"].mean()
    unex = grp["F_unexplained"].mean()
    drag = grp["F_drag"].mean()
    rr = grp["F_rr"].mean()
    corner = grp["F_corner"].mean()
    tfb = grp["Torque Feedback"].mean()
    print(f"{name:>8} | {len(grp):>5} | {fr:>6.1f} N | {fs:>6.1f} N | {unex:>+6.1f} N | {drag:>5.1f} | {rr:>5.1f} | {corner:>6.1f} | {tfb:>+5.1f} Nm")

# Overall
fr_tot = d3["F_resist_implied"].mean()
fs_tot = d3["F_resist_sim"].mean()
unex_tot = d3["F_unexplained"].mean()
print(f"\n  OVERALL: Real resist = {fr_tot:.1f} N, Sim resist = {fs_tot:.1f} N, Unexplained = {unex_tot:+.1f} N")
print(f"  The sim UNDER-PREDICTS resistance by {(fr_tot - fs_tot):.0f} N ({(fr_tot/fs_tot - 1)*100:.0f}%)")

# ══════════════════════════════════════════════════════════════════════
# STEP 8: What's the unexplained resistance? Grade?
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 8: Could slope/grade explain the unexplained resistance? ---")
if "GPS Slope" in d3.columns:
    print(f"  GPS Slope available")
    slope_vals = d3["GPS Slope"].dropna()
    print(f"  Mean GPS Slope:   {slope_vals.mean():.2f} deg")
    print(f"  Std GPS Slope:    {slope_vals.std():.2f} deg")
    print(f"  Min/Max:          {slope_vals.min():.2f} / {slope_vals.max():.2f} deg")
    # Grade force = m * g * sin(slope_deg * pi/180)
    mean_grade_force = MASS_KG * G * np.sin(np.radians(slope_vals.mean()))
    max_grade_force = MASS_KG * G * np.sin(np.radians(slope_vals.abs().quantile(0.95)))
    print(f"  Mean grade force:   {mean_grade_force:+.1f} N")
    print(f"  P95 |grade| force:  {max_grade_force:+.1f} N")
else:
    print("  No GPS Slope column")

# ══════════════════════════════════════════════════════════════════════
# STEP 9: Cross-check with Torque Feedback vs LVCU Torque Req
# ══════════════════════════════════════════════════════════════════════
print("\n--- STEP 9: Torque Feedback vs LVCU Torque Req ---")
print("  If TorqueFb < LVCU Req during driving, the motor isn't producing")
print("  what was commanded (current limit, field weakening, etc.)")

mask_driving = (d["GPS Speed"] > 10) & (d["LVCU Torque Req"] > 10) & d["Torque Feedback"].notna()
dd = d[mask_driving]

print(f"\n  Driving samples (torque req > 10, speed > 10): {len(dd)}")
print(f"  Mean LVCU Torque Req:  {dd['LVCU Torque Req'].mean():.1f} Nm")
print(f"  Mean Torque Command:   {dd['Torque Command'].mean():.1f} Nm")
print(f"  Mean Torque Feedback:  {dd['Torque Feedback'].mean():.1f} Nm")
print(f"  Mean Torque Limit:     {dd['MCU Torque Limit'].mean():.1f} Nm")
print(f"  Ratio Feedback/Req:    {dd['Torque Feedback'].mean() / dd['LVCU Torque Req'].mean():.3f}")

dd2 = dd.copy()
dd2["speed_bin"] = pd.cut(dd2["GPS Speed"], bins=speed_bins,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins[:-1], speed_bins[1:])])

print(f"\n{'Speed':>8} | {'N':>5} | {'LVCU Req':>9} | {'TorqueCmd':>10} | {'TorqueFb':>9} | {'TrqLimit':>9} | {'Fb/Req':>6} | {'RPM':>6}")
print("-" * 80)

for name, grp in dd2.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    lvcu = grp["LVCU Torque Req"].mean()
    tcmd = grp["Torque Command"].mean()
    tfb = grp["Torque Feedback"].mean()
    tlim = grp["MCU Torque Limit"].mean()
    ratio = tfb / lvcu if lvcu > 1 else 0
    rpm = grp["Motor RPM"].mean()
    print(f"{name:>8} | {len(grp):>5} | {lvcu:>7.1f} Nm | {tcmd:>8.1f} Nm | {tfb:>7.1f} Nm | {tlim:>7.1f} Nm | {ratio:>5.2f} | {rpm:>5.0f}")

# ══════════════════════════════════════════════════════════════════════
# FINAL: QUANTIFIED ROOT CAUSES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("QUANTIFIED ROOT CAUSES")
print("=" * 90)

# 1. Efficiency double-dip in wheel_force
eff_force_loss = d3["Torque Feedback"].clip(lower=0).mean() * GEAR_RATIO * (0.97 - 0.92) / TIRE_RADIUS_M
print(f"\n1. EFFICIENCY DOUBLE-DIP in wheel_force():")
print(f"   Using 0.92 instead of 0.97 loses {eff_force_loss:.1f} N of drive force")
print(f"   This is a {(0.97/0.92 - 1)*100:.1f}% error in drive force")

# 2. Missing resistance
print(f"\n2. MISSING RESISTANCE:")
print(f"   Real resistance: {fr_tot:.0f} N (from Torque Feedback + GPS accel)")
print(f"   Sim resistance:  {fs_tot:.0f} N (drag + RR + cornering)")
print(f"   Gap:             {fr_tot - fs_tot:.0f} N ({(fr_tot/fs_tot - 1)*100:.0f}% under-predicted)")
print(f"   This is MUCH larger than the drive force error.")

# 3. What the gap could be
print(f"\n3. POSSIBLE SOURCES OF MISSING RESISTANCE:")
rr_to_match = (fr_tot - fs_tot) + (MASS_KG * G + d3["F_downforce"].mean()) * CRR
normal_force = MASS_KG * G + d3["F_downforce"].mean()
crr_needed = fr_tot / normal_force  # crude: if we attribute ALL resistance to RR
print(f"   If attributed to rolling resistance: Crr = {crr_needed:.4f} (vs current {CRR})")
print(f"   Hoosier LC0 on FSAE car: typical Crr = 0.015-0.025")
print(f"   But with 10\" wheels + low pressure: could be 0.025-0.035")
print(f"   Remaining gap likely: drivetrain drag, bearing friction, chain tension")
