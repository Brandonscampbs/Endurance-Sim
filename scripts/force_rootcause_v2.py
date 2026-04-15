"""Root-cause force analysis v2: carefully handle sign conventions.

GPS LonAcc: positive = forward acceleration, negative = braking/deceleration
LVCU Torque Req: positive = forward drive torque

Key question: the sim is 11.4% too slow.  Is it because:
(a) too much resistance (sim over-predicts drag/rolling/cornering)
(b) too little drive force (efficiency double-dip)
(c) both?

Approach: use coasting data (zero torque) to isolate resistance,
then use acceleration data to check drive force pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from fsae_sim.data.loader import load_cleaned_csv

# ── Parameters ────────────────────────────────────────────────────────
MASS_KG = 288.0
GEAR_RATIO = 3.6363
TIRE_RADIUS_M = 0.228
DRIVETRAIN_EFF = 0.92
CDA = 1.502
CLA = 2.18
CRR = 0.015
RHO = 1.225
G = 9.81
GEARBOX_EFF = 0.97

# Rotational inertia
ROTOR_INERTIA = 0.06
WHEEL_INERTIA = 0.3
j_eff = ROTOR_INERTIA * GEAR_RATIO**2 * DRIVETRAIN_EFF + 4 * WHEEL_INERTIA
M_EFFECTIVE = MASS_KG + j_eff / (TIRE_RADIUS_M**2)

# Cornering drag parameters (from dynamics.py analytical formula)
MU_PEAK = 1.5
ALPHA_PEAK = 0.15
C_ALPHA_TOTAL = MASS_KG * G * MU_PEAK / ALPHA_PEAK

# ── Load data ─────────────────────────────────────────────────────────
_, df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")

print("=" * 90)
print("FORCE ROOT-CAUSE ANALYSIS v2")
print("=" * 90)

# ── Filter ────────────────────────────────────────────────────────────
mask = (
    (df["GPS Speed"] > 5.0)
    & df["GPS LonAcc"].notna()
    & df["LVCU Torque Req"].notna()
    & df["GPS LatAcc"].notna()
)
d = df[mask].copy()
d["speed_ms"] = d["GPS Speed"] / 3.6
d["a_lon"] = d["GPS LonAcc"] * G  # m/s^2, positive = forward
d["a_lat_abs"] = d["GPS LatAcc"].abs() * G
d["motor_rpm"] = d["Motor RPM"]

# ── Sim resistance at each point ──────────────────────────────────────
d["F_drag"] = 0.5 * RHO * CDA * d["speed_ms"]**2
d["F_downforce"] = 0.5 * RHO * CLA * d["speed_ms"]**2
d["F_rr"] = (MASS_KG * G + d["F_downforce"]) * CRR
d["F_lat"] = MASS_KG * d["a_lat_abs"]
d["F_corner"] = d["F_lat"]**2 / C_ALPHA_TOTAL
d["F_resist_sim"] = d["F_drag"] + d["F_rr"] + d["F_corner"]

# ── Drive force at each point ─────────────────────────────────────────
d["F_drive_sim"] = d["LVCU Torque Req"].clip(lower=0) * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M
d["F_drive_gearbox"] = d["LVCU Torque Req"].clip(lower=0) * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M

# ══════════════════════════════════════════════════════════════════════
# PART A: COASTING ANALYSIS (cleanest resistance measurement)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART A: COASTING ANALYSIS (torque < 2 Nm, throttle < 5%)")
print("=" * 90)
print("During coasting: decel = resistance / mass")
print("So: F_resist_real = mass * |decel| = -mass * a_lon (since a_lon < 0)")
print()

mask_coast = (
    (d["LVCU Torque Req"].abs() < 2.0)
    & (d["Throttle Pos"] < 5.0)
    & (d["GPS Speed"] > 8.0)
    & (d["a_lon"] < 0)  # must be decelerating
)
dc = d[mask_coast].copy()
dc["F_resist_real"] = -MASS_KG * dc["a_lon"]  # positive resistance

print(f"Coasting samples (decelerating): {len(dc)}")

# Separate straight vs cornering coasting
mask_straight = dc["GPS LatAcc"].abs() < 0.15
mask_corner = dc["GPS LatAcc"].abs() >= 0.15

dc_straight = dc[mask_straight]
dc_corner = dc[mask_corner]

print(f"  Straight coasting (|lat| < 0.15g): {len(dc_straight)}")
print(f"  Cornering coasting (|lat| >= 0.15g): {len(dc_corner)}")

# ── Straight-line coasting: isolates drag + rolling resistance ────────
print("\n--- Straight-line coasting (drag + rolling resistance only) ---")
speed_bins_kmh = [8, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
dc_straight = dc_straight.copy()
dc_straight["speed_bin"] = pd.cut(
    dc_straight["GPS Speed"],
    bins=speed_bins_kmh,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins_kmh[:-1], speed_bins_kmh[1:])],
)

print(f"\n{'Speed':>8} | {'N':>4} | {'F_real':>8} | {'F_sim':>8} | {'Drag':>6} | {'RR':>6} | {'Corner':>7} | {'Sim/Real':>8} | {'Decel(g)':>8}")
print("-" * 85)

for name, grp in dc_straight.groupby("speed_bin", observed=True):
    if len(grp) < 5:
        continue
    fr = grp["F_resist_real"].mean()
    fs = grp["F_resist_sim"].mean()
    drag = grp["F_drag"].mean()
    rr = grp["F_rr"].mean()
    corner = grp["F_corner"].mean()
    ratio = fs / fr if fr > 1 else 0
    decel = grp["GPS LonAcc"].mean()
    print(f"{name:>8} | {len(grp):>4} | {fr:>6.1f} N | {fs:>6.1f} N | {drag:>5.1f} | {rr:>5.1f} | {corner:>6.1f} | {ratio:>7.2f}x | {decel:>+7.4f}")

# ── Crr calibration from straight coasting ────────────────────────────
print("\n--- Rolling resistance calibration ---")
# F_real = drag + RR => RR = F_real - drag
# RR = (m*g + F_downforce) * Crr => Crr = RR / (m*g + F_downforce)
dc_straight2 = dc_straight.copy()
dc_straight2["F_rr_implied"] = dc_straight2["F_resist_real"] - dc_straight2["F_drag"]
dc_straight2["F_normal"] = MASS_KG * G + dc_straight2["F_downforce"]
dc_straight2["crr_implied"] = dc_straight2["F_rr_implied"] / dc_straight2["F_normal"]

# Filter out outliers (some coasting data has bumps/slope)
crr_vals = dc_straight2["crr_implied"]
crr_clean = crr_vals[(crr_vals > -0.05) & (crr_vals < 0.15)]
print(f"  Crr samples (after outlier filter): {len(crr_clean)} / {len(crr_vals)}")
print(f"  Implied Crr mean:   {crr_clean.mean():.4f}")
print(f"  Implied Crr median: {crr_clean.median():.4f}")
print(f"  Implied Crr std:    {crr_clean.std():.4f}")
print(f"  Current sim Crr:    {CRR:.4f}")

# Per speed bin
print(f"\n  {'Speed':>8} | {'Crr implied':>12} | {'N':>4}")
print("  " + "-" * 35)
for name, grp in dc_straight2.groupby("speed_bin", observed=True):
    if len(grp) < 5:
        continue
    crr = grp["crr_implied"].median()
    print(f"  {name:>8} | {crr:>11.4f}  | {len(grp):>4}")

# ══════════════════════════════════════════════════════════════════════
# PART B: ACCELERATION ANALYSIS (drive force pipeline)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART B: ACCELERATION ANALYSIS (torque > 5 Nm, positive accel)")
print("=" * 90)
print("During acceleration: F_drive - F_resist = m * a")
print("=> F_resist_real = F_drive - m * a")
print()

mask_accel = (
    (d["LVCU Torque Req"] > 5.0)
    & (d["a_lon"] > 0.05)  # clearly accelerating
    & (d["GPS Speed"] > 10.0)
)
da = d[mask_accel].copy()
da["F_resist_real"] = da["F_drive_gearbox"] - MASS_KG * da["a_lon"]

print(f"Acceleration samples: {len(da)}")

da["speed_bin"] = pd.cut(
    da["GPS Speed"],
    bins=speed_bins_kmh,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins_kmh[:-1], speed_bins_kmh[1:])],
)

print(f"\n{'Speed':>8} | {'N':>4} | {'F_real':>8} | {'F_sim':>8} | {'Diff':>8} | {'F_drv_sim':>9} | {'F_drv_gbx':>9} | {'Lost':>6} | {'Torque':>7}")
print("-" * 95)

for name, grp in da.groupby("speed_bin", observed=True):
    if len(grp) < 5:
        continue
    fr = grp["F_resist_real"].mean()
    fs = grp["F_resist_sim"].mean()
    diff = fs - fr
    fds = grp["F_drive_sim"].mean()
    fdg = grp["F_drive_gearbox"].mean()
    lost = fdg - fds
    torq = grp["LVCU Torque Req"].mean()
    print(f"{name:>8} | {len(grp):>4} | {fr:>6.1f} N | {fs:>6.1f} N | {diff:>+6.1f} N | {fds:>7.1f} N | {fdg:>7.1f} N | {lost:>+4.1f} N | {torq:>5.1f} Nm")

# ══════════════════════════════════════════════════════════════════════
# PART C: THE FULL PICTURE -- What makes the sim too slow?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART C: FULL PICTURE -- Why is the sim 11.4% too slow?")
print("=" * 90)

# Overall means during powered driving (torque > 2)
mask_powered = d["LVCU Torque Req"] > 2.0
dp = d[mask_powered].copy()

mean_speed = dp["speed_ms"].mean()
mean_torque = dp["LVCU Torque Req"].mean()
mean_rpm = dp["motor_rpm"].mean()
mean_a_lon = dp["a_lon"].mean()
mean_a_lat = dp["a_lat_abs"].mean()

# What GPS says the net force is
F_net_real = MASS_KG * mean_a_lon

# What the sim computes for drive force
F_drive_sim = mean_torque * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M
F_drive_gbx = mean_torque * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M

# Sim resistance
mean_drag = dp["F_drag"].mean()
mean_rr = dp["F_rr"].mean()
mean_corner = dp["F_corner"].mean()
F_resist_sim = mean_drag + mean_rr + mean_corner

# Sim net force
F_net_sim = F_drive_sim - F_resist_sim

# Correct net force (gearbox eff for force, keep sim resistance as-is for now)
F_net_corrected = F_drive_gbx - F_resist_sim

# What net force SHOULD be (from GPS)
# Note: sim uses m_effective, so sim_accel = F_net / m_eff
sim_accel = F_net_sim / M_EFFECTIVE
corrected_accel = F_net_corrected / M_EFFECTIVE
real_accel = mean_a_lon

print(f"\n  Mean operating point:")
print(f"    Speed:       {mean_speed:.1f} m/s ({mean_speed*3.6:.1f} km/h)")
print(f"    Torque:      {mean_torque:.1f} Nm")
print(f"    RPM:         {mean_rpm:.0f}")
print(f"    Lon Accel:   {mean_a_lon:.3f} m/s^2 ({mean_a_lon/G:.4f} g)")
print(f"    Lat Accel:   {mean_a_lat:.3f} m/s^2 ({mean_a_lat/G:.3f} g)")

print(f"\n  Force breakdown:")
print(f"    F_drive (sim, eff=0.92):     {F_drive_sim:>+8.1f} N")
print(f"    F_drive (gearbox, eff=0.97): {F_drive_gbx:>+8.1f} N")
print(f"    Drive force lost by sim:     {F_drive_sim - F_drive_gbx:>+8.1f} N")
print(f"    F_resist (sim total):        {F_resist_sim:>+8.1f} N")
print(f"      Aero drag:                 {mean_drag:>+8.1f} N ({mean_drag/F_resist_sim*100:.0f}%)")
print(f"      Rolling resistance:        {mean_rr:>+8.1f} N ({mean_rr/F_resist_sim*100:.0f}%)")
print(f"      Cornering drag:            {mean_corner:>+8.1f} N ({mean_corner/F_resist_sim*100:.0f}%)")
print(f"    F_net (real, GPS):           {F_net_real:>+8.1f} N")
print(f"    F_net (sim):                 {F_net_sim:>+8.1f} N")
print(f"    F_net (corrected):           {F_net_corrected:>+8.1f} N")

print(f"\n  Acceleration comparison:")
print(f"    Real (GPS):                  {real_accel:>+.4f} m/s^2")
print(f"    Sim (F_net_sim / m_eff):     {sim_accel:>+.4f} m/s^2")
print(f"    Corrected:                   {corrected_accel:>+.4f} m/s^2")

# ══════════════════════════════════════════════════════════════════════
# PART D: EFFECTIVE MASS DOUBLE-CHECK
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART D: EFFECTIVE MASS IMPACT")
print("=" * 90)
print(f"\n  The sim divides net force by m_effective = {M_EFFECTIVE:.1f} kg")
print(f"  but real car has chassis mass = {MASS_KG:.1f} kg")
print(f"  The rotational inertia adds {M_EFFECTIVE - MASS_KG:.1f} kg ({(M_EFFECTIVE/MASS_KG-1)*100:.1f}%)")
print(f"\n  At same net force, sim acceleration is {MASS_KG/M_EFFECTIVE:.3f}x of what chassis-mass would give")
print(f"  This is CORRECT physics -- rotational inertia must be accelerated too")
print(f"  But GPS measures CHASSIS accel, which IS the real forward accel")
print(f"  So using m_effective with chassis accel data: the real F_net = m_eff * a_gps")

# Let's redo with m_effective for F_net
F_net_real_meff = M_EFFECTIVE * mean_a_lon
F_resist_back_derived = F_drive_gbx - F_net_real_meff

print(f"\n  Using m_effective for F_net:")
print(f"    F_net_real = m_eff * a_gps = {F_net_real_meff:.1f} N")
print(f"    F_resist_back_derived = F_drive - F_net = {F_resist_back_derived:.1f} N")
print(f"    F_resist_sim = {F_resist_sim:.1f} N")
print(f"    Difference: {F_resist_sim - F_resist_back_derived:+.1f} N")

# ══════════════════════════════════════════════════════════════════════
# PART E: RECONSIDER -- the sim is too SLOW, meaning it takes MORE time.
# If the sim has MORE net force, it should be FASTER, not slower.
# Let's check what "too slow" means in context.
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART E: REALITY CHECK -- What does '11.4% too slow' mean?")
print("=" * 90)
print("""
  "Too slow" = sim lap times are 11.4% longer than real.
  This means sim speed is LOWER than real speed at equivalent points.
  Which means sim acceleration is LOWER, or sim braking/coasting is MORE.

  Possible causes:
  1. Sim drive force too LOW (efficiency double-dip reduces F_drive)
  2. Sim resistance too HIGH (over-predicted drag/rr/cornering)
  3. Sim cornering speed too LOW (enters corners slower)
  4. Sim coasts too aggressively (enters braking earlier)
  5. Strategy/driver model mismatch (different throttle/brake points)

  The FORCE analysis above shows that at the SAME operating point,
  the sim's drive force (eff=0.92) is 5.2% less than correct (eff=0.97).
  But the sim also has LESS resistance than GPS back-derives.

  CRITICAL: maybe the issue is not in force magnitude but in WHEN
  forces are applied -- the driver model / strategy timing matters.
""")

# ── Let's look at what fraction of time is spent in each state ────────
total = len(d)
n_accel = (d["LVCU Torque Req"] > 5).sum()
n_coast = ((d["LVCU Torque Req"].abs() < 2) & (d["Throttle Pos"] < 5)).sum()
n_brake = (d["FBrakePressure"] > 1.0).sum()
n_other = total - n_accel - n_coast - n_brake

print(f"\n  Telemetry time distribution:")
print(f"    Accelerating (torque > 5 Nm):  {n_accel:>5} ({n_accel/total*100:.1f}%)")
print(f"    Coasting (torque~0, throt~0):  {n_coast:>5} ({n_coast/total*100:.1f}%)")
print(f"    Braking (brake press > 1 bar): {n_brake:>5} ({n_brake/total*100:.1f}%)")
print(f"    Other/transition:              {n_other:>5} ({n_other/total*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# PART F: Speed-matched comparison at IDENTICAL torque commands
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART F: SPEED-MATCHED NET FORCE COMPARISON")
print("=" * 90)
print("  At each telemetry point, compute what the SIM would predict")
print("  for acceleration, and compare to real GPS acceleration.")
print()

# For every point, the sim would compute:
# a_sim = (F_drive_sim - F_resist_sim) / m_effective
d["F_net_sim"] = d["F_drive_sim"] - d["F_resist_sim"]
d["a_sim"] = d["F_net_sim"] / M_EFFECTIVE
d["a_real"] = d["a_lon"]
d["a_error"] = d["a_sim"] - d["a_real"]

# Also with corrected drive force
d["F_net_corrected"] = d["F_drive_gearbox"] - d["F_resist_sim"]
d["a_corrected"] = d["F_net_corrected"] / M_EFFECTIVE
d["a_error_corrected"] = d["a_corrected"] - d["a_real"]

speed_bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
d["speed_bin"] = pd.cut(
    d["GPS Speed"],
    bins=speed_bins,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins[:-1], speed_bins[1:])],
)

print(f"{'Speed':>8} | {'N':>5} | {'a_real':>8} | {'a_sim':>8} | {'a_err':>8} | {'a_corrected':>11} | {'a_err_corr':>10}")
print("-" * 80)

for name, grp in d.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    ar = grp["a_real"].mean()
    asim = grp["a_sim"].mean()
    aerr = grp["a_error"].mean()
    acorr = grp["a_corrected"].mean()
    aerrc = grp["a_error_corrected"].mean()
    print(f"{name:>8} | {len(grp):>5} | {ar:>+6.3f} g | {asim/G:>+6.3f} g | {aerr/G:>+6.3f} g | {acorr/G:>+9.3f} g | {aerrc/G:>+8.3f} g")

# Overall
print(f"\n  Overall mean:")
ar = d["a_real"].mean()
asim = d["a_sim"].mean()
acorr = d["a_corrected"].mean()
print(f"    Real accel:        {ar/G:>+.4f} g ({ar:>+.3f} m/s^2)")
print(f"    Sim accel:         {asim/G:>+.4f} g ({asim:>+.3f} m/s^2)")
print(f"    Corrected accel:   {acorr/G:>+.4f} g ({acorr:>+.3f} m/s^2)")
print(f"    Sim error:         {(asim-ar)/G:>+.4f} g")
print(f"    Corrected error:   {(acorr-ar)/G:>+.4f} g")

# ══════════════════════════════════════════════════════════════════════
# PART G: THE SMOKING GUN -- Compare during THROTTLE-ON periods only
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART G: THROTTLE-ON ANALYSIS (torque > 10 Nm)")
print("=" * 90)

mask_throttle = d["LVCU Torque Req"] > 10.0
dt = d[mask_throttle]

print(f"Samples with torque > 10 Nm: {len(dt)}")
print()

# What speed change does the sim predict vs what actually happened?
# Over the whole dataset, integrate: delta_v = sum(a * dt)
dt2 = dt.copy()
dt2["dt"] = dt2["Time"].diff().fillna(0.05).clip(0, 0.2)

# Time-weighted mean acceleration
total_time = dt2["dt"].sum()
real_dv = (dt2["a_real"] * dt2["dt"]).sum()
sim_dv = (dt2["a_sim"] * dt2["dt"]).sum()
corr_dv = (dt2["a_corrected"] * dt2["dt"]).sum()

print(f"  Total throttle-on time: {total_time:.1f} s")
print(f"  Real delta-v:     {real_dv:>+.1f} m/s")
print(f"  Sim delta-v:      {sim_dv:>+.1f} m/s  (error: {sim_dv - real_dv:>+.1f} m/s)")
print(f"  Corrected delta-v:{corr_dv:>+.1f} m/s  (error: {corr_dv - real_dv:>+.1f} m/s)")

if abs(real_dv) > 0:
    print(f"  Sim over-predicts velocity gain by {(sim_dv/real_dv - 1)*100:+.1f}%")
    print(f"  Corrected over-predicts by {(corr_dv/real_dv - 1)*100:+.1f}%")

# ══════════════════════════════════════════════════════════════════════
# PART H: BACK-DERIVE CdA FROM HIGH-SPEED STRAIGHT COASTING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("PART H: CdA CALIBRATION FROM HIGH-SPEED STRAIGHT COASTING")
print("=" * 90)

# At high speed, drag dominates.  During straight coasting:
# F_resist = F_drag + F_rr
# F_drag = F_resist - F_rr = m*|a| - (m*g + F_df)*Crr
# CdA = 2 * F_drag / (rho * v^2)

mask_hs_coast = (
    (d["LVCU Torque Req"].abs() < 2.0)
    & (d["Throttle Pos"] < 5.0)
    & (d["GPS Speed"] > 30.0)
    & (d["GPS LatAcc"].abs() < 0.15)
    & (d["a_lon"] < 0)
)
dh = d[mask_hs_coast].copy()

if len(dh) > 10:
    dh["F_resist_real"] = -MASS_KG * dh["a_lon"]
    dh["F_rr_est"] = (MASS_KG * G + dh["F_downforce"]) * CRR
    dh["F_drag_real"] = dh["F_resist_real"] - dh["F_rr_est"]
    dh["CdA_implied"] = 2 * dh["F_drag_real"] / (RHO * dh["speed_ms"]**2)

    # Filter outliers
    cda_vals = dh["CdA_implied"]
    cda_clean = cda_vals[(cda_vals > 0) & (cda_vals < 5)]

    print(f"\n  High-speed straight coasting samples: {len(dh)} ({len(cda_clean)} after outlier filter)")
    print(f"  Implied CdA mean:   {cda_clean.mean():.3f}")
    print(f"  Implied CdA median: {cda_clean.median():.3f}")
    print(f"  Implied CdA std:    {cda_clean.std():.3f}")
    print(f"  Current sim CdA:    {CDA:.3f}")
else:
    print(f"  Only {len(dh)} samples -- not enough")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("FINAL ROOT-CAUSE SUMMARY")
print("=" * 90)
print(f"""
KEY FINDINGS:

1. DRIVE FORCE DOUBLE-DIP (confirmed, 5.2% force loss):
   The sim applies drivetrain_efficiency (0.92) to LVCU torque command
   in wheel_force(), but the torque command IS the motor shaft output.
   Only gearbox losses (~3%) reduce wheel torque.  Motor+inverter
   efficiency only affects electrical power draw.
   Impact: -5.2% drive force at all speeds.

2. EFFECTIVE MASS OVERHEAD ({(M_EFFECTIVE/MASS_KG-1)*100:.1f}%):
   m_effective = {M_EFFECTIVE:.1f} kg vs chassis mass {MASS_KG:.1f} kg.
   The sim accelerates a {M_EFFECTIVE/MASS_KG:.3f}x heavier car.
   This is physically correct (rotational inertia must be accelerated),
   but when validating against GPS accel data, the force balance must
   use m_effective consistently.

3. NET FORCE ERROR DIRECTION:
   Surprisingly, during acceleration the sim predicts MORE net force
   than reality (sim accel > GPS accel).  This means the sim would
   actually be FASTER, not slower, at the same speed with the same torque.

   The 11.4% time error likely comes from the DRIVER MODEL (when/how
   much throttle) and CORNER SPEED LIMITS (how fast through turns),
   NOT from the force magnitude at a given operating point.
""")

# Final check: what's the distribution of sim-vs-real accel by driving state?
print("Accel error by driving state:")
for label, mask_fn in [
    ("Throttle (>10 Nm)", d["LVCU Torque Req"] > 10),
    ("Light throttle (2-10 Nm)", (d["LVCU Torque Req"] > 2) & (d["LVCU Torque Req"] <= 10)),
    ("Coasting", (d["LVCU Torque Req"].abs() < 2) & (d["Throttle Pos"] < 5)),
    ("Braking", d["FBrakePressure"] > 1.0),
]:
    subset = d[mask_fn]
    if len(subset) < 10:
        continue
    ar = subset["a_real"].mean() / G
    asim = subset["a_sim"].mean() / G
    print(f"  {label:<30}  N={len(subset):>5}  real={ar:>+.4f}g  sim={asim:>+.4f}g  err={asim-ar:>+.4f}g")
