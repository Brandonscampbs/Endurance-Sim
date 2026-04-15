"""Root-cause force analysis: back-derive real forces from GPS accel data
and compare against the sim's force model.

Answers: is the sim too slow because of too much resistance, too little
drive force, or both?  And which resistance component is wrong?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from fsae_sim.data.loader import load_cleaned_csv

# ── Vehicle parameters (from ct16ev.yaml / DSS) ──────────────────────
MASS_KG = 288.0
GEAR_RATIO = 3.6363
TIRE_RADIUS_M = 0.228
DRIVETRAIN_EFF = 0.92  # fixed scalar (used as baseline)
CDA = 1.502            # drag_coefficient * frontal_area = 1.502 * 1.0
CLA = 2.18             # downforce coefficient (ClA)
ROLLING_RESISTANCE = 0.015
AIR_DENSITY = 1.225
G = 9.81

# Rotational inertia correction (from VehicleDynamics.__init__)
ROTOR_INERTIA = 0.06
WHEEL_INERTIA = 0.3
j_eff = ROTOR_INERTIA * GEAR_RATIO**2 * DRIVETRAIN_EFF + 4 * WHEEL_INERTIA
M_EFFECTIVE = MASS_KG + j_eff / (TIRE_RADIUS_M**2)

# Motor efficiency map (gearbox * motor+inverter)
GEARBOX_EFF = 0.97

print("=" * 80)
print("FORCE ROOT-CAUSE ANALYSIS: Real GPS Accel vs Sim Force Model")
print("=" * 80)
print(f"\nVehicle mass:        {MASS_KG} kg")
print(f"Effective mass:      {M_EFFECTIVE:.1f} kg (with rotational inertia)")
print(f"CdA:                 {CDA}")
print(f"ClA:                 {CLA}")
print(f"Rolling resistance:  {ROLLING_RESISTANCE}")
print(f"Gear ratio:          {GEAR_RATIO}")
print(f"Drivetrain eff:      {DRIVETRAIN_EFF}")
print(f"Gearbox eff:         {GEARBOX_EFF}")

# ── Load telemetry ────────────────────────────────────────────────────
_, df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
print(f"\nLoaded {len(df)} telemetry rows")

# ── Filter to driving periods ─────────────────────────────────────────
# GPS Speed > 5 km/h to avoid standstill noise
mask_moving = df["GPS Speed"] > 5.0
# Need valid acceleration and torque data
mask_valid = (
    df["GPS LonAcc"].notna()
    & df["LVCU Torque Req"].notna()
    & df["GPS LatAcc"].notna()
    & df["Motor RPM"].notna()
)
d = df[mask_moving & mask_valid].copy()
print(f"After filtering (speed > 5 km/h, valid data): {len(d)} rows")

# ── Derived quantities ────────────────────────────────────────────────
d["speed_ms"] = d["GPS Speed"] / 3.6
d["a_lon_ms2"] = d["GPS LonAcc"] * G       # GPS LonAcc is in g
d["a_lat_ms2"] = d["GPS LatAcc"].abs() * G  # lateral (always positive magnitude)
d["motor_rpm_actual"] = d["Motor RPM"]

# ── 1. Real net force from GPS acceleration ───────────────────────────
# F_net = m_eff * a_lon  (using effective mass since accel includes rotational)
# Actually: GPS measures chassis acceleration, which is the linear component.
# The rotational inertia matters for how MUCH force is needed to produce that
# accel, but GPS LonAcc measures the actual chassis accel.
# So: F_net_linear = m * a_lon (chassis force)
# But the motor also has to accelerate the rotating parts.
# When we compute F_drive and subtract F_net to get resistance, we should
# use the SAME mass basis.  Let's use m_chassis = 288 kg for F_net since
# GPS measures chassis accel, and note that the sim uses m_effective for
# its acceleration calc.
d["F_net_real"] = MASS_KG * d["a_lon_ms2"]  # Net force on chassis (N)

# ── 2. Drive force from LVCU Torque Req ───────────────────────────────
# LVCU Torque Req is the final torque command sent to the inverter (Nm at motor)
# Drive force at wheels = torque * gear_ratio * efficiency / tire_radius
# Using fixed drivetrain efficiency first, then motor map
d["F_drive_fixed_eff"] = (
    d["LVCU Torque Req"] * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M
)

# For accelerating periods, also compute with gearbox-only efficiency
# (motor+inverter eff affects power, not force -- the torque command IS the
# motor shaft torque, gear losses reduce wheel torque)
d["F_drive_gearbox_only"] = (
    d["LVCU Torque Req"] * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
)

# ── CRITICAL INSIGHT ──────────────────────────────────────────────────
# The sim's wheel_force() method uses:
#   wheel_force = motor_torque * gear_ratio * drivetrain_efficiency / tire_radius
# But drivetrain_efficiency = 0.92 includes motor+inverter losses.
# The LVCU Torque Req IS the motor shaft torque command -- it's what the
# motor actually produces.  The only loss between motor shaft and wheels
# is the GEARBOX (~3% loss), not the full 8% drivetrain loss.
# Using 0.92 instead of 0.97 loses 5% of drive force!

# ── 3. Back-derive real total resistance ──────────────────────────────
# F_drive - F_net = F_resist  (for accelerating, coasting, everything)
# Use gearbox-only efficiency (correct physics)
d["F_resist_real_gearbox"] = d["F_drive_gearbox_only"] - d["F_net_real"]
d["F_resist_real_fixed"] = d["F_drive_fixed_eff"] - d["F_net_real"]

# ── 4. Sim's predicted resistance ────────────────────────────────────
# Aero drag
d["F_drag_sim"] = 0.5 * AIR_DENSITY * CDA * d["speed_ms"]**2

# Rolling resistance (includes downforce-augmented normal force)
d["F_downforce"] = 0.5 * AIR_DENSITY * CLA * d["speed_ms"]**2
d["F_rr_sim"] = (MASS_KG * G + d["F_downforce"]) * ROLLING_RESISTANCE

# Cornering drag (analytical formula from dynamics.py)
# F_lat = m * v^2 * kappa, but we have actual lateral accel from GPS
# Cornering drag = F_lat^2 / C_alpha_total
# where C_alpha_total = m * g * mu_peak / alpha_peak
MU_PEAK = 1.5
ALPHA_PEAK = 0.15  # rad
C_ALPHA_TOTAL = MASS_KG * G * MU_PEAK / ALPHA_PEAK
d["F_lat_real"] = MASS_KG * d["a_lat_ms2"]
d["F_cornering_drag_sim"] = d["F_lat_real"]**2 / C_ALPHA_TOTAL

# Total sim resistance
d["F_resist_sim"] = d["F_drag_sim"] + d["F_rr_sim"] + d["F_cornering_drag_sim"]

# ── 5. Analysis by speed bins ────────────────────────────────────────
speed_bins_kmh = [5, 10, 15, 20, 25, 30, 35, 40, 45]
d["speed_bin"] = pd.cut(
    d["GPS Speed"],
    bins=speed_bins_kmh,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins_kmh[:-1], speed_bins_kmh[1:])],
)

print("\n" + "=" * 80)
print("SECTION 1: RESISTANCE COMPARISON (Real back-derived vs Sim predicted)")
print("=" * 80)
print("  F_resist_real = F_drive(gearbox_eff) - F_net(GPS)")
print("  F_resist_sim  = drag + rolling_resistance + cornering_drag")
print()

# Filter to acceleration periods only for cleanest resistance analysis
# (during coasting, LVCU torque = 0, so F_resist_real = -F_net which is just decel)
mask_accel = (d["LVCU Torque Req"] > 2.0) & (d["a_lon_ms2"] > -5.0)
d_accel = d[mask_accel]

print(f"Analyzing {len(d_accel)} accelerating rows (torque > 2 Nm)\n")

header = f"{'Speed Bin':>10} | {'Count':>5} | {'F_resist_real':>13} | {'F_resist_sim':>12} | {'Difference':>10} | {'% Over':>7} | {'Drag':>6} | {'RR':>6} | {'Corner':>8}"
print(header)
print("-" * len(header))

grouped_accel = d_accel.groupby("speed_bin", observed=True)
for name, group in grouped_accel:
    if len(group) < 10:
        continue
    fr_real = group["F_resist_real_gearbox"].mean()
    fr_sim = group["F_resist_sim"].mean()
    diff = fr_sim - fr_real
    pct = (diff / fr_real * 100) if abs(fr_real) > 1 else 0
    drag = group["F_drag_sim"].mean()
    rr = group["F_rr_sim"].mean()
    corner = group["F_cornering_drag_sim"].mean()
    print(f"{name:>10} | {len(group):>5} | {fr_real:>10.1f} N  | {fr_sim:>9.1f} N  | {diff:>+8.1f} N | {pct:>+6.1f}% | {drag:>5.1f} | {rr:>5.1f} | {corner:>7.1f}")

# ── 6. Drive force analysis ──────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: DRIVE FORCE -- Fixed eff (0.92) vs Gearbox-only (0.97)")
print("=" * 80)
print("  Shows how much force the sim LOSES by applying full drivetrain_eff")
print("  to motor torque command (which is already the shaft output).")
print()

header2 = f"{'Speed Bin':>10} | {'Count':>5} | {'F_drive(0.92)':>13} | {'F_drive(0.97)':>13} | {'Lost Force':>10} | {'% Lost':>7} | {'Mean Torque':>11}"
print(header2)
print("-" * len(header2))

for name, group in grouped_accel:
    if len(group) < 10:
        continue
    fd_fixed = group["F_drive_fixed_eff"].mean()
    fd_gearbox = group["F_drive_gearbox_only"].mean()
    lost = fd_gearbox - fd_fixed
    pct = (lost / fd_gearbox * 100) if fd_gearbox > 1 else 0
    torq = group["LVCU Torque Req"].mean()
    print(f"{name:>10} | {len(group):>5} | {fd_fixed:>10.1f} N  | {fd_gearbox:>10.1f} N  | {lost:>+8.1f} N | {pct:>+6.1f}% | {torq:>8.1f} Nm")

# ── 7. COASTING analysis ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: COASTING DECELERATION (zero torque) -- Pure resistance check")
print("=" * 80)
print("  When LVCU torque ~ 0 and throttle ~ 0, deceleration = resistance / mass")
print("  This gives the cleanest resistance measurement (no drive force uncertainty)")
print()

mask_coast = (
    (d["LVCU Torque Req"].abs() < 2.0)
    & (d["Throttle Pos"] < 5.0)
    & (d["GPS Speed"] > 8.0)  # need enough speed for meaningful resistance
)
d_coast = d[mask_coast]
print(f"Analyzing {len(d_coast)} coasting rows (torque < 2 Nm, throttle < 5%)\n")

# During coasting: F_net = -F_resist (decel from resistance)
d_coast = d_coast.copy()
d_coast["F_resist_coast_real"] = -d_coast["F_net_real"]  # flip sign: decel -> positive resistance

header3 = f"{'Speed Bin':>10} | {'Count':>5} | {'F_resist(GPS)':>13} | {'F_resist_sim':>12} | {'Difference':>10} | {'% Over':>7} | {'Decel (g)':>9} | {'Drag':>6} | {'RR':>6} | {'Corner':>8}"
print(header3)
print("-" * len(header3))

grouped_coast = d_coast.groupby("speed_bin", observed=True)
for name, group in grouped_coast:
    if len(group) < 5:
        continue
    fr_real = group["F_resist_coast_real"].mean()
    fr_sim = group["F_resist_sim"].mean()
    diff = fr_sim - fr_real
    pct = (diff / fr_real * 100) if abs(fr_real) > 1 else 0
    decel_g = group["GPS LonAcc"].mean()
    drag = group["F_drag_sim"].mean()
    rr = group["F_rr_sim"].mean()
    corner = group["F_cornering_drag_sim"].mean()
    print(f"{name:>10} | {len(group):>5} | {fr_real:>10.1f} N  | {fr_sim:>9.1f} N  | {diff:>+8.1f} N | {pct:>+6.1f}% | {decel_g:>+8.4f} | {drag:>5.1f} | {rr:>5.1f} | {corner:>7.1f}")

# ── 8. Effective mass analysis ────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: EFFECTIVE MASS CHECK")
print("=" * 80)
print(f"  Chassis mass:     {MASS_KG:.1f} kg")
print(f"  Rotational J_eff: {j_eff:.3f} kg*m^2")
print(f"  Effective mass:   {M_EFFECTIVE:.1f} kg  (+{M_EFFECTIVE - MASS_KG:.1f} kg rotational)")
print(f"  Sim uses m_eff for acceleration calc: a = F_net / {M_EFFECTIVE:.1f}")
print(f"  GPS measures CHASSIS accel, not accounting for rotational inertia")
print(f"  This means the sim needs MORE force to produce the same accel")
print(f"  Rotational inertia overhead: {(M_EFFECTIVE/MASS_KG - 1)*100:.1f}%")

# ── 9. Drive force pipeline trace ────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: SIM DRIVE FORCE PIPELINE TRACE (at median conditions)")
print("=" * 80)

median_torque = d_accel["LVCU Torque Req"].median()
median_speed = d_accel["speed_ms"].median()
median_rpm = d_accel["motor_rpm_actual"].median()

print(f"\nMedian operating point: {median_torque:.1f} Nm, {median_speed:.1f} m/s ({median_speed*3.6:.1f} km/h), {median_rpm:.0f} RPM")

# What the sim does (engine.py line 212):
#   drive_f = self.powertrain.wheel_force(motor_torque)
# wheel_force = motor_torque * gear_ratio * drivetrain_efficiency / tire_radius
f_sim_pipeline = median_torque * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M

# What physics says (gearbox loss only):
f_physics = median_torque * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M

# What we'd get with NO efficiency loss (upper bound):
f_no_loss = median_torque * GEAR_RATIO * 1.0 / TIRE_RADIUS_M

print(f"\n  Motor torque command:     {median_torque:.1f} Nm")
print(f"  x gear ratio ({GEAR_RATIO}):    {median_torque * GEAR_RATIO:.1f} Nm at wheel")
print(f"  / tire radius ({TIRE_RADIUS_M}m):    {f_no_loss:.1f} N (no losses)")
print(f"  x gearbox eff (0.97):     {f_physics:.1f} N (correct: gearbox loss only)")
print(f"  x drivetrain eff (0.92):  {f_sim_pipeline:.1f} N (sim uses this)")
print(f"  Force lost by sim:        {f_physics - f_sim_pipeline:.1f} N ({(1 - DRIVETRAIN_EFF/GEARBOX_EFF)*100:.1f}%)")
print(f"\n  The torque COMMAND is the motor shaft output.")
print(f"  Motor+inverter efficiency affects POWER draw, not output torque.")
print(f"  Only gearbox friction reduces the torque reaching the wheels.")

# ── 10. Overall impact summary ────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: OVERALL IMPACT SUMMARY")
print("=" * 80)

# Average during acceleration
mean_F_drive_sim = d_accel["F_drive_fixed_eff"].mean()
mean_F_drive_correct = d_accel["F_drive_gearbox_only"].mean()
mean_F_resist_real = d_accel["F_resist_real_gearbox"].mean()
mean_F_resist_sim = d_accel["F_resist_sim"].mean()
mean_F_net_real = d_accel["F_net_real"].mean()

# What the sim gets for net force:
sim_net_force = mean_F_drive_sim - mean_F_resist_sim
correct_net_force = mean_F_drive_correct - mean_F_resist_real

print(f"\nDuring acceleration ({len(d_accel)} samples):")
print(f"  Real net force (GPS):              {mean_F_net_real:>+8.1f} N")
print(f"  Real resistance (back-derived):    {mean_F_resist_real:>+8.1f} N")
print(f"  Sim resistance:                    {mean_F_resist_sim:>+8.1f} N  ({mean_F_resist_sim - mean_F_resist_real:>+.1f} N error)")
print(f"  Sim drive force (eff=0.92):        {mean_F_drive_sim:>+8.1f} N")
print(f"  Correct drive force (eff=0.97):    {mean_F_drive_correct:>+8.1f} N  ({mean_F_drive_correct - mean_F_drive_sim:>+.1f} N lost)")
print(f"  Sim net force (drive-resist):      {sim_net_force:>+8.1f} N")
print(f"  Correct net force:                 {correct_net_force:>+8.1f} N")
print(f"  Net force error:                   {sim_net_force - correct_net_force:>+8.1f} N")

resist_error = mean_F_resist_sim - mean_F_resist_real
drive_error = mean_F_drive_sim - mean_F_drive_correct
total_error = resist_error + drive_error  # negative = sim has less net force

print(f"\n  DECOMPOSITION OF NET FORCE ERROR:")
print(f"    Excess resistance (sim too high):   {resist_error:>+8.1f} N ({resist_error/abs(total_error)*100:>+5.1f}% of total)")
print(f"    Missing drive force (eff too low):   {drive_error:>+8.1f} N ({drive_error/abs(total_error)*100:>+5.1f}% of total)")
print(f"    TOTAL net force deficit:             {total_error:>+8.1f} N")

# Impact on acceleration
sim_accel = sim_net_force / M_EFFECTIVE
correct_accel = correct_net_force / M_EFFECTIVE
real_accel = mean_F_net_real / MASS_KG  # GPS uses chassis mass

print(f"\n  IMPACT ON ACCELERATION:")
print(f"    Real mean accel (GPS):     {real_accel:>+.4f} m/s^2  ({real_accel/G:>+.4f} g)")
print(f"    Sim accel (m_eff={M_EFFECTIVE:.0f}):   {sim_accel:>+.4f} m/s^2  ({sim_accel/G:>+.4f} g)")
print(f"    Correct accel:             {correct_accel:>+.4f} m/s^2  ({correct_accel/G:>+.4f} g)")

# ── 11. Cornering drag deep-dive ─────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 7: CORNERING DRAG ANALYSIS")
print("=" * 80)
print("  Comparing analytical cornering drag formula vs what GPS says")

lat_bins = [0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
d_accel_copy = d_accel.copy()
d_accel_copy["lat_g_bin"] = pd.cut(
    d_accel_copy["GPS LatAcc"].abs(),
    bins=lat_bins,
    labels=[f"{a}-{b}g" for a, b in zip(lat_bins[:-1], lat_bins[1:])],
)

print(f"\n{'Lat Accel':>10} | {'Count':>5} | {'Corner Drag':>11} | {'Total Resist':>12} | {'% of Resist':>11} | {'Mean Speed':>10}")
print("-" * 75)

grouped_lat = d_accel_copy.groupby("lat_g_bin", observed=True)
for name, group in grouped_lat:
    if len(group) < 5:
        continue
    cd = group["F_cornering_drag_sim"].mean()
    tr = group["F_resist_sim"].mean()
    pct = (cd / tr * 100) if tr > 1 else 0
    spd = group["GPS Speed"].mean()
    print(f"{name:>10} | {len(group):>5} | {cd:>8.1f} N  | {tr:>9.1f} N  | {pct:>9.1f}%  | {spd:>7.1f} km/h")

# ── 12. Per-resistance-component breakdown ────────────────────────────
print("\n" + "=" * 80)
print("SECTION 8: RESISTANCE COMPONENT BREAKDOWN (overall means)")
print("=" * 80)

all_moving = d[d["GPS Speed"] > 5.0]
drag_mean = all_moving["F_drag_sim"].mean()
rr_mean = all_moving["F_rr_sim"].mean()
corner_mean = all_moving["F_cornering_drag_sim"].mean()
total_mean = all_moving["F_resist_sim"].mean()

print(f"\n  Aero drag:        {drag_mean:>8.1f} N  ({drag_mean/total_mean*100:>5.1f}%)")
print(f"  Rolling resist:   {rr_mean:>8.1f} N  ({rr_mean/total_mean*100:>5.1f}%)")
print(f"  Cornering drag:   {corner_mean:>8.1f} N  ({corner_mean/total_mean*100:>5.1f}%)")
print(f"  Total sim resist: {total_mean:>8.1f} N  (100.0%)")

# Back-derived real resistance during coasting for comparison
if len(d_coast) > 0:
    real_resist_coast = d_coast["F_resist_coast_real"].mean()
    sim_resist_coast = d_coast["F_resist_sim"].mean()
    print(f"\n  Coasting validation (cleanest data):")
    print(f"    Real resistance (GPS decel): {real_resist_coast:>8.1f} N")
    print(f"    Sim resistance at same pts:  {sim_resist_coast:>8.1f} N")
    print(f"    Sim over-predicts by:        {sim_resist_coast - real_resist_coast:>+8.1f} N ({(sim_resist_coast/real_resist_coast - 1)*100:>+.1f}%)")

# ── 13. What rolling resistance value would match reality? ────────────
print("\n" + "=" * 80)
print("SECTION 9: ROLLING RESISTANCE CALIBRATION (from coasting data)")
print("=" * 80)

# During coasting on straights (low lat accel), resistance = drag + RR
# So RR = F_resist_real - F_drag
mask_straight_coast = (
    (d["LVCU Torque Req"].abs() < 2.0)
    & (d["Throttle Pos"] < 5.0)
    & (d["GPS Speed"] > 10.0)
    & (d["GPS LatAcc"].abs() < 0.15)  # nearly straight
)
d_straight_coast = d[mask_straight_coast].copy()
d_straight_coast["F_resist_coast_real"] = -d_straight_coast["F_net_real"]

if len(d_straight_coast) > 20:
    # RR_real = F_resist_real - F_drag
    d_straight_coast["F_rr_implied"] = (
        d_straight_coast["F_resist_coast_real"] - d_straight_coast["F_drag_sim"]
    )
    # Normal force at each speed
    d_straight_coast["F_normal"] = MASS_KG * G + d_straight_coast["F_downforce"]
    d_straight_coast["crr_implied"] = d_straight_coast["F_rr_implied"] / d_straight_coast["F_normal"]

    crr_mean = d_straight_coast["crr_implied"].mean()
    crr_median = d_straight_coast["crr_implied"].median()
    crr_std = d_straight_coast["crr_implied"].std()

    print(f"\n  Straight-line coasting samples: {len(d_straight_coast)}")
    print(f"  Implied Crr (mean):   {crr_mean:.4f}")
    print(f"  Implied Crr (median): {crr_median:.4f}")
    print(f"  Implied Crr (std):    {crr_std:.4f}")
    print(f"  Current sim Crr:      {ROLLING_RESISTANCE:.4f}")
    print(f"  Difference:           {crr_mean - ROLLING_RESISTANCE:+.4f}")

    if crr_mean < ROLLING_RESISTANCE:
        saved_force = (ROLLING_RESISTANCE - crr_mean) * MASS_KG * G
        print(f"  Reducing Crr to {crr_mean:.4f} saves ~{saved_force:.1f} N of resistance")
else:
    print(f"\n  Only {len(d_straight_coast)} straight coasting samples -- not enough for calibration")

print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print("""
The sim applies drivetrain_efficiency (0.92) to the LVCU torque command when
computing wheel_force(). But the LVCU torque command IS the motor shaft
torque -- it's what the motor actually produces.  The efficiency that
affects force transmission from motor shaft to wheel is ONLY the gearbox
efficiency (~0.97), not the full motor+inverter+gearbox chain (0.92).

Motor+inverter efficiency affects how much ELECTRICAL POWER the battery
must supply to produce that shaft torque, but it does NOT reduce the
mechanical torque output.  The sim's electrical_power() method already
divides by efficiency to get power -- so efficiency is being applied TWICE:
once to reduce force, and once to increase power consumption.

FIX: In PowertrainModel.wheel_force() / wheel_torque(), use gearbox
efficiency (0.97) instead of drivetrain_efficiency (0.92).  The motor
efficiency map should ONLY be used in electrical_power() for energy calc.
""")
