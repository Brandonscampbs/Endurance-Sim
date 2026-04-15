"""Root-cause v4: Clean analysis with proper outlier filtering.

Key findings so far:
- Torque Feedback has huge outliers (3000+ Nm) at low speed -- CAN noise
- Torque Feedback/LVCU Req ratio is 0.88-0.95 -- motor produces LESS than commanded
- Sim under-predicts resistance significantly
- Drive force efficiency double-dip is 5.2% but resistance gap is much larger

This version:
1. Filters Torque Feedback outliers (cap at 120 Nm, the physical max of EMRAX at 85Nm inverter limit)
2. Uses LVCU Torque Req for drive force (what the sim uses) -- not Torque Feedback
3. Carefully separates the question into force-model accuracy vs driver-model accuracy
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
GEARBOX_EFF = 0.97
DRIVETRAIN_EFF = 0.92

# Cornering drag
MU_PEAK = 1.5
ALPHA_PEAK = 0.15
C_ALPHA_TOTAL = MASS_KG * G * MU_PEAK / ALPHA_PEAK

# Effective mass
ROTOR_INERTIA = 0.06
WHEEL_INERTIA = 0.3
j_eff = ROTOR_INERTIA * GEAR_RATIO**2 * DRIVETRAIN_EFF + 4 * WHEEL_INERTIA
M_EFFECTIVE = MASS_KG + j_eff / (TIRE_RADIUS_M**2)

_, df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")

print("=" * 95)
print("FORCE ROOT-CAUSE ANALYSIS v4 (cleaned, filtered)")
print("=" * 95)

# ── Clean data ────────────────────────────────────────────────────────
d = df[(df["GPS Speed"] > 5) & df["GPS LonAcc"].notna()].copy()
d["speed_ms"] = d["GPS Speed"] / 3.6
d["a_lon"] = d["GPS LonAcc"] * G  # m/s^2, positive = forward
d["a_lat_abs"] = d["GPS LatAcc"].fillna(0).abs() * G

# Cap Torque Feedback at physical limits (EMRAX 228 max ~230 Nm, inverter limits to 85)
d["torque_fb_clean"] = d["Torque Feedback"].clip(-100, 120)

# Sim resistance
d["F_drag"] = 0.5 * RHO * CDA * d["speed_ms"]**2
d["F_downforce"] = 0.5 * RHO * CLA * d["speed_ms"]**2
d["F_rr"] = (MASS_KG * G + d["F_downforce"]) * CRR
d["F_lat"] = MASS_KG * d["a_lat_abs"]
d["F_corner"] = d["F_lat"]**2 / C_ALPHA_TOTAL
d["F_resist_sim"] = d["F_drag"] + d["F_rr"] + d["F_corner"]

# Drive forces
d["F_drive_lvcu_sim"] = d["LVCU Torque Req"].clip(lower=0) * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M
d["F_drive_lvcu_gbx"] = d["LVCU Torque Req"].clip(lower=0) * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M

# Speed bins
speed_bins = [5, 15, 25, 35, 45, 55, 65]
d["speed_bin"] = pd.cut(
    d["GPS Speed"],
    bins=speed_bins,
    labels=[f"{a}-{b}" for a, b in zip(speed_bins[:-1], speed_bins[1:])],
)

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: COASTING on STRAIGHTS -- purest resistance measurement
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 1: STRAIGHT-LINE COASTING (torque < 2, throttle < 3%, |lat| < 0.1g)")
print("=" * 95)
print("This isolates drag + rolling resistance with no drive force or cornering uncertainty.")

mask_pure_coast = (
    (d["LVCU Torque Req"].abs() < 2)
    & (d["Throttle Pos"] < 3)
    & (d["GPS LatAcc"].abs() < 0.1)
    & (d["a_lon"] < 0)  # must be decelerating
    & (d["GPS Speed"] > 15)
    # Also exclude corrupted torque feedback
    & (d["torque_fb_clean"] < 5)
    & (d["torque_fb_clean"] > -5)
)
dpc = d[mask_pure_coast].copy()
dpc["F_resist_real"] = -MASS_KG * dpc["a_lon"]
dpc["F_resist_sim_straight"] = dpc["F_drag"] + dpc["F_rr"]  # no cornering

print(f"\nPure straight coasting samples: {len(dpc)}")

dpc["speed_bin2"] = pd.cut(dpc["GPS Speed"], bins=[15, 25, 35, 45, 55, 65],
    labels=["15-25", "25-35", "35-45", "45-55", "55-65"])

print(f"\n{'Speed':>8} | {'N':>4} | {'F_real':>8} | {'F_sim':>8} | {'Gap':>8} | {'Drag':>6} | {'RR':>6} | {'Decel':>7} | {'Crr_impl':>8}")
print("-" * 85)

for name, grp in dpc.groupby("speed_bin2", observed=True):
    if len(grp) < 3:
        continue
    fr = grp["F_resist_real"].mean()
    fs = grp["F_resist_sim_straight"].mean()
    gap = fr - fs
    drag = grp["F_drag"].mean()
    rr = grp["F_rr"].mean()
    decel = grp["GPS LonAcc"].mean()
    # Implied Crr: F_real = drag + (m*g + downforce)*Crr => Crr = (F_real - drag) / (m*g + downforce)
    normal = (MASS_KG * G + grp["F_downforce"]).mean()
    crr_impl = (fr - drag) / normal
    print(f"{name:>8} | {len(grp):>4} | {fr:>6.1f} N | {fs:>6.1f} N | {gap:>+6.1f} N | {drag:>5.1f} | {rr:>5.1f} | {decel:>+6.4f} | {crr_impl:>7.4f}")

# Overall Crr
fr_all = dpc["F_resist_real"].mean()
drag_all = dpc["F_drag"].mean()
normal_all = (MASS_KG * G + dpc["F_downforce"]).mean()
crr_all = (fr_all - drag_all) / normal_all
print(f"\n  Overall implied Crr: {crr_all:.4f}  (sim uses {CRR})")
print(f"  Overall real resistance: {fr_all:.1f} N  vs  sim: {(drag_all + normal_all * CRR):.1f} N")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: TORQUE FEEDBACK vs LVCU TORQUE REQ
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 2: ACTUAL MOTOR TORQUE vs COMMANDED TORQUE")
print("=" * 95)
print("If the motor produces less than commanded, the sim over-estimates drive force.")

mask_drive = (d["LVCU Torque Req"] > 10) & (d["torque_fb_clean"].notna())
dd = d[mask_drive]

print(f"\nDriving samples (LVCU > 10 Nm): {len(dd)}")

# Per speed bin
print(f"\n{'Speed':>8} | {'N':>5} | {'LVCU Req':>9} | {'TorqueFb':>9} | {'Fb/Req':>7} | {'Lost Nm':>8} | {'Lost N':>7} | {'RPM':>6}")
print("-" * 75)

for name, grp in dd.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    lvcu = grp["LVCU Torque Req"].mean()
    tfb = grp["torque_fb_clean"].mean()
    ratio = tfb / lvcu
    lost_nm = lvcu - tfb
    lost_n = lost_nm * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
    rpm = grp["Motor RPM"].mean()
    print(f"{name:>8} | {len(grp):>5} | {lvcu:>7.1f} Nm | {tfb:>7.1f} Nm | {ratio:>6.3f} | {lost_nm:>+6.1f} Nm | {lost_n:>+5.0f} N | {rpm:>5.0f}")

overall_lvcu = dd["LVCU Torque Req"].mean()
overall_fb = dd["torque_fb_clean"].mean()
overall_ratio = overall_fb / overall_lvcu
lost_overall = (overall_lvcu - overall_fb) * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
print(f"\n  Overall: LVCU={overall_lvcu:.1f} Nm, Feedback={overall_fb:.1f} Nm, "
      f"Ratio={overall_ratio:.3f}, Lost={lost_overall:.0f} N")
print(f"  The motor consistently produces {(1-overall_ratio)*100:.1f}% less torque than commanded.")
print(f"  This is likely motor+inverter efficiency loss at the SHAFT level.")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: REAL NET FORCE using LVCU Torque Req (what sim uses)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 3: FORCE BALANCE USING LVCU TORQUE REQ")
print("=" * 95)
print("F_net_real = m * a_GPS")
print("F_drive_sim = LVCU_torque * gear * eff / r")
print("F_resist_back = F_drive_sim - F_net_real  (during acceleration)")

mask_accel = (d["LVCU Torque Req"] > 10) & (d["a_lon"] > 0.05)
da = d[mask_accel]

print(f"\nAcceleration samples (torque > 10, a > 0.05): {len(da)}")

# The key question: is F_drive_sim - F_resist_sim = F_net_real?
da2 = da.copy()
da2["F_net_real"] = MASS_KG * da2["a_lon"]
da2["F_net_sim"] = da2["F_drive_lvcu_sim"] - da2["F_resist_sim"]
da2["F_net_gbx"] = da2["F_drive_lvcu_gbx"] - da2["F_resist_sim"]
# Using Torque Feedback instead
da2["F_drive_fb"] = da2["torque_fb_clean"] * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
da2["F_net_fb"] = da2["F_drive_fb"] - da2["F_resist_sim"]

print(f"\n{'Speed':>8} | {'N':>5} | {'F_net_real':>10} | {'F_net_sim':>10} | {'F_net_fb':>10} | {'Err(sim)':>8} | {'Err(fb)':>8}")
print("-" * 80)

for name, grp in da2.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    fr = grp["F_net_real"].mean()
    fs = grp["F_net_sim"].mean()
    ffb = grp["F_net_fb"].mean()
    err_s = fs - fr
    err_fb = ffb - fr
    print(f"{name:>8} | {len(grp):>5} | {fr:>+8.1f} N | {fs:>+8.1f} N | {ffb:>+8.1f} N | {err_s:>+6.1f} N | {err_fb:>+6.1f} N")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: THE SIM'S m_effective -- how it changes acceleration
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 4: EFFECTIVE MASS IMPACT ON SIMULATED ACCELERATION")
print("=" * 95)

print(f"""
  The sim computes: a_sim = F_net / m_effective = F_net / {M_EFFECTIVE:.1f} kg
  But GPS gives:    a_real = F_net_real / m_chassis = F_net_real / {MASS_KG:.1f} kg

  The sim's acceleration for a given net force is {MASS_KG/M_EFFECTIVE:.3f}x of
  what a chassis-mass-based calc would give.

  However, m_effective is CORRECT physics -- the motor must accelerate
  the rotor and wheels along with the chassis.  The question is whether
  the GPS acceleration already "sees" this rotational effect.

  GPS measures the acceleration of the chassis (GPS antenna), which IS
  the real forward acceleration.  The total force needed to produce this
  accel is F = m_eff * a_gps (because some force goes to spinning rotational
  components).

  So if the sim has the SAME net force as reality, it should get the
  SAME acceleration because both use m_effective (sim explicitly, reality
  implicitly through physics).

  KEY: The issue is whether F_net_sim matches F_net_real, NOT mass.
""")

# Quick check
overall_fnet_real = da2["F_net_real"].mean()
overall_fnet_sim = da2["F_net_sim"].mean()
overall_fnet_fb = da2["F_net_fb"].mean()

print(f"  During acceleration:")
print(f"    F_net_real (m_chassis * a_gps): {overall_fnet_real:+.1f} N")
print(f"    F_net_sim (LVCU*eff - resist):  {overall_fnet_sim:+.1f} N")
print(f"    F_net_fb (TrqFb*gbx - resist):  {overall_fnet_fb:+.1f} N")
print(f"    Sim overestimates F_net by:     {overall_fnet_sim - overall_fnet_real:+.1f} N")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: THE REAL QUESTION -- If sim force model is actually
# giving MORE net force than reality, why is the sim SLOWER?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 5: IF SIM HAS MORE NET FORCE, WHY IS IT SLOWER?")
print("=" * 95)
print("""
  The sim is 11.4% too slow (longer lap times).
  But at the same speed/torque, the sim has MORE net force.

  The answer MUST be one of:
  1. The sim operates at DIFFERENT speeds than reality (lower speed)
     -> at lower speed, even with more net force, it takes longer
  2. The sim operates with DIFFERENT torque commands
     -> driver model commands less throttle or more coasting
  3. The sim has lower CORNERING SPEEDS
     -> enters/exits corners slower, entire speed profile shifts down

  Let's check what the sim actually does vs reality.
""")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 6: KEY FINDING -- Torque Feedback consistently < LVCU Req
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 6: ROOT CAUSE -- TORQUE SHORTFALL")
print("=" * 95)

# The sim uses LVCU Torque Req and multiplies by eff for drive force.
# But actual motor output (Torque Feedback) is less than LVCU Req.
# So the sim's drive force using LVCU Req * 0.92 vs reality Torque Fb * 0.97:

dd2 = d[(d["LVCU Torque Req"] > 5) & d["torque_fb_clean"].notna()].copy()

# Sim's drive force: LVCU * gear * drivetrain_eff / r
dd2["F_drive_sim"] = dd2["LVCU Torque Req"] * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M
# Real drive force: TorqueFb * gear * gearbox_eff / r
dd2["F_drive_real"] = dd2["torque_fb_clean"] * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
# What sim SHOULD compute: LVCU * gear * gearbox_eff / r
dd2["F_drive_sim_corrected"] = dd2["LVCU Torque Req"] * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M

print(f"\n{'Speed':>8} | {'N':>5} | {'F_sim(0.92)':>11} | {'F_sim(0.97)':>11} | {'F_real(fb)':>10} | {'Sim/Real':>8} | {'Corr/Real':>9}")
print("-" * 80)

for name, grp in dd2.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    fsim = grp["F_drive_sim"].mean()
    fcorr = grp["F_drive_sim_corrected"].mean()
    freal = grp["F_drive_real"].mean()
    ratio_sim = fsim / freal if freal > 1 else 0
    ratio_corr = fcorr / freal if freal > 1 else 0
    print(f"{name:>8} | {len(grp):>5} | {fsim:>8.1f} N  | {fcorr:>8.1f} N  | {freal:>7.1f} N  | {ratio_sim:>7.3f} | {ratio_corr:>8.3f}")

# Overall
fsim_all = dd2["F_drive_sim"].mean()
fcorr_all = dd2["F_drive_sim_corrected"].mean()
freal_all = dd2["F_drive_real"].mean()

print(f"\n  Overall drive force comparison:")
print(f"    Sim (LVCU * 0.92):     {fsim_all:.1f} N")
print(f"    Corrected (LVCU * 0.97): {fcorr_all:.1f} N")
print(f"    Real (TrqFb * 0.97):   {freal_all:.1f} N")
print(f"    Sim / Real ratio:      {fsim_all/freal_all:.3f}")
print(f"    Corrected / Real:      {fcorr_all/freal_all:.3f}")

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 7: NET FORCE COMPARISON -- All three methods
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("ANALYSIS 7: ACCELERATION COMPARISON -- Sim vs Reality at SAME POINT")
print("=" * 95)
print("For each telemetry row, compute what the sim would predict for accel")
print("vs what GPS actually measured.\n")

d3 = d[(d["GPS Speed"] > 5) & d["torque_fb_clean"].notna()].copy()

# Sim's computation: F = LVCU * G * eta_drivetrain / r - resist_sim
d3["F_drive_sim"] = d3["LVCU Torque Req"].clip(lower=0) * GEAR_RATIO * DRIVETRAIN_EFF / TIRE_RADIUS_M
d3["F_net_sim"] = d3["F_drive_sim"] - d3["F_resist_sim"]
d3["a_sim"] = d3["F_net_sim"] / M_EFFECTIVE

# Corrected: LVCU * gearbox / r - resist_sim
d3["F_drive_corr"] = d3["LVCU Torque Req"].clip(lower=0) * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
d3["F_net_corr"] = d3["F_drive_corr"] - d3["F_resist_sim"]
d3["a_corr"] = d3["F_net_corr"] / M_EFFECTIVE

# Real: TorqueFb * gearbox / r - resist_sim (using sim resistance)
d3["F_drive_fb"] = d3["torque_fb_clean"] * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
d3["F_net_fb"] = d3["F_drive_fb"] - d3["F_resist_sim"]
d3["a_fb"] = d3["F_net_fb"] / M_EFFECTIVE

d3["a_real"] = d3["a_lon"]

print(f"{'Speed':>8} | {'N':>5} | {'a_GPS':>8} | {'a_sim':>8} | {'a_corr':>8} | {'a_fb':>8} | {'sim err':>8} | {'fb err':>8}")
print("-" * 85)

for name, grp in d3.groupby("speed_bin", observed=True):
    if len(grp) < 10:
        continue
    ar = grp["a_real"].mean() / G
    asim = grp["a_sim"].mean() / G
    acorr = grp["a_corr"].mean() / G
    afb = grp["a_fb"].mean() / G
    err_sim = (asim - ar)
    err_fb = (afb - ar)
    print(f"{name:>8} | {len(grp):>5} | {ar:>+6.4f} | {asim:>+6.4f} | {acorr:>+6.4f} | {afb:>+6.4f} | {err_sim:>+6.4f} | {err_fb:>+6.4f}")

ar_tot = d3["a_real"].mean()
asim_tot = d3["a_sim"].mean()
acorr_tot = d3["a_corr"].mean()
afb_tot = d3["a_fb"].mean()

print(f"\n  Overall mean accel (all moving data):")
print(f"    GPS:       {ar_tot/G:>+.5f} g ({ar_tot:>+.3f} m/s^2)")
print(f"    Sim:       {asim_tot/G:>+.5f} g ({asim_tot:>+.3f} m/s^2)  err={asim_tot/G - ar_tot/G:>+.5f} g")
print(f"    Corrected: {acorr_tot/G:>+.5f} g ({acorr_tot:>+.3f} m/s^2)  err={acorr_tot/G - ar_tot/G:>+.5f} g")
print(f"    TrqFb:     {afb_tot/G:>+.5f} g ({afb_tot:>+.3f} m/s^2)  err={afb_tot/G - ar_tot/G:>+.5f} g")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("FINAL ROOT-CAUSE SUMMARY")
print("=" * 95)
print(f"""
FORCE MODEL FINDINGS:

1. EFFICIENCY DOUBLE-DIP IN wheel_force() (CONFIRMED BUG):
   The sim computes: F = torque * gear * drivetrain_eff(0.92) / r
   Should be:        F = torque * gear * gearbox_eff(0.97) / r
   Impact: 5.4% drive force reduction.

   Explanation: LVCU Torque Req is the motor shaft torque command.
   Motor+inverter efficiency converts electrical power to shaft torque,
   but it does NOT reduce the shaft output.  The only mechanical loss
   from shaft to wheel is gearbox friction (~3%).

2. ACTUAL MOTOR OUTPUT < COMMANDED TORQUE:
   Torque Feedback / LVCU Req = {overall_ratio:.3f} (motor produces only {overall_ratio*100:.1f}% of commanded)
   This ~{(1-overall_ratio)*100:.0f}% shortfall is real motor physics, not a sim bug.
   However, the sim uses LVCU Torque Req as if the motor hits it exactly.

   When BOTH effects combine:
   - Sim drive: LVCU * 0.92 = {overall_lvcu * DRIVETRAIN_EFF:.1f} effective Nm
   - Real drive: TorqueFb * 0.97 = {overall_fb * GEARBOX_EFF:.1f} effective Nm
   - Sim is at {overall_lvcu * DRIVETRAIN_EFF / (overall_fb * GEARBOX_EFF):.3f}x of reality

   The sim's force is {(overall_lvcu * DRIVETRAIN_EFF / (overall_fb * GEARBOX_EFF) - 1)*100:+.1f}% vs
   real output (eff=0.92 on a larger torque almost cancels eff=0.97 on a smaller torque).

3. RESISTANCE UNDER-PREDICTION:
   Straight-line coasting shows real resistance ~2x sim's prediction.
   Implied Crr = {crr_all:.4f} vs sim's {CRR}.
   This gap (~{(crr_all - CRR)*normal_all:.0f} N) is too large for just rolling resistance.
   It likely includes: bearing drag, chain tension, seal friction, brake drag.
   These are real-world parasitic losses not modeled in the sim.

4. NET EFFECT ON SIM TIMING:
   The sim predicts HIGHER acceleration than GPS shows (more net force).
   But the sim is 11.4% SLOWER.
   This means the timing error is NOT in the force model -- it's in:
   - Corner speed limits (how fast through turns)
   - Driver strategy (when/how much throttle, when to coast)
   - Speed envelope computation
   The force model errors partially cancel (less drive force + less resistance).
""")
