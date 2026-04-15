"""RPM deep-dive: investigate anomalous speed bins and coast torque feedback.

Follow-up to rpm_power_diagnostic.py findings:
- RPM error is -3% mean but varies by speed bin (low speed bins show large errors)
- Back-derived effective radius = 0.1979 vs sim 0.2042 (3.1% diff)
- Torque Feedback = 27 Nm during coast while LVCU commands 0 (phantom torque)
- Speed bias of +1.65 km/h explains ~51s of the 53.6s gap
"""

import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from fsae_sim.data.loader import load_cleaned_csv

print("=" * 72)
print("RPM DEEP-DIVE: ANOMALOUS SPEED BINS AND COAST BEHAVIOR")
print("=" * 72)

_, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")

# Filter to moving
moving = aim_df[aim_df["GPS Speed"] > 5.0].copy()
speed_ms = moving["LFspeed"].values / 3.6
motor_rpm = moving["Motor RPM"].values

GEAR_RATIO = 3.6363
TIRE_RADIUS = 0.2042

# =====================================================================
# 1. LFspeed vs Motor RPM scatter analysis - WHY do 25-35 km/h have
#    12% error while 45-80 km/h have <1.4%?
# =====================================================================
print("\n" + "=" * 72)
print("1. LFspeed vs Motor RPM: INVESTIGATING PER-BIN ERRORS")
print("=" * 72)

# Hypothesis: at low speeds, the front wheel speed sensor (LFspeed)
# and motor RPM may decouple due to wheel slip, clutch behavior,
# or sensor noise.

# For each speed bin, also check throttle and brake state
throttle = moving["Throttle Pos"].values
torque_req = moving["LVCU Torque Req"].values
torque_fb = moving["Torque Feedback"].values

r_eff = speed_ms * GEAR_RATIO * 60.0 / (2.0 * math.pi * motor_rpm)

print(f"\n  Per-bin analysis with throttle/brake context:")
print(f"  {'Speed bin':>15s}  {'N':>6s}  {'r_eff':>8s}  {'Throttle%':>10s}  {'TorqReq':>8s}  {'TorqFB':>8s}  {'RPM':>8s}  {'RPM_pred':>10s}")
for lo, hi in [(5,10), (10,15), (15,20), (20,25), (25,30), (30,35), (35,40), (40,45), (45,50), (50,55), (55,60), (60,65), (65,70)]:
    mask = (speed_ms * 3.6 >= lo) & (speed_ms * 3.6 < hi) & (motor_rpm > 50) & (speed_ms > 0.5)
    if mask.sum() < 10:
        continue
    r = r_eff[mask]
    r_valid = r[(r > 0.1) & (r < 0.5)]
    rpm_pred = speed_ms[mask] * GEAR_RATIO * 60.0 / (2.0 * math.pi * TIRE_RADIUS)
    print(f"  {lo:>3d}-{hi:<3d} km/h  {mask.sum():>6d}  {np.median(r_valid):.4f}  {np.mean(throttle[mask]):>10.1f}  {np.mean(torque_req[mask]):>8.1f}  {np.mean(torque_fb[mask]):>8.1f}  {np.mean(motor_rpm[mask]):>8.0f}  {np.mean(rpm_pred):>10.0f}")

# =====================================================================
# 2. COAST TORQUE FEEDBACK -- the 27 Nm phantom torque
# =====================================================================
print("\n" + "=" * 72)
print("2. COAST TORQUE FEEDBACK: THE 27 Nm PHANTOM")
print("=" * 72)

# When throttle < 5% and LVCU Torque Req < 1, the Torque Feedback
# reads ~27 Nm. This is suspicious. Is it a sensor offset?

coast_mask = (moving["Throttle Pos"] < 5.0) & (moving["LVCU Torque Req"] < 1.0) & (moving["GPS Speed"] > 10.0)
coast = moving[coast_mask]

print(f"\n  Coast samples: {len(coast)} (throttle < 5%, LVCU < 1 Nm, speed > 10 km/h)")
print(f"  Torque Feedback stats:")
print(f"    Mean:   {coast['Torque Feedback'].mean():.2f} Nm")
print(f"    Median: {coast['Torque Feedback'].median():.2f} Nm")
print(f"    Std:    {coast['Torque Feedback'].std():.2f} Nm")
print(f"    Min:    {coast['Torque Feedback'].min():.2f} Nm")
print(f"    Max:    {coast['Torque Feedback'].max():.2f} Nm")

# Compare with Torque Command channel
print(f"\n  Torque Command (inverter internal):")
print(f"    Mean:   {coast['Torque Command'].mean():.2f} Nm")
print(f"    Median: {coast['Torque Command'].median():.2f} Nm")

# MCU DC Current during coast
print(f"\n  MCU DC Current during coast:")
print(f"    Mean:   {coast['MCU DC Current'].mean():.2f} A")
print(f"    Median: {coast['MCU DC Current'].median():.2f} A")

# Pack Current during coast (negative = regen / charging)
print(f"\n  Pack Current during coast:")
print(f"    Mean:   {coast['Pack Current'].mean():.2f} A")
print(f"    Median: {coast['Pack Current'].median():.2f} A")

# The -537W pack power + 27 Nm torque feedback is a strong signal
# If torque feedback is 27 Nm but current is -1.29 A, the motor is
# being DRAGGED -- the torque is resistive, not motive.
# OR the torque feedback sensor has an offset.
print(f"\n  CHECK: Is Torque Feedback an offset/calibration artifact?")
print(f"  If 27 Nm was real motive torque, power would be:")
mech_power = 27.0 * coast["Motor RPM"].mean() * math.pi / 30
print(f"    P_mech = 27 Nm * {coast['Motor RPM'].mean():.0f} RPM * pi/30 = {mech_power:.0f} W")
print(f"    But pack is ABSORBING {abs(coast['Pack Current'].mean() * coast['Pack Voltage'].mean()):.0f} W")
print(f"    --> Torque Feedback = 27 Nm during coast is likely a SENSOR OFFSET or")
print(f"        represents cogging/iron-loss torque that the inverter reads as 'feedback'")

# Check: when the motor is clearly producing torque (full throttle),
# what's the offset?
full_throttle = moving[(moving["Throttle Pos"] > 90.0) & (moving["GPS Speed"] > 20.0)]
if len(full_throttle) > 10:
    print(f"\n  Full throttle reference (throttle > 90%):")
    print(f"    Mean LVCU Torque Req: {full_throttle['LVCU Torque Req'].mean():.1f} Nm")
    print(f"    Mean Torque Feedback: {full_throttle['Torque Feedback'].mean():.1f} Nm")
    print(f"    Mean Torque Command:  {full_throttle['Torque Command'].mean():.1f} Nm")
    print(f"    Ratio FB/Req:         {full_throttle['Torque Feedback'].mean() / full_throttle['LVCU Torque Req'].mean():.3f}")

# =====================================================================
# 3. THE 2000-4000m ANOMALY: sim gains +18.2s
# =====================================================================
print("\n" + "=" * 72)
print("3. THE 2000-4000m ANOMALY: WHY DOES SIM GAIN +18.2s?")
print("=" * 72)

# What's happening at 2000-4000m in telemetry? Is there a slow period?
dist = moving["Distance on GPS Speed"].values
mask_2k4k = (dist >= 2000) & (dist < 4000)
telem_2k4k = moving[mask_2k4k]

if len(telem_2k4k) > 0:
    print(f"  Telemetry 2000-4000 m:")
    print(f"    Duration:  {telem_2k4k['Time'].iloc[-1] - telem_2k4k['Time'].iloc[0]:.1f} s")
    print(f"    Mean speed: {telem_2k4k['GPS Speed'].mean():.1f} km/h")
    print(f"    Min speed:  {telem_2k4k['GPS Speed'].min():.1f} km/h")
    print(f"    Samples below 10 km/h: {(telem_2k4k['GPS Speed'] < 10).sum()}")

# Also check the full telemetry (including stopped) for this distance range
full_dist = aim_df["Distance on GPS Speed"].values
full_2k4k_mask = (full_dist >= 2000) & (full_dist < 4000)
full_2k4k = aim_df[full_2k4k_mask]
if len(full_2k4k) > 0:
    print(f"\n  Full telemetry (incl. stopped) 2000-4000 m:")
    print(f"    Duration:  {full_2k4k['Time'].iloc[-1] - full_2k4k['Time'].iloc[0]:.1f} s")
    print(f"    Mean speed: {full_2k4k['GPS Speed'].mean():.1f} km/h")
    print(f"    Samples below 5 km/h: {(full_2k4k['GPS Speed'] < 5).sum()}")
    print(f"    --> Stopped time not in 'moving' filter explains sim gaining time")

# Also 10000-12000m (another big gap window: -12.7s)
mask_10k12k = (dist >= 10000) & (dist < 12000)
telem_10k12k = moving[mask_10k12k]
if len(telem_10k12k) > 0:
    print(f"\n  Telemetry 10000-12000 m:")
    print(f"    Duration:  {telem_10k12k['Time'].iloc[-1] - telem_10k12k['Time'].iloc[0]:.1f} s")
    print(f"    Mean speed: {telem_10k12k['GPS Speed'].mean():.1f} km/h")
    # Is this the driver change region?
    print(f"    --> This is near the driver change point (~10.3 km)")

# =====================================================================
# 4. CHECK: does the sim skip the driver change gap correctly?
# =====================================================================
print("\n" + "=" * 72)
print("4. DRIVER CHANGE GAP IN TELEMETRY")
print("=" * 72)

# Find where speed drops below 5 km/h for extended periods
speed_full = aim_df["GPS Speed"].values
time_full = aim_df["Time"].values
dist_full = aim_df["Distance on GPS Speed"].values

stopped_mask = speed_full < 5.0
transitions = np.diff(stopped_mask.astype(int))
stop_starts = np.where(transitions == 1)[0]
stop_ends = np.where(transitions == -1)[0]

print(f"  Stopped periods (speed < 5 km/h):")
for i, (start, end) in enumerate(zip(stop_starts[:10], stop_ends[:min(10, len(stop_ends))])):
    duration = time_full[end] - time_full[start]
    at_dist = dist_full[start]
    if duration > 2.0:
        print(f"    Stop {i+1}: t={time_full[start]:.1f}-{time_full[end]:.1f} s, dur={duration:.1f} s, dist={at_dist:.0f} m")

# The ReplayStrategy filters out stopped periods via min_speed_kmh=5.0
# So the sim's distance timeline skips them. But the telemetry time
# used for comparison may or may not include them.
print(f"\n  Total telemetry time: {time_full[-1] - time_full[0]:.1f} s")
print(f"  Moving time (>5 km/h): {telem_moving_t:.1f} s" if 'telem_moving_t' in dir() else "")

# Compute time spent stopped
stopped_time = np.sum(np.diff(time_full)[stopped_mask[:-1]])
print(f"  Time spent stopped (< 5 km/h): {stopped_time:.1f} s")
print(f"  Time spent moving:  {time_full[-1] - time_full[0] - stopped_time:.1f} s")

# =====================================================================
# 5. LOADED vs UNLOADED tire radius
# =====================================================================
print("\n" + "=" * 72)
print("5. LOADED vs UNLOADED TIRE RADIUS")
print("=" * 72)

# The sim uses UNLOADED radius = 0.2042 m
# Under the car's weight (288 kg, ~706 N per tire on average),
# the tire compresses. The back-derived radius of 0.1979 m is
# 3.1% smaller, which is consistent with tire deflection.

# For a 16x7.5-10 tire at ~657 N nominal load (from TIR FNOMIN):
# Typical deflection is 5-15% of sidewall height
# Sidewall height = 0.2042 (tire radius) - 0.127 (rim radius) = 0.0772 m
# 5% deflection = 3.86 mm, 10% = 7.72 mm

sidewall = 0.2042 - 0.127
deflection = 0.2042 - 0.1979
deflection_pct = deflection / sidewall * 100

print(f"  Unloaded radius (TIR file):  0.2042 m")
print(f"  Rim radius (TIR file):       0.127 m")
print(f"  Sidewall height:             {sidewall*1000:.1f} mm")
print(f"  Back-derived effective:      0.1979 m (median 0.2025 m)")
print(f"  Deflection (mean):           {deflection*1000:.1f} mm ({deflection_pct:.1f}% of sidewall)")
print(f"  Deflection (median):         {(0.2042-0.2025)*1000:.1f} mm ({(0.2042-0.2025)/sidewall*100:.1f}% of sidewall)")

# Using median is more robust to outliers
print(f"\n  Using MEDIAN back-derived radius = 0.2025 m:")
r_loaded = 0.2025
speed_diff_pct = (TIRE_RADIUS - r_loaded) / TIRE_RADIUS * 100
rpm_diff_pct = (r_loaded - TIRE_RADIUS) / TIRE_RADIUS * 100  # opposite sign for RPM
print(f"    Speed error from radius: {speed_diff_pct:.2f}%")
print(f"    At 52 km/h avg, this is {52 * speed_diff_pct/100:.2f} km/h")
print(f"    Over 1600s, this produces {1600 * speed_diff_pct/100:.0f} s time error")

# =====================================================================
# 6. QUANTIFYING SPEED BIAS SOURCES
# =====================================================================
print("\n" + "=" * 72)
print("6. SPEED BIAS DECOMPOSITION")
print("=" * 72)

# The sim is 1.65 km/h faster. Where does this come from?
# Source 1: tire radius error (unloaded vs loaded)
radius_speed_bias_kmh = 52.0 * (TIRE_RADIUS - r_loaded) / TIRE_RADIUS
print(f"  Source 1: Tire radius (unloaded vs loaded)")
print(f"    Using median r_eff = {r_loaded:.4f} m vs sim {TIRE_RADIUS:.4f} m")
print(f"    Speed bias: {radius_speed_bias_kmh:.2f} km/h")
print(f"    Time impact: {1600 * radius_speed_bias_kmh / 52:.0f} s")

# Source 2: missing motor drag during coast
# Coast is ~754/4420 = 17% of segments
# During coast, sim has zero drag but real car has ~70N parasitic + motor drag
# But wait -- the sim already has 70N parasitic drag constant
# The additional missing drag is the motor iron loss / inverter standby
# Pack power during coast = -537W (motor is absorbing, not producing)
# This represents ~537W / (52/3.6 m/s) = ~37N of additional drag
print(f"\n  Source 2: Missing motor electrical drag during coast")
print(f"    Real car absorbs ~537 W during coast (from pack)")
print(f"    At ~52 km/h = 14.4 m/s, P=F*v -> F = 537/14.4 = {537/14.4:.0f} N")
print(f"    Coast fraction: {754/4420*100:.0f}% of segments")
print(f"    But this is drag that SLOWS the car -- sim missing it = sim goes faster")
print(f"    Speed impact during coast: small (already has 70N parasitic)")

# Source 3: sim force balance errors
# The sim is +2.41 km/h faster during throttle, -0.31 during coast
# The throttle segments dominate (3357 vs 754)
print(f"\n  Source 3: Force balance during throttle (dominant)")
print(f"    Sim is +2.41 km/h during throttle (3357 segments, 76% of total)")
print(f"    Sim is -0.31 km/h during coast (754 segments, 17%)")
print(f"    Sim is -1.90 km/h during brake (309 segments, 7%)")
print(f"    Weighted avg: {(2.41*3357 - 0.31*754 - 1.90*309) / 4420:.2f} km/h")

print("\nDone.")
