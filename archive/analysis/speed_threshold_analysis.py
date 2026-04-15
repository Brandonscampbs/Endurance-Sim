"""Telemetry speed threshold analysis.

Determines how much of the 48-second gap between sim (1560s) and
telemetry (1608.75s) is due to stopped or slow periods.
"""

import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from fsae_sim.data.loader import load_cleaned_csv

# Load telemetry
_, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")

time = aim_df["Time"].values
speed_kmh = aim_df["GPS Speed"].values  # aliased from LFspeed
speed_ms = speed_kmh / 3.6

# Total elapsed time
total_elapsed = time[-1] - time[0]
print("=== TELEMETRY TIME ANALYSIS ===")
print(f"Total samples: {len(aim_df)}")
print(f"Sample rate: ~{1.0/np.median(np.diff(time)):.1f} Hz")
print(f"Time range: {time[0]:.2f}s to {time[-1]:.2f}s")
print(f"Total elapsed time: {total_elapsed:.2f}s")
print()

# dt array (same method as validation.py)
dt_arr = np.diff(time, prepend=time[0])
total_dt = np.sum(dt_arr)
print(f"Sum of dt array: {total_dt:.2f}s")
print()

# ============================================================
# Speed threshold analysis
# ============================================================
print("=== SPEED THRESHOLD ANALYSIS ===")

mask_stopped = speed_kmh < 1.0
mask_crawling = (speed_kmh >= 1.0) & (speed_kmh < 5.0)
mask_very_slow = (speed_kmh >= 5.0) & (speed_kmh < 10.0)
mask_normal = speed_kmh >= 10.0

t_stopped = np.sum(dt_arr[mask_stopped])
t_crawling = np.sum(dt_arr[mask_crawling])
t_very_slow = np.sum(dt_arr[mask_very_slow])
t_normal = np.sum(dt_arr[mask_normal])

print(f"Time at < 1 km/h (stopped):       {t_stopped:8.2f}s  ({t_stopped/total_dt*100:.1f}%)")
print(f"Time at 1-5 km/h (crawling):      {t_crawling:8.2f}s  ({t_crawling/total_dt*100:.1f}%)")
print(f"Time at 5-10 km/h (very slow):    {t_very_slow:8.2f}s  ({t_very_slow/total_dt*100:.1f}%)")
print(f"Time at > 10 km/h (normal):       {t_normal:8.2f}s  ({t_normal/total_dt*100:.1f}%)")
print(f"SUM check:                        {t_stopped+t_crawling+t_very_slow+t_normal:8.2f}s")
print()

# ============================================================
# Exactly reproduce validation.py telem_driving_time
# ============================================================
print("=== VALIDATION.PY TELEM_DRIVING_TIME REPRODUCTION ===")
telem_driving_time = float(np.sum(dt_arr[speed_kmh > 5]))
print(f"telem_driving_time (speed > 5 km/h): {telem_driving_time:.2f}s")
print(f"This EXCLUDES: stopped + crawling = {t_stopped + t_crawling:.2f}s")
print(f"This INCLUDES: very slow (5-10) + normal (>10) = {t_very_slow + t_normal:.2f}s")
print()

# ============================================================
# Find driver change: longest period of near-zero speed
# ============================================================
print("=== DRIVER CHANGE DETECTION ===")

# Identify contiguous slow periods (speed < 5 km/h)
slow = speed_kmh < 5.0
transitions = np.diff(slow.astype(int))
starts = np.where(transitions == 1)[0] + 1
ends = np.where(transitions == -1)[0] + 1

if slow[0]:
    starts = np.insert(starts, 0, 0)
if slow[-1]:
    ends = np.append(ends, len(slow))

slow_periods = []
for s, e in zip(starts, ends):
    duration = time[min(e, len(time)-1)] - time[s]
    mean_speed = np.mean(speed_kmh[s:e])
    slow_periods.append((s, e, duration, mean_speed))

slow_periods.sort(key=lambda x: x[2], reverse=True)

print(f"Found {len(slow_periods)} periods where speed < 5 km/h")
print(f"Top 10 longest slow periods:")
for i, (s, e, dur, ms) in enumerate(slow_periods[:10]):
    t_start = time[s]
    t_end = time[min(e, len(time)-1)]
    print(f"  #{i+1}: {dur:7.2f}s  (t={t_start:.1f}-{t_end:.1f}s, samples={e-s}, mean_speed={ms:.1f} km/h)")

driver_change = slow_periods[0]
print()
print(f"Driver change (longest stop): {driver_change[2]:.2f}s")
print(f"  At time: {time[driver_change[0]]:.1f}s to {time[min(driver_change[1], len(time)-1)]:.1f}s")
print()

# ============================================================
# Is driver change included in 1608.75s?
# ============================================================
print("=== DRIVER CHANGE vs VALIDATION NUMBER ===")
print(f"telem_driving_time: {telem_driving_time:.2f}s")
print(f"Driver change duration: {driver_change[2]:.2f}s")
print(f"If driver change WERE included: {telem_driving_time + driver_change[2]:.2f}s")
print(f"Total elapsed time: {total_elapsed:.2f}s")
print()

# ============================================================
# Time below 0.5 m/s OUTSIDE driver change
# ============================================================
print("=== TIME BELOW 0.5 M/S (SIM MIN SPEED) OUTSIDE DRIVER CHANGE ===")

dc_start_idx = driver_change[0]
dc_end_idx = driver_change[1]

below_min = speed_ms < 0.5
not_driver_change = np.ones(len(speed_ms), dtype=bool)
not_driver_change[dc_start_idx:dc_end_idx] = False

driving_below_min = below_min & not_driver_change
t_below_min_driving = np.sum(dt_arr[driving_below_min])
n_samples_below = np.sum(driving_below_min)

print(f"Time below 0.5 m/s (1.8 km/h) outside driver change: {t_below_min_driving:.2f}s ({n_samples_below} samples)")
print()

below_5_not_dc = (speed_kmh < 5.0) & not_driver_change
t_below_5_not_dc = np.sum(dt_arr[below_5_not_dc])
n_below_5_not_dc = np.sum(below_5_not_dc)
print(f"Time below 5 km/h outside driver change: {t_below_5_not_dc:.2f}s ({n_below_5_not_dc} samples)")
print()

# ============================================================
# Replay strategy filtering effect
# ============================================================
print("=== REPLAY STRATEGY FILTERING ===")
print("ReplayStrategy.from_full_endurance() filters aim_df to speed > 5 km/h")
print("It then creates distance-indexed interpolation from the filtered data.")
print("The sim engine runs segment-by-segment. Time = distance / speed.")
print("So the sim NEVER encounters the slow-speed driving periods.")
print()

# Show what the replay strategy actually keeps vs drops
replay_moving = speed_kmh > 5.0
replay_kept_time = np.sum(dt_arr[replay_moving])
replay_dropped_time = np.sum(dt_arr[~replay_moving])
print(f"ReplayStrategy keeps: {replay_kept_time:.2f}s of telemetry time")
print(f"ReplayStrategy drops: {replay_dropped_time:.2f}s of telemetry time")
print(f"  Of dropped time: {driver_change[2]:.2f}s is driver change")
print(f"  Of dropped time: {t_below_5_not_dc:.2f}s is other slow/stopped driving")
print()

# ============================================================
# Distribution of speeds during driving (speed > 5 km/h)
# ============================================================
print("=== DISTRIBUTION OF SPEEDS DURING DRIVING (speed > 5 km/h) ===")
driving_mask = speed_kmh > 5
driving_speeds = speed_kmh[driving_mask]
driving_dt = dt_arr[driving_mask]

for threshold in [5, 10, 15, 20, 25]:
    lower = threshold
    upper = threshold + 5
    mask_band = (driving_speeds >= lower) & (driving_speeds < upper)
    t_band = np.sum(driving_dt[mask_band])
    print(f"  {lower:2d}-{upper:2d} km/h: {t_band:7.2f}s")

mask_gt30 = driving_speeds >= 30
t_gt30 = np.sum(driving_dt[mask_gt30])
print(f"  30+ km/h:  {t_gt30:7.2f}s")
print(f"  Total driving: {np.sum(driving_dt):7.2f}s")
print()

# ============================================================
# Detailed: list ALL slow periods outside driver change
# ============================================================
print("=== ALL SLOW PERIODS OUTSIDE DRIVER CHANGE (speed < 5 km/h, > 0.5s) ===")
non_dc_slow = [(s, e, d, ms) for s, e, d, ms in slow_periods
               if not (s >= dc_start_idx and e <= dc_end_idx)
               and d > 0.5]
total_non_dc_slow = sum(d for _, _, d, _ in non_dc_slow)
print(f"Count: {len(non_dc_slow)} periods, total {total_non_dc_slow:.2f}s")
for i, (s, e, dur, ms) in enumerate(non_dc_slow[:20]):
    t_start = time[s]
    t_end = time[min(e, len(time)-1)]
    print(f"  #{i+1}: {dur:5.2f}s at t={t_start:.1f}s (mean speed {ms:.1f} km/h)")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total elapsed time in CSV:                {total_elapsed:.2f}s")
print(f"Validation telem_driving_time (>5 km/h):  {telem_driving_time:.2f}s")
print(f"Driver change (longest stop):             {driver_change[2]:.2f}s")
print(f"Other slow periods (<5 km/h, not DC):     {t_below_5_not_dc:.2f}s")
print(f"Sum (driving + DC + other slow):          {telem_driving_time + driver_change[2] + t_below_5_not_dc:.2f}s")
print()
print(f"Sim driving time:                         1560.00s")
print(f"Telem driving time reference:             {telem_driving_time:.2f}s")
print(f"Gap to explain:                           {telem_driving_time - 1560.0:.2f}s")
print()
print("CONCLUSION:")
print(f"The validation (line 221 of validation.py) computes:")
print(f"  telem_driving_time = sum(dt where speed > 5 km/h)")
print(f"This ALREADY excludes the driver change ({driver_change[2]:.2f}s) and other stops ({t_below_5_not_dc:.2f}s).")
print(f"So the 48s gap is NOT a measurement error from including stops.")
print(f"The sim genuinely finishes the driving portions {telem_driving_time - 1560.0:.1f}s faster than reality.")
