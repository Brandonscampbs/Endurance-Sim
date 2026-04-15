"""
Detailed SOC analysis: why does BMS SOC show non-linear behavior?
The BMS SOC drops from 94.5% to 80.5% in the first ~750s, then jumps
down to 60.5% and stays flat. This is unusual -- investigate.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

TELEMETRY_PATH = Path(r"C:\Users\brand\Development-BC\Real-Car-Data-And-Stats\CleanedEndurance.csv")
OUTPUT_DIR = Path(r"C:\Users\brand\Development-BC\analysis\thermal_output")

df = pd.read_csv(TELEMETRY_PATH, encoding='latin-1', skiprows=[1])
for col in ['Time', 'Pack Temp', 'Pack Voltage', 'Pack Current',
            'State of Charge', 'Min Cell Voltage', 'GPS Speed', 'Motor RPM']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

time = df['Time'].values
soc = df['State of Charge'].values
current = df['Pack Current'].values
voltage = df['Pack Voltage'].values
temp = df['Pack Temp'].values
speed = df['GPS Speed'].values if 'GPS Speed' in df.columns else np.zeros_like(time)

# Detailed SOC profile
print("=" * 70)
print("SOC PROFILE DETAIL")
print("=" * 70)

# Find all SOC transitions
soc_diff = np.diff(soc)
transitions = np.where(np.abs(soc_diff) > 0.3)[0]  # Large SOC jumps
print(f"\nLarge SOC transitions (>0.3%):")
for idx in transitions[:20]:
    print(f"  t={time[idx]:.1f}s: SOC {soc[idx]:.1f}% -> {soc[idx+1]:.1f}% "
          f"(delta={soc_diff[idx]:.1f}%), I={current[idx]:.1f}A, V={voltage[idx]:.1f}V")

# SOC at every 100s
print(f"\nSOC at 100s intervals:")
for t_target in range(0, 1700, 100):
    idx = np.argmin(np.abs(time - t_target))
    print(f"  t={time[idx]:6.0f}s: SOC={soc[idx]:5.1f}%, I={current[idx]:6.1f}A, "
          f"V={voltage[idx]:5.1f}V, T={temp[idx]:4.1f}C, "
          f"Speed={speed[idx]:4.1f}km/h")

# Compute cumulative Ah and compare to SOC
dt = np.diff(time, prepend=time[0])
cum_ah = np.cumsum(current * dt) / 3600

# The big question: around t=800-1100s, SOC drops from 80.5 to 60.5.
# Is the car actually drawing huge current, or is the BMS recalibrating?
print(f"\nFocusing on SOC drop region (t=700-1200s):")
mask_drop = (time >= 700) & (time <= 1200)
print(f"  SOC: {soc[mask_drop][0]:.1f}% -> {soc[mask_drop][-1]:.1f}%")
print(f"  Duration: {time[mask_drop][-1] - time[mask_drop][0]:.1f}s")
print(f"  Avg current: {np.mean(current[mask_drop]):.1f}A")
print(f"  Max current: {np.max(current[mask_drop]):.1f}A")
print(f"  Charge (Ah): {np.sum(current[mask_drop] * dt[mask_drop]) / 3600:.2f} Ah")
print(f"  Avg speed: {np.mean(speed[mask_drop]):.1f} km/h")

# Is there a driver change? Look for extended stopped period
stopped = speed < 2.0
stopped_runs = []
in_run = False
run_start = 0
for i in range(len(stopped)):
    if stopped[i]:
        if not in_run:
            in_run = True
            run_start = i
    else:
        if in_run:
            run_len = time[i] - time[run_start]
            if run_len > 10:
                stopped_runs.append((time[run_start], time[i], run_len))
            in_run = False

print(f"\nStopped periods (speed < 2 km/h, > 10s):")
for start_t, end_t, dur in stopped_runs:
    idx_s = np.argmin(np.abs(time - start_t))
    idx_e = np.argmin(np.abs(time - end_t))
    print(f"  t={start_t:.0f}-{end_t:.0f}s (dur={dur:.0f}s): "
          f"SOC {soc[idx_s]:.1f}% -> {soc[idx_e]:.1f}%, "
          f"avg I={np.mean(current[idx_s:idx_e]):.1f}A")

# This is likely the driver change. The BMS might reset/recalibrate SOC during pause.
# Segment the event into driver stints
print(f"\n--- DRIVER STINT ANALYSIS ---")
# Find the longest stopped period = driver change
if stopped_runs:
    longest = max(stopped_runs, key=lambda x: x[2])
    dc_start, dc_end = longest[0], longest[1]
    print(f"Driver change at t={dc_start:.0f}-{dc_end:.0f}s (duration={longest[2]:.0f}s)")

    # Stint 1: before driver change
    mask1 = time < dc_start
    if np.sum(mask1) > 100:
        ah1 = np.sum(current[mask1] * dt[mask1]) / 3600
        print(f"\nStint 1 (t=0 to {dc_start:.0f}s):")
        print(f"  SOC: {soc[mask1][0]:.1f}% -> {soc[mask1][-1]:.1f}% (consumed {soc[mask1][0] - soc[mask1][-1]:.1f}%)")
        print(f"  Coulomb: {ah1:.2f} Ah (pack), {ah1/4:.2f} Ah (cell)")
        print(f"  Implied capacity: {ah1/4 / ((soc[mask1][0] - soc[mask1][-1])/100):.2f} Ah")
        print(f"  Duration: {time[mask1][-1]:.0f}s")

    # Stint 2: after driver change
    mask2 = time > dc_end
    if np.sum(mask2) > 100:
        ah2 = np.sum(current[mask2] * dt[mask2]) / 3600
        print(f"\nStint 2 (t={dc_end:.0f} to {time[-1]:.0f}s):")
        print(f"  SOC: {soc[mask2][0]:.1f}% -> {soc[mask2][-1]:.1f}% (consumed {soc[mask2][0] - soc[mask2][-1]:.1f}%)")
        if (soc[mask2][0] - soc[mask2][-1]) > 1:
            print(f"  Coulomb: {ah2:.2f} Ah (pack), {ah2/4:.2f} Ah (cell)")
            print(f"  Implied capacity: {ah2/4 / ((soc[mask2][0] - soc[mask2][-1])/100):.2f} Ah")
        else:
            print(f"  SOC barely changed -- BMS may have recalibrated")
        print(f"  Duration: {time[mask2][-1] - time[mask2][0]:.0f}s")

# Plot detailed SOC
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax1.plot(time, soc, 'b-', linewidth=1)
ax1.set_ylabel('SOC (%)')
ax1.set_title('SOC, Current, and Speed vs Time')
ax1.grid(True, alpha=0.3)
if stopped_runs:
    for s, e, _ in stopped_runs:
        ax1.axvspan(s, e, alpha=0.2, color='red')

ax2.plot(time, current, 'r-', linewidth=0.3, alpha=0.5)
ax2.set_ylabel('Pack Current (A)')
ax2.grid(True, alpha=0.3)

ax3.plot(time, speed, 'g-', linewidth=0.3, alpha=0.5)
ax3.set_ylabel('GPS Speed (km/h)')
ax3.set_xlabel('Time (s)')
ax3.grid(True, alpha=0.3)

plt.savefig(OUTPUT_DIR / 'soc_detail.png', dpi=150)
plt.close()

# Recompute SOC tracking for just stint 1 (before driver change)
if stopped_runs:
    mask1 = time < dc_start
    time1 = time[mask1]
    current1 = current[mask1]
    soc1 = soc[mask1]
    dt1 = dt[mask1]

    cum_ah1 = np.cumsum(current1 * dt1) / 3600

    print(f"\n--- SOC TRACKING (STINT 1 ONLY) ---")
    for cap in [4.0, 4.3, 4.5, 4.7, 5.0]:
        sim_soc = soc1[0] - (cum_ah1 / (cap * 4)) * 100
        err = sim_soc - soc1
        print(f"  Cap={cap:.1f} Ah: final sim={sim_soc[-1]:.1f}% vs real={soc1[-1]:.1f}%, "
              f"max err={np.max(np.abs(err)):.2f}%, mean err={np.mean(np.abs(err)):.2f}%")

print("\nDone.")
