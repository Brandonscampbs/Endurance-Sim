"""Final SOC flat-region analysis."""
import numpy as np
import pandas as pd
from pathlib import Path

TELEMETRY_PATH = Path(r"C:\Users\brand\Development-BC\Real-Car-Data-And-Stats\CleanedEndurance.csv")
df = pd.read_csv(TELEMETRY_PATH, encoding='latin-1', skiprows=[1])
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

time = df['Time'].values
soc = df['State of Charge'].values
current = df['Pack Current'].values
dt = np.diff(time, prepend=time[0])

# SOC flat regions
print("SOC FLAT REGIONS (>30s at same value):")
i = 0
while i < len(soc) - 1:
    start_i = i
    while i < len(soc) - 1 and abs(soc[i+1] - soc[start_i]) < 0.1:
        i += 1
    duration = time[i] - time[start_i]
    if duration > 30:
        avg_i = np.mean(current[start_i:i+1])
        total_ah = np.sum(current[start_i:i+1] * dt[start_i:i+1]) / 3600
        print(f"  t={time[start_i]:.0f}-{time[i]:.0f}s ({duration:.0f}s): "
              f"SOC={soc[start_i]:.1f}%, avg I={avg_i:.1f}A, Ah drawn={total_ah:.2f}")
    i += 1

# Check: SOC is flat at 94.5% for first 350s while car draws current
# and flat at 60.5% for last 450s while car draws current
# This means BMS SOC has saturation limits or is voltage-based, not pure coulomb
print(f"\nFirst 350s: SOC flat at {soc[0]:.1f}%")
mask_early = time < 350
ah_early = np.sum(current[mask_early] * dt[mask_early]) / 3600
print(f"  Current drawn: {ah_early:.2f} Ah (pack), {ah_early/4:.3f} Ah (cell)")
print(f"  At 4.5 Ah/cell, this is {ah_early/4/4.5*100:.1f}% SOC")

print(f"\nLast 450s (t>1163): SOC flat at {soc[-1]:.1f}%")
mask_late = time > 1163
ah_late = np.sum(current[mask_late] * dt[mask_late]) / 3600
print(f"  Current drawn: {ah_late:.2f} Ah (pack), {ah_late/4:.3f} Ah (cell)")
print(f"  At 4.5 Ah/cell, this is {ah_late/4/4.5*100:.1f}% SOC")

# Active SOC region only
mask_active = (time >= 350) & (time <= 1163)
time_active = time[mask_active]
soc_active = soc[mask_active]
current_active = current[mask_active]
dt_active = dt[mask_active]

cum_ah_active = np.cumsum(current_active * dt_active) / 3600
soc_drop_active = soc_active[0] - soc_active[-1]
ah_active = cum_ah_active[-1]

print(f"\nActive SOC region (t=350-1163s, {time_active[-1]-time_active[0]:.0f}s):")
print(f"  SOC: {soc_active[0]:.1f}% -> {soc_active[-1]:.1f}% (drop={soc_drop_active:.1f}%)")
print(f"  Coulomb: {ah_active:.2f} Ah pack, {ah_active/4:.2f} Ah cell")
print(f"  Implied capacity: {ah_active/4 / (soc_drop_active/100):.2f} Ah")

# Coulomb tracking in active region
for cap in [3.5, 4.0, 4.3, 4.5, 4.7, 5.0]:
    sim_soc = soc_active[0] - (cum_ah_active / (cap * 4)) * 100
    err = sim_soc - soc_active
    print(f"  Cap={cap:.1f}: final={sim_soc[-1]:.1f}% (err={sim_soc[-1]-soc_active[-1]:.1f}%), "
          f"max_err={np.max(np.abs(err)):.1f}%")
