"""Quick check on speed and SOC data quality."""
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

# Check all speed-like columns
speed_cols = [c for c in df.columns if 'speed' in c.lower() or 'Speed' in c or 'LF' in c.lower()]
print("Speed-like columns:", speed_cols)
for c in speed_cols:
    vals = df[c].dropna().values
    print(f"  {c}: min={np.min(vals):.1f}, max={np.max(vals):.1f}, mean={np.mean(vals):.1f}")

# GPS Speed may need different parsing
print(f"\nGPS Speed sample values: {df['GPS Speed'].values[:20] if 'GPS Speed' in df.columns else 'N/A'}")

# Use LFspeed as car speed proxy
if 'LFspeed' in df.columns:
    lfspeed = df['LFspeed'].values
    print(f"\nLFspeed: min={np.nanmin(lfspeed):.1f}, max={np.nanmax(lfspeed):.1f}")
    moving = lfspeed > 5
    print(f"Moving samples (>5 km/h): {np.sum(moving)} of {len(lfspeed)}")

# SOC unique values and pattern
soc_unique = np.sort(np.unique(soc[~np.isnan(soc)]))
print(f"\nSOC unique values ({len(soc_unique)} total):")
print(f"  {soc_unique[:20]}")
print(f"  ...")
print(f"  {soc_unique[-20:]}")

# Is SOC quantized to 0.5%?
soc_diffs = np.diff(soc_unique)
print(f"\nSOC step sizes: {np.unique(np.round(soc_diffs, 3))[:20]}")

# Show SOC change profile: where does SOC actually change?
soc_changes = np.where(np.abs(np.diff(soc)) > 0.01)[0]
print(f"\nSOC change events: {len(soc_changes)}")
if len(soc_changes) > 0:
    print(f"  First change: t={time[soc_changes[0]]:.1f}s, SOC {soc[soc_changes[0]]:.2f}% -> {soc[soc_changes[0]+1]:.2f}%")
    print(f"  Last change: t={time[soc_changes[-1]]:.1f}s, SOC {soc[soc_changes[-1]]:.2f}% -> {soc[soc_changes[-1]+1]:.2f}%")

# SOC flat periods
flat_regions = []
i = 0
while i < len(soc) - 1:
    start_i = i
    while i < len(soc) - 1 and abs(soc[i+1] - soc[start_i]) < 0.1:
        i += 1
    duration = time[i] - time[start_i]
    if duration > 30:  # >30s flat
        flat_regions.append((time[start_i], time[i], duration, soc[start_i]))
    i += 1

print(f"\nSOC flat regions (>30s at same value):")
for s, e, d, sv in flat_regions:
    idx_s = np.argmin(np.abs(time - s))
    idx_e = np.argmin(np.abs(time - e))
    avg_i = np.mean(current[idx_s:idx_e])
    total_ah = np.sum(current[idx_s:idx_e] * np.diff(time[idx_s:idx_e+1], prepend=time[idx_s])) / 3600
    print(f"  t={s:.0f}-{e:.0f}s ({d:.0f}s): SOC={sv:.1f}%, avg I={avg_i:.1f}A, total Ah={total_ah:.2f}")
