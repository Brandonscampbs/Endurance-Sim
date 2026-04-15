"""
Battery Thermal Model Accuracy Analysis
Analyzes telemetry data to evaluate the simulation's thermal model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
TELEMETRY_PATH = Path(r"C:\Users\brand\Development-BC\Real-Car-Data-And-Stats\CleanedEndurance.csv")
VOLTT_CELL_PATH = Path(r"C:\Users\brand\Development-BC\Real-Car-Data-And-Stats\About-Energy-Volt-Simulations-2025-Pack\2025_Pack_cell.csv")
OUTPUT_DIR = Path(r"C:\Users\brand\Development-BC\analysis\thermal_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load telemetry
# ============================================================
print("=" * 70)
print("LOADING TELEMETRY DATA")
print("=" * 70)

df = pd.read_csv(TELEMETRY_PATH, encoding='latin-1', skiprows=[1])
# Convert relevant columns to numeric
for col in ['Time', 'Pack Temp', 'Pack Voltage', 'Pack Current',
            'State of Charge', 'Min Cell Voltage', 'Motor RPM', 'GPS Speed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Time', 'Pack Temp', 'Pack Voltage', 'Pack Current'])
print(f"Loaded {len(df)} rows, time range: {df['Time'].min():.1f} to {df['Time'].max():.1f} s")
print(f"Columns available: {list(df.columns)}")

# ============================================================
# 2. Pack Temperature Profile
# ============================================================
print("\n" + "=" * 70)
print("SECTION 1: PACK TEMPERATURE vs TIME")
print("=" * 70)

time = df['Time'].values
temp = df['Pack Temp'].values
current = df['Pack Current'].values
voltage = df['Pack Voltage'].values
soc = df['State of Charge'].values

print(f"Temperature range: {temp.min():.1f}C to {temp.max():.1f}C")
print(f"Starting temperature (first 10 samples avg): {np.mean(temp[:10]):.2f} C")
print(f"Starting temperature (first 50 samples avg): {np.mean(temp[:50]):.2f} C")
print(f"Starting temperature (first sample): {temp[0]:.1f} C")
print(f"Ending temperature (last 10 samples avg): {np.mean(temp[-10:]):.2f} C")
print(f"Ending temperature (last sample): {temp[-1]:.1f} C")
print(f"Total temperature rise: {temp[-1] - temp[0]:.2f} C")
print(f"Event duration: {time[-1] - time[0]:.1f} s")

# Temperature at key milestones
quartiles = [0, 0.25, 0.50, 0.75, 1.0]
print("\nTemperature at event quartiles:")
for q in quartiles:
    idx = int(q * (len(time) - 1))
    print(f"  {q*100:.0f}% (t={time[idx]:.0f}s): {temp[idx]:.1f} C, SOC={soc[idx]:.1f}%, I={current[idx]:.1f}A")

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(time, temp, 'r-', linewidth=0.5, alpha=0.7, label='Pack Temp')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pack Temperature (C)', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(time, current, 'b-', linewidth=0.3, alpha=0.3, label='Pack Current')
ax2.set_ylabel('Pack Current (A)', color='b')
ax2.tick_params(axis='y', labelcolor='b')
plt.title('Pack Temperature and Current vs Time')
fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'temp_vs_time.png', dpi=150)
plt.close()

# ============================================================
# 3. Starting Temperature Analysis
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: STARTING TEMPERATURE ANALYSIS")
print("=" * 70)

# Look at first 30 seconds
mask_start = time < (time[0] + 30)
print(f"First 30s temperature stats:")
print(f"  Mean: {np.mean(temp[mask_start]):.2f} C")
print(f"  Std:  {np.std(temp[mask_start]):.3f} C")
print(f"  Min:  {np.min(temp[mask_start]):.1f} C")
print(f"  Max:  {np.max(temp[mask_start]):.1f} C")

# Temperature sensor resolution
temp_unique = np.unique(temp[:200])
if len(temp_unique) > 1:
    diffs = np.diff(np.sort(temp_unique))
    resolution = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 0
    print(f"\nTemperature sensor resolution (est): {resolution:.2f} C")
    print(f"Unique temp values in first 200 samples: {temp_unique}")

# ============================================================
# 4. Compute I^2*R Heat Generation
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: HEAT GENERATION ANALYSIS (I^2*R)")
print("=" * 70)

# Load Voltt cell data for OCV-SOC curve
voltt = pd.read_csv(VOLTT_CELL_PATH, comment='#')
voltt_soc = voltt['SOC [%]'].values
voltt_ocv = voltt['OCV [V]'].values

# Build OCV interpolator
sort_idx = np.argsort(voltt_soc)
voltt_soc_sorted = voltt_soc[sort_idx]
voltt_ocv_sorted = voltt_ocv[sort_idx]
soc_grid = np.linspace(voltt_soc_sorted.min(), voltt_soc_sorted.max(), 200)
ocv_grid = np.interp(soc_grid, voltt_soc_sorted, voltt_ocv_sorted)

def ocv_lookup(soc_pct):
    return np.interp(soc_pct, soc_grid, ocv_grid)

# Estimate pack OCV and internal resistance from telemetry
# For each point: V_pack = OCV_cell * N_series - I * R_pack
# So: R_pack = (OCV_cell * N_series - V_pack) / I
N_SERIES = 110
N_PARALLEL = 4

# Cell-level approach: pack_current / N_parallel = cell current
cell_current = current / N_PARALLEL
pack_ocv_estimated = ocv_lookup(soc) * N_SERIES

# Only compute R where current is meaningful (> 5A pack, i.e. > 1.25A/cell)
mask_load = current > 5.0
if np.sum(mask_load) > 100:
    r_pack_est = (pack_ocv_estimated[mask_load] - voltage[mask_load]) / current[mask_load]
    valid_r = (r_pack_est > 0.05) & (r_pack_est < 10.0)
    r_pack_median = np.median(r_pack_est[valid_r])
    r_cell_median = r_pack_median * N_PARALLEL / N_SERIES  # R_pack = R_cell * N_s / N_p
    print(f"Estimated pack resistance: {r_pack_median:.4f} Ohm")
    print(f"Estimated cell resistance: {r_cell_median*1000:.2f} mOhm")

    # Resistance vs SOC
    soc_load = soc[mask_load][valid_r]
    r_cell_values = r_pack_est[valid_r] * N_PARALLEL / N_SERIES

    # Bin by SOC
    soc_bins = np.arange(40, 100, 5)
    print("\nResistance vs SOC:")
    r_vs_soc_centers = []
    r_vs_soc_medians = []
    for i in range(len(soc_bins) - 1):
        bmask = (soc_load >= soc_bins[i]) & (soc_load < soc_bins[i+1])
        if np.sum(bmask) > 10:
            r_med = np.median(r_cell_values[bmask]) * 1000
            r_vs_soc_centers.append((soc_bins[i] + soc_bins[i+1]) / 2)
            r_vs_soc_medians.append(r_med)
            print(f"  SOC {soc_bins[i]}-{soc_bins[i+1]}%: R_cell = {r_med:.2f} mOhm (n={np.sum(bmask)})")

# Compute cumulative I^2*R heat
dt = np.diff(time, prepend=time[0])
# Use per-cell values for comparison with model
# The model does: heat_per_cell = I_cell^2 * R_cell, total = heat_per_cell * N_cells
# Equivalent to: I_pack^2 * R_pack (since R_pack = R_cell * Ns/Np, I_cell = I_pack/Np)
# I_cell^2 * R_cell * Ns * Np = (I_pack/Np)^2 * R_cell * Ns * Np = I_pack^2 * R_cell * Ns / Np = I_pack^2 * R_pack

heat_rate_w = current**2 * r_pack_median  # Total pack I^2*R heating (W)
cumulative_heat_j = np.cumsum(heat_rate_w * dt)

print(f"\nTotal I^2*R heat generated: {cumulative_heat_j[-1]:.0f} J = {cumulative_heat_j[-1]/1000:.2f} kJ")
print(f"Average heating rate: {np.mean(heat_rate_w):.1f} W")
print(f"Peak heating rate: {np.max(heat_rate_w):.1f} W")

# What temperature rise does this predict?
N_CELLS = N_SERIES * N_PARALLEL
CELL_MASS_KG = 0.070
CELL_CP = 1000.0  # J/(kg*K)
thermal_mass = CELL_MASS_KG * CELL_CP * N_CELLS
predicted_dT = cumulative_heat_j[-1] / thermal_mass
print(f"\nThermal mass (model): {thermal_mass:.0f} J/K")
print(f"  = {N_CELLS} cells * {CELL_MASS_KG} kg * {CELL_CP} J/(kg*K)")
print(f"Predicted temperature rise from I^2*R: {predicted_dT:.2f} C")
print(f"Actual temperature rise: {temp[-1] - temp[0]:.2f} C")
print(f"Ratio (actual/predicted): {(temp[-1] - temp[0]) / predicted_dT:.2f}")

# Time-resolved: predicted temp vs actual
predicted_temp = temp[0] + np.cumsum(heat_rate_w * dt) / thermal_mass

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(time, temp, 'r-', linewidth=1, label='Actual (telemetry)')
ax1.plot(time, predicted_temp, 'b--', linewidth=1, label=f'Predicted (I^2*R, R={r_pack_median:.3f} Ohm)')
ax1.set_ylabel('Temperature (C)')
ax1.set_xlabel('Time (s)')
ax1.legend()
ax1.set_title('Pack Temperature: Actual vs I^2*R Model Prediction')
ax1.grid(True, alpha=0.3)

ax2.plot(time, temp - predicted_temp, 'g-', linewidth=1)
ax2.set_ylabel('Error: Actual - Predicted (C)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Temperature Prediction Error')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'temp_actual_vs_predicted.png', dpi=150)
plt.close()

# ============================================================
# 5. Thermal Rise Rate vs Current Magnitude
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: THERMAL RISE RATE vs CURRENT")
print("=" * 70)

# Smooth temperature to get dT/dt (temp sensor has limited resolution)
# Use 50-sample (~2.5s) rolling average
window = 50
temp_smooth = pd.Series(temp).rolling(window=window, center=True).mean().values
dT_dt = np.gradient(temp_smooth, time)

# Remove NaNs from rolling average edges
valid = ~np.isnan(dT_dt)
time_v = time[valid]
dT_dt_v = dT_dt[valid]
current_v = current[valid]
current_sq_v = current_v**2

# Bin dT/dt by current magnitude
current_bins = np.arange(0, 110, 10)
print("dT/dt vs current magnitude (binned):")
bin_centers = []
bin_dTdt_means = []
bin_current_sq_means = []
for i in range(len(current_bins) - 1):
    bmask = (np.abs(current_v) >= current_bins[i]) & (np.abs(current_v) < current_bins[i+1])
    if np.sum(bmask) > 20:
        mean_dTdt = np.mean(dT_dt_v[bmask]) * 1000  # mC/s
        mean_Isq = np.mean(current_v[bmask]**2)
        bin_centers.append((current_bins[i] + current_bins[i+1]) / 2)
        bin_dTdt_means.append(mean_dTdt)
        bin_current_sq_means.append(mean_Isq)
        print(f"  |I|={current_bins[i]}-{current_bins[i+1]}A: dT/dt = {mean_dTdt:.3f} mC/s, I^2 = {mean_Isq:.0f} A^2 (n={np.sum(bmask)})")

# Correlation: dT/dt vs I^2
corr_r, corr_p = stats.pearsonr(current_sq_v[current_v > 1], dT_dt_v[current_v > 1])
print(f"\nCorrelation (dT/dt vs I^2): r = {corr_r:.4f}, p = {corr_p:.2e}")

# Linear regression: dT/dt = a * I^2 + b
slope, intercept, r_sq, p_val, se = stats.linregress(current_sq_v[current_v > 1], dT_dt_v[current_v > 1])
print(f"Linear fit: dT/dt = {slope:.2e} * I^2 + {intercept:.2e}")
print(f"  R^2 = {r_sq**2:.4f}")
print(f"  Implied R_pack from slope * thermal_mass = {slope * thermal_mass:.4f} Ohm")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.scatter(np.abs(current_v[::10]), dT_dt_v[::10]*1000, s=1, alpha=0.1, c='blue')
if len(bin_centers) > 0:
    ax1.scatter(bin_centers, bin_dTdt_means, s=50, c='red', zorder=5, label='Bin medians')
ax1.set_xlabel('|Current| (A)')
ax1.set_ylabel('dT/dt (mC/s)')
ax1.set_title('Thermal Rise Rate vs Current Magnitude')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.scatter(current_sq_v[::10], dT_dt_v[::10]*1000, s=1, alpha=0.1, c='blue')
if len(bin_current_sq_means) > 0:
    ax2.scatter(bin_current_sq_means, bin_dTdt_means, s=50, c='red', zorder=5, label='Bin medians')
x_fit = np.linspace(0, max(current_sq_v), 100)
ax2.plot(x_fit, (slope * x_fit + intercept)*1000, 'r-', linewidth=2, label=f'Fit: R^2={r_sq**2:.3f}')
ax2.set_xlabel('I^2 (A^2)')
ax2.set_ylabel('dT/dt (mC/s)')
ax2.set_title('Thermal Rise Rate vs I^2')
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'dTdt_vs_current.png', dpi=150)
plt.close()

# ============================================================
# 6. Temperature Rise Linearity
# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: TEMPERATURE RISE LINEARITY")
print("=" * 70)

# Fit a linear model to temp vs time
slope_t, intercept_t, r_val_t, p_val_t, se_t = stats.linregress(time, temp)
temp_linear_fit = slope_t * time + intercept_t
residual = temp - temp_linear_fit

print(f"Linear fit: T = {slope_t*1000:.4f} mC/s * t + {intercept_t:.2f} C")
print(f"R^2 = {r_val_t**2:.4f}")
print(f"Max residual: {np.max(np.abs(residual)):.3f} C")
print(f"Mean absolute residual: {np.mean(np.abs(residual)):.3f} C")

# Is the residual pattern consistent with I^2 driving? (non-linear)
# Compute cumulative I^2 (normalized) and compare to temp profile
cum_i2 = np.cumsum(current**2 * dt) / np.sum(current**2 * dt)
norm_temp = (temp - temp[0]) / (temp[-1] - temp[0]) if temp[-1] != temp[0] else np.zeros_like(temp)
norm_time = (time - time[0]) / (time[-1] - time[0])

corr_i2_temp, _ = stats.pearsonr(cum_i2, norm_temp)
corr_time_temp, _ = stats.pearsonr(norm_time, norm_temp)
print(f"\nCorrelation of normalized temp rise with:")
print(f"  Cumulative I^2*dt (normalized): r = {corr_i2_temp:.4f}")
print(f"  Linear time (normalized):       r = {corr_time_temp:.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(norm_time, norm_temp, 'r-', linewidth=1, label='Normalized temp rise')
ax.plot(norm_time, cum_i2, 'b-', linewidth=1, label='Normalized cumulative I^2*dt')
ax.plot(norm_time, norm_time, 'k--', linewidth=0.5, label='Linear reference')
ax.set_xlabel('Normalized time')
ax.set_ylabel('Normalized value')
ax.legend()
ax.set_title('Temperature Rise Shape: I^2*dt vs Linear')
ax.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / 'temp_linearity.png', dpi=150)
plt.close()

# ============================================================
# 7. Min Cell Voltage Analysis
# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: MIN CELL VOLTAGE PROFILE")
print("=" * 70)

if 'Min Cell Voltage' in df.columns:
    min_v = df['Min Cell Voltage'].values
    # Check if units are mV (values > 100 suggest mV)
    if np.nanmedian(min_v) > 100:
        min_v = min_v / 1000.0  # convert mV to V
        print("Min Cell Voltage appears to be in mV, converted to V")

    # Find absolute minimum
    valid_v = ~np.isnan(min_v)
    min_v_clean = min_v[valid_v]
    time_clean = time[valid_v]
    soc_clean = soc[valid_v]
    current_clean = current[valid_v]
    temp_clean = temp[valid_v]

    abs_min_idx = np.argmin(min_v_clean)
    print(f"\nAbsolute minimum cell voltage: {min_v_clean[abs_min_idx]:.4f} V")
    print(f"  Occurred at: t = {time_clean[abs_min_idx]:.1f} s")
    print(f"  SOC at that time: {soc_clean[abs_min_idx]:.1f}%")
    print(f"  Current at that time: {current_clean[abs_min_idx]:.1f} A")
    print(f"  Temperature at that time: {temp_clean[abs_min_idx]:.1f} C")

    # Voltage under load vs SOC
    print(f"\nMin cell voltage statistics:")
    print(f"  Mean: {np.nanmean(min_v_clean):.4f} V")
    print(f"  Std:  {np.nanstd(min_v_clean):.4f} V")
    print(f"  P5:   {np.nanpercentile(min_v_clean, 5):.4f} V")
    print(f"  P1:   {np.nanpercentile(min_v_clean, 1):.4f} V")

    # Minimum voltage at different SOC ranges
    print("\nMin cell voltage vs SOC:")
    soc_bins_v = np.arange(40, 100, 5)
    for i in range(len(soc_bins_v) - 1):
        bmask = (soc_clean >= soc_bins_v[i]) & (soc_clean < soc_bins_v[i+1])
        if np.sum(bmask) > 10:
            print(f"  SOC {soc_bins_v[i]}-{soc_bins_v[i+1]}%: min={np.min(min_v_clean[bmask]):.4f} V, "
                  f"mean={np.mean(min_v_clean[bmask]):.4f} V")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(time_clean, min_v_clean, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=2.55, color='r', linestyle='--', label='Cell min (2.55V)')
    ax1.axhline(y=min_v_clean[abs_min_idx], color='orange', linestyle=':',
                label=f'Absolute min ({min_v_clean[abs_min_idx]:.3f}V)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Min Cell Voltage (V)')
    ax1.set_title('Minimum Cell Voltage vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(soc_clean[::5], min_v_clean[::5], s=1, alpha=0.2, c=current_clean[::5], cmap='viridis')
    ax2.set_xlabel('SOC (%)')
    ax2.set_ylabel('Min Cell Voltage (V)')
    ax2.set_title('Min Cell Voltage vs SOC (colored by current)')
    cb = plt.colorbar(ax2.collections[0], ax=ax2)
    cb.set_label('Pack Current (A)')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / 'min_cell_voltage.png', dpi=150)
    plt.close()

# ============================================================
# 8. SOC Profile Analysis
# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: SOC PROFILE AND COULOMB COUNTING")
print("=" * 70)

print(f"SOC range: {soc[0]:.1f}% to {soc[-1]:.1f}%")
print(f"Total SOC consumed: {soc[0] - soc[-1]:.1f}%")

# Coulomb counting: integrate current over time
# pack current in A, time in s -> Ah
cum_charge_ah = np.cumsum(current * dt) / 3600  # pack Ah
total_charge_ah = cum_charge_ah[-1]
cell_charge_ah = total_charge_ah / N_PARALLEL
print(f"\nCoulomb counting results:")
print(f"  Total pack charge discharged: {total_charge_ah:.3f} Ah")
print(f"  Per-cell charge discharged: {cell_charge_ah:.3f} Ah")

# What cell capacity is implied?
soc_change_frac = (soc[0] - soc[-1]) / 100.0
implied_capacity_cell = cell_charge_ah / soc_change_frac if soc_change_frac > 0 else float('nan')
print(f"  Implied cell capacity: {implied_capacity_cell:.3f} Ah")
print(f"  Nominal cell capacity: 4.5 Ah")
print(f"  Ratio: {implied_capacity_cell / 4.5:.3f}")

# Simulate SOC via coulomb counting with different capacities
capacities_to_try = [4.0, 4.3, 4.5, 4.7, 5.0, implied_capacity_cell]
print(f"\nSOC tracking error for different cell capacities:")
print(f"{'Capacity (Ah)':>14s} | {'Final SOC sim':>13s} | {'Final SOC real':>14s} | {'Max error':>10s} | {'Mean error':>10s}")
print("-" * 75)

best_capacity = None
best_max_err = float('inf')

for cap in capacities_to_try:
    pack_cap_ah = cap * N_PARALLEL
    sim_soc = soc[0] - (cum_charge_ah / pack_cap_ah) * 100
    err = sim_soc - soc
    max_err = np.max(np.abs(err))
    mean_err = np.mean(np.abs(err))
    print(f"{cap:14.3f} | {sim_soc[-1]:13.1f}% | {soc[-1]:13.1f}% | {max_err:9.2f}% | {mean_err:9.2f}%")
    if max_err < best_max_err:
        best_max_err = max_err
        best_capacity = cap

print(f"\nBest capacity: {best_capacity:.3f} Ah (min max error = {best_max_err:.2f}%)")

# Linearity of SOC discharge
slope_soc, intercept_soc, r_soc, _, _ = stats.linregress(time, soc)
soc_linear = slope_soc * time + intercept_soc
soc_residual = soc - soc_linear
print(f"\nSOC linearity:")
print(f"  Linear fit: SOC = {slope_soc*60:.3f} %/min * t + {intercept_soc:.1f}%")
print(f"  R^2 = {r_soc**2:.6f}")
print(f"  Max deviation from linear: {np.max(np.abs(soc_residual)):.2f}%")

# Plot SOC: actual vs coulomb counting
sim_soc_best = soc[0] - (cum_charge_ah / (best_capacity * N_PARALLEL)) * 100
sim_soc_nominal = soc[0] - (cum_charge_ah / (4.5 * N_PARALLEL)) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(time, soc, 'b-', linewidth=1, label='BMS SOC (telemetry)')
ax1.plot(time, sim_soc_nominal, 'r--', linewidth=1, label='Coulomb counting (4.5 Ah)')
ax1.plot(time, sim_soc_best, 'g--', linewidth=1, label=f'Coulomb counting ({best_capacity:.2f} Ah)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('SOC (%)')
ax1.legend()
ax1.set_title('State of Charge: BMS vs Coulomb Counting')
ax1.grid(True, alpha=0.3)

ax2.plot(time, sim_soc_nominal - soc, 'r-', linewidth=1, label='Error (4.5 Ah)')
ax2.plot(time, sim_soc_best - soc, 'g-', linewidth=1, label=f'Error ({best_capacity:.2f} Ah)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('SOC Error (%)')
ax2.legend()
ax2.set_title('SOC Error Over Time')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / 'soc_tracking.png', dpi=150)
plt.close()

# ============================================================
# 9. Simulation Thermal Model Assessment
# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: SIMULATION THERMAL MODEL ASSESSMENT")
print("=" * 70)

print(f"\n--- Model Parameters ---")
print(f"Cell mass: {CELL_MASS_KG} kg")
print(f"Cell Cp: {CELL_CP} J/(kg*K)")
print(f"Number of cells: {N_CELLS}")
print(f"Thermal mass: {thermal_mass:.0f} J/K ({thermal_mass/1000:.1f} kJ/K)")

print(f"\n--- What the data says ---")
actual_rise = temp[-1] - temp[0]
total_heat = cumulative_heat_j[-1]
implied_thermal_mass = total_heat / actual_rise if actual_rise > 0 else float('nan')
print(f"Actual temperature rise: {actual_rise:.1f} C")
print(f"Total I^2*R heat (R={r_pack_median:.4f} Ohm): {total_heat:.0f} J")
print(f"Implied thermal mass: {implied_thermal_mass:.0f} J/K")
print(f"Model thermal mass: {thermal_mass:.0f} J/K")
print(f"Ratio (implied/model): {implied_thermal_mass / thermal_mass:.3f}")

# What if we also consider heat dissipation (cooling)?
# If model is too cold, either: too little heat, or too much thermal mass, or missing heat sources
# Let's check: what R would be needed to match the observed temperature rise?
cum_i2_dt = np.sum(current**2 * dt)  # A^2 * s
r_pack_needed = actual_rise * thermal_mass / cum_i2_dt
r_cell_needed = r_pack_needed * N_PARALLEL / N_SERIES
print(f"\n--- Required R to match observed temp rise ---")
print(f"R_pack needed: {r_pack_needed:.4f} Ohm")
print(f"R_cell needed: {r_cell_needed*1000:.2f} mOhm")
print(f"R_cell from V-I fit: {r_cell_median*1000:.2f} mOhm")
print(f"Ratio (needed/measured): {r_cell_needed/r_cell_median:.2f}")

# Energy analysis
total_energy_j = np.sum(voltage * current * dt)
total_energy_kwh = total_energy_j / 3.6e6
print(f"\n--- Energy Summary ---")
print(f"Total energy discharged: {total_energy_kwh:.3f} kWh")
print(f"Total I^2*R loss: {total_heat/1000:.2f} kJ = {total_heat/3.6e6:.4f} kWh")
print(f"Resistive loss fraction: {total_heat/total_energy_j*100:.2f}%")

# ============================================================
# 10. Summary Report
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

print(f"""
1. STARTING TEMPERATURE:
   - Real car starts at {temp[0]:.0f}C (telemetry)
   - Simulation assumes 25C
   - Recommendation: Use {temp[0]:.0f}C as initial temperature
   - Impact: {temp[0] - 25:.0f}C offset propagates through entire event

2. THERMAL MODEL HEAT GENERATION:
   - Pack resistance from telemetry V-I: {r_pack_median:.4f} Ohm ({r_cell_median*1000:.1f} mOhm/cell)
   - Total I^2*R heat: {total_heat:.0f} J over {time[-1]:.0f}s
   - Predicted temp rise: {predicted_dT:.2f}C
   - Actual temp rise: {actual_rise:.1f}C
   - The I^2*R model predicts {predicted_dT/actual_rise*100:.0f}% of the observed rise
   - dT/dt correlates well with I^2 (r = {corr_r:.3f})

3. THERMAL MASS:
   - Model: {thermal_mass:.0f} J/K ({N_CELLS} cells x {CELL_MASS_KG} kg x {CELL_CP} J/kg/K)
   - Implied from data: {implied_thermal_mass:.0f} J/K
   - Ratio (implied/model): {implied_thermal_mass/thermal_mass:.2f}
   - The model thermal mass is {'too high' if implied_thermal_mass < thermal_mass else 'too low'} by {abs(1 - implied_thermal_mass/thermal_mass)*100:.0f}%

4. SOC TRACKING:
   - BMS SOC: {soc[0]:.1f}% -> {soc[-1]:.1f}% (consumed {soc[0]-soc[-1]:.1f}%)
   - Coulomb counting with 4.5 Ah -> final SOC = {sim_soc_nominal[-1]:.1f}%
   - Implied cell capacity: {implied_capacity_cell:.2f} Ah
   - SOC discharge is {'approximately linear' if r_soc**2 > 0.99 else 'non-linear'} (R^2 = {r_soc**2:.4f})

5. MIN CELL VOLTAGE:
   - Absolute minimum: {min_v_clean[abs_min_idx]:.3f}V at t={time_clean[abs_min_idx]:.0f}s
   - Headroom above floor (2.55V): {min_v_clean[abs_min_idx] - 2.55:.3f}V
   - SOC at minimum: {soc_clean[abs_min_idx]:.1f}%, Current: {current_clean[abs_min_idx]:.1f}A

6. KEY CORRECTIONS NEEDED:
   a) Set sim initial temperature to {temp[0]:.0f}C (not 25C) -- accounts for {temp[0]-25:.0f}C of the {38-34.5:.1f}C error
   b) Verify R_cell calibration matches {r_cell_median*1000:.1f} mOhm
   c) Remaining temp error after start-temp fix: {actual_rise - predicted_dT:.2f}C
""")

print("Analysis complete. Plots saved to:", OUTPUT_DIR)
