"""
Deep-dive into battery resistance estimation.
The initial analysis showed R_cell = 42.7 mOhm which is 2x+ the typical P45B value.
Need to understand why, and whether the thermal model error is from R or thermal mass.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

TELEMETRY_PATH = Path(r"C:\Users\brand\Development-BC\Real-Car-Data-And-Stats\CleanedEndurance.csv")
VOLTT_CELL_PATH = Path(r"C:\Users\brand\Development-BC\Real-Car-Data-And-Stats\About-Energy-Volt-Simulations-2025-Pack\2025_Pack_cell.csv")
OUTPUT_DIR = Path(r"C:\Users\brand\Development-BC\analysis\thermal_output")

N_SERIES = 110
N_PARALLEL = 4
N_CELLS = N_SERIES * N_PARALLEL

# Load data
df = pd.read_csv(TELEMETRY_PATH, encoding='latin-1', skiprows=[1])
for col in ['Time', 'Pack Temp', 'Pack Voltage', 'Pack Current',
            'State of Charge', 'Min Cell Voltage']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Time', 'Pack Temp', 'Pack Voltage', 'Pack Current'])

time = df['Time'].values
temp = df['Pack Temp'].values
current = df['Pack Current'].values
voltage = df['Pack Voltage'].values
soc = df['State of Charge'].values
dt = np.diff(time, prepend=time[0])

# Load Voltt OCV
voltt = pd.read_csv(VOLTT_CELL_PATH, comment='#')
voltt_soc = voltt['SOC [%]'].values
voltt_ocv = voltt['OCV [V]'].values
sort_idx = np.argsort(voltt_soc)
soc_grid = np.linspace(voltt_soc[sort_idx].min(), voltt_soc[sort_idx].max(), 200)
ocv_grid = np.interp(soc_grid, voltt_soc[sort_idx], voltt_ocv[sort_idx])

def cell_ocv(s):
    return np.interp(s, soc_grid, ocv_grid)

# ============================================================
# Deep dive: What does Voltt say about R?
# ============================================================
print("=" * 70)
print("VOLTT CELL DATA RESISTANCE CHECK")
print("=" * 70)

voltt_v = voltt['Voltage [V]'].values
voltt_i = voltt['Current [A]'].values
voltt_r_mask = voltt_i < -0.5  # Discharge
if np.sum(voltt_r_mask) > 10:
    voltt_r = (voltt_ocv[voltt_r_mask] - voltt_v[voltt_r_mask]) / np.abs(voltt_i[voltt_r_mask])
    valid = (voltt_r > 0.001) & (voltt_r < 0.5)
    print(f"Voltt cell R (discharge, median): {np.median(voltt_r[valid])*1000:.2f} mOhm")
    print(f"Voltt cell R (discharge, mean):   {np.mean(voltt_r[valid])*1000:.2f} mOhm")
    print(f"Voltt cell R (discharge, P25):    {np.percentile(voltt_r[valid], 25)*1000:.2f} mOhm")
    print(f"Voltt cell R (discharge, P75):    {np.percentile(voltt_r[valid], 75)*1000:.2f} mOhm")
    print(f"Valid samples: {np.sum(valid)}")

# ============================================================
# The core question: is R really 42 mOhm per cell?
# ============================================================
print("\n" + "=" * 70)
print("TELEMETRY RESISTANCE ANALYSIS - CAREFUL APPROACH")
print("=" * 70)

# The issue: our OCV curve may not match the BMS SOC definition.
# If OCV(SOC) is wrong, R estimate is biased.
# Better approach: use pairs of measurements at same SOC but different currents.

# Method 1: Direct V-I regression at narrow SOC bands
print("\nMethod 1: V-I slope at constant SOC bands")
print("(R_pack = -dV/dI at constant SOC)")
soc_bands = [(90, 92), (85, 87), (80, 82), (75, 77), (70, 72), (65, 67), (60, 62)]
r_pack_by_soc = []
for soc_lo, soc_hi in soc_bands:
    mask = (soc >= soc_lo) & (soc <= soc_hi) & (current > 0)  # Discharge only
    if np.sum(mask) > 50:
        i_band = current[mask]
        v_band = voltage[mask]
        # Linear regression: V = OCV - I * R_pack
        slope, intercept, r_val, _, se = stats.linregress(i_band, v_band)
        r_pack = -slope  # V decreases with I, so slope is negative
        r_cell = r_pack * N_PARALLEL / N_SERIES
        r_pack_by_soc.append((soc_lo, soc_hi, r_pack, r_cell, r_val**2, np.sum(mask)))
        print(f"  SOC {soc_lo}-{soc_hi}%: R_pack = {r_pack:.4f} Ohm, "
              f"R_cell = {r_cell*1000:.1f} mOhm, R^2 = {r_val**2:.3f}, n = {np.sum(mask)}")

# Average cell resistance from V-I regression
if r_pack_by_soc:
    r_cells_vi = [x[3] for x in r_pack_by_soc]
    r_cell_vi_avg = np.mean(r_cells_vi)
    r_cell_vi_med = np.median(r_cells_vi)
    print(f"\n  Average R_cell (V-I regression): {r_cell_vi_avg*1000:.1f} mOhm")
    print(f"  Median R_cell (V-I regression):  {r_cell_vi_med*1000:.1f} mOhm")

# Method 2: Instantaneous voltage drops at throttle transients
print("\nMethod 2: Voltage drop during rapid current changes")
# Find points where current changes rapidly
di_dt = np.abs(np.gradient(current, time))
# Large current step = good for R measurement
transient_mask = di_dt > 100  # A/s, large transient
print(f"  Found {np.sum(transient_mask)} high-transient samples")

# Method 3: Pack OCV from zero-current moments
print("\nMethod 3: Pack OCV from near-zero current moments")
zero_i_mask = np.abs(current) < 2.0
if np.sum(zero_i_mask) > 20:
    soc_zero = soc[zero_i_mask]
    v_zero = voltage[zero_i_mask]
    # These are approximately OCV points
    cell_v_zero = v_zero / N_SERIES
    print(f"  Found {np.sum(zero_i_mask)} near-zero current samples")
    print(f"  SOC range at zero-current: {soc_zero.min():.1f}% to {soc_zero.max():.1f}%")

    # Compare with Voltt OCV
    voltt_at_soc = cell_ocv(soc_zero)
    ocv_error = cell_v_zero - voltt_at_soc
    print(f"  Cell voltage at zero-I vs Voltt OCV:")
    print(f"    Mean error: {np.mean(ocv_error)*1000:.1f} mV")
    print(f"    Std error:  {np.std(ocv_error)*1000:.1f} mV")
    print(f"    This matters because a {np.mean(np.abs(ocv_error))*1000:.0f} mV OCV error at 20A cell current")
    print(f"    biases R estimate by {np.mean(np.abs(ocv_error))/0.005:.0f} mOhm (=dV/I_cell_typical)")

    # Plot OCV comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(soc_zero[::5], cell_v_zero[::5]*1000, s=5, alpha=0.3, label='Telemetry (V=0 current)')
    soc_plot = np.linspace(55, 98, 100)
    ax.plot(soc_plot, cell_ocv(soc_plot)*1000, 'r-', linewidth=2, label='Voltt OCV')
    ax.set_xlabel('SOC (%)')
    ax.set_ylabel('Cell Voltage (mV)')
    ax.legend()
    ax.set_title('Cell OCV: Telemetry vs Voltt Model')
    ax.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'ocv_comparison.png', dpi=150)
    plt.close()

# ============================================================
# Use the V-I regression R (most reliable)
# ============================================================
print("\n" + "=" * 70)
print("THERMAL MODEL RECOMPUTATION WITH V-I REGRESSION R")
print("=" * 70)

r_pack_best = np.median([x[2] for x in r_pack_by_soc]) if r_pack_by_soc else 1.0
r_cell_best = r_pack_best * N_PARALLEL / N_SERIES

print(f"Using R_pack = {r_pack_best:.4f} Ohm, R_cell = {r_cell_best*1000:.1f} mOhm")

# Compute heat with this R
heat_rate = current**2 * r_pack_best
total_heat = np.sum(heat_rate * dt)
cum_heat = np.cumsum(heat_rate * dt)

# Thermal mass
CELL_MASS = 0.070
CELL_CP = 1000.0
thermal_mass_model = CELL_MASS * CELL_CP * N_CELLS

actual_rise = temp[-1] - temp[0]
predicted_rise = total_heat / thermal_mass_model

print(f"\nTotal I^2*R heat: {total_heat:.0f} J ({total_heat/1000:.1f} kJ)")
print(f"Thermal mass (model): {thermal_mass_model:.0f} J/K")
print(f"Predicted temp rise: {predicted_rise:.2f} C")
print(f"Actual temp rise: {actual_rise:.1f} C")
print(f"Overprediction factor: {predicted_rise / actual_rise:.2f}x")

# ============================================================
# Is the thermal mass wrong, or is there cooling?
# ============================================================
print("\n" + "=" * 70)
print("INVESTIGATING: THERMAL MASS vs COOLING")
print("=" * 70)

# The pack is 110S4P. But what is the physical structure?
# 5 segments x 22S x 4P. Each segment has housing, busbars, etc.
# The thermal mass should include:
# 1. Cell mass: 440 * 0.070 = 30.8 kg
# 2. Module housing, busbars, cooling plates, insulation
# Total pack mass from DSS: check

# Effective thermal mass from data:
effective_tm = total_heat / actual_rise
print(f"Effective thermal mass from data: {effective_tm:.0f} J/K")
print(f"Model thermal mass (cells only): {thermal_mass_model:.0f} J/K")
print(f"Ratio: {effective_tm / thermal_mass_model:.2f}")

# Possibility 1: Real thermal mass is higher (includes non-cell mass)
# What total mass would explain it?
implied_total_mass = effective_tm / CELL_CP
print(f"\nIf Cp = {CELL_CP}: implied total pack mass = {implied_total_mass:.1f} kg")
print(f"Cell-only mass: {CELL_MASS * N_CELLS:.1f} kg")
print(f"This means {implied_total_mass - CELL_MASS * N_CELLS:.1f} kg of non-cell thermal mass")

# Possibility 2: Convective cooling removes some heat
# If there's airflow, h * A * (T - T_ambient) removes heat
# Check: is dT/dt lower when T is higher? (would indicate cooling)
temp_smooth = pd.Series(temp).rolling(50, center=True).mean().values
dTdt = np.gradient(temp_smooth, time)
valid = ~np.isnan(dTdt)

# Bin dT/dt by temperature (not current)
temp_bins = np.arange(29, 39, 1)
print("\ndT/dt vs Temperature (checking for cooling effect):")
for i in range(len(temp_bins) - 1):
    bmask = valid & (temp >= temp_bins[i]) & (temp < temp_bins[i+1])
    if np.sum(bmask) > 50:
        mean_dTdt = np.mean(dTdt[bmask]) * 1000
        mean_i2 = np.mean(current[bmask]**2)
        # Expected dT/dt from I^2*R alone
        expected_dTdt = mean_i2 * r_pack_best / thermal_mass_model * 1000
        print(f"  T={temp_bins[i]}-{temp_bins[i+1]}C: dT/dt = {mean_dTdt:.2f} mC/s, "
              f"expected = {expected_dTdt:.2f} mC/s, "
              f"mean I^2 = {mean_i2:.0f} A^2")

# ============================================================
# Check: what does the Voltt simulation predict for heat?
# ============================================================
print("\n" + "=" * 70)
print("VOLTT SIMULATION HEAT GENERATION DATA")
print("=" * 70)

if 'Heat Generation [W]' in voltt.columns:
    voltt_heat = voltt['Heat Generation [W]'].values
    voltt_r_heat = voltt['Resistive Heat [W]'].values if 'Resistive Heat [W]' in voltt.columns else None
    voltt_rev_heat = voltt['Reversible Heat [W]'].values if 'Reversible Heat [W]' in voltt.columns else None
    voltt_time = voltt['Time [s]'].values
    voltt_dt = np.diff(voltt_time, prepend=voltt_time[0])

    total_voltt_heat = np.sum(voltt_heat * voltt_dt)
    print(f"Voltt total heat per cell: {total_voltt_heat:.2f} J")
    if voltt_r_heat is not None:
        total_voltt_r = np.sum(voltt_r_heat * voltt_dt)
        print(f"Voltt resistive heat per cell: {total_voltt_r:.2f} J")
    if voltt_rev_heat is not None:
        total_voltt_rev = np.sum(voltt_rev_heat * voltt_dt)
        print(f"Voltt reversible heat per cell: {total_voltt_rev:.2f} J")

    # Voltt temp profile
    if 'Temperature [°C]' in voltt.columns:
        voltt_temp = voltt['Temperature [°C]'].values
        print(f"\nVoltt cell temperature: {voltt_temp[0]:.1f}C -> {voltt_temp[-1]:.1f}C")
        print(f"Voltt temperature rise: {voltt_temp[-1] - voltt_temp[0]:.2f}C")

# ============================================================
# Pack temp sensor: what is it actually measuring?
# ============================================================
print("\n" + "=" * 70)
print("PACK TEMP SENSOR ANALYSIS")
print("=" * 70)

# The "Pack Temp" might be a single thermistor on the pack housing,
# not the cell temperature. Thermal lag is expected.
# Check response time by looking at temp vs heat input phase relationship

# Cross-correlation of heat_rate and dT/dt
from scipy.signal import correlate
heat_smooth = pd.Series(heat_rate).rolling(50, center=True).mean().values
valid2 = ~np.isnan(heat_smooth) & ~np.isnan(dTdt)
if np.sum(valid2) > 100:
    h = heat_smooth[valid2]
    t_rate = dTdt[valid2]
    # Normalize
    h_n = (h - np.mean(h)) / (np.std(h) + 1e-10)
    t_n = (t_rate - np.mean(t_rate)) / (np.std(t_rate) + 1e-10)
    corr = correlate(t_n, h_n, mode='full')
    lags = np.arange(-len(h_n)+1, len(h_n))
    lag_time = lags * np.mean(np.diff(time))
    # Find peak correlation
    peak_idx = np.argmax(corr)
    peak_lag = lag_time[peak_idx]
    print(f"Cross-correlation peak lag (heat->temp): {peak_lag:.1f} s")
    print(f"  (positive = temp lags behind heat, expected for thermal inertia)")

# ============================================================
# SOC deep dive
# ============================================================
print("\n" + "=" * 70)
print("SOC DEEP DIVE")
print("=" * 70)

# The BMS reports SOC. Our coulomb counting with 4.5 Ah gives wrong final SOC.
# Implied capacity is 5.9 Ah for a 4.5 Ah cell. This is suspicious.
# Possible causes:
# 1. Pack Current sensor calibration error
# 2. BMS uses a different SOC definition (e.g., based on voltage, not pure coulomb)
# 3. The BMS capacity setting is different from 4.5 Ah

# Check: is the current sensor consistent with power?
power = voltage * current  # W
energy_v_i = np.sum(power * dt) / 3600  # Wh
print(f"Energy from V*I: {energy_v_i:.1f} Wh")
print(f"Energy from SOC change: {(soc[0]-soc[-1])/100 * 4.5 * 4 * np.mean(voltage):.1f} Wh (4.5 Ah)")
print(f"Energy from SOC change: {(soc[0]-soc[-1])/100 * 5.9 * 4 * np.mean(voltage):.1f} Wh (5.9 Ah)")

# Time-resolved SOC error with 4.5 Ah
pack_cap_45 = 4.5 * N_PARALLEL
cum_ah = np.cumsum(current * dt) / 3600
sim_soc_45 = soc[0] - (cum_ah / pack_cap_45) * 100

# Find where the error grows
soc_err = sim_soc_45 - soc
print(f"\nSOC error profile (4.5 Ah capacity):")
quarters = [0, 0.25, 0.5, 0.75, 1.0]
for q in quarters:
    idx = int(q * (len(time) - 1))
    print(f"  t={time[idx]:.0f}s ({q*100:.0f}%): BMS={soc[idx]:.1f}%, "
          f"Sim={sim_soc_45[idx]:.1f}%, Error={soc_err[idx]:.1f}%")

# Check if SOC includes driver change period (car sitting with no current)
# Look for long periods of near-zero current
zero_runs = []
in_run = False
run_start = 0
for i in range(len(current)):
    if np.abs(current[i]) < 1.0:
        if not in_run:
            in_run = True
            run_start = i
    else:
        if in_run:
            run_len = time[i] - time[run_start]
            if run_len > 5:  # > 5s of zero current
                zero_runs.append((time[run_start], time[i], run_len,
                                  soc[run_start], soc[i]))
            in_run = False

print(f"\nPeriods of near-zero current (>5s):")
for start_t, end_t, dur, soc_s, soc_e in zero_runs:
    print(f"  t={start_t:.0f}-{end_t:.0f}s (dur={dur:.0f}s): "
          f"SOC {soc_s:.1f}% -> {soc_e:.1f}% (change={soc_e-soc_s:.2f}%)")

# Check: is SOC step-like (integer resolution)?
soc_unique = np.unique(soc)
soc_diffs = np.diff(np.sort(soc_unique))
print(f"\nSOC sensor:")
print(f"  Unique values: {len(soc_unique)}")
print(f"  Resolution: {np.min(soc_diffs[soc_diffs > 0]):.2f}% (minimum step)")
print(f"  Values: {np.sort(soc_unique)[:10]}... to ...{np.sort(soc_unique)[-5:]}")

# ============================================================
# Final thermal model recommendation
# ============================================================
print("\n" + "=" * 70)
print("FINAL THERMAL MODEL ASSESSMENT")
print("=" * 70)

# Use V-I regression R
r_cell_final = r_cell_vi_med if r_pack_by_soc else 0.020
r_pack_final = r_cell_final * N_SERIES / N_PARALLEL

# Recompute heat with this R
heat_final = current**2 * r_pack_final
total_heat_final = np.sum(heat_final * dt)

# The model predicts much more temp rise than observed.
# This means either:
# A) Thermal mass is wrong (too low -- need more mass)
# B) There IS cooling (convective? conductive to chassis?)
# C) Resistance is wrong

# To separate A from B:
# If no cooling, cumulative heat curve should match temp curve shape.
# If cooling exists, temp should saturate / lag behind heat.

predicted_temp_model = temp[0] + np.cumsum(heat_final * dt) / thermal_mass_model
predicted_temp_4x = temp[0] + np.cumsum(heat_final * dt) / (thermal_mass_model * 4)

# Fit optimal thermal mass + cooling coefficient
# T(t) = T_amb + Q_cum / (m*Cp) - integral(h*(T-T_amb))
# Simplified: effective_tm = Q_total / dT

# But let's also try: m*Cp * dT/dt = I^2*R - h*(T - T_amb)
# Linear regression: dT/dt = a * I^2 - b * (T - T_amb) + c
# where a = R_pack / (m*Cp), b = h / (m*Cp)

valid3 = ~np.isnan(dTdt) & ~np.isnan(temp_smooth)
T_minus_Tamb = temp_smooth[valid3] - temp[0]  # Using start temp as ambient proxy
I_squared = current[valid3]**2
dTdt_v = dTdt[valid3]

# Multiple regression: dT/dt = a * I^2 + b * (T-Tamb) + c
X = np.column_stack([I_squared, T_minus_Tamb, np.ones(np.sum(valid3))])
result = np.linalg.lstsq(X, dTdt_v, rcond=None)
a, b, c = result[0]

print(f"\nMultiple regression: dT/dt = a*I^2 + b*(T-Tamb) + c")
print(f"  a (R/thermal_mass) = {a:.2e}")
print(f"  b (cooling/thermal_mass) = {b:.2e}")
print(f"  c (constant) = {c:.2e}")

if a > 0:
    implied_tm_from_R = r_pack_final / a
    print(f"\n  Implied thermal mass from a and R_pack: {implied_tm_from_R:.0f} J/K")
    print(f"  Model thermal mass: {thermal_mass_model:.0f} J/K")
    print(f"  Ratio: {implied_tm_from_R / thermal_mass_model:.2f}")

if b < 0:
    print(f"\n  Cooling coefficient b is NEGATIVE (= cooling present)")
    h_eff = -b * implied_tm_from_R if a > 0 else -b * thermal_mass_model
    print(f"  Effective cooling coefficient h*A: {h_eff:.2f} W/K")
    print(f"  At dT=9C, cooling power: {h_eff * 9:.0f} W")
else:
    print(f"\n  Cooling coefficient b is positive or zero (no clear cooling signal)")

# Simulate with fitted parameters
if a > 0:
    sim_temp_fitted = np.zeros(len(time))
    sim_temp_fitted[0] = temp[0]
    for i in range(1, len(time)):
        dTdt_sim = a * current[i]**2 + b * (sim_temp_fitted[i-1] - temp[0]) + c
        sim_temp_fitted[i] = sim_temp_fitted[i-1] + dTdt_sim * dt[i]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(time, temp, 'r-', linewidth=1, label='Actual (telemetry)')
    ax1.plot(time, predicted_temp_model, 'b--', linewidth=1,
             label=f'Model (R={r_pack_final:.3f}, TM={thermal_mass_model:.0f})')
    ax1.plot(time, predicted_temp_4x, 'g--', linewidth=1,
             label=f'Model 4x thermal mass')
    ax1.plot(time, sim_temp_fitted, 'k-', linewidth=1.5,
             label='Fitted (I^2*R + cooling)')
    ax1.set_ylabel('Temperature (C)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(fontsize=8)
    ax1.set_title('Temperature Models Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(25, 75)

    err_model = predicted_temp_model - temp
    err_4x = predicted_temp_4x - temp
    err_fitted = sim_temp_fitted - temp
    ax2.plot(time, err_model, 'b-', linewidth=0.5, label=f'Model error (RMSE={np.sqrt(np.mean(err_model**2)):.1f}C)')
    ax2.plot(time, err_4x, 'g-', linewidth=0.5, label=f'4x TM error (RMSE={np.sqrt(np.mean(err_4x**2)):.1f}C)')
    ax2.plot(time, err_fitted, 'k-', linewidth=1, label=f'Fitted error (RMSE={np.sqrt(np.mean(err_fitted**2)):.1f}C)')
    ax2.set_ylabel('Error (C)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(fontsize=8)
    ax2.set_title('Temperature Prediction Error')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linewidth=0.5)
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temp_models_comparison.png', dpi=150)
    plt.close()

# ============================================================
# Summary of cell R from all methods
# ============================================================
print("\n" + "=" * 70)
print("RESISTANCE SUMMARY (ALL METHODS)")
print("=" * 70)
print(f"{'Method':<35s} | {'R_cell (mOhm)':<15s}")
print("-" * 55)
if r_pack_by_soc:
    for soc_lo, soc_hi, rp, rc, r2, n in r_pack_by_soc:
        print(f"V-I regression SOC {soc_lo}-{soc_hi}%       | {rc*1000:.1f}")
    print(f"{'V-I regression (median)':35s} | {r_cell_vi_med*1000:.1f}")
    print(f"{'V-I regression (mean)':35s} | {r_cell_vi_avg*1000:.1f}")
if np.sum(voltt_r_mask) > 10:
    print(f"{'Voltt simulation':35s} | {np.median(voltt_r[valid])*1000:.1f}")
print(f"{'Typical P45B datasheet':35s} | 15-25")
if a > 0:
    r_from_thermal = a * implied_tm_from_R * N_PARALLEL / N_SERIES
    print(f"{'From thermal fit (a*TM)':35s} | {r_from_thermal*1000:.1f}")

print(f"\nExpected R_cell for P45B at 25-30C: ~15-25 mOhm (datasheet)")
print(f"V-I regression gives: {r_cell_vi_med*1000:.1f} mOhm")
print(f"This is {'within' if 15 <= r_cell_vi_med*1000 <= 50 else 'outside'} expected range")
print(f"\nNote: V-I regression R includes connector resistance, busbar resistance,")
print(f"and measurement noise. True cell R is likely lower.")
