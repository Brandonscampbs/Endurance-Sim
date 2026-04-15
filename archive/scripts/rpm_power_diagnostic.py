"""RPM and Electrical Power diagnostic: sim vs telemetry.

Investigates why the replay sim finishes ~48 seconds too fast by comparing:
1. Motor RPM: sim prediction vs telemetry, and independent gear-ratio/tire-radius validation
2. Electrical power: sim vs telemetry (Pack Voltage * Pack Current)
3. Motor drag at zero throttle (iron/windage losses the sim may be missing)
"""

import math
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Setup: load data and run replay sim (same as validate_tier3.py)
# ------------------------------------------------------------------
from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.powertrain_model import PowertrainModel
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine

print("=" * 72)
print("RPM & ELECTRICAL POWER DIAGNOSTIC: SIM vs TELEMETRY")
print("=" * 72)

# Load config and data
config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
_, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
voltt_df = load_voltt_csv(
    "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
)
track = Track.from_telemetry(df=aim_df)

# Battery
battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
battery.calibrate(voltt_df)
battery.calibrate_pack_from_telemetry(aim_df)

# Replay sim
initial_soc = float(aim_df["State of Charge"].iloc[0])
initial_temp = float(aim_df["Pack Temp"].iloc[0])
initial_speed = float(aim_df["GPS Speed"].iloc[0]) / 3.6
total_distance = aim_df["Distance on GPS Speed"].iloc[-1]
num_laps = round(total_distance / track.total_distance_m)

replay = ReplayStrategy.from_full_endurance(aim_df, track.lap_distance_m)

batt_replay = BatteryModel(config.battery, cell_capacity_ah=4.5)
batt_replay.calibrate(voltt_df)
batt_replay.calibrate_pack_from_telemetry(aim_df)

engine = SimulationEngine(config, track, replay, batt_replay)
print("Running replay simulation...")
result = engine.run(
    num_laps=num_laps,
    initial_soc_pct=initial_soc,
    initial_temp_c=initial_temp,
    initial_speed_ms=max(initial_speed, 0.5),
)
sim_df = result.states

print(f"Sim total time: {result.total_time_s:.1f} s")
print(f"Sim total energy: {result.total_energy_kwh:.3f} kWh")
print(f"Sim laps: {result.laps_completed}")

# Telemetry total time (moving samples, GPS Speed > 5 km/h)
telem_moving = aim_df[aim_df["GPS Speed"] > 5.0]
telem_total_time = float(telem_moving["Time"].iloc[-1] - telem_moving["Time"].iloc[0])
print(f"Telemetry total time (moving): {telem_total_time:.1f} s")
print(f"Time difference: {result.total_time_s - telem_total_time:.1f} s (negative = sim is faster)")

# ==================================================================
# SECTION 1: GEAR RATIO AND TIRE RADIUS CONSTANTS
# ==================================================================
print("\n" + "=" * 72)
print("1. GEAR RATIO AND TIRE RADIUS CONSTANTS")
print("=" * 72)

gear_ratio = config.powertrain.gear_ratio
tire_radius = PowertrainModel.TIRE_RADIUS_M

print(f"  Sim gear_ratio:      {gear_ratio}")
print(f"  Expected (40/11):    {40/11:.6f}")
print(f"  Match:               {abs(gear_ratio - 40/11) < 0.001}")
print(f"  Sim TIRE_RADIUS_M:   {tire_radius:.4f} m")
print(f"  TIR file (10psi):    0.2042 m (UNLOADED_RADIUS)")
print(f"  Match:               {abs(tire_radius - 0.2042) < 0.001}")

# Dynamics module tire radius
print(f"  dynamics.py radius:  0.2042 m (hardcoded for m_effective)")
print(f"  Older scripts use:   0.228 m  (WRONG -- that is 16in diameter / 2, nominal not actual)")

# What max speed does the sim predict at 2900 RPM?
max_speed_sim = tire_radius * 2 * math.pi * 2900 / (60 * gear_ratio)
print(f"\n  Max vehicle speed at 2900 RPM (sim):  {max_speed_sim:.2f} m/s = {max_speed_sim*3.6:.1f} km/h")

# ==================================================================
# SECTION 2: RPM VALIDATION -- TELEMETRY vs PREDICTED
# ==================================================================
print("\n" + "=" * 72)
print("2. RPM VALIDATION: TELEMETRY Motor RPM vs PREDICTED from LFspeed")
print("=" * 72)

# Filter to moving samples
moving_mask = aim_df["GPS Speed"] > 5.0
telem = aim_df[moving_mask].copy()

# LFspeed is in km/h (front wheel speed sensor)
telem_speed_ms = telem["LFspeed"].values / 3.6
telem_motor_rpm = telem["Motor RPM"].values

# Predicted RPM from speed using sim's constants
# motor_rpm = speed_ms / tire_radius * 60 / (2*pi) * gear_ratio
predicted_rpm = telem_speed_ms * gear_ratio * 60.0 / (2.0 * math.pi * tire_radius)

# Also try with 0.228 radius (what old scripts used)
predicted_rpm_228 = telem_speed_ms * gear_ratio * 60.0 / (2.0 * math.pi * 0.228)

# Filter out very low RPM (noise) and NaN
valid = (telem_motor_rpm > 100) & (telem_speed_ms > 1.0) & np.isfinite(telem_motor_rpm)
rpm_actual = telem_motor_rpm[valid]
rpm_pred = predicted_rpm[valid]
rpm_pred_228 = predicted_rpm_228[valid]
speed_valid = telem_speed_ms[valid]

rpm_error = rpm_pred - rpm_actual
rpm_error_228 = rpm_pred_228 - rpm_actual
rpm_pct_error = (rpm_error / rpm_actual) * 100
rpm_pct_error_228 = (rpm_error_228 / rpm_actual) * 100

print(f"\n  With TIRE_RADIUS = {tire_radius:.4f} m (sim value, from TIR file):")
print(f"    Mean RPM error:      {np.mean(rpm_error):.1f} RPM ({np.mean(rpm_pct_error):.2f}%)")
print(f"    Median RPM error:    {np.median(rpm_error):.1f} RPM ({np.median(rpm_pct_error):.2f}%)")
print(f"    Std RPM error:       {np.std(rpm_error):.1f} RPM")
print(f"    Max abs RPM error:   {np.max(np.abs(rpm_error)):.0f} RPM")
print(f"    RMSE:                {np.sqrt(np.mean(rpm_error**2)):.1f} RPM")

print(f"\n  With TIRE_RADIUS = 0.228 m (old scripts, WRONG):")
print(f"    Mean RPM error:      {np.mean(rpm_error_228):.1f} RPM ({np.mean(rpm_pct_error_228):.2f}%)")
print(f"    Median RPM error:    {np.median(rpm_error_228):.1f} RPM ({np.median(rpm_pct_error_228):.2f}%)")

# Back-derive effective tire radius from telemetry
# tire_radius_eff = speed_ms * gear_ratio * 60 / (2*pi * motor_rpm)
r_eff = speed_valid * gear_ratio * 60.0 / (2.0 * math.pi * rpm_actual)
print(f"\n  Back-derived effective tire radius from telemetry:")
print(f"    Mean:    {np.mean(r_eff):.4f} m")
print(f"    Median:  {np.median(r_eff):.4f} m")
print(f"    Std:     {np.std(r_eff):.4f} m")
print(f"    Min:     {np.min(r_eff):.4f} m")
print(f"    Max:     {np.max(r_eff):.4f} m")
print(f"    Sim uses: {tire_radius:.4f} m")

# Speed-binned RPM comparison
print(f"\n  Speed-binned RPM comparison (r={tire_radius:.4f} m):")
print(f"  {'Speed bin':>15s}  {'Actual RPM':>12s}  {'Predicted':>12s}  {'Error':>8s}  {'Error%':>8s}")
for lo, hi in [(5, 15), (15, 25), (25, 35), (35, 45), (45, 60), (60, 80)]:
    mask = (speed_valid * 3.6 >= lo) & (speed_valid * 3.6 < hi)
    if mask.sum() < 10:
        continue
    act_mean = np.mean(rpm_actual[mask])
    pred_mean = np.mean(rpm_pred[mask])
    err = pred_mean - act_mean
    pct = err / act_mean * 100
    print(f"  {lo:>3d}-{hi:<3d} km/h    {act_mean:>10.0f}    {pred_mean:>10.0f}  {err:>+7.0f}  {pct:>+7.2f}%")

# ==================================================================
# SECTION 3: SIM RPM vs TELEMETRY RPM (at matched distances)
# ==================================================================
print("\n" + "=" * 72)
print("3. SIM MOTOR RPM vs TELEMETRY MOTOR RPM (distance-matched)")
print("=" * 72)

# Interpolate telemetry Motor RPM to sim distance points
telem_dist = telem["Distance on GPS Speed"].values
telem_rpm_vals = telem["Motor RPM"].values

from scipy.interpolate import interp1d

# Valid range for interpolation
dist_min = max(sim_df["distance_m"].min(), telem_dist.min())
dist_max = min(sim_df["distance_m"].max(), telem_dist.max())

# Build interpolators
rpm_interp = interp1d(telem_dist, telem_rpm_vals, kind="linear",
                      bounds_error=False, fill_value=np.nan)

# Sample at sim distance points
sim_dist = sim_df["distance_m"].values
sim_rpm = sim_df["motor_rpm"].values
telem_rpm_at_sim = rpm_interp(sim_dist)

# Filter valid
valid_mask = np.isfinite(telem_rpm_at_sim) & (sim_dist >= dist_min) & (sim_dist <= dist_max)
sim_rpm_v = sim_rpm[valid_mask]
telem_rpm_v = telem_rpm_at_sim[valid_mask]
sim_dist_v = sim_dist[valid_mask]

rpm_diff = sim_rpm_v - telem_rpm_v
rpm_diff_pct = np.where(telem_rpm_v > 50, rpm_diff / telem_rpm_v * 100, 0)

print(f"  Matched points: {valid_mask.sum()}")
print(f"  Mean sim RPM:      {np.mean(sim_rpm_v):.0f}")
print(f"  Mean telem RPM:    {np.mean(telem_rpm_v):.0f}")
print(f"  Mean RPM diff:     {np.mean(rpm_diff):.0f} ({np.mean(rpm_diff_pct):.2f}%)")
print(f"  Median RPM diff:   {np.median(rpm_diff):.0f}")
print(f"  Max abs RPM diff:  {np.max(np.abs(rpm_diff)):.0f}")

# ==================================================================
# SECTION 4: ELECTRICAL POWER COMPARISON
# ==================================================================
print("\n" + "=" * 72)
print("4. ELECTRICAL POWER: SIM vs TELEMETRY")
print("=" * 72)

# Telemetry power = Pack Voltage * Pack Current
telem["Elec_Power_W"] = telem["Pack Voltage"].values * telem["Pack Current"].values

# Interpolate telemetry power to sim distance points
power_interp = interp1d(telem_dist, telem["Elec_Power_W"].values, kind="linear",
                        bounds_error=False, fill_value=np.nan)

telem_power_at_sim = power_interp(sim_dist)
sim_power = sim_df["electrical_power_w"].values

valid_p = np.isfinite(telem_power_at_sim) & (sim_dist >= dist_min) & (sim_dist <= dist_max)
sim_pow_v = sim_power[valid_p]
telem_pow_v = telem_power_at_sim[valid_p]
sim_dist_p = sim_dist[valid_p]

power_diff = sim_pow_v - telem_pow_v

print(f"  Matched points: {valid_p.sum()}")
print(f"  Mean sim power:     {np.mean(sim_pow_v):.0f} W")
print(f"  Mean telem power:   {np.mean(telem_pow_v):.0f} W")
print(f"  Mean power diff:    {np.mean(power_diff):.0f} W ({np.mean(power_diff)/max(1,abs(np.mean(telem_pow_v)))*100:.1f}%)")
print(f"  Median power diff:  {np.median(power_diff):.0f} W")
print(f"  Max power diff:     {np.max(power_diff):.0f} W")
print(f"  Min power diff:     {np.min(power_diff):.0f} W")
print(f"  Std power diff:     {np.std(power_diff):.0f} W")

# Power comparison in speed bins
print(f"\n  Speed-binned power comparison:")
sim_speed = sim_df["speed_ms"].values[valid_p]
print(f"  {'Speed bin':>15s}  {'Sim Power':>12s}  {'Telem Power':>12s}  {'Diff':>10s}  {'Diff%':>8s}")
for lo, hi in [(5, 15), (15, 25), (25, 35), (35, 45), (45, 60), (60, 80)]:
    mask = (sim_speed * 3.6 >= lo) & (sim_speed * 3.6 < hi)
    if mask.sum() < 10:
        continue
    sp = np.mean(sim_pow_v[mask])
    tp = np.mean(telem_pow_v[mask])
    diff = sp - tp
    pct = diff / max(1, abs(tp)) * 100
    print(f"  {lo:>3d}-{hi:<3d} km/h    {sp:>10.0f} W  {tp:>10.0f} W  {diff:>+9.0f} W  {pct:>+7.1f}%")

# Worst operating points
print(f"\n  Top 10 largest power discrepancy points:")
abs_diff = np.abs(power_diff)
worst_idx = np.argsort(abs_diff)[-10:][::-1]
sim_torque = sim_df["motor_torque_nm"].values[valid_p]
sim_action = sim_df["action"].values[valid_p]
print(f"  {'Dist(m)':>8s}  {'SimPow(W)':>10s}  {'TelemPow(W)':>12s}  {'Diff(W)':>10s}  {'Speed(km/h)':>12s}  {'Torque(Nm)':>11s}  {'Action':>8s}")
for i in worst_idx:
    print(f"  {sim_dist_p[i]:>8.0f}  {sim_pow_v[i]:>10.0f}  {telem_pow_v[i]:>12.0f}  {power_diff[i]:>+10.0f}  {sim_speed[i]*3.6:>12.1f}  {sim_torque[i]:>11.1f}  {sim_action[i]:>8s}")

# ==================================================================
# SECTION 5: MOTOR DRAG AT ZERO THROTTLE
# ==================================================================
print("\n" + "=" * 72)
print("5. MOTOR DRAG AT ZERO THROTTLE (Iron/Windage Losses)")
print("=" * 72)

# In telemetry: when throttle is ~0, what does Pack Current show?
telem_full = aim_df[aim_df["GPS Speed"] > 10.0].copy()
telem_full["Throttle_frac"] = telem_full["Throttle Pos"] / 100.0

# Zero throttle: Throttle Pos < 5%
coast_mask = (telem_full["Throttle Pos"] < 5.0) & (telem_full["GPS Speed"] > 10.0)
coast_data = telem_full[coast_mask]

if len(coast_data) > 0:
    coast_current = coast_data["Pack Current"].values
    coast_voltage = coast_data["Pack Voltage"].values
    coast_power = coast_voltage * coast_current
    coast_rpm = coast_data["Motor RPM"].values
    coast_speed = coast_data["GPS Speed"].values
    coast_torque_fb = coast_data["Torque Feedback"].values
    coast_lvcu = coast_data["LVCU Torque Req"].values
    coast_mcu_dc = coast_data["MCU DC Current"].values

    print(f"  Telemetry coasting samples (throttle < 5%): {len(coast_data)}")
    print(f"  Mean Pack Current (coasting):  {np.mean(coast_current):.2f} A")
    print(f"  Mean Pack Power (coasting):    {np.mean(coast_power):.0f} W")
    print(f"  Mean Motor RPM (coasting):     {np.mean(coast_rpm):.0f} RPM")
    print(f"  Mean Speed (coasting):         {np.mean(coast_speed):.1f} km/h")
    print(f"  Mean Torque Feedback:          {np.mean(coast_torque_fb):.2f} Nm")
    print(f"  Mean LVCU Torque Req:          {np.mean(coast_lvcu):.2f} Nm")
    print(f"  Mean MCU DC Current:           {np.mean(coast_mcu_dc):.2f} A")

    # Is there measurable power draw with zero throttle command?
    zero_cmd = coast_data[coast_data["LVCU Torque Req"] < 1.0]
    if len(zero_cmd) > 10:
        print(f"\n  Samples with LVCU Torque Req < 1 Nm AND throttle < 5%: {len(zero_cmd)}")
        print(f"    Mean Pack Current:  {np.mean(zero_cmd['Pack Current'].values):.2f} A")
        print(f"    Mean Pack Power:    {np.mean(zero_cmd['Pack Voltage'].values * zero_cmd['Pack Current'].values):.0f} W")
        print(f"    Mean MCU DC Current: {np.mean(zero_cmd['MCU DC Current'].values):.2f} A")
        print(f"    Mean Torque Feedback: {np.mean(zero_cmd['Torque Feedback'].values):.2f} Nm")
        print(f"    Mean Motor RPM:     {np.mean(zero_cmd['Motor RPM'].values):.0f} RPM")
        print(f"    --> This is the motor's parasitic draw (iron loss + windage + inverter standby)")
    else:
        print(f"  Only {len(zero_cmd)} samples with LVCU Torque Req < 1 Nm")

    # Sim behavior during coast: does it produce any motor drag?
    print(f"\n  Sim behavior during COAST action:")
    coast_sim = sim_df[sim_df["action"] == "coast"]
    if len(coast_sim) > 0:
        print(f"    COAST segments:          {len(coast_sim)}")
        print(f"    Mean motor_torque_nm:    {coast_sim['motor_torque_nm'].mean():.2f} Nm")
        print(f"    Mean electrical_power_w: {coast_sim['electrical_power_w'].mean():.0f} W")
        print(f"    Mean drive_force_n:      {coast_sim['drive_force_n'].mean():.1f} N")
        print(f"    Mean regen_force_n:      {coast_sim['regen_force_n'].mean():.1f} N")
        print(f"    --> If all zeros, sim has NO motor drag during coast")
        print(f"    --> Real car still draws power for inverter standby and motor iron losses")
    else:
        print(f"    No COAST segments found in sim!")

    # Check phase currents during coast for true motor drag
    phase_cols = ["Phase A Current", "Phase B Current", "Phase C Current"]
    if all(c in coast_data.columns for c in phase_cols):
        phase_a = coast_data["Phase A Current"].values
        phase_b = coast_data["Phase B Current"].values
        phase_c = coast_data["Phase C Current"].values
        rms_phase = np.sqrt((phase_a**2 + phase_b**2 + phase_c**2) / 3)
        print(f"\n  Phase current during coast (throttle < 5%):")
        print(f"    RMS phase current:  {np.mean(rms_phase):.1f} A")
        print(f"    --> Non-zero phase current = motor iron losses (the motor is a generator being dragged)")
else:
    print("  No coasting samples found!")

# ==================================================================
# SECTION 6: TIME ACCUMULATION ANALYSIS
# ==================================================================
print("\n" + "=" * 72)
print("6. WHERE IS THE 48-SECOND GAP ACCUMULATING?")
print("=" * 72)

# Sim: cumulative time vs distance
# Telemetry: Time vs Distance on GPS Speed
# Compare how time accumulates vs distance

telem_time_interp = interp1d(telem_dist, telem["Time"].values, kind="linear",
                             bounds_error=False, fill_value=np.nan)

sim_cum_time = sim_df["time_s"].values
telem_time_at_sim = telem_time_interp(sim_dist)

valid_t = np.isfinite(telem_time_at_sim) & (sim_dist >= dist_min) & (sim_dist <= dist_max)
time_diff = sim_cum_time[valid_t] - telem_time_at_sim[valid_t]

print(f"  Time difference vs distance (sim - telemetry):")
dist_checkpoints = [1000, 2000, 5000, 8000, 10000, 12000, 15000, 18000, 20000]
for cp in dist_checkpoints:
    idx = np.argmin(np.abs(sim_dist[valid_t] - cp))
    if abs(sim_dist[valid_t][idx] - cp) < 500:
        print(f"    At {cp:>6d} m:  sim is {time_diff[idx]:>+6.1f} s vs telemetry")

# Where does time gap grow fastest?
print(f"\n  Time gap rate of change (1000m windows):")
for start in range(0, 19000, 2000):
    end = start + 2000
    mask_window = (sim_dist[valid_t] >= start) & (sim_dist[valid_t] < end)
    if mask_window.sum() < 5:
        continue
    td = time_diff[mask_window]
    gap_change = td[-1] - td[0]
    print(f"    {start:>6d} - {end:<6d} m:  gap change = {gap_change:>+5.1f} s")

# ==================================================================
# SECTION 7: SPEED COMPARISON
# ==================================================================
print("\n" + "=" * 72)
print("7. SPEED COMPARISON: SIM vs TELEMETRY (at matched distances)")
print("=" * 72)

speed_telem_interp = interp1d(telem_dist, telem["LFspeed"].values / 3.6, kind="linear",
                              bounds_error=False, fill_value=np.nan)
telem_speed_at_sim = speed_telem_interp(sim_dist)

valid_s = np.isfinite(telem_speed_at_sim) & (sim_dist >= dist_min) & (sim_dist <= dist_max)
sim_spd = sim_df["speed_ms"].values[valid_s]
telem_spd = telem_speed_at_sim[valid_s]
speed_diff = sim_spd - telem_spd

print(f"  Mean sim speed:    {np.mean(sim_spd)*3.6:.1f} km/h")
print(f"  Mean telem speed:  {np.mean(telem_spd)*3.6:.1f} km/h")
print(f"  Mean speed diff:   {np.mean(speed_diff)*3.6:.2f} km/h ({np.mean(speed_diff)/np.mean(telem_spd)*100:.2f}%)")
print(f"  Max speed diff:    {np.max(speed_diff)*3.6:.1f} km/h")
print(f"  Min speed diff:    {np.min(speed_diff)*3.6:.1f} km/h")

# Break down by action type
for action in ["throttle", "coast", "brake"]:
    act_mask = sim_df["action"].values[valid_s] == action
    if act_mask.sum() < 5:
        continue
    sd = speed_diff[act_mask]
    print(f"  During {action:>8s}:  mean speed diff = {np.mean(sd)*3.6:>+.2f} km/h  ({act_mask.sum()} segments)")

# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "=" * 72)
print("SUMMARY OF FINDINGS")
print("=" * 72)

print(f"\n  GEAR RATIO: {gear_ratio} (correct, matches 40/11 = {40/11:.4f})")
print(f"  TIRE RADIUS: {tire_radius:.4f} m (matches TIR file UNLOADED_RADIUS)")
r_eff_mean = np.mean(r_eff)
r_eff_pct_diff = (r_eff_mean - tire_radius) / tire_radius * 100
print(f"  Back-derived effective radius from telemetry: {r_eff_mean:.4f} m ({r_eff_pct_diff:+.1f}% vs sim)")

# RPM verdict
mean_rpm_err_pct = np.mean(rpm_pct_error)
print(f"\n  RPM ACCURACY: predicted vs telemetry error = {mean_rpm_err_pct:+.2f}%")
if abs(mean_rpm_err_pct) < 2:
    print(f"    --> Gear ratio and tire radius are CORRECT (< 2% error)")
else:
    print(f"    --> SIGNIFICANT RPM error: gear ratio or tire radius may be wrong!")

# Power verdict
mean_power_diff = np.mean(power_diff)
print(f"\n  POWER: sim vs telemetry mean diff = {mean_power_diff:+.0f} W")

# Motor drag verdict
if len(coast_sim) > 0 and coast_sim['electrical_power_w'].mean() == 0:
    print(f"\n  MOTOR DRAG: sim produces ZERO power during coast")
    print(f"    Real car draws ~{abs(np.mean(coast_power)):.0f} W during coast (iron losses + standby)")
    print(f"    This means sim UNDERESTIMATES resistance during coasting")
    print(f"    Missing motor drag = the motor provides LESS retardation than reality")
    print(f"    --> This would make the sim FASTER, not slower, during coast phases")
    print(f"    --> But missing motor electrical losses mean energy accounting is wrong")

print(f"\n  SPEED BIAS: sim is {np.mean(speed_diff)*3.6:+.2f} km/h faster on average")
print(f"    If sim is systematically faster, it covers distance in less time")
print(f"    For {total_distance:.0f} m at {np.mean(telem_spd)*3.6:.1f} km/h avg:")
print(f"    A {abs(np.mean(speed_diff)*3.6):.2f} km/h speed bias produces ~{abs(np.mean(speed_diff)/np.mean(telem_spd) * telem_total_time):.0f} s time difference")

print("\nDone.")
