"""Per-segment comparison: simulation vs telemetry.

Loads cleaned AiM telemetry, computes per-5m-segment averages across
all 22 laps, runs the CalibratedStrategy simulation, and produces a
detailed segment-by-segment error analysis.
"""

import sys
import os

# Ensure the project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ======================================================================
# 1. Load cleaned telemetry
# ======================================================================

CSV_PATH = os.path.join(PROJECT_ROOT, "Real-Car-Data-And-Stats", "CleanedEndurance.csv")

print("=" * 80)
print("STEP 1: Loading cleaned telemetry")
print("=" * 80)

from fsae_sim.data.loader import load_cleaned_csv

_, aim_df = load_cleaned_csv(CSV_PATH)
print(f"  Loaded {len(aim_df)} rows, time range: {aim_df['Time'].iloc[0]:.2f} - {aim_df['Time'].iloc[-1]:.2f} s")
print(f"  Columns: {list(aim_df.columns[:15])} ... ({len(aim_df.columns)} total)")

# ======================================================================
# 2. Compute cumulative distance and detect laps
# ======================================================================

print("\n" + "=" * 80)
print("STEP 2: Computing distance and detecting laps")
print("=" * 80)

time_arr = aim_df["Time"].values
speed_kmh = aim_df["LFspeed"].values
speed_ms = speed_kmh / 3.6
dt = np.diff(time_arr, prepend=time_arr[0])
cum_dist = np.cumsum(speed_ms * dt)

print(f"  Total distance: {cum_dist[-1]:.1f} m")
print(f"  Expected: ~22 laps x ~1005 m = ~22110 m")

# Detect lap boundaries using GPS latitude crossing (same method as Track)
from fsae_sim.analysis.validation import detect_lap_boundaries
lap_boundaries = detect_lap_boundaries(aim_df)

print(f"  Detected {len(lap_boundaries)} laps")
for i, (s, e, d) in enumerate(lap_boundaries):
    t_start = aim_df["Time"].iloc[s]
    t_end = aim_df["Time"].iloc[e]
    lap_time = t_end - t_start
    print(f"    Lap {i}: rows {s}-{e}, dist={d:.1f}m, time={lap_time:.1f}s")

# ======================================================================
# 3. Per-segment telemetry averages (5m bins, all laps)
# ======================================================================

print("\n" + "=" * 80)
print("STEP 3: Computing per-segment telemetry means across laps")
print("=" * 80)

SEGMENT_LENGTH = 5.0  # meters
LAP_DISTANCE_M = 1005.0  # approximate, will use detected

# Filter to valid laps (skip short ones, skip outliers)
median_dist = float(np.median([d for _, _, d in lap_boundaries]))
valid_laps = []
for i, (s, e, d) in enumerate(lap_boundaries):
    if abs(d - median_dist) / median_dist < 0.15:
        valid_laps.append((i, s, e, d))

print(f"  Median lap distance: {median_dist:.1f} m")
print(f"  Valid laps (within 15% of median): {len(valid_laps)}")
actual_lap_dist = median_dist
num_segments = int(actual_lap_dist // SEGMENT_LENGTH)
print(f"  Segments per lap: {num_segments} (at {SEGMENT_LENGTH}m each)")

# For each valid lap, compute per-segment values
all_lap_speeds = []        # shape: (n_valid_laps, num_segments)
all_lap_torques = []       # Torque Feedback
all_lap_lvcu_torques = []  # LVCU Torque Req
all_lap_throttles = []     # Throttle Pos (%)
all_lap_currents = []      # Pack Current
all_lap_times = []         # segment time
all_lap_rpms = []          # Motor RPM

for lap_idx, start_idx, end_idx, lap_dist in valid_laps:
    lap_df = aim_df.iloc[start_idx:end_idx].copy()

    # Compute within-lap distance
    lap_cum_dist = cum_dist[start_idx:end_idx]
    lap_d = lap_cum_dist - lap_cum_dist[0]

    lap_speed = lap_df["LFspeed"].values
    lap_torque = lap_df["Torque Feedback"].values
    lap_lvcu = lap_df["LVCU Torque Req"].values
    lap_throttle = lap_df["Throttle Pos"].values
    lap_current = lap_df["Pack Current"].values
    lap_time = lap_df["Time"].values
    lap_rpm = lap_df["Motor RPM"].values

    seg_speeds = np.zeros(num_segments)
    seg_torques = np.zeros(num_segments)
    seg_lvcu = np.zeros(num_segments)
    seg_throttles = np.zeros(num_segments)
    seg_currents = np.zeros(num_segments)
    seg_times = np.zeros(num_segments)
    seg_rpms = np.zeros(num_segments)

    for si in range(num_segments):
        bin_lo = si * SEGMENT_LENGTH
        bin_hi = (si + 1) * SEGMENT_LENGTH
        mask = (lap_d >= bin_lo) & (lap_d < bin_hi)

        if np.sum(mask) > 0:
            seg_speeds[si] = float(np.mean(lap_speed[mask]))
            seg_torques[si] = float(np.mean(lap_torque[mask]))
            seg_lvcu[si] = float(np.mean(lap_lvcu[mask]))
            seg_throttles[si] = float(np.mean(lap_throttle[mask]))
            seg_currents[si] = float(np.mean(lap_current[mask]))
            seg_rpms[si] = float(np.mean(lap_rpm[mask]))
            # Time through segment: time of last sample in bin - time of first
            t_in_bin = lap_time[mask]
            if len(t_in_bin) > 1:
                seg_times[si] = t_in_bin[-1] - t_in_bin[0]
            else:
                # Single sample: estimate from speed
                v = seg_speeds[si] / 3.6
                seg_times[si] = SEGMENT_LENGTH / max(v, 0.5)
        else:
            # No samples in this bin - interpolate from neighbors
            nearest = np.argmin(np.abs(lap_d - (bin_lo + bin_hi) / 2))
            seg_speeds[si] = lap_speed[nearest]
            seg_torques[si] = lap_torque[nearest]
            seg_lvcu[si] = lap_lvcu[nearest]
            seg_throttles[si] = lap_throttle[nearest]
            seg_currents[si] = lap_current[nearest]
            seg_rpms[si] = lap_rpm[nearest]
            v = seg_speeds[si] / 3.6
            seg_times[si] = SEGMENT_LENGTH / max(v, 0.5)

    all_lap_speeds.append(seg_speeds)
    all_lap_torques.append(seg_torques)
    all_lap_lvcu_torques.append(seg_lvcu)
    all_lap_throttles.append(seg_throttles)
    all_lap_currents.append(seg_currents)
    all_lap_times.append(seg_times)
    all_lap_rpms.append(seg_rpms)

# Stack and compute means
speed_matrix = np.array(all_lap_speeds)       # (n_laps, n_segs)
torque_matrix = np.array(all_lap_torques)
lvcu_matrix = np.array(all_lap_lvcu_torques)
throttle_matrix = np.array(all_lap_throttles)
current_matrix = np.array(all_lap_currents)
time_matrix = np.array(all_lap_times)
rpm_matrix = np.array(all_lap_rpms)

telem_mean_speed = np.mean(speed_matrix, axis=0)
telem_mean_torque = np.mean(torque_matrix, axis=0)
telem_mean_lvcu = np.mean(lvcu_matrix, axis=0)
telem_mean_throttle = np.mean(throttle_matrix, axis=0)
telem_mean_current = np.mean(current_matrix, axis=0)
telem_mean_time = np.mean(time_matrix, axis=0)
telem_mean_rpm = np.mean(rpm_matrix, axis=0)

# Also compute p90 speed (what the calibrated strategy uses)
telem_p90_speed = np.percentile(speed_matrix, 90, axis=0)

# Lap-to-lap std for understanding variability
telem_std_speed = np.std(speed_matrix, axis=0)

print(f"  Telemetry mean speed range: {telem_mean_speed.min():.1f} - {telem_mean_speed.max():.1f} km/h")
print(f"  Telemetry mean torque range: {telem_mean_torque.min():.1f} - {telem_mean_torque.max():.1f} Nm")
print(f"  Telemetry mean current range: {telem_mean_current.min():.1f} - {telem_mean_current.max():.1f} A")
print(f"  Telemetry total lap time (mean): {telem_mean_time.sum():.2f} s")

# ======================================================================
# 4. Run the simulation
# ======================================================================

print("\n" + "=" * 80)
print("STEP 4: Running CalibratedStrategy simulation")
print("=" * 80)

from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.track.track import Track
from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimulationEngine

config = VehicleConfig.from_yaml(os.path.join(PROJECT_ROOT, "configs", "ct16ev.yaml"))
voltt_df = pd.read_csv(
    os.path.join(PROJECT_ROOT, "Real-Car-Data-And-Stats",
                 "About-Energy-Volt-Simulations-2025-Pack", "2025_Pack_cell.csv"),
    comment="#",
)
track = Track.from_telemetry(df=aim_df)
strategy = CalibratedStrategy.from_telemetry(aim_df, track)
battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
battery.calibrate(voltt_df)
battery.calibrate_pack_from_telemetry(aim_df)

print(f"  Track: {track.num_segments} segments, {track.total_distance_m:.1f} m total")
print(f"  Strategy: {len(strategy.zones)} zones")

# Print zone summary
print("\n  Zone Summary:")
for z in strategy.zones:
    print(f"    Zone {z.zone_id:2d}: seg {z.segment_start:3d}-{z.segment_end:3d} "
          f"({z.distance_start_m:6.1f}-{z.distance_end_m:6.1f}m) "
          f"{z.action.value:8s} int={z.intensity:.3f} "
          f"max_v={z.max_speed_ms*3.6:.1f}km/h  {z.label}")

# Run simulation
engine = SimulationEngine(config, track, strategy, battery)
result = engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=25.0)

print(f"\n  Simulation completed:")
print(f"    Total time: {result.total_time_s:.2f} s")
print(f"    Total energy: {result.total_energy_kwh:.3f} kWh")
print(f"    Final SOC: {result.final_soc:.2f}%")
print(f"    Laps completed: {result.laps_completed}")

# ======================================================================
# 5. Extract sim per-segment values for a representative lap
# ======================================================================

print("\n" + "=" * 80)
print("STEP 5: Extracting sim per-segment values")
print("=" * 80)

sim_states = result.states

# Use lap 10 as representative mid-race lap (0-indexed)
rep_lap = 10
lap10 = sim_states[sim_states["lap"] == rep_lap].copy()
print(f"  Lap {rep_lap}: {len(lap10)} segments")

# Also compute average across all laps
sim_num_segs = track.num_segments
sim_laps_data = []
for lap_num in range(result.laps_completed):
    lap_data = sim_states[sim_states["lap"] == lap_num]
    if len(lap_data) == sim_num_segs:
        sim_laps_data.append(lap_data)

print(f"  Complete laps for averaging: {len(sim_laps_data)}")

# Build sim per-segment arrays (average across all laps)
sim_speed_matrix = np.array([ld["speed_kmh"].values for ld in sim_laps_data])
sim_torque_matrix = np.array([ld["motor_torque_nm"].values for ld in sim_laps_data])
sim_current_matrix = np.array([ld["pack_current_a"].values for ld in sim_laps_data])
sim_time_matrix = np.array([ld["segment_time_s"].values for ld in sim_laps_data])
sim_action_matrix = np.array([ld["action"].values for ld in sim_laps_data])

sim_mean_speed = np.mean(sim_speed_matrix, axis=0)
sim_mean_torque = np.mean(sim_torque_matrix, axis=0)
sim_mean_current = np.mean(sim_current_matrix, axis=0)
sim_mean_time = np.mean(sim_time_matrix, axis=0)

# Representative lap 10 values
sim_lap10_speed = lap10["speed_kmh"].values
sim_lap10_torque = lap10["motor_torque_nm"].values
sim_lap10_current = lap10["pack_current_a"].values
sim_lap10_time = lap10["segment_time_s"].values
sim_lap10_action = lap10["action"].values

# Ensure arrays are same length (track may have different seg count than telem bins)
n_compare = min(num_segments, sim_num_segs)
print(f"  Telemetry segments: {num_segments}, Sim segments: {sim_num_segs}")
print(f"  Comparing {n_compare} segments")

# Trim arrays to matching length
t_speed = telem_mean_speed[:n_compare]
t_torque = telem_mean_torque[:n_compare]
t_lvcu = telem_mean_lvcu[:n_compare]
t_throttle = telem_mean_throttle[:n_compare]
t_current = telem_mean_current[:n_compare]
t_time = telem_mean_time[:n_compare]
t_p90_speed = telem_p90_speed[:n_compare]
t_std_speed = telem_std_speed[:n_compare]

s_speed = sim_mean_speed[:n_compare]
s_torque = sim_mean_torque[:n_compare]
s_current = sim_mean_current[:n_compare]
s_time = sim_mean_time[:n_compare]

s10_speed = sim_lap10_speed[:n_compare]
s10_torque = sim_lap10_torque[:n_compare]
s10_current = sim_lap10_current[:n_compare]
s10_time = sim_lap10_time[:n_compare]
s10_action = sim_lap10_action[:n_compare]

# Get curvature for each segment
curvature = np.array([track.segments[i].curvature for i in range(n_compare)])

# Get speed targets used by CalibratedStrategy
speed_targets_ms = np.array([strategy.speed_target_ms(i) for i in range(n_compare)])
speed_targets_kmh = speed_targets_ms * 3.6

# ======================================================================
# 6. Compute errors
# ======================================================================

print("\n" + "=" * 80)
print("STEP 6: Per-segment error analysis (sim avg across laps vs telem mean)")
print("=" * 80)

# Speed errors
speed_err_abs = s_speed - t_speed  # km/h
speed_err_pct = np.where(t_speed > 1.0, (speed_err_abs / t_speed) * 100, 0.0)

# Torque errors
torque_err_abs = s_torque - t_torque  # Nm
torque_err_pct = np.where(np.abs(t_torque) > 0.5, (torque_err_abs / t_torque) * 100, 0.0)

# Current errors
current_err_abs = s_current - t_current  # A
current_err_pct = np.where(np.abs(t_current) > 0.5, (current_err_abs / t_current) * 100, 0.0)

# Time errors
time_err_abs = s_time - t_time  # s
time_err_pct = np.where(t_time > 0.01, (time_err_abs / t_time) * 100, 0.0)

# ======================================================================
# 7. Report: Overall summary
# ======================================================================

print("\n" + "-" * 70)
print("OVERALL SPEED ERRORS")
print("-" * 70)
mae_speed = np.mean(np.abs(speed_err_abs))
rms_speed = np.sqrt(np.mean(speed_err_abs**2))
mean_bias_speed = np.mean(speed_err_abs)
within_5pct = np.mean(np.abs(speed_err_pct) <= 5.0) * 100
within_10pct = np.mean(np.abs(speed_err_pct) <= 10.0) * 100
within_2pct = np.mean(np.abs(speed_err_pct) <= 2.0) * 100

print(f"  Mean Absolute Error (MAE):  {mae_speed:.3f} km/h")
print(f"  RMS Error:                  {rms_speed:.3f} km/h")
print(f"  Mean bias (sim - telem):    {mean_bias_speed:+.3f} km/h")
print(f"  Segments within  2% error:  {within_2pct:.1f}%")
print(f"  Segments within  5% error:  {within_5pct:.1f}%")
print(f"  Segments within 10% error:  {within_10pct:.1f}%")

print("\n" + "-" * 70)
print("OVERALL TORQUE ERRORS")
print("-" * 70)
mae_torque = np.mean(np.abs(torque_err_abs))
rms_torque = np.sqrt(np.mean(torque_err_abs**2))
mean_bias_torque = np.mean(torque_err_abs)
print(f"  Mean Absolute Error (MAE):  {mae_torque:.3f} Nm")
print(f"  RMS Error:                  {rms_torque:.3f} Nm")
print(f"  Mean bias (sim - telem):    {mean_bias_torque:+.3f} Nm")

print("\n" + "-" * 70)
print("OVERALL CURRENT ERRORS")
print("-" * 70)
mae_current = np.mean(np.abs(current_err_abs))
rms_current = np.sqrt(np.mean(current_err_abs**2))
mean_bias_current = np.mean(current_err_abs)
print(f"  Mean Absolute Error (MAE):  {mae_current:.3f} A")
print(f"  RMS Error:                  {rms_current:.3f} A")
print(f"  Mean bias (sim - telem):    {mean_bias_current:+.3f} A")

print("\n" + "-" * 70)
print("TIME COMPARISON")
print("-" * 70)
total_telem_time = t_time.sum()
total_sim_time = s_time.sum()
time_diff = total_sim_time - total_telem_time
print(f"  Telemetry total segment time (per lap): {total_telem_time:.3f} s")
print(f"  Simulation total segment time (per lap): {total_sim_time:.3f} s")
print(f"  Difference: {time_diff:+.3f} s ({time_diff/total_telem_time*100:+.2f}%)")
print(f"  Over 22 laps: {time_diff*22:+.2f} s cumulative error")

# ======================================================================
# 8. Top 10 segments by speed error
# ======================================================================

print("\n" + "=" * 80)
print("TOP 10 SEGMENTS BY ABSOLUTE SPEED ERROR")
print("=" * 80)
print(f"  {'Seg':>4s}  {'Dist(m)':>8s}  {'Telem(km/h)':>12s}  {'Sim(km/h)':>10s}  "
      f"{'Err(km/h)':>10s}  {'Err(%)':>7s}  {'Curv(1/m)':>10s}  {'SimAction':>10s}")
print("  " + "-" * 88)

top10_speed = np.argsort(np.abs(speed_err_abs))[::-1][:10]
for idx in top10_speed:
    dist = idx * SEGMENT_LENGTH + SEGMENT_LENGTH / 2
    action_str = s10_action[idx] if idx < len(s10_action) else "?"
    print(f"  {idx:4d}  {dist:8.1f}  {t_speed[idx]:12.2f}  {s_speed[idx]:10.2f}  "
          f"{speed_err_abs[idx]:+10.2f}  {speed_err_pct[idx]:+7.1f}  "
          f"{curvature[idx]:10.4f}  {action_str:>10s}")

# ======================================================================
# 9. Top 10 segments by torque error
# ======================================================================

print("\n" + "=" * 80)
print("TOP 10 SEGMENTS BY ABSOLUTE TORQUE ERROR")
print("=" * 80)
print(f"  {'Seg':>4s}  {'Dist(m)':>8s}  {'Telem(Nm)':>10s}  {'Sim(Nm)':>10s}  "
      f"{'Err(Nm)':>10s}  {'Err(%)':>7s}  {'Curv(1/m)':>10s}  {'SimAction':>10s}")
print("  " + "-" * 82)

top10_torque = np.argsort(np.abs(torque_err_abs))[::-1][:10]
for idx in top10_torque:
    dist = idx * SEGMENT_LENGTH + SEGMENT_LENGTH / 2
    action_str = s10_action[idx] if idx < len(s10_action) else "?"
    print(f"  {idx:4d}  {dist:8.1f}  {t_torque[idx]:10.2f}  {s_torque[idx]:10.2f}  "
          f"{torque_err_abs[idx]:+10.2f}  {torque_err_pct[idx]:+7.1f}  "
          f"{curvature[idx]:10.4f}  {action_str:>10s}")

# ======================================================================
# 10. Speed error vs curvature correlation
# ======================================================================

print("\n" + "=" * 80)
print("SPEED ERROR vs CURVATURE CORRELATION")
print("=" * 80)

abs_curv = np.abs(curvature)
abs_speed_err = np.abs(speed_err_abs)

# Pearson correlation
r_pearson, p_pearson = sp_stats.pearsonr(abs_curv, abs_speed_err)
# Spearman (for monotonic, non-linear)
r_spearman, p_spearman = sp_stats.spearmanr(abs_curv, abs_speed_err)

print(f"  Pearson r:   {r_pearson:+.4f}  (p={p_pearson:.2e})")
print(f"  Spearman rho: {r_spearman:+.4f}  (p={p_spearman:.2e})")

# Break down by curvature bins
curv_bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
print(f"\n  {'Curvature bin':>20s}  {'N segs':>6s}  {'MAE speed':>10s}  {'Mean bias':>10s}  "
      f"{'MAE torque':>10s}")
print("  " + "-" * 66)
for i in range(len(curv_bins) - 1):
    mask = (abs_curv >= curv_bins[i]) & (abs_curv < curv_bins[i+1])
    n = np.sum(mask)
    if n > 0:
        mae_s = np.mean(np.abs(speed_err_abs[mask]))
        bias_s = np.mean(speed_err_abs[mask])
        mae_t = np.mean(np.abs(torque_err_abs[mask]))
        print(f"  [{curv_bins[i]:.3f}, {curv_bins[i+1]:.3f})  {n:6d}  {mae_s:10.3f}  "
              f"{bias_s:+10.3f}  {mae_t:10.3f}")

# ======================================================================
# 11. p90 speed target analysis
# ======================================================================

print("\n" + "=" * 80)
print("P90 SPEED TARGET vs ACTUAL TELEMETRY MEAN")
print("=" * 80)

# The calibrated strategy uses p90 speed across laps as the speed target
p90_vs_mean_diff = t_p90_speed - t_speed  # p90 - mean (should be positive)
p90_vs_mean_pct = np.where(t_speed > 1.0, p90_vs_mean_diff / t_speed * 100, 0.0)

# Strategy speed target vs telemetry mean
target_vs_mean_diff = speed_targets_kmh - t_speed
target_vs_mean_pct = np.where(t_speed > 1.0, target_vs_mean_diff / t_speed * 100, 0.0)

# Strategy speed target vs telemetry p90
target_vs_p90_diff = speed_targets_kmh - t_p90_speed
target_vs_p90_pct = np.where(t_p90_speed > 1.0, target_vs_p90_diff / t_p90_speed * 100, 0.0)

print(f"\n  Telemetry p90 speed vs telemetry mean speed:")
print(f"    Mean difference (p90 - mean): {np.mean(p90_vs_mean_diff):+.3f} km/h")
print(f"    Mean pct difference:          {np.mean(p90_vs_mean_pct):+.2f}%")

print(f"\n  Strategy speed target vs telemetry mean:")
print(f"    Mean difference (target - mean): {np.mean(target_vs_mean_diff):+.3f} km/h")
print(f"    Mean pct difference:             {np.mean(target_vs_mean_pct):+.2f}%")
print(f"    Target higher than mean at {np.sum(target_vs_mean_diff > 0)}/{n_compare} segments")

print(f"\n  Strategy speed target vs telemetry p90:")
print(f"    Mean difference (target - p90): {np.mean(target_vs_p90_diff):+.3f} km/h")
print(f"    Mean pct difference:            {np.mean(target_vs_p90_pct):+.2f}%")
print(f"    Target higher than p90 at {np.sum(target_vs_p90_diff > 0)}/{n_compare} segments")

# Show segments where target is most wrong
print(f"\n  Top 10 segments where sim speed target is ABOVE telemetry p90:")
target_above_p90 = target_vs_p90_diff.copy()
top10_high = np.argsort(target_above_p90)[::-1][:10]
print(f"  {'Seg':>4s}  {'Dist(m)':>8s}  {'Mean(km/h)':>10s}  {'p90(km/h)':>10s}  "
      f"{'Target(km/h)':>12s}  {'Target-p90':>10s}  {'Curv':>8s}")
print("  " + "-" * 76)
for idx in top10_high:
    dist = idx * SEGMENT_LENGTH + SEGMENT_LENGTH / 2
    print(f"  {idx:4d}  {dist:8.1f}  {t_speed[idx]:10.2f}  {t_p90_speed[idx]:10.2f}  "
          f"{speed_targets_kmh[idx]:12.2f}  {target_vs_p90_diff[idx]:+10.2f}  "
          f"{curvature[idx]:8.4f}")

print(f"\n  Top 10 segments where sim speed target is BELOW telemetry p90:")
top10_low = np.argsort(target_above_p90)[:10]
print(f"  {'Seg':>4s}  {'Dist(m)':>8s}  {'Mean(km/h)':>10s}  {'p90(km/h)':>10s}  "
      f"{'Target(km/h)':>12s}  {'Target-p90':>10s}  {'Curv':>8s}")
print("  " + "-" * 76)
for idx in top10_low:
    dist = idx * SEGMENT_LENGTH + SEGMENT_LENGTH / 2
    print(f"  {idx:4d}  {dist:8.1f}  {t_speed[idx]:10.2f}  {t_p90_speed[idx]:10.2f}  "
          f"{speed_targets_kmh[idx]:12.2f}  {target_vs_p90_diff[idx]:+10.2f}  "
          f"{curvature[idx]:8.4f}")

# ======================================================================
# 12. Full per-segment table (every segment)
# ======================================================================

print("\n" + "=" * 80)
print("FULL PER-SEGMENT TABLE (every 5m segment)")
print("=" * 80)
print(f"  {'Seg':>4s}  {'Dist':>6s}  {'T_spd':>6s}  {'S_spd':>6s}  {'dSpd':>6s}  "
      f"{'T_trq':>6s}  {'S_trq':>6s}  {'dTrq':>6s}  "
      f"{'T_cur':>6s}  {'S_cur':>6s}  {'dCur':>6s}  "
      f"{'Curv':>7s}  {'Act':>7s}  {'Tgt':>6s}  {'stdV':>5s}")
print("  " + "-" * 120)

for i in range(n_compare):
    dist = i * SEGMENT_LENGTH + SEGMENT_LENGTH / 2
    action_str = s10_action[i] if i < len(s10_action) else "?"
    print(f"  {i:4d}  {dist:6.1f}  {t_speed[i]:6.1f}  {s_speed[i]:6.1f}  {speed_err_abs[i]:+6.1f}  "
          f"{t_torque[i]:6.1f}  {s_torque[i]:6.1f}  {torque_err_abs[i]:+6.1f}  "
          f"{t_current[i]:6.1f}  {s_current[i]:6.1f}  {current_err_abs[i]:+6.1f}  "
          f"{curvature[i]:7.4f}  {action_str:>7s}  {speed_targets_kmh[i]:6.1f}  {t_std_speed[i]:5.2f}")

# ======================================================================
# 13. Error by driver action type
# ======================================================================

print("\n" + "=" * 80)
print("ERROR BREAKDOWN BY SIM ACTION TYPE (lap 10)")
print("=" * 80)

for action_type in ["throttle", "coast", "brake"]:
    mask = np.array([a == action_type for a in s10_action[:n_compare]])
    n = np.sum(mask)
    if n > 0:
        mae_s = np.mean(np.abs(speed_err_abs[mask]))
        bias_s = np.mean(speed_err_abs[mask])
        mae_t = np.mean(np.abs(torque_err_abs[mask]))
        bias_t = np.mean(torque_err_abs[mask])
        mae_c = np.mean(np.abs(current_err_abs[mask]))
        bias_c = np.mean(current_err_abs[mask])
        print(f"\n  {action_type.upper()} ({n} segments):")
        print(f"    Speed  MAE: {mae_s:.3f} km/h,  bias: {bias_s:+.3f} km/h")
        print(f"    Torque MAE: {mae_t:.3f} Nm,    bias: {bias_t:+.3f} Nm")
        print(f"    Current MAE: {mae_c:.3f} A,    bias: {bias_c:+.3f} A")

# ======================================================================
# 14. Corner speed limit analysis
# ======================================================================

print("\n" + "=" * 80)
print("CORNER SPEED LIMIT vs ACTUAL SPEED")
print("=" * 80)

corner_limits_ms = np.array([
    float(lap10["corner_speed_limit_ms"].values[i]) if i < len(lap10) else float("inf")
    for i in range(n_compare)
])
corner_limits_kmh = corner_limits_ms * 3.6
# Where corner limit is binding (lower than speed target or actual speed)
is_corner = abs_curv > 0.005
corner_segs = np.where(is_corner)[0]

print(f"  Corner segments (curvature > 0.005): {len(corner_segs)}")
if len(corner_segs) > 0:
    print(f"\n  {'Seg':>4s}  {'Dist':>6s}  {'Curv':>7s}  {'CornerLim':>10s}  "
          f"{'TelemSpd':>10s}  {'SimSpd':>10s}  {'SpeedTgt':>10s}")
    print("  " + "-" * 70)
    for idx in corner_segs:
        dist = idx * SEGMENT_LENGTH + SEGMENT_LENGTH / 2
        cl = corner_limits_kmh[idx] if corner_limits_kmh[idx] < 500 else float("inf")
        cl_str = f"{cl:10.1f}" if cl < 500 else "       inf"
        print(f"  {idx:4d}  {dist:6.1f}  {curvature[idx]:7.4f}  {cl_str}  "
              f"{t_speed[idx]:10.1f}  {s_speed[idx]:10.1f}  {speed_targets_kmh[idx]:10.1f}")

# ======================================================================
# 15. Summary statistics and diagnosis
# ======================================================================

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

# Where is sim too fast vs too slow?
sim_too_fast = np.sum(speed_err_abs > 1.0)  # >1 km/h too fast
sim_too_slow = np.sum(speed_err_abs < -1.0)  # >1 km/h too slow
sim_close = np.sum(np.abs(speed_err_abs) <= 1.0)

print(f"\n  Speed accuracy breakdown (threshold = 1 km/h):")
print(f"    Sim too fast by >1 km/h:  {sim_too_fast}/{n_compare} segments ({sim_too_fast/n_compare*100:.1f}%)")
print(f"    Sim too slow by >1 km/h:  {sim_too_slow}/{n_compare} segments ({sim_too_slow/n_compare*100:.1f}%)")
print(f"    Within 1 km/h:            {sim_close}/{n_compare} segments ({sim_close/n_compare*100:.1f}%)")

# Where is torque wrong direction?
torque_sign_match = np.sum(np.sign(s_torque) == np.sign(t_torque))
print(f"\n  Torque sign agreement: {torque_sign_match}/{n_compare} segments ({torque_sign_match/n_compare*100:.1f}%)")

# Energy balance per lap
telem_energy_per_lap = np.sum(t_current * t_time)  # rough: A*s proxy
sim_energy_per_lap = np.sum(s_current * s_time)
print(f"\n  Energy proxy (current * time) per lap:")
print(f"    Telemetry: {telem_energy_per_lap:.1f} A*s")
print(f"    Simulation: {sim_energy_per_lap:.1f} A*s")
print(f"    Ratio: {sim_energy_per_lap/telem_energy_per_lap:.3f}" if telem_energy_per_lap != 0 else "    N/A")

# Sim speed vs target: how well does sim track its own targets?
sim_vs_target = s_speed - speed_targets_kmh
print(f"\n  Sim speed vs its own speed targets:")
print(f"    Mean (sim - target): {np.mean(sim_vs_target):+.3f} km/h")
print(f"    Segments where sim > target: {np.sum(sim_vs_target > 0.1)}/{n_compare}")
print(f"    Segments where sim < target by >2 km/h: {np.sum(sim_vs_target < -2.0)}/{n_compare}")

# Compute lap 10 sim vs telem specifically
print(f"\n  Lap 10 sim vs telem mean:")
s10_err = s10_speed - t_speed
print(f"    Speed MAE: {np.mean(np.abs(s10_err)):.3f} km/h")
print(f"    Speed RMS: {np.sqrt(np.mean(s10_err**2)):.3f} km/h")
print(f"    Speed bias: {np.mean(s10_err):+.3f} km/h")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
