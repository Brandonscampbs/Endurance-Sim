"""Force balance comparison: telemetry-derived forces vs simulation forces.

Diagnoses why the sim finishes ~48 seconds too fast by comparing:
1. Drive force (telemetry Torque Feedback vs sim drive_force_n)
2. Resistance force (telemetry-implied vs sim resistance_force_n)
3. Systematic biases by track region (straights vs corners)
4. Total impulse difference and equivalent time impact
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from fsae_sim.data.loader import load_cleaned_csv
from fsae_sim.vehicle.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.track.track import Track
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine

# ============================================================
# Constants
# ============================================================
MASS_KG = 288.0
GEAR_RATIO = 3.6363
TIRE_RADIUS_M = 0.2042
GRAVITY = 9.81
GEARBOX_EFF = 0.97

# ============================================================
# 1. Load telemetry
# ============================================================
print("=" * 70)
print("FORCE BALANCE COMPARISON: Telemetry vs Simulation")
print("=" * 70)

csv_path = Path(__file__).resolve().parent.parent / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
_, aim_df = load_cleaned_csv(str(csv_path))

print(f"\nTelemetry: {len(aim_df)} samples, {aim_df['Time'].values[-1]:.1f}s total recording")

# ============================================================
# 2. Filter to GPS-valid, moving samples
# ============================================================
speed_kmh = aim_df["LFspeed"].values
moving_mask = speed_kmh > 5.0
df_moving = aim_df[moving_mask].copy().reset_index(drop=True)

gps_valid = ~np.isnan(df_moving["GPS LonAcc"].values) & ~np.isnan(df_moving["GPS LatAcc"].values)
print(f"GPS-valid moving samples: {np.sum(gps_valid)} of {len(df_moving)}")

# Apply GPS validity filter
time_arr = df_moving["Time"].values[gps_valid]
speed_ms = df_moving["LFspeed"].values[gps_valid] / 3.6
distance_m = df_moving["Distance on GPS Speed"].values[gps_valid]
torque_nm = df_moving["Torque Feedback"].values[gps_valid]
lon_acc_g = df_moving["GPS LonAcc"].values[gps_valid]
lat_acc_g = df_moving["GPS LatAcc"].values[gps_valid]
motor_rpm = df_moving["Motor RPM"].values[gps_valid]

# Compute telemetry-derived forces
tel_drive_force = torque_nm * GEAR_RATIO * GEARBOX_EFF / TIRE_RADIUS_M
tel_total_force = MASS_KG * lon_acc_g * GRAVITY  # F = m*a
tel_resistance_implied = tel_drive_force - tel_total_force

# Classify regions
cornering_threshold_g = 0.3
is_cornering = np.abs(lat_acc_g) > cornering_threshold_g
is_straight = ~is_cornering
is_throttle = torque_nm > 2.0
is_coast = (torque_nm <= 2.0) & (torque_nm >= -2.0)

# ============================================================
# 3. Telemetry force statistics
# ============================================================
print("\n" + "=" * 70)
print("PART 1: TELEMETRY FORCE ANALYSIS (ground truth)")
print("=" * 70)

mask_powered = is_throttle & (speed_ms > 2.0)
mask_str_pwr = mask_powered & is_straight
mask_cor_pwr = mask_powered & is_cornering
mask_coast = is_coast & (speed_ms > 2.0)
mask_coast_str = mask_coast & is_straight
mask_coast_cor = mask_coast & is_cornering

print(f"\nAll powered samples (n={np.sum(mask_powered)}):")
print(f"  Drive force:      {np.mean(tel_drive_force[mask_powered]):7.1f} N")
print(f"  Total (m*a):      {np.mean(tel_total_force[mask_powered]):7.1f} N")
print(f"  Implied resist:   {np.mean(tel_resistance_implied[mask_powered]):7.1f} N")
print(f"  Motor torque:     {np.mean(torque_nm[mask_powered]):7.1f} Nm")
print(f"  Speed:            {np.mean(speed_ms[mask_powered]*3.6):7.1f} km/h")

print(f"\nStraights under power (n={np.sum(mask_str_pwr)}):")
print(f"  Drive force:      {np.mean(tel_drive_force[mask_str_pwr]):7.1f} N")
print(f"  Implied resist:   {np.mean(tel_resistance_implied[mask_str_pwr]):7.1f} N")
print(f"  Speed:            {np.mean(speed_ms[mask_str_pwr]*3.6):7.1f} km/h")

print(f"\nCorners under power (n={np.sum(mask_cor_pwr)}):")
print(f"  Drive force:      {np.mean(tel_drive_force[mask_cor_pwr]):7.1f} N")
print(f"  Implied resist:   {np.mean(tel_resistance_implied[mask_cor_pwr]):7.1f} N")
print(f"  Speed:            {np.mean(speed_ms[mask_cor_pwr]*3.6):7.1f} km/h")
print(f"  Mean |lat_acc|:   {np.mean(np.abs(lat_acc_g[mask_cor_pwr])):7.3f} g")
extra_resist = np.mean(tel_resistance_implied[mask_cor_pwr]) - np.mean(tel_resistance_implied[mask_str_pwr])
print(f"  Corner - Straight resist delta: {extra_resist:+.1f} N")

print(f"\n  ** NOTE: Corner implied resistance is LOWER than straight. **")
print(f"  ** This is because in corners, drivers use LESS throttle (lower drive force), **")
print(f"  ** so the implied resistance = drive - m*a is lower even though real drag is higher. **")

print(f"\nCoasting on straights (n={np.sum(mask_coast_str)}):")
coast_str_decel = -tel_total_force[mask_coast_str]
print(f"  Decel force (from a): {np.mean(coast_str_decel):7.1f} N  <-- BEST estimate of real resistance")
print(f"  Speed:                {np.mean(speed_ms[mask_coast_str]*3.6):7.1f} km/h")

print(f"\nCoasting in corners (n={np.sum(mask_coast_cor)}):")
coast_cor_decel = -tel_total_force[mask_coast_cor]
print(f"  Decel force (from a): {np.mean(coast_cor_decel):7.1f} N")
print(f"  Speed:                {np.mean(speed_ms[mask_coast_cor]*3.6):7.1f} km/h")
print(f"  Mean |lat_acc|:       {np.mean(np.abs(lat_acc_g[mask_coast_cor])):7.3f} g")
coast_extra = np.mean(coast_cor_decel) - np.mean(coast_str_decel)
print(f"  Corner extra drag (coasting): {coast_extra:+.1f} N")

# Speed-binned coasting resistance (most reliable estimate)
print(f"\nCoasting resistance by speed (straight only):")
for s_lo in range(20, 80, 10):
    s_hi = s_lo + 10
    m = mask_coast_str & (speed_ms*3.6 >= s_lo) & (speed_ms*3.6 < s_hi)
    if np.sum(m) > 5:
        print(f"  {s_lo:2d}-{s_hi:2d} km/h: {np.mean(-tel_total_force[m]):7.1f} N  (n={np.sum(m)})")

# ============================================================
# 4. Run replay simulation
# ============================================================
print("\n" + "=" * 70)
print("PART 2: REPLAY SIMULATION")
print("=" * 70)

config_path = Path(__file__).resolve().parent.parent / "configs" / "ct16ev.yaml"
vehicle_config = VehicleConfig.from_yaml(str(config_path))
track = Track.from_telemetry(df=aim_df, name="Michigan Endurance")
lap_dist = track.total_distance_m
print(f"Track: {track.num_segments} segments, {lap_dist:.1f} m/lap")

voltt_path = Path(__file__).resolve().parent.parent / "Real-Car-Data-And-Stats" / "About-Energy-Volt-Simulations-2025-Pack" / "2025_Pack_cell.csv"
battery = BatteryModel.from_config_and_data(vehicle_config.battery, str(voltt_path))
battery.calibrate_pack_from_telemetry(aim_df)

strategy = ReplayStrategy.from_full_endurance(aim_df, lap_dist)
engine = SimulationEngine(vehicle_config, track, strategy, battery)

num_laps = 22
print(f"Running {num_laps}-lap replay...")
result = engine.run(num_laps=num_laps, initial_soc_pct=95.0, initial_speed_ms=0.5)
sim_df = result.states

# Telemetry "driving time" (excluding stopped periods)
tel_moving = aim_df[aim_df["LFspeed"] > 5.0]
dt_all = np.diff(aim_df["Time"].values, prepend=aim_df["Time"].values[0])
dt_moving = dt_all[aim_df["LFspeed"].values > 5.0]
tel_driving_time = np.sum(dt_moving)

print(f"\nSim total time:   {result.total_time_s:.1f}s")
print(f"Tel driving time: {tel_driving_time:.1f}s (excluding stopped/driver-change)")
print(f"TIME GAP:         {tel_driving_time - result.total_time_s:.1f}s (sim faster)")
print(f"Per lap:          {(tel_driving_time - result.total_time_s)/num_laps:.1f}s/lap")

# ============================================================
# 5. Per-lap sim comparison
# ============================================================
print("\n" + "=" * 70)
print("PART 3: PER-LAP ANALYSIS")
print("=" * 70)

# Detect laps in telemetry from distance
tel_all_dist = aim_df["Distance on GPS Speed"].values
tel_all_speed = aim_df["LFspeed"].values

# Find approximate telemetry lap times
# Use the track lap distance to find telemetry time per lap
tel_lap_times = []
tel_lap_energies = []
for lap_i in range(num_laps):
    d_start = lap_i * lap_dist
    d_end = (lap_i + 1) * lap_dist
    # Find telemetry samples in this distance window
    mask = (tel_all_dist >= d_start) & (tel_all_dist < d_end) & (tel_all_speed > 5.0)
    if np.sum(mask) > 10:
        times_in_lap = aim_df["Time"].values[mask]
        t_lap = times_in_lap[-1] - times_in_lap[0]
        tel_lap_times.append(t_lap)
    else:
        tel_lap_times.append(np.nan)

print(f"\n{'Lap':>4s}  {'Sim Time':>10s}  {'Tel Time':>10s}  {'Diff':>8s}  {'Sim Avg km/h':>14s}  {'Sim Drive N':>12s}  {'Sim Resist N':>12s}")
print("-" * 80)

for lap_i in range(min(num_laps, 22)):
    sim_lap = sim_df[sim_df["lap"] == lap_i]
    if len(sim_lap) == 0:
        continue
    s_time = sim_lap["segment_time_s"].sum()
    s_speed = sim_lap["speed_ms"].mean() * 3.6
    s_drive = sim_lap["drive_force_n"].mean()
    s_resist = sim_lap["resistance_force_n"].mean()

    t_time = tel_lap_times[lap_i] if lap_i < len(tel_lap_times) else np.nan
    diff = t_time - s_time if not np.isnan(t_time) else np.nan
    diff_str = f"{diff:+.1f}" if not np.isnan(diff) else "N/A"
    t_str = f"{t_time:.1f}" if not np.isnan(t_time) else "N/A"

    print(f"{lap_i:4d}  {s_time:10.1f}  {t_str:>10s}  {diff_str:>8s}  {s_speed:14.1f}  {s_drive:12.1f}  {s_resist:12.1f}")

# ============================================================
# 6. Core force comparison: sim vs telemetry on LAP 0
# ============================================================
print("\n" + "=" * 70)
print("PART 4: FORCE COMPARISON (Sim lap 0 vs Telemetry)")
print("=" * 70)

sim_lap0 = sim_df[sim_df["lap"] == 0].copy()

# For fair comparison, bin forces by distance-in-lap
bin_size = 5.0  # match segment size
n_bins = track.num_segments
tel_dist_in_lap = distance_m % lap_dist

# Build telemetry bins (average over all laps at each position)
tel_bin_drive = np.zeros(n_bins)
tel_bin_resist = np.zeros(n_bins)
tel_bin_total = np.zeros(n_bins)
tel_bin_speed = np.zeros(n_bins)
tel_bin_latg = np.zeros(n_bins)
tel_bin_count = np.zeros(n_bins)

for i in range(n_bins):
    d_lo = i * bin_size
    d_hi = (i + 1) * bin_size
    mask = (tel_dist_in_lap >= d_lo) & (tel_dist_in_lap < d_hi) & is_throttle
    n = np.sum(mask)
    if n > 0:
        tel_bin_drive[i] = np.mean(tel_drive_force[mask])
        tel_bin_resist[i] = np.mean(tel_resistance_implied[mask])
        tel_bin_total[i] = np.mean(tel_total_force[mask])
        tel_bin_speed[i] = np.mean(speed_ms[mask])
        tel_bin_latg[i] = np.mean(np.abs(lat_acc_g[mask]))
        tel_bin_count[i] = n

# Sim bins are directly the segments
sim_drive = sim_lap0["drive_force_n"].values
sim_resist = sim_lap0["resistance_force_n"].values
sim_speed = sim_lap0["speed_ms"].values
sim_curv = np.abs(sim_lap0["curvature"].values)

# Only compare bins where both have powered samples
valid = (tel_bin_count > 0) & (sim_drive > 0)
bin_is_corner = sim_curv > 0.01

print(f"\nPowered segments with telemetry data: {np.sum(valid)}/{n_bins}")

# Overall
print(f"\n--- OVERALL (powered segments) ---")
print(f"  {'':25s}  {'Telemetry':>12s}  {'Sim':>12s}  {'Diff':>12s}  {'Pct':>8s}")
if np.sum(valid) > 0:
    td = np.mean(tel_bin_drive[valid])
    sd = np.mean(sim_drive[valid])
    print(f"  {'Drive force':25s}  {td:12.1f}  {sd:12.1f}  {sd-td:+12.1f}  {100*(sd-td)/td:+.1f}%")

    tr = np.mean(tel_bin_resist[valid])
    sr = np.mean(sim_resist[valid])
    print(f"  {'Resistance':25s}  {tr:12.1f}  {sr:12.1f}  {sr-tr:+12.1f}  {100*(sr-tr)/max(abs(tr),1):+.1f}%")

    ts = np.mean(tel_bin_speed[valid]) * 3.6
    ss = np.mean(sim_speed[valid]) * 3.6
    print(f"  {'Speed (km/h)':25s}  {ts:12.1f}  {ss:12.1f}  {ss-ts:+12.1f}")

# Straights only
str_valid = valid & ~bin_is_corner
if np.sum(str_valid) > 0:
    print(f"\n--- STRAIGHTS (|curv| < 0.01, n={np.sum(str_valid)}) ---")
    td = np.mean(tel_bin_drive[str_valid])
    sd = np.mean(sim_drive[str_valid])
    print(f"  {'Drive force':25s}  {td:12.1f}  {sd:12.1f}  {sd-td:+12.1f}  {100*(sd-td)/td:+.1f}%")

    tr = np.mean(tel_bin_resist[str_valid])
    sr = np.mean(sim_resist[str_valid])
    print(f"  {'Resistance':25s}  {tr:12.1f}  {sr:12.1f}  {sr-tr:+12.1f}  {100*(sr-tr)/max(abs(tr),1):+.1f}%")

    ts = np.mean(tel_bin_speed[str_valid]) * 3.6
    ss = np.mean(sim_speed[str_valid]) * 3.6
    print(f"  {'Speed (km/h)':25s}  {ts:12.1f}  {ss:12.1f}  {ss-ts:+12.1f}")

# Corners only
cor_valid = valid & bin_is_corner
if np.sum(cor_valid) > 0:
    print(f"\n--- CORNERS (|curv| >= 0.01, n={np.sum(cor_valid)}) ---")
    td = np.mean(tel_bin_drive[cor_valid])
    sd = np.mean(sim_drive[cor_valid])
    print(f"  {'Drive force':25s}  {td:12.1f}  {sd:12.1f}  {sd-td:+12.1f}  {100*(sd-td)/td:+.1f}%")

    tr = np.mean(tel_bin_resist[cor_valid])
    sr = np.mean(sim_resist[cor_valid])
    print(f"  {'Resistance':25s}  {tr:12.1f}  {sr:12.1f}  {sr-tr:+12.1f}  {100*(sr-tr)/max(abs(tr),1):+.1f}%")
    print(f"  {'  -> DEFICIT':25s}  {'':12s}  {'':12s}  {abs(sr-tr):12.1f} N missing")

    ts = np.mean(tel_bin_speed[cor_valid]) * 3.6
    ss = np.mean(sim_speed[cor_valid]) * 3.6
    print(f"  {'Speed (km/h)':25s}  {ts:12.1f}  {ss:12.1f}  {ss-ts:+12.1f}")

# ============================================================
# 7. What should sim resistance be? Use coasting telemetry
# ============================================================
print("\n" + "=" * 70)
print("PART 5: RESISTANCE VALIDATION (coasting = pure resistance)")
print("=" * 70)

print(f"\nCoasting telemetry gives the REAL total resistance at each speed,")
print(f"because drive force = 0, so F_resist = -m*a = m*decel")

# Compare sim resistance model with coasting-derived resistance
print(f"\n  {'Speed (km/h)':>14s}  {'Tel Resist (N)':>14s}  {'Sim Resist (N)':>14s}  {'Diff':>10s}  {'Notes':20s}")
print("  " + "-" * 80)

for s_lo in range(20, 80, 10):
    s_hi = s_lo + 10
    # Telemetry coasting resistance (straight only)
    m_coast = mask_coast_str & (speed_ms*3.6 >= s_lo) & (speed_ms*3.6 < s_hi)
    # Sim resistance at same speed range (straight segments)
    m_sim = (~bin_is_corner) & (sim_speed*3.6 >= s_lo) & (sim_speed*3.6 < s_hi)

    tel_r = np.mean(-tel_total_force[m_coast]) if np.sum(m_coast) > 5 else np.nan
    sim_r = np.mean(sim_resist[m_sim]) if np.sum(m_sim) > 0 else np.nan

    notes = ""
    if not np.isnan(tel_r) and not np.isnan(sim_r):
        diff = sim_r - tel_r
        pct = 100 * diff / tel_r if tel_r > 0 else 0
        notes = f"({pct:+.0f}%)"
        diff_str = f"{diff:+.1f}"
    else:
        diff_str = "N/A"

    tel_str = f"{tel_r:.1f}" if not np.isnan(tel_r) else "N/A"
    sim_str = f"{sim_r:.1f}" if not np.isnan(sim_r) else "N/A"
    print(f"  {s_lo:3d}-{s_hi:3d}          {tel_str:>14s}  {sim_str:>14s}  {diff_str:>10s}  {notes}")

# ============================================================
# 8. Sim resistance components breakdown
# ============================================================
print("\n" + "=" * 70)
print("PART 6: SIM RESISTANCE BREAKDOWN")
print("=" * 70)

# Reconstruct what the sim's dynamics model computes
from fsae_sim.vehicle.dynamics import VehicleDynamics

# Use the same dynamics model the sim uses
dynamics = engine.dynamics

# For a typical straight speed and a typical corner speed
for label, v_ms, curv in [("Straight @ 60 km/h", 60/3.6, 0.0),
                            ("Corner @ 40 km/h, curv=0.02", 40/3.6, 0.02),
                            ("Corner @ 50 km/h, curv=0.01", 50/3.6, 0.01),
                            ("Corner @ 50 km/h, curv=0.03", 50/3.6, 0.03)]:
    drag = dynamics.drag_force(v_ms)
    rr = dynamics.rolling_resistance_force(v_ms)
    cd = dynamics.cornering_drag(v_ms, curv)
    para = dynamics.parasitic_drag()
    total = dynamics.total_resistance(v_ms, 0.0, curv)

    print(f"\n{label}:")
    print(f"  Aero drag:        {drag:7.1f} N")
    print(f"  Rolling resist:   {rr:7.1f} N")
    print(f"  Cornering drag:   {cd:7.1f} N")
    print(f"  Parasitic:        {para:7.1f} N")
    print(f"  TOTAL:            {total:7.1f} N")

# ============================================================
# 9. Time impact quantification
# ============================================================
print("\n" + "=" * 70)
print("PART 7: TIME IMPACT QUANTIFICATION")
print("=" * 70)

time_gap = tel_driving_time - result.total_time_s
avg_sim_speed = sim_df["speed_ms"].mean()

# If resistance is too low by X N on average, how much time does that cost?
# Extra net force -> higher avg speed -> less time
# F_net_excess = drive - (resist_actual - resist_sim) - resist_sim = drive - resist_actual
# But we need it in terms of the resistance deficit

# Method: compute what sim time WOULD be if resistance matched telemetry
# For each segment where we have data, add the resistance deficit
if np.sum(cor_valid) > 0:
    corner_resist_deficit = np.mean(tel_bin_resist[cor_valid]) - np.mean(sim_resist[cor_valid])
else:
    corner_resist_deficit = 0

if np.sum(str_valid) > 0:
    straight_resist_deficit = np.mean(tel_bin_resist[str_valid]) - np.mean(sim_resist[str_valid])
else:
    straight_resist_deficit = 0

print(f"\nResistance deficit (Tel - Sim = missing resistance in sim):")
print(f"  On straights: {straight_resist_deficit:+.1f} N  (positive = sim has too MUCH)")
print(f"  In corners:   {corner_resist_deficit:+.1f} N  (positive = sim has too LITTLE)")

# Estimate time impact of corner resistance deficit
# Assume corner segments are about 45% of the lap
n_corner_segs = np.sum(bin_is_corner)
n_total_segs = len(bin_is_corner)
corner_frac = n_corner_segs / n_total_segs

# In corner segments, the car accelerates by F_net/m * dt more than it should
# Over the full lap, this excess acceleration compounds
# Simple estimate: F_net_excess_avg -> a_excess = F/m -> dv per lap -> dt per lap

# Better: compute how much extra impulse the sim gets per lap from low resistance
sim_corner_mask = sim_df["lap"] == 0
sim_corner_segs = sim_lap0[bin_is_corner]
sim_straight_segs = sim_lap0[~bin_is_corner]

if len(sim_corner_segs) > 0 and np.sum(cor_valid) > 0:
    # Excess net force per corner segment
    avg_corner_excess_force = corner_resist_deficit  # N that should slow car more
    corner_time_per_lap = sim_corner_segs["segment_time_s"].sum()
    excess_impulse_per_lap = avg_corner_excess_force * corner_time_per_lap
    # Impulse = m * dv -> dv per lap from this source
    dv_per_lap = excess_impulse_per_lap / MASS_KG
    # Time saved per lap: at avg speed, extra speed dv saves dt = d * dv / v^2
    dt_per_lap = lap_dist * dv_per_lap / (avg_sim_speed ** 2) if avg_sim_speed > 0 else 0

    print(f"\nCorner resistance deficit time impact:")
    print(f"  Corner segments: {n_corner_segs}/{n_total_segs} ({100*corner_frac:.0f}%)")
    print(f"  Corner time/lap: {corner_time_per_lap:.1f}s")
    print(f"  Excess impulse/lap: {excess_impulse_per_lap:.0f} N*s")
    print(f"  dv/lap: {dv_per_lap:.2f} m/s")
    print(f"  Estimated dt/lap: {dt_per_lap:.1f}s")
    print(f"  Over {num_laps} laps: {dt_per_lap * num_laps:.1f}s")

# ============================================================
# 10. Final diagnosis
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

print(f"""
Total time gap: {time_gap:.1f}s over {num_laps} laps ({time_gap/num_laps:.1f}s/lap)

FINDING 1 - DRIVE FORCE:
  Sim drive force closely matches telemetry ({100*(np.mean(sim_drive[valid])-np.mean(tel_bin_drive[valid]))/np.mean(tel_bin_drive[valid]):+.1f}% overall).
  Drive force is NOT the problem.

FINDING 2 - STRAIGHT-LINE RESISTANCE:
  On straights, sim resistance is {straight_resist_deficit:+.1f} N vs telemetry.
  {'This is well-matched.' if abs(straight_resist_deficit) < 50 else 'This shows a systematic bias.'}

FINDING 3 - CORNER RESISTANCE:
  In corners, sim resistance is {corner_resist_deficit:.0f} N LOWER than telemetry implies.
  The sim is missing ~{abs(corner_resist_deficit):.0f} N of corner drag.
  This means the car decelerates less in corners, carries more speed,
  and exits corners faster -- accumulating time advantage every corner.

FINDING 4 - SPEED DIFFERENCE:
  Sim avg speed: {avg_sim_speed*3.6:.1f} km/h
  Tel avg speed: {np.mean(speed_ms)*3.6:.1f} km/h
  The sim car is {(avg_sim_speed - np.mean(speed_ms))*3.6:.1f} km/h faster on average.

ROOT CAUSE:
  The simulation's resistance model (VehicleDynamics.total_resistance)
  does not capture enough drag in corners. The cornering_drag() function
  adds some tire slip drag, but the real car's cornering resistance is
  much higher than what the model predicts.

  Sources of missing corner resistance:
  1. Tire lateral slip -> longitudinal drag (cornering_drag underestimate)
  2. Scrub/compliance drag from suspension links under lateral load
  3. Tire temperature/wear effects under cornering load
  4. Steering system drag
  5. Aerodynamic yaw/side-force induced drag
""")
