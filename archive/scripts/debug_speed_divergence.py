"""Diagnose WHERE the replay simulation diverges from telemetry.

Compares sim vs telemetry speed profiles at matched distance points,
identifies the worst divergence regions, and traces cumulative time error.
"""

import math
import numpy as np
import pandas as pd

from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.analysis.validation import detect_lap_boundaries


def main():
    # ── Load data ──
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    track = Track.from_telemetry(df=aim_df)

    # Battery setup (same as validate_tier3.py)
    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)
    battery.calibrate_pack_from_telemetry(aim_df)

    # Initial conditions from telemetry
    initial_soc = float(aim_df["State of Charge"].iloc[0])
    initial_temp = float(aim_df["Pack Temp"].iloc[0])
    initial_speed = float(aim_df["GPS Speed"].iloc[0]) / 3.6

    total_distance = aim_df["Distance on GPS Speed"].iloc[-1]
    num_laps = round(total_distance / track.total_distance_m)

    # ── Run replay simulation ──
    replay = ReplayStrategy.from_full_endurance(aim_df, track.lap_distance_m)
    engine = SimulationEngine(config, track, replay, battery)
    result = engine.run(
        num_laps=num_laps,
        initial_soc_pct=initial_soc,
        initial_temp_c=initial_temp,
        initial_speed_ms=max(initial_speed, 0.5),
    )

    sim_states = result.states
    print("=" * 70)
    print("SPEED DIVERGENCE ANALYSIS: Replay Sim vs Telemetry")
    print("=" * 70)
    print(f"  Sim total time:      {result.total_time_s:.2f} s")
    print(f"  Sim laps completed:  {result.laps_completed}")
    print(f"  Sim total distance:  {sim_states['distance_m'].iloc[-1]:.1f} m")

    # ── Build telemetry speed-vs-distance (moving samples only) ──
    # Filter to moving (>5 km/h) to exclude driver change / stopped periods
    moving_mask = aim_df["GPS Speed"].values > 5.0
    telem_moving = aim_df[moving_mask].copy()
    telem_dist = telem_moving["Distance on GPS Speed"].values
    telem_speed_kmh = telem_moving["GPS Speed"].values
    telem_time = telem_moving["Time"].values

    # Compute telemetry driving time
    dt_arr = np.diff(aim_df["Time"].values, prepend=aim_df["Time"].values[0])
    telem_driving_time = float(np.sum(dt_arr[aim_df["GPS Speed"].values > 5.0]))
    print(f"  Telem driving time:  {telem_driving_time:.2f} s")
    print(f"  TIME GAP:            {result.total_time_s - telem_driving_time:.2f} s (sim - telem)")
    print(f"  Telem total distance:{telem_dist[-1]:.1f} m")
    print()

    # ── Resample both to common distance grid (every 5m) ──
    max_dist = min(sim_states["distance_m"].iloc[-1], telem_dist[-1])
    dist_grid = np.arange(0, max_dist, 5.0)

    # Sim: distance_m is segment start distance. speed_kmh is avg speed in segment.
    sim_dist = sim_states["distance_m"].values
    sim_speed = sim_states["speed_kmh"].values
    sim_seg_time = sim_states["segment_time_s"].values
    sim_laps = sim_states["lap"].values

    # Interpolate sim speed onto grid
    sim_speed_on_grid = np.interp(dist_grid, sim_dist, sim_speed)

    # Interpolate telemetry speed onto grid
    telem_speed_on_grid = np.interp(dist_grid, telem_dist, telem_speed_kmh)

    # ── Compute speed difference ──
    speed_diff = sim_speed_on_grid - telem_speed_on_grid  # positive = sim faster
    abs_diff = np.abs(speed_diff)

    print("GLOBAL SPEED COMPARISON (resampled every 5m)")
    print("-" * 50)
    print(f"  Mean speed diff (sim - telem):  {np.mean(speed_diff):+.3f} km/h")
    print(f"  Mean |speed diff|:              {np.mean(abs_diff):.3f} km/h")
    print(f"  Max |speed diff|:               {np.max(abs_diff):.3f} km/h")
    print(f"  Std speed diff:                 {np.std(speed_diff):.3f} km/h")
    print(f"  Median speed diff:              {np.median(speed_diff):+.3f} km/h")
    print(f"  Mean sim speed:                 {np.mean(sim_speed_on_grid):.3f} km/h")
    print(f"  Mean telem speed:               {np.mean(telem_speed_on_grid):.3f} km/h")
    print()

    # ── Cumulative time error ──
    # For each 5m bin, compute time = 5m / speed_ms
    # (time to traverse 5m at the local speed)
    ds = 5.0  # meters per grid step
    sim_speed_ms = sim_speed_on_grid / 3.6
    telem_speed_ms = telem_speed_on_grid / 3.6

    # Clamp to avoid divide by zero
    sim_speed_ms = np.maximum(sim_speed_ms, 0.5)
    telem_speed_ms = np.maximum(telem_speed_ms, 0.5)

    sim_dt = ds / sim_speed_ms
    telem_dt = ds / telem_speed_ms
    dt_error = sim_dt - telem_dt  # negative = sim is faster (less time)
    cum_time_error = np.cumsum(dt_error)

    print("CUMULATIVE TIME ERROR GROWTH")
    print("-" * 50)
    # Report at distance milestones
    milestones = [1000, 2000, 5000, 10000, 15000, 20000, max_dist]
    for milestone in milestones:
        if milestone > max_dist:
            milestone = max_dist
        idx = np.searchsorted(dist_grid, milestone)
        if idx >= len(cum_time_error):
            idx = len(cum_time_error) - 1
        lap_num = int(dist_grid[idx] / track.total_distance_m)
        print(f"  At {dist_grid[idx]:>8.0f}m (lap ~{lap_num+1:>2d}): "
              f"cum time error = {cum_time_error[idx]:+.2f} s")
    print()

    # ── Per-lap breakdown ──
    print("PER-LAP BREAKDOWN")
    print("-" * 70)
    lap_distance = track.total_distance_m

    # Detect telemetry lap boundaries
    telem_laps = detect_lap_boundaries(aim_df)
    print(f"  Telemetry laps detected: {len(telem_laps)}")

    # Compute sim lap times
    sim_lap_times = sim_states.groupby("lap")["segment_time_s"].sum()

    # Compute telemetry lap times from detected boundaries
    telem_lap_times = []
    for start_idx, end_idx, lap_dist in telem_laps:
        t_start = aim_df["Time"].iloc[start_idx]
        t_end = aim_df["Time"].iloc[end_idx]
        telem_lap_times.append(t_end - t_start)

    print(f"\n  {'Lap':>4s}  {'Sim Time':>10s}  {'Telem Time':>10s}  "
          f"{'Delta':>8s}  {'Mean SPD diff':>14s}  {'Max SPD diff':>14s}")
    print(f"  {'':>4s}  {'(s)':>10s}  {'(s)':>10s}  {'(s)':>8s}  "
          f"{'(km/h)':>14s}  {'(km/h)':>14s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*14}  {'-'*14}")

    total_sim = 0.0
    total_telem = 0.0
    lap_deltas = []
    num_compare = min(len(sim_lap_times), len(telem_lap_times))

    for lap_idx in range(num_compare):
        sim_t = sim_lap_times.iloc[lap_idx]
        telem_t = telem_lap_times[lap_idx]
        delta = sim_t - telem_t
        total_sim += sim_t
        total_telem += telem_t

        # Speed diff for this lap's distance range
        d_start = lap_idx * lap_distance
        d_end = (lap_idx + 1) * lap_distance
        mask = (dist_grid >= d_start) & (dist_grid < d_end)
        if mask.any():
            lap_speed_diff = speed_diff[mask]
            mean_sdiff = np.mean(lap_speed_diff)
            max_sdiff_signed = lap_speed_diff[np.argmax(np.abs(lap_speed_diff))]
        else:
            mean_sdiff = 0.0
            max_sdiff_signed = 0.0

        lap_deltas.append((lap_idx, delta, mean_sdiff))
        print(f"  {lap_idx+1:>4d}  {sim_t:>10.2f}  {telem_t:>10.2f}  "
              f"{delta:>+8.2f}  {mean_sdiff:>+14.3f}  {max_sdiff_signed:>+14.3f}")

    # Any remaining sim laps without telemetry match
    for lap_idx in range(num_compare, len(sim_lap_times)):
        sim_t = sim_lap_times.iloc[lap_idx]
        total_sim += sim_t
        print(f"  {lap_idx+1:>4d}  {sim_t:>10.2f}  {'N/A':>10s}  "
              f"{'N/A':>8s}  {'N/A':>14s}  {'N/A':>14s}")

    print(f"  {'':>4s}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'SUM':>4s}  {total_sim:>10.2f}  {total_telem:>10.2f}  "
          f"{total_sim - total_telem:>+8.2f}")

    # Rank laps by contribution to error
    print(f"\n  LAPS RANKED BY TIME ERROR (largest negative = sim too fast):")
    sorted_laps = sorted(lap_deltas, key=lambda x: x[1])
    for rank, (lap_idx, delta, mean_sdiff) in enumerate(sorted_laps[:10], 1):
        print(f"    #{rank}: Lap {lap_idx+1:>2d}  delta={delta:+.2f}s  "
              f"mean speed diff={mean_sdiff:+.3f} km/h")

    # ── TOP 10 worst distance ranges ──
    print()
    print("TOP 10 WORST 50m DISTANCE RANGES (by |time error|)")
    print("-" * 70)

    # Compute time error in 50m bins
    bin_size = 50.0  # meters
    num_bins = int(max_dist / bin_size)
    bin_time_errors = np.zeros(num_bins)
    bin_speed_diffs = np.zeros(num_bins)
    bin_sim_speeds = np.zeros(num_bins)
    bin_telem_speeds = np.zeros(num_bins)

    for b in range(num_bins):
        d_start = b * bin_size
        d_end = (b + 1) * bin_size
        mask = (dist_grid >= d_start) & (dist_grid < d_end)
        if mask.any():
            bin_time_errors[b] = np.sum(dt_error[mask])
            bin_speed_diffs[b] = np.mean(speed_diff[mask])
            bin_sim_speeds[b] = np.mean(sim_speed_on_grid[mask])
            bin_telem_speeds[b] = np.mean(telem_speed_on_grid[mask])

    # Sort by absolute time error
    worst_bins = np.argsort(np.abs(bin_time_errors))[::-1][:10]

    print(f"  {'Rank':>4s}  {'Dist Range':>18s}  {'Lap':>4s}  {'Lap Dist':>10s}  "
          f"{'Time Err':>10s}  {'Sim Spd':>10s}  {'Tel Spd':>10s}  {'Spd Diff':>10s}")
    print(f"  {'':>4s}  {'(m)':>18s}  {'':>4s}  {'(m)':>10s}  "
          f"{'(s)':>10s}  {'(km/h)':>10s}  {'(km/h)':>10s}  {'(km/h)':>10s}")
    print(f"  {'-'*4}  {'-'*18}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for rank, b in enumerate(worst_bins, 1):
        d_start = b * bin_size
        d_end = (b + 1) * bin_size
        lap_num = int(d_start / lap_distance) + 1
        lap_dist = d_start % lap_distance
        print(f"  {rank:>4d}  {d_start:>8.0f}-{d_end:>7.0f}  {lap_num:>4d}  {lap_dist:>10.0f}  "
              f"{bin_time_errors[b]:>+10.3f}  {bin_sim_speeds[b]:>10.1f}  "
              f"{bin_telem_speeds[b]:>10.1f}  {bin_speed_diffs[b]:>+10.1f}")

    # ── Within-lap position analysis ──
    # Where on the lap does error accumulate?
    print()
    print("WITHIN-LAP ERROR PATTERN (averaged across all laps)")
    print("-" * 70)
    print("  Which part of the lap contributes most error?")
    print()

    # Divide each lap into 20 equal sections
    num_sections = 20
    section_len = lap_distance / num_sections
    section_time_errors = np.zeros(num_sections)
    section_counts = np.zeros(num_sections)
    section_speed_diffs = np.zeros(num_sections)

    for i in range(len(dist_grid)):
        lap_pos = dist_grid[i] % lap_distance
        section = min(int(lap_pos / section_len), num_sections - 1)
        section_time_errors[section] += dt_error[i]
        section_speed_diffs[section] += speed_diff[i]
        section_counts[section] += 1

    # Average per-pass speed diff
    section_avg_speed_diff = np.where(
        section_counts > 0,
        section_speed_diffs / section_counts,
        0.0,
    )

    print(f"  {'Section':>7s}  {'Dist Range':>18s}  {'Total Time Err':>15s}  "
          f"{'Avg Spd Diff':>14s}  {'Passes':>7s}")
    print(f"  {'-'*7}  {'-'*18}  {'-'*15}  {'-'*14}  {'-'*7}")

    for s in range(num_sections):
        d_start_sec = s * section_len
        d_end_sec = (s + 1) * section_len
        print(f"  {s+1:>7d}  {d_start_sec:>8.0f}-{d_end_sec:>7.0f}  "
              f"{section_time_errors[s]:>+15.2f}  "
              f"{section_avg_speed_diff[s]:>+14.3f}  "
              f"{section_counts[s]:>7.0f}")

    print()
    print(f"  Sections ranked by |time error|:")
    ranked = sorted(range(num_sections), key=lambda s: abs(section_time_errors[s]), reverse=True)
    for rank, s in enumerate(ranked[:5], 1):
        d_s = s * section_len
        d_e = (s + 1) * section_len
        print(f"    #{rank}: Section {s+1} ({d_s:.0f}-{d_e:.0f}m): "
              f"total time error = {section_time_errors[s]:+.2f}s, "
              f"avg speed diff = {section_avg_speed_diff[s]:+.3f} km/h")

    # ── Speed histogram comparison ──
    print()
    print("SPEED DISTRIBUTION COMPARISON")
    print("-" * 50)
    bins = [0, 10, 20, 30, 40, 50, 60, 70]
    sim_hist, _ = np.histogram(sim_speed_on_grid, bins=bins)
    telem_hist, _ = np.histogram(telem_speed_on_grid, bins=bins)
    total_pts = len(dist_grid)
    print(f"  {'Speed Range':>15s}  {'Sim %':>8s}  {'Telem %':>8s}  {'Diff %':>8s}")
    for i in range(len(bins) - 1):
        s_pct = 100.0 * sim_hist[i] / total_pts
        t_pct = 100.0 * telem_hist[i] / total_pts
        print(f"  {bins[i]:>5d}-{bins[i+1]:>3d} km/h  {s_pct:>8.1f}  {t_pct:>8.1f}  {s_pct-t_pct:>+8.1f}")

    # ── Final summary ──
    print()
    print("=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    # Is the error gradual or concentrated?
    # Check: what fraction of the total error comes from the worst 20% of bins?
    sorted_abs_errors = np.sort(np.abs(bin_time_errors))[::-1]
    top_20pct = int(0.2 * num_bins)
    top_20pct_error = np.sum(sorted_abs_errors[:top_20pct])
    total_abs_error = np.sum(sorted_abs_errors)
    print(f"  Total time error (from resampled grid): {cum_time_error[-1]:+.2f} s")
    print(f"  Top 20% of 50m bins account for {100*top_20pct_error/total_abs_error:.1f}% of |time error|")

    # Is sim consistently faster or mixed?
    faster_bins = np.sum(bin_time_errors < -0.01)
    slower_bins = np.sum(bin_time_errors > 0.01)
    neutral_bins = num_bins - faster_bins - slower_bins
    print(f"  Bins where sim is faster: {faster_bins}/{num_bins} ({100*faster_bins/num_bins:.1f}%)")
    print(f"  Bins where sim is slower: {slower_bins}/{num_bins} ({100*slower_bins/num_bins:.1f}%)")
    print(f"  Neutral bins:             {neutral_bins}/{num_bins} ({100*neutral_bins/num_bins:.1f}%)")

    # Average error per lap
    if num_compare > 0:
        avg_error_per_lap = (total_sim - total_telem) / num_compare
        print(f"  Average time error per lap: {avg_error_per_lap:+.2f} s/lap")
        print(f"  Total gap over {num_compare} matched laps: {total_sim - total_telem:+.2f} s")

    # Check if the sim is faster at low speeds (corners) or high speeds (straights)
    low_speed_mask = telem_speed_on_grid < 30
    high_speed_mask = telem_speed_on_grid >= 30
    if low_speed_mask.any():
        low_speed_err = np.mean(speed_diff[low_speed_mask])
        low_speed_time_err = np.sum(dt_error[low_speed_mask])
        print(f"\n  At telem speed < 30 km/h (corners):")
        print(f"    Mean speed diff: {low_speed_err:+.3f} km/h")
        print(f"    Cumulative time error: {low_speed_time_err:+.2f} s")
        print(f"    Fraction of distance: {100*low_speed_mask.sum()/len(dist_grid):.1f}%")
    if high_speed_mask.any():
        high_speed_err = np.mean(speed_diff[high_speed_mask])
        high_speed_time_err = np.sum(dt_error[high_speed_mask])
        print(f"  At telem speed >= 30 km/h (straights/fast sections):")
        print(f"    Mean speed diff: {high_speed_err:+.3f} km/h")
        print(f"    Cumulative time error: {high_speed_time_err:+.2f} s")
        print(f"    Fraction of distance: {100*high_speed_mask.sum()/len(dist_grid):.1f}%")


if __name__ == "__main__":
    main()
