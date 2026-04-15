"""Diagnose WHERE the force-based replay sim diverges from real telemetry.

Runs SimulationEngine with ReplayStrategy.from_full_endurance() for 22 laps,
then compares the sim speed profile against real GPS Speed, segment by segment.

Key questions:
1. Is the error everywhere evenly, or concentrated in specific track sections?
2. Does the error grow over time (compounding) or is it consistent per lap?
3. Are corners or straights the problem?
4. Which specific segments have the largest speed discrepancy?
"""

import sys
sys.path.insert(0, "src")

import math
import numpy as np
import pandas as pd

from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.track.track import Track
from fsae_sim.vehicle.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine


def classify_segment(curvature, threshold=0.005):
    k = abs(curvature)
    if k < threshold:
        return "straight"
    elif k < 0.02:
        return "gentle_corner"
    elif k < 0.04:
        return "medium_corner"
    else:
        return "tight_corner"


def main():
    print("=" * 90)
    print("REPLAY STRATEGY SPEED PROFILE DIAGNOSIS")
    print("Force-based sim with exact LVCU torque commands vs real telemetry")
    print("=" * 90)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[1] Loading data...")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")

    # Build track from cleaned telemetry
    track = Track.from_telemetry(df=aim_df)
    lap_dist = track.total_distance_m
    n_segments = track.num_segments

    print(f"  Track: {track.name}, {n_segments} segments, {lap_dist:.1f}m per lap")

    # Battery model
    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)

    # Build ReplayStrategy from full endurance
    strategy = ReplayStrategy.from_full_endurance(aim_df, lap_dist)
    print(f"  ReplayStrategy built: wrap={strategy._wrap}, "
          f"total_distance={strategy._total_distance_m:.0f}m")

    # ================================================================
    # RUN SIM
    # ================================================================
    print("\n[2] Running 22-lap force-based replay sim...")
    engine = SimulationEngine(config, track, strategy, battery)
    result = engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=29.0)

    sim_df = result.states
    print(f"  Sim completed: {result.total_time_s:.1f}s, "
          f"{result.total_energy_kwh:.2f} kWh, "
          f"final SOC={result.final_soc:.1f}%")

    # ================================================================
    # BUILD TELEMETRY REFERENCE (per-segment speed by distance)
    # ================================================================
    print("\n[3] Building telemetry reference speed profile...")

    # Filter to moving samples
    tel = aim_df[aim_df["GPS Speed"] > 5.0].copy()
    tel_dist = tel["Distance on GPS Speed"].values
    tel_speed_kmh = tel["GPS Speed"].values
    tel_speed_ms = tel_speed_kmh / 3.6
    tel_torque = tel["LVCU Torque Req"].values

    total_tel_dist = float(tel_dist[-1] - tel_dist[0])
    # Telemetry time: sum dt for moving samples
    tel_time = tel["Time"].values
    tel_dt = np.diff(tel_time, prepend=tel_time[0])
    total_tel_time = float(np.sum(tel_dt))

    print(f"  Telemetry: {total_tel_dist:.0f}m total distance, "
          f"{total_tel_time:.1f}s driving time")
    print(f"  Sim:       {sim_df['distance_m'].iloc[-1]:.0f}m total distance, "
          f"{result.total_time_s:.1f}s")
    print(f"  Time gap:  {result.total_time_s - total_tel_time:+.1f}s "
          f"({(result.total_time_s - total_tel_time) / total_tel_time * 100:+.1f}%)")

    # ================================================================
    # MAP TELEMETRY TO SIM SEGMENTS
    # For each sim segment (lap, seg_idx), find the corresponding
    # telemetry speed at that cumulative distance
    # ================================================================
    print("\n[4] Mapping telemetry speed to sim segments...")

    from scipy.interpolate import interp1d

    # Build telemetry interpolator: cumulative distance -> speed
    tel_speed_interp = interp1d(
        tel_dist, tel_speed_ms, kind="linear",
        bounds_error=False, fill_value=(float(tel_speed_ms[0]), float(tel_speed_ms[-1]))
    )
    tel_torque_interp = interp1d(
        tel_dist, tel_torque, kind="linear",
        bounds_error=False, fill_value=(float(tel_torque[0]), float(tel_torque[-1]))
    )

    # For each sim row, look up real speed at the same cumulative distance
    sim_distances = sim_df["distance_m"].values
    sim_speeds_ms = sim_df["speed_ms"].values

    real_speeds_ms = tel_speed_interp(sim_distances)
    real_torques = tel_torque_interp(sim_distances)

    sim_df = sim_df.copy()
    sim_df["real_speed_ms"] = real_speeds_ms
    sim_df["real_speed_kmh"] = real_speeds_ms * 3.6
    sim_df["real_torque_nm"] = real_torques
    sim_df["speed_error_ms"] = sim_speeds_ms - real_speeds_ms
    sim_df["speed_error_kmh"] = sim_df["speed_error_ms"] * 3.6
    sim_df["speed_error_pct"] = (sim_df["speed_error_ms"] / np.maximum(real_speeds_ms, 0.5)) * 100
    sim_df["seg_type"] = sim_df["curvature"].apply(classify_segment)

    # Telemetry time per segment: distance / real_speed
    seg_len = 5.0  # meters per segment
    sim_df["real_seg_time_s"] = seg_len / np.maximum(real_speeds_ms, 0.5)
    sim_df["time_error_s"] = sim_df["segment_time_s"] - sim_df["real_seg_time_s"]

    # ================================================================
    # SECTION A: OVERALL SUMMARY
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION A: OVERALL SPEED ERROR SUMMARY")
    print("=" * 90)

    mean_err = sim_df["speed_error_kmh"].mean()
    median_err = sim_df["speed_error_kmh"].median()
    std_err = sim_df["speed_error_kmh"].std()
    mean_abs_err = sim_df["speed_error_kmh"].abs().mean()

    print(f"  Mean speed error:     {mean_err:+.2f} km/h (negative = sim slower)")
    print(f"  Median speed error:   {median_err:+.2f} km/h")
    print(f"  Std dev:              {std_err:.2f} km/h")
    print(f"  Mean absolute error:  {mean_abs_err:.2f} km/h")
    print(f"  Sim mean speed:       {sim_df['speed_kmh'].mean():.1f} km/h")
    print(f"  Tel mean speed:       {sim_df['real_speed_kmh'].mean():.1f} km/h")

    # How many segments is sim slower vs faster?
    n_slower = (sim_df["speed_error_ms"] < -0.5).sum()
    n_faster = (sim_df["speed_error_ms"] > 0.5).sum()
    n_close = ((sim_df["speed_error_ms"].abs()) <= 0.5).sum()
    n_total = len(sim_df)
    print(f"\n  Sim SLOWER by >0.5 m/s: {n_slower} segments ({n_slower/n_total*100:.1f}%)")
    print(f"  Sim FASTER by >0.5 m/s: {n_faster} segments ({n_faster/n_total*100:.1f}%)")
    print(f"  Within 0.5 m/s:         {n_close} segments ({n_close/n_total*100:.1f}%)")

    # ================================================================
    # SECTION B: LAP-BY-LAP ERROR ANALYSIS
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION B: LAP-BY-LAP ERROR (does error grow over time?)")
    print("=" * 90)

    print(f"\n  {'Lap':>4} {'SimTime':>8} {'TelTime':>8} {'TimeDiff':>9} "
          f"{'SimSpd':>8} {'TelSpd':>8} {'SpdErr':>8} {'AbsSpdE':>8} {'CumDist':>8}")
    print("  " + "-" * 80)

    lap_summaries = []
    for lap in range(22):
        lap_data = sim_df[sim_df["lap"] == lap]
        if len(lap_data) == 0:
            continue

        sim_lap_time = lap_data["segment_time_s"].sum()
        tel_lap_time = lap_data["real_seg_time_s"].sum()
        time_diff = sim_lap_time - tel_lap_time
        sim_avg_speed = lap_data["speed_kmh"].mean()
        tel_avg_speed = lap_data["real_speed_kmh"].mean()
        speed_err = lap_data["speed_error_kmh"].mean()
        abs_speed_err = lap_data["speed_error_kmh"].abs().mean()
        cum_dist = lap_data["distance_m"].iloc[-1]

        lap_summaries.append({
            "lap": lap,
            "sim_time": sim_lap_time,
            "tel_time": tel_lap_time,
            "time_diff": time_diff,
            "sim_avg_speed": sim_avg_speed,
            "tel_avg_speed": tel_avg_speed,
            "speed_err": speed_err,
            "abs_speed_err": abs_speed_err,
            "cum_dist": cum_dist,
        })

        print(f"  {lap:4d} {sim_lap_time:7.1f}s {tel_lap_time:7.1f}s {time_diff:+8.1f}s "
              f"{sim_avg_speed:7.1f} {tel_avg_speed:7.1f} {speed_err:+7.1f} "
              f"{abs_speed_err:7.1f} {cum_dist:7.0f}m")

    lap_sum_df = pd.DataFrame(lap_summaries)
    # Check for trend
    if len(lap_sum_df) > 3:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(lap_sum_df["lap"].values, lap_sum_df["speed_err"].values, 1)
        slope_per_lap = coeffs[1]
        print(f"\n  Speed error trend: {slope_per_lap:+.3f} km/h per lap "
              f"({'growing' if slope_per_lap < -0.05 else 'shrinking' if slope_per_lap > 0.05 else 'stable'})")
        print(f"  Lap 0 mean error: {lap_sum_df.iloc[0]['speed_err']:+.1f} km/h")
        print(f"  Lap 21 mean error: {lap_sum_df.iloc[-1]['speed_err']:+.1f} km/h")

        # Time error trend
        coeffs_t = P.polyfit(lap_sum_df["lap"].values, lap_sum_df["time_diff"].values, 1)
        print(f"  Time diff trend: {coeffs_t[1]:+.3f}s per lap "
              f"({'growing' if coeffs_t[1] > 0.05 else 'shrinking' if coeffs_t[1] < -0.05 else 'stable'})")

    # ================================================================
    # SECTION C: ERROR BY SEGMENT TYPE (straights vs corners)
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION C: ERROR BY SEGMENT TYPE (straights vs corners)")
    print("=" * 90)

    for seg_type in ["straight", "gentle_corner", "medium_corner", "tight_corner"]:
        subset = sim_df[sim_df["seg_type"] == seg_type]
        if len(subset) == 0:
            continue

        n_segs = len(subset)
        pct_segs = n_segs / n_total * 100
        mean_spd_err = subset["speed_error_kmh"].mean()
        median_spd_err = subset["speed_error_kmh"].median()
        time_err_total = subset["time_error_s"].sum()
        sim_avg = subset["speed_kmh"].mean()
        tel_avg = subset["real_speed_kmh"].mean()

        print(f"\n  {seg_type.upper()} ({n_segs} segments, {pct_segs:.1f}% of track)")
        print(f"    Mean speed error:     {mean_spd_err:+.2f} km/h")
        print(f"    Median speed error:   {median_spd_err:+.2f} km/h")
        print(f"    Total time error:     {time_err_total:+.1f}s (over 22 laps)")
        print(f"    Sim avg speed:        {sim_avg:.1f} km/h")
        print(f"    Tel avg speed:        {tel_avg:.1f} km/h")

    # Also by action
    print("\n  By driver action (over 22 laps):")
    for action in ["throttle", "coast", "brake"]:
        subset = sim_df[sim_df["action"] == action]
        if len(subset) == 0:
            continue
        time_err = subset["time_error_s"].sum()
        spd_err = subset["speed_error_kmh"].mean()
        print(f"    {action:>10}: {len(subset):5d} segs, "
              f"time_err={time_err:+.1f}s, "
              f"speed_err={spd_err:+.1f} km/h")

    # ================================================================
    # SECTION D: PER-SEGMENT ERROR PROFILE (averaged across laps)
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION D: PER-SEGMENT ERROR (averaged across all 22 laps)")
    print("=" * 90)

    seg_avg = sim_df.groupby("segment_idx").agg({
        "speed_error_kmh": "mean",
        "speed_error_ms": "mean",
        "time_error_s": "mean",
        "speed_kmh": "mean",
        "real_speed_kmh": "mean",
        "curvature": "first",
        "seg_type": "first",
        "motor_torque_nm": "mean",
        "real_torque_nm": "mean",
        "drive_force_n": "mean",
        "resistance_force_n": "mean",
        "net_force_n": "mean",
        "action": "first",
        "corner_speed_limit_ms": "first",
    }).reset_index()

    # Top 20 worst segments by absolute speed error
    seg_avg["abs_speed_error"] = seg_avg["speed_error_kmh"].abs()
    worst20 = seg_avg.nlargest(20, "abs_speed_error")

    print(f"\n  TOP 20 WORST SEGMENTS (by mean |speed error| across laps):")
    print(f"  {'Seg':>4} {'Type':>15} {'Dist_m':>7} {'SimSpd':>7} {'TelSpd':>7} "
          f"{'SpdErr':>7} {'TimeErr':>8} {'Curv':>8} {'Action':>8} "
          f"{'SimTrq':>7} {'TelTrq':>7} {'CorLim':>7}")
    print("  " + "-" * 115)

    for _, row in worst20.iterrows():
        seg_idx = int(row["segment_idx"])
        dist_m = seg_idx * 5
        corner_lim_kmh = row["corner_speed_limit_ms"] * 3.6
        print(f"  {seg_idx:4d} {row['seg_type']:>15} {dist_m:6d}m "
              f"{row['speed_kmh']:6.1f} {row['real_speed_kmh']:6.1f} "
              f"{row['speed_error_kmh']:+6.1f} {row['time_error_s']:+7.3f}s "
              f"{row['curvature']:+7.4f} {row['action']:>8} "
              f"{row['motor_torque_nm']:6.1f} {row['real_torque_nm']:6.1f} "
              f"{corner_lim_kmh:6.1f}")

    # ================================================================
    # SECTION E: CONTIGUOUS ERROR REGIONS
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION E: CONTIGUOUS ERROR REGIONS (where sim is consistently wrong)")
    print("=" * 90)

    # Find runs of consecutive segments where sim is > 3 km/h slower
    threshold_kmh = 3.0
    in_run = False
    run_start = None
    regions = []

    for i in range(n_segments):
        err = seg_avg.loc[seg_avg["segment_idx"] == i, "speed_error_kmh"]
        if len(err) == 0:
            continue
        err_val = float(err.values[0])

        if err_val < -threshold_kmh:  # sim is slower
            if not in_run:
                in_run = True
                run_start = i
        else:
            if in_run:
                run_end = i - 1
                region_data = seg_avg[(seg_avg["segment_idx"] >= run_start) &
                                      (seg_avg["segment_idx"] <= run_end)]
                regions.append({
                    "start": run_start,
                    "end": run_end,
                    "n_segs": run_end - run_start + 1,
                    "avg_speed_err": region_data["speed_error_kmh"].mean(),
                    "max_speed_err": region_data["speed_error_kmh"].min(),  # most negative
                    "avg_time_err": region_data["time_error_s"].mean(),
                    "total_time_err_per_lap": region_data["time_error_s"].sum(),
                    "avg_curv": region_data["curvature"].abs().mean(),
                    "actions": region_data["action"].value_counts().to_dict(),
                    "seg_types": region_data["seg_type"].value_counts().to_dict(),
                    "dist_start": run_start * 5,
                    "dist_end": (run_end + 1) * 5,
                })
                in_run = False

    if in_run:
        run_end = n_segments - 1
        region_data = seg_avg[(seg_avg["segment_idx"] >= run_start) &
                              (seg_avg["segment_idx"] <= run_end)]
        regions.append({
            "start": run_start,
            "end": run_end,
            "n_segs": run_end - run_start + 1,
            "avg_speed_err": region_data["speed_error_kmh"].mean(),
            "max_speed_err": region_data["speed_error_kmh"].min(),
            "avg_time_err": region_data["time_error_s"].mean(),
            "total_time_err_per_lap": region_data["time_error_s"].sum(),
            "avg_curv": region_data["curvature"].abs().mean(),
            "actions": region_data["action"].value_counts().to_dict(),
            "seg_types": region_data["seg_type"].value_counts().to_dict(),
            "dist_start": run_start * 5,
            "dist_end": (run_end + 1) * 5,
        })

    # Sort by worst total time error per lap
    regions.sort(key=lambda r: r["total_time_err_per_lap"], reverse=True)

    total_region_time = sum(r["total_time_err_per_lap"] for r in regions)
    print(f"\n  Found {len(regions)} regions where sim is > {threshold_kmh} km/h slower")
    print(f"  Total time error from these regions: {total_region_time:+.2f}s per lap")

    print(f"\n  {'#':>3} {'Segs':>8} {'Dist':>14} {'AvgSpdE':>8} {'MaxSpdE':>8} "
          f"{'TimeE/lap':>10} {'AvgCurv':>8} {'Types':>25} {'Actions':>20}")
    print("  " + "-" * 115)

    for i, r in enumerate(regions[:15]):
        types_str = ", ".join(f"{k}:{v}" for k, v in r["seg_types"].items())
        actions_str = ", ".join(f"{k}:{v}" for k, v in r["actions"].items())
        print(f"  {i+1:3d} {r['start']:3d}-{r['end']:3d} "
              f"{r['dist_start']:5d}-{r['dist_end']:5d}m "
              f"{r['avg_speed_err']:+7.1f} {r['max_speed_err']:+7.1f} "
              f"{r['total_time_err_per_lap']:+9.3f}s "
              f"{r['avg_curv']:7.4f} "
              f"{types_str:>25} {actions_str:>20}")

    # Also find regions where sim is FASTER
    regions_fast = []
    in_run = False
    for i in range(n_segments):
        err = seg_avg.loc[seg_avg["segment_idx"] == i, "speed_error_kmh"]
        if len(err) == 0:
            continue
        err_val = float(err.values[0])

        if err_val > threshold_kmh:  # sim is faster
            if not in_run:
                in_run = True
                run_start = i
        else:
            if in_run:
                run_end = i - 1
                region_data = seg_avg[(seg_avg["segment_idx"] >= run_start) &
                                      (seg_avg["segment_idx"] <= run_end)]
                regions_fast.append({
                    "start": run_start,
                    "end": run_end,
                    "n_segs": run_end - run_start + 1,
                    "avg_speed_err": region_data["speed_error_kmh"].mean(),
                    "total_time_err_per_lap": region_data["time_error_s"].sum(),
                    "avg_curv": region_data["curvature"].abs().mean(),
                    "dist_start": run_start * 5,
                    "dist_end": (run_end + 1) * 5,
                })
                in_run = False

    if in_run:
        run_end = n_segments - 1
        region_data = seg_avg[(seg_avg["segment_idx"] >= run_start) &
                              (seg_avg["segment_idx"] <= run_end)]
        regions_fast.append({
            "start": run_start,
            "end": run_end,
            "n_segs": run_end - run_start + 1,
            "avg_speed_err": region_data["speed_error_kmh"].mean(),
            "total_time_err_per_lap": region_data["time_error_s"].sum(),
            "avg_curv": region_data["curvature"].abs().mean(),
            "dist_start": run_start * 5,
            "dist_end": (run_end + 1) * 5,
        })

    print(f"\n  Found {len(regions_fast)} regions where sim is > {threshold_kmh} km/h FASTER")
    for i, r in enumerate(regions_fast[:10]):
        print(f"  {i+1:3d} segs {r['start']:3d}-{r['end']:3d} "
              f"({r['dist_start']}-{r['dist_end']}m) "
              f"avg_err={r['avg_speed_err']:+.1f} km/h, "
              f"time_err={r['total_time_err_per_lap']:+.3f}s/lap, "
              f"curv={r['avg_curv']:.4f}")

    # ================================================================
    # SECTION F: FORCE BALANCE DIAGNOSIS FOR WORST SEGMENTS
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION F: FORCE BALANCE DIAGNOSIS (why is the sim slow?)")
    print("=" * 90)

    # For the top 5 worst regions, look at force details
    for i, r in enumerate(regions[:5]):
        start, end = r["start"], r["end"]
        # Get full sim data for this region (all laps)
        region_all_laps = sim_df[
            (sim_df["segment_idx"] >= start) & (sim_df["segment_idx"] <= end)
        ]
        # Averaged
        region_avg = seg_avg[
            (seg_avg["segment_idx"] >= start) & (seg_avg["segment_idx"] <= end)
        ]

        print(f"\n  --- Region {i+1}: segs {start}-{end} "
              f"({r['dist_start']}-{r['dist_end']}m) ---")
        print(f"  Speed error: {r['avg_speed_err']:+.1f} km/h avg, "
              f"{r['max_speed_err']:+.1f} km/h worst")
        print(f"  Time error:  {r['total_time_err_per_lap']:+.3f}s per lap, "
              f"{r['total_time_err_per_lap'] * 22:+.1f}s over 22 laps")

        # Forces
        print(f"  Drive force:      {region_avg['drive_force_n'].mean():+.1f} N")
        print(f"  Resistance force: {region_avg['resistance_force_n'].mean():+.1f} N")
        print(f"  Net force:        {region_avg['net_force_n'].mean():+.1f} N")

        # Torque comparison
        sim_trq = region_avg["motor_torque_nm"].mean()
        tel_trq = region_avg["real_torque_nm"].mean()
        print(f"  Sim motor torque:  {sim_trq:.1f} Nm")
        print(f"  Tel LVCU torque:   {tel_trq:.1f} Nm")
        print(f"  Torque diff:       {sim_trq - tel_trq:+.1f} Nm")

        # Speeds
        print(f"  Sim speed:  {region_avg['speed_kmh'].mean():.1f} km/h")
        print(f"  Tel speed:  {region_avg['real_speed_kmh'].mean():.1f} km/h")

        # Corner limits
        corner_lim = region_avg["corner_speed_limit_ms"].mean() * 3.6
        print(f"  Corner limit: {corner_lim:.1f} km/h")

        # Check if corner limit is binding
        n_at_limit = (region_all_laps["speed_kmh"] >=
                      region_all_laps["corner_speed_limit_ms"] * 3.6 - 1.0).sum()
        pct_at_limit = n_at_limit / len(region_all_laps) * 100
        print(f"  At corner limit:  {pct_at_limit:.0f}% of segments")

    # ================================================================
    # SECTION G: DISTANCE DRIFT CHECK
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION G: DISTANCE DRIFT CHECK")
    print("=" * 90)

    # The sim always traverses exactly lap_dist * 22 = n_segments * 5m * 22
    sim_total_dist = n_segments * 5.0 * 22
    tel_total_dist = total_tel_dist

    print(f"  Sim total distance:  {sim_total_dist:.0f}m ({n_segments} segs * 5m * 22 laps)")
    print(f"  Tel total distance:  {tel_total_dist:.0f}m")
    print(f"  Difference:          {sim_total_dist - tel_total_dist:+.0f}m "
          f"({(sim_total_dist - tel_total_dist) / tel_total_dist * 100:+.1f}%)")

    # Check if the telemetry covers more or less distance per lap
    # Estimate laps from telemetry distance
    tel_approx_laps = tel_total_dist / lap_dist
    print(f"  Tel approx laps:     {tel_approx_laps:.2f} (track lap = {lap_dist:.1f}m)")
    print(f"  Sim laps:            22")

    # ================================================================
    # SECTION H: CUMULATIVE TIME DIVERGENCE
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION H: CUMULATIVE TIME DIVERGENCE")
    print("=" * 90)

    # Track cumulative time error as we go through the sim
    cum_time_err = np.cumsum(sim_df["time_error_s"].values)

    # Sample at lap boundaries
    for lap in range(22):
        lap_end_idx = (lap + 1) * n_segments - 1
        if lap_end_idx < len(cum_time_err):
            print(f"  After lap {lap:2d}: cum_time_error = {cum_time_err[lap_end_idx]:+.1f}s, "
                  f"sim_time = {sim_df.iloc[lap_end_idx]['time_s']:.1f}s")

    # ================================================================
    # SECTION I: ENTRY SPEED MISMATCH CHECK
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION I: ENTRY SPEED VS EXIT SPEED MISMATCH")
    print("=" * 90)

    # The sim is quasi-static: each segment's entry speed = previous exit speed.
    # But in the force model, the sim speed at segment N depends on all prior
    # segments. If the sim gets one segment wrong, it snowballs.
    # Check: how much does the sim's speed at each segment differ from what
    # the torque command SHOULD produce given the real entry speed?

    # For the first lap, compute what exit speed SHOULD be if entry speed
    # matched telemetry
    lap0 = sim_df[sim_df["lap"] == 0].copy()

    # For each segment, compute: given REAL entry speed + sim forces, what
    # exit speed would the force model produce?
    hypothetical_speeds = np.zeros(len(lap0))
    hypothetical_speeds[0] = real_speeds_ms[0]  # start from real speed

    for i in range(len(lap0)):
        row = lap0.iloc[i]

        if i == 0:
            entry_speed = float(real_speeds_ms[0])
        else:
            # Use the REAL speed as entry (no compounding)
            entry_speed = float(lap0.iloc[i]["real_speed_ms"])

        # What net force does the sim compute?
        net_f = float(row["net_force_n"])
        corner_lim = float(row["corner_speed_limit_ms"])
        m_eff = config.vehicle.mass_kg  # approximate, doesn't include rotational inertia here

        # v_exit^2 = v_entry^2 + 2*a*d
        a = net_f / m_eff
        v_sq = entry_speed**2 + 2.0 * a * 5.0
        if v_sq < 0:
            v_sq = 0.0
        exit_speed = math.sqrt(v_sq)
        exit_speed = min(exit_speed, corner_lim)
        exit_speed = max(exit_speed, 0.5)

        avg_speed = (entry_speed + exit_speed) / 2.0
        hypothetical_speeds[i] = avg_speed

    hyp_error = hypothetical_speeds - lap0["real_speed_ms"].values
    actual_error = lap0["speed_ms"].values - lap0["real_speed_ms"].values

    print(f"  Lap 0 analysis (isolating compounding from per-segment error):")
    print(f"  ")
    print(f"  With sim's OWN entry speeds (actual sim):")
    print(f"    Mean speed error: {np.mean(actual_error) * 3.6:+.2f} km/h")
    print(f"    Mean |error|:     {np.mean(np.abs(actual_error)) * 3.6:.2f} km/h")
    print(f"  ")
    print(f"  With REAL entry speeds (no compounding):")
    print(f"    Mean speed error: {np.mean(hyp_error) * 3.6:+.2f} km/h")
    print(f"    Mean |error|:     {np.mean(np.abs(hyp_error)) * 3.6:.2f} km/h")
    print(f"  ")
    compounding_contribution = np.mean(np.abs(actual_error)) - np.mean(np.abs(hyp_error))
    print(f"  Compounding adds: {compounding_contribution * 3.6:.2f} km/h to mean |error|")
    print(f"  This means {'compounding is a major factor' if compounding_contribution * 3.6 > 1.0 else 'the error is primarily per-segment, not compounding'}")

    # ================================================================
    # SECTION J: QUICK PHYSICS SANITY CHECK
    # ================================================================
    print("\n" + "=" * 90)
    print("SECTION J: PHYSICS SANITY CHECKS")
    print("=" * 90)

    # Check effective mass
    from fsae_sim.vehicle.powertrain_model import PowertrainModel
    pt = PowertrainModel(config.powertrain)

    # At 40 km/h (typical speed), what's the force balance?
    v_test = 40.0 / 3.6  # m/s
    rpm_test = pt.motor_rpm_from_speed(v_test)
    print(f"\n  At {v_test*3.6:.0f} km/h ({v_test:.1f} m/s), motor RPM = {rpm_test:.0f}")

    # Typical torque from telemetry at this speed
    mask_40 = (sim_df["real_speed_kmh"] > 38) & (sim_df["real_speed_kmh"] < 42)
    if mask_40.sum() > 0:
        typical_torque = sim_df.loc[mask_40, "real_torque_nm"].mean()
        print(f"  Typical telemetry torque at 40 km/h: {typical_torque:.1f} Nm")

        # Wheel force from this torque
        wf = pt.wheel_force(typical_torque)
        print(f"  Wheel force: {wf:.1f} N")

        # Build dynamics for resistance check
        from fsae_sim.vehicle.dynamics import VehicleDynamics
        vp = config.vehicle
        # Simple dynamics without tire model for comparison
        dyn_simple = VehicleDynamics(vp, powertrain_config=config.powertrain)
        resist = dyn_simple.total_resistance(v_test, 0.0, 0.0)
        print(f"  Resistance at {v_test*3.6:.0f} km/h (straight): {resist:.1f} N")
        print(f"  Net force: {wf - resist:+.1f} N")
        print(f"  Expected accel: {(wf - resist) / dyn_simple.m_effective:.2f} m/s^2")
        print(f"  m_effective: {dyn_simple.m_effective:.1f} kg (bare mass: {vp.mass_kg:.0f} kg)")

        # With cornering drag at typical curvature
        typical_curv = 0.02
        resist_corner = dyn_simple.total_resistance(v_test, 0.0, typical_curv)
        print(f"  Resistance at curvature={typical_curv}: {resist_corner:.1f} N "
              f"(+{resist_corner - resist:.1f} N cornering drag)")

    # Check what the full dynamics model (with Pacejka tires) gives
    engine2 = SimulationEngine(config, track, strategy, battery)
    dyn_full = engine2.dynamics
    resist_full = dyn_full.total_resistance(v_test, 0.0, 0.0)
    resist_full_corner = dyn_full.total_resistance(v_test, 0.0, 0.02)
    print(f"\n  Full dynamics model (Pacejka tires):")
    print(f"  Resistance at {v_test*3.6:.0f} km/h straight: {resist_full:.1f} N")
    print(f"  Resistance at {v_test*3.6:.0f} km/h + curv=0.02: {resist_full_corner:.1f} N "
          f"(+{resist_full_corner - resist_full:.1f} N cornering)")
    print(f"  m_effective: {dyn_full.m_effective:.1f} kg")

    print("\n" + "=" * 90)
    print("DIAGNOSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
