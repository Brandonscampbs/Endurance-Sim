"""Deep-dive analysis: account for ALL of the ~165s gap between sim and telemetry.

Compares the CalibratedStrategy force-based sim (single lap, then full endurance)
against AiM telemetry to identify WHERE and WHY the sim is faster.

Key question: sim ~ 1500s, telemetry ~ 1665s => ~165s gap over ~21 laps.
That's ~7.9s per lap. Where does it come from?
"""

import math
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from fsae_sim.data.loader import load_aim_csv, load_voltt_csv
from fsae_sim.track.track import Track
from fsae_sim.vehicle.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.powertrain_model import PowertrainModel
from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.analysis.validation import detect_lap_boundaries


def classify_segment(curvature, threshold=0.005):
    """Classify segment as straight, corner, or transition."""
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
    print("=" * 80)
    print("DEEP-DIVE TIME ERROR ANALYSIS")
    print("Accounting for ALL 165 seconds of gap (sim: ~1500s, telemetry: ~1665s)")
    print("=" * 80)

    # Load data
    _, aim_df = load_aim_csv("Real-Car-Data-And-Stats/2025 Endurance Data.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    track = Track.from_telemetry("Real-Car-Data-And-Stats/2025 Endurance Data.csv")

    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)
    battery.calibrate_pack_from_telemetry(aim_df)

    powertrain = PowertrainModel(config.powertrain)

    # Detect laps
    lap_boundaries = detect_lap_boundaries(aim_df)
    n_laps = len(lap_boundaries)

    # Calibrate driver model
    strategy = CalibratedStrategy.from_telemetry(aim_df, track)

    # ====================================================================
    # SECTION 1: Per-lap telemetry times to understand the 1665s total
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: TELEMETRY LAP TIMES (what makes up the 1665s?)")
    print("=" * 80)

    total_tel_time = 0.0
    tel_lap_times = []
    for i, (s, e, d) in enumerate(lap_boundaries):
        t = aim_df["Time"].iloc[e] - aim_df["Time"].iloc[s]
        avg_speed = aim_df["GPS Speed"].iloc[s:e].mean()
        tel_lap_times.append(t)
        total_tel_time += t
        driver = "D1" if i < 10 else "D2"
        print(f"  Lap {i+1:2d} ({driver}): {t:6.1f}s, dist={d:6.0f}m, "
              f"avg_speed={avg_speed:.1f} km/h")

    # Find gaps between laps (driver change, etc.)
    speed_arr = aim_df["GPS Speed"].values
    time_arr = aim_df["Time"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    total_recording = time_arr[-1] - time_arr[0]
    driving_time = float(np.sum(dt_arr[speed_arr > 5]))
    stopped_time = total_recording - driving_time

    print(f"\n  Total recording time: {total_recording:.1f}s")
    print(f"  Sum of lap times:    {total_tel_time:.1f}s")
    print(f"  Driving time (>5km/h): {driving_time:.1f}s")
    print(f"  Stopped/slow time:   {stopped_time:.1f}s")

    # ====================================================================
    # SECTION 2: Single-lap detailed comparison (sim vs telemetry)
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: SINGLE-LAP SIM vs AVERAGED TELEMETRY (laps 3-8)")
    print("=" * 80)

    engine = SimulationEngine(config, track, strategy, battery)
    result = engine.run(num_laps=1, initial_soc_pct=92.0, initial_temp_c=28.0)
    sim_df = result.states

    sim_lap_time = sim_df["segment_time_s"].sum()

    # Find "normal" laps (exclude short, driver change, outliers)
    normal_laps = []
    for i, t in enumerate(tel_lap_times):
        dist = lap_boundaries[i][2]
        if 60 < t < 100 and 900 < dist < 1100:  # normal full-speed laps
            normal_laps.append((i, t))

    normal_times = [t for _, t in normal_laps]
    avg_tel_normal = np.mean(normal_times)
    d1_normal = [t for i, t in normal_laps if i < 10]
    d2_normal = [t for i, t in normal_laps if i >= 10]

    print(f"\n  Sim 1-lap time:        {sim_lap_time:.1f}s")
    print(f"  Normal laps found:     {len(normal_laps)} (excluding driver change, outliers)")
    print(f"  Avg normal lap time:   {avg_tel_normal:.1f}s")
    print(f"  D1 avg normal:         {np.mean(d1_normal):.1f}s ({len(d1_normal)} laps)")
    print(f"  D2 avg normal:         {np.mean(d2_normal):.1f}s ({len(d2_normal)} laps)")
    print(f"  Per-lap gap (normal):  {avg_tel_normal - sim_lap_time:.1f}s "
          f"(sim is {(avg_tel_normal - sim_lap_time) / avg_tel_normal * 100:.1f}% faster)")
    print(f"  Projected 21-lap gap:  {(avg_tel_normal - sim_lap_time) * 21:.1f}s")

    # Compare against actual total
    n_sim_laps = 21
    print(f"\n  Actual full-endurance comparison:")
    print(f"    Tel driving time: {driving_time:.1f}s over {n_laps} laps")
    print(f"    Tel avg lap time: {driving_time / n_laps:.1f}s (includes outliers)")
    print(f"    Sum of detected lap times: {total_tel_time:.1f}s")

    # Get segment telemetry (averaged over laps 3-8)
    # Import the function from the diagnostic script directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diagnose_accuracy",
        "C:/Users/brand/Development-BC/scripts/diagnose_accuracy.py",
    )
    diag_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag_mod)
    tel_avg = diag_mod.segment_telemetry_averaged(aim_df, track, lap_boundaries, range(2, 8))
    merged = sim_df.merge(tel_avg, on="segment_idx", how="inner")

    # Per-segment time comparison
    # Telemetry time per segment = segment_length / telemetry_speed
    seg_len = 5.0  # 5m segments
    merged["tel_time_s"] = seg_len / (merged["tel_speed_ms"].clip(lower=0.5))
    merged["time_error_s"] = merged["segment_time_s"] - merged["tel_time_s"]
    merged["segment_type"] = merged["curvature"].apply(classify_segment)

    total_time_err = merged["time_error_s"].sum()
    print(f"\n  Sum of segment time errors: {total_time_err:.2f}s")

    # ====================================================================
    # SECTION 3: TOP 10 WORST SEGMENTS BY TIME ERROR
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: TOP 10 SEGMENTS BY TIME ERROR (per-segment)")
    print("=" * 80)

    merged["time_error_abs"] = merged["time_error_s"].abs()
    merged["speed_error_kmh"] = merged["speed_kmh"] - merged["tel_speed_kmh"]
    worst10 = merged.nlargest(10, "time_error_abs")

    print(f"\n  {'Seg':>4} {'Type':>15} {'SimSpd':>8} {'TelSpd':>8} {'SpdErr':>8} "
          f"{'SimTime':>8} {'TelTime':>8} {'TimeErr':>8} {'Curv':>8} {'Action':>8} "
          f"{'SimTrq':>7} {'TelTrq':>7}")
    print("  " + "-" * 112)

    for _, row in worst10.iterrows():
        print(f"  {int(row['segment_idx']):4d} "
              f"{row['segment_type']:>15} "
              f"{row['speed_kmh']:7.1f} "
              f"{row['tel_speed_kmh']:7.1f} "
              f"{row['speed_error_kmh']:+7.1f} "
              f"{row['segment_time_s']:7.3f} "
              f"{row['tel_time_s']:7.3f} "
              f"{row['time_error_s']:+7.3f} "
              f"{row['curvature']:+7.4f} "
              f"{row['action']:>8} "
              f"{row['motor_torque_nm']:6.1f} "
              f"{row['tel_torque_nm']:6.1f}")

    # ====================================================================
    # SECTION 4: TIME ERROR BREAKDOWN BY SEGMENT TYPE
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: TIME ERROR BREAKDOWN BY SEGMENT TYPE")
    print("=" * 80)

    for seg_type in ["straight", "gentle_corner", "medium_corner", "tight_corner"]:
        subset = merged[merged["segment_type"] == seg_type]
        if len(subset) == 0:
            continue
        type_time_err = subset["time_error_s"].sum()
        count = len(subset)
        avg_speed_err = (subset["speed_kmh"] - subset["tel_speed_kmh"]).mean()
        pct_of_total = type_time_err / total_time_err * 100 if total_time_err != 0 else 0
        print(f"\n  {seg_type} ({count} segments):")
        print(f"    Total time error:    {type_time_err:+.3f}s ({pct_of_total:.0f}% of total)")
        print(f"    Mean speed error:    {avg_speed_err:+.1f} km/h")
        print(f"    Sim avg speed:       {subset['speed_kmh'].mean():.1f} km/h")
        print(f"    Tel avg speed:       {subset['tel_speed_kmh'].mean():.1f} km/h")

    # Group by driver action
    print("\n  By driver action:")
    for act in ["throttle", "coast", "brake"]:
        subset = merged[merged["action"] == act]
        if len(subset) == 0:
            continue
        type_time_err = subset["time_error_s"].sum()
        pct = type_time_err / total_time_err * 100 if total_time_err != 0 else 0
        print(f"    {act:>10}: {type_time_err:+.3f}s ({pct:.0f}%), "
              f"{len(subset)} segments, "
              f"avg speed err={((subset['speed_kmh'] - subset['tel_speed_kmh']).mean()):+.1f} km/h")

    # ====================================================================
    # SECTION 5: WHERE IS SIM FASTER vs SLOWER?
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: SIM FASTER vs SIM SLOWER")
    print("=" * 80)

    sim_faster = merged[merged["speed_error_kmh"] > 0]
    sim_slower = merged[merged["speed_error_kmh"] < 0]

    sim_faster_time = sim_faster["time_error_s"].sum()
    sim_slower_time = sim_slower["time_error_s"].sum()

    print(f"\n  Sim FASTER segments:  {len(sim_faster)} segs, "
          f"time contribution: {sim_faster_time:+.3f}s")
    print(f"  Sim SLOWER segments:  {len(sim_slower)} segs, "
          f"time contribution: {sim_slower_time:+.3f}s")
    print(f"  Net:                  {sim_faster_time + sim_slower_time:+.3f}s")

    print(f"\n  Where sim is FASTER:")
    print(f"    Mean speed excess:  {sim_faster['speed_error_kmh'].mean():+.1f} km/h")
    print(f"    Max speed excess:   {sim_faster['speed_error_kmh'].max():+.1f} km/h")
    print(f"    Mean |curvature|:   {sim_faster['curvature'].abs().mean():.4f}")

    print(f"\n  Where sim is SLOWER:")
    print(f"    Mean speed deficit: {sim_slower['speed_error_kmh'].mean():+.1f} km/h")
    print(f"    Min speed deficit:  {sim_slower['speed_error_kmh'].min():+.1f} km/h")
    print(f"    Mean |curvature|:   {sim_slower['curvature'].abs().mean():.4f}")

    # ====================================================================
    # SECTION 6: IDENTIFY CONTIGUOUS PROBLEM REGIONS
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: CONTIGUOUS ERROR REGIONS (clusters of bad segments)")
    print("=" * 80)

    # Find runs of consecutive segments where sim is >10 km/h faster
    in_run = False
    run_start = None
    regions = []

    for i, row in merged.iterrows():
        if row["speed_error_kmh"] > 8:  # sim is 8+ km/h faster
            if not in_run:
                in_run = True
                run_start = int(row["segment_idx"])
        else:
            if in_run:
                run_end = int(merged.iloc[i - 1]["segment_idx"]) if i > 0 else run_start
                region_data = merged[(merged["segment_idx"] >= run_start) &
                                     (merged["segment_idx"] <= run_end)]
                regions.append({
                    "start": run_start,
                    "end": run_end,
                    "n_segs": run_end - run_start + 1,
                    "time_err": region_data["time_error_s"].sum(),
                    "avg_speed_err": (region_data["speed_kmh"] - region_data["tel_speed_kmh"]).mean(),
                    "max_speed_err": (region_data["speed_kmh"] - region_data["tel_speed_kmh"]).max(),
                    "avg_curv": region_data["curvature"].abs().mean(),
                    "dist_start": run_start * 5,
                    "dist_end": (run_end + 1) * 5,
                })
                in_run = False

    # Catch trailing run
    if in_run:
        run_end = int(merged.iloc[-1]["segment_idx"])
        region_data = merged[(merged["segment_idx"] >= run_start) &
                             (merged["segment_idx"] <= run_end)]
        regions.append({
            "start": run_start,
            "end": run_end,
            "n_segs": run_end - run_start + 1,
            "time_err": region_data["time_error_s"].sum(),
            "avg_speed_err": (region_data["speed_kmh"] - region_data["tel_speed_kmh"]).mean(),
            "max_speed_err": (region_data["speed_kmh"] - region_data["tel_speed_kmh"]).max(),
            "avg_curv": region_data["curvature"].abs().mean(),
            "dist_start": run_start * 5,
            "dist_end": (run_end + 1) * 5,
        })

    # Sort by time error (most negative first = sim most too fast)
    regions.sort(key=lambda r: r["time_err"])

    total_region_time_err = sum(r["time_err"] for r in regions)
    print(f"\n  Found {len(regions)} problem regions (sim > 8 km/h faster)")
    print(f"  Total time error from these regions: {total_region_time_err:.3f}s per lap")
    print(f"  That's {total_region_time_err / total_time_err * 100:.0f}% of total per-lap error")

    print(f"\n  {'Region':>7} {'Segs':>6} {'Dist':>12} {'TimeErr':>8} {'AvgSpdE':>8} "
          f"{'MaxSpdE':>8} {'AvgCurv':>8}")
    print("  " + "-" * 70)

    for i, r in enumerate(regions):
        print(f"  R{i+1:3d}   "
              f"{r['start']:3d}-{r['end']:3d} "
              f"{r['dist_start']:5d}-{r['dist_end']:5d}m "
              f"{r['time_err']:+7.3f}s "
              f"{r['avg_speed_err']:+7.1f} "
              f"{r['max_speed_err']:+7.1f} "
              f"{r['avg_curv']:7.4f}")

    # ====================================================================
    # SECTION 7: PHYSICS ROOT CAUSE ANALYSIS
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: ROOT CAUSE ANALYSIS -- WHY IS THE SIM FASTER?")
    print("=" * 80)

    # A) Cornering speed: sim corner speed limit vs actual cornering speed
    corner_segs = merged[merged["curvature"].abs() >= 0.005]
    if len(corner_segs) > 0:
        corner_limit_kmh = corner_segs["corner_speed_limit_ms"] * 3.6
        print(f"\n  A) CORNER SPEED LIMITS:")
        print(f"     Corner segments: {len(corner_segs)}")
        print(f"     Sim corner speed limit (mean): {corner_limit_kmh.mean():.1f} km/h")
        print(f"     Sim actual corner speed:        {corner_segs['speed_kmh'].mean():.1f} km/h")
        print(f"     Tel actual corner speed:         {corner_segs['tel_speed_kmh'].mean():.1f} km/h")
        print(f"     Limit vs tel: sim limits are {(corner_limit_kmh.mean() - corner_segs['tel_speed_kmh'].mean()):+.1f} km/h above telemetry")

        # Distribution of how far under the limit the sim runs
        margin_pct = (corner_limit_kmh.values - corner_segs["speed_kmh"].values) / corner_limit_kmh.values * 100
        print(f"     Sim speed margin below limit: {np.mean(margin_pct):.1f}% mean, {np.min(margin_pct):.1f}% min")

    # B) Acceleration phase analysis
    # Find segments where the car is accelerating (speed increasing)
    merged["speed_change_kmh"] = merged["speed_kmh"].diff().fillna(0)
    accel_segs = merged[merged["speed_change_kmh"] > 0.5]
    decel_segs = merged[merged["speed_change_kmh"] < -0.5]
    const_segs = merged[merged["speed_change_kmh"].abs() <= 0.5]

    print(f"\n  B) ACCELERATION PHASES:")
    print(f"     Accelerating segments: {len(accel_segs)}, "
          f"avg speed err: {(accel_segs['speed_kmh'] - accel_segs['tel_speed_kmh']).mean():+.1f} km/h")
    print(f"     Decelerating segments: {len(decel_segs)}, "
          f"avg speed err: {(decel_segs['speed_kmh'] - decel_segs['tel_speed_kmh']).mean():+.1f} km/h")
    print(f"     Constant speed segs:   {len(const_segs)}, "
          f"avg speed err: {(const_segs['speed_kmh'] - const_segs['tel_speed_kmh']).mean():+.1f} km/h")

    # C) Force analysis: what forces are the sim and telemetry seeing?
    print(f"\n  C) FORCE BALANCE:")
    print(f"     Sim mean drive force:      {merged['drive_force_n'].mean():.1f} N")
    print(f"     Sim mean resistance:       {merged['resistance_force_n'].mean():.1f} N")
    print(f"     Sim mean net force:        {merged['net_force_n'].mean():.1f} N")

    # Estimate telemetry implied force from speed changes
    tel_accel_ms2 = np.gradient(merged["tel_speed_ms"].values, merged.index) / (seg_len / merged["tel_speed_ms"].clip(lower=0.5).values)
    tel_net_force = tel_accel_ms2 * config.vehicle.mass_kg
    print(f"     Tel implied mean net force: {np.mean(tel_net_force):.1f} N (from speed changes)")

    # D) Startup/first-segment artifact
    first_few = merged[merged["segment_idx"] < 10]
    first_few_time_err = first_few["time_error_s"].sum()
    print(f"\n  D) STARTUP ARTIFACT:")
    print(f"     First 10 segments time error: {first_few_time_err:+.3f}s")
    print(f"     (Sim starts from 0 speed, telemetry is mid-lap)")

    # E) Coasting behavior
    coast_segs = merged[merged["action"] == "coast"]
    throttle_segs = merged[merged["action"] == "throttle"]

    if len(coast_segs) > 0:
        print(f"\n  E) COASTING BEHAVIOR:")
        print(f"     Coast segments: {len(coast_segs)}")
        coast_speed_err = (coast_segs["speed_kmh"] - coast_segs["tel_speed_kmh"]).mean()
        print(f"     Mean speed error during coast: {coast_speed_err:+.1f} km/h")
        print(f"     Sim coast speed: {coast_segs['speed_kmh'].mean():.1f} km/h")
        print(f"     Tel coast speed: {coast_segs['tel_speed_kmh'].mean():.1f} km/h")

    # ====================================================================
    # SECTION 8: FULL ENDURANCE EXTRAPOLATION
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: FULL ENDURANCE TIME GAP ACCOUNTING")
    print("=" * 80)

    # Run full endurance sim
    result_full = engine.run(num_laps=21, initial_soc_pct=95.0, initial_temp_c=25.0)
    sim_total = result_full.total_time_s

    # Break down telemetry time
    # Total event time from AiM
    first_lap_start = aim_df["Time"].iloc[lap_boundaries[0][0]]
    last_lap_end = aim_df["Time"].iloc[lap_boundaries[-1][1]]
    total_event_time = last_lap_end - first_lap_start

    # Driving time only
    tel_driving = float(driving_time)
    # Driver change gap
    if len(lap_boundaries) > 10:
        d1_end = aim_df["Time"].iloc[lap_boundaries[9][1]]  # end of lap 10
        d2_start = aim_df["Time"].iloc[lap_boundaries[10][0]]  # start of lap 11
        driver_change = d2_start - d1_end
    else:
        driver_change = 0.0

    print(f"\n  Telemetry total event time:   {total_event_time:.1f}s")
    print(f"  Telemetry driving time:       {tel_driving:.1f}s")
    print(f"  Driver change gap:            {driver_change:.1f}s")
    print(f"  Other stopped time:           {(total_event_time - tel_driving - driver_change):.1f}s")
    print(f"\n  Sim total time (21 laps):     {sim_total:.1f}s")
    print(f"  Time gap (tel - sim):         {tel_driving - sim_total:.1f}s")

    # Break down the gap
    per_lap_gap = (tel_driving - sim_total) / 21.0
    print(f"  Per-lap gap:                  {per_lap_gap:.1f}s")

    # Estimate breakdown of per-lap gap by category
    # From Section 4 analysis
    straight_err = merged[merged["segment_type"] == "straight"]["time_error_s"].sum()
    gentle_err = merged[merged["segment_type"] == "gentle_corner"]["time_error_s"].sum()
    medium_err = merged[merged["segment_type"] == "medium_corner"]["time_error_s"].sum()
    tight_err = merged[merged["segment_type"] == "tight_corner"]["time_error_s"].sum()

    print(f"\n  Per-lap error breakdown:")
    print(f"    Straight segments:     {straight_err:+.3f}s")
    print(f"    Gentle corners:        {gentle_err:+.3f}s")
    print(f"    Medium corners:        {medium_err:+.3f}s")
    print(f"    Tight corners:         {tight_err:+.3f}s")
    print(f"    Total per-segment sum: {total_time_err:+.3f}s")

    # Over 21 laps
    print(f"\n  Projected over 21 laps:")
    print(f"    Straight segments:     {straight_err * 21:+.1f}s")
    print(f"    Gentle corners:        {gentle_err * 21:+.1f}s")
    print(f"    Medium corners:        {medium_err * 21:+.1f}s")
    print(f"    Tight corners:         {tight_err * 21:+.1f}s")
    print(f"    Total projected:       {total_time_err * 21:+.1f}s")

    # ====================================================================
    # SECTION 9: SPECIFIC TRACK SECTION DEEP DIVES
    # ====================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: DEEP DIVE INTO WORST TRACK SECTIONS")
    print("=" * 80)

    # Deep dive into each problem region
    for i, r in enumerate(regions[:5]):  # top 5 worst
        start, end = r["start"], r["end"]
        region = merged[(merged["segment_idx"] >= start) & (merged["segment_idx"] <= end)]

        print(f"\n  --- Problem Region {i+1}: segs {start}-{end} "
              f"({r['dist_start']}-{r['dist_end']}m) ---")
        print(f"  Time error: {r['time_err']:+.3f}s | "
              f"Speed error: {r['avg_speed_err']:+.1f} km/h avg, {r['max_speed_err']:+.1f} km/h max")

        # What actions are in this region?
        actions = region["action"].value_counts()
        print(f"  Actions: {dict(actions)}")

        # Curvature profile
        curvatures = region["curvature"].values
        print(f"  Curvature: min={curvatures.min():+.4f}, max={curvatures.max():+.4f}, "
              f"mean={np.mean(np.abs(curvatures)):.4f}")

        # Corner speed limits vs actual
        limits = region["corner_speed_limit_ms"].values * 3.6
        sim_speeds = region["speed_kmh"].values
        tel_speeds = region["tel_speed_kmh"].values
        print(f"  Corner limits: {limits.min():.0f}-{limits.max():.0f} km/h")
        print(f"  Sim speeds:    {sim_speeds.min():.1f}-{sim_speeds.max():.1f} km/h")
        print(f"  Tel speeds:    {tel_speeds.min():.1f}-{tel_speeds.max():.1f} km/h")

        # Force analysis
        print(f"  Sim drive force:  {region['drive_force_n'].mean():.0f} N")
        print(f"  Sim resist force: {region['resistance_force_n'].mean():.0f} N")
        print(f"  Sim net force:    {region['net_force_n'].mean():.0f} N")

        # Diagnosis
        if limits.min() > tel_speeds.max() + 10:
            print(f"  >> DIAGNOSIS: Corner speed limits too high -- Pacejka model with "
                  f"grip_scale={config.tire.grip_scale} allows {limits.min():.0f} km/h "
                  f"but real car only reaches {tel_speeds.max():.0f} km/h")
        elif sim_speeds.mean() > tel_speeds.mean() + 10:
            print(f"  >> DIAGNOSIS: Sim maintains too much speed -- real driver is slower "
                  f"here. May be: (a) driver caution not modeled, (b) missing resistance, "
                  f"(c) driver model overcalibrated throttle in this zone")


if __name__ == "__main__":
    main()
