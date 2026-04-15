"""Diagnose where the 48s time error accumulates in replay mode.

Traces error per lap, per track section, and per action type.
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
    print("=" * 80)
    print("REPLAY MODE TIME ERROR DIAGNOSIS")
    print("=" * 80)

    # ── Load everything ──
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    track = Track.from_telemetry(df=aim_df)

    print(f"Track: {track.num_segments} segments, {track.total_distance_m:.1f} m/lap")

    # ── Run replay simulation ──
    initial_soc = float(aim_df["State of Charge"].iloc[0])
    initial_temp = float(aim_df["Pack Temp"].iloc[0])
    initial_speed = float(aim_df["GPS Speed"].iloc[0]) / 3.6

    total_distance = aim_df["Distance on GPS Speed"].iloc[-1]
    num_laps = round(total_distance / track.total_distance_m)
    print(f"Endurance: {num_laps} laps, {total_distance:.0f} m total")

    replay = ReplayStrategy.from_full_endurance(aim_df, track.lap_distance_m)

    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)
    battery.calibrate_pack_from_telemetry(aim_df)

    engine = SimulationEngine(config, track, replay, battery)
    result = engine.run(
        num_laps=num_laps,
        initial_soc_pct=initial_soc,
        initial_temp_c=initial_temp,
        initial_speed_ms=max(initial_speed, 0.5),
    )
    sim_states = result.states

    print(f"\nSim total time: {result.total_time_s:.2f} s")
    print(f"Sim total distance: {sim_states['distance_m'].iloc[-1]:.1f} m")

    # ── Detect lap boundaries in telemetry ──
    laps_telem = detect_lap_boundaries(aim_df)
    print(f"Telemetry laps detected: {len(laps_telem)}")

    # ── Compute telemetry driving time (speed > 5 km/h) ──
    speed_arr = aim_df["GPS Speed"].values
    time_arr = aim_df["Time"].values
    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    telem_driving_time = float(np.sum(dt_arr[speed_arr > 5]))
    print(f"Telemetry driving time (speed>5): {telem_driving_time:.2f} s")
    print(f"Time error: {result.total_time_s - telem_driving_time:.2f} s")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 1: PER-LAP TIME ERROR TABLE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 1: PER-LAP TIME ERROR")
    print("=" * 80)

    # Compute telemetry lap times using detected boundaries
    telem_lap_times = []
    for start_idx, end_idx, lap_dist in laps_telem:
        # Driving time only (speed > 5 km/h) within this lap
        lap_mask = np.zeros(len(aim_df), dtype=bool)
        lap_mask[start_idx:end_idx] = True
        lap_dt = dt_arr[lap_mask & (speed_arr > 5)]
        lap_time = float(np.sum(lap_dt))
        telem_lap_times.append(lap_time)

    # Sim lap times from states
    sim_lap_times = []
    for lap_num in range(num_laps):
        lap_states = sim_states[sim_states["lap"] == lap_num]
        sim_lap_times.append(float(lap_states["segment_time_s"].sum()))

    # Build comparison table
    n_compare = min(len(telem_lap_times), len(sim_lap_times))

    print(f"\n{'Lap':>4s} | {'Telem(s)':>9s} | {'Sim(s)':>9s} | {'Delta(s)':>9s} | {'CumErr(s)':>9s} | {'%Err':>6s}")
    print("-" * 60)

    cumulative_error = 0.0
    lap_deltas = []
    for i in range(n_compare):
        delta = sim_lap_times[i] - telem_lap_times[i]
        cumulative_error += delta
        pct_err = delta / telem_lap_times[i] * 100 if telem_lap_times[i] > 0 else 0
        lap_deltas.append(delta)
        print(f"{i+1:>4d} | {telem_lap_times[i]:>9.2f} | {sim_lap_times[i]:>9.2f} | {delta:>+9.2f} | {cumulative_error:>+9.2f} | {pct_err:>+5.1f}%")

    # Totals
    total_telem = sum(telem_lap_times[:n_compare])
    total_sim = sum(sim_lap_times[:n_compare])
    print("-" * 60)
    print(f"{'SUM':>4s} | {total_telem:>9.2f} | {total_sim:>9.2f} | {total_sim - total_telem:>+9.2f} |")

    # Any laps not covered?
    if len(sim_lap_times) > n_compare:
        uncovered_sim = sum(sim_lap_times[n_compare:])
        print(f"\n  Sim laps {n_compare+1}-{len(sim_lap_times)}: {uncovered_sim:.2f} s (no telem boundary match)")
    if len(telem_lap_times) > n_compare:
        uncovered_telem = sum(telem_lap_times[n_compare:])
        print(f"\n  Telem laps {n_compare+1}-{len(telem_lap_times)}: {uncovered_telem:.2f} s (no sim boundary match)")

    # Uniformity check
    if len(lap_deltas) >= 3:
        mean_delta = np.mean(lap_deltas)
        std_delta = np.std(lap_deltas)
        print(f"\n  Mean delta: {mean_delta:+.2f} s/lap, Std: {std_delta:.2f} s")
        print(f"  CoV: {std_delta/abs(mean_delta)*100:.0f}%" if abs(mean_delta) > 0.01 else "  CoV: N/A (mean~0)")
        print(f"  First 5 laps mean: {np.mean(lap_deltas[:5]):+.2f} s")
        print(f"  Last 5 laps mean:  {np.mean(lap_deltas[-5:]):+.2f} s")
        trend = np.polyfit(range(len(lap_deltas)), lap_deltas, 1)
        print(f"  Linear trend: {trend[0]:+.3f} s/lap (drift)")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 2: PER-SECTION BREAKDOWN FOR REPRESENTATIVE LAP
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 2: PER-SECTION BREAKDOWN (Lap 5)")
    print("=" * 80)

    rep_lap = 4  # 0-indexed, so lap 5
    if rep_lap < n_compare:
        # Telemetry for this lap
        t_start_idx, t_end_idx, t_lap_dist = laps_telem[rep_lap]
        telem_lap = aim_df.iloc[t_start_idx:t_end_idx].copy()
        telem_dist = telem_lap["Distance on GPS Speed"].values
        telem_base_dist = telem_dist[0]
        telem_lap_local_dist = telem_dist - telem_base_dist
        telem_time_vals = telem_lap["Time"].values
        telem_speed_vals = telem_lap["GPS Speed"].values
        telem_dt = np.diff(telem_time_vals, prepend=telem_time_vals[0])

        # Sim for this lap
        sim_lap = sim_states[sim_states["lap"] == rep_lap].copy()
        sim_dist_in_lap = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]

        lap_length = track.total_distance_m
        n_sections = 10
        section_size = lap_length / n_sections

        print(f"\n  Lap distance: {lap_length:.1f} m ({n_sections} sections of {section_size:.0f} m)")
        print(f"\n{'Section':>8s} | {'Dist(m)':>12s} | {'Telem(s)':>9s} | {'Sim(s)':>9s} | {'Delta(s)':>9s} | {'Type':>12s}")
        print("-" * 70)

        for sec in range(n_sections):
            d_lo = sec * section_size
            d_hi = (sec + 1) * section_size

            # Telemetry time in this section
            sec_mask = (telem_lap_local_dist >= d_lo) & (telem_lap_local_dist < d_hi) & (telem_speed_vals > 5)
            telem_sec_time = float(np.sum(telem_dt[sec_mask]))

            # Sim time in this section
            sim_sec_mask = (sim_dist_in_lap >= d_lo) & (sim_dist_in_lap < d_hi)
            sim_sec_time = float(sim_lap.loc[sim_sec_mask, "segment_time_s"].sum())

            # Characterize section: straight or corner
            seg_start = int(d_lo / 5)
            seg_end = min(int(d_hi / 5), track.num_segments - 1)
            curvatures = [abs(track.segments[s].curvature) for s in range(seg_start, seg_end + 1)]
            mean_curv = np.mean(curvatures) if curvatures else 0
            if mean_curv > 0.02:
                sec_type = "CORNER"
            elif mean_curv > 0.005:
                sec_type = "TRANSITION"
            else:
                sec_type = "STRAIGHT"

            delta = sim_sec_time - telem_sec_time
            print(f"{sec+1:>8d} | {d_lo:>5.0f}-{d_hi:>5.0f} | {telem_sec_time:>9.2f} | {sim_sec_time:>9.2f} | {delta:>+9.2f} | {sec_type:>12s}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 3: ERROR BY ACTION TYPE (FULL ENDURANCE)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 3: TIME BY ACTION TYPE (Full Endurance)")
    print("=" * 80)

    # Sim breakdown by action
    for action_name in ["throttle", "coast", "brake"]:
        mask = sim_states["action"] == action_name
        n_segs = mask.sum()
        sim_time = float(sim_states.loc[mask, "segment_time_s"].sum())
        avg_speed = float(sim_states.loc[mask, "speed_kmh"].mean()) if n_segs > 0 else 0
        print(f"  {action_name.upper():>10s}: {n_segs:>5d} segments, {sim_time:>8.2f} s, avg {avg_speed:.1f} km/h")

    # Try to classify telemetry into action types
    print("\n  Telemetry action classification (throttle>5%, brake>2 bar, else coast):")
    telem_throttle_mask = aim_df["Throttle Pos"].values > 5.0
    telem_brake_raw = np.maximum(aim_df["FBrakePressure"].values, aim_df["RBrakePressure"].values)
    telem_brake_mask = telem_brake_raw > 2.0
    telem_coast_mask = ~telem_throttle_mask & ~telem_brake_mask
    moving = speed_arr > 5.0

    for name, mask in [("THROTTLE", telem_throttle_mask), ("COAST", telem_coast_mask), ("BRAKE", telem_brake_mask)]:
        combined = mask & moving
        t = float(np.sum(dt_arr[combined]))
        n = int(combined.sum())
        print(f"  {name:>10s}: {n:>5d} samples, {t:>8.2f} s")

    # Now compare sim vs telemetry by action type using distance-binned approach
    print("\n  Distance-binned action comparison (5m bins across full endurance):")
    total_dist = sim_states["distance_m"].iloc[-1]
    bin_size = 5.0
    n_bins = int(total_dist / bin_size)

    # Build telemetry distance-to-action mapping
    telem_dist_arr = aim_df["Distance on GPS Speed"].values
    telem_throttle_arr = aim_df["Throttle Pos"].values
    telem_brake_arr = np.maximum(aim_df["FBrakePressure"].values, aim_df["RBrakePressure"].values)
    telem_speed_ms_arr = speed_arr / 3.6

    # For each action type, compute total time in sim vs telemetry
    action_time_sim = {"throttle": 0.0, "coast": 0.0, "brake": 0.0}
    action_time_telem = {"throttle": 0.0, "coast": 0.0, "brake": 0.0}
    action_count = {"throttle": 0, "coast": 0, "brake": 0}

    # Group sim states
    for _, row in sim_states.iterrows():
        act = row["action"]
        action_time_sim[act] += row["segment_time_s"]

    # For telemetry, match to sim action at same distance
    # This is more robust: bin telemetry by distance, check action at that distance
    for i in range(len(sim_states)):
        row = sim_states.iloc[i]
        d = row["distance_m"]
        act = row["action"]

        # Find telemetry samples in the same distance bin
        d_lo = d
        d_hi = d + 5.0  # segment is 5m
        telem_mask = (telem_dist_arr >= d_lo) & (telem_dist_arr < d_hi) & (speed_arr > 5)
        if telem_mask.any():
            telem_time_in_bin = float(np.sum(dt_arr[telem_mask]))
            action_time_telem[act] += telem_time_in_bin
            action_count[act] += 1

    print(f"\n{'Action':>10s} | {'SimTime(s)':>10s} | {'TelemTime(s)':>12s} | {'Delta(s)':>9s} | {'Segments':>8s}")
    print("-" * 60)
    total_delta_by_action = 0.0
    for act in ["throttle", "coast", "brake"]:
        delta = action_time_sim[act] - action_time_telem[act]
        total_delta_by_action += delta
        print(f"{act.upper():>10s} | {action_time_sim[act]:>10.2f} | {action_time_telem[act]:>12.2f} | {delta:>+9.2f} | {action_count[act]:>8d}")
    print("-" * 60)
    print(f"{'TOTAL':>10s} | {sum(action_time_sim.values()):>10.2f} | {sum(action_time_telem.values()):>12.2f} | {total_delta_by_action:>+9.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 4: SPEED COMPARISON AT KEY POINTS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 4: SPEED COMPARISON (Lap 5 detail)")
    print("=" * 80)

    if rep_lap < n_compare:
        # For each sim segment in lap 5, compare speed to telemetry at that distance
        sim_lap5 = sim_states[sim_states["lap"] == rep_lap].copy()
        t_start_idx5, t_end_idx5, _ = laps_telem[rep_lap]
        telem5 = aim_df.iloc[t_start_idx5:t_end_idx5].copy()
        telem5_dist = telem5["Distance on GPS Speed"].values - telem5["Distance on GPS Speed"].values[0]
        telem5_speed = telem5["GPS Speed"].values  # km/h

        sim5_dist = sim_lap5["distance_m"].values - sim_lap5["distance_m"].values[0]
        sim5_speed = sim_lap5["speed_kmh"].values
        sim5_action = sim_lap5["action"].values
        sim5_curv = sim_lap5["curvature"].values

        # Interpolate telemetry speed at sim distances
        from scipy.interpolate import interp1d
        telem_speed_interp = interp1d(telem5_dist, telem5_speed, bounds_error=False, fill_value="extrapolate")
        telem_at_sim_dist = telem_speed_interp(sim5_dist)

        speed_delta = sim5_speed - telem_at_sim_dist

        # Show segments with biggest speed differences
        sorted_idx = np.argsort(np.abs(speed_delta))[::-1]

        print(f"\n  Top 20 segments with largest speed discrepancy (Lap 5):")
        print(f"{'Seg':>5s} | {'Dist(m)':>8s} | {'SimSpd':>8s} | {'TelemSpd':>8s} | {'Delta':>8s} | {'Action':>8s} | {'Curv':>8s}")
        print("-" * 70)
        for rank in range(min(20, len(sorted_idx))):
            idx = sorted_idx[rank]
            print(f"{int(sim_lap5.iloc[idx]['segment_idx']):>5d} | {sim5_dist[idx]:>8.1f} | {sim5_speed[idx]:>7.1f} | {telem_at_sim_dist[idx]:>7.1f} | {speed_delta[idx]:>+7.1f} | {sim5_action[idx]:>8s} | {sim5_curv[idx]:>8.4f}")

        # Average speed delta by action type
        print(f"\n  Average speed delta by action (Lap 5):")
        for act in ["throttle", "coast", "brake"]:
            mask = sim5_action == act
            if mask.any():
                avg_d = float(np.mean(speed_delta[mask]))
                avg_abs_d = float(np.mean(np.abs(speed_delta[mask])))
                print(f"    {act.upper():>10s}: mean delta = {avg_d:+.2f} km/h, mean |delta| = {avg_abs_d:.2f} km/h, n={mask.sum()}")

        # Speed delta by curvature bin
        print(f"\n  Speed delta by curvature (Lap 5):")
        curv_abs = np.abs(sim5_curv)
        for label, lo, hi in [("Straight", 0, 0.005), ("Gentle", 0.005, 0.02), ("Medium", 0.02, 0.05), ("Tight", 0.05, 1.0)]:
            mask = (curv_abs >= lo) & (curv_abs < hi)
            if mask.any():
                avg_d = float(np.mean(speed_delta[mask]))
                avg_seg_time_sim = float(np.mean(sim_lap5.iloc[np.where(mask)[0]]["segment_time_s"].values))
                print(f"    {label:>10s} (k={lo:.3f}-{hi:.3f}): mean speed delta = {avg_d:+.2f} km/h, n={mask.sum()}, avg seg time = {avg_seg_time_sim:.3f} s")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 5: DISTANCE DRIFT CHECK
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 5: DISTANCE DRIFT CHECK")
    print("=" * 80)

    # Compare cumulative distance at each lap boundary
    print(f"\n  Track lap distance: {track.total_distance_m:.1f} m")
    for i in range(n_compare):
        sim_lap_end_dist = float(sim_states[sim_states["lap"] == i]["distance_m"].iloc[-1])
        expected_sim_dist = (i + 1) * track.total_distance_m
        telem_lap_dist = laps_telem[i][2]  # lap distance from detect_lap_boundaries
        print(f"  Lap {i+1:>2d}: sim dist = {sim_lap_end_dist:>9.1f} m (expected {expected_sim_dist:.1f}), telem lap dist = {telem_lap_dist:.1f} m")

    # Compare total
    sim_total_dist_end = float(sim_states["distance_m"].iloc[-1])
    telem_total_dist = float(aim_df["Distance on GPS Speed"].iloc[-1])
    print(f"\n  Sim total:   {sim_total_dist_end:.1f} m ({num_laps} laps x {track.total_distance_m:.1f} m = {num_laps * track.total_distance_m:.1f} m)")
    print(f"  Telem total: {telem_total_dist:.1f} m")
    print(f"  Difference:  {sim_total_dist_end - telem_total_dist:+.1f} m")

    # If track lap length differs from telemetry average lap length
    avg_telem_lap = np.mean([l[2] for l in laps_telem])
    print(f"\n  Mean telem lap distance: {avg_telem_lap:.1f} m")
    print(f"  Track model lap distance: {track.total_distance_m:.1f} m")
    print(f"  Mismatch per lap: {track.total_distance_m - avg_telem_lap:+.1f} m")
    print(f"  Over {num_laps} laps: {num_laps * (track.total_distance_m - avg_telem_lap):+.1f} m total distance drift")

    # What would the time impact be?
    avg_speed_ms = float(sim_states["speed_ms"].mean())
    dist_mismatch_total = num_laps * (track.total_distance_m - avg_telem_lap)
    time_from_dist = dist_mismatch_total / avg_speed_ms
    print(f"  At avg sim speed of {avg_speed_ms:.2f} m/s ({avg_speed_ms*3.6:.1f} km/h), distance drift accounts for ~{time_from_dist:+.1f} s")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION 6: SEGMENT TIME DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 6: SEGMENT TIME STATISTICS")
    print("=" * 80)

    seg_times = sim_states["segment_time_s"].values
    print(f"  Segment time: min={seg_times.min():.4f} s, max={seg_times.max():.4f} s, mean={seg_times.mean():.4f} s, median={np.median(seg_times):.4f} s")
    print(f"  Segments with time > 1.0 s: {(seg_times > 1.0).sum()} ({(seg_times > 1.0).sum() / len(seg_times) * 100:.1f}%)")
    print(f"  Segments with time > 2.0 s: {(seg_times > 2.0).sum()}")

    # Slowest segments
    slowest_idx = np.argsort(seg_times)[::-1][:10]
    print(f"\n  10 slowest segments:")
    print(f"{'Row':>5s} | {'Lap':>4s} | {'Seg':>5s} | {'Time(s)':>8s} | {'Speed':>8s} | {'Action':>8s} | {'Curv':>8s}")
    print("-" * 55)
    for idx in slowest_idx:
        r = sim_states.iloc[idx]
        print(f"{idx:>5d} | {int(r['lap']):>4d} | {int(r['segment_idx']):>5d} | {r['segment_time_s']:>8.4f} | {r['speed_kmh']:>7.1f} | {r['action']:>8s} | {r['curvature']:>8.4f}")


if __name__ == "__main__":
    main()
