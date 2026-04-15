"""Braking behavior comparison: sim (regen only) vs telemetry (regen + hydraulic).

Quantifies how much deceleration the sim is MISSING because it only models
regenerative braking through the motor, while the real car has hydraulic
disc brakes that provide additional stopping force.

Also checks coast deceleration: is the real car decelerating more during
coast than the sim's aero drag + rolling resistance model predicts?
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


MASS_KG = 288.0  # car + driver


def identify_braking_events(df, brake_col="RBrakePressure", min_pressure=0.5,
                            min_duration_samples=3, speed_col="GPS Speed"):
    """Find contiguous braking events in telemetry.

    Returns list of dicts with start_idx, end_idx, duration, peak_pressure,
    entry_speed, exit_speed, mean_decel, distance_start, distance_end.
    """
    brake = df[brake_col].values
    speed_kmh = df[speed_col].values
    speed_ms = speed_kmh / 3.6
    lon_acc = df["GPS LonAcc"].values  # in g
    dist = df["Distance on GPS Speed"].values
    time = df["Time"].values

    is_braking = brake > min_pressure
    events = []
    i = 0
    while i < len(is_braking):
        if is_braking[i]:
            start = i
            while i < len(is_braking) and is_braking[i]:
                i += 1
            end = i  # exclusive
            if (end - start) >= min_duration_samples:
                events.append({
                    "start_idx": start,
                    "end_idx": end,
                    "duration_samples": end - start,
                    "duration_s": time[end - 1] - time[start],
                    "peak_pressure_bar": float(np.max(brake[start:end])),
                    "mean_pressure_bar": float(np.mean(brake[start:end])),
                    "entry_speed_ms": float(speed_ms[start]),
                    "exit_speed_ms": float(speed_ms[end - 1]),
                    "speed_drop_ms": float(speed_ms[start] - speed_ms[end - 1]),
                    "mean_decel_g": float(np.mean(lon_acc[start:end])),
                    "min_decel_g": float(np.min(lon_acc[start:end])),
                    "distance_start_m": float(dist[start]),
                    "distance_end_m": float(dist[end - 1]),
                    "distance_span_m": float(dist[end - 1] - dist[start]),
                })
        else:
            i += 1
    return events


def main():
    print("=" * 90)
    print("BRAKING BEHAVIOR ANALYSIS: SIM (regen only) vs TELEMETRY (regen + hydraulic)")
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
    track = Track.from_telemetry(df=aim_df)
    lap_dist = track.total_distance_m
    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)

    print(f"  Track: {track.num_segments} segments, {lap_dist:.1f}m per lap")

    # ================================================================
    # RUN REPLAY SIM
    # ================================================================
    print("\n[2] Running 22-lap replay sim...")
    strategy = ReplayStrategy.from_full_endurance(aim_df, lap_dist)
    engine = SimulationEngine(config, track, strategy, battery)
    result = engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=29.0)
    sim_df = result.states

    # Telemetry reference (moving samples only)
    tel = aim_df[aim_df["GPS Speed"] > 5.0].copy()

    print(f"  Sim: {result.total_time_s:.1f}s")
    tel_time = tel["Time"].values
    tel_dt = np.diff(tel_time, prepend=tel_time[0])
    total_tel_time = float(np.sum(tel_dt))
    print(f"  Telemetry (moving): {total_tel_time:.1f}s")
    print(f"  Time gap: {total_tel_time - result.total_time_s:.1f}s (sim finishes faster)")

    # ================================================================
    # [3] IDENTIFY BRAKING EVENTS IN TELEMETRY
    # ================================================================
    print("\n" + "=" * 90)
    print("[3] TELEMETRY BRAKING EVENTS")
    print("=" * 90)

    # FBrakePressure is bad (all negative ~-18), use RBrakePressure only
    events = identify_braking_events(tel, brake_col="RBrakePressure",
                                     min_pressure=0.5, min_duration_samples=3)

    print(f"\n  Total braking events (RBrakePressure > 0.5 bar, >= 3 samples): {len(events)}")
    total_brake_time = sum(e["duration_s"] for e in events)
    total_brake_dist = sum(e["distance_span_m"] for e in events)
    print(f"  Total braking time: {total_brake_time:.1f}s")
    print(f"  Total braking distance: {total_brake_dist:.1f}m")

    # Stats
    decels_g = np.array([e["mean_decel_g"] for e in events])
    peak_pressures = np.array([e["peak_pressure_bar"] for e in events])
    speed_drops = np.array([e["speed_drop_ms"] for e in events])
    durations = np.array([e["duration_s"] for e in events])

    print(f"\n  Deceleration (GPS LonAcc, g):")
    print(f"    Mean across events: {np.mean(decels_g):.3f} g")
    print(f"    Min (hardest braking): {np.min(decels_g):.3f} g")
    print(f"    Max (lightest braking): {np.max(decels_g):.3f} g")
    print(f"    Median: {np.median(decels_g):.3f} g")

    print(f"\n  Brake pressure (bar):")
    print(f"    Mean peak: {np.mean(peak_pressures):.2f} bar")
    print(f"    Max peak: {np.max(peak_pressures):.2f} bar")

    print(f"\n  Speed drop per event (m/s):")
    print(f"    Mean: {np.mean(speed_drops):.2f} m/s  ({np.mean(speed_drops)*3.6:.1f} km/h)")
    print(f"    Max: {np.max(speed_drops):.2f} m/s  ({np.max(speed_drops)*3.6:.1f} km/h)")
    print(f"    Total speed-loss across all events: {np.sum(speed_drops):.1f} m/s")

    # ================================================================
    # [4] SIM DECELERATION DURING BRAKING AT SAME DISTANCES
    # ================================================================
    print("\n" + "=" * 90)
    print("[4] SIM vs TELEMETRY DECELERATION DURING BRAKING")
    print("=" * 90)

    sim_dist = sim_df["distance_m"].values
    sim_speed = sim_df["speed_ms"].values
    sim_regen = sim_df["regen_force_n"].values
    sim_resist = sim_df["resistance_force_n"].values
    sim_action = sim_df["action"].values
    sim_brake_pct = sim_df["brake_pct"].values

    # For each telemetry braking event, find the corresponding sim segment
    # and compare deceleration
    comparisons = []
    for ev in events:
        d_start = ev["distance_start_m"]
        d_end = ev["distance_end_m"]

        # Find sim segments overlapping this distance range
        # Sim distance wraps per lap, but for full endurance replay it's cumulative
        mask = (sim_dist >= d_start - 5) & (sim_dist <= d_end + 5)
        if not np.any(mask):
            continue

        sim_regen_here = sim_regen[mask]
        sim_resist_here = sim_resist[mask]
        sim_speed_here = sim_speed[mask]
        sim_action_here = sim_action[mask]
        sim_brake_here = sim_brake_pct[mask]

        # Sim deceleration: (regen + resistance) / mass (both oppose motion)
        # regen_force_n is negative (opposing), so total decel force = abs(regen) + resist
        sim_decel_force = np.abs(sim_regen_here) + sim_resist_here
        sim_decel_g = np.mean(sim_decel_force) / (MASS_KG * 9.81)

        # Telemetry deceleration (GPS LonAcc is in g, negative = decelerating)
        tel_decel_g = abs(ev["mean_decel_g"])

        # How much is the sim classified as brake vs coast?
        n_brake_sim = np.sum(sim_action_here == "brake")
        n_coast_sim = np.sum(sim_action_here == "coast")
        n_throttle_sim = np.sum(sim_action_here == "throttle")
        total_sim = len(sim_action_here)

        comparisons.append({
            "tel_decel_g": tel_decel_g,
            "sim_decel_g": sim_decel_g,
            "missing_decel_g": tel_decel_g - sim_decel_g,
            "tel_speed_entry": ev["entry_speed_ms"],
            "tel_speed_drop": ev["speed_drop_ms"],
            "tel_pressure_bar": ev["peak_pressure_bar"],
            "tel_duration_s": ev["duration_s"],
            "sim_brake_frac": n_brake_sim / total_sim if total_sim > 0 else 0,
            "sim_coast_frac": n_coast_sim / total_sim if total_sim > 0 else 0,
            "mean_regen_force_n": float(np.mean(np.abs(sim_regen_here))),
            "mean_resist_force_n": float(np.mean(sim_resist_here)),
        })

    comp_df = pd.DataFrame(comparisons)

    print(f"\n  Matched braking events: {len(comp_df)}")
    print(f"\n  Telemetry deceleration (abs, g):")
    print(f"    Mean: {comp_df['tel_decel_g'].mean():.4f} g  ({comp_df['tel_decel_g'].mean()*9.81:.3f} m/s2)")
    print(f"    Median: {comp_df['tel_decel_g'].median():.4f} g")
    print(f"    Max: {comp_df['tel_decel_g'].max():.4f} g  ({comp_df['tel_decel_g'].max()*9.81:.3f} m/s2)")

    print(f"\n  Sim deceleration during same intervals (abs, g):")
    print(f"    Mean: {comp_df['sim_decel_g'].mean():.4f} g  ({comp_df['sim_decel_g'].mean()*9.81:.3f} m/s2)")
    print(f"    Median: {comp_df['sim_decel_g'].median():.4f} g")

    print(f"\n  MISSING deceleration (telemetry - sim, g):")
    print(f"    Mean: {comp_df['missing_decel_g'].mean():.4f} g  ({comp_df['missing_decel_g'].mean()*9.81:.3f} m/s2)")
    print(f"    Median: {comp_df['missing_decel_g'].median():.4f} g")
    print(f"    Events where sim < telemetry: {(comp_df['missing_decel_g'] > 0).sum()} / {len(comp_df)}")

    print(f"\n  Sim action classification during telemetry braking:")
    print(f"    Mean brake fraction: {comp_df['sim_brake_frac'].mean()*100:.1f}%")
    print(f"    Mean coast fraction: {comp_df['sim_coast_frac'].mean()*100:.1f}%")
    print(f"    Events fully classified as coast in sim: "
          f"{(comp_df['sim_brake_frac'] == 0).sum()} / {len(comp_df)}")

    print(f"\n  Sim forces during braking events:")
    print(f"    Mean regen force: {comp_df['mean_regen_force_n'].mean():.1f} N")
    print(f"    Mean resistance force: {comp_df['mean_resist_force_n'].mean():.1f} N")

    # ================================================================
    # [5] BELOW-THRESHOLD BRAKING (light braking classified as coast)
    # ================================================================
    print("\n" + "=" * 90)
    print("[5] LIGHT BRAKING BELOW DETECTION THRESHOLD")
    print("=" * 90)

    rb = tel["RBrakePressure"].values
    brake_raw = rb.copy()  # FBrakePressure is bad, use R only
    bmax = float(np.percentile(brake_raw[brake_raw > 0], 99)) if np.any(brake_raw > 0) else 1.0
    brake_norm = np.clip(brake_raw / bmax, 0, 1)

    # ReplayStrategy threshold is 0.05 on normalized brake
    n_total = len(brake_norm)
    n_any_brake = np.sum(brake_raw > 0)
    n_above_thresh = np.sum(brake_norm > 0.05)
    n_below_thresh = np.sum((brake_raw > 0) & (brake_norm <= 0.05))

    print(f"\n  Brake normalization: bmax = {bmax:.2f} bar (P99 of positive values)")
    print(f"  Total samples: {n_total}")
    print(f"  Any brake pressure > 0: {n_any_brake} ({100*n_any_brake/n_total:.1f}%)")
    print(f"  Above 0.05 threshold (sim sees as BRAKE): {n_above_thresh} ({100*n_above_thresh/n_total:.1f}%)")
    print(f"  Below 0.05 threshold (sim sees as COAST): {n_below_thresh} ({100*n_below_thresh/n_total:.1f}%)")

    # What deceleration do these light-braking samples have?
    light_brake_mask = (brake_raw > 0) & (brake_norm <= 0.05)
    if np.sum(light_brake_mask) > 0:
        light_decel = tel["GPS LonAcc"].values[light_brake_mask]
        light_speed = tel["GPS Speed"].values[light_brake_mask] / 3.6
        print(f"\n  Light-braking samples (brake > 0 but below threshold):")
        print(f"    Count: {np.sum(light_brake_mask)}")
        print(f"    Mean GPS LonAcc: {np.mean(light_decel):.4f} g ({np.mean(light_decel)*9.81:.3f} m/s2)")
        print(f"    Mean speed: {np.mean(light_speed):.1f} m/s ({np.mean(light_speed)*3.6:.1f} km/h)")
        print(f"    Mean brake pressure: {np.mean(brake_raw[light_brake_mask]):.3f} bar")
        print(f"    Pressure range: {np.min(brake_raw[light_brake_mask]):.3f} - "
              f"{np.max(brake_raw[light_brake_mask]):.3f} bar")

    # ================================================================
    # [6] COAST DECELERATION: TELEMETRY vs SIM
    # ================================================================
    print("\n" + "=" * 90)
    print("[6] COAST DECELERATION: TELEMETRY vs SIM")
    print("=" * 90)

    throttle = tel["Throttle Pos"].values
    speed_ms_tel = tel["GPS Speed"].values / 3.6

    # Coast in telemetry: throttle < 5%, brake < threshold, speed > 5 m/s
    coast_mask = (throttle < 5.0) & (brake_raw < 0.5) & (speed_ms_tel > 5.0)
    n_coast = np.sum(coast_mask)

    print(f"\n  Coast definition: throttle < 5%, brake < 0.5 bar, speed > 5 m/s")
    print(f"  Coast samples in telemetry: {n_coast} ({100*n_coast/n_total:.1f}%)")

    if n_coast > 0:
        coast_decel_g = tel["GPS LonAcc"].values[coast_mask]
        coast_speeds = speed_ms_tel[coast_mask]

        print(f"\n  Telemetry coast deceleration (GPS LonAcc, g):")
        print(f"    Mean: {np.mean(coast_decel_g):.4f} g ({np.mean(coast_decel_g)*9.81:.3f} m/s2)")
        print(f"    Median: {np.median(coast_decel_g):.4f} g")
        print(f"    Std: {np.std(coast_decel_g):.4f} g")

        # Speed-binned coast decel
        speed_bins = [(5, 10), (10, 15), (15, 20)]
        print(f"\n  Coast deceleration by speed bin:")
        for lo, hi in speed_bins:
            bin_mask = (coast_speeds >= lo) & (coast_speeds < hi)
            if np.sum(bin_mask) > 5:
                bin_decel = coast_decel_g[bin_mask]
                avg_speed = np.mean(coast_speeds[bin_mask])
                print(f"    {lo}-{hi} m/s ({lo*3.6:.0f}-{hi*3.6:.0f} km/h): "
                      f"mean {np.mean(bin_decel):.4f} g ({np.mean(bin_decel)*9.81:.3f} m/s2), "
                      f"n={np.sum(bin_mask)}")

        # Sim coast: what deceleration does the sim produce during coast?
        sim_coast_mask = sim_df["action"].values == "coast"
        if np.sum(sim_coast_mask) > 0:
            sim_coast_resist = sim_df["resistance_force_n"].values[sim_coast_mask]
            sim_coast_speed = sim_df["speed_ms"].values[sim_coast_mask]
            sim_coast_decel_g = sim_coast_resist / (MASS_KG * 9.81)

            print(f"\n  Sim coast deceleration (resistance / mass):")
            print(f"    Mean: {np.mean(sim_coast_decel_g):.4f} g ({np.mean(sim_coast_decel_g)*9.81:.3f} m/s2)")
            print(f"    Median: {np.median(sim_coast_decel_g):.4f} g")

            for lo, hi in speed_bins:
                bin_mask = (sim_coast_speed >= lo) & (sim_coast_speed < hi)
                if np.sum(bin_mask) > 5:
                    print(f"    {lo}-{hi} m/s: "
                          f"mean {np.mean(sim_coast_decel_g[bin_mask]):.4f} g "
                          f"({np.mean(sim_coast_decel_g[bin_mask])*9.81:.3f} m/s2), "
                          f"n={np.sum(bin_mask)}")

            # Compare at similar speeds
            print(f"\n  Coast decel gap (telemetry - sim) at matched speeds:")
            for lo, hi in speed_bins:
                tel_bin = coast_decel_g[(coast_speeds >= lo) & (coast_speeds < hi)]
                sim_bin = sim_coast_decel_g[(sim_coast_speed >= lo) & (sim_coast_speed < hi)]
                if len(tel_bin) > 5 and len(sim_bin) > 5:
                    gap = abs(np.mean(tel_bin)) - np.mean(sim_bin)
                    print(f"    {lo}-{hi} m/s: telemetry {abs(np.mean(tel_bin)):.4f} g, "
                          f"sim {np.mean(sim_bin):.4f} g, "
                          f"gap = {gap:.4f} g ({gap*9.81:.3f} m/s2)")

    # ================================================================
    # [7] TIME IMPACT ESTIMATE
    # ================================================================
    print("\n" + "=" * 90)
    print("[7] TIME IMPACT OF MISSING HYDRAULIC BRAKING")
    print("=" * 90)

    # Method 1: From braking events
    # If the sim decelerates less, it arrives at the corner exit at a higher speed
    # and spends less time decelerating. For each braking event:
    #   real decel distance: d = v0^2 / (2*a_real)  (simplified)
    #   sim decel distance: d = v0^2 / (2*a_sim)
    #   time = 2*d / (v0 + v_exit)
    total_time_diff_braking = 0.0
    for _, row in comp_df.iterrows():
        v0 = row["tel_speed_entry"]
        tel_a = row["tel_decel_g"] * 9.81  # m/s2, magnitude
        sim_a = row["sim_decel_g"] * 9.81
        duration = row["tel_duration_s"]

        if tel_a > 0.01 and sim_a > 0.01:
            # Speed at end of braking zone in telemetry
            v_exit_tel = max(0.1, v0 - tel_a * duration)
            # Speed at end of same distance in sim (less decel -> higher exit speed)
            v_exit_sim = max(0.1, v0 - sim_a * duration)

            # Time through this zone: distance / avg_speed
            dist_zone = row["tel_speed_drop"]  # this is speed drop, need distance
            # Use distance from event
            d_zone = (comp_df.iloc[0]["tel_duration_s"] if "distance_span_m" not in row.index
                      else 0)

        # Simpler approach: time = distance / avg_speed
        # With lower decel, avg speed is higher, so time is shorter
        if tel_a > 0.01 and sim_a > 0.01 and duration > 0.01:
            v_exit_tel = max(0.1, v0 - tel_a * duration)
            v_exit_sim = max(0.1, v0 - sim_a * duration)
            dist_zone = v0 * duration - 0.5 * tel_a * duration**2
            if dist_zone > 0:
                time_tel = dist_zone / ((v0 + v_exit_tel) / 2)
                time_sim = dist_zone / ((v0 + v_exit_sim) / 2)
                total_time_diff_braking += (time_tel - time_sim)

    print(f"\n  Method 1: Per-event braking time deficit")
    print(f"    Total time sim is faster during braking zones: {total_time_diff_braking:.2f}s")

    # Method 2: Global approach
    # Total braking time in telemetry vs sim
    sim_brake_mask = sim_df["action"].values == "brake"
    sim_brake_time = sim_df["segment_time_s"].values[sim_brake_mask].sum()
    sim_coast_time = sim_df["segment_time_s"].values[sim_coast_mask].sum()

    print(f"\n  Method 2: Global time breakdown")
    print(f"    Sim segments classified as BRAKE: {np.sum(sim_brake_mask)}")
    print(f"    Sim total brake time: {sim_brake_time:.1f}s")
    print(f"    Sim total coast time: {sim_coast_time:.1f}s")
    print(f"    Sim total throttle time: "
          f"{sim_df['segment_time_s'].values[sim_df['action'].values == 'throttle'].sum():.1f}s")

    # Method 3: Missing decel force -> speed -> time
    # Average missing deceleration across all braking events
    mean_missing_g = comp_df["missing_decel_g"].mean()
    mean_missing_ms2 = mean_missing_g * 9.81

    # Over total braking distance, how much extra speed does the sim carry?
    # delta_v = a_missing * t_braking
    total_brake_time_tel = sum(e["duration_s"] for e in events)
    extra_speed = mean_missing_ms2 * total_brake_time_tel
    print(f"\n  Method 3: Missing deceleration -> speed accumulation")
    print(f"    Mean missing decel: {mean_missing_g:.4f} g ({mean_missing_ms2:.3f} m/s2)")
    print(f"    Total braking time in telemetry: {total_brake_time_tel:.1f}s")
    print(f"    Cumulative extra speed from missing decel: {extra_speed:.1f} m/s ({extra_speed*3.6:.1f} km/h)")
    print(f"    If this extra speed means the sim doesn't slow enough,")
    print(f"    the sim traverses corners faster -> finishes sooner")

    # Direct estimate: for every braking second, sim accumulates
    # mean_missing_ms2 * dt extra velocity. This extra velocity means
    # each subsequent meter is traversed in less time.
    # delta_t = delta_d / delta_v ~ proportional to speed ratio
    avg_tel_speed = tel["GPS Speed"].values[tel["GPS Speed"].values > 5].mean() / 3.6
    total_distance_m = tel["Distance on GPS Speed"].values[-1]
    # If average speed is higher by some amount, total time is:
    # t = D / v_avg. dt/dv = -D/v^2
    # So time saved per m/s of extra avg speed: D/v^2

    # From braking events: total speed*time "impulse" that's missing
    # Each event: the sim exits with (sim_a / tel_a) * tel_exit_speed
    # The extra speed carries until the next throttle application
    # Rough estimate: extra speed decays over coast distance
    total_excess_speed_x_time = 0
    for _, row in comp_df.iterrows():
        if row["tel_decel_g"] > 0.01 and row["sim_decel_g"] > 0.01:
            excess_v = (row["tel_decel_g"] - row["sim_decel_g"]) * 9.81 * row["tel_duration_s"]
            # This excess speed persists for some distance (assume coast decay ~5s)
            decay_time = 3.0
            total_excess_speed_x_time += excess_v * decay_time

    time_saved_estimate = total_excess_speed_x_time / avg_tel_speed
    print(f"\n  Rough time-saved estimate from excess corner-exit speed:")
    print(f"    Average telemetry speed: {avg_tel_speed:.1f} m/s ({avg_tel_speed*3.6:.0f} km/h)")
    print(f"    Estimated time saved: {time_saved_estimate:.1f}s")

    # ================================================================
    # [8] TOP 10 HEAVIEST BRAKING EVENTS
    # ================================================================
    print("\n" + "=" * 90)
    print("[8] TOP 10 HEAVIEST BRAKING EVENTS (by deceleration)")
    print("=" * 90)

    sorted_events = sorted(events, key=lambda e: e["mean_decel_g"])
    print(f"\n  {'#':>3} {'Dist(m)':>10} {'Duration':>8} {'EntrySpd':>10} {'SpeedDrop':>10} "
          f"{'Decel(g)':>9} {'Pressure':>9}")
    for i, ev in enumerate(sorted_events[:10]):
        print(f"  {i+1:3d} {ev['distance_start_m']:10.0f} {ev['duration_s']:7.2f}s "
              f"{ev['entry_speed_ms']*3.6:9.1f}km/h {ev['speed_drop_ms']*3.6:9.1f}km/h "
              f"{ev['mean_decel_g']:8.4f}g {ev['peak_pressure_bar']:8.2f}bar")

    # ================================================================
    # [9] SIM ACTION DISTRIBUTION IN BRAKING ZONES
    # ================================================================
    print("\n" + "=" * 90)
    print("[9] WHAT DOES THE SIM DO WHEN THE REAL CAR BRAKES?")
    print("=" * 90)

    # For each telemetry sample with brake > 0, what is the sim doing?
    tel_dist_arr = tel["Distance on GPS Speed"].values
    tel_brake_arr = tel["RBrakePressure"].values

    braking_samples = tel_brake_arr > 0.5
    n_brake_samples = np.sum(braking_samples)
    brake_distances = tel_dist_arr[braking_samples]

    # Find nearest sim segment for each braking sample
    sim_actions_at_brake = []
    for d in brake_distances[:500]:  # sample first 500 for speed
        idx = np.argmin(np.abs(sim_dist - d))
        sim_actions_at_brake.append(sim_action[idx])

    sim_actions_at_brake = np.array(sim_actions_at_brake)
    n_sim_brake = np.sum(sim_actions_at_brake == "brake")
    n_sim_coast = np.sum(sim_actions_at_brake == "coast")
    n_sim_throttle = np.sum(sim_actions_at_brake == "throttle")
    n_total_check = len(sim_actions_at_brake)

    print(f"\n  At distances where telemetry has brake > 0.5 bar (sampled {n_total_check}):")
    print(f"    Sim BRAKE:    {n_sim_brake:4d} ({100*n_sim_brake/n_total_check:.1f}%)")
    print(f"    Sim COAST:    {n_sim_coast:4d} ({100*n_sim_coast/n_total_check:.1f}%)")
    print(f"    Sim THROTTLE: {n_sim_throttle:4d} ({100*n_sim_throttle/n_total_check:.1f}%)")

    # ================================================================
    # [10] QUANTIFY: HOW MUCH FORCE IS MISSING?
    # ================================================================
    print("\n" + "=" * 90)
    print("[10] FORCE DEFICIT QUANTIFICATION")
    print("=" * 90)

    # For each braking event, the telemetry deceleration comes from:
    #   a_total = a_regen + a_hydraulic + a_aero + a_rolling + a_cornering
    # The sim only has:
    #   a_sim = a_regen + a_aero + a_rolling + a_cornering
    # So the missing force is:
    #   F_hydraulic = m * (a_telemetry - a_sim) = m * missing_decel
    missing_forces = comp_df["missing_decel_g"] * 9.81 * MASS_KG

    print(f"\n  Missing hydraulic brake force estimate:")
    print(f"    Mean: {missing_forces.mean():.1f} N")
    print(f"    Median: {missing_forces.median():.1f} N")
    print(f"    Max: {missing_forces.max():.1f} N")

    # Compare to regen force
    print(f"\n  For comparison:")
    print(f"    Mean sim regen force during braking: {comp_df['mean_regen_force_n'].mean():.1f} N")
    print(f"    Mean sim resistance force during braking: {comp_df['mean_resist_force_n'].mean():.1f} N")
    if comp_df['mean_regen_force_n'].mean() > 0:
        ratio = missing_forces.mean() / comp_df['mean_regen_force_n'].mean()
        print(f"    Missing force / regen force ratio: {ratio:.2f}x")
        print(f"    -> The hydraulic brakes provide ~{ratio:.1f}x as much stopping force as regen")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"""
  The sim finishes {total_tel_time - result.total_time_s:.0f}s faster than telemetry.

  BRAKING:
  - {len(events)} braking events in telemetry ({total_brake_time:.1f}s, {total_brake_dist:.0f}m)
  - Telemetry mean decel during braking: {abs(np.mean(decels_g)):.4f} g ({abs(np.mean(decels_g))*9.81:.3f} m/s2)
  - Sim mean decel during same zones:    {comp_df['sim_decel_g'].mean():.4f} g ({comp_df['sim_decel_g'].mean()*9.81:.3f} m/s2)
  - Missing deceleration:                {comp_df['missing_decel_g'].mean():.4f} g ({comp_df['missing_decel_g'].mean()*9.81:.3f} m/s2)
  - Missing hydraulic brake force:       ~{missing_forces.mean():.0f} N mean

  LIGHT BRAKING (below threshold):
  - {n_below_thresh} samples have brake > 0 but fall below the 0.05 threshold
  - These are classified as COAST in the sim, losing their deceleration

  COAST DECELERATION:
  - Real car decelerates at ~{abs(np.mean(coast_decel_g)):.4f} g during coast
  - Sim coast deceleration: ~{np.mean(sim_coast_decel_g):.4f} g
  - Gap: ~{abs(np.mean(coast_decel_g)) - np.mean(sim_coast_decel_g):.4f} g

  KEY FINDING: The sim only models regen braking ({comp_df['mean_regen_force_n'].mean():.0f} N avg).
  The real car's hydraulic brakes add ~{missing_forces.mean():.0f} N of stopping force.
  This means the sim decelerates less before corners, carries higher speed
  through braking zones, and finishes faster.
""")


if __name__ == "__main__":
    main()
