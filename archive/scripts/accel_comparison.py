"""Acceleration comparison: Sim vs Telemetry.

Diagnose WHERE the force model is wrong (too much drive force? too little
drag? too little braking?) causing the replay sim to finish 48s too fast.
"""

import math
import numpy as np
import pandas as pd

from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.tire_model import PacejkaTireModel
from fsae_sim.vehicle.load_transfer import LoadTransferModel
from fsae_sim.vehicle.cornering_solver import CorneringSolver
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine


def main():
    # ── Load everything (same as validate_tier3.py section 1) ──
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    track = Track.from_telemetry(df=aim_df)

    # Battery with two-step calibration
    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)
    battery.calibrate_pack_from_telemetry(aim_df)

    # ── Run replay sim ──
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
    result = engine.run(
        num_laps=num_laps,
        initial_soc_pct=initial_soc,
        initial_temp_c=initial_temp,
        initial_speed_ms=max(initial_speed, 0.5),
    )

    sim = result.states
    MASS_KG = config.vehicle.mass_kg  # 288
    G = 9.81

    # Effective mass used by the sim for acceleration
    tire_radius = 0.2042
    gear_ratio = config.powertrain.gear_ratio
    eta = config.powertrain.drivetrain_efficiency
    j_eff = (config.vehicle.rotor_inertia_kg_m2 * gear_ratio**2 * eta
             + 4 * config.vehicle.wheel_inertia_kg_m2)
    M_EFF = MASS_KG + j_eff / (tire_radius**2)

    print("=" * 70)
    print("ACCELERATION COMPARISON: SIM vs TELEMETRY")
    print("=" * 70)
    print(f"  Sim total time:       {result.total_time_s:.1f} s")
    print(f"  Telemetry total time: {aim_df['Time'].iloc[-1]:.1f} s")
    print(f"  Time error:           {result.total_time_s - aim_df['Time'].iloc[-1]:.1f} s")
    print(f"  Bare mass:            {MASS_KG:.0f} kg")
    print(f"  Effective mass:       {M_EFF:.1f} kg")
    print(f"  Sim segments:         {len(sim)}")
    print(f"  Telemetry samples:    {len(aim_df)}")

    # ================================================================
    # 1. COMPUTE SIM LONGITUDINAL ACCELERATION (in g)
    # ================================================================
    # The sim uses m_effective for resolve_exit_speed, so:
    # a_lon = net_force_n / m_effective / g
    sim["accel_lon_g"] = sim["net_force_n"] / M_EFF / G

    # Also compute separate components in g
    sim["drive_accel_g"] = sim["drive_force_n"] / M_EFF / G
    sim["resist_accel_g"] = sim["resistance_force_n"] / M_EFF / G
    sim["regen_accel_g"] = sim["regen_force_n"] / M_EFF / G

    # ================================================================
    # 2. RESAMPLE TELEMETRY TO MATCH SIM DISTANCE POINTS
    # ================================================================
    # Filter telemetry to moving only (same as ReplayStrategy does)
    telem = aim_df[aim_df["GPS Speed"] > 5.0].copy()
    telem_dist = telem["Distance on GPS Speed"].values
    telem_lonaccel = telem["GPS LonAcc"].values
    telem_lataccel = telem["GPS LatAcc"].values
    telem_speed_kmh = telem["GPS Speed"].values

    sim_dist = sim["distance_m"].values

    # Interpolate telemetry onto sim distance points
    telem_lon_at_sim = np.interp(sim_dist, telem_dist, telem_lonaccel)
    telem_lat_at_sim = np.interp(sim_dist, telem_dist, telem_lataccel)
    telem_speed_at_sim = np.interp(sim_dist, telem_dist, telem_speed_kmh)

    sim["telem_lon_g"] = telem_lon_at_sim
    sim["telem_lat_g"] = telem_lat_at_sim
    sim["telem_speed_kmh"] = telem_speed_at_sim

    # Acceleration difference: sim - telemetry (positive = sim accelerates more)
    sim["accel_diff_g"] = sim["accel_lon_g"] - sim["telem_lon_g"]

    # ================================================================
    # 3. BREAKDOWN BY DRIVING MODE
    # ================================================================
    print("\n" + "=" * 70)
    print("3. LONGITUDINAL ACCELERATION COMPARISON BY MODE")
    print("=" * 70)

    accel_mask = sim["action"] == "throttle"
    brake_mask = sim["action"] == "brake"
    coast_mask = sim["action"] == "coast"

    for label, mask in [("ALL", np.ones(len(sim), dtype=bool)),
                        ("THROTTLE", accel_mask),
                        ("COAST", coast_mask),
                        ("BRAKE", brake_mask)]:
        subset = sim[mask]
        if len(subset) == 0:
            print(f"\n  {label}: no segments")
            continue
        diff = subset["accel_diff_g"]
        print(f"\n  {label} ({len(subset)} segments, {len(subset)/len(sim)*100:.0f}% of track):")
        print(f"    Mean sim accel:    {subset['accel_lon_g'].mean():+.4f} g")
        print(f"    Mean telem accel:  {subset['telem_lon_g'].mean():+.4f} g")
        print(f"    Mean difference:   {diff.mean():+.4f} g  (sim-telem, + = sim too fast)")
        print(f"    RMS difference:    {np.sqrt((diff**2).mean()):.4f} g")
        print(f"    Max sim>telem:     {diff.max():+.4f} g at d={subset.loc[diff.idxmax(), 'distance_m']:.0f} m")
        print(f"    Max telem>sim:     {diff.min():+.4f} g at d={subset.loc[diff.idxmin(), 'distance_m']:.0f} m")

    # ================================================================
    # 4. WORST DISCREPANCY LOCATIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("4. TOP 20 WORST DISCREPANCY LOCATIONS (sim too fast)")
    print("=" * 70)
    print(f"  {'Dist(m)':>8}  {'Lap':>3}  {'Action':>8}  {'SimG':>7}  {'TelemG':>7}  {'DiffG':>7}  {'SimSpd':>7}  {'TelemSpd':>8}  {'Curv':>8}")

    worst_fast = sim.nlargest(20, "accel_diff_g")
    for _, row in worst_fast.iterrows():
        print(f"  {row['distance_m']:8.0f}  {row['lap']:3.0f}  {row['action']:>8s}  "
              f"{row['accel_lon_g']:+7.3f}  {row['telem_lon_g']:+7.3f}  {row['accel_diff_g']:+7.3f}  "
              f"{row['speed_kmh']:7.1f}  {row['telem_speed_kmh']:8.1f}  {row['curvature']:8.4f}")

    print(f"\n  TOP 20 WORST DISCREPANCY LOCATIONS (sim too slow)")
    print(f"  {'Dist(m)':>8}  {'Lap':>3}  {'Action':>8}  {'SimG':>7}  {'TelemG':>7}  {'DiffG':>7}  {'SimSpd':>7}  {'TelemSpd':>8}  {'Curv':>8}")

    worst_slow = sim.nsmallest(20, "accel_diff_g")
    for _, row in worst_slow.iterrows():
        print(f"  {row['distance_m']:8.0f}  {row['lap']:3.0f}  {row['action']:>8s}  "
              f"{row['accel_lon_g']:+7.3f}  {row['telem_lon_g']:+7.3f}  {row['accel_diff_g']:+7.3f}  "
              f"{row['speed_kmh']:7.1f}  {row['telem_speed_kmh']:8.1f}  {row['curvature']:8.4f}")

    # ================================================================
    # 5. LATERAL ACCELERATION & CORNERING DRAG ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("5. LATERAL ACCELERATION & CORNERING DRAG")
    print("=" * 70)

    high_lat = sim[abs(sim["telem_lat_g"]) > 0.5].copy()
    low_lat = sim[abs(sim["telem_lat_g"]) <= 0.5].copy()

    print(f"\n  Segments with |GPS LatAcc| > 0.5g: {len(high_lat)} ({len(high_lat)/len(sim)*100:.0f}%)")
    print(f"  Segments with |GPS LatAcc| <= 0.5g: {len(low_lat)} ({len(low_lat)/len(sim)*100:.0f}%)")

    if len(high_lat) > 0:
        print(f"\n  HIGH CORNERING (|LatAcc| > 0.5g):")
        print(f"    Mean |GPS LatAcc|:       {abs(high_lat['telem_lat_g']).mean():.3f} g")
        print(f"    Mean resistance force:   {high_lat['resistance_force_n'].mean():.1f} N")
        print(f"    Mean sim lon accel:      {high_lat['accel_lon_g'].mean():+.4f} g")
        print(f"    Mean telem lon accel:    {high_lat['telem_lon_g'].mean():+.4f} g")
        print(f"    Mean accel diff:         {high_lat['accel_diff_g'].mean():+.4f} g (+ = sim too fast)")
        print(f"    Mean speed:              {high_lat['speed_kmh'].mean():.1f} km/h")
        print(f"    Mean curvature:          {abs(high_lat['curvature']).mean():.4f} 1/m")

    if len(low_lat) > 0:
        print(f"\n  STRAIGHT-LINE (|LatAcc| <= 0.5g):")
        print(f"    Mean |GPS LatAcc|:       {abs(low_lat['telem_lat_g']).mean():.3f} g")
        print(f"    Mean resistance force:   {low_lat['resistance_force_n'].mean():.1f} N")
        print(f"    Mean sim lon accel:      {low_lat['accel_lon_g'].mean():+.4f} g")
        print(f"    Mean telem lon accel:    {low_lat['telem_lon_g'].mean():+.4f} g")
        print(f"    Mean accel diff:         {low_lat['accel_diff_g'].mean():+.4f} g (+ = sim too fast)")

    # Check what cornering drag the sim actually computes
    print(f"\n  CORNERING DRAG BREAKDOWN (high-lat segments):")
    if len(high_lat) > 0:
        # Sample cornering drag computation at a few representative points
        dynamics = engine.dynamics
        print(f"    {'Speed':>7}  {'Curv':>8}  {'AeroDrag':>9}  {'RollRes':>8}  {'CornDrag':>9}  {'Parasit':>8}  {'Total':>8}")
        # Pick 10 representative high-lat segments
        sample_idx = np.linspace(0, len(high_lat)-1, min(10, len(high_lat)), dtype=int)
        for i in sample_idx:
            row = high_lat.iloc[i]
            spd = row["speed_ms"]
            curv = row["curvature"]
            aero = dynamics.drag_force(spd)
            rr = dynamics.rolling_resistance_force(spd)
            cd = dynamics.cornering_drag(spd, curv)
            para = dynamics.parasitic_drag()
            total = dynamics.total_resistance(spd, 0.0, curv)
            print(f"    {spd*3.6:7.1f}  {curv:8.4f}  {aero:9.1f}  {rr:8.1f}  {cd:9.1f}  {para:8.1f}  {total:8.1f}")

    # ================================================================
    # 5b. EXPECTED CORNERING DRAG FROM TELEMETRY LAT-G
    # ================================================================
    print(f"\n  EXPECTED vs ACTUAL CORNERING DRAG:")
    # For each sim segment, compute what cornering drag SHOULD be based on
    # telemetry lateral g, and compare to what the sim actually uses
    if len(high_lat) > 0:
        # Cornering drag ~ F_lat * sin(alpha) ~ F_lat^2 / C_alpha
        # F_lat = m * a_lat = m * g * lat_g
        f_lat_from_telem = MASS_KG * G * abs(high_lat["telem_lat_g"].values)
        # The sim uses curvature-based lateral force: m * v^2 * kappa
        f_lat_from_sim = MASS_KG * (high_lat["speed_ms"].values ** 2) * abs(high_lat["curvature"].values)

        print(f"    Lateral force from telemetry GPS LatAcc:  mean={f_lat_from_telem.mean():.0f} N")
        print(f"    Lateral force from sim (m*v^2*kappa):     mean={f_lat_from_sim.mean():.0f} N")
        print(f"    Ratio (telem/sim):                        {f_lat_from_telem.mean()/max(f_lat_from_sim.mean(),1):.2f}")

    # ================================================================
    # 6. TOTAL IMPULSE DIFFERENCE
    # ================================================================
    print("\n" + "=" * 70)
    print("6. TOTAL IMPULSE DIFFERENCE (FORCE x TIME)")
    print("=" * 70)

    # Impulse from sim: net_force * segment_time for each segment
    sim_impulse_per_seg = sim["net_force_n"].values * sim["segment_time_s"].values  # N*s

    # Impulse from telemetry: m_eff * a * dt (resampled to sim distances)
    # But we need time-based integration. Use segment_time_s and telem accel at each point
    telem_impulse_per_seg = M_EFF * G * sim["telem_lon_g"].values * sim["segment_time_s"].values  # N*s

    total_sim_impulse = sim_impulse_per_seg.sum()
    total_telem_impulse = telem_impulse_per_seg.sum()
    impulse_diff = total_sim_impulse - total_telem_impulse

    print(f"  Total sim impulse:         {total_sim_impulse:+.0f} N*s")
    print(f"  Total telem impulse:       {total_telem_impulse:+.0f} N*s")
    print(f"  Impulse difference:        {impulse_diff:+.0f} N*s (sim - telem)")
    print(f"  Equiv speed diff @ end:    {impulse_diff/M_EFF:.1f} m/s = {impulse_diff/M_EFF*3.6:.1f} km/h")

    # Break impulse diff down by action type
    print(f"\n  IMPULSE DIFFERENCE BY ACTION TYPE:")
    for label, mask in [("THROTTLE", accel_mask), ("COAST", coast_mask), ("BRAKE", brake_mask)]:
        if mask.sum() == 0:
            continue
        sim_imp = (sim.loc[mask, "net_force_n"] * sim.loc[mask, "segment_time_s"]).sum()
        tel_imp = (M_EFF * G * sim.loc[mask, "telem_lon_g"] * sim.loc[mask, "segment_time_s"]).sum()
        diff = sim_imp - tel_imp
        print(f"    {label:>8}: sim={sim_imp:+8.0f}  telem={tel_imp:+8.0f}  diff={diff:+8.0f} N*s")

    # ================================================================
    # 6b. CUMULATIVE IMPULSE DIFFERENCE OVER DISTANCE
    # ================================================================
    print(f"\n  CUMULATIVE IMPULSE DIFFERENCE BY LAP:")
    impulse_diff_per_seg = sim_impulse_per_seg - telem_impulse_per_seg
    cum_impulse_diff = np.cumsum(impulse_diff_per_seg)

    for lap_num in range(num_laps):
        lap_mask = sim["lap"] == lap_num
        if lap_mask.sum() == 0:
            continue
        lap_diff = impulse_diff_per_seg[lap_mask.values].sum()
        last_idx = np.where(lap_mask.values)[0][-1]
        cum_at_end = cum_impulse_diff[last_idx]
        lap_time_sim = sim.loc[lap_mask, "segment_time_s"].sum()
        print(f"    Lap {lap_num+1:2d}: impulse_diff={lap_diff:+7.0f} N*s  "
              f"cumulative={cum_at_end:+8.0f} N*s  "
              f"sim_lap_time={lap_time_sim:.1f}s")

    # ================================================================
    # 7. FORCE COMPONENT SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("7. FORCE COMPONENT SUMMARY (mean values)")
    print("=" * 70)
    print(f"  {'':>10}  {'Drive':>9}  {'Resist':>9}  {'Regen':>9}  {'Net':>9}  {'TelemNet':>9}  {'Diff':>9}")
    for label, mask in [("ALL", np.ones(len(sim), dtype=bool)),
                        ("THROTTLE", accel_mask),
                        ("COAST", coast_mask),
                        ("BRAKE", brake_mask)]:
        subset = sim[mask]
        if len(subset) == 0:
            continue
        telem_net = M_EFF * G * subset["telem_lon_g"].mean()
        print(f"  {label:>10}  {subset['drive_force_n'].mean():9.1f}  "
              f"{subset['resistance_force_n'].mean():9.1f}  "
              f"{subset['regen_force_n'].mean():9.1f}  "
              f"{subset['net_force_n'].mean():9.1f}  "
              f"{telem_net:9.1f}  "
              f"{subset['net_force_n'].mean() - telem_net:+9.1f}")

    # ================================================================
    # 8. SPEED COMPARISON OVER DISTANCE
    # ================================================================
    print("\n" + "=" * 70)
    print("8. SPEED COMPARISON")
    print("=" * 70)
    speed_diff = sim["speed_kmh"].values - sim["telem_speed_kmh"].values
    print(f"  Mean speed diff (sim-telem): {speed_diff.mean():+.2f} km/h")
    print(f"  RMS speed diff:              {np.sqrt((speed_diff**2).mean()):.2f} km/h")
    print(f"  Max sim>telem:               {speed_diff.max():+.2f} km/h at d={sim.loc[np.argmax(speed_diff), 'distance_m']:.0f} m")
    print(f"  Max telem>sim:               {speed_diff.min():+.2f} km/h at d={sim.loc[np.argmin(speed_diff), 'distance_m']:.0f} m")

    # ================================================================
    # 9. RESISTANCE FORCE AT DIFFERENT SPEED RANGES
    # ================================================================
    print("\n" + "=" * 70)
    print("9. RESISTANCE FORCE AT DIFFERENT SPEEDS")
    print("=" * 70)
    dynamics = engine.dynamics
    print(f"  {'Speed':>7}  {'AeroDrag':>9}  {'RollRes':>8}  {'Parasit':>8}  {'Total(str)':>11}  {'Decel(g)':>9}")
    for v_kmh in [10, 20, 30, 40, 50, 60, 70, 80]:
        v_ms = v_kmh / 3.6
        aero = dynamics.drag_force(v_ms)
        rr = dynamics.rolling_resistance_force(v_ms)
        para = dynamics.parasitic_drag()
        total = aero + rr + para  # straight line, no curvature
        decel_g = total / M_EFF / G
        print(f"  {v_kmh:5d}    {aero:9.1f}  {rr:8.1f}  {para:8.1f}  {total:11.1f}  {decel_g:9.4f}")

    # ================================================================
    # 10. COASTING DECELERATION COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("10. COASTING DECELERATION COMPARISON")
    print("=" * 70)
    coast_data = sim[coast_mask & (abs(sim["telem_lat_g"]) < 0.3)].copy()  # straight coasting only
    if len(coast_data) > 0:
        # Bin by speed
        coast_data["speed_bin"] = pd.cut(coast_data["speed_kmh"], bins=[0,15,25,35,45,55,65,80])
        grouped = coast_data.groupby("speed_bin", observed=True)
        print(f"  {'Speed Bin':>15}  {'N':>5}  {'SimDecel(g)':>12}  {'TelemDecel(g)':>14}  {'Diff(g)':>9}")
        for name, group in grouped:
            if len(group) > 5:
                sim_decel = group["accel_lon_g"].mean()
                tel_decel = group["telem_lon_g"].mean()
                print(f"  {str(name):>15}  {len(group):5d}  {sim_decel:+12.4f}  {tel_decel:+14.4f}  {sim_decel-tel_decel:+9.4f}")

    # ================================================================
    # 11. ACTION DISTRIBUTION COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("11. ACTION DISTRIBUTION")
    print("=" * 70)
    total_segs = len(sim)
    for action_name in ["throttle", "coast", "brake"]:
        mask = sim["action"] == action_name
        n = mask.sum()
        time_s = sim.loc[mask, "segment_time_s"].sum()
        dist_m = 0
        # Sum segment lengths from distance differences
        if n > 0:
            # Use segment count * avg segment length
            dist_m = n * (sim["distance_m"].diff().median() if len(sim) > 1 else 0)
        print(f"  {action_name:>8}: {n:5d} segments ({n/total_segs*100:5.1f}%), {time_s:7.1f}s ({time_s/result.total_time_s*100:5.1f}%)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
