"""Segment-by-segment comparison: sim vs telemetry.

Compares CalibratedStrategy force-based sim against real AiM telemetry
at each track segment to expose WHERE and WHY the sim diverges.
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


def segment_telemetry_for_lap(aim_df, track, lap_boundaries, lap_idx):
    """Extract per-segment telemetry averages for a specific lap."""
    start_idx, end_idx, _ = lap_boundaries[lap_idx]
    lap_df = aim_df.iloc[start_idx:end_idx].copy()

    # Distance within lap
    time_arr = lap_df["Time"].values
    speed_ms = lap_df["GPS Speed"].values * (1000.0 / 3600.0)
    dt = np.diff(time_arr, prepend=time_arr[0])
    dist = np.cumsum(speed_ms * dt)
    lap_length = track.total_distance_m

    # Map each telemetry sample to a segment
    seg_length = track.segments[0].length_m
    seg_indices = np.clip((dist / seg_length).astype(int), 0, track.num_segments - 1)

    # Per-segment averages
    records = []
    for seg_idx in range(track.num_segments):
        mask = seg_indices == seg_idx
        if mask.sum() == 0:
            continue
        records.append({
            "segment_idx": seg_idx,
            "tel_speed_kmh": float(np.mean(lap_df["GPS Speed"].values[mask])),
            "tel_speed_ms": float(np.mean(speed_ms[mask])),
            "tel_throttle": float(np.mean(lap_df["Throttle Pos"].values[mask])),
            "tel_torque_nm": float(np.mean(lap_df["Torque Feedback"].values[mask])),
            "tel_rpm": float(np.mean(lap_df["RPM"].values[mask])),
            "tel_voltage": float(np.mean(lap_df["Pack Voltage"].values[mask])),
            "tel_current": float(np.mean(lap_df["Pack Current"].values[mask])),
            "tel_power_w": float(np.mean(
                lap_df["Pack Voltage"].values[mask] * lap_df["Pack Current"].values[mask]
            )),
            "tel_soc": float(np.mean(lap_df["State of Charge"].values[mask])),
            "tel_lat_g": float(np.mean(np.abs(lap_df["GPS LatAcc"].values[mask]))),
            "tel_n_samples": int(mask.sum()),
        })
    return pd.DataFrame(records)


def segment_telemetry_averaged(aim_df, track, lap_boundaries, lap_range):
    """Average per-segment telemetry across multiple laps."""
    frames = []
    for lap_idx in lap_range:
        if lap_idx < len(lap_boundaries):
            df = segment_telemetry_for_lap(aim_df, track, lap_boundaries, lap_idx)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.groupby("segment_idx").mean().reset_index()


def main():
    print("=" * 70)
    print("SEGMENT-BY-SEGMENT SIMULATION DIAGNOSTIC")
    print("=" * 70)

    # Load everything
    print("\nLoading data...")
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

    print(f"  Track: {track.num_segments} segments, {track.total_distance_m:.0f}m per lap")

    # Detect laps
    lap_boundaries = detect_lap_boundaries(aim_df)
    print(f"  Detected {len(lap_boundaries)} laps")

    # Calibrate driver model
    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    print(f"  Driver model: {len(strategy.zones)} zones")

    # Run sim for 1 lap first (easier to compare)
    print("\n--- Single-Lap Comparison (Lap 3 equivalent) ---")
    engine = SimulationEngine(config, track, strategy, battery)
    result_1 = engine.run(num_laps=1, initial_soc_pct=92.0, initial_temp_c=28.0)
    sim_df = result_1.states

    # Get averaged telemetry for driver 1 steady-state laps (laps 2-8, 0-indexed)
    tel_avg = segment_telemetry_averaged(aim_df, track, lap_boundaries, range(2, 8))

    # Merge sim and telemetry
    merged = sim_df.merge(tel_avg, on="segment_idx", how="inner")

    # ====================================================================
    # SPEED COMPARISON
    # ====================================================================
    print("\n" + "=" * 70)
    print("SPEED COMPARISON (sim vs telemetry, per segment)")
    print("=" * 70)

    speed_err = merged["speed_kmh"] - merged["tel_speed_kmh"]
    speed_err_pct = speed_err / merged["tel_speed_kmh"].replace(0, np.nan) * 100

    print(f"\n  Overall speed stats:")
    print(f"    Sim mean speed:       {merged['speed_kmh'].mean():.1f} km/h")
    print(f"    Telemetry mean speed: {merged['tel_speed_kmh'].mean():.1f} km/h")
    print(f"    Mean speed error:     {speed_err.mean():+.1f} km/h ({speed_err_pct.mean():+.1f}%)")
    print(f"    RMS speed error:      {np.sqrt((speed_err**2).mean()):.1f} km/h")
    print(f"    Max sim-faster:       {speed_err.max():+.1f} km/h at seg {speed_err.idxmax()}")
    print(f"    Max sim-slower:       {speed_err.min():+.1f} km/h at seg {speed_err.idxmin()}")

    # Speed by track zone (corners vs straights)
    merged["curvature_abs"] = merged["curvature"].abs()
    straights = merged[merged["curvature_abs"] < 0.005]
    corners = merged[merged["curvature_abs"] >= 0.005]

    if len(straights) > 0:
        s_err = straights["speed_kmh"] - straights["tel_speed_kmh"]
        print(f"\n  Straights ({len(straights)} segments):")
        print(f"    Sim: {straights['speed_kmh'].mean():.1f} km/h, Tel: {straights['tel_speed_kmh'].mean():.1f} km/h, Err: {s_err.mean():+.1f} km/h")
    if len(corners) > 0:
        c_err = corners["speed_kmh"] - corners["tel_speed_kmh"]
        print(f"  Corners ({len(corners)} segments):")
        print(f"    Sim: {corners['speed_kmh'].mean():.1f} km/h, Tel: {corners['tel_speed_kmh'].mean():.1f} km/h, Err: {c_err.mean():+.1f} km/h")

    # Segment-by-segment detail (binned into ~10 groups)
    n = len(merged)
    bin_size = max(n // 15, 1)
    print(f"\n  Speed by track position (every ~{bin_size * 5}m):")
    print(f"  {'Seg':>5} {'Dist':>6} {'SimSpd':>7} {'TelSpd':>7} {'Err':>6} {'Curv':>7} {'Action':>8} {'Throt%':>7}")
    for i in range(0, n, bin_size):
        chunk = merged.iloc[i:i + bin_size]
        seg = int(chunk["segment_idx"].iloc[0])
        dist = seg * 5.0
        sim_spd = chunk["speed_kmh"].mean()
        tel_spd = chunk["tel_speed_kmh"].mean()
        err = sim_spd - tel_spd
        curv = chunk["curvature"].mean()
        action = chunk["action"].mode().iloc[0] if len(chunk) > 0 else ""
        throt = chunk["throttle_pct"].mean() * 100
        print(f"  {seg:5d} {dist:5.0f}m {sim_spd:6.1f} {tel_spd:6.1f} {err:+5.1f} {curv:+7.4f} {action:>8} {throt:6.1f}%")

    # ====================================================================
    # TORQUE AND POWER COMPARISON
    # ====================================================================
    print("\n" + "=" * 70)
    print("TORQUE AND POWER COMPARISON")
    print("=" * 70)

    # Sim motor torque vs telemetry torque feedback
    torque_err = merged["motor_torque_nm"] - merged["tel_torque_nm"]
    print(f"\n  Motor torque:")
    print(f"    Sim mean:  {merged['motor_torque_nm'].mean():.1f} Nm")
    print(f"    Tel mean:  {merged['tel_torque_nm'].mean():.1f} Nm")
    print(f"    Error:     {torque_err.mean():+.1f} Nm")

    # Sim RPM vs telemetry RPM
    rpm_err = merged["motor_rpm"] - merged["tel_rpm"]
    print(f"\n  Motor RPM:")
    print(f"    Sim mean:  {merged['motor_rpm'].mean():.0f}")
    print(f"    Tel mean:  {merged['tel_rpm'].mean():.0f}")
    print(f"    Error:     {rpm_err.mean():+.0f}")

    # Electrical power
    power_err = merged["electrical_power_w"] - merged["tel_power_w"]
    print(f"\n  Electrical power:")
    print(f"    Sim mean:  {merged['electrical_power_w'].mean():.0f} W")
    print(f"    Tel mean:  {merged['tel_power_w'].mean():.0f} W")
    print(f"    Error:     {power_err.mean():+.0f} W ({power_err.mean() / max(merged['tel_power_w'].mean(), 1) * 100:+.1f}%)")

    # ====================================================================
    # FORCE ANALYSIS
    # ====================================================================
    print("\n" + "=" * 70)
    print("FORCE ANALYSIS")
    print("=" * 70)

    print(f"\n  Drive force:      mean={merged['drive_force_n'].mean():.1f} N")
    print(f"  Resistance force: mean={merged['resistance_force_n'].mean():.1f} N")
    print(f"  Net force:        mean={merged['net_force_n'].mean():.1f} N")

    # What telemetry implies: F = m*a (from GPS longitudinal acceleration)
    # We don't have direct lon_acc in merged, but we can estimate from speed changes

    # ====================================================================
    # POWERTRAIN OPERATING POINT ANALYSIS
    # ====================================================================
    print("\n" + "=" * 70)
    print("POWERTRAIN OPERATING POINT ANALYSIS")
    print("=" * 70)

    # Check where the sim is operating on the torque-RPM curve
    for _, row in merged.iterrows():
        pass  # just using aggregate stats below

    # Breakdown: how much torque the sim commands vs what's available
    throttle_segs = merged[merged["action"] == "throttle"]
    if len(throttle_segs) > 0:
        max_torques = []
        for _, row in throttle_segs.iterrows():
            mt = powertrain.max_motor_torque(row["motor_rpm"])
            max_torques.append(mt)
        throttle_segs = throttle_segs.copy()
        throttle_segs["max_torque_available"] = max_torques
        throttle_segs["torque_utilization"] = (
            throttle_segs["motor_torque_nm"] / throttle_segs["max_torque_available"].replace(0, np.nan)
        )

        print(f"\n  Throttle segments ({len(throttle_segs)}):")
        print(f"    Mean commanded torque:  {throttle_segs['motor_torque_nm'].mean():.1f} Nm")
        print(f"    Mean available torque:  {throttle_segs['max_torque_available'].mean():.1f} Nm")
        print(f"    Mean utilization:       {throttle_segs['torque_utilization'].mean() * 100:.1f}%")
        print(f"    Telemetry mean torque:  {throttle_segs['tel_torque_nm'].mean():.1f} Nm")

        # How many segments are in field-weakening?
        fw_mask = throttle_segs["motor_rpm"] > config.powertrain.brake_speed_rpm
        if fw_mask.sum() > 0:
            fw_segs = throttle_segs[fw_mask]
            print(f"\n  Field-weakening segments ({fw_mask.sum()}):")
            print(f"    RPM range:    {fw_segs['motor_rpm'].min():.0f} - {fw_segs['motor_rpm'].max():.0f}")
            print(f"    Speed range:  {fw_segs['speed_kmh'].min():.1f} - {fw_segs['speed_kmh'].max():.1f} km/h")
            print(f"    Sim torque:   {fw_segs['motor_torque_nm'].mean():.1f} Nm (linear taper)")
            print(f"    Tel torque:   {fw_segs['tel_torque_nm'].mean():.1f} Nm (real motor)")
            print(f"    Max available (linear): {fw_segs['max_torque_available'].mean():.1f} Nm")
            # What constant-power would give
            p_max = config.powertrain.torque_limit_inverter_nm * config.powertrain.brake_speed_rpm * (math.pi / 30.0)
            cp_torques = p_max / (fw_segs["motor_rpm"] * math.pi / 30.0)
            print(f"    Max available (const-P): {cp_torques.mean():.1f} Nm")
            print(f"    Linear taper underestimates by: {(cp_torques.mean() - fw_segs['max_torque_available'].mean()):.1f} Nm ({(cp_torques.mean() / fw_segs['max_torque_available'].mean() - 1) * 100:.0f}%)")

    # Check corner speed limit vs actual speed
    corner_segs = merged[merged["corner_speed_limit_ms"] < 100]
    if len(corner_segs) > 0:
        print(f"\n  Corner-limited segments ({len(corner_segs)}):")
        print(f"    Sim speed:  {corner_segs['speed_kmh'].mean():.1f} km/h")
        print(f"    Tel speed:  {corner_segs['tel_speed_kmh'].mean():.1f} km/h")
        print(f"    Limit:      {(corner_segs['corner_speed_limit_ms'].mean() * 3.6):.1f} km/h")

    # ====================================================================
    # ENERGY ACCOUNTING BREAKDOWN
    # ====================================================================
    print("\n" + "=" * 70)
    print("ENERGY ACCOUNTING (1 LAP)")
    print("=" * 70)

    pos_power = merged[merged["electrical_power_w"] > 0]
    neg_power = merged[merged["electrical_power_w"] < 0]

    sim_energy_j = (merged["electrical_power_w"] * merged["segment_time_s"]).sum()
    tel_energy_j = (merged["tel_power_w"] * merged["segment_time_s"]).sum()

    print(f"\n  Sim energy this lap:  {sim_energy_j / 3.6e6:.3f} kWh")
    print(f"  Tel energy (est):     {tel_energy_j / 3.6e6:.3f} kWh")
    print(f"  Sim lap time:         {merged['segment_time_s'].sum():.1f} s")

    if len(pos_power) > 0:
        drive_energy = (pos_power["electrical_power_w"] * pos_power["segment_time_s"]).sum()
        print(f"  Driving energy:       {drive_energy / 3.6e6:.3f} kWh")
    if len(neg_power) > 0:
        regen_energy = (neg_power["electrical_power_w"] * neg_power["segment_time_s"]).sum()
        print(f"  Regen recovered:      {regen_energy / 3.6e6:.4f} kWh")

    # What efficiency are we getting from powertrain?
    mech_power_drive = pos_power["motor_torque_nm"] * pos_power["motor_rpm"] * math.pi / 30.0
    if mech_power_drive.sum() > 0:
        implied_eff = pos_power["electrical_power_w"].sum() / mech_power_drive.sum()
        print(f"  Implied drivetrain eff: {1.0/implied_eff:.3f} (config: {config.powertrain.drivetrain_efficiency})")

    # ====================================================================
    # TOP WORST SEGMENTS
    # ====================================================================
    print("\n" + "=" * 70)
    print("TOP 20 WORST SPEED ERRORS (sim vs telemetry)")
    print("=" * 70)

    merged["speed_error_kmh"] = merged["speed_kmh"] - merged["tel_speed_kmh"]
    merged["speed_error_abs"] = merged["speed_error_kmh"].abs()
    worst = merged.nlargest(20, "speed_error_abs")

    print(f"\n  {'Seg':>5} {'Dist':>6} {'SimSpd':>7} {'TelSpd':>7} {'Err':>7} {'Curv':>7} {'Action':>8} {'SimTrq':>7} {'TelTrq':>7}")
    for _, row in worst.iterrows():
        seg = int(row["segment_idx"])
        dist = seg * 5.0
        print(
            f"  {seg:5d} {dist:5.0f}m "
            f"{row['speed_kmh']:6.1f} {row['tel_speed_kmh']:6.1f} "
            f"{row['speed_error_kmh']:+6.1f} "
            f"{row['curvature']:+7.4f} "
            f"{row['action']:>8} "
            f"{row['motor_torque_nm']:6.1f} {row['tel_torque_nm']:6.1f}"
        )

    # ====================================================================
    # FULL SEGMENT DUMP (for plotting)
    # ====================================================================
    print("\n" + "=" * 70)
    print("FULL SEGMENT PROFILE (every segment)")
    print("=" * 70)
    print(f"\n  {'Seg':>4} {'SimSpd':>7} {'TelSpd':>7} {'SpdErr':>7} {'SimTrq':>7} {'TelTrq':>7} {'SimPwr':>8} {'TelPwr':>8} {'Action':>8}")
    for _, row in merged.iterrows():
        seg = int(row["segment_idx"])
        print(
            f"  {seg:4d} "
            f"{row['speed_kmh']:6.1f} {row['tel_speed_kmh']:6.1f} "
            f"{row['speed_error_kmh']:+6.1f} "
            f"{row['motor_torque_nm']:6.1f} {row['tel_torque_nm']:6.1f} "
            f"{row['electrical_power_w']:7.0f} {row['tel_power_w']:7.0f} "
            f"{row['action']:>8}"
        )


if __name__ == "__main__":
    main()
