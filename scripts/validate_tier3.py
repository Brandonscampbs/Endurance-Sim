"""Tier 3 Simulation Validation: Compare sim results to Michigan 2025 telemetry."""

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
from fsae_sim.driver.strategies import CoastOnlyStrategy, ThresholdBrakingStrategy, ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.analysis.validation import validate_full_endurance, detect_lap_boundaries
from fsae_sim.analysis.validation_plots import plot_validation


def _annotate_laps(aim_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate aim_df with a 1-based ``lap`` column using lap boundaries.

    Used to drive the ``holdout_laps`` argument of
    :meth:`BatteryModel.calibrate_pack_from_telemetry`.  Rows outside
    detected laps (pre-start, post-finish, driver change) receive
    ``lap = 0``.
    """
    laps = detect_lap_boundaries(aim_df)
    lap_col = np.zeros(len(aim_df), dtype=int)
    for lap_num, (start_idx, end_idx, _) in enumerate(laps, start=1):
        lap_col[start_idx:end_idx] = lap_num
    out = aim_df.copy()
    out["lap"] = lap_col
    return out


def main():
    # ── Load everything ──
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    track = Track.from_telemetry(df=aim_df)

    # C15 fix: decouple battery calibration from validation.
    # Calibrate from Voltt cell-level data ONLY; pack-level parameters
    # scale geometrically via series/parallel.  We do NOT call
    # calibrate_pack_from_telemetry on aim_df here — doing so fits
    # OCV and pack R against the same data we then validate against.
    aim_df = _annotate_laps(aim_df)
    HOLDOUT_LAPS = list(range(13, 22))  # validate on laps 13-21

    battery = BatteryModel(config.battery)
    battery.calibrate_from_voltt(voltt_df)

    # ── Tire model components ──
    tire = PacejkaTireModel(config.tire.tir_file)
    lt = LoadTransferModel(config.vehicle, config.suspension)
    solver = CorneringSolver(
        tire, lt, config.vehicle.mass_kg,
        math.radians(config.tire.static_camber_front_deg),
        math.radians(config.tire.static_camber_rear_deg),
        config.suspension.roll_camber_front_deg_per_deg,
        config.suspension.roll_camber_rear_deg_per_deg,
    )

    legacy_dyn = VehicleDynamics(config.vehicle)
    new_dyn = VehicleDynamics(config.vehicle, tire, lt, solver)

    print("=" * 70)
    print("TIER 3 SIMULATION ACCURACY REPORT")
    print("Michigan 2025 Endurance -- CT-16EV")
    print("=" * 70)

    # ── 1. Full Endurance Replay Validation ──
    print("\n1. REPLAY MODE -- Full Endurance Validation")
    print("-" * 50)

    initial_soc = float(aim_df["State of Charge"].iloc[0])
    initial_temp = float(aim_df["Pack Temp"].iloc[0])
    initial_speed = float(aim_df["GPS Speed"].iloc[0]) / 3.6

    total_distance = aim_df["Distance on GPS Speed"].iloc[-1]
    num_laps = round(total_distance / track.total_distance_m)

    replay = ReplayStrategy.from_full_endurance(aim_df, track.lap_distance_m)

    # Need fresh battery for each run. Voltt-only calibration (no AiM
    # pack fit) so the replay validation is an honest comparison
    # against the held-out stint.
    batt_replay = BatteryModel(config.battery)
    batt_replay.calibrate_from_voltt(voltt_df)

    engine = SimulationEngine(config, track, replay, batt_replay)
    result = engine.run(
        num_laps=num_laps,
        initial_soc_pct=initial_soc,
        initial_temp_c=initial_temp,
        initial_speed_ms=max(initial_speed, 0.5),
    )

    # C15 fix: validate on held-out stint only (laps 13-21).  The full
    # endurance report is still printed for reference below.
    aim_holdout = aim_df[aim_df["lap"].isin(HOLDOUT_LAPS)].reset_index(drop=True)
    sim_holdout = result.states[result.states["lap"].isin(HOLDOUT_LAPS)].reset_index(drop=True)

    if len(aim_holdout) > 0 and len(sim_holdout) > 0:
        # Approximate per-stint totals for sim; the report API still
        # accepts scalars so we hand it the held-out stint aggregates.
        holdout_time = float(sim_holdout["time_s"].iloc[-1] - sim_holdout["time_s"].iloc[0])
        # D-05: report NET energy (discharge - regen), consistent with
        # SimResult.total_energy_kwh and the telemetry-side net.
        holdout_energy_j_arr = sim_holdout["electrical_power_w"].values * sim_holdout["segment_time_s"].values
        holdout_discharge_j = float(np.sum(np.maximum(holdout_energy_j_arr, 0.0)))
        holdout_regen_j = float(np.sum(np.maximum(-holdout_energy_j_arr, 0.0)))
        holdout_energy_kwh = (holdout_discharge_j - holdout_regen_j) / 3.6e6
        holdout_final_soc = float(sim_holdout["soc_pct"].iloc[-1])

        report_holdout = validate_full_endurance(
            sim_holdout, aim_holdout,
            holdout_time, holdout_final_soc,
            holdout_energy_kwh, int(sim_holdout["lap"].nunique()),
            target_pct=5.0,
        )
        print("  Validation stint: laps", HOLDOUT_LAPS[0], "-", HOLDOUT_LAPS[-1])
        print(report_holdout.summary())
        report = report_holdout
    else:
        print("  WARNING: no held-out laps found; falling back to full endurance")
        report = validate_full_endurance(
            result.states, aim_df,
            result.total_time_s, result.final_soc,
            result.total_energy_kwh, result.laps_completed,
            target_pct=5.0,
        )
        print(report.summary())
    print(f"  Laps completed: {result.laps_completed}/{num_laps}")

    # ── Validation plots ──
    plot_path = "results/validation_plots.png"
    plot_validation(result.states, aim_df, output_path=plot_path)
    print(f"  Validation plots saved to {plot_path}")

    # ── 2. Corner Speed Comparison ──
    print("\n2. CORNER SPEED PREDICTION -- Pacejka vs Legacy")
    print("-" * 50)

    corner_segments = [s for s in track.segments if abs(s.curvature) > 0.01]
    rows = []
    for seg in corner_segments:
        legacy_v = legacy_dyn.max_cornering_speed(seg.curvature, seg.grip_factor)
        new_v = new_dyn.max_cornering_speed(seg.curvature, seg.grip_factor)
        radius = 1.0 / abs(seg.curvature)
        rows.append({
            "radius_m": radius,
            "legacy_kmh": min(legacy_v * 3.6, 999),
            "pacejka_kmh": min(new_v * 3.6, 999),
        })

    df = pd.DataFrame(rows).sort_values("radius_m")
    tight = df[df.radius_m < 15]
    medium = df[(df.radius_m >= 15) & (df.radius_m < 30)]
    wide = df[df.radius_m >= 30]

    print(f"  Corner Type     Count   Legacy 1.3G    Pacejka Tire")
    if len(tight) > 0:
        print(f"  Tight (<15m)    {len(tight):>3d}     {tight.legacy_kmh.mean():>5.1f} km/h     {tight.pacejka_kmh.mean():>5.1f} km/h")
    if len(medium) > 0:
        print(f"  Medium (15-30m) {len(medium):>3d}     {medium.legacy_kmh.mean():>5.1f} km/h     {medium.pacejka_kmh.mean():>5.1f} km/h")
    if len(wide) > 0:
        print(f"  Wide (>30m)     {len(wide):>3d}     {wide.legacy_kmh.mean():>5.1f} km/h     {wide.pacejka_kmh.mean():>5.1f} km/h")

    # Telemetry lateral G reference
    moving = aim_df[aim_df["GPS Speed"] > 10.0]
    actual_peak_g = moving["GPS LatAcc"].abs().quantile(0.95)
    cornering_mask = moving["GPS LatAcc"].abs() > 0.5
    actual_mean_g = moving.loc[cornering_mask, "GPS LatAcc"].abs().mean()

    print(f"\n  Telemetry lateral G: peak(95th)={actual_peak_g:.2f}g, mean(corners)={actual_mean_g:.2f}g")
    print(f"  Legacy assumption: 1.30g constant")
    pdy1 = abs(tire.lateral["PDY1"])
    print(f"  Pacejka peak mu at nominal load: {pdy1:.2f}")

    # ── 3. Force-Based Strategy Comparison ──
    print("\n3. FORCE-BASED STRATEGIES -- 1-Lap Comparison")
    print("-" * 50)

    # CoastOnly with Pacejka dynamics
    coast_strat = CoastOnlyStrategy(new_dyn, coast_margin_ms=2.0)
    batt_coast = BatteryModel(config.battery)
    batt_coast.calibrate_from_voltt(voltt_df)
    engine_coast = SimulationEngine(config, track, coast_strat, batt_coast)
    # Override dynamics to use our Pacejka-equipped one
    engine_coast.dynamics = new_dyn
    result_coast = engine_coast.run(num_laps=1, initial_soc_pct=initial_soc, initial_temp_c=initial_temp)

    # ThresholdBraking with Pacejka
    brake_strat = ThresholdBrakingStrategy(new_dyn, coast_margin_ms=3.0, brake_threshold_ms=1.0, brake_intensity=0.5)
    batt_brake = BatteryModel(config.battery)
    batt_brake.calibrate_from_voltt(voltt_df)
    engine_brake = SimulationEngine(config, track, brake_strat, batt_brake)
    engine_brake.dynamics = new_dyn
    result_brake = engine_brake.run(num_laps=1, initial_soc_pct=initial_soc, initial_temp_c=initial_temp)

    # CoastOnly with legacy dynamics
    coast_legacy = CoastOnlyStrategy(legacy_dyn, coast_margin_ms=2.0)
    batt_leg = BatteryModel(config.battery)
    batt_leg.calibrate_from_voltt(voltt_df)
    engine_leg = SimulationEngine(config, track, coast_legacy, batt_leg)
    engine_leg.dynamics = legacy_dyn
    result_leg = engine_leg.run(num_laps=1, initial_soc_pct=initial_soc, initial_temp_c=initial_temp)

    # Telemetry lap reference
    laps = detect_lap_boundaries(aim_df)
    telem_lap = laps[1][2] if len(laps) >= 2 else 70.6

    print(f"  Strategy               Lap Time    Avg Speed    Energy/Lap")
    print(f"  Telemetry (Lap 2)      {telem_lap:>6.1f} s     --           --")
    print(f"  CoastOnly (Pacejka)    {result_coast.total_time_s:>6.1f} s     {result_coast.states['speed_kmh'].mean():>5.1f} km/h   {result_coast.total_energy_kwh:.3f} kWh")
    print(f"  ThreshBrake (Pacejka)  {result_brake.total_time_s:>6.1f} s     {result_brake.states['speed_kmh'].mean():>5.1f} km/h   {result_brake.total_energy_kwh:.3f} kWh")
    print(f"  CoastOnly (Legacy)     {result_leg.total_time_s:>6.1f} s     {result_leg.states['speed_kmh'].mean():>5.1f} km/h   {result_leg.total_energy_kwh:.3f} kWh")

    # ── 4. Tire Model Characteristics ──
    print("\n4. TIRE MODEL CHARACTERISTICS (Hoosier LC0 10psi)")
    print("-" * 50)
    print(f"  Load (N)   Peak Fy (N)   mu_y     Peak Fx (N)")
    for fz in [300, 500, 657, 800, 1000]:
        peak_fy = tire.peak_lateral_force(float(fz))
        peak_fx = tire.peak_longitudinal_force(float(fz))
        mu_y = peak_fy / fz
        print(f"  {fz:>7.0f}     {peak_fy:>8.0f}     {mu_y:.2f}     {peak_fx:>8.0f}")

    # ── 5. Load Transfer at Key Conditions ──
    print("\n5. LOAD TRANSFER AT KEY CONDITIONS")
    print("-" * 50)
    fl, fr, rl, rr = lt.tire_loads(0.0, 0.0, 0.0)
    print(f"  Static:   FL={fl:.0f}  FR={fr:.0f}  RL={rl:.0f}  RR={rr:.0f}  Total={fl+fr+rl+rr:.0f} N")
    fl, fr, rl, rr = lt.tire_loads(22.22, 0.0, 0.0)
    print(f"  80 kph:   FL={fl:.0f}  FR={fr:.0f}  RL={rl:.0f}  RR={rr:.0f}  Total={fl+fr+rl+rr:.0f} N")
    fl, fr, rl, rr = lt.tire_loads(22.22, 1.0, 0.0)
    print(f"  80kph+1G: FL={fl:.0f}  FR={fr:.0f}  RL={rl:.0f}  RR={rr:.0f}  Total={fl+fr+rl+rr:.0f} N")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Replay validation:     {report.num_passed}/{report.num_total} metrics pass")
    print(f"  Worst metric error:    {max(m.relative_error_pct for m in report.metrics):.1f}%")
    print(f"  Pacejka vs Legacy:     Higher corner speeds (mu={pdy1:.1f} vs 1.3)")
    print(f"  Real car lateral G:    {actual_peak_g:.2f}g peak, {actual_mean_g:.2f}g mean in corners")
    print(f"  Force-based lap delta: {result_coast.total_time_s - result_leg.total_time_s:+.1f}s (Pacejka vs Legacy coast)")


if __name__ == "__main__":
    main()
