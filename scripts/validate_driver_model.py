"""End-to-end validation of the calibrated driver model.

Pipeline:
1. Load AiM telemetry and track geometry
2. Calibrate CalibratedStrategy from telemetry
3. Run 22-lap simulation with the calibrated strategy
4. Compare sim results to telemetry
5. Score with FSAE scoring function
"""

import sys

import numpy as np

from fsae_sim.data.loader import load_aim_csv, load_cleaned_csv, load_voltt_csv
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.driver.strategies import PedalProfileStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.analysis.validation import validate_full_endurance, detect_lap_boundaries
from fsae_sim.analysis.scoring import FSAEScoring


def main():
    print("=" * 70)
    print("CALIBRATED DRIVER MODEL VALIDATION")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/6] Loading data...")
    config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
    _, aim_df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    voltt_df = load_voltt_csv(
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
    )
    track = Track.from_telemetry(df=aim_df)

    print(f"  Track: {track.name}, {track.num_segments} segments, "
          f"{track.total_distance_m:.0f}m per lap")

    # ── Detect laps ──
    print("\n[2/6] Detecting laps...")
    laps = detect_lap_boundaries(aim_df)
    print(f"  Detected {len(laps)} laps")

    # ── Calibrate strategy ──
    print("\n[3/6] Calibrating pedal-profile driver model from telemetry...")
    strategy = PedalProfileStrategy.from_telemetry(aim_df, track)

    # Profile summary
    n_throttle = int(np.sum(strategy._actions == 1))
    n_coast = int(np.sum(strategy._actions == 0))
    n_brake = int(np.sum(strategy._actions == 2))
    print(f"  {strategy.num_segments} segments: "
          f"{n_throttle} throttle, {n_coast} coast, {n_brake} brake")
    throttle_segs = strategy._throttle_pct[strategy._actions == 1]
    if len(throttle_segs) > 0:
        print(f"  Throttle pedal: mean={np.mean(throttle_segs)*100:.1f}%, "
              f"median={np.median(throttle_segs)*100:.1f}%, "
              f"max={np.max(throttle_segs)*100:.1f}%")
    brake_segs = strategy._brake_pct[strategy._actions == 2]
    if len(brake_segs) > 0:
        print(f"  Brake pressure: mean={np.mean(brake_segs)*100:.1f}%, "
              f"median={np.median(brake_segs)*100:.1f}%, "
              f"max={np.max(brake_segs)*100:.1f}%")

    # ── Profile quality checks ──
    print("\n[4/6] Profile quality checks...")
    all_ok = True
    if n_throttle == 0:
        print("  WARNING: No throttle segments found")
        all_ok = False
    if n_coast == 0:
        print("  WARNING: No coast segments found")
        all_ok = False
    total_pct = 100.0 * (n_throttle + n_coast + n_brake) / strategy.num_segments
    if total_pct < 99.9:
        print(f"  WARNING: Only {total_pct:.1f}% of segments classified")
        all_ok = False
    if all_ok:
        print("  All profile quality checks passed")

    # ── Run simulation ──
    print("\n[5/6] Running 22-lap simulation...")
    battery = BatteryModel(config.battery, cell_capacity_ah=4.5)
    battery.calibrate(voltt_df)
    battery.calibrate_pack_from_telemetry(aim_df)

    engine = SimulationEngine(config, track, strategy, battery)
    # Start temp from telemetry: real car started at 29°C
    result = engine.run(
        num_laps=22,
        initial_soc_pct=95.0,
        initial_temp_c=29.0,
    )

    print(f"  Completed {result.laps_completed} laps")
    print(f"  Total time:   {result.total_time_s:.1f}s")
    print(f"  Total energy: {result.total_energy_kwh:.3f} kWh")
    print(f"  Final SOC:    {result.final_soc:.1f}%")

    # ── Validate against telemetry ──
    print("\n[6/6] Validating against telemetry...")
    report = validate_full_endurance(
        result.states, aim_df,
        sim_total_time_s=result.total_time_s,
        sim_final_soc=result.final_soc,
        sim_total_energy_kwh=result.total_energy_kwh,
        sim_laps=result.laps_completed,
    )
    print(f"\n{report.summary()}")

    # ── FSAE Scoring ──
    print("\n" + "=" * 70)
    print("FSAE SCORING (Michigan 2025 Field)")
    print("=" * 70)

    scorer = FSAEScoring.michigan_2025_field()
    track_km = track.total_distance_m / 1000.0
    score = scorer.score_sim_result(result, track_distance_km=track_km)

    print(f"\n  Endurance time score:  {score.endurance_time_score:6.1f} / 250")
    print(f"  Endurance laps score: {score.endurance_laps_score:6.1f} /  25")
    print(f"  Endurance total:      {score.endurance_total:6.1f} / 275")
    print(f"  Efficiency factor:    {score.efficiency_factor:6.3f}")
    print(f"  Efficiency score:     {score.efficiency_score:6.1f} / 100")
    print(f"  Combined score:       {score.combined_score:6.1f} / 375")
    print(f"\n  Corrected time:   {score.your_time_s:.1f}s")
    print(f"  Energy consumed:  {score.your_energy_kwh:.3f} kWh")
    print(f"  CO2 equivalent:   {score.your_co2_kg:.3f} kg")
    print(f"  Avg lap time:     {score.your_avg_lap_s:.2f}s")
    print(f"  CO2 per lap:      {score.your_co2_per_lap:.4f} kg")

    # ── Compare to actual 2025 results ──
    print(f"\n  Actual 2025 results:")
    print(f"    Endurance: 152.9 pts")
    print(f"    Efficiency: 100.0 pts")
    print(f"    Combined: 252.9 pts")

    actual_combined = 252.9
    score_error = abs(score.combined_score - actual_combined) / actual_combined * 100
    print(f"\n  Score error: {score_error:.1f}% (target: <5%)")

    print("\n" + "=" * 70)
    if report.all_passed:
        print("VALIDATION PASSED")
    else:
        print(f"VALIDATION: {report.num_passed}/{report.num_total} metrics passed")
    print("=" * 70)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
