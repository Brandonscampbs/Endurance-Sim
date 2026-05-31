"""Numerical snapshot for the simulator.

Runs both the calibrated and replay engines and prints the per-segment
hash + summary totals. Use before and after a refactor to confirm whether
aggregate outputs and per-channel arrays changed.
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.analysis.validation import (  # noqa: E402
    detect_lap_boundaries,
    telemetry_speed_col,
)
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.driver.strategies import (  # noqa: E402
    CalibratedStrategy,
    ReplayStrategy,
)
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402

CONFIG = REPO / "configs" / "ct16ev.yaml"
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = (
    REPO / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)


def array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def snapshot_calibrated():
    vehicle = VehicleConfig.from_yaml(str(CONFIG))
    _, aim_df = load_cleaned_csv(str(TELEM))
    speed_col = telemetry_speed_col(aim_df)
    track = Track.from_telemetry(df=aim_df)
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
    battery.calibrate_pack_from_telemetry(aim_df, allow_same_run_validation=True)
    strategy = CalibratedStrategy.from_telemetry(
        aim_df, track, speed_col=speed_col, use_observed_speed_caps=False,
    )
    engine = SimulationEngine(
        vehicle, track, strategy, battery, mode=SimulationMode.VALIDATION,
    )
    laps = detect_lap_boundaries(aim_df)
    initial_speed_ms = float(aim_df[speed_col].iloc[laps[0][0]]) / 3.6
    result = engine.run(
        num_laps=len(laps) + 1, initial_soc_pct=95.0, initial_temp_c=29.0,
        initial_speed_ms=initial_speed_ms,
    )
    return result


def snapshot_replay():
    vehicle = VehicleConfig.from_yaml(str(CONFIG))
    _, aim_df = load_cleaned_csv(str(TELEM))
    speed_col = telemetry_speed_col(aim_df)
    track = Track.from_telemetry_full_recording(df=aim_df)
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
    battery.calibrate_pack_from_telemetry(aim_df, allow_same_run_validation=True)
    strategy = ReplayStrategy.from_full_endurance(
        aim_df, track.total_distance_m, trim_to_lap_start=False,
    )
    engine = SimulationEngine(
        vehicle, track, strategy, battery, mode=SimulationMode.REPLAY,
    )
    laps = detect_lap_boundaries(aim_df)
    initial_speed_ms = float(aim_df[speed_col].iloc[laps[0][0]]) / 3.6
    result = engine.run(
        num_laps=1, initial_soc_pct=95.0, initial_temp_c=29.0,
        initial_speed_ms=initial_speed_ms,
    )
    return result


def report(label: str, result) -> None:
    states = result.states
    print(f"=== {label} ===")
    print(f"  segments         : {len(states)}")
    print(f"  laps_completed   : {result.laps_completed}")
    print(f"  total_time_s     : {result.total_time_s:.6f}")
    print(f"  net_energy_kwh   : {result.net_energy_kwh:.9f}")
    print(f"  discharge_kwh    : {result.discharge_energy_kwh:.9f}")
    print(f"  regen_kwh        : {result.regen_energy_kwh:.9f}")
    print(f"  net_charge_ah    : {result.net_charge_ah:.6f}")
    print(f"  final_soc        : {result.final_soc:.6f}")
    print(f"  envelope_recomp. : {result.envelope_recomputes}")
    # Per-channel hashes: exact array fingerprints. These can change from
    # roundoff-level operation-order differences even when aggregate metrics
    # are unchanged, so interpret them alongside the summary totals.
    for col in (
        "speed_ms", "soc_pct", "pack_current_a", "motor_torque_nm",
        "drive_force_n", "regen_force_n", "brake_force_n",
        "resistance_force_n", "electrical_power_w",
    ):
        if col in states.columns:
            print(f"  hash[{col:<22}] = {array_hash(states[col].values)}")


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = snapshot_calibrated()
        report("calibrated", cal)
        rep = snapshot_replay()
        report("replay", rep)


if __name__ == "__main__":
    main()
