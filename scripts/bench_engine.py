"""Throwaway micro-benchmark for the simulator engine.

Mirrors `sim_compare.py` setup but times just `SimulationEngine.run` for
both `replay` and `driver` strategies. Also runs cProfile on the
calibrated path and prints top-30 cumulative time and total-time hot
spots. Intended for local one-off perf investigation; not a regression
fixture.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

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
from fsae_sim.vehicle.dynamics import VehicleDynamics  # noqa: E402


CONFIG = REPO / "configs" / "ct16ev.yaml"
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = (
    REPO / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)


def build_calibrated():
    vehicle = VehicleConfig.from_yaml(str(CONFIG))
    _, aim_df = load_cleaned_csv(str(TELEM))
    speed_col = telemetry_speed_col(aim_df)
    track = Track.from_telemetry(df=aim_df)
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
    battery.calibrate_pack_from_telemetry(aim_df, allow_same_run_validation=True)
    strategy = CalibratedStrategy.from_telemetry(
        aim_df, track, speed_col=speed_col,
        use_observed_speed_caps=False,
    )
    engine = SimulationEngine(
        vehicle, track, strategy, battery,
        mode=SimulationMode.VALIDATION,
    )
    _laps = detect_lap_boundaries(aim_df)
    return engine, len(_laps), aim_df, speed_col, _laps


def build_replay():
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
    return engine, aim_df, speed_col


def time_run(engine, num_laps, initial_speed_ms, label):
    # Warm-up to let any first-call costs (envelope, jit) hit before timing.
    t0 = time.perf_counter()
    result = engine.run(
        num_laps=num_laps, initial_soc_pct=95.0, initial_temp_c=29.0,
        initial_speed_ms=initial_speed_ms,
    )
    t1 = time.perf_counter()
    print(
        f"[{label}] warm-up: {t1 - t0:.3f}s  "
        f"segments={len(result.states)}  laps={result.laps_completed}"
    )

    # Three timed runs (each is a fresh engine.run on the same engine
    # instance, since engine.run is stateless on its arguments).
    times: list[float] = []
    for i in range(3):
        t0 = time.perf_counter()
        result = engine.run(
            num_laps=num_laps, initial_soc_pct=95.0, initial_temp_c=29.0,
            initial_speed_ms=initial_speed_ms,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"[{label}] run {i + 1}: {t1 - t0:.3f}s")
    print(
        f"[{label}] min={min(times):.3f}s  median={sorted(times)[1]:.3f}s  "
        f"mean={sum(times) / len(times):.3f}s"
    )
    return result, times


def profile_run(engine, num_laps, initial_speed_ms, label, top_n=30):
    pr = cProfile.Profile()
    pr.enable()
    engine.run(
        num_laps=num_laps, initial_soc_pct=95.0, initial_temp_c=29.0,
        initial_speed_ms=initial_speed_ms,
    )
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).strip_dirs()

    print(f"\n=== [{label}] Top {top_n} by cumulative time ===")
    stats.sort_stats("cumulative").print_stats(top_n)
    print(buf.getvalue())

    buf2 = io.StringIO()
    stats2 = pstats.Stats(pr, stream=buf2).strip_dirs()
    print(f"\n=== [{label}] Top {top_n} by total (self) time ===")
    stats2.sort_stats("tottime").print_stats(top_n)
    print(buf2.getvalue())


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("--- Building calibrated engine ---")
        cal_engine, num_lap_intervals, aim_df, speed_col, _laps = build_calibrated()
        initial_speed_ms_cal = (
            float(aim_df[speed_col].iloc[_laps[0][0]]) / 3.6
            if _laps else 0.0
        )
        cal_num_laps = (len(_laps) + 1) if _laps else 22

        print("--- Building replay engine ---")
        rep_engine, aim_df_r, speed_col_r = build_replay()
        _laps_r = detect_lap_boundaries(aim_df_r)
        initial_speed_ms_rep = (
            float(aim_df_r[speed_col_r].iloc[_laps_r[0][0]]) / 3.6
            if _laps_r else 0.0
        )

        print("\n=== Calibrated timing ===")
        time_run(cal_engine, cal_num_laps, initial_speed_ms_cal, "calibrated")

        print("\n=== Replay timing ===")
        time_run(rep_engine, 1, initial_speed_ms_rep, "replay")

        print("\n--- cProfile calibrated ---")
        profile_run(cal_engine, cal_num_laps, initial_speed_ms_cal, "calibrated")

        print("\n--- cProfile replay ---")
        profile_run(rep_engine, 1, initial_speed_ms_rep, "replay")


if __name__ == "__main__":
    main()
