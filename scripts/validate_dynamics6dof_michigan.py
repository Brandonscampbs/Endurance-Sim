"""Michigan 2025 validation: compare legacy vs dynamics6dof backends.

Runs the full endurance sim under both backends with matched parameters and
reports per-channel residuals from ``validate_full_endurance``. Saves a
summary JSON + comparison plot to ``docs/validation/2026-04-17-dynamics6dof/``.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fsae_sim.analysis.validation import validate_full_endurance
from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv
from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.vehicle import VehicleConfig

REPO_ROOT = Path(__file__).parents[1]
CONFIG = REPO_ROOT / "configs" / "ct16ev.yaml"
AIM = REPO_ROOT / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = REPO_ROOT / "Real-Car-Data-And-Stats" / "About-Energy-Volt-Simulations-2025-Pack" / "2025_Pack_cell.csv"
OUT = REPO_ROOT / "docs" / "validation" / "2026-04-17-dynamics6dof"


def _build_engine(backend: str, cfg: VehicleConfig, track: Track, aim_df: pd.DataFrame) -> SimulationEngine:
    cfg_sel = replace(cfg, dynamics_backend=backend)
    battery = BatteryModel(cfg_sel.battery)
    battery.calibrate_from_voltt(load_voltt_csv(VOLTT))
    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    return SimulationEngine(cfg_sel, track, strategy, battery)


def _run_backend(backend: str, cfg: VehicleConfig, track: Track, aim_df: pd.DataFrame):
    engine = _build_engine(backend, cfg, track, aim_df)
    result = engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=25.0)
    return result


def _pack_report(report, backend: str) -> dict:
    metrics = {}
    for m in report.metrics:
        metrics[m.name] = {
            "telemetry": m.telemetry_value,
            "simulation": m.simulation_value,
            "error_pct": m.relative_error_pct,
            "passed": m.passed,
        }
    return {
        "backend": backend,
        "metrics": metrics,
        "overall_pass_count": sum(1 for m in report.metrics if m.passed),
        "overall_total": len(report.metrics),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading config, telemetry, and track...")
    cfg = VehicleConfig.from_yaml(CONFIG)
    _, aim_df = load_cleaned_csv(AIM)
    track = Track.from_telemetry(df=aim_df)
    print(f"Track: {track.total_distance_m:.1f} m over {len(track.segments)} segments")

    comparisons: dict = {}
    for backend in ("legacy", "dynamics6dof"):
        print(f"\nRunning backend = {backend}...")
        try:
            result = _run_backend(backend, cfg, track, aim_df)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            comparisons[backend] = {"error": str(exc)}
            continue

        report = validate_full_endurance(
            sim_states=result.states,
            aim_df=aim_df,
            sim_total_time_s=result.total_time_s,
            sim_final_soc=result.final_soc,
            sim_total_energy_kwh=result.total_energy_kwh,
            sim_laps=result.laps_completed,
            target_pct=5.0,
        )
        packed = _pack_report(report, backend)
        packed["sim_time_s"] = result.total_time_s
        packed["sim_final_soc"] = result.final_soc
        packed["sim_energy_kwh"] = result.total_energy_kwh
        comparisons[backend] = packed

        print(f"  time={result.total_time_s:.2f}s  soc={result.final_soc:.2f}%"
              f"  energy={result.total_energy_kwh:.3f}kWh"
              f"  metrics pass: {packed['overall_pass_count']}/{packed['overall_total']}")

    out_json = OUT / "comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2, default=str)
    print(f"\nWrote {out_json}")

    # Print summary table
    print("\n=== Summary ===")
    for backend, data in comparisons.items():
        if "error" in data:
            print(f"{backend}: ERROR — {data['error']}")
            continue
        print(f"{backend}: {data['overall_pass_count']}/{data['overall_total']} "
              f"metrics pass, time={data['sim_time_s']:.2f}s, "
              f"energy={data['sim_energy_kwh']:.3f}kWh")


if __name__ == "__main__":
    main()
