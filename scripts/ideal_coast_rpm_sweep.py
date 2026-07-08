"""RPM sweep for the ideal no-brake coast driver."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.analysis.scoring import FSAEScoring  # noqa: E402
from fsae_sim.analysis.validation import (  # noqa: E402
    detect_lap_boundaries,
    telemetry_speed_col,
)
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.driver.strategies import CoastOptimalStrategy  # noqa: E402
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402

CONFIG = REPO / "configs" / "ct16ev.yaml"
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT = (
    REPO
    / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)
RESULTS_PDF = REPO / "Real-Car-Data-And-Stats" / "FSAE_2025_MI6_results.pdf"

RPM_VALUES = tuple(range(2500, 4001, 100))
ENDURANCE_LAPS = 22
VALIDATION_HOLDOUT_LAPS_ZERO_BASED = tuple(range(12, 22))

_CTX: dict[str, Any] = {}


def _init_worker() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vehicle = VehicleConfig.from_yaml(str(CONFIG))
        _, aim_df = load_cleaned_csv(str(TELEM))
        speed_col = telemetry_speed_col(aim_df)
        detected_laps = detect_lap_boundaries(aim_df)
        validation_laps = [
            lap for lap in VALIDATION_HOLDOUT_LAPS_ZERO_BASED
            if lap < len(detected_laps)
        ]
        track = Track.from_telemetry(df=aim_df)
        battery = BatteryModel.from_config_and_data(
            vehicle.battery,
            str(VOLTT),
        )
        battery.calibrate_pack_from_telemetry(
            aim_df,
            holdout_laps=tuple(lap + 1 for lap in validation_laps),
        )
        initial_speed_ms = (
            float(aim_df[speed_col].iloc[detected_laps[0][0]]) / 3.6
            if detected_laps else 0.0
        )

    _CTX.update({
        "vehicle": vehicle,
        "track": track,
        "battery": battery,
        "initial_speed_ms": initial_speed_ms,
    })


def _vehicle_with_rpm(rpm: int) -> VehicleConfig:
    vehicle = _CTX["vehicle"]
    return replace(
        vehicle,
        powertrain=replace(
            vehicle.powertrain,
            motor_speed_max_rpm=float(rpm),
            lvcu_overspeed_rpm=float(rpm),
        ),
    )


def run_case(rpm: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vehicle = _vehicle_with_rpm(rpm)
            track = _CTX["track"]
            battery = copy.copy(_CTX["battery"])
            battery.violations = []
            strategy = CoastOptimalStrategy.from_config(vehicle, track)
            result = SimulationEngine(
                vehicle,
                track,
                strategy,
                battery,
                mode=SimulationMode.PREDICTION,
                allow_telemetry_track=True,
                allow_empirical_grip=True,
            ).run(
                num_laps=ENDURANCE_LAPS,
                initial_soc_pct=95.0,
                initial_temp_c=29.0,
                initial_speed_ms=float(_CTX["initial_speed_ms"]),
            )

        track_km_per_lap = track.total_distance_m / 1000.0
        score = FSAEScoring.michigan_2025_field().score(
            total_time_s=float(result.total_time_s),
            total_energy_kwh=float(result.net_energy_kwh),
            laps_completed=ENDURANCE_LAPS,
            total_distance_km=track_km_per_lap * ENDURANCE_LAPS,
            track_km_per_lap=track_km_per_lap,
            driver_change_completed=True,
        )
        states = result.states
        action_counts = (
            states["action"].value_counts().to_dict()
            if "action" in states else {}
        )
        return {
            "status": "ok",
            "rpm": int(rpm),
            "combined_score": float(score.combined_score),
            "endurance_total": float(score.endurance_total),
            "endurance_time_score": float(score.endurance_time_score),
            "endurance_laps_score": float(score.endurance_laps_score),
            "efficiency_score": float(score.efficiency_score),
            "efficiency_factor": float(score.efficiency_factor),
            "time_s": float(result.total_time_s),
            "net_kwh": float(result.net_energy_kwh),
            "net_ah": float(result.net_charge_ah),
            "final_soc_pct": float(result.final_soc),
            "peak_speed_kmh": float(states["speed_kmh"].max()),
            "peak_motor_rpm": float(states["motor_rpm"].max()),
            "rpm_limited_pct": float(
                (states["motor_rpm"] >= float(rpm) * 0.995).mean() * 100.0
            ),
            "brake_force_max_n": float(states["brake_force_n"].max()),
            "regen_force_min_n": float(states["regen_force_n"].min()),
            "throttle_segments": int(action_counts.get("throttle", 0)),
            "coast_segments": int(action_counts.get("coast", 0)),
            "duration_s": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "status": "error",
            "rpm": int(rpm),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: int(row["rpm"]))
    fields = [
        "rpm",
        "combined_score",
        "endurance_total",
        "endurance_time_score",
        "endurance_laps_score",
        "efficiency_score",
        "efficiency_factor",
        "time_s",
        "net_kwh",
        "net_ah",
        "final_soc_pct",
        "peak_speed_kmh",
        "peak_motor_rpm",
        "rpm_limited_pct",
        "brake_force_max_n",
        "regen_force_min_n",
        "throttle_segments",
        "coast_segments",
        "duration_s",
    ]
    with (out_dir / "ideal_coast_rpm_sweep.csv").open(
        "w", newline="", encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "ideal_coast",
        "rpm_values": list(RPM_VALUES),
        "scoring": {
            "source": str(RESULTS_PDF.relative_to(REPO)),
            "endurance_tmin_s": 1369.936,
            "efficiency_co2min_kg_per_lap": 0.0967,
            "efficiency_efmax": 0.848,
            "laps_completed": ENDURANCE_LAPS,
            "penalties": "none",
        },
        "rows": rows,
    }
    (out_dir / "ideal_coast_rpm_sweep.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Ideal Coast RPM Sweep",
        "",
        "Model: `ideal_coast`. Scored against 2025 FSAE Michigan endurance "
        "and efficiency field, no penalties.",
        "",
        "| RPM | Points | Endurance | Efficiency | Time (s) | Net kWh | Peak mph | Peak RPM | RPM-limited % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        peak_mph = float(row["peak_speed_kmh"]) * 0.621371
        lines.append(
            "| "
            + " | ".join([
                str(row["rpm"]),
                f"{row['combined_score']:.1f}",
                f"{row['endurance_total']:.1f}",
                f"{row['efficiency_score']:.1f}",
                f"{row['time_s']:.1f}",
                f"{row['net_kwh']:.3f}",
                f"{peak_mph:.1f}",
                f"{row['peak_motor_rpm']:.0f}",
                f"{row['rpm_limited_pct']:.1f}",
            ])
            + " |"
        )
    (out_dir / "ideal_coast_rpm_sweep.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rpms", type=int, nargs="*", default=list(RPM_VALUES))
    args = parser.parse_args()

    out_dir = args.out_dir or (
        REPO / "results" / f"ideal_coast_rpm_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    jobs = [int(rpm) for rpm in args.rpms]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"output_dir={out_dir}")
    print(f"jobs={len(jobs)} workers={args.workers}")

    with progress_path.open("a", encoding="utf-8") as progress:
        progress.write(json.dumps({
            "event": "start",
            "jobs": len(jobs),
            "workers": args.workers,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }) + "\n")
        progress.flush()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
        ) as pool:
            future_to_rpm = {pool.submit(run_case, rpm): rpm for rpm in jobs}
            for idx, future in enumerate(as_completed(future_to_rpm), start=1):
                row = future.result()
                row["completed_index"] = idx
                row["total_jobs"] = len(jobs)
                row["elapsed_s"] = time.perf_counter() - started
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                if row.get("status") == "ok":
                    rows.append(row)
                    print(
                        f"[{idx:02d}/{len(jobs)}] {row['rpm']} rpm -> "
                        f"{row['combined_score']:.1f} pts "
                        f"({row['time_s']:.1f}s, {row['net_kwh']:.3f}kWh)"
                    )
                else:
                    print(f"[{idx:02d}/{len(jobs)}] ERROR {row['rpm']}: {row['error']}")

    if len(rows) != len(jobs):
        print(f"completed {len(rows)} of {len(jobs)} successfully")
        print(f"see {progress_path}")
        return 2
    _write_outputs(rows, out_dir)
    print(f"wrote {out_dir / 'ideal_coast_rpm_sweep.csv'}")
    print(f"wrote {out_dir / 'ideal_coast_rpm_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
