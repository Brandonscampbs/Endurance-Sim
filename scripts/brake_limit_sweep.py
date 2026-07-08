"""Sweep lighter mechanical braking for the corrected no-regen command-cap model."""

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
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import scripts.command_power_limit_sweep as command_sweep  # noqa: E402
from fsae_sim.analysis.scoring import FSAEScoring  # noqa: E402
from fsae_sim.driver.adaptive import AdaptiveDriverParams  # noqa: E402
from fsae_sim.driver.strategies import AdaptiveStrategy  # noqa: E402
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.sim.speed_envelope import SpeedEnvelope  # noqa: E402

ENDURANCE_LAPS = 22


def _init_worker() -> None:
    command_sweep._init_worker()


def run_case(
    torque_nm: float,
    rpm: int,
    power_kw: float,
    brake_g: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vehicle = command_sweep._vehicle_with_limits(torque_nm, rpm)
            track = command_sweep._CTX["track"]
            battery = copy.copy(command_sweep._CTX["battery"])
            battery.violations = []
            params = AdaptiveDriverParams(regen_enabled=False)
            strategy = AdaptiveStrategy.from_config(vehicle, track, params=params)
            engine = SimulationEngine(
                vehicle,
                track,
                strategy,
                battery,
                mode=SimulationMode.PREDICTION,
                allow_telemetry_track=True,
                allow_empirical_grip=True,
            )
            engine.powertrain = command_sweep.CommandPowerLimitedPowertrain(
                engine.powertrain, power_kw,
            )
            engine.dynamics.max_brake_decel_g = float(brake_g)
            engine.dynamics._MAX_BRAKE_DECEL_G = float(brake_g)
            strategy.bind_models(engine.dynamics, engine.powertrain)
            engine._envelope = SpeedEnvelope(engine.dynamics, engine.powertrain, track)
            result = engine.run(
                num_laps=ENDURANCE_LAPS,
                initial_soc_pct=95.0,
                initial_temp_c=29.0,
                initial_speed_ms=float(command_sweep._CTX["initial_speed_ms"]),
            )

        states = result.states
        track_km_per_lap = track.total_distance_m / 1000.0
        score = FSAEScoring.michigan_2025_field().score(
            total_time_s=float(result.total_time_s),
            total_energy_kwh=float(result.net_energy_kwh),
            laps_completed=int(result.laps_completed),
            total_distance_km=track_km_per_lap * int(result.laps_completed),
            track_km_per_lap=track_km_per_lap,
            driver_change_completed=result.laps_completed >= ENDURANCE_LAPS,
        )
        action_counts = states["action"].value_counts().to_dict()
        braking = states[(states["brake_pct"] > 1e-6) | (states["brake_force_n"] > 1e-6)]
        brake_time_s = float(braking["segment_time_s"].sum()) if len(braking) else 0.0
        brake_energy_kwh = (
            float(states["mechanical_brake_energy_j"].sum()) / 3.6e6
            if "mechanical_brake_energy_j" in states else 0.0
        )
        return {
            "status": "ok",
            "torque_nm": float(torque_nm),
            "rpm": int(rpm),
            "power_limit_kw": float(power_kw),
            "brake_g": float(brake_g),
            "combined_score": float(score.combined_score),
            "endurance_total": float(score.endurance_total),
            "efficiency_score": float(score.efficiency_score),
            "efficiency_factor": float(score.efficiency_factor),
            "time_s": float(result.total_time_s),
            "net_kwh": float(result.net_energy_kwh),
            "discharge_kwh": float(result.discharge_energy_kwh),
            "regen_kwh": float(result.regen_energy_kwh),
            "brake_energy_kwh": brake_energy_kwh,
            "brake_time_s": brake_time_s,
            "brake_time_pct": float(brake_time_s / result.total_time_s * 100.0),
            "mean_brake_pct_when_braking": (
                float(braking["brake_pct"].mean()) if len(braking) else 0.0
            ),
            "max_brake_pct": float(states["brake_pct"].max()),
            "final_soc_pct": float(result.final_soc),
            "laps_completed": int(result.laps_completed),
            "peak_speed_kmh": float(states["speed_kmh"].max()),
            "peak_motor_rpm": float(states["motor_rpm"].max()),
            "peak_motor_torque_nm": float(states["motor_torque_nm"].max()),
            "peak_pack_current_a": float(states["pack_current_a"].max()),
            "max_cell_temp_c": float(states["cell_temp_c"].max()),
            "min_longitudinal_g": float(states["longitudinal_g"].min()),
            "speed_limited_pct": float(states["speed_limit_active"].mean() * 100.0),
            "speed_limit_violations": int(states["speed_limit_violation"].sum()),
            "throttle_segments": int(action_counts.get("throttle", 0)),
            "brake_segments": int(action_counts.get("brake", 0)),
            "coast_segments": int(action_counts.get("coast", 0)),
            "duration_s": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "status": "error",
            "torque_nm": float(torque_nm),
            "rpm": int(rpm),
            "power_limit_kw": float(power_kw),
            "brake_g": float(brake_g),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(
        rows,
        key=lambda r: (
            float(r["brake_g"]),
            float(r["torque_nm"]),
            int(r["rpm"]),
            float(r["power_limit_kw"]),
        ),
    )
    fields = [
        "brake_g", "torque_nm", "rpm", "power_limit_kw",
        "combined_score", "endurance_total", "efficiency_score",
        "efficiency_factor", "time_s", "net_kwh", "discharge_kwh",
        "regen_kwh", "brake_energy_kwh", "brake_time_s", "brake_time_pct",
        "mean_brake_pct_when_braking", "max_brake_pct", "final_soc_pct",
        "laps_completed", "peak_speed_kmh", "peak_motor_rpm",
        "peak_motor_torque_nm", "peak_pack_current_a", "max_cell_temp_c",
        "min_longitudinal_g", "speed_limited_pct", "speed_limit_violations",
        "throttle_segments", "brake_segments", "coast_segments", "duration_s",
    ]
    with (out_dir / "brake_limit_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    ranked = sorted(rows, key=lambda r: float(r["combined_score"]), reverse=True)
    best_by_brake: dict[float, dict[str, Any]] = {}
    for row in ranked:
        best_by_brake.setdefault(float(row["brake_g"]), row)

    (out_dir / "brake_limit_sweep.json").write_text(
        json.dumps({
            "rows": rows,
            "best": ranked[0],
            "best_by_brake": [
                best_by_brake[key] for key in sorted(best_by_brake)
            ],
        }, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Brake Limit Sweep",
        "",
        "Model: corrected no-regen command power cap. Mechanical brake limit "
        "is varied and the speed envelope is recomputed for each case.",
        "",
        "## Best Overall",
        "",
    ]
    best = ranked[0]
    lines.append(
        f"- `{best['brake_g']:.2f} g` brake cap, `{best['torque_nm']:.1f} Nm`, "
        f"`{best['rpm']} rpm`, `{best['power_limit_kw']:.1f} kW` = "
        f"`{best['combined_score']:.3f}` points."
    )
    lines.extend([
        "",
        "## Best By Brake Cap",
        "",
        "| Brake cap g | Torque | RPM | Power kW | Points | Time | Net kWh | Brake heat kWh | Brake time % | Max temp C |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in sorted(best_by_brake):
        row = best_by_brake[key]
        lines.append(
            f"| {row['brake_g']:.2f} | {row['torque_nm']:.1f} | {row['rpm']} | "
            f"{row['power_limit_kw']:.1f} | {row['combined_score']:.3f} | "
            f"{row['time_s']:.1f} | {row['net_kwh']:.3f} | "
            f"{row['brake_energy_kwh']:.3f} | {row['brake_time_pct']:.1f} | "
            f"{row['max_cell_temp_c']:.1f} |"
        )

    (out_dir / "brake_limit_sweep.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--torques", type=float, nargs="*", default=[62.5, 65.0, 67.5, 70.0, 72.5])
    parser.add_argument("--rpms", type=int, nargs="*", default=[3200, 3250, 3300, 3350])
    parser.add_argument("--power-kw", type=float, nargs="*", default=[12.0, 13.0, 14.0, 15.0, 16.0])
    parser.add_argument("--brake-g", type=float, nargs="*", default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    args = parser.parse_args()

    out_dir = args.out_dir or (
        REPO / "results" / f"brake_limit_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    jobs = [
        (float(torque), int(rpm), float(power_kw), float(brake_g))
        for brake_g in args.brake_g
        for torque in args.torques
        for rpm in args.rpms
        for power_kw in args.power_kw
    ]
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
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            future_to_job = {
                pool.submit(run_case, torque, rpm, power_kw, brake_g): (
                    torque, rpm, power_kw, brake_g,
                )
                for torque, rpm, power_kw, brake_g in jobs
            }
            for idx, future in enumerate(as_completed(future_to_job), start=1):
                row = future.result()
                row["completed_index"] = idx
                row["total_jobs"] = len(jobs)
                row["elapsed_s"] = time.perf_counter() - started
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                if row.get("status") == "ok":
                    rows.append(row)
                    print(
                        f"[{idx:03d}/{len(jobs)}] "
                        f"{row['brake_g']:.2f}g {row['torque_nm']:.1f}Nm "
                        f"{row['rpm']}rpm {row['power_limit_kw']:.1f}kW -> "
                        f"{row['combined_score']:.3f} pts"
                    )
                else:
                    print(f"[{idx:03d}/{len(jobs)}] ERROR {row['error']}")

    if len(rows) != len(jobs):
        print(f"completed {len(rows)} of {len(jobs)} successfully")
        return 2
    _write_outputs(rows, out_dir)
    best = sorted(rows, key=lambda r: float(r["combined_score"]), reverse=True)[0]
    print(
        f"best={best['brake_g']:.2f}g {best['torque_nm']:.1f}Nm "
        f"{best['rpm']}rpm {best['power_limit_kw']:.1f}kW "
        f"{best['combined_score']:.3f}pts"
    )
    print(f"wrote {out_dir / 'brake_limit_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
