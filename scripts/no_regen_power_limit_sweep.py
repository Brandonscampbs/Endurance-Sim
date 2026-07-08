"""No-regen sweep over torque, RPM, and LVCU power limit."""

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
from fsae_sim.driver.adaptive import AdaptiveDriverParams  # noqa: E402
from fsae_sim.driver.strategies import AdaptiveStrategy  # noqa: E402
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

ENDURANCE_LAPS = 22
VALIDATION_HOLDOUT_LAPS_ZERO_BASED = tuple(range(12, 22))
DEFAULT_TORQUES = (65.0, 70.0, 72.5, 75.0, 77.5, 80.0)
DEFAULT_RPMS = (3350, 3450, 3550)
DEFAULT_POWER_KW = (18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 40.0)

_CTX: dict[str, Any] = {}


def _init_worker() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vehicle = VehicleConfig.from_yaml(str(CONFIG))
        _, aim_df = load_cleaned_csv(str(TELEM))
        speed_col = telemetry_speed_col(aim_df)
        detected_laps = detect_lap_boundaries(aim_df)
        validation_laps = [
            lap
            for lap in VALIDATION_HOLDOUT_LAPS_ZERO_BASED
            if lap < len(detected_laps)
        ]
        track = Track.from_telemetry(df=aim_df)
        battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
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


def _lvcu_constant_for_power_kw(vehicle: VehicleConfig, power_kw: float) -> float:
    """Return LVCU power constant for nominal real shaft kW at 30 C.

    Firmware uses ``torque = constant * effective_bms_current /
    (rpm * lvcu_rpm_scale)``. Convert target real mechanical shaft power
    ``T * rpm*pi/30`` into that firmware constant at the 30 C BMS limit.
    """
    raw_bms_a = vehicle.battery.discharge_limits[0].max_current_a
    effective_a = max(1e-9, raw_bms_a - vehicle.powertrain.lvcu_bms_current_offset_a)
    return (
        float(power_kw)
        * 1000.0
        * 30.0
        * vehicle.powertrain.lvcu_rpm_scale
        / (effective_a * 3.141592653589793)
    )


def _vehicle_with_limits(
    torque_nm: float,
    rpm: int,
    power_kw: float,
) -> VehicleConfig:
    vehicle = _CTX["vehicle"]
    lvcu_power_constant = _lvcu_constant_for_power_kw(vehicle, power_kw)
    return replace(
        vehicle,
        powertrain=replace(
            vehicle.powertrain,
            torque_limit_inverter_nm=float(torque_nm),
            motor_speed_max_rpm=float(rpm),
            lvcu_overspeed_rpm=float(rpm),
            lvcu_power_constant=float(lvcu_power_constant),
        ),
    )


def _power_knee_rpm(torque_nm: float, power_kw: float) -> float:
    if torque_nm <= 0.0:
        return 0.0
    return float(power_kw) * 1000.0 * 30.0 / (float(torque_nm) * 3.141592653589793)


def run_case(torque_nm: float, rpm: int, power_kw: float) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vehicle = _vehicle_with_limits(torque_nm, rpm, power_kw)
            track = _CTX["track"]
            battery = copy.copy(_CTX["battery"])
            battery.violations = []
            params = AdaptiveDriverParams(regen_enabled=False)
            strategy = AdaptiveStrategy.from_config(vehicle, track, params=params)
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
        brake_energy_kwh = (
            float(states["mechanical_brake_energy_j"].sum()) / 3.6e6
            if "mechanical_brake_energy_j" in states else 0.0
        )
        return {
            "status": "ok",
            "torque_nm": float(torque_nm),
            "rpm": int(rpm),
            "power_limit_kw": float(power_kw),
            "power_knee_rpm": _power_knee_rpm(torque_nm, power_kw),
            "lvcu_power_constant": _lvcu_constant_for_power_kw(_CTX["vehicle"], power_kw),
            "combined_score": float(score.combined_score),
            "endurance_total": float(score.endurance_total),
            "efficiency_score": float(score.efficiency_score),
            "efficiency_factor": float(score.efficiency_factor),
            "time_s": float(result.total_time_s),
            "net_kwh": float(result.net_energy_kwh),
            "discharge_kwh": float(result.discharge_energy_kwh),
            "regen_kwh": float(result.regen_energy_kwh),
            "brake_energy_kwh": brake_energy_kwh,
            "final_soc_pct": float(result.final_soc),
            "laps_completed": int(result.laps_completed),
            "peak_speed_kmh": float(states["speed_kmh"].max()),
            "peak_motor_rpm": float(states["motor_rpm"].max()),
            "peak_motor_torque_nm": float(states["motor_torque_nm"].max()),
            "peak_pack_current_a": float(states["pack_current_a"].max()),
            "min_pack_current_a": float(states["pack_current_a"].min()),
            "max_cell_temp_c": float(states["cell_temp_c"].max()),
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
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: (float(r["torque_nm"]), int(r["rpm"]), float(r["power_limit_kw"])))
    fields = [
        "torque_nm",
        "rpm",
        "power_limit_kw",
        "power_knee_rpm",
        "lvcu_power_constant",
        "combined_score",
        "endurance_total",
        "efficiency_score",
        "efficiency_factor",
        "time_s",
        "net_kwh",
        "discharge_kwh",
        "regen_kwh",
        "brake_energy_kwh",
        "final_soc_pct",
        "laps_completed",
        "peak_speed_kmh",
        "peak_motor_rpm",
        "peak_motor_torque_nm",
        "peak_pack_current_a",
        "min_pack_current_a",
        "max_cell_temp_c",
        "speed_limited_pct",
        "speed_limit_violations",
        "throttle_segments",
        "brake_segments",
        "coast_segments",
        "duration_s",
    ]
    with (out_dir / "no_regen_power_limit_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    ranked = sorted(rows, key=lambda r: float(r["combined_score"]), reverse=True)
    (out_dir / "no_regen_power_limit_sweep.json").write_text(
        json.dumps({"rows": rows, "best": ranked[0]}, indent=2),
        encoding="utf-8",
    )
    best = ranked[0]
    lines = [
        "# No-Regen Power-Limit Sweep",
        "",
        "Model: adaptive friction-only braking. Full 22-lap endurance, "
        "2025 Michigan scoring, no penalties.",
        "",
        "## Recommendation",
        "",
        (
            f"- Best: `{best['torque_nm']:.1f} Nm`, `{best['rpm']} rpm`, "
            f"`{best['power_limit_kw']:.1f} kW` = "
            f"`{best['combined_score']:.3f}` points."
        ),
        (
            f"- Torque fade starts around `{best['power_knee_rpm']:.0f} rpm` "
            f"for that torque/power pair."
        ),
        (
            f"- Firmware-equivalent `lvcu_power_constant`: "
            f"`{best['lvcu_power_constant']:.1f}` at the 30 C/100 A BMS limit."
        ),
        "",
        "## Top 20",
        "",
        "| Rank | Torque | RPM | Power kW | Knee RPM | Points | Endurance | Efficiency | Time | Net kWh | Max Temp C | Peak A |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked[:20], start=1):
        lines.append(
            f"| {idx} | {row['torque_nm']:.1f} | {row['rpm']} | "
            f"{row['power_limit_kw']:.1f} | {row['power_knee_rpm']:.0f} | "
            f"{row['combined_score']:.3f} | {row['endurance_total']:.3f} | "
            f"{row['efficiency_score']:.3f} | {row['time_s']:.1f} | "
            f"{row['net_kwh']:.3f} | {row['max_cell_temp_c']:.1f} | "
            f"{row['peak_pack_current_a']:.1f} |"
        )
    (out_dir / "no_regen_power_limit_sweep.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--torques", type=float, nargs="*", default=list(DEFAULT_TORQUES))
    parser.add_argument("--rpms", type=int, nargs="*", default=list(DEFAULT_RPMS))
    parser.add_argument("--power-kw", type=float, nargs="*", default=list(DEFAULT_POWER_KW))
    args = parser.parse_args()

    out_dir = args.out_dir or (
        REPO / "results" / f"no_regen_power_limit_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    jobs = [
        (float(torque), int(rpm), float(power_kw))
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
                pool.submit(run_case, torque, rpm, power_kw): (torque, rpm, power_kw)
                for torque, rpm, power_kw in jobs
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
                        f"{row['torque_nm']:.1f}Nm {row['rpm']}rpm "
                        f"{row['power_limit_kw']:.1f}kW -> "
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
        f"best={best['torque_nm']:.1f}Nm {best['rpm']}rpm "
        f"{best['power_limit_kw']:.1f}kW {best['combined_score']:.3f}pts"
    )
    print(f"wrote {out_dir / 'no_regen_power_limit_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
