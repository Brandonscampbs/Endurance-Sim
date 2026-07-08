"""2D torque/RPM sweep using the adaptive prediction driver."""

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
RESULTS_PDF = REPO / "Real-Car-Data-And-Stats" / "FSAE_2025_MI6_results.pdf"

DEFAULT_TORQUES = tuple(range(40, 86, 5))
DEFAULT_RPMS = tuple(range(2500, 4001, 100))
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


def _vehicle_with_limits(torque_nm: float, rpm: int) -> VehicleConfig:
    vehicle = _CTX["vehicle"]
    return replace(
        vehicle,
        powertrain=replace(
            vehicle.powertrain,
            torque_limit_inverter_nm=float(torque_nm),
            motor_speed_max_rpm=float(rpm),
            lvcu_overspeed_rpm=float(rpm),
        ),
    )


def run_case(torque_nm: float, rpm: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vehicle = _vehicle_with_limits(torque_nm, rpm)
            track = _CTX["track"]
            battery = copy.copy(_CTX["battery"])
            battery.violations = []
            strategy = AdaptiveStrategy.from_config(vehicle, track)
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
            laps_completed=int(result.laps_completed),
            total_distance_km=track_km_per_lap * int(result.laps_completed),
            track_km_per_lap=track_km_per_lap,
            driver_change_completed=result.laps_completed >= ENDURANCE_LAPS,
        )
        states = result.states
        action_counts = (
            states["action"].value_counts().to_dict()
            if "action" in states else {}
        )
        return {
            "status": "ok",
            "torque_nm": float(torque_nm),
            "rpm": int(rpm),
            "combined_score": float(score.combined_score),
            "endurance_total": float(score.endurance_total),
            "endurance_time_score": float(score.endurance_time_score),
            "endurance_laps_score": float(score.endurance_laps_score),
            "efficiency_score": float(score.efficiency_score),
            "efficiency_factor": float(score.efficiency_factor),
            "time_s": float(result.total_time_s),
            "net_kwh": float(result.net_energy_kwh),
            "discharge_kwh": float(result.discharge_energy_kwh),
            "regen_kwh": float(result.regen_energy_kwh),
            "net_ah": float(result.net_charge_ah),
            "final_soc_pct": float(result.final_soc),
            "laps_completed": int(result.laps_completed),
            "peak_speed_kmh": float(states["speed_kmh"].max()),
            "peak_motor_rpm": float(states["motor_rpm"].max()),
            "peak_motor_torque_nm": float(states["motor_torque_nm"].max()),
            "peak_pack_current_a": float(states["pack_current_a"].max()),
            "min_pack_current_a": float(states["pack_current_a"].min()),
            "max_cell_temp_c": float(states["cell_temp_c"].max()),
            "rpm_limited_pct": float(
                (states["motor_rpm"] >= float(rpm) * 0.995).mean() * 100.0
            ),
            "speed_limited_pct": float(
                states["speed_limit_active"].mean() * 100.0
                if "speed_limit_active" in states else 0.0
            ),
            "speed_limit_violations": int(
                states["speed_limit_violation"].sum()
                if "speed_limit_violation" in states else 0
            ),
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
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _write_csv(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "torque_nm",
        "rpm",
        "combined_score",
        "endurance_total",
        "endurance_time_score",
        "endurance_laps_score",
        "efficiency_score",
        "efficiency_factor",
        "time_s",
        "net_kwh",
        "discharge_kwh",
        "regen_kwh",
        "net_ah",
        "final_soc_pct",
        "laps_completed",
        "peak_speed_kmh",
        "peak_motor_rpm",
        "peak_motor_torque_nm",
        "peak_pack_current_a",
        "min_pack_current_a",
        "max_cell_temp_c",
        "rpm_limited_pct",
        "speed_limited_pct",
        "speed_limit_violations",
        "throttle_segments",
        "brake_segments",
        "coast_segments",
        "duration_s",
    ]
    with (out_dir / "adaptive_torque_rpm_sweep.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    torques: list[float],
    rpms: list[int],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "adaptive",
        "torques_nm": torques,
        "rpms": rpms,
        "scoring": {
            "source": str(RESULTS_PDF.relative_to(REPO)),
            "endurance_tmin_s": 1369.936,
            "efficiency_co2min_kg_per_lap": 0.0967,
            "efficiency_efmax": 0.848,
            "laps_requested": ENDURANCE_LAPS,
            "penalties": "none",
        },
        "rows": rows,
    }
    (out_dir / "adaptive_torque_rpm_sweep.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["combined_score"]),
            -float(row["max_cell_temp_c"]),
            -float(row["peak_pack_current_a"]),
        ),
        reverse=True,
    )


def _write_markdown(rows: list[dict[str, Any]], out_dir: Path) -> None:
    best_rows = _ranked(rows)
    best = best_rows[0]
    best_score = float(best["combined_score"])
    near_best = [
        row for row in best_rows
        if best_score - float(row["combined_score"]) <= 1.0
    ]
    conservative = sorted(
        near_best,
        key=lambda row: (
            float(row["torque_nm"]),
            int(row["rpm"]),
            float(row["peak_pack_current_a"]),
        ),
    )[0]

    lines = [
        "# Adaptive Torque/RPM Sweep",
        "",
        "Model: `adaptive`. Full 22-lap endurance prediction, scored against "
        "2025 FSAE Michigan endurance and efficiency field, no penalties.",
        "",
        "## Recommendation",
        "",
        (
            f"- Best score: `{best['combined_score']:.2f}` points at "
            f"`{best['torque_nm']:.0f} Nm`, `{best['rpm']} rpm`."
        ),
        (
            f"- Conservative 1-point setting: `{conservative['torque_nm']:.0f} Nm`, "
            f"`{conservative['rpm']} rpm` with "
            f"`{conservative['combined_score']:.2f}` points."
        ),
        "",
        "## Top 20",
        "",
        "| Rank | Torque | RPM | Points | Endurance | Efficiency | Time (s) | Net kWh | Peak mph | Max Temp C | Peak A |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(best_rows[:20], start=1):
        lines.append(
            "| "
            + " | ".join([
                str(idx),
                f"{row['torque_nm']:.0f}",
                str(row["rpm"]),
                f"{row['combined_score']:.2f}",
                f"{row['endurance_total']:.2f}",
                f"{row['efficiency_score']:.2f}",
                f"{row['time_s']:.1f}",
                f"{row['net_kwh']:.3f}",
                f"{float(row['peak_speed_kmh']) * 0.621371:.1f}",
                f"{row['max_cell_temp_c']:.1f}",
                f"{row['peak_pack_current_a']:.1f}",
            ])
            + " |"
        )
    lines.extend([
        "",
        "Files:",
        "- `adaptive_torque_rpm_sweep.csv`",
        "- `adaptive_torque_rpm_sweep.json`",
        "- `adaptive_torque_rpm_heatmap.html`",
        "- `adaptive_torque_rpm_heatmap.png`",
    ])
    (out_dir / "adaptive_torque_rpm_sweep.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="torque_nm", columns="rpm", values="combined_score")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6.8), facecolor="#111318")
    ax.set_facecolor("#111318")
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            min(pivot.columns),
            max(pivot.columns),
            min(pivot.index),
            max(pivot.index),
        ],
    )
    ax.set_title("Adaptive Driver Points vs Torque/RPM", color="white", fontsize=16)
    ax.set_xlabel("RPM cap", color="#d7dae0")
    ax.set_ylabel("Torque cap (Nm)", color="#d7dae0")
    ax.tick_params(colors="#d7dae0")
    for spine in ax.spines.values():
        spine.set_color("#3c434f")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Predicted points", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    best = _ranked(rows)[0]
    ax.scatter(
        [best["rpm"]],
        [best["torque_nm"]],
        s=80,
        c="white",
        edgecolors="#111318",
        linewidths=1.5,
    )
    ax.text(
        best["rpm"],
        best["torque_nm"],
        " best",
        color="white",
        va="center",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive_torque_rpm_heatmap.png", dpi=220)
    plt.close(fig)

    custom = []
    for torque in pivot.index:
        row_custom = []
        for rpm in pivot.columns:
            match = df[
                (df["torque_nm"] == torque)
                & (df["rpm"] == rpm)
            ].iloc[0]
            row_custom.append([
                match["endurance_total"],
                match["efficiency_score"],
                match["time_s"],
                match["net_kwh"],
                match["peak_speed_kmh"] * 0.621371,
                match["max_cell_temp_c"],
                match["peak_pack_current_a"],
            ])
        custom.append(row_custom)

    html_fig = go.Figure(data=go.Heatmap(
        x=list(pivot.columns),
        y=list(pivot.index),
        z=pivot.values,
        colorscale="Viridis",
        colorbar={"title": "Points"},
        customdata=custom,
        hovertemplate=(
            "RPM: %{x}<br>"
            "Torque: %{y:.0f} Nm<br>"
            "Points: %{z:.2f}<br>"
            "Endurance: %{customdata[0]:.2f}<br>"
            "Efficiency: %{customdata[1]:.2f}<br>"
            "Time: %{customdata[2]:.1f} s<br>"
            "Energy: %{customdata[3]:.3f} kWh<br>"
            "Peak speed: %{customdata[4]:.1f} mph<br>"
            "Max temp: %{customdata[5]:.1f} C<br>"
            "Peak current: %{customdata[6]:.1f} A<extra></extra>"
        ),
    ))
    html_fig.add_trace(go.Scatter(
        x=[best["rpm"]],
        y=[best["torque_nm"]],
        mode="markers+text",
        marker={"size": 12, "color": "white", "line": {"color": "#111318", "width": 2}},
        text=["best"],
        textposition="middle right",
        showlegend=False,
        hoverinfo="skip",
    ))
    html_fig.update_layout(
        title="Adaptive Driver Points vs Torque/RPM",
        template="plotly_dark",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        xaxis_title="RPM cap",
        yaxis_title="Torque cap (Nm)",
        width=1050,
        height=720,
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
    )
    html_fig.write_html(
        str(out_dir / "adaptive_torque_rpm_heatmap.html"),
        include_plotlyjs="cdn",
        full_html=True,
    )


def _write_outputs(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    torques: list[float],
    rpms: list[int],
) -> None:
    rows = sorted(rows, key=lambda row: (float(row["torque_nm"]), int(row["rpm"])))
    _write_csv(rows, out_dir)
    _write_json(rows, out_dir, torques=torques, rpms=rpms)
    _write_markdown(rows, out_dir)
    _write_plots(rows, out_dir)


def _parse_torques(values: list[str] | None) -> list[float]:
    if not values:
        return [float(x) for x in DEFAULT_TORQUES]
    return [float(x) for x in values]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--torques", nargs="*", default=None)
    parser.add_argument("--rpms", type=int, nargs="*", default=list(DEFAULT_RPMS))
    args = parser.parse_args()

    out_dir = args.out_dir or (
        REPO
        / "results"
        / f"adaptive_torque_rpm_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    torques = _parse_torques(args.torques)
    rpms = [int(rpm) for rpm in args.rpms]
    jobs = [(torque, rpm) for torque in torques for rpm in rpms]

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"output_dir={out_dir}")
    print(f"jobs={len(jobs)} workers={args.workers}")
    print(f"torques={torques}")
    print(f"rpms={rpms}")

    with progress_path.open("a", encoding="utf-8") as progress:
        progress.write(json.dumps({
            "event": "start",
            "jobs": len(jobs),
            "workers": args.workers,
            "torques": torques,
            "rpms": rpms,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }) + "\n")
        progress.flush()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
        ) as pool:
            future_to_job = {
                pool.submit(run_case, torque, rpm): (torque, rpm)
                for torque, rpm in jobs
            }
            for idx, future in enumerate(as_completed(future_to_job), start=1):
                row = future.result()
                row["completed_index"] = idx
                row["total_jobs"] = len(jobs)
                row["elapsed_s"] = time.perf_counter() - started
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                torque, rpm = future_to_job[future]
                if row.get("status") == "ok":
                    rows.append(row)
                    print(
                        f"[{idx:03d}/{len(jobs)}] "
                        f"{torque:.0f} Nm {rpm} rpm -> "
                        f"{row['combined_score']:.2f} pts "
                        f"({row['time_s']:.1f}s, {row['net_kwh']:.3f}kWh)"
                    )
                else:
                    print(
                        f"[{idx:03d}/{len(jobs)}] ERROR "
                        f"{torque:.0f} Nm {rpm} rpm: {row['error']}"
                    )

    if len(rows) != len(jobs):
        print(f"completed {len(rows)} of {len(jobs)} successfully")
        print(f"see {progress_path}")
        return 2
    _write_outputs(rows, out_dir, torques=torques, rpms=rpms)
    best = _ranked(rows)[0]
    print(
        "best="
        f"{best['torque_nm']:.0f}Nm "
        f"{best['rpm']}rpm "
        f"{best['combined_score']:.2f}pts"
    )
    print(f"wrote {out_dir / 'adaptive_torque_rpm_sweep.csv'}")
    print(f"wrote {out_dir / 'adaptive_torque_rpm_sweep.md'}")
    print(f"wrote {out_dir / 'adaptive_torque_rpm_heatmap.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
