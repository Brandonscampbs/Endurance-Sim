"""Search strategy/tune variants and generate a drive map for the winner."""

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

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from fsae_sim.analysis.scoring import FSAEScoring  # noqa: E402
from fsae_sim.analysis.validation import (  # noqa: E402
    detect_lap_boundaries,
    telemetry_speed_col,
)
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.driver.adaptive import AdaptiveDriverParams  # noqa: E402
from fsae_sim.driver.energy_shaper import (  # noqa: E402
    EnergyShaper,
    EnergyShaperConfig,
)
from fsae_sim.driver.strategies import (  # noqa: E402
    AdaptiveStrategy,
    CoastOptimalStrategy,
)
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.track.track import Track  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402
from scripts.plot_ideal_throttle_map import averaged_centerline  # noqa: E402

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
        "aim_df": aim_df,
        "lap_boundaries": detected_laps,
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


def _strategy_for_case(
    strategy_name: str,
    vehicle: VehicleConfig,
    track: Track,
    budget_kwh: float | None,
) -> object:
    if strategy_name == "ideal_coast":
        return CoastOptimalStrategy.from_config(vehicle, track)

    if strategy_name == "adaptive_full":
        return AdaptiveStrategy.from_config(vehicle, track)

    if strategy_name == "adaptive_friction_only":
        params = AdaptiveDriverParams(regen_enabled=False)
        return AdaptiveStrategy.from_config(vehicle, track, params=params)

    if strategy_name.startswith("adaptive_lbp"):
        if budget_kwh is None:
            raise ValueError("adaptive_lbp requires budget_kwh")
        params = AdaptiveDriverParams(
            energy_shaper=EnergyShaper(EnergyShaperConfig(
                strategy="lbp",
                total_budget_kwh=float(budget_kwh),
                laps_total=ENDURANCE_LAPS,
            )),
        )
        return AdaptiveStrategy.from_config(vehicle, track, params=params)

    if strategy_name.startswith("adaptive_fcfb"):
        if budget_kwh is None:
            raise ValueError("adaptive_fcfb requires budget_kwh")
        params = AdaptiveDriverParams(
            energy_shaper=EnergyShaper(EnergyShaperConfig(
                strategy="fcfb",
                total_budget_kwh=float(budget_kwh),
                laps_total=ENDURANCE_LAPS,
            )),
        )
        return AdaptiveStrategy.from_config(vehicle, track, params=params)

    raise ValueError(f"unknown strategy {strategy_name!r}")


def _case_label(strategy_name: str, budget_kwh: float | None) -> str:
    if budget_kwh is None:
        return strategy_name
    return f"{strategy_name}_{budget_kwh:.2f}kWh"


def _run_sim(case: dict[str, Any], *, laps: int) -> tuple[object, object, Track]:
    vehicle = _vehicle_with_limits(case["torque_nm"], case["rpm"])
    track = _CTX["track"]
    battery = copy.copy(_CTX["battery"])
    battery.violations = []
    strategy = _strategy_for_case(
        case["strategy"],
        vehicle,
        track,
        case.get("budget_kwh"),
    )
    result = SimulationEngine(
        vehicle,
        track,
        strategy,
        battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    ).run(
        num_laps=laps,
        initial_soc_pct=95.0,
        initial_temp_c=29.0,
        initial_speed_ms=float(_CTX["initial_speed_ms"]),
    )
    return result, strategy, track


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _strategy, track = _run_sim(case, laps=ENDURANCE_LAPS)
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
        action_counts = states["action"].value_counts().to_dict()
        brake_energy_kwh = (
            float(states["mechanical_brake_energy_j"].sum()) / 3.6e6
            if "mechanical_brake_energy_j" in states else 0.0
        )
        return {
            "status": "ok",
            "strategy": case["strategy"],
            "label": _case_label(case["strategy"], case.get("budget_kwh")),
            "budget_kwh": case.get("budget_kwh"),
            "torque_nm": float(case["torque_nm"]),
            "rpm": int(case["rpm"]),
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
            "strategy": case["strategy"],
            "label": _case_label(case["strategy"], case.get("budget_kwh")),
            "budget_kwh": case.get("budget_kwh"),
            "torque_nm": float(case["torque_nm"]),
            "rpm": int(case["rpm"]),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _candidate_cases() -> list[dict[str, Any]]:
    top_tunes = [
        (72.5, 3450),
        (75.0, 3450),
        (75.0, 3500),
        (76.0, 3500),
        (76.0, 3550),
        (80.0, 3400),
        (85.0, 3400),
    ]
    cases: list[dict[str, Any]] = []
    for torque, rpm in top_tunes:
        cases.append({"strategy": "adaptive_full", "torque_nm": torque, "rpm": rpm})
        cases.append({"strategy": "adaptive_friction_only", "torque_nm": torque, "rpm": rpm})
    for budget in (2.6, 2.8, 3.0, 3.2, 3.4):
        for torque in (72.5, 75.0, 76.0, 80.0):
            for rpm in (3450, 3500, 3550):
                cases.append({
                    "strategy": "adaptive_lbp",
                    "budget_kwh": budget,
                    "torque_nm": torque,
                    "rpm": rpm,
                })
    for budget in (2.8, 3.0, 3.2):
        for torque, rpm in ((75.0, 3500), (76.0, 3500), (80.0, 3400)):
            cases.append({
                "strategy": "adaptive_fcfb",
                "budget_kwh": budget,
                "torque_nm": torque,
                "rpm": rpm,
            })
    for torque, rpm in ((75.0, 3500), (76.0, 3500), (85.0, 3700)):
        cases.append({"strategy": "ideal_coast", "torque_nm": torque, "rpm": rpm})
    return cases


def _write_rows(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "strategy",
        "label",
        "budget_kwh",
        "torque_nm",
        "rpm",
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
    with (out_dir / "strategy_tune_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (out_dir / "strategy_tune_search.json").write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r["combined_score"]), reverse=True)


def _write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    ranked = _rank(rows)
    best = ranked[0]
    best_full = next(r for r in ranked if r["strategy"] == "adaptive_full")
    best_friction = next(r for r in ranked if r["strategy"] == "adaptive_friction_only")
    best_ideal = next(r for r in ranked if r["strategy"] == "ideal_coast")
    best_energy = next((r for r in ranked if r["strategy"].startswith("adaptive_lbp")), None)
    lines = [
        "# Strategy/Tune Search",
        "",
        "Full 22-lap endurance predictions, 2025 FSAE Michigan scoring, no penalties.",
        "",
        "## Recommendation",
        "",
        (
            f"- Use `{best['torque_nm']:.1f} Nm` and `{best['rpm']} rpm` "
            f"with `{best['strategy']}`."
        ),
        (
            f"- Predicted score: `{best['combined_score']:.3f}` points "
            f"({best['time_s']:.1f} s, {best['net_kwh']:.3f} kWh)."
        ),
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Torque | RPM | Budget | Points | Time | Net kWh | Regen kWh | Brake Heat kWh | Max Temp C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [best_full, best_friction, best_energy, best_ideal]:
        if row is None:
            continue
        budget = "" if row.get("budget_kwh") in (None, "") else f"{row['budget_kwh']:.2f}"
        lines.append(
            f"| {row['label']} | {row['torque_nm']:.1f} | {row['rpm']} | {budget} | "
            f"{row['combined_score']:.3f} | {row['time_s']:.1f} | {row['net_kwh']:.3f} | "
            f"{row['regen_kwh']:.3f} | {row['brake_energy_kwh']:.3f} | {row['max_cell_temp_c']:.1f} |"
        )
    lines.extend([
        "",
        "## Top 20",
        "",
        "| Rank | Strategy | Torque | RPM | Budget | Points | Time | Net kWh | Regen kWh | Brake Heat kWh |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for idx, row in enumerate(ranked[:20], start=1):
        budget = "" if row.get("budget_kwh") in (None, "") else f"{row['budget_kwh']:.2f}"
        lines.append(
            f"| {idx} | {row['label']} | {row['torque_nm']:.1f} | {row['rpm']} | {budget} | "
            f"{row['combined_score']:.3f} | {row['time_s']:.1f} | {row['net_kwh']:.3f} | "
            f"{row['regen_kwh']:.3f} | {row['brake_energy_kwh']:.3f} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `strategy_tune_search.csv`",
        "- `strategy_tune_search.json`",
        "- `winning_drive_map.html`",
        "- `winning_drive_map.png`",
        "- `winning_lap_trace.csv`",
    ])
    (out_dir / "strategy_tune_search.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lap_trace(track: Track, states: pd.DataFrame, centerline: pd.DataFrame) -> pd.DataFrame:
    lap = int(states["lap"].min())
    lap_states = states[states["lap"] == lap].sort_values("segment_idx")
    max_regen = max(float(abs(lap_states["regen_force_n"].min())), 1e-6)
    rows = []
    for _, row in lap_states.iterrows():
        seg_idx = int(row["segment_idx"])
        seg = track.segments[seg_idx]
        mid = seg.distance_start_m + seg.length_m / 2.0
        x = float(np.interp(mid, centerline["distance_m"], centerline["x_m"]))
        y = float(np.interp(mid, centerline["distance_m"], centerline["y_m"]))
        throttle = float(row["throttle_pct"]) * 100.0
        brake = float(row["brake_pct"]) * 100.0
        regen = float(row["regen_force_n"])
        if throttle > 1.0:
            command_signed = throttle
            mode = "throttle"
        elif brake > 1.0 or regen < -1.0:
            regen_pct = min(100.0, abs(regen) / max_regen * 100.0)
            command_signed = -max(brake, regen_pct)
            mode = "brake_regen"
        else:
            command_signed = 0.0
            mode = "coast"
        rows.append({
            "segment_idx": seg_idx,
            "distance_m": mid,
            "x_m": x,
            "y_m": y,
            "mode": mode,
            "command_signed_pct": command_signed,
            "throttle_pct": throttle,
            "brake_pct": brake,
            "speed_mph": float(row["speed_kmh"]) * 0.621371,
            "motor_rpm": float(row["motor_rpm"]),
            "regen_force_n": regen,
            "brake_force_n": float(row["brake_force_n"]),
        })
    return pd.DataFrame(rows)


def _write_drive_map(
    out_dir: Path,
    row: dict[str, Any],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, _strategy, track = _run_sim(row, laps=1)
    aim_df = _CTX["aim_df"]
    centerline = averaged_centerline(aim_df, _CTX["lap_boundaries"])
    trace = _lap_trace(track, result.states, centerline)
    trace.to_csv(out_dir / "winning_lap_trace.csv", index=False)
    result.states.to_parquet(out_dir / "winning_lap_states.parquet")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import plotly.graph_objects as go

    x = centerline["x_m"].to_numpy(float)
    y = centerline["y_m"].to_numpy(float)
    points = np.column_stack([x, y])
    segs = np.stack([points[:-1], points[1:]], axis=1)
    edge_mid = centerline["distance_m"].to_numpy(float)[:-1] + np.diff(
        centerline["distance_m"].to_numpy(float)
    ) / 2.0
    cmd = np.interp(edge_mid, trace["distance_m"], trace["command_signed_pct"])
    cmap = LinearSegmentedColormap.from_list(
        "drive_cmd",
        [(0.0, "#d62728"), (0.5, "#3b4252"), (1.0, "#2ecc71")],
    )
    norm = Normalize(vmin=-100.0, vmax=100.0)

    fig, ax = plt.subplots(figsize=(7.5, 10), facecolor="#111318")
    ax.set_facecolor("#111318")
    ax.plot(x, y, color="#252a33", linewidth=8, solid_capstyle="round")
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=5, capstyle="round")
    lc.set_array(cmd)
    ax.add_collection(lc)
    ax.scatter([x[0]], [y[0]], s=90, c="white", edgecolors="#111318", zorder=3)
    ax.text(x[0], y[0], " Start/Finish", color="white", va="center", ha="left")
    cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Brake/Regen < 0    Command %    Throttle > 0", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    ax.set_title(
        f"Winning Drive Map - {row['torque_nm']:.1f} Nm / {row['rpm']} rpm",
        color="white",
        fontsize=14,
    )
    ax.set_xlabel("East/West position (m)", color="#d7dae0")
    ax.set_ylabel("North/South position (m)", color="#d7dae0")
    ax.tick_params(colors="#d7dae0")
    for spine in ax.spines.values():
        spine.set_color("#3c434f")
    ax.grid(True, color="#2a2f38", linewidth=0.7, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=1.0)
    fig.savefig(out_dir / "winning_drive_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Scattergl(
        x=centerline["x_m"],
        y=centerline["y_m"],
        mode="lines",
        line={"color": "rgba(210,215,225,0.25)", "width": 8},
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_html.add_trace(go.Scattergl(
        x=trace["x_m"],
        y=trace["y_m"],
        mode="markers",
        marker={
            "size": 7,
            "color": trace["command_signed_pct"],
            "colorscale": [
                [0.0, "#d62728"],
                [0.5, "#3b4252"],
                [1.0, "#2ecc71"],
            ],
            "cmin": -100,
            "cmax": 100,
            "colorbar": {"title": "Command %"},
        },
        customdata=np.column_stack([
            trace["mode"],
            trace["distance_m"],
            trace["throttle_pct"],
            trace["brake_pct"],
            trace["speed_mph"],
            trace["motor_rpm"],
            trace["regen_force_n"],
        ]),
        hovertemplate=(
            "Mode: %{customdata[0]}<br>"
            "Distance: %{customdata[1]:.1f} m<br>"
            "Throttle: %{customdata[2]:.1f}%<br>"
            "Brake: %{customdata[3]:.1f}%<br>"
            "Speed: %{customdata[4]:.1f} mph<br>"
            "Motor: %{customdata[5]:.0f} rpm<br>"
            "Regen force: %{customdata[6]:.0f} N<extra></extra>"
        ),
        showlegend=False,
    ))
    fig_html.add_trace(go.Scattergl(
        x=[float(centerline["x_m"].iloc[0])],
        y=[float(centerline["y_m"].iloc[0])],
        mode="markers+text",
        marker={"size": 13, "color": "white", "line": {"color": "#111318", "width": 2}},
        text=["Start/Finish"],
        textposition="middle right",
        textfont={"color": "white"},
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_html.update_layout(
        title=f"Winning Drive Map - {row['label']} - {row['torque_nm']:.1f} Nm / {row['rpm']} rpm",
        template="plotly_dark",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        xaxis_title="East/West position (m)",
        yaxis_title="North/South position (m)",
        width=820,
        height=1000,
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
    )
    fig_html.update_xaxes(scaleanchor="y", scaleratio=1, gridcolor="#2a2f38", zeroline=False)
    fig_html.update_yaxes(gridcolor="#2a2f38", zeroline=False)
    fig_html.write_html(str(out_dir / "winning_drive_map.html"), include_plotlyjs="cdn")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (
        REPO / "results" / f"strategy_tune_search_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    cases = _candidate_cases()
    print(f"output_dir={out_dir}")
    print(f"jobs={len(cases)} workers={args.workers}")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    with progress_path.open("a", encoding="utf-8") as progress:
        progress.write(json.dumps({
            "event": "start",
            "jobs": len(cases),
            "workers": args.workers,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }) + "\n")
        progress.flush()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
            future_to_case = {pool.submit(run_case, case): case for case in cases}
            for idx, future in enumerate(as_completed(future_to_case), start=1):
                row = future.result()
                row["completed_index"] = idx
                row["total_jobs"] = len(cases)
                row["elapsed_s"] = time.perf_counter() - started
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                if row.get("status") == "ok":
                    rows.append(row)
                    print(
                        f"[{idx:03d}/{len(cases)}] {row['label']} "
                        f"{row['torque_nm']:.1f}Nm {row['rpm']}rpm -> "
                        f"{row['combined_score']:.3f} pts"
                    )
                else:
                    print(f"[{idx:03d}/{len(cases)}] ERROR {row['label']}: {row['error']}")

    if len(rows) != len(cases):
        print(f"completed {len(rows)} of {len(cases)} successfully")
        return 2

    rows = _rank(rows)
    _write_rows(rows, out_dir)
    _write_summary(rows, out_dir)

    # Generate map in the main process so it can reuse plotting libraries
    # without every worker importing them.
    _CTX.clear()
    _init_worker()
    _write_drive_map(out_dir, rows[0])

    print(
        f"best={rows[0]['label']} "
        f"{rows[0]['torque_nm']:.1f}Nm {rows[0]['rpm']}rpm "
        f"{rows[0]['combined_score']:.3f}pts"
    )
    print(f"wrote {out_dir / 'strategy_tune_search.md'}")
    print(f"wrote {out_dir / 'winning_drive_map.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
