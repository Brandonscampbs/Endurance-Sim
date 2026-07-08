"""RPM sweep for driver/replay endurance points.

Runs the CT-16EV Michigan 2025 simulator at a list of motor RPM caps for
the calibrated driver model and full-recording replay model, then scores
each run against the 2025 FSAE Michigan endurance/efficiency field.

Outputs are written to ``results/rpm_sweep_<timestamp>/``:
- ``rpm_sweep_results.csv``
- ``rpm_sweep_results.json``
- ``rpm_sweep_report.md``
- ``rpm_sweep_report.html``
- ``progress.jsonl``
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
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
from fsae_sim.driver.strategies import CalibratedStrategy, ReplayStrategy  # noqa: E402
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
    """Build expensive model inputs once per worker process."""
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

        driver_track = Track.from_telemetry(df=aim_df)
        replay_track = Track.from_telemetry_full_recording(df=aim_df)

        driver_battery = BatteryModel.from_config_and_data(
            vehicle.battery,
            str(VOLTT),
        )
        driver_battery.calibrate_pack_from_telemetry(
            aim_df,
            holdout_laps=tuple(lap + 1 for lap in validation_laps),
        )

        replay_battery = BatteryModel.from_config_and_data(
            vehicle.battery,
            str(VOLTT),
        )
        replay_battery.calibrate_pack_from_telemetry(
            aim_df,
            allow_same_run_validation=True,
        )

        driver_strategy = CalibratedStrategy.from_telemetry(
            aim_df,
            driver_track,
            speed_col=speed_col,
            holdout_laps=validation_laps,
            use_observed_speed_caps=False,
        )
        replay_strategy = ReplayStrategy.from_full_endurance(
            aim_df,
            replay_track.total_distance_m,
            trim_to_lap_start=False,
        )

        initial_speed_ms = (
            float(aim_df[speed_col].iloc[detected_laps[0][0]]) / 3.6
            if detected_laps else 0.0
        )

    _CTX.update({
        "vehicle": vehicle,
        "speed_col": speed_col,
        "detected_laps": detected_laps,
        "validation_laps": validation_laps,
        "driver_track": driver_track,
        "replay_track": replay_track,
        "driver_battery": driver_battery,
        "replay_battery": replay_battery,
        "driver_strategy": driver_strategy,
        "replay_strategy": replay_strategy,
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


def _fresh_battery(key: str) -> BatteryModel:
    battery = copy.copy(_CTX[key])
    battery.violations = []
    return battery


def _score_result(
    *,
    model: str,
    total_time_s: float,
    total_energy_kwh: float,
) -> Any:
    scorer = FSAEScoring.michigan_2025_field()
    if model == "driver":
        track = _CTX["driver_track"]
        total_distance_km = track.total_distance_m * ENDURANCE_LAPS / 1000.0
        track_km_per_lap = track.total_distance_m / 1000.0
    else:
        track = _CTX["replay_track"]
        total_distance_km = track.total_distance_m / 1000.0
        track_km_per_lap = track.total_distance_m / ENDURANCE_LAPS / 1000.0

    return scorer.score(
        total_time_s=total_time_s,
        total_energy_kwh=total_energy_kwh,
        laps_completed=ENDURANCE_LAPS,
        total_distance_km=total_distance_km,
        track_km_per_lap=track_km_per_lap,
        driver_change_completed=True,
    )


def run_case(job: tuple[str, int]) -> dict[str, Any]:
    model, rpm = job
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vehicle = _vehicle_with_rpm(rpm)
            if model == "driver":
                track = _CTX["driver_track"]
                strategy = _CTX["driver_strategy"]
                battery = _fresh_battery("driver_battery")
                mode = SimulationMode.VALIDATION
                num_laps = ENDURANCE_LAPS
            elif model == "replay":
                track = _CTX["replay_track"]
                strategy = _CTX["replay_strategy"]
                battery = _fresh_battery("replay_battery")
                mode = SimulationMode.REPLAY
                num_laps = 1
            else:
                raise ValueError(f"unknown model {model!r}")

            result = SimulationEngine(
                vehicle,
                track,
                strategy,
                battery,
                mode=mode,
            ).run(
                num_laps=num_laps,
                initial_soc_pct=95.0,
                initial_temp_c=29.0,
                initial_speed_ms=float(_CTX["initial_speed_ms"]),
            )

        states = result.states
        score = _score_result(
            model=model,
            total_time_s=float(result.total_time_s),
            total_energy_kwh=float(result.net_energy_kwh),
        )
        peak_motor_rpm = (
            float(states["motor_rpm"].max())
            if "motor_rpm" in states and not states.empty else 0.0
        )
        peak_speed_kmh = (
            float(states["speed_kmh"].max())
            if "speed_kmh" in states and not states.empty else 0.0
        )
        rpm_limited_pct = (
            float((states["motor_rpm"] >= float(rpm) * 0.995).mean() * 100.0)
            if "motor_rpm" in states and not states.empty else 0.0
        )
        speed_limit_active_pct = (
            float(states["speed_limit_active"].mean() * 100.0)
            if "speed_limit_active" in states and not states.empty else 0.0
        )
        distance_km = (
            float(states["distance_m"].iloc[-1] / 1000.0)
            if "distance_m" in states and not states.empty else 0.0
        )

        return {
            "status": "ok",
            "model": model,
            "rpm": int(rpm),
            "sim_laps_completed": int(result.laps_completed),
            "score_laps_completed": ENDURANCE_LAPS,
            "time_s": float(result.total_time_s),
            "net_kwh": float(result.net_energy_kwh),
            "net_ah": float(result.net_charge_ah),
            "final_soc_pct": float(result.final_soc),
            "distance_km": distance_km,
            "peak_speed_kmh": peak_speed_kmh,
            "peak_motor_rpm": peak_motor_rpm,
            "rpm_limited_pct": rpm_limited_pct,
            "speed_limit_active_pct": speed_limit_active_pct,
            "endurance_time_score": float(score.endurance_time_score),
            "endurance_laps_score": float(score.endurance_laps_score),
            "endurance_total": float(score.endurance_total),
            "efficiency_factor": float(score.efficiency_factor),
            "efficiency_score": float(score.efficiency_score),
            "combined_score": float(score.combined_score),
            "duration_s": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "status": "error",
            "model": model,
            "rpm": int(rpm),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _fmt(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _svg_line_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    y_key: str,
    y_label: str,
    colors: dict[str, str],
    width: int = 980,
    height: int = 420,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 68, 26, 42, 54
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xs = sorted({int(r["rpm"]) for r in rows})
    ys = [float(r[y_key]) for r in rows]
    y_min = min(ys)
    y_max = max(ys)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    span = y_max - y_min
    y_min -= span * 0.08
    y_max += span * 0.08

    def sx(x: float) -> float:
        return pad_l + (x - xs[0]) / (xs[-1] - xs[0]) * plot_w

    def sy(y: float) -> float:
        return pad_t + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="24" class="chart-title">{html.escape(title)}</text>',
    ]

    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = sy(y)
        parts.append(
            f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width-pad_r}" '
            f'y2="{py:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{pad_l-10}" y="{py+4:.1f}" text-anchor="end" '
            f'class="tick">{y:.1f}</text>'
        )

    for x in xs:
        px = sx(x)
        parts.append(
            f'<line x1="{px:.1f}" y1="{pad_t}" x2="{px:.1f}" '
            f'y2="{height-pad_b}" class="grid minor"/>'
        )
        if x % 200 == 0:
            parts.append(
                f'<text x="{px:.1f}" y="{height-22}" text-anchor="middle" '
                f'class="tick">{x}</text>'
            )

    parts.append(
        f'<text x="{width/2:.1f}" y="{height-4}" text-anchor="middle" '
        f'class="axis">RPM cap</text>'
    )
    parts.append(
        f'<text x="16" y="{height/2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 16 {height/2:.1f})" '
        f'class="axis">{html.escape(y_label)}</text>'
    )

    for model in ("driver", "replay"):
        model_rows = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: int(r["rpm"]),
        )
        pts = " ".join(
            f'{sx(float(r["rpm"])):.1f},{sy(float(r[y_key])):.1f}'
            for r in model_rows
        )
        color = colors[model]
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for r in model_rows:
            parts.append(
                f'<circle cx="{sx(float(r["rpm"])):.1f}" '
                f'cy="{sy(float(r[y_key])):.1f}" r="4" fill="{color}">'
                f'<title>{model} {r["rpm"]}: {float(r[y_key]):.2f}</title>'
                f'</circle>'
            )

    legend_y = 28
    legend_x = width - 190
    for i, model in enumerate(("driver", "replay")):
        y = legend_y + i * 22
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" '
            f'stroke="{colors[model]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x+36}" y="{y+5}" class="legend">'
            f'{html.escape(model.title())}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: (int(r["rpm"]), str(r["model"])))
    csv_path = out_dir / "rpm_sweep_results.csv"
    json_path = out_dir / "rpm_sweep_results.json"
    md_path = out_dir / "rpm_sweep_report.md"
    html_path = out_dir / "rpm_sweep_report.html"

    fieldnames = [
        "model",
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
        "distance_km",
        "peak_speed_kmh",
        "peak_motor_rpm",
        "rpm_limited_pct",
        "speed_limit_active_pct",
        "sim_laps_completed",
        "score_laps_completed",
        "duration_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rpm_values": list(RPM_VALUES),
        "scoring": {
            "source": str(RESULTS_PDF.relative_to(REPO)),
            "endurance_tmin_s": 1369.936,
            "efficiency_co2min_kg_per_lap": 0.0967,
            "efficiency_efmax": 0.848,
            "score_laps_completed": ENDURANCE_LAPS,
            "penalties": "none",
        },
        "notes": [
            "driver = CalibratedStrategy, validation holdout laps 13-21, observed speed caps disabled",
            "replay = ReplayStrategy on the full cleaned recording; scored as 22 laps because the track distance is the full 22.1 km event even though the lap detector only tags 21 closed intervals",
            "RPM override sets both motor_speed_max_rpm and lvcu_overspeed_rpm",
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by_model = {
        model: sorted([r for r in rows if r["model"] == model], key=lambda r: r["rpm"])
        for model in ("driver", "replay")
    }
    best = {
        model: max(model_rows, key=lambda r: float(r["combined_score"]))
        for model, model_rows in by_model.items()
    }
    baseline = {
        model: next(r for r in model_rows if int(r["rpm"]) == 3000)
        for model, model_rows in by_model.items()
    }

    compare_rows = []
    for rpm in RPM_VALUES:
        d = next(r for r in by_model["driver"] if int(r["rpm"]) == rpm)
        p = next(r for r in by_model["replay"] if int(r["rpm"]) == rpm)
        compare_rows.append(
            "| "
            + " | ".join([
                str(rpm),
                _fmt(d["combined_score"], 1),
                _fmt(p["combined_score"], 1),
                _fmt(float(d["combined_score"]) - float(p["combined_score"]), 1),
                _fmt(d["time_s"], 1),
                _fmt(p["time_s"], 1),
                _fmt(d["net_kwh"], 3),
                _fmt(p["net_kwh"], 3),
            ])
            + " |"
        )

    md_lines = [
        "# RPM Sweep: Competition Points",
        "",
        "Scored against `Real-Car-Data-And-Stats/FSAE_2025_MI6_results.pdf`: "
        "endurance Tmin 1369.936 s, efficiency CO2min 0.0967 kg/lap, "
        "EFmax 0.848. No penalties were applied.",
        "",
        "Important scoring convention: replay is scored as 22 laps because "
        "the full-recording track distance is the full ~22.1 km event; the "
        "lap detector only labels 21 closed intervals because the recording "
        "does not include a final clean crossing.",
        "",
        "## Best Cases",
        "",
        "| Model | Best RPM | Combined pts | Endurance pts | Efficiency pts | Time (s) | Net kWh |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("driver", "replay"):
        row = best[model]
        md_lines.append(
            "| "
            + " | ".join([
                model,
                str(row["rpm"]),
                _fmt(row["combined_score"], 1),
                _fmt(row["endurance_total"], 1),
                _fmt(row["efficiency_score"], 1),
                _fmt(row["time_s"], 1),
                _fmt(row["net_kwh"], 3),
            ])
            + " |"
        )

    md_lines.extend([
        "",
        "## RPM vs Points",
        "",
        "| RPM | Driver pts | Replay pts | Driver-Replay | Driver time | Replay time | Driver kWh | Replay kWh |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *compare_rows,
        "",
        "## Detail: Driver",
        "",
        "| RPM | Combined | Endurance | Efficiency | Time (s) | kWh | Peak speed | Peak RPM | RPM-limited % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in by_model["driver"]:
        md_lines.append(
            "| "
            + " | ".join([
                str(row["rpm"]),
                _fmt(row["combined_score"], 1),
                _fmt(row["endurance_total"], 1),
                _fmt(row["efficiency_score"], 1),
                _fmt(row["time_s"], 1),
                _fmt(row["net_kwh"], 3),
                _fmt(row["peak_speed_kmh"], 1),
                _fmt(row["peak_motor_rpm"], 0),
                _fmt(row["rpm_limited_pct"], 1),
            ])
            + " |"
        )

    md_lines.extend([
        "",
        "## Detail: Replay",
        "",
        "| RPM | Combined | Endurance | Efficiency | Time (s) | kWh | Peak speed | Peak RPM | RPM-limited % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in by_model["replay"]:
        md_lines.append(
            "| "
            + " | ".join([
                str(row["rpm"]),
                _fmt(row["combined_score"], 1),
                _fmt(row["endurance_total"], 1),
                _fmt(row["efficiency_score"], 1),
                _fmt(row["time_s"], 1),
                _fmt(row["net_kwh"], 3),
                _fmt(row["peak_speed_kmh"], 1),
                _fmt(row["peak_motor_rpm"], 0),
                _fmt(row["rpm_limited_pct"], 1),
            ])
            + " |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    colors = {"driver": "#1d4ed8", "replay": "#b45309"}
    chart_points = _svg_line_chart(
        rows,
        title="RPM Cap vs Combined Competition Points",
        y_key="combined_score",
        y_label="Combined points",
        colors=colors,
    )
    chart_time = _svg_line_chart(
        rows,
        title="RPM Cap vs Simulated Endurance Time",
        y_key="time_s",
        y_label="Time (s)",
        colors=colors,
    )
    chart_energy = _svg_line_chart(
        rows,
        title="RPM Cap vs Net Energy",
        y_key="net_kwh",
        y_label="Net kWh",
        colors=colors,
    )

    def table_html(model_rows: list[dict[str, Any]]) -> str:
        body = []
        for row in model_rows:
            body.append(
                "<tr>"
                f"<td>{row['rpm']}</td>"
                f"<td>{float(row['combined_score']):.1f}</td>"
                f"<td>{float(row['endurance_total']):.1f}</td>"
                f"<td>{float(row['efficiency_score']):.1f}</td>"
                f"<td>{float(row['time_s']):.1f}</td>"
                f"<td>{float(row['net_kwh']):.3f}</td>"
                f"<td>{float(row['peak_speed_kmh']):.1f}</td>"
                f"<td>{float(row['peak_motor_rpm']):.0f}</td>"
                f"<td>{float(row['rpm_limited_pct']):.1f}%</td>"
                "</tr>"
            )
        return "\n".join(body)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RPM Sweep: Competition Points</title>
<style>
:root {{
  --bg: #f7f7f4;
  --ink: #1f2933;
  --muted: #5f6b76;
  --line: #d8d9d2;
  --panel: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
main {{
  max-width: 1160px;
  margin: 0 auto;
  padding: 28px 20px 52px;
}}
h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
p {{ color: var(--muted); line-height: 1.45; }}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin: 18px 0 22px;
}}
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}}
.label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.big {{ font-size: 30px; font-weight: 720; margin-top: 4px; }}
.sub {{ color: var(--muted); margin-top: 2px; font-size: 13px; }}
.chart {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0;
  overflow-x: auto;
}}
svg {{ min-width: 760px; width: 100%; height: auto; }}
.chart-title {{ font-size: 17px; font-weight: 700; fill: var(--ink); }}
.tick {{ font-size: 12px; fill: var(--muted); }}
.axis {{ font-size: 12px; fill: var(--muted); font-weight: 600; }}
.legend {{ font-size: 13px; fill: var(--ink); }}
.grid {{ stroke: #d8d9d2; stroke-width: 1; }}
.grid.minor {{ stroke: #ecece8; }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}}
th, td {{
  padding: 8px 10px;
  border-bottom: 1px solid #ecece8;
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}}
th:first-child, td:first-child {{ text-align: left; }}
th {{ font-size: 12px; color: var(--muted); background: #fbfbf9; }}
tr:last-child td {{ border-bottom: 0; }}
.table-wrap {{ overflow-x: auto; }}
code {{ background: #ecece8; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
  <h1>RPM Sweep: Competition Points</h1>
  <p>Scored against <code>{html.escape(str(RESULTS_PDF.relative_to(REPO)))}</code>:
  endurance Tmin 1369.936 s, efficiency CO2min 0.0967 kg/lap, EFmax 0.848.
  No penalties were applied. RPM override sets both <code>motor_speed_max_rpm</code>
  and <code>lvcu_overspeed_rpm</code>.</p>

  <div class="cards">
    <div class="card">
      <div class="label">Best Driver</div>
      <div class="big">{best['driver']['rpm']} RPM</div>
      <div class="sub">{float(best['driver']['combined_score']):.1f} pts, {float(best['driver']['time_s']):.1f} s, {float(best['driver']['net_kwh']):.3f} kWh</div>
    </div>
    <div class="card">
      <div class="label">Best Replay</div>
      <div class="big">{best['replay']['rpm']} RPM</div>
      <div class="sub">{float(best['replay']['combined_score']):.1f} pts, {float(best['replay']['time_s']):.1f} s, {float(best['replay']['net_kwh']):.3f} kWh</div>
    </div>
    <div class="card">
      <div class="label">3000 RPM Baseline Spread</div>
      <div class="big">{float(baseline['driver']['combined_score']) - float(baseline['replay']['combined_score']):+.1f}</div>
      <div class="sub">driver pts minus replay pts at 3000 RPM</div>
    </div>
  </div>

  <div class="chart">{chart_points}</div>
  <div class="chart">{chart_time}</div>
  <div class="chart">{chart_energy}</div>

  <h2>Driver Detail</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>RPM</th><th>Combined</th><th>Endurance</th><th>Efficiency</th><th>Time s</th><th>Net kWh</th><th>Peak km/h</th><th>Peak RPM</th><th>RPM-limited</th></tr></thead>
      <tbody>{table_html(by_model['driver'])}</tbody>
    </table>
  </div>

  <h2>Replay Detail</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>RPM</th><th>Combined</th><th>Endurance</th><th>Efficiency</th><th>Time s</th><th>Net kWh</th><th>Peak km/h</th><th>Peak RPM</th><th>RPM-limited</th></tr></thead>
      <tbody>{table_html(by_model['replay'])}</tbody>
    </table>
  </div>

  <p>Replay is scored as 22 laps because the full-recording track distance is
  the full ~22.1 km event; the lap detector only labels 21 closed intervals.</p>
</main>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rpms", type=int, nargs="*", default=list(RPM_VALUES))
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPO / "results" / f"rpm_sweep_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    jobs = [(model, int(rpm)) for rpm in args.rpms for model in ("driver", "replay")]
    rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    print(f"output_dir={out_dir}")
    print(f"jobs={len(jobs)} workers={args.workers}")
    with progress_path.open("a", encoding="utf-8") as progress:
        progress.write(json.dumps({
            "event": "start",
            "jobs": len(jobs),
            "workers": args.workers,
            "out_dir": str(out_dir),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }) + "\n")
        progress.flush()

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
        ) as pool:
            future_to_job = {pool.submit(run_case, job): job for job in jobs}
            for idx, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                row = future.result()
                row["completed_index"] = idx
                row["total_jobs"] = len(jobs)
                row["elapsed_s"] = time.perf_counter() - started_at
                progress.write(json.dumps(row) + "\n")
                progress.flush()
                if row.get("status") == "ok":
                    rows.append(row)
                    print(
                        f"[{idx:02d}/{len(jobs)}] {row['model']} "
                        f"{row['rpm']} rpm -> {row['combined_score']:.1f} pts "
                        f"({row['time_s']:.1f}s, {row['net_kwh']:.3f}kWh)"
                    )
                else:
                    print(f"[{idx:02d}/{len(jobs)}] ERROR {job}: {row.get('error')}")

    expected = len(jobs)
    if len(rows) != expected:
        print(f"completed {len(rows)} of {expected} successfully; see {progress_path}")
        return 2

    _write_outputs(rows, out_dir)
    print(f"wrote {out_dir / 'rpm_sweep_results.csv'}")
    print(f"wrote {out_dir / 'rpm_sweep_report.md'}")
    print(f"wrote {out_dir / 'rpm_sweep_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
