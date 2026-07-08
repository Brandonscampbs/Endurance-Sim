"""Compare brake timing policies and map where the best one brakes.

This is an analysis-only script. It keeps the corrected no-regen command
power cap, then compares:

* the current adaptive speed-envelope strategy ("early_brake_envelope"),
* a late-brake strategy that accelerates until the latest feasible brake
  point, and
* coast-before-brake variants with different lift distances.

The delayed strategies use a corner-only speed cap so the normal
forward/backward speed envelope does not force early braking.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import scripts.command_power_limit_sweep as command_sweep  # noqa: E402
from fsae_sim.analysis.scoring import FSAEScoring  # noqa: E402
from fsae_sim.analysis.validation import detect_lap_boundaries  # noqa: E402
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.driver.adaptive import AdaptiveDriverParams  # noqa: E402
from fsae_sim.driver.strategies import AdaptiveStrategy  # noqa: E402
from fsae_sim.driver.strategy import (  # noqa: E402
    ControlAction,
    ControlCommand,
    DriverStrategy,
    SimState,
)
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.sim.speed_envelope import SpeedEnvelope  # noqa: E402
from fsae_sim.track.track import Segment, Track  # noqa: E402
from scripts.plot_ideal_throttle_map import averaged_centerline  # noqa: E402


ENDURANCE_LAPS = 22


class FixedEnvelope:
    """Speed-envelope object that returns a precomputed cap array."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self.last_built_bms_limit_a: float | None = None

    def compute(
        self,
        initial_speed: float = 0.5,
        bms_current_limit_a: float | None = None,
    ) -> np.ndarray:
        self.last_built_bms_limit_a = bms_current_limit_a
        return self._values.copy()


@dataclass(frozen=True)
class ApproachPolicy:
    name: str
    coast_window_m: float
    brake_margin_m: float
    lookahead_m: float = 260.0
    speed_deadband_ms: float = 0.15
    brake_distance_scale: float = 1.0
    target_speed_margin_ms: float = 0.0


class LatestBrakeStrategy(DriverStrategy):
    """Full-accel, optional-coast, then brake for the next corner cap."""

    name = "latest_brake"

    def __init__(
        self,
        track: Track,
        dynamics: Any,
        corner_limits_ms: np.ndarray,
        policy: ApproachPolicy,
    ) -> None:
        self._track = track
        self._dyn = dynamics
        self._corner_limits = np.asarray(corner_limits_ms, dtype=np.float64)
        self._policy = policy
        self._distance_starts = np.array(
            [seg.distance_start_m for seg in track.segments],
            dtype=np.float64,
        )
        self._lap_distance = float(track.total_distance_m)
        mean_len = self._lap_distance / max(1, track.num_segments)
        self._lookahead_segments = max(1, int(math.ceil(policy.lookahead_m / mean_len)))

    @property
    def uses_observed_speed_caps(self) -> bool:
        return False

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        idx = int(state.segment_idx) % self._track.num_segments
        v = max(float(state.speed), 0.0)

        current_cap = float(self._corner_limits[idx])
        current_target = max(
            0.5,
            current_cap - self._policy.target_speed_margin_ms,
        )
        if math.isfinite(current_cap) and v > current_target + self._policy.speed_deadband_ms:
            return self._brake()

        critical = self._critical_future_cap(idx, v)
        if critical is None:
            return self._throttle()

        slack_m = critical["distance_m"] - critical["brake_distance_m"]
        if slack_m <= self._policy.brake_margin_m:
            return self._brake()
        if (
            self._policy.coast_window_m > 0.0
            and slack_m <= self._policy.brake_margin_m + self._policy.coast_window_m
        ):
            return self._coast()
        return self._throttle()

    def _critical_future_cap(self, idx: int, speed_ms: float) -> dict[str, float] | None:
        best: dict[str, float] | None = None
        n = self._track.num_segments
        for offset in range(1, min(self._lookahead_segments, n) + 1):
            j = (idx + offset) % n
            target = float(self._corner_limits[j])
            if not math.isfinite(target):
                continue
            target = max(0.5, target - self._policy.target_speed_margin_ms)
            if target >= speed_ms - self._policy.speed_deadband_ms:
                continue
            distance_m = self._distance_to_segment_start(idx, j)
            if distance_m <= 0.0:
                continue
            brake_distance_m = self._brake_distance(
                speed_ms,
                target,
                self._track.segments[idx],
            ) * self._policy.brake_distance_scale
            slack_m = distance_m - brake_distance_m
            candidate = {
                "segment_idx": float(j),
                "target_ms": target,
                "distance_m": distance_m,
                "brake_distance_m": brake_distance_m,
                "slack_m": slack_m,
            }
            if best is None or slack_m < best["slack_m"]:
                best = candidate
        return best

    def _distance_to_segment_start(self, idx: int, target_idx: int) -> float:
        here = float(self._distance_starts[idx])
        there = float(self._distance_starts[target_idx])
        if there >= here:
            return there - here
        return self._lap_distance - here + there

    def _brake_distance(self, v0: float, v1: float, seg: Segment) -> float:
        if v0 <= v1:
            return 0.0
        v_avg = max(0.5, 0.5 * (v0 + v1))
        active_brake = float(self._dyn.mechanical_brake_force(1.0, v_avg))
        resist = float(
            self._dyn.total_resistance(
                v_avg,
                grade=float(seg.grade),
                curvature=float(seg.curvature),
            )
        )
        accel = max((active_brake + resist) / float(self._dyn.m_effective), 1e-6)
        return max(0.0, (v0 * v0 - v1 * v1) / (2.0 * accel))

    def _throttle(self) -> ControlCommand:
        return ControlCommand(
            action=ControlAction.THROTTLE,
            throttle_pct=1.0,
            brake_pct=0.0,
            regen_request_pct=0.0,
        )

    def _coast(self) -> ControlCommand:
        return ControlCommand(
            action=ControlAction.COAST,
            throttle_pct=0.0,
            brake_pct=0.0,
            regen_request_pct=0.0,
        )

    def _brake(self) -> ControlCommand:
        return ControlCommand(
            action=ControlAction.BRAKE,
            throttle_pct=0.0,
            brake_pct=1.0,
            regen_request_pct=0.0,
        )


def _init_worker() -> None:
    command_sweep._init_worker()
    _, aim_df = load_cleaned_csv(str(command_sweep.TELEM))
    command_sweep._CTX["aim_df"] = aim_df


def _corner_only_limits(track: Track, dynamics: Any) -> np.ndarray:
    limits: list[float] = []
    for seg in track.segments:
        if abs(float(seg.curvature)) < 1e-8:
            limits.append(float("inf"))
        else:
            limits.append(
                float(dynamics.max_cornering_speed(seg.curvature, seg.grip_factor))
            )
    return np.asarray(limits, dtype=np.float64)


def _score_result(result: Any, track: Track) -> Any:
    track_km_per_lap = track.total_distance_m / 1000.0
    return FSAEScoring.michigan_2025_field().score(
        total_time_s=float(result.total_time_s),
        total_energy_kwh=float(result.net_energy_kwh),
        laps_completed=int(result.laps_completed),
        total_distance_km=track_km_per_lap * int(result.laps_completed),
        track_km_per_lap=track_km_per_lap,
        driver_change_completed=result.laps_completed >= ENDURANCE_LAPS,
    )


def _make_engine(
    *,
    torque_nm: float,
    rpm: int,
    power_kw: float,
    brake_g: float,
    strategy_kind: str,
    policy: ApproachPolicy | None,
) -> tuple[SimulationEngine, Track]:
    vehicle = command_sweep._vehicle_with_limits(torque_nm, rpm)
    track = command_sweep._CTX["track"]
    battery = copy.copy(command_sweep._CTX["battery"])
    battery.violations = []

    placeholder = AdaptiveStrategy.from_config(
        vehicle,
        track,
        params=AdaptiveDriverParams(regen_enabled=False),
    )
    engine = SimulationEngine(
        vehicle,
        track,
        placeholder,
        battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    )
    engine.powertrain = command_sweep.CommandPowerLimitedPowertrain(
        engine.powertrain,
        power_kw,
    )
    engine.dynamics.max_brake_decel_g = float(brake_g)
    engine.dynamics._MAX_BRAKE_DECEL_G = float(brake_g)

    if strategy_kind == "early_brake_envelope":
        strategy = AdaptiveStrategy.from_config(
            vehicle,
            track,
            params=AdaptiveDriverParams(regen_enabled=False),
        )
        strategy.bind_models(engine.dynamics, engine.powertrain)
        engine.strategy = strategy
        engine._envelope = SpeedEnvelope(engine.dynamics, engine.powertrain, track)
    elif strategy_kind == "latest_brake":
        if policy is None:
            raise ValueError("latest_brake requires an ApproachPolicy")
        corner_limits = _corner_only_limits(track, engine.dynamics)
        strategy = LatestBrakeStrategy(track, engine.dynamics, corner_limits, policy)
        engine.strategy = strategy
        engine._envelope = FixedEnvelope(corner_limits)
    else:
        raise ValueError(f"unknown strategy_kind={strategy_kind!r}")

    return engine, track


def run_case(
    name: str,
    strategy_kind: str,
    coast_window_m: float,
    brake_margin_m: float,
    lookahead_m: float,
    brake_distance_scale: float,
    target_speed_margin_ms: float,
    torque_nm: float,
    rpm: int,
    power_kw: float,
    brake_g: float,
    keep_states: bool = False,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            policy = None
            if strategy_kind == "latest_brake":
                policy = ApproachPolicy(
                    name=name,
                    coast_window_m=float(coast_window_m),
                    brake_margin_m=float(brake_margin_m),
                    lookahead_m=float(lookahead_m),
                    brake_distance_scale=float(brake_distance_scale),
                    target_speed_margin_ms=float(target_speed_margin_ms),
                )
            engine, track = _make_engine(
                torque_nm=torque_nm,
                rpm=rpm,
                power_kw=power_kw,
                brake_g=brake_g,
                strategy_kind=strategy_kind,
                policy=policy,
            )
            result = engine.run(
                num_laps=ENDURANCE_LAPS,
                initial_soc_pct=95.0,
                initial_temp_c=29.0,
                initial_speed_ms=float(command_sweep._CTX["initial_speed_ms"]),
            )

        states = result.states
        score = _score_result(result, track)
        action_counts = states["action"].value_counts().to_dict()
        braking = states[(states["brake_pct"] > 1e-6) | (states["brake_force_n"] > 1e-6)]
        coasting = states[
            (states["throttle_pct"] <= 1e-6)
            & (states["brake_pct"] <= 1e-6)
            & (states["brake_force_n"] <= 1e-6)
        ]
        row: dict[str, Any] = {
            "status": "ok",
            "name": name,
            "strategy_kind": strategy_kind,
            "coast_window_m": float(coast_window_m),
            "brake_margin_m": float(brake_margin_m),
            "lookahead_m": float(lookahead_m),
            "brake_distance_scale": float(brake_distance_scale),
            "target_speed_margin_ms": float(target_speed_margin_ms),
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
            "brake_energy_kwh": float(states["mechanical_brake_energy_j"].sum()) / 3.6e6,
            "brake_time_s": float(braking["segment_time_s"].sum()) if len(braking) else 0.0,
            "brake_time_pct": (
                float(braking["segment_time_s"].sum()) / float(result.total_time_s) * 100.0
                if len(braking) else 0.0
            ),
            "coast_time_s": float(coasting["segment_time_s"].sum()) if len(coasting) else 0.0,
            "coast_time_pct": (
                float(coasting["segment_time_s"].sum()) / float(result.total_time_s) * 100.0
                if len(coasting) else 0.0
            ),
            "mean_brake_pct_when_braking": (
                float(braking["brake_pct"].mean()) if len(braking) else 0.0
            ),
            "max_brake_pct": float(states["brake_pct"].max()),
            "final_soc_pct": float(result.final_soc),
            "peak_speed_kmh": float(states["speed_kmh"].max()),
            "peak_motor_rpm": float(states["motor_rpm"].max()),
            "peak_motor_torque_nm": float(states["motor_torque_nm"].max()),
            "peak_pack_current_a": float(states["pack_current_a"].max()),
            "max_cell_temp_c": float(states["cell_temp_c"].max()),
            "speed_limited_pct": float(states["speed_limit_active"].mean() * 100.0),
            "speed_limit_violations": int(states["speed_limit_violation"].sum()),
            "throttle_segments": int(action_counts.get("throttle", 0)),
            "brake_segments": int(action_counts.get("brake", 0)),
            "coast_segments": int(action_counts.get("coast", 0)),
            "duration_s": time.perf_counter() - start,
        }
        if keep_states:
            row["_states"] = states
        return row
    except Exception as exc:
        return {
            "status": "error",
            "name": name,
            "strategy_kind": strategy_kind,
            "coast_window_m": float(coast_window_m),
            "brake_margin_m": float(brake_margin_m),
            "lookahead_m": float(lookahead_m),
            "brake_distance_scale": float(brake_distance_scale),
            "target_speed_margin_ms": float(target_speed_margin_ms),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_s": time.perf_counter() - start,
        }


def _representative_lap(states: pd.DataFrame) -> pd.DataFrame:
    if states.empty:
        return states.copy()
    lap = 1 if 1 in set(states["lap"].astype(int)) else int(states["lap"].min())
    return states[states["lap"].astype(int) == lap].copy().sort_values("segment_idx")


def _trace_with_xy(track: Track, states: pd.DataFrame, centerline: pd.DataFrame) -> pd.DataFrame:
    lap_states = _representative_lap(states)
    rows: list[dict[str, Any]] = []
    for _, row in lap_states.iterrows():
        seg_idx = int(row["segment_idx"])
        if seg_idx < 0 or seg_idx >= track.num_segments:
            continue
        seg = track.segments[seg_idx]
        mid_d = seg.distance_start_m + seg.length_m / 2.0
        rows.append({
            "lap": int(row["lap"]),
            "segment_idx": seg_idx,
            "distance_m": mid_d,
            "time_s": float(row["time_s"]),
            "x_m": float(np.interp(mid_d, centerline["distance_m"], centerline["x_m"])),
            "y_m": float(np.interp(mid_d, centerline["distance_m"], centerline["y_m"])),
            "speed_mph": float(row["speed_kmh"]) * 0.621371,
            "throttle_pct": float(row["throttle_pct"]) * 100.0,
            "brake_pct": max(float(row["brake_pct"]) * 100.0, 100.0 if float(row["brake_force_n"]) > 1e-6 else 0.0),
            "brake_force_n": float(row["brake_force_n"]),
            "action": str(row["action"]),
            "curvature": float(row["curvature"]),
            "lateral_g": float(row["lateral_g"]),
            "longitudinal_g": float(row["longitudinal_g"]),
            "segment_time_s": float(row["segment_time_s"]),
        })
    return pd.DataFrame(rows)


def _extract_zones(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    is_brake = (trace["brake_pct"] > 1e-6) | (trace["brake_force_n"] > 1e-6)
    rows: list[dict[str, Any]] = []
    start_idx: int | None = None
    zone_id = 1
    flags = is_brake.to_numpy(dtype=bool)
    for i, active in enumerate(flags.tolist() + [False]):
        if active and start_idx is None:
            start_idx = i
        if (not active) and start_idx is not None:
            end_idx = i - 1
            z = trace.iloc[start_idx:end_idx + 1]
            prev = trace.iloc[:start_idx]
            coast_before = 0.0
            if not prev.empty:
                j = len(prev) - 1
                while j >= 0 and str(prev.iloc[j]["action"]) == "coast":
                    coast_before += float(prev.iloc[j]["segment_time_s"])
                    j -= 1
            rows.append({
                "zone": zone_id,
                "start_distance_m": float(z["distance_m"].iloc[0]),
                "end_distance_m": float(z["distance_m"].iloc[-1]),
                "start_time_s": float(z["time_s"].iloc[0]),
                "end_time_s": float(z["time_s"].iloc[-1]),
                "duration_s": float(z["segment_time_s"].sum()),
                "entry_speed_mph": float(z["speed_mph"].iloc[0]),
                "exit_speed_mph": float(z["speed_mph"].iloc[-1]),
                "avg_brake_pct": float(z["brake_pct"].mean()),
                "max_brake_pct": float(z["brake_pct"].max()),
                "min_longitudinal_g": float(z["longitudinal_g"].min()),
                "coast_before_s": coast_before,
            })
            zone_id += 1
            start_idx = None
    return pd.DataFrame(rows)


def _write_action_map(
    centerline: pd.DataFrame,
    trace: pd.DataFrame,
    zones: pd.DataFrame,
    out_dir: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    action_value = {"throttle": 0.0, "coast": 1.0, "brake": 2.0}
    colors = np.array(["#2ca25f", "#f2c94c", "#d7191c"])

    x = centerline["x_m"].to_numpy(dtype=float)
    y = centerline["y_m"].to_numpy(dtype=float)
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    edge_mid = (
        centerline["distance_m"].to_numpy(dtype=float)[:-1]
        + np.diff(centerline["distance_m"].to_numpy(dtype=float)) / 2.0
    )
    values = np.interp(
        edge_mid,
        trace["distance_m"].to_numpy(dtype=float),
        np.array([action_value.get(str(a), 1.0) for a in trace["action"]]),
    )

    fig, ax = plt.subplots(figsize=(8.5, 10.5), facecolor="#111318")
    ax.set_facecolor("#111318")
    ax.plot(x, y, color="#2b3038", linewidth=8, solid_capstyle="round", zorder=1)

    # Draw action colors in three passes so categorical colors stay crisp.
    for value, color, label in [
        (0.0, colors[0], "Throttle"),
        (1.0, colors[1], "Coast"),
        (2.0, colors[2], "Brake"),
    ]:
        mask = np.abs(values - value) < 0.5
        if not np.any(mask):
            continue
        lc = LineCollection(
            segments[mask],
            colors=[color],
            linewidth=5.0,
            capstyle="round",
            joinstyle="round",
            label=label,
            zorder=2 + int(value),
        )
        ax.add_collection(lc)

    ax.scatter([x[0]], [y[0]], s=90, c="#ffffff", edgecolors="#111318", zorder=5)
    ax.text(x[0], y[0], " Start/Finish", color="#ffffff", fontsize=10, va="center")

    for _, zone in zones.iterrows():
        sx = float(np.interp(zone["start_distance_m"], centerline["distance_m"], centerline["x_m"]))
        sy = float(np.interp(zone["start_distance_m"], centerline["distance_m"], centerline["y_m"]))
        ax.scatter([sx], [sy], s=58, c="#ffffff", edgecolors="#d7191c", linewidths=2, zorder=6)
        ax.text(
            sx,
            sy,
            f" B{int(zone['zone'])}",
            color="#ffffff",
            fontsize=8,
            va="center",
            ha="left",
            zorder=7,
        )

    ax.set_title(title, color="#ffffff", fontsize=15, pad=14)
    ax.set_xlabel("East/West position (m)", color="#d7dae0")
    ax.set_ylabel("North/South position (m)", color="#d7dae0")
    ax.tick_params(colors="#d7dae0")
    for spine in ax.spines.values():
        spine.set_color("#3c434f")
    leg = ax.legend(loc="upper right", facecolor="#111318", edgecolor="#3c434f")
    for txt in leg.get_texts():
        txt.set_color("#ffffff")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.08)
    ax.grid(True, color="#2a2f38", linewidth=0.7, alpha=0.7)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_dir / "brake_timing_action_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    import plotly.graph_objects as go

    color_map = {"throttle": "#2ca25f", "coast": "#f2c94c", "brake": "#d7191c"}
    fig2 = go.Figure()
    fig2.add_trace(go.Scattergl(
        x=centerline["x_m"],
        y=centerline["y_m"],
        mode="lines",
        line={"color": "rgba(210, 215, 225, 0.24)", "width": 8},
        hoverinfo="skip",
        showlegend=False,
    ))
    for action in ["throttle", "coast", "brake"]:
        t = trace[trace["action"] == action]
        if t.empty:
            continue
        fig2.add_trace(go.Scattergl(
            x=t["x_m"],
            y=t["y_m"],
            mode="markers",
            marker={"size": 7, "color": color_map[action]},
            name=action.title(),
            customdata=np.column_stack([
                t["distance_m"],
                t["time_s"],
                t["speed_mph"],
                t["throttle_pct"],
                t["brake_pct"],
                t["longitudinal_g"],
            ]),
            hovertemplate=(
                "Distance: %{customdata[0]:.1f} m<br>"
                "Lap time: %{customdata[1]:.2f} s<br>"
                "Speed: %{customdata[2]:.1f} mph<br>"
                "Throttle: %{customdata[3]:.1f}%<br>"
                "Brake: %{customdata[4]:.1f}%<br>"
                "Long g: %{customdata[5]:.3f}<extra></extra>"
            ),
        ))
    fig2.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        width=900,
        height=1050,
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    fig2.update_xaxes(
        title="East/West position (m)",
        scaleanchor="y",
        scaleratio=1,
        gridcolor="#2a2f38",
        zeroline=False,
    )
    fig2.update_yaxes(
        title="North/South position (m)",
        gridcolor="#2a2f38",
        zeroline=False,
    )
    fig2.write_html(
        str(out_dir / "brake_timing_action_map.html"),
        include_plotlyjs="cdn",
        full_html=True,
    )


def _write_outputs(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    best_states: pd.DataFrame,
    best_name: str,
) -> None:
    fields = [
        "name", "strategy_kind", "coast_window_m", "brake_margin_m",
        "lookahead_m", "brake_distance_scale", "target_speed_margin_ms",
        "torque_nm", "rpm", "power_limit_kw", "brake_g",
        "combined_score", "endurance_total", "efficiency_score",
        "efficiency_factor", "time_s", "net_kwh", "brake_energy_kwh",
        "brake_time_s", "brake_time_pct", "coast_time_s", "coast_time_pct",
        "mean_brake_pct_when_braking", "max_brake_pct", "final_soc_pct",
        "peak_speed_kmh", "peak_motor_rpm", "peak_motor_torque_nm",
        "peak_pack_current_a", "max_cell_temp_c", "speed_limited_pct",
        "speed_limit_violations", "throttle_segments", "brake_segments",
        "coast_segments", "duration_s",
    ]
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    ranked_all = sorted(clean_rows, key=lambda r: float(r["combined_score"]), reverse=True)
    ranked_valid = [
        row for row in ranked_all
        if int(row.get("speed_limit_violations", 0)) == 0
    ]
    ranked = ranked_valid if ranked_valid else ranked_all
    with (out_dir / "brake_timing_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked_all:
            writer.writerow({field: row.get(field, "") for field in fields})
    (out_dir / "brake_timing_sweep.json").write_text(
        json.dumps({
            "best_valid": ranked_valid[0] if ranked_valid else None,
            "best_raw": ranked_all[0],
            "rows": ranked_all,
        }, indent=2),
        encoding="utf-8",
    )

    _, aim_df = load_cleaned_csv(str(command_sweep.TELEM))
    lap_boundaries = detect_lap_boundaries(aim_df)
    centerline = averaged_centerline(aim_df, lap_boundaries)
    track = command_sweep._CTX["track"]
    trace = _trace_with_xy(track, best_states, centerline)
    zones = _extract_zones(trace)
    trace.to_csv(out_dir / "best_strategy_trace.csv", index=False)
    zones.to_csv(out_dir / "best_strategy_brake_zones.csv", index=False)
    best_states.to_parquet(out_dir / "best_strategy_states.parquet")
    _write_action_map(
        centerline,
        trace,
        zones,
        out_dir,
        title=f"Brake Timing Map - {best_name}",
    )

    lines = [
        "# Brake Timing Strategy Sweep",
        "",
        "Model: corrected no-regen command power cap. The delayed-brake cases "
        "use a corner-only speed cap so lift/brake timing is chosen by the "
        "strategy, not by the normal backward speed envelope.",
        "",
        "## Best Strategy",
        "",
    ]
    best = ranked[0]
    raw_best = ranked_all[0]
    lines.extend([
        f"- Name: `{best['name']}`",
        f"- Tune: `{best['torque_nm']:.1f} Nm`, `{best['rpm']} rpm`, "
        f"`{best['power_limit_kw']:.1f} kW`, `{best['brake_g']:.3f} g` brake cap",
        f"- Policy: coast window `{best['coast_window_m']:.1f} m`, "
        f"brake margin `{best['brake_margin_m']:.1f} m`, "
        f"brake-distance scale `{best['brake_distance_scale']:.3f}`, "
        f"target-speed cushion `{best['target_speed_margin_ms']:.2f} m/s`",
        f"- Points: `{best['combined_score']:.3f}`",
        f"- Time: `{best['time_s']:.1f} s`",
        f"- Net energy: `{best['net_kwh']:.3f} kWh`",
        f"- Brake heat: `{best['brake_energy_kwh']:.3f} kWh`",
        f"- Brake time: `{best['brake_time_pct']:.1f}%`",
        f"- Coast time: `{best['coast_time_pct']:.1f}%`",
        f"- Speed-limit violations: `{best['speed_limit_violations']}`",
        f"- Best raw score: `{raw_best['name']}` at `{raw_best['combined_score']:.3f}` "
        f"points with `{raw_best['speed_limit_violations']}` speed-limit violations.",
        "",
        "## Top Valid Strategies",
        "",
        "| Rank | Name | Torque | RPM | Power | Brake g | Coast m | Margin m | Scale | Cushion m/s | Points | Time s | Net kWh | Brake heat kWh | Brake % | Coast % | Violations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for idx, row in enumerate(ranked[:12], start=1):
        lines.append(
            f"| {idx} | {row['name']} | {row['torque_nm']:.1f} | "
            f"{row['rpm']} | {row['power_limit_kw']:.1f} | "
            f"{row['brake_g']:.3f} | {row['coast_window_m']:.1f} | "
            f"{row['brake_margin_m']:.1f} | {row['brake_distance_scale']:.3f} | "
            f"{row['target_speed_margin_ms']:.2f} | {row['combined_score']:.3f} | "
            f"{row['time_s']:.1f} | {row['net_kwh']:.3f} | "
            f"{row['brake_energy_kwh']:.3f} | {row['brake_time_pct']:.1f} | "
            f"{row['coast_time_pct']:.1f} | {row['speed_limit_violations']} |"
        )
    lines.extend([
        "",
        "## Top Raw Strategies",
        "",
        "| Rank | Name | Torque | RPM | Power | Brake g | Coast m | Margin m | Scale | Cushion m/s | Points | Time s | Net kWh | Brake heat kWh | Brake % | Coast % | Violations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for idx, row in enumerate(ranked_all[:12], start=1):
        lines.append(
            f"| {idx} | {row['name']} | {row['torque_nm']:.1f} | "
            f"{row['rpm']} | {row['power_limit_kw']:.1f} | "
            f"{row['brake_g']:.3f} | {row['coast_window_m']:.1f} | "
            f"{row['brake_margin_m']:.1f} | {row['brake_distance_scale']:.3f} | "
            f"{row['target_speed_margin_ms']:.2f} | {row['combined_score']:.3f} | "
            f"{row['time_s']:.1f} | {row['net_kwh']:.3f} | "
            f"{row['brake_energy_kwh']:.3f} | {row['brake_time_pct']:.1f} | "
            f"{row['coast_time_pct']:.1f} | {row['speed_limit_violations']} |"
        )
    lines.extend([
        "",
        "## Brake Zones On Representative Lap",
        "",
        "| Zone | Start m | End m | Start lap-time s | Duration s | Entry mph | Exit mph | Avg brake % | Coast before s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, zone in zones.iterrows():
        lines.append(
            f"| {int(zone['zone'])} | {zone['start_distance_m']:.1f} | "
            f"{zone['end_distance_m']:.1f} | {zone['start_time_s']:.2f} | "
            f"{zone['duration_s']:.2f} | {zone['entry_speed_mph']:.1f} | "
            f"{zone['exit_speed_mph']:.1f} | {zone['avg_brake_pct']:.1f} | "
            f"{zone['coast_before_s']:.2f} |"
        )
    lines.extend([
        "",
        "Files:",
        "- `brake_timing_action_map.html`",
        "- `brake_timing_action_map.png`",
        "- `brake_timing_sweep.csv`",
        "- `best_strategy_brake_zones.csv`",
        "- `best_strategy_trace.csv`",
    ])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--torque-nm", type=float, default=75.0)
    parser.add_argument("--torques", type=float, nargs="*", default=None)
    parser.add_argument("--rpm", type=int, default=3500)
    parser.add_argument("--rpms", type=int, nargs="*", default=None)
    parser.add_argument("--power-kw", type=float, nargs="*", default=None)
    parser.add_argument("--brake-g", type=float, nargs="*", default=None)
    parser.add_argument("--coast-windows", type=float, nargs="*", default=[0, 5, 10, 20, 30, 40, 60])
    parser.add_argument("--brake-margins", type=float, nargs="*", default=[0, 2, 5, 10])
    parser.add_argument("--lookahead-m", type=float, nargs="*", default=[260.0])
    parser.add_argument("--brake-distance-scales", type=float, nargs="*", default=[1.0])
    parser.add_argument("--target-speed-margins-ms", type=float, nargs="*", default=[0.0])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (
        REPO / "results" / f"brake_timing_strategy_map_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    torques = args.torques if args.torques else [args.torque_nm]
    rpms = args.rpms if args.rpms else [args.rpm]
    power_limits = args.power_kw if args.power_kw else [27.5]
    brake_caps = args.brake_g if args.brake_g else [0.125]

    jobs: list[tuple[str, str, float, float, float, float, float, float, int, float, float]] = []
    for torque_nm in torques:
        for rpm in rpms:
            for power_kw in power_limits:
                for brake_g in brake_caps:
                    jobs.append((
                        f"early_brake_envelope_{torque_nm:g}Nm_{rpm}rpm_{power_kw:g}kW_{brake_g:g}g",
                        "early_brake_envelope",
                        0.0,
                        0.0,
                        260.0,
                        1.0,
                        0.0,
                        float(torque_nm),
                        int(rpm),
                        float(power_kw),
                        float(brake_g),
                    ))
                    for margin in args.brake_margins:
                        for window in args.coast_windows:
                            for lookahead_m in args.lookahead_m:
                                for scale in args.brake_distance_scales:
                                    for target_margin in args.target_speed_margins_ms:
                                        if window == 0:
                                            name = (
                                                f"late_brake_margin_{margin:g}m"
                                                f"_scale_{scale:g}_cushion_{target_margin:g}"
                                            )
                                        else:
                                            name = (
                                                f"coast_{window:g}m_then_brake_margin_{margin:g}m"
                                                f"_scale_{scale:g}_cushion_{target_margin:g}"
                                            )
                                        name = (
                                            f"{name}_{torque_nm:g}Nm_{rpm}rpm_"
                                            f"{power_kw:g}kW_{brake_g:g}g"
                                        )
                                        jobs.append((
                                            name,
                                            "latest_brake",
                                            float(window),
                                            float(margin),
                                            float(lookahead_m),
                                            float(scale),
                                            float(target_margin),
                                            float(torque_nm),
                                            int(rpm),
                                            float(power_kw),
                                            float(brake_g),
                                        ))

    print(f"output_dir={out_dir}")
    print(f"jobs={len(jobs)} workers={args.workers}")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
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
                pool.submit(
                    run_case,
                    name,
                    kind,
                    window,
                    margin,
                    lookahead_m,
                    scale,
                    target_margin,
                    torque_nm,
                    rpm,
                    power_kw,
                    brake_g,
                    False,
                ): (name, kind, window, margin, lookahead_m, scale, target_margin)
                for (
                    name,
                    kind,
                    window,
                    margin,
                    lookahead_m,
                    scale,
                    target_margin,
                    torque_nm,
                    rpm,
                    power_kw,
                    brake_g,
                ) in jobs
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
                        f"[{idx:03d}/{len(jobs)}] {row['name']} -> "
                        f"{row['combined_score']:.3f} pts, "
                        f"{row['time_s']:.1f}s, {row['net_kwh']:.3f}kWh, "
                        f"coast={row['coast_time_pct']:.1f}%"
                    )
                else:
                    print(f"[{idx:03d}/{len(jobs)}] ERROR {row['name']}: {row['error']}")

    if len(rows) != len(jobs):
        print(f"completed {len(rows)} of {len(jobs)} successfully")
        return 2

    valid_rows = [
        row for row in rows
        if int(row.get("speed_limit_violations", 0)) == 0
    ]
    best_pool = valid_rows if valid_rows else rows
    best = max(best_pool, key=lambda row: float(row["combined_score"]))
    _init_worker()
    best_with_states = run_case(
        best["name"],
        best["strategy_kind"],
        best["coast_window_m"],
        best["brake_margin_m"],
        best["lookahead_m"],
        best["brake_distance_scale"],
        best["target_speed_margin_ms"],
        best["torque_nm"],
        int(best["rpm"]),
        best["power_limit_kw"],
        best["brake_g"],
        True,
    )
    if best_with_states.get("status") != "ok" or "_states" not in best_with_states:
        print("failed to rerun best case for map")
        return 3
    _write_outputs(
        rows,
        out_dir,
        best_states=best_with_states["_states"],
        best_name=best["name"],
    )
    print(
        f"best={best['name']} {best['combined_score']:.3f}pts "
        f"time={best['time_s']:.1f}s energy={best['net_kwh']:.3f}kWh"
    )
    print(f"wrote {out_dir / 'README.md'}")
    print(f"map_html {out_dir / 'brake_timing_action_map.html'}")
    print(f"map_png {out_dir / 'brake_timing_action_map.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
