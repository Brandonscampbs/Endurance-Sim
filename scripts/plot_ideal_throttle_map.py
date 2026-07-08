"""Generate a 2D track map colored by ideal-driver throttle input."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.analysis.validation import (  # noqa: E402
    detect_lap_boundaries,
    telemetry_speed_col,
)
from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402
from fsae_sim.driver.strategies import CoastOptimalStrategy  # noqa: E402
from fsae_sim.sim.engine import SimulationEngine, SimulationMode  # noqa: E402
from fsae_sim.track.track import (  # noqa: E402
    Track,
    _M_PER_DEG_LAT,
    _periodic_gaussian_filter,
)
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

ENDURANCE_HOLDOUT_LAPS_ZERO_BASED = tuple(range(12, 22))


def averaged_centerline(
    aim_df: pd.DataFrame,
    lap_boundaries: list[tuple[int, int, float]],
    *,
    bin_size_m: float = 0.5,
    centerline_sigma_m: float = 1.25,
) -> pd.DataFrame:
    """Return the same averaged GPS centerline shape used for the track."""
    if not lap_boundaries:
        raise RuntimeError("No lap boundaries detected; cannot build map.")
    if not {"GPS Latitude", "GPS Longitude"}.issubset(aim_df.columns):
        raise RuntimeError("GPS Latitude/Longitude are required for 2D map.")

    lat_med = float(np.median(aim_df["GPS Latitude"].values))
    m_per_deg_lon = _M_PER_DEG_LAT * float(np.cos(np.radians(lat_med)))

    ref_idx = lap_boundaries[0][0]
    lat0 = float(aim_df["GPS Latitude"].iloc[ref_idx])
    lon0 = float(aim_df["GPS Longitude"].iloc[ref_idx])

    lap_lengths = np.array([lap_d for _, _, lap_d in lap_boundaries], dtype=float)
    mean_lap_length = float(lap_lengths.mean())
    n_grid = int(np.ceil(mean_lap_length / bin_size_m)) + 1
    s_grid = np.linspace(0.0, mean_lap_length, n_grid)

    x_stack: list[np.ndarray] = []
    y_stack: list[np.ndarray] = []
    weights: list[float] = []
    has_pos_acc = "GPS PosAccuracy" in aim_df.columns

    for s_idx, e_idx, _lap_d in lap_boundaries:
        lap = aim_df.iloc[s_idx:e_idx]
        lat = lap["GPS Latitude"].values
        lon = lap["GPS Longitude"].values
        dist = lap["Distance on GPS Speed"].values

        if (
            not np.all(np.isfinite(lat))
            or not np.all(np.isfinite(lon))
            or not np.all(np.isfinite(dist))
            or dist[-1] - dist[0] <= 0.0
        ):
            continue

        x = (lon - lon0) * m_per_deg_lon
        y = (lat - lat0) * _M_PER_DEG_LAT
        dist_lap = dist - dist[0]
        s_lap = dist_lap * (mean_lap_length / dist_lap[-1])

        if has_pos_acc:
            bad_frac = float(np.mean(lap["GPS PosAccuracy"].values == 200.0))
            weight = max(0.0, 1.0 - bad_frac)
            if weight <= 0.0:
                continue
        else:
            weight = 1.0

        x_stack.append(np.interp(s_grid, s_lap, x))
        y_stack.append(np.interp(s_grid, s_lap, y))
        weights.append(weight)

    if not x_stack:
        raise RuntimeError("No usable GPS laps found for centerline map.")

    x_arr = np.stack(x_stack, axis=0)
    y_arr = np.stack(y_stack, axis=0)
    w = np.asarray(weights, dtype=float)
    x_mean = (x_arr * w[:, None]).sum(axis=0) / float(w.sum())
    y_mean = (y_arr * w[:, None]).sum(axis=0) / float(w.sum())

    ds = mean_lap_length / (n_grid - 1)
    sigma_samples = max(centerline_sigma_m / ds, 1e-6)
    x_smooth = _periodic_gaussian_filter(x_mean, sigma_samples)
    y_smooth = _periodic_gaussian_filter(y_mean, sigma_samples)

    return pd.DataFrame({
        "distance_m": s_grid,
        "x_m": x_smooth,
        "y_m": y_smooth,
    })


def run_ideal_lap(rpm: int) -> tuple[Track, pd.DataFrame, object]:
    vehicle = VehicleConfig.from_yaml(str(CONFIG))
    _, aim_df = load_cleaned_csv(str(TELEM))
    speed_col = telemetry_speed_col(aim_df)
    lap_boundaries = detect_lap_boundaries(aim_df)
    validation_laps = [
        lap for lap in ENDURANCE_HOLDOUT_LAPS_ZERO_BASED
        if lap < len(lap_boundaries)
    ]

    vehicle = replace(
        vehicle,
        powertrain=replace(
            vehicle.powertrain,
            motor_speed_max_rpm=float(rpm),
            lvcu_overspeed_rpm=float(rpm),
        ),
    )
    track = Track.from_telemetry(df=aim_df)
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(VOLTT))
    battery.calibrate_pack_from_telemetry(
        aim_df,
        holdout_laps=tuple(lap + 1 for lap in validation_laps),
    )
    strategy = CoastOptimalStrategy.from_config(vehicle, track)
    initial_speed_ms = (
        float(aim_df[speed_col].iloc[lap_boundaries[0][0]]) / 3.6
        if lap_boundaries else 0.0
    )

    result = SimulationEngine(
        vehicle,
        track,
        strategy,
        copy.copy(battery),
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    ).run(
        num_laps=1,
        initial_soc_pct=95.0,
        initial_temp_c=29.0,
        initial_speed_ms=initial_speed_ms,
    )
    return track, aim_df, result


def segment_trace(track: Track, states: pd.DataFrame, centerline: pd.DataFrame) -> pd.DataFrame:
    lap = int(states["lap"].min()) if "lap" in states and not states.empty else 0
    lap_states = states[states["lap"] == lap].copy()
    lap_states = lap_states.sort_values("segment_idx")

    rows: list[dict[str, float]] = []
    for _, row in lap_states.iterrows():
        seg_idx = int(row["segment_idx"])
        if seg_idx < 0 or seg_idx >= track.num_segments:
            continue
        seg = track.segments[seg_idx]
        mid_d = seg.distance_start_m + seg.length_m / 2.0
        x = float(np.interp(mid_d, centerline["distance_m"], centerline["x_m"]))
        y = float(np.interp(mid_d, centerline["distance_m"], centerline["y_m"]))
        rows.append({
            "segment_idx": float(seg_idx),
            "distance_m": mid_d,
            "x_m": x,
            "y_m": y,
            "throttle_pct": float(row["throttle_pct"]) * 100.0,
            "speed_mph": float(row["speed_kmh"]) * 0.621371,
            "motor_rpm": float(row["motor_rpm"]),
            "action": row["action"],
        })
    return pd.DataFrame(rows)


def write_static_png(
    centerline: pd.DataFrame,
    trace: pd.DataFrame,
    out_path: Path,
    *,
    rpm: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    x = centerline["x_m"].to_numpy(dtype=float)
    y = centerline["y_m"].to_numpy(dtype=float)
    points = np.column_stack([x, y])
    line_segments = np.stack([points[:-1], points[1:]], axis=1)

    edge_mid = (
        centerline["distance_m"].to_numpy(dtype=float)[:-1]
        + np.diff(centerline["distance_m"].to_numpy(dtype=float)) / 2.0
    )
    throttle = np.interp(
        edge_mid,
        trace["distance_m"].to_numpy(dtype=float),
        trace["throttle_pct"].to_numpy(dtype=float),
        left=float(trace["throttle_pct"].iloc[0]),
        right=float(trace["throttle_pct"].iloc[-1]),
    )

    fig, ax = plt.subplots(figsize=(7.5, 10.0), facecolor="#111318")
    ax.set_facecolor("#111318")
    ax.plot(x, y, color="#2b3038", linewidth=8, solid_capstyle="round", zorder=1)

    collection = LineCollection(
        line_segments,
        cmap="turbo",
        norm=Normalize(vmin=0.0, vmax=100.0),
        linewidth=5.0,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    collection.set_array(throttle)
    ax.add_collection(collection)

    ax.scatter([x[0]], [y[0]], s=90, c="#ffffff", edgecolors="#111318", zorder=3)
    ax.text(
        x[0],
        y[0],
        " Start/Finish",
        color="#ffffff",
        fontsize=10,
        va="center",
        ha="left",
        zorder=4,
    )

    cbar = fig.colorbar(collection, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Throttle input (%)", color="#ffffff")
    cbar.ax.yaxis.set_tick_params(color="#ffffff")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#ffffff")

    ax.set_title(
        f"Throttle Map - Ideal Coast ({rpm} rpm)",
        color="#ffffff",
        fontsize=15,
        pad=14,
    )
    ax.set_xlabel("East/West position (m)", color="#d7dae0")
    ax.set_ylabel("North/South position (m)", color="#d7dae0")
    ax.tick_params(colors="#d7dae0")
    for spine in ax.spines.values():
        spine.set_color("#3c434f")
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.08)
    ax.grid(True, color="#2a2f38", linewidth=0.7, alpha=0.7)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_interactive_html(
    centerline: pd.DataFrame,
    trace: pd.DataFrame,
    out_path: Path,
    *,
    rpm: int,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=centerline["x_m"],
        y=centerline["y_m"],
        mode="lines",
        line={"color": "rgba(210, 215, 225, 0.26)", "width": 8},
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
        x=trace["x_m"],
        y=trace["y_m"],
        mode="markers",
        marker={
            "size": 7,
            "color": trace["throttle_pct"],
            "colorscale": "Turbo",
            "cmin": 0,
            "cmax": 100,
            "colorbar": {"title": "Throttle %"},
        },
        customdata=np.column_stack([
            trace["distance_m"],
            trace["throttle_pct"],
            trace["speed_mph"],
            trace["motor_rpm"],
            trace["action"],
        ]),
        hovertemplate=(
            "Distance: %{customdata[0]:.1f} m<br>"
            "Throttle: %{customdata[1]:.1f}%<br>"
            "Speed: %{customdata[2]:.1f} mph<br>"
            "Motor: %{customdata[3]:.0f} rpm<br>"
            "Action: %{customdata[4]}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
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

    fig.update_layout(
        title=f"Throttle Map - Ideal Coast ({rpm} rpm)",
        template="plotly_dark",
        paper_bgcolor="#111318",
        plot_bgcolor="#111318",
        width=820,
        height=1000,
        margin={"l": 40, "r": 30, "t": 70, "b": 40},
    )
    fig.update_xaxes(
        title="East/West position (m)",
        scaleanchor="y",
        scaleratio=1,
        gridcolor="#2a2f38",
        zeroline=False,
    )
    fig.update_yaxes(
        title="North/South position (m)",
        gridcolor="#2a2f38",
        zeroline=False,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)


def write_summary(
    out_dir: Path,
    *,
    rpm: int,
    result: object,
    trace: pd.DataFrame,
) -> None:
    states = result.states
    summary = {
        "model": "ideal_coast",
        "rpm": int(rpm),
        "lap_time_s": float(result.total_time_s),
        "net_energy_kwh": float(result.net_energy_kwh),
        "peak_speed_mph": float(states["speed_kmh"].max()) * 0.621371,
        "peak_motor_rpm": float(states["motor_rpm"].max()),
        "mean_throttle_pct": float(trace["throttle_pct"].mean()),
        "full_throttle_pct_of_segments": float((trace["throttle_pct"] >= 99.0).mean() * 100.0),
        "coast_pct_of_segments": float((trace["throttle_pct"] <= 1.0).mean() * 100.0),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Ideal Coast Throttle Map",
        "",
        f"RPM cap: `{rpm}`",
        "",
        f"- Lap time: `{summary['lap_time_s']:.2f} s`",
        f"- Peak speed: `{summary['peak_speed_mph']:.1f} mph`",
        f"- Peak motor speed: `{summary['peak_motor_rpm']:.0f} rpm`",
        f"- Mean throttle input: `{summary['mean_throttle_pct']:.1f}%`",
        f"- Full-throttle segments: `{summary['full_throttle_pct_of_segments']:.1f}%`",
        f"- Coast segments: `{summary['coast_pct_of_segments']:.1f}%`",
        "",
        "Files:",
        "- `ideal_coast_throttle_map.html`",
        "- `ideal_coast_throttle_map.png`",
        "- `ideal_coast_throttle_trace.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpm", type=int, default=3700)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/ideal_coast_throttle_map_<timestamp>.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO / "results" / f"ideal_coast_throttle_map_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        track, aim_df, result = run_ideal_lap(args.rpm)
        lap_boundaries = detect_lap_boundaries(aim_df)
        centerline = averaged_centerline(aim_df, lap_boundaries)
        trace = segment_trace(track, result.states, centerline)

    trace.to_csv(out_dir / "ideal_coast_throttle_trace.csv", index=False)
    result.states.to_parquet(out_dir / "ideal_coast_lap_states.parquet")

    write_static_png(centerline, trace, out_dir / "ideal_coast_throttle_map.png", rpm=args.rpm)
    write_interactive_html(centerline, trace, out_dir / "ideal_coast_throttle_map.html", rpm=args.rpm)
    write_summary(out_dir, rpm=args.rpm, result=result, trace=trace)

    print(f"wrote {out_dir}")
    print(f"html {out_dir / 'ideal_coast_throttle_map.html'}")
    print(f"png  {out_dir / 'ideal_coast_throttle_map.png'}")


if __name__ == "__main__":
    main()
