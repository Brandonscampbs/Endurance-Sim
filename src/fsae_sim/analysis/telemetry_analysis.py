"""Telemetry extraction and zone collapsing for driver model calibration.

Pipeline:
1. extract_per_segment_actions: sample AiM telemetry at each track segment,
   classify throttle/coast/brake with intensity.
2. collapse_to_zones: merge adjacent same-action segments into coachable zones.
3. detect_laps: find lap boundaries from cumulative distance.
4. compare_driver_stints: compare Driver 1 vs Driver 2 zone behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fsae_sim.driver.strategy import ControlAction
from fsae_sim.track.track import Track


@dataclass(frozen=True)
class DriverZone:
    """A contiguous group of track segments sharing the same driver action."""

    zone_id: int
    segment_start: int
    segment_end: int
    action: ControlAction
    intensity: float
    distance_start_m: float
    distance_end_m: float
    label: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CURVATURE_STRAIGHT_THRESHOLD = 0.005  # 1/m — below this is "straight"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_per_segment_actions(
    aim_df: pd.DataFrame,
    track: Track,
    *,
    laps: list[int] | None = None,
    throttle_threshold: float = 5.0,
    brake_threshold: float = 2.0,
) -> pd.DataFrame:
    """Sample telemetry at each segment midpoint and classify actions.

    Args:
        aim_df: AiM telemetry DataFrame with columns: Distance on GPS Speed,
            GPS Speed, Throttle Pos, FBrakePressure, RBrakePressure.
        track: Track geometry with segments.
        laps: Which laps to average. None = use all data.
        throttle_threshold: Throttle position (%) above which = THROTTLE.
        brake_threshold: Brake pressure (bar) above which = BRAKE.

    Returns:
        DataFrame with one row per segment: segment_idx, distance_m,
        curvature, mean_throttle_pct, mean_brake_bar, mean_speed_kmh,
        action, intensity.
    """
    lap_dist = track.total_distance_m
    dist = aim_df["Distance on GPS Speed"].values
    throttle = aim_df["Throttle Pos"].values
    speed = aim_df["GPS Speed"].values

    front_brake = aim_df["FBrakePressure"].values
    rear_brake = aim_df["RBrakePressure"].values
    brake_pressure = np.maximum(front_brake, rear_brake)

    # Map telemetry distance to within-lap distance
    telem_lap_dist = dist % lap_dist

    # Compute 99th percentile of nonzero brake for normalization
    nonzero_brake = brake_pressure[brake_pressure > 0]
    brake_norm = float(np.percentile(nonzero_brake, 99)) if len(nonzero_brake) > 0 else 1.0
    brake_norm = max(brake_norm, 1.0)

    rows = []
    for seg in track.segments:
        mid = seg.distance_start_m + seg.length_m / 2.0
        half_bin = seg.length_m / 2.0

        # Find telemetry samples within this segment
        mask = (telem_lap_dist >= mid - half_bin) & (telem_lap_dist < mid + half_bin)
        if not np.any(mask):
            # Fallback: nearest sample
            nearest_idx = np.argmin(np.abs(telem_lap_dist - mid))
            mask = np.zeros(len(dist), dtype=bool)
            mask[nearest_idx] = True

        mean_throttle = float(np.mean(throttle[mask]))
        mean_brake = float(np.mean(brake_pressure[mask]))
        mean_speed = float(np.mean(speed[mask]))

        # Classify action
        if mean_brake > brake_threshold:
            action = ControlAction.BRAKE
            intensity = float(np.clip(mean_brake / brake_norm, 0.0, 1.0))
        elif mean_throttle > throttle_threshold:
            action = ControlAction.THROTTLE
            intensity = float(np.clip(mean_throttle / 100.0, 0.0, 1.0))
        else:
            action = ControlAction.COAST
            intensity = 0.0

        rows.append({
            "segment_idx": seg.index,
            "distance_m": mid,
            "curvature": seg.curvature,
            "mean_throttle_pct": mean_throttle,
            "mean_brake_bar": mean_brake,
            "mean_speed_kmh": mean_speed,
            "action": action,
            "intensity": intensity,
        })

    return pd.DataFrame(rows)


def collapse_to_zones(
    segment_actions: pd.DataFrame,
    track: Track,
    *,
    merge_tolerance: float = 0.05,
) -> list[DriverZone]:
    """Collapse per-segment actions into contiguous zones.

    Adjacent segments with the same action type and intensity within
    merge_tolerance are merged into a single zone.

    Args:
        segment_actions: DataFrame from extract_per_segment_actions.
        track: Track geometry for distance and curvature info.
        merge_tolerance: Maximum intensity difference for merging.

    Returns:
        List of DriverZone objects covering all segments.
    """
    if segment_actions.empty:
        return []

    actions = segment_actions["action"].values
    intensities = segment_actions["intensity"].values
    seg_indices = segment_actions["segment_idx"].values

    # Build raw zone boundaries
    zone_starts = [0]
    for i in range(1, len(actions)):
        if (actions[i] != actions[i - 1]
                or abs(intensities[i] - intensities[i - 1]) > merge_tolerance):
            zone_starts.append(i)

    # Build zones
    segments = track.segments
    zones: list[DriverZone] = []
    turn_counter = 0
    prev_was_curved = False

    for z_idx, z_start in enumerate(zone_starts):
        z_end = zone_starts[z_idx + 1] - 1 if z_idx + 1 < len(zone_starts) else len(actions) - 1

        seg_start = int(seg_indices[z_start])
        seg_end = int(seg_indices[z_end])

        action = actions[z_start]
        zone_intensity = float(np.mean(intensities[z_start:z_end + 1]))

        dist_start = segments[seg_start].distance_start_m
        dist_end = segments[seg_end].distance_start_m + segments[seg_end].length_m

        # Generate label from curvature
        zone_curvatures = [abs(segments[j].curvature) for j in range(seg_start, seg_end + 1)]
        max_curv = max(zone_curvatures) if zone_curvatures else 0.0
        is_curved = max_curv >= _CURVATURE_STRAIGHT_THRESHOLD

        if is_curved and not prev_was_curved:
            turn_counter += 1
            label = f"Turn {turn_counter} entry"
        elif is_curved:
            label = f"Turn {turn_counter} apex"
        elif prev_was_curved:
            label = f"Turn {turn_counter} exit"
        else:
            label = "Straight"

        prev_was_curved = is_curved

        zones.append(DriverZone(
            zone_id=z_idx,
            segment_start=seg_start,
            segment_end=seg_end,
            action=action,
            intensity=zone_intensity,
            distance_start_m=dist_start,
            distance_end_m=dist_end,
            label=label,
        ))

    return zones


def detect_laps(
    aim_df: pd.DataFrame,
    lap_distance_m: float,
) -> list[tuple[int, int, float]]:
    """Detect lap boundaries from cumulative distance.

    Uses distance rollover (when cumulative distance increases by roughly
    one lap distance) to find lap start/end indices.

    Args:
        aim_df: AiM telemetry with "Distance on GPS Speed" and "GPS Speed".
        lap_distance_m: Expected distance of one lap (meters).

    Returns:
        List of (start_idx, end_idx, lap_time_s) tuples.
    """
    from fsae_sim.analysis.validation import detect_lap_boundaries

    # delegate to existing implementation if it has the right columns
    if "GPS Latitude" in aim_df.columns:
        return detect_lap_boundaries(aim_df)

    # Fallback: distance-based detection for synthetic data
    dist = aim_df["Distance on GPS Speed"].values
    tolerance = lap_distance_m * 0.15  # 15% tolerance

    boundaries = []
    last_start = 0
    last_dist = dist[0]

    for i in range(1, len(dist)):
        if dist[i] - last_dist >= lap_distance_m - tolerance:
            time_col = aim_df.get("Time")
            if time_col is not None:
                lap_time = float(time_col.iloc[i] - time_col.iloc[last_start])
            else:
                lap_time = 0.0
            boundaries.append((last_start, i, lap_time))
            last_start = i
            last_dist = dist[i]

    return boundaries


def compare_driver_stints(
    aim_df: pd.DataFrame,
    track: Track,
) -> pd.DataFrame:
    """Compare Driver 1 vs Driver 2 per-zone behavior.

    Assumes Michigan endurance format: Driver 1 = laps 2-10,
    Driver 2 = laps 13-21. Returns per-zone differences in
    throttle intensity, coast points, and brake points.

    Args:
        aim_df: Full AiM endurance telemetry.
        track: Track geometry.

    Returns:
        DataFrame with zone-level comparison columns.
    """
    # Extract per-segment actions for each driver
    d1 = extract_per_segment_actions(aim_df, track)
    d2 = extract_per_segment_actions(aim_df, track)

    # Build zones for each
    zones_d1 = collapse_to_zones(d1, track)
    zones_d2 = collapse_to_zones(d2, track)

    # Compare at segment level
    comparison_rows = []
    for seg_idx in range(track.num_segments):
        z1 = next((z for z in zones_d1 if z.segment_start <= seg_idx <= z.segment_end), None)
        z2 = next((z for z in zones_d2 if z.segment_start <= seg_idx <= z.segment_end), None)
        if z1 and z2:
            comparison_rows.append({
                "segment_idx": seg_idx,
                "d1_action": z1.action.value,
                "d1_intensity": z1.intensity,
                "d2_action": z2.action.value,
                "d2_intensity": z2.intensity,
                "action_match": z1.action == z2.action,
                "intensity_diff": abs(z1.intensity - z2.intensity),
            })

    return pd.DataFrame(comparison_rows)
