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
    max_speed_ms: float = 0.0  # observed peak speed in this zone (m/s), 0 = no limit


# ---------------------------------------------------------------------------
# Tire grip calibration
# ---------------------------------------------------------------------------


def extract_tire_grip_scale(
    aim_df: pd.DataFrame,
    mass_kg: float,
    cla: float,
    tire_model,
    fz_representative: float,
    *,
    min_speed_kmh: float = 15.0,
    min_lat_g: float = 0.3,
    percentile: float = 95.0,
    rho: float = 1.225,
) -> dict:
    """Extract tire grip scale factor from endurance telemetry.

    Computes the car's real effective friction coefficient from lateral
    acceleration data and compares to the Pacejka model's peak mu.
    The ratio is the LMUY scaling factor needed to calibrate TTC rig
    data to on-car grip.

    Args:
        aim_df: AiM telemetry DataFrame with GPS LatAcc (g) and GPS Speed (km/h).
        mass_kg: Total vehicle mass including driver (kg).
        cla: Downforce coefficient * area (ClA, m^2).
        tire_model: PacejkaTireModel instance (uncalibrated).
        fz_representative: Representative per-tire normal load (N) for
            computing Pacejka peak mu.
        min_speed_kmh: Minimum speed to include (filters parking/pit).
        min_lat_g: Minimum lateral G to include (filters straights).
        percentile: Percentile for peak grip extraction (default 95th).
        rho: Air density (kg/m^3).

    Returns:
        Dict with keys: grip_scale, effective_mu_95, pacejka_mu,
        n_samples, peak_lat_g.
    """
    g = 9.81
    speed_kmh = aim_df["GPS Speed"].values
    lat_g = np.abs(aim_df["GPS LatAcc"].values)

    # Filter: moving and cornering
    mask = (speed_kmh > min_speed_kmh) & (lat_g > min_lat_g)
    if np.sum(mask) < 10:
        raise ValueError(
            f"Not enough cornering samples: {np.sum(mask)} "
            f"(need >= 10 with speed > {min_speed_kmh} km/h and |lat_g| > {min_lat_g})"
        )

    speed_ms = speed_kmh[mask] * (1000.0 / 3600.0)
    lat_g_filtered = lat_g[mask]

    # Effective mu: accounts for downforce augmenting normal force
    lateral_force = mass_kg * lat_g_filtered * g
    downforce = 0.5 * rho * cla * speed_ms ** 2
    total_normal = mass_kg * g + downforce
    effective_mu = lateral_force / total_normal

    mu_at_percentile = float(np.percentile(effective_mu, percentile))

    # Pacejka peak mu at representative load
    pacejka_peak_fy = tire_model.peak_lateral_force(fz_representative)
    pacejka_mu = pacejka_peak_fy / fz_representative

    grip_scale = mu_at_percentile / pacejka_mu

    return {
        "grip_scale": float(grip_scale),
        "effective_mu_95": float(mu_at_percentile),
        "pacejka_mu": float(pacejka_mu),
        "n_samples": int(np.sum(mask)),
        "peak_lat_g": float(np.percentile(lat_g_filtered, percentile)),
    }


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
    brake_max_pressure_bar: float | None = None,
    throttle_col: str = "Throttle Pos",
    front_brake_col: str = "FBrakePressure",
    rear_brake_col: str = "RBrakePressure",
    speed_col: str = "GPS Speed",
    distance_col: str = "Distance on GPS Speed",
) -> pd.DataFrame:
    """Sample telemetry at each segment midpoint and classify actions.

    Classifies each segment per-lap independently, then aggregates across
    laps using majority vote for action and median for intensity. This
    prevents cross-lap averaging from washing out coast/throttle contrast.

    If lap boundaries cannot be detected (e.g. synthetic data without GPS),
    falls back to single-pass classification using all data.

    Args:
        aim_df: AiM telemetry DataFrame with columns: Distance on GPS Speed,
            GPS Speed, Throttle Pos, FBrakePressure, RBrakePressure.
        track: Track geometry with segments.
        laps: Which lap indices (0-based into detected laps) to average.
            None = auto-select non-outlier laps (skip first, skip short laps,
            skip driver-change lap).
        throttle_threshold: Throttle position (%) above which = THROTTLE.
        brake_threshold: Brake pressure (bar) above which = BRAKE.

    Returns:
        DataFrame with one row per segment: segment_idx, distance_m,
        curvature, mean_throttle_pct, mean_brake_bar, mean_speed_kmh,
        action, intensity.
    """
    # Try to detect lap boundaries for per-lap classification
    lap_boundaries = _detect_lap_boundaries_safe(aim_df)

    if lap_boundaries and len(lap_boundaries) >= 2:
        return _extract_per_lap_then_aggregate(
            aim_df, track, lap_boundaries,
            laps=laps,
            throttle_threshold=throttle_threshold,
            brake_threshold=brake_threshold,
            brake_max_pressure_bar=brake_max_pressure_bar,
            throttle_col=throttle_col,
            front_brake_col=front_brake_col,
            rear_brake_col=rear_brake_col,
            speed_col=speed_col,
            distance_col=distance_col,
        )
    else:
        # Fallback for synthetic/simple data: single-pass
        return _extract_single_pass(
            aim_df, track,
            throttle_threshold=throttle_threshold,
            brake_threshold=brake_threshold,
            brake_max_pressure_bar=brake_max_pressure_bar,
            throttle_col=throttle_col,
            front_brake_col=front_brake_col,
            rear_brake_col=rear_brake_col,
            speed_col=speed_col,
            distance_col=distance_col,
        )


def _auto_select_laps(
    aim_df: pd.DataFrame,
    lap_boundaries: list[tuple[int, int, float]],
    *,
    distance_tolerance: float = 0.15,
    time_tolerance: float = 0.20,
    min_mean_speed_fraction: float = 0.70,
) -> list[tuple[int, int, float]]:
    """D-18: filter laps by distance AND time AND mean_speed.

    The old filter only checked distance (±15% of median), which let the
    driver-change lap through (normal distance, long dwell time at
    stationary pause, low mean speed). Now we also reject laps whose
    duration is >20% off the median and whose mean moving speed is
    below 70% of the across-laps median mean speed.

    Always skips lap index 0 (warmup / pit-out partial).
    """
    if len(lap_boundaries) == 0:
        return []

    # Compute per-lap duration and mean speed.
    durations: list[float] = []
    mean_speeds: list[float] = []
    has_time = "Time" in aim_df.columns
    speed_col = aim_df.get("GPS Speed")
    n_rows = len(aim_df)
    for start_idx, end_idx, _ in lap_boundaries:
        # end_idx is exclusive; clamp for `iloc` to stay within bounds.
        end_ref = min(end_idx, n_rows) - 1
        start_ref = max(0, min(start_idx, n_rows - 1))
        if has_time and end_ref > start_ref:
            t = aim_df["Time"].iloc[end_ref] - aim_df["Time"].iloc[start_ref]
            durations.append(float(t))
        else:
            durations.append(float("nan"))
        if speed_col is not None:
            v = speed_col.iloc[start_idx:end_idx].values
            mean_speeds.append(float(np.mean(v)) if len(v) else 0.0)
        else:
            mean_speeds.append(float("nan"))

    median_dist = float(np.median([d for _, _, d in lap_boundaries]))
    # Medians robust to the outlier we're trying to reject.
    finite_dur = [d for d in durations if np.isfinite(d)]
    median_dur = float(np.median(finite_dur)) if finite_dur else float("nan")
    finite_speed = [s for s in mean_speeds if np.isfinite(s)]
    median_speed = float(np.median(finite_speed)) if finite_speed else float("nan")

    selected: list[tuple[int, int, float]] = []
    for i, (s, e, d) in enumerate(lap_boundaries):
        if i == 0:
            continue
        if abs(d - median_dist) > median_dist * distance_tolerance:
            continue
        if np.isfinite(median_dur) and np.isfinite(durations[i]):
            if abs(durations[i] - median_dur) > median_dur * time_tolerance:
                continue
        if np.isfinite(median_speed) and np.isfinite(mean_speeds[i]):
            if mean_speeds[i] < min_mean_speed_fraction * median_speed:
                continue
        selected.append((s, e, d))

    return selected


def _detect_lap_boundaries_safe(
    aim_df: pd.DataFrame,
) -> list[tuple[int, int, float]]:
    """Detect lap boundaries, returning empty list on failure."""
    if "GPS Latitude" not in aim_df.columns:
        return []
    # NF-13: catch only the narrow "no boundary crossings detected" / "missing
    # expected column" failure modes. A true loader/parse failure must surface
    # loudly — silent empty-list returns produce "calibrated from no laps"
    # results that look fine and are silently wrong.
    try:
        from fsae_sim.analysis.validation import detect_lap_boundaries
        return detect_lap_boundaries(aim_df)
    except (KeyError, IndexError):
        return []


def _extract_per_lap_then_aggregate(
    aim_df: pd.DataFrame,
    track: Track,
    lap_boundaries: list[tuple[int, int, float]],
    *,
    laps: list[int] | None = None,
    throttle_threshold: float = 5.0,
    brake_threshold: float = 2.0,
    brake_max_pressure_bar: float | None = None,
    throttle_col: str = "Throttle Pos",
    front_brake_col: str = "FBrakePressure",
    rear_brake_col: str = "RBrakePressure",
    speed_col: str = "GPS Speed",
    distance_col: str = "Distance on GPS Speed",
) -> pd.DataFrame:
    """Classify per-lap, then aggregate across laps with majority vote.

    Uses LVCU Torque Req (if available) for throttle intensity instead of
    raw pedal position, since the pedal-to-torque mapping in the real car
    is nonlinear (LVCU limit 150 Nm, inverter clips to 85 Nm). This ensures
    the sim's ``throttle_pct * max_torque(rpm)`` produces the same force as
    the real car's torque request.
    """
    num_segments = track.num_segments

    # Check if we have LVCU torque data for better intensity calibration
    has_torque = "LVCU Torque Req" in aim_df.columns

    # Compute inverter torque limit for normalizing torque → throttle_pct
    # The sim uses: drive_force = throttle_pct * max_motor_torque(rpm)
    # where max_motor_torque = min(inverter, lvcu) = 85 Nm below brake_speed.
    # So to match: throttle_pct = actual_torque / 85
    _INVERTER_TORQUE_LIMIT = 85.0

    # Select which laps to use
    if laps is not None:
        selected = [lap_boundaries[i] for i in laps if i < len(lap_boundaries)]
    else:
        selected = _auto_select_laps(aim_df, lap_boundaries)

    if not selected:
        selected = lap_boundaries

    # D-08: brake normalization is now data-independent. Prefer the
    # DSS-derived `brake_max_pressure_bar` passed in by the caller;
    # fall back to a 60 bar FSAE-typical constant. The old 99th-percentile
    # approach made `brake_pct` depend on which laps were in `aim_df`.
    if brake_max_pressure_bar is not None:
        brake_norm = float(brake_max_pressure_bar)
    else:
        brake_norm = 60.0
    brake_norm = max(brake_norm, 1.0)

    # Per-lap, per-segment classification
    action_codes = []
    intensity_vals = []  # normalized intensity for the winning action
    throttle_vals = []   # raw throttle % for reporting
    brake_vals = []
    speed_vals = []

    for start_idx, end_idx, _ in selected:
        lap_df = aim_df.iloc[start_idx:end_idx]
        lap_dist_raw = lap_df[distance_col].values
        # D-07: rescale each lap's arc-length onto the track total distance.
        # The AiM "Distance on GPS Speed" channel accumulates with some
        # drift across laps (GPS noise, slight route variation). A raw
        # offset `lap_dist_raw - lap_dist_raw[0]` leaves each lap with a
        # different terminal distance, so per-segment midpoint lookups land
        # at physically wrong locations on later laps — medians wrap around
        # and wash out the turn-1/turn-N contrast.
        lap_span = float(lap_dist_raw[-1] - lap_dist_raw[0])
        if lap_span > 0.0:
            lap_d = (lap_dist_raw - lap_dist_raw[0]) * (
                track.total_distance_m / lap_span
            )
        else:
            lap_d = lap_dist_raw - lap_dist_raw[0]
        lap_throttle = lap_df[throttle_col].values
        lap_speed = lap_df[speed_col].values
        lap_fbr = lap_df[front_brake_col].values
        lap_rbr = lap_df[rear_brake_col].values
        lap_brake = np.maximum(lap_fbr, lap_rbr)

        if has_torque:
            lap_torque = lap_df["LVCU Torque Req"].values
        else:
            lap_torque = None

        lap_actions = np.zeros(num_segments, dtype=int)
        lap_intensities = np.zeros(num_segments)
        lap_throttles = np.zeros(num_segments)
        lap_brakes = np.zeros(num_segments)
        lap_speeds = np.zeros(num_segments)

        last_idx = len(track.segments) - 1
        for seg in track.segments:
            mid = seg.distance_start_m + seg.length_m / 2.0
            half_bin = seg.length_m / 2.0

            # Final segment uses an inclusive upper bound so the very
            # last telemetry sample (d == lap_length) isn't dropped into
            # an empty window (previously fell back to the nearest-sample
            # heuristic instead of being binned with its segment).
            if seg.index == last_idx:
                mask = (lap_d >= mid - half_bin) & (lap_d <= mid + half_bin)
            else:
                mask = (lap_d >= mid - half_bin) & (lap_d < mid + half_bin)
            if not np.any(mask):
                nearest_idx = np.argmin(np.abs(lap_d - mid))
                mask = np.zeros(len(lap_d), dtype=bool)
                mask[nearest_idx] = True

            seg_throttle = float(np.median(lap_throttle[mask]))
            seg_brake = float(np.median(lap_brake[mask]))
            seg_speed = float(np.mean(lap_speed[mask]))

            if seg_brake > brake_threshold:
                lap_actions[seg.index] = 2
                lap_intensities[seg.index] = float(np.clip(seg_brake / brake_norm, 0.0, 1.0))
            elif seg_throttle > throttle_threshold:
                lap_actions[seg.index] = 1
                # Use torque-based intensity if available — this is the
                # effective torque fraction after LVCU processing.
                if lap_torque is not None:
                    # NF-25: LVCU Torque Req has sensor dropouts (NaN).
                    # `np.median` propagates NaN; drop non-finite samples
                    # before reducing. Fallback to throttle-based intensity
                    # if no finite torque samples remain.
                    raw_torque = np.clip(lap_torque[mask], 0, None)
                    finite = raw_torque[np.isfinite(raw_torque)]
                    if finite.size:
                        seg_torque = float(np.median(finite))
                        lap_intensities[seg.index] = float(np.clip(
                            seg_torque / _INVERTER_TORQUE_LIMIT, 0.0, 1.0,
                        ))
                    else:
                        lap_intensities[seg.index] = float(np.clip(
                            seg_throttle / 100.0, 0.0, 1.0,
                        ))
                else:
                    lap_intensities[seg.index] = float(np.clip(seg_throttle / 100.0, 0.0, 1.0))
            else:
                lap_actions[seg.index] = 0
                lap_intensities[seg.index] = 0.0

            lap_throttles[seg.index] = seg_throttle
            lap_brakes[seg.index] = seg_brake
            lap_speeds[seg.index] = seg_speed

        action_codes.append(lap_actions)
        intensity_vals.append(lap_intensities)
        throttle_vals.append(lap_throttles)
        brake_vals.append(lap_brakes)
        speed_vals.append(lap_speeds)

    # Stack into arrays: (num_laps, num_segments)
    action_matrix = np.array(action_codes)
    intensity_matrix = np.array(intensity_vals)
    throttle_matrix = np.array(throttle_vals)
    brake_matrix = np.array(brake_vals)
    speed_matrix = np.array(speed_vals)

    # Aggregate: majority vote for action, median for intensity
    rows = []
    for seg in track.segments:
        i = seg.index
        mid = seg.distance_start_m + seg.length_m / 2.0

        action_col = action_matrix[:, i]
        counts = np.bincount(action_col, minlength=3)
        winner = int(np.argmax(counts))

        # Median intensity across laps where the winning action was chosen.
        # NF-25: guard against NaN propagation from upstream dropouts.
        winner_mask = action_col == winner
        if np.any(winner_mask):
            intensities_for_winner = intensity_matrix[:, i][winner_mask]
            finite = intensities_for_winner[np.isfinite(intensities_for_winner)]
            med_intensity = float(np.median(finite)) if finite.size else 0.0
        else:
            col = intensity_matrix[:, i]
            finite = col[np.isfinite(col)]
            med_intensity = float(np.median(finite)) if finite.size else 0.0

        med_throttle = float(np.median(throttle_matrix[:, i]))
        med_brake = float(np.median(brake_matrix[:, i]))
        mean_speed = float(np.mean(speed_matrix[:, i]))

        if winner == 2:
            action = ControlAction.BRAKE
        elif winner == 1:
            action = ControlAction.THROTTLE
        else:
            action = ControlAction.COAST
            med_intensity = 0.0

        rows.append({
            "segment_idx": i,
            "distance_m": mid,
            "curvature": seg.curvature,
            "mean_throttle_pct": med_throttle,
            "mean_brake_bar": med_brake,
            "mean_speed_kmh": mean_speed,
            "action": action,
            "intensity": med_intensity,
        })

    return pd.DataFrame(rows)


def _extract_single_pass(
    aim_df: pd.DataFrame,
    track: Track,
    *,
    throttle_threshold: float = 5.0,
    brake_threshold: float = 2.0,
    brake_max_pressure_bar: float | None = None,
    throttle_col: str = "Throttle Pos",
    front_brake_col: str = "FBrakePressure",
    rear_brake_col: str = "RBrakePressure",
    speed_col: str = "GPS Speed",
    distance_col: str = "Distance on GPS Speed",
) -> pd.DataFrame:
    """Single-pass extraction for synthetic data without lap boundaries."""
    lap_dist = track.total_distance_m
    dist = aim_df[distance_col].values
    throttle = aim_df[throttle_col].values
    speed = aim_df[speed_col].values

    front_brake = aim_df[front_brake_col].values
    rear_brake = aim_df[rear_brake_col].values
    brake_pressure = np.maximum(front_brake, rear_brake)

    telem_lap_dist = dist % lap_dist

    # D-08: data-independent brake normalization (see _extract_per_lap_then_aggregate).
    if brake_max_pressure_bar is not None:
        brake_norm = float(brake_max_pressure_bar)
    else:
        brake_norm = 60.0
    brake_norm = max(brake_norm, 1.0)

    rows = []
    last_idx = len(track.segments) - 1
    for seg in track.segments:
        mid = seg.distance_start_m + seg.length_m / 2.0
        half_bin = seg.length_m / 2.0

        # Inclusive upper bound on the final segment so the lap-end
        # sample lands in its segment window.
        if seg.index == last_idx:
            mask = (telem_lap_dist >= mid - half_bin) & (telem_lap_dist <= mid + half_bin)
        else:
            mask = (telem_lap_dist >= mid - half_bin) & (telem_lap_dist < mid + half_bin)
        if not np.any(mask):
            nearest_idx = np.argmin(np.abs(telem_lap_dist - mid))
            mask = np.zeros(len(dist), dtype=bool)
            mask[nearest_idx] = True

        mean_throttle = float(np.mean(throttle[mask]))
        mean_brake = float(np.mean(brake_pressure[mask]))
        mean_speed = float(np.mean(speed[mask]))

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
    merge_tolerance: float = 0.15,
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

        # S10: use 95th percentile of per-segment speeds as the zone's
        # observed peak — mean-of-means washes out the corner-exit peak
        # that downstream speed-cap logic needs. Robust to single-sample
        # outliers (vs np.max) while still tracking the true peak.
        if "mean_speed_kmh" in segment_actions.columns:
            zone_speeds = segment_actions["mean_speed_kmh"].values[z_start:z_end + 1]
            max_speed_kmh = float(np.percentile(zone_speeds, 95))
            max_speed_ms = max_speed_kmh / 3.6
        else:
            max_speed_ms = 0.0

        zones.append(DriverZone(
            zone_id=z_idx,
            segment_start=seg_start,
            segment_end=seg_end,
            action=action,
            intensity=zone_intensity,
            distance_start_m=dist_start,
            distance_end_m=dist_end,
            label=label,
            max_speed_ms=max_speed_ms,
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


def _auto_split_driver_laps(
    aim_df: pd.DataFrame,
    lap_boundaries: list[tuple[int, int, float]],
) -> tuple[list[int], list[int]]:
    """Auto-detect driver change by finding the longest stationary pause.

    Returns ``(pre_change_laps, post_change_laps)`` as index lists into
    ``lap_boundaries``.  If no lap-internal pause is detected, falls back
    to a simple midpoint split.
    """
    speed_col = "GPS Speed" if "GPS Speed" in aim_df.columns else None
    if speed_col is None or not lap_boundaries:
        half = len(lap_boundaries) // 2
        return list(range(half)), list(range(half, len(lap_boundaries)))

    # Find the lap whose minimum speed stays nearest zero the longest —
    # that's the lap the driver-change pit stop lands in.
    speeds = aim_df[speed_col].values
    best_lap, best_stopped = -1, 0
    for i, (s, e, _) in enumerate(lap_boundaries):
        stopped = int(np.sum(speeds[s:e] < 1.0))
        if stopped > best_stopped:
            best_stopped = stopped
            best_lap = i

    if best_lap < 0 or best_stopped < 10:
        half = len(lap_boundaries) // 2
        return list(range(half)), list(range(half, len(lap_boundaries)))

    pre = list(range(0, best_lap))
    post = list(range(best_lap + 1, len(lap_boundaries)))
    return pre, post


def compare_driver_stints(
    aim_df: pd.DataFrame,
    track: Track,
    *,
    driver1_laps: list[int] | None = None,
    driver2_laps: list[int] | None = None,
) -> pd.DataFrame:
    """Compare Driver 1 vs Driver 2 per-zone behavior (D-12 + D-21).

    Both drivers are zone-aligned through the shared track-segment
    decomposition: each driver's telemetry is extracted separately,
    collapsed to zones on the same track, and compared at the zone
    level.  Returns one row per Driver 1 zone so result rows are
    zone-indexed (D-21), not segment-indexed.

    Args:
        aim_df: Full AiM endurance telemetry.
        track: Track geometry.
        driver1_laps: Lap indices (0-based into detected boundaries)
            for driver 1. ``None`` (default) detects the driver-change
            lap via stationary pause and uses all laps before it.
        driver2_laps: Lap indices for driver 2. ``None`` uses all laps
            after the auto-detected driver-change lap.

    Returns:
        DataFrame with columns: zone_id, segment_start, segment_end,
        d1_action, d1_intensity, d2_action, d2_intensity, action_match,
        intensity_diff.
    """
    lap_boundaries = _detect_lap_boundaries_safe(aim_df)

    if driver1_laps is None or driver2_laps is None:
        auto_d1, auto_d2 = _auto_split_driver_laps(aim_df, lap_boundaries)
        if driver1_laps is None:
            driver1_laps = auto_d1
        if driver2_laps is None:
            driver2_laps = auto_d2

    # Extract per-segment actions for each driver's lap set separately.
    d1 = extract_per_segment_actions(aim_df, track, laps=driver1_laps)
    d2 = extract_per_segment_actions(aim_df, track, laps=driver2_laps)

    # Collapse to zones on the same track so both share the segment
    # decomposition used for zone-aligned comparison.
    zones_d1 = collapse_to_zones(d1, track)
    zones_d2 = collapse_to_zones(d2, track)

    # Zone-indexed output: use D1's zones as the anchor and look up the
    # D2 zone covering each D1 zone's midpoint.
    rows = []
    for z1 in zones_d1:
        mid_seg = (z1.segment_start + z1.segment_end) // 2
        z2 = next(
            (z for z in zones_d2 if z.segment_start <= mid_seg <= z.segment_end),
            None,
        )
        if z2 is None:
            continue
        rows.append({
            "zone_id": z1.zone_id,
            "segment_start": z1.segment_start,
            "segment_end": z1.segment_end,
            "d1_action": z1.action.value,
            "d1_intensity": z1.intensity,
            "d2_action": z2.action.value,
            "d2_intensity": z2.intensity,
            "action_match": z1.action == z2.action,
            "intensity_diff": abs(z1.intensity - z2.intensity),
        })

    return pd.DataFrame(rows)
