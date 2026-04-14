"""Concrete driver strategy implementations.

- ReplayStrategy: reproduce recorded telemetry behavior
- CoastOnlyStrategy: full throttle on straights, coast into corners
- ThresholdBrakingStrategy: coast + brake when speed exceeds corner limit
- CalibratedStrategy: zone-based model calibrated from telemetry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from fsae_sim.driver.strategy import (
    ControlAction,
    ControlCommand,
    DriverStrategy,
    SimState,
)
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.analysis.telemetry_analysis import (
    DriverZone,
    extract_per_segment_actions,
    collapse_to_zones,
)


class ReplayStrategy(DriverStrategy):
    """Replay recorded driver behavior from AiM telemetry.

    Supports two modes:
    - **Single-lap**: wraps a one-lap recording for multi-lap sims (use
      ``from_aim_data`` with lap slice).
    - **Full-endurance**: uses the entire AiM recording indexed by
      cumulative distance (use ``from_full_endurance``).

    The engine uses ``target_speed()`` to constrain speed directly and
    ``target_torque()`` for power calculation.
    """

    name = "replay"

    def __init__(
        self,
        distances_m: np.ndarray,
        throttle_pct: np.ndarray,
        brake_pct: np.ndarray,
        speeds_ms: np.ndarray,
        torque_nm: np.ndarray,
        lap_distance_m: float,
        *,
        wrap: bool = True,
    ) -> None:
        self._lap_distance_m = lap_distance_m
        self._total_distance_m = float(distances_m[-1])
        self._wrap = wrap
        self._throttle_interp = interp1d(
            distances_m, throttle_pct, kind="linear",
            bounds_error=False, fill_value=(float(throttle_pct[0]), float(throttle_pct[-1])),
        )
        self._brake_interp = interp1d(
            distances_m, brake_pct, kind="linear",
            bounds_error=False, fill_value=(float(brake_pct[0]), float(brake_pct[-1])),
        )
        self._speed_interp = interp1d(
            distances_m, speeds_ms, kind="linear",
            bounds_error=False, fill_value=(float(speeds_ms[0]), float(speeds_ms[-1])),
        )
        self._torque_interp = interp1d(
            distances_m, torque_nm, kind="linear",
            bounds_error=False, fill_value=(float(torque_nm[0]), float(torque_nm[-1])),
        )

    @classmethod
    def from_aim_data(
        cls,
        aim_df: "pd.DataFrame",
        lap_start_idx: int,
        lap_end_idx: int,
        lap_distance_m: float,
    ) -> ReplayStrategy:
        """Build single-lap replay from a slice of AiM DataFrame."""
        lap = aim_df.iloc[lap_start_idx:lap_end_idx].copy()
        dist = lap["Distance on GPS Speed"].values - lap["Distance on GPS Speed"].values[0]

        throttle = np.clip(lap["Throttle Pos"].values / 100.0, 0.0, 1.0)

        brake_raw = np.maximum(
            lap["FBrakePressure"].values, lap["RBrakePressure"].values,
        )
        bmax = max(np.percentile(brake_raw[brake_raw > 0], 99), 1.0) if np.any(brake_raw > 0) else 1.0
        brake = np.clip(brake_raw / bmax, 0.0, 1.0)

        speeds = lap["GPS Speed"].values / 3.6
        inverter_torque_limit = 85.0
        torque = np.clip(lap["LVCU Torque Req"].values, 0.0, inverter_torque_limit)

        return cls(dist, throttle, brake, speeds, torque, lap_distance_m, wrap=True)

    @classmethod
    def from_full_endurance(
        cls,
        aim_df: "pd.DataFrame",
        lap_distance_m: float,
        min_speed_kmh: float = 5.0,
    ) -> ReplayStrategy:
        """Build replay from the full AiM endurance recording.

        Filters out stopped periods (driver change, stalls) to produce
        a clean distance-indexed driving profile.  Indexes by cumulative
        distance rather than wrapping per-lap.
        """
        # Filter to moving samples to cut driver change and stopped periods
        moving = aim_df["GPS Speed"].values > min_speed_kmh
        clean = aim_df[moving].copy()

        dist = clean["Distance on GPS Speed"].values.copy()
        throttle = np.clip(clean["Throttle Pos"].values / 100.0, 0.0, 1.0)

        brake_raw = np.maximum(
            clean["FBrakePressure"].values, clean["RBrakePressure"].values,
        )
        bmax = max(np.percentile(brake_raw[brake_raw > 0], 99), 1.0) if np.any(brake_raw > 0) else 1.0
        brake = np.clip(brake_raw / bmax, 0.0, 1.0)

        speeds = clean["GPS Speed"].values / 3.6
        inverter_torque_limit = 85.0
        torque = np.clip(clean["LVCU Torque Req"].values, 0.0, inverter_torque_limit)

        # Use mean driving speed as fill value (not last point, which may
        # be very slow at end of event) so sim doesn't crawl if cumulative
        # distance exceeds the AiM data range.
        mean_speed = float(np.mean(speeds))
        mean_torque = float(np.mean(torque))

        # Extend interpolation data with a point beyond the last distance
        # at mean speed, so extrapolation uses a reasonable value
        max_dist = float(dist[-1])
        dist_ext = np.append(dist, max_dist + lap_distance_m)
        speeds_ext = np.append(speeds, mean_speed)
        throttle_ext = np.append(throttle, 0.5)
        brake_ext = np.append(brake, 0.0)
        torque_ext = np.append(torque, mean_torque)

        return cls(dist_ext, throttle_ext, brake_ext, speeds_ext, torque_ext,
                   lap_distance_m, wrap=False)

    def _resolve_distance(self, cumulative_distance_m: float) -> float:
        """Map cumulative sim distance to interpolation distance."""
        if self._wrap:
            return cumulative_distance_m % self._lap_distance_m
        return min(cumulative_distance_m, self._total_distance_m)

    def target_speed(self, distance_m: float) -> float:
        """Recorded speed (m/s) at given cumulative distance."""
        return float(self._speed_interp(self._resolve_distance(distance_m)))

    def target_torque(self, distance_m: float) -> float:
        """Recorded motor torque (Nm) at given cumulative distance."""
        return float(self._torque_interp(self._resolve_distance(distance_m)))

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        d = self._resolve_distance(state.distance)
        throttle = float(np.clip(self._throttle_interp(d), 0.0, 1.0))
        brake = float(np.clip(self._brake_interp(d), 0.0, 1.0))

        if brake > 0.05:
            return ControlCommand(ControlAction.BRAKE, throttle_pct=0.0, brake_pct=brake)
        elif throttle > 0.05:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=throttle, brake_pct=0.0)
        else:
            return ControlCommand(ControlAction.COAST, throttle_pct=0.0, brake_pct=0.0)


class CoastOnlyStrategy(DriverStrategy):
    """Full throttle on straights, coast into corners. No braking.

    This approximates the 2025 CT-16EV strategy where the team was
    instructed to never use brakes (coast-only approach).
    """

    name = "coast_only"

    def __init__(
        self,
        dynamics: VehicleDynamics,
        coast_margin_ms: float = 2.0,
    ) -> None:
        """
        Args:
            dynamics: Vehicle dynamics model for cornering speed limits.
            coast_margin_ms: Start coasting when speed is within this margin
                of the corner speed limit (m/s).
        """
        self._dynamics = dynamics
        self._coast_margin = coast_margin_ms

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        if not upcoming:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)

        # Find the minimum corner speed limit in upcoming segments
        min_corner_speed = float("inf")
        for seg in upcoming:
            v_max = self._dynamics.max_cornering_speed(seg.curvature, seg.grip_factor)
            min_corner_speed = min(min_corner_speed, v_max)

        if state.speed > min_corner_speed - self._coast_margin:
            return ControlCommand(ControlAction.COAST)
        else:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)


class ThresholdBrakingStrategy(DriverStrategy):
    """Coast + brake when speed exceeds corner limit.

    Adds active braking to the coast-only approach for tighter corners
    where coasting alone doesn't slow the car enough.
    """

    name = "threshold_braking"

    def __init__(
        self,
        dynamics: VehicleDynamics,
        coast_margin_ms: float = 3.0,
        brake_threshold_ms: float = 1.0,
        brake_intensity: float = 0.5,
    ) -> None:
        """
        Args:
            dynamics: Vehicle dynamics model.
            coast_margin_ms: Start coasting at this speed margin above corner limit.
            brake_threshold_ms: Apply brakes when speed exceeds corner limit by this much.
            brake_intensity: Brake pedal fraction (0-1) when braking.
        """
        self._dynamics = dynamics
        self._coast_margin = coast_margin_ms
        self._brake_threshold = brake_threshold_ms
        self._brake_intensity = brake_intensity

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        if not upcoming:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)

        min_corner_speed = float("inf")
        for seg in upcoming:
            v_max = self._dynamics.max_cornering_speed(seg.curvature, seg.grip_factor)
            min_corner_speed = min(min_corner_speed, v_max)

        if state.speed > min_corner_speed + self._brake_threshold:
            return ControlCommand(
                ControlAction.BRAKE,
                brake_pct=self._brake_intensity,
            )
        elif state.speed > min_corner_speed - self._coast_margin:
            return ControlCommand(ControlAction.COAST)
        else:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)


class CalibratedStrategy(DriverStrategy):
    """Zone-based driver model calibrated from telemetry.

    Collapses per-segment telemetry data into coachable zones, each with
    an action (throttle/coast/brake) and intensity. The decide() method
    looks up the zone for the current segment and returns the corresponding
    command. The sim engine resolves physics from force balance.

    Construction paths:
    - Direct: pass zones and num_segments
    - from_telemetry(): calibrate from AiM telemetry DataFrame
    - from_zone_list(): build from manual zone definitions
    """

    name = "calibrated"

    def __init__(
        self,
        zones: list[DriverZone],
        num_segments: int,
        name: str = "calibrated",
    ) -> None:
        self.name = name
        self._zones = list(zones)
        self._num_segments = num_segments

        # Build flat lookup: segment_idx -> (action, intensity)
        self._segment_actions: list[tuple[ControlAction, float]] = [
            (ControlAction.COAST, 0.0)
        ] * num_segments
        for zone in zones:
            for seg_idx in range(zone.segment_start, zone.segment_end + 1):
                if 0 <= seg_idx < num_segments:
                    self._segment_actions[seg_idx] = (zone.action, zone.intensity)

    @property
    def zones(self) -> list[DriverZone]:
        return list(self._zones)

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        idx = state.segment_idx % self._num_segments
        action, intensity = self._segment_actions[idx]

        if action == ControlAction.THROTTLE:
            return ControlCommand(action, throttle_pct=intensity, brake_pct=0.0)
        elif action == ControlAction.BRAKE:
            return ControlCommand(action, throttle_pct=0.0, brake_pct=intensity)
        else:
            return ControlCommand(ControlAction.COAST, throttle_pct=0.0, brake_pct=0.0)

    def zone_for_segment(self, segment_idx: int) -> DriverZone:
        """Return the zone containing the given segment index."""
        idx = segment_idx % self._num_segments
        for zone in self._zones:
            if zone.segment_start <= idx <= zone.segment_end:
                return zone
        raise ValueError(f"No zone found for segment {idx}")

    def to_dataframe(self) -> pd.DataFrame:
        """Export zone table as a DataFrame."""
        rows = []
        for z in self._zones:
            rows.append({
                "zone_id": z.zone_id,
                "segment_start": z.segment_start,
                "segment_end": z.segment_end,
                "action": z.action.value,
                "intensity": z.intensity,
                "distance_start_m": z.distance_start_m,
                "distance_end_m": z.distance_end_m,
                "label": z.label,
            })
        return pd.DataFrame(rows)

    def to_driver_brief(self) -> str:
        """Format zone table as human-readable driver coaching text."""
        lines = ["Driver Strategy Brief", "=" * 40]
        for z in self._zones:
            action_str = z.action.value.upper()
            if z.action == ControlAction.THROTTLE:
                detail = f"{z.intensity * 100:.0f}% throttle"
            elif z.action == ControlAction.BRAKE:
                detail = f"{z.intensity * 100:.0f}% brake"
            else:
                detail = "coast"
            dist = f"{z.distance_start_m:.0f}-{z.distance_end_m:.0f}m"
            lines.append(f"  Zone {z.zone_id}: {z.label} ({dist}) — {detail}")
        return "\n".join(lines)

    def with_zone_override(
        self,
        zone_id: int,
        action: ControlAction,
        intensity: float,
    ) -> CalibratedStrategy:
        """Return a new strategy with one zone's action/intensity changed."""
        new_zones = []
        for z in self._zones:
            if z.zone_id == zone_id:
                new_zones.append(DriverZone(
                    zone_id=z.zone_id,
                    segment_start=z.segment_start,
                    segment_end=z.segment_end,
                    action=action,
                    intensity=intensity,
                    distance_start_m=z.distance_start_m,
                    distance_end_m=z.distance_end_m,
                    label=z.label,
                ))
            else:
                new_zones.append(z)
        return CalibratedStrategy(new_zones, self._num_segments, name=self.name)

    @classmethod
    def from_telemetry(
        cls,
        aim_df: pd.DataFrame,
        track: Track,
        *,
        laps: list[int] | None = None,
        throttle_col: str = "Throttle Pos",
        front_brake_col: str = "FBrakePressure",
        rear_brake_col: str = "RBrakePressure",
        speed_col: str = "GPS Speed",
        distance_col: str = "Distance on GPS Speed",
        throttle_threshold: float = 5.0,
        brake_threshold: float = 2.0,
        merge_tolerance: float = 0.05,
        name: str = "calibrated",
    ) -> CalibratedStrategy:
        """Calibrate from AiM telemetry.

        Samples telemetry at each track segment midpoint, classifies
        actions, and collapses into coachable zones.
        """
        seg_actions = extract_per_segment_actions(
            aim_df, track,
            laps=laps,
            throttle_threshold=throttle_threshold,
            brake_threshold=brake_threshold,
        )
        zones = collapse_to_zones(seg_actions, track, merge_tolerance=merge_tolerance)
        return cls(zones, track.num_segments, name=name)

    @classmethod
    def from_zone_list(
        cls,
        zones: list[dict],
        track: Track,
        name: str = "calibrated",
    ) -> CalibratedStrategy:
        """Build from manual zone definitions.

        Args:
            zones: List of dicts with keys: segments (start, end),
                action ("throttle"/"coast"/"brake"), intensity (0-1), label.
            track: Track geometry.
            name: Strategy name.
        """
        action_map = {
            "throttle": ControlAction.THROTTLE,
            "coast": ControlAction.COAST,
            "brake": ControlAction.BRAKE,
        }
        driver_zones = []
        for i, z in enumerate(zones):
            seg_start, seg_end = z["segments"]
            action = action_map[z["action"]]
            intensity = z.get("intensity", 0.0)
            label = z.get("label", f"Zone {i}")

            dist_start = track.segments[seg_start].distance_start_m
            dist_end = (
                track.segments[seg_end].distance_start_m
                + track.segments[seg_end].length_m
            )
            driver_zones.append(DriverZone(
                zone_id=i,
                segment_start=seg_start,
                segment_end=seg_end,
                action=action,
                intensity=intensity,
                distance_start_m=dist_start,
                distance_end_m=dist_end,
                label=label,
            ))
        return cls(driver_zones, track.num_segments, name=name)
