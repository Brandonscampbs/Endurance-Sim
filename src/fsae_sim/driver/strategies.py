"""Concrete driver strategy implementations.

- PedalProfileStrategy: per-segment pedal profile from telemetry (primary)
- ReplayStrategy: reproduce recorded telemetry behavior
- CoastOnlyStrategy: full throttle on straights, coast into corners
- ThresholdBrakingStrategy: coast + brake when speed exceeds corner limit
- CalibratedStrategy: zone-based model calibrated from telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class DriverParams:
    """Tunable driver behavior parameters for sweeps.

    All multipliers default to 1.0 (baseline = telemetry behavior).
    """

    throttle_scale: float = 1.0
    brake_scale: float = 1.0
    coast_throttle: float = 0.0
    max_throttle: float = 1.0
    max_brake: float = 1.0


class ReplayStrategy(DriverStrategy):
    """Replay recorded driver inputs from AiM telemetry through the force model.

    Provides the exact torque commands, throttle, and brake pressure the
    real driver applied, indexed by cumulative distance.  The engine
    resolves speed from forces — no speed-constrained shortcuts.

    Supports two modes:
    - **Single-lap**: wraps a one-lap recording for multi-lap sims (use
      ``from_aim_data`` with lap slice).
    - **Full-endurance**: uses the entire AiM recording indexed by
      cumulative distance (use ``from_full_endurance``).
    """

    name = "replay"

    def __init__(
        self,
        distances_m: np.ndarray,
        throttle_pct: np.ndarray,
        brake_pct: np.ndarray,
        torque_nm: np.ndarray,
        lap_distance_m: float,
        *,
        wrap: bool = True,
        electrical_power_w: np.ndarray | None = None,
    ) -> None:
        self._lap_distance_m = lap_distance_m
        self._total_distance_m = float(distances_m[-1])
        self._wrap = wrap
        self._has_power = electrical_power_w is not None
        if self._has_power:
            self._power_interp = interp1d(
                distances_m, electrical_power_w, kind="linear",
                bounds_error=False, fill_value=(0.0, 0.0),
            )
        self._throttle_interp = interp1d(
            distances_m, throttle_pct, kind="linear",
            bounds_error=False, fill_value=(float(throttle_pct[0]), float(throttle_pct[-1])),
        )
        self._brake_interp = interp1d(
            distances_m, brake_pct, kind="linear",
            bounds_error=False, fill_value=(float(brake_pct[0]), float(brake_pct[-1])),
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

        inverter_torque_limit = 85.0
        torque = np.clip(lap["LVCU Torque Req"].values, 0.0, inverter_torque_limit)

        return cls(dist, throttle, brake, torque, lap_distance_m, wrap=True)

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

        inverter_torque_limit = 85.0
        torque = np.clip(clean["LVCU Torque Req"].values, 0.0, inverter_torque_limit)

        mean_torque = float(np.mean(torque))

        # Measured electrical power from pack: V × I (W).
        # Positive = discharge, negative = regen/charging.
        # This is the most accurate energy measurement — direct from
        # the battery terminals, bypassing any torque sensor bias or
        # motor efficiency model uncertainty.
        elec_power = clean["Pack Voltage"].values * clean["Pack Current"].values

        # Extend interpolation data with a point beyond the last distance
        # so extrapolation uses reasonable values if sim distance exceeds
        # the AiM data range.
        max_dist = float(dist[-1])
        dist_ext = np.append(dist, max_dist + lap_distance_m)
        throttle_ext = np.append(throttle, 0.5)
        brake_ext = np.append(brake, 0.0)
        torque_ext = np.append(torque, mean_torque)
        power_ext = np.append(elec_power, 0.0)

        return cls(dist_ext, throttle_ext, brake_ext, torque_ext,
                   lap_distance_m, wrap=False, electrical_power_w=power_ext)

    def _resolve_distance(self, cumulative_distance_m: float) -> float:
        """Map cumulative sim distance to interpolation distance."""
        if self._wrap:
            return cumulative_distance_m % self._lap_distance_m
        return min(cumulative_distance_m, self._total_distance_m)

    def target_torque(self, distance_m: float) -> float:
        """Recorded motor torque (Nm) at given cumulative distance."""
        return float(self._torque_interp(self._resolve_distance(distance_m)))

    @property
    def has_electrical_power(self) -> bool:
        """True if this replay includes measured V×I power data."""
        return self._has_power

    def measured_electrical_power(self, distance_m: float) -> float:
        """Recorded Pack Voltage × Pack Current (W) at given distance.

        Returns the actual electrical power measured at the battery
        terminals.  Positive = discharging, negative = charging.
        Only available when constructed with ``electrical_power_w``.
        """
        if not self._has_power:
            return 0.0
        return float(self._power_interp(self._resolve_distance(distance_m)))

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
        segment_intensities: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self._zones = list(zones)
        self._num_segments = num_segments
        self._segment_intensities = segment_intensities

        # Build flat lookup: segment_idx -> (action, intensity, max_speed_ms)
        self._segment_actions: list[tuple[ControlAction, float, float]] = [
            (ControlAction.COAST, 0.0, 0.0)
        ] * num_segments
        for zone in zones:
            for seg_idx in range(zone.segment_start, zone.segment_end + 1):
                if 0 <= seg_idx < num_segments:
                    self._segment_actions[seg_idx] = (
                        zone.action, zone.intensity, zone.max_speed_ms,
                    )

        # Override with per-segment intensities when available.
        # Zones still define action type and structure; per-segment
        # intensities capture the real driver's fine-grained modulation.
        if segment_intensities is not None:
            for i in range(num_segments):
                action, _, max_speed = self._segment_actions[i]
                self._segment_actions[i] = (
                    action, float(segment_intensities[i]), max_speed,
                )

    @property
    def zones(self) -> list[DriverZone]:
        return list(self._zones)

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        idx = state.segment_idx % self._num_segments
        action, intensity, max_speed_ms = self._segment_actions[idx]

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
                    max_speed_ms=z.max_speed_ms,
                ))
            else:
                new_zones.append(z)
        # Carry per-segment intensities, update overridden zone's segments
        new_seg_intensities = None
        if self._segment_intensities is not None:
            new_seg_intensities = self._segment_intensities.copy()
            zone = next(z for z in self._zones if z.zone_id == zone_id)
            for seg_idx in range(zone.segment_start, zone.segment_end + 1):
                if 0 <= seg_idx < self._num_segments:
                    new_seg_intensities[seg_idx] = intensity
        return CalibratedStrategy(new_zones, self._num_segments, name=self.name,
                                  segment_intensities=new_seg_intensities)

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
        merge_tolerance: float = 0.15,
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

        # Extract per-segment intensities for fine-grained driver modulation
        segment_intensities = seg_actions["intensity"].values.copy()

        return cls(zones, track.num_segments, name=name,
                   segment_intensities=segment_intensities)

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


class PedalProfileStrategy(DriverStrategy):
    """Per-segment pedal-profile driver model.

    Stores raw throttle position and brake pressure per track segment,
    extracted from telemetry.  At runtime, outputs pedal values that
    the engine routes through ``lvcu_torque_command()`` — the same
    firmware chain the real car uses.

    For sweeps, ``DriverParams`` multipliers scale pedal inputs (driver
    behavior) while ``PowertrainConfig`` changes affect LVCU processing
    (car tune).  Both sweep independently.
    """

    name = "pedal_profile"

    def __init__(
        self,
        throttle_pct: np.ndarray,
        brake_pct: np.ndarray,
        actions: np.ndarray,
        ref_speed_ms: np.ndarray,
        num_segments: int,
        *,
        params: DriverParams | None = None,
    ) -> None:
        if not (len(throttle_pct) == len(brake_pct) == len(actions) == len(ref_speed_ms) == num_segments):
            raise ValueError(
                f"All arrays must have the same length as num_segments ({num_segments}), "
                f"got throttle={len(throttle_pct)}, brake={len(brake_pct)}, "
                f"actions={len(actions)}, ref_speed={len(ref_speed_ms)}"
            )
        self._throttle_pct = np.asarray(throttle_pct, dtype=np.float64)
        self._brake_pct = np.asarray(brake_pct, dtype=np.float64)
        self._actions = np.asarray(actions, dtype=np.int32)
        self._ref_speed_ms = np.asarray(ref_speed_ms, dtype=np.float64)
        self._num_segments = num_segments
        self.params = params or DriverParams()

    @property
    def num_segments(self) -> int:
        return self._num_segments

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        seg_idx = state.segment_idx % self._num_segments
        action_code = int(self._actions[seg_idx])

        if action_code == 1:  # THROTTLE
            throttle = float(self._throttle_pct[seg_idx]) * self.params.throttle_scale
            throttle = min(throttle, self.params.max_throttle)
            throttle = max(0.0, min(1.0, throttle))
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=throttle, brake_pct=0.0)

        elif action_code == 2:  # BRAKE
            brake = float(self._brake_pct[seg_idx]) * self.params.brake_scale
            brake = min(brake, self.params.max_brake)
            brake = max(0.0, min(1.0, brake))
            return ControlCommand(ControlAction.BRAKE, throttle_pct=0.0, brake_pct=brake)

        else:  # COAST (0)
            return ControlCommand(
                ControlAction.COAST,
                throttle_pct=max(0.0, min(1.0, self.params.coast_throttle)),
                brake_pct=0.0,
            )

    def with_params(self, **kwargs) -> PedalProfileStrategy:
        """Return a new strategy with modified DriverParams.

        Shares the underlying profile arrays (numpy views).
        Only the DriverParams are replaced.

        Args:
            **kwargs: Fields of DriverParams to override.

        Returns:
            New PedalProfileStrategy with updated params.
        """
        new_params = replace(self.params, **kwargs)
        return PedalProfileStrategy(
            throttle_pct=self._throttle_pct,
            brake_pct=self._brake_pct,
            actions=self._actions,
            ref_speed_ms=self._ref_speed_ms,
            num_segments=self._num_segments,
            params=new_params,
        )
