"""Concrete driver strategy implementations.

- ReplayStrategy: reproduce recorded telemetry behavior
- CoastOnlyStrategy: full throttle on straights, coast into corners
- ThresholdBrakingStrategy: coast + brake when speed exceeds corner limit
- CalibratedStrategy: zone-based model calibrated from telemetry
- PreviewDriverStrategy: physics-envelope preview speed tracker
"""

from __future__ import annotations

import math
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
    _detect_lap_boundaries_safe,
)


# D-24: module-level action thresholds for ReplayStrategy.decide.
# Post-normalization these line up with the calibration pipeline's
# classifier in extract_per_segment_actions / PedalProfileStrategy:
#   throttle: 5% pedal (Throttle Pos / 100 ≥ 0.05)
#   brake:    2 bar line pressure (brake_raw / brake_max_pressure_bar
#             with brake_max_pressure_bar=60 → ~0.033 fraction, but
#             Replay normalizes by the recorded 99th percentile
#             — see ReplayStrategy.from_aim_data — so a 5% brake
#             fraction there corresponds to roughly the same "driver
#             is actively braking" regime.)
# Same constants are intentionally reused so a sim run and a
# calibration pass over the same telemetry agree on what counts
# as throttle vs brake vs coast.
_THROTTLE_ACTION_THRESHOLD = 0.05
_BRAKE_ACTION_THRESHOLD = 0.05
_BRAKE_ACTION_THRESHOLD_BAR = 2.0


def _telemetry_speed_col(df: pd.DataFrame) -> str:
    """Prefer the wheel-speed truth channel in cleaned telemetry."""
    if "LFspeed" in df.columns:
        return "LFspeed"
    return "GPS Speed"


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
        brake_action_threshold: float | None = None,
        torque_is_delivered: bool = False,
        electrical_power_w: np.ndarray | None = None,
    ) -> None:
        self._lap_distance_m = lap_distance_m
        self._total_distance_m = float(distances_m[-1])
        self._wrap = wrap
        self._torque_is_delivered = bool(torque_is_delivered)
        self._brake_action_threshold = (
            float(brake_action_threshold)
            if brake_action_threshold is not None
            else _BRAKE_ACTION_THRESHOLD
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
        self._power_interp = None
        if electrical_power_w is not None:
            self._power_interp = interp1d(
                distances_m, electrical_power_w, kind="linear",
                bounds_error=False,
                fill_value=(float(electrical_power_w[0]), float(electrical_power_w[-1])),
            )

    @property
    def torque_is_delivered(self) -> bool:
        return self._torque_is_delivered

    @property
    def has_electrical_power(self) -> bool:
        return self._power_interp is not None

    @staticmethod
    def _clean_torque_channel(values: np.ndarray, limit_nm: float) -> np.ndarray:
        """Interpolate finite torque samples and clip CAN dropouts."""
        s = pd.Series(np.asarray(values, dtype=float))
        s = s.where(np.isfinite(s) & (s.abs() <= limit_nm * 2.0))
        s = s.interpolate(limit_direction="both").fillna(0.0)
        return np.clip(s.to_numpy(), -limit_nm, limit_nm)

    @classmethod
    def from_aim_data(
        cls,
        aim_df: "pd.DataFrame",
        lap_start_idx: int,
        lap_end_idx: int,
        lap_distance_m: float,
        *,
        prefer_torque_feedback: bool = False,
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
        brake_action_threshold = min(
            1.0, _BRAKE_ACTION_THRESHOLD_BAR / bmax,
        )

        # S18: preserve regen. Previously clipped to [0, +limit] which
        # silently deleted negative torque commands (coast-regen). Keep
        # symmetric [-limit, +limit] so replay drives the true recorded
        # electrical profile.
        inverter_torque_limit = 85.0
        torque_is_delivered = (
            prefer_torque_feedback and "Torque Feedback" in lap.columns
        )
        if torque_is_delivered:
            torque = cls._clean_torque_channel(
                lap["Torque Feedback"].values,
                inverter_torque_limit,
            )
        else:
            torque = np.clip(
                lap["LVCU Torque Req"].values,
                -inverter_torque_limit, inverter_torque_limit,
            )

        electrical_power = None
        if {"Pack Voltage", "Pack Current"}.issubset(lap.columns):
            electrical_power = (
                lap["Pack Voltage"].values * lap["Pack Current"].values
            )

        return cls(
            dist, throttle, brake, torque, lap_distance_m, wrap=True,
            brake_action_threshold=brake_action_threshold,
            torque_is_delivered=torque_is_delivered,
            electrical_power_w=electrical_power,
        )

    @classmethod
    def from_full_endurance(
        cls,
        aim_df: "pd.DataFrame",
        lap_distance_m: float,
        min_speed_kmh: float = 5.0,
        *,
        prefer_torque_feedback: bool = False,
        trim_to_lap_start: bool = True,
    ) -> ReplayStrategy:
        """Build replay from the full AiM endurance recording.

        Filters out stopped periods (driver change, stalls) to produce
        a clean distance-indexed driving profile.  Indexes by cumulative
        distance rather than wrapping per-lap.

        Distance alignment: the AiM "Distance on GPS Speed" channel
        accumulates from the first moving sample of the recording, so
        lap 1 may start at a non-zero global distance (warm-up / pit-out
        precedes the start/finish gate). Sim cumulative distance starts
        at 0 at the start/finish line. We trim the recording to begin at
        the first detected start/finish crossing (lap 1 entry) and
        re-zero distances so sim and replay share the same origin.
        Without this trim, sim distance 200 m maps to pit-out throttle,
        not lap-1 driver inputs.
        """
        if trim_to_lap_start:
            # Trim to lap 1 start so the distance origin matches the sim's.
            try:
                from fsae_sim.analysis.validation import detect_lap_boundaries
                laps = detect_lap_boundaries(aim_df)
            except Exception:
                laps = []
            if laps:
                start_row = laps[0][0]
                aim_df = aim_df.iloc[start_row:].copy()

        # Filter to moving samples to cut driver change and stopped periods
        moving = aim_df[_telemetry_speed_col(aim_df)].values > min_speed_kmh
        clean = aim_df[moving].copy()

        dist = clean["Distance on GPS Speed"].values.copy()
        # Re-zero so sim distance 0 corresponds to the first telem sample
        # we are now using (lap 1 start).
        if len(dist):
            dist = dist - dist[0]
        throttle = np.clip(clean["Throttle Pos"].values / 100.0, 0.0, 1.0)

        brake_raw = np.maximum(
            clean["FBrakePressure"].values, clean["RBrakePressure"].values,
        )
        bmax = max(np.percentile(brake_raw[brake_raw > 0], 99), 1.0) if np.any(brake_raw > 0) else 1.0
        brake = np.clip(brake_raw / bmax, 0.0, 1.0)
        brake_action_threshold = min(
            1.0, _BRAKE_ACTION_THRESHOLD_BAR / bmax,
        )

        # S18: preserve regen (see from_aim_data).
        inverter_torque_limit = 85.0
        torque_is_delivered = (
            prefer_torque_feedback and "Torque Feedback" in clean.columns
        )
        if torque_is_delivered:
            torque = cls._clean_torque_channel(
                clean["Torque Feedback"].values,
                inverter_torque_limit,
            )
        else:
            torque = np.clip(
                clean["LVCU Torque Req"].values,
                -inverter_torque_limit, inverter_torque_limit,
            )

        electrical_power = None
        if {"Pack Voltage", "Pack Current"}.issubset(clean.columns):
            electrical_power = (
                clean["Pack Voltage"].values * clean["Pack Current"].values
            )

        order = np.argsort(dist)
        dist = dist[order]
        throttle = throttle[order]
        brake = brake[order]
        torque = torque[order]
        if electrical_power is not None:
            electrical_power = electrical_power[order]
        keep = np.concatenate(([True], np.diff(dist) > 1e-6))
        dist = dist[keep]
        throttle = throttle[keep]
        brake = brake[keep]
        torque = torque[keep]
        if electrical_power is not None:
            electrical_power = electrical_power[keep]

        mean_torque = float(np.mean(torque))
        mean_power = (
            float(np.mean(electrical_power))
            if electrical_power is not None else None
        )

        # Extend interpolation data with a point beyond the last distance
        # so extrapolation uses reasonable values if sim distance exceeds
        # the AiM data range.
        max_dist = float(dist[-1])
        dist_ext = np.append(dist, max_dist + lap_distance_m)
        throttle_ext = np.append(throttle, 0.5)
        brake_ext = np.append(brake, 0.0)
        torque_ext = np.append(torque, mean_torque)
        power_ext = (
            np.append(electrical_power, mean_power)
            if electrical_power is not None else None
        )

        return cls(
            dist_ext, throttle_ext, brake_ext, torque_ext,
            lap_distance_m, wrap=False,
            brake_action_threshold=brake_action_threshold,
            torque_is_delivered=torque_is_delivered,
            electrical_power_w=power_ext,
        )

    def _resolve_distance(self, cumulative_distance_m: float) -> float:
        """Map cumulative sim distance to interpolation distance."""
        if self._wrap:
            return cumulative_distance_m % self._lap_distance_m
        return min(cumulative_distance_m, self._total_distance_m)

    def target_torque(self, distance_m: float) -> float:
        """Recorded motor torque (Nm) at given cumulative distance."""
        return float(self._torque_interp(self._resolve_distance(distance_m)))

    def measured_electrical_power(self, distance_m: float) -> float:
        """Recorded pack electrical power (W) at given cumulative distance."""
        if self._power_interp is None:
            raise ValueError("ReplayStrategy has no electrical power channel")
        return float(self._power_interp(self._resolve_distance(distance_m)))

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        d = self._resolve_distance(state.distance)
        throttle = float(np.clip(self._throttle_interp(d), 0.0, 1.0))
        brake = float(np.clip(self._brake_interp(d), 0.0, 1.0))

        # Replay is a measured-input mode: motor torque and hydraulic
        # brake pressure are independent physical channels.  Preserve the
        # brake fraction even when the action label is THROTTLE so the
        # engine can model trail-braking/overlap instead of deleting one
        # measured channel.
        if brake > self._brake_action_threshold:
            if throttle > _THROTTLE_ACTION_THRESHOLD:
                return ControlCommand(
                    ControlAction.THROTTLE,
                    throttle_pct=throttle,
                    brake_pct=brake,
                )
            return ControlCommand(
                ControlAction.BRAKE,
                throttle_pct=0.0,
                brake_pct=brake,
            )
        elif throttle > _THROTTLE_ACTION_THRESHOLD:
            return ControlCommand(
                ControlAction.THROTTLE,
                throttle_pct=throttle,
                brake_pct=brake,
            )
        else:
            return ControlCommand(
                ControlAction.COAST,
                throttle_pct=0.0,
                brake_pct=brake,
            )


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
        envelope: "SpeedEnvelope | None" = None,
    ) -> None:
        """
        Args:
            dynamics: Vehicle dynamics model for cornering speed limits.
            coast_margin_ms: Start coasting when speed is within this margin
                of the corner speed limit (m/s).
            envelope: D-20. Optional pre-computed forward-backward speed
                envelope. When provided, ``decide`` consults the
                per-segment ceiling for upcoming segments instead of the
                isolated ``dynamics.max_cornering_speed``, so the driver
                looks ahead past single-corner grip (matters for
                back-to-back tight corners where a downstream corner
                forces earlier lifting).
        """
        self._dynamics = dynamics
        self._coast_margin = coast_margin_ms
        self._envelope: np.ndarray | None = None
        if envelope is not None:
            self.set_envelope(envelope)

    def set_envelope(self, envelope: "SpeedEnvelope | np.ndarray") -> None:
        """Attach (or replace) a pre-computed speed envelope.

        Accepts either a SpeedEnvelope instance (we call ``compute()``)
        or a pre-computed NumPy array of per-segment ceilings.
        """
        if hasattr(envelope, "compute"):
            self._envelope = np.asarray(envelope.compute())
        else:
            self._envelope = np.asarray(envelope)

    def _min_upcoming_limit(self, state: SimState, upcoming: list[Segment]) -> float:
        if self._envelope is not None:
            n = len(self._envelope)
            min_v = float("inf")
            for i, _ in enumerate(upcoming):
                idx = (state.segment_idx + i) % n
                min_v = min(min_v, float(self._envelope[idx]))
            return min_v
        # Fallback: per-corner max from dynamics.
        min_v = float("inf")
        for seg in upcoming:
            v_max = self._dynamics.max_cornering_speed(seg.curvature, seg.grip_factor)
            min_v = min(min_v, v_max)
        return min_v

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        if not upcoming:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)

        min_corner_speed = self._min_upcoming_limit(state, upcoming)

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
        envelope: "SpeedEnvelope | None" = None,
    ) -> None:
        """
        Args:
            dynamics: Vehicle dynamics model.
            coast_margin_ms: Start coasting at this speed margin above corner limit.
            brake_threshold_ms: Apply brakes when speed exceeds corner limit by this much.
            brake_intensity: Brake pedal fraction (0-1) when braking.
            envelope: D-20. Optional pre-computed speed envelope; see
                ``CoastOnlyStrategy`` for rationale.
        """
        self._dynamics = dynamics
        self._coast_margin = coast_margin_ms
        self._brake_threshold = brake_threshold_ms
        self._brake_intensity = brake_intensity
        self._envelope: np.ndarray | None = None
        if envelope is not None:
            self.set_envelope(envelope)

    def set_envelope(self, envelope: "SpeedEnvelope | np.ndarray") -> None:
        if hasattr(envelope, "compute"):
            self._envelope = np.asarray(envelope.compute())
        else:
            self._envelope = np.asarray(envelope)

    def _min_upcoming_limit(self, state: SimState, upcoming: list[Segment]) -> float:
        if self._envelope is not None:
            n = len(self._envelope)
            min_v = float("inf")
            for i, _ in enumerate(upcoming):
                idx = (state.segment_idx + i) % n
                min_v = min(min_v, float(self._envelope[idx]))
            return min_v
        min_v = float("inf")
        for seg in upcoming:
            v_max = self._dynamics.max_cornering_speed(seg.curvature, seg.grip_factor)
            min_v = min(min_v, v_max)
        return min_v

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        if not upcoming:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)

        min_corner_speed = self._min_upcoming_limit(state, upcoming)

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
        params: "DriverParams | None" = None,
        use_observed_speed_caps: bool = True,
    ) -> None:
        self.name = name
        self._zones = list(zones)
        self._num_segments = num_segments
        self._use_observed_speed_caps = bool(use_observed_speed_caps)
        # D-28: shared DriverParams surface with PedalProfileStrategy.
        # Defaults to an identity (all 1.0) so existing behavior is
        # unchanged. Applied in ``decide`` to zone intensity.
        self._params: DriverParams | None = params

        # D-02: per-segment intensity override has been removed entirely.
        # Zones are the unit of control: `decide()`, `to_driver_brief()`,
        # and `to_dataframe()` all read `zone.intensity` exclusively. The
        # previous two-pass init (write zone intensity, then overwrite
        # from `segment_intensities`) created a three-way inconsistency
        # between runtime behavior, exported dataframe, and driver brief.

        # Build flat lookup: segment_idx -> (action, intensity, max_speed_ms)
        self._segment_actions: list[tuple[ControlAction, float, float]] = [
            (ControlAction.COAST, 0.0, 0.0)
        ] * num_segments
        # D-25: O(1) segment→zone index map.  -1 means "no zone covers
        # this segment" (fallback behavior below matches the old linear
        # scan's ValueError path at the call site).
        self._segment_to_zone: np.ndarray = np.full(num_segments, -1, dtype=np.int32)
        for z_pos, zone in enumerate(zones):
            for seg_idx in range(zone.segment_start, zone.segment_end + 1):
                if 0 <= seg_idx < num_segments:
                    self._segment_actions[seg_idx] = (
                        zone.action, zone.intensity, zone.max_speed_ms,
                    )
                    self._segment_to_zone[seg_idx] = z_pos

    @property
    def zones(self) -> list[DriverZone]:
        return list(self._zones)

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        idx = state.segment_idx % self._num_segments
        action, intensity, max_speed_ms = self._segment_actions[idx]

        # D-09: pass the zone's observed peak speed through metadata so
        # the engine can cap exit_speed. Only populated when a real cap
        # is available (>0) to leave straight-line zones unconstrained.
        meta: dict | None = None
        if self._use_observed_speed_caps and max_speed_ms > 0.0:
            meta = {"max_speed_ms": float(max_speed_ms)}

        # D-28: apply optional DriverParams scale/cap.  Multiplier
        # applies to intensity, cap clamps the post-multiplier value.
        p = self._params
        if action == ControlAction.THROTTLE:
            value = intensity
            if p is not None:
                value = min(value * p.throttle_scale, p.max_throttle)
            value = max(0.0, min(1.0, value))
            return ControlCommand(
                action, throttle_pct=value, brake_pct=0.0, metadata=meta,
            )
        elif action == ControlAction.BRAKE:
            value = intensity
            if p is not None:
                value = min(value * p.brake_scale, p.max_brake)
            value = max(0.0, min(1.0, value))
            return ControlCommand(
                action, throttle_pct=0.0, brake_pct=value, metadata=meta,
            )
        else:
            return ControlCommand(
                ControlAction.COAST, throttle_pct=0.0, brake_pct=0.0, metadata=meta,
            )

    def zone_for_segment(self, segment_idx: int) -> DriverZone:
        """Return the zone containing the given segment index.

        D-25: O(1) lookup via pre-computed ``_segment_to_zone`` index
        array.  Behavior matches the prior linear scan: returns the
        first zone covering ``idx``, or raises ``ValueError`` if no
        zone covers it.
        """
        idx = segment_idx % self._num_segments
        z_pos = int(self._segment_to_zone[idx])
        if z_pos < 0:
            raise ValueError(f"No zone found for segment {idx}")
        return self._zones[z_pos]

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
                "max_speed_ms": z.max_speed_ms,
                "max_speed_source": z.max_speed_source,
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
                    max_speed_source=z.max_speed_source,
                ))
            else:
                new_zones.append(z)
        return CalibratedStrategy(
            new_zones,
            self._num_segments,
            name=self.name,
            params=self._params,
            use_observed_speed_caps=self._use_observed_speed_caps,
        )

    def with_params(self, params: "DriverParams") -> CalibratedStrategy:
        """D-28: return a new strategy with ``DriverParams`` applied at decide.

        Shares the underlying zone list — only the params attribute is
        replaced.  Multiplier/cap semantics match ``PedalProfileStrategy``
        so both strategies now expose a single sweep surface.
        """
        return CalibratedStrategy(
            self._zones,
            self._num_segments,
            name=self.name,
            params=params,
            use_observed_speed_caps=self._use_observed_speed_caps,
        )

    @property
    def params(self) -> "DriverParams | None":
        return self._params

    @property
    def uses_observed_speed_caps(self) -> bool:
        return self._use_observed_speed_caps and any(
            z.max_speed_ms > 0.0 and z.max_speed_source != "none"
            for z in self._zones
        )

    def without_observed_speed_caps(self) -> CalibratedStrategy:
        """Return a copy that ignores telemetry-derived speed caps."""
        return CalibratedStrategy(
            self._zones,
            self._num_segments,
            name=self.name,
            params=self._params,
            use_observed_speed_caps=False,
        )

    @classmethod
    def from_telemetry(
        cls,
        aim_df: pd.DataFrame,
        track: Track,
        *,
        laps: list[int] | None = None,
        holdout_laps: list[int] | None = None,
        throttle_col: str = "Throttle Pos",
        front_brake_col: str = "FBrakePressure",
        rear_brake_col: str = "RBrakePressure",
        speed_col: str | None = None,
        distance_col: str = "Distance on GPS Speed",
        throttle_threshold: float = 5.0,
        brake_threshold: float = 2.0,
        brake_max_pressure_bar: float | None = None,
        merge_tolerance: float = 0.15,
        name: str = "calibrated",
        use_observed_speed_caps: bool = True,
    ) -> CalibratedStrategy:
        """Calibrate from AiM telemetry.

        Samples telemetry at each track segment midpoint, classifies
        actions, and collapses into coachable zones.

        Args:
            laps: explicit lap indices (0-based) to calibrate on. None =
                auto-select non-outlier laps.
            holdout_laps: S9 — lap indices to EXCLUDE from calibration so
                they can be used as an unseen validation set. Applied
                after `laps` selection (or auto-selection). Prevents the
                same laps being used to both fit and validate.
        """
        effective_laps = laps
        if holdout_laps is not None:
            holdout_set = set(holdout_laps)
            # Resolve the candidate lap set: if the caller didn't pick
            # explicit laps, subtract holdouts from "all laps" so the
            # auto-selector inside `extract_per_segment_actions` sees a
            # filtered list. We pass positive indices; the helper accepts
            # a concrete `laps` list.
            if laps is not None:
                effective_laps = [i for i in laps if i not in holdout_set]
            else:
                # Determine lap count to build the complement.
                boundaries = _detect_lap_boundaries_safe(aim_df)
                if boundaries:
                    effective_laps = [
                        i for i in range(len(boundaries)) if i not in holdout_set
                    ]
            if effective_laps is not None and len(effective_laps) == 0:
                raise ValueError(
                    "holdout_laps removed every candidate lap; "
                    "nothing left to calibrate on."
                )

        # D-19: plumb column-name kwargs through to the extractor.
        # Previously accepted but dropped on the floor.
        resolved_speed_col = speed_col or _telemetry_speed_col(aim_df)
        seg_actions = extract_per_segment_actions(
            aim_df, track,
            laps=effective_laps,
            throttle_threshold=throttle_threshold,
            brake_threshold=brake_threshold,
            brake_max_pressure_bar=brake_max_pressure_bar,
            throttle_col=throttle_col,
            front_brake_col=front_brake_col,
            rear_brake_col=rear_brake_col,
            speed_col=resolved_speed_col,
            distance_col=distance_col,
        )
        zones = collapse_to_zones(seg_actions, track, merge_tolerance=merge_tolerance)

        # D-02: no per-segment intensity passed — zone intensity is the
        # single source of runtime truth.
        return cls(
            zones,
            track.num_segments,
            name=name,
            use_observed_speed_caps=use_observed_speed_caps,
        )

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


@dataclass(frozen=True)
class DriverParams:
    """Tunable driver behavior parameters for sweeps.

    All multipliers default to 1.0 (baseline = telemetry behavior).
    """

    throttle_scale: float = 1.0
    brake_scale: float = 1.0
    max_throttle: float = 1.0
    max_brake: float = 1.0


@dataclass(frozen=True)
class PreviewDriverParams:
    """Tunable parameters for the preview speed-tracking driver.

    This is a compact approximation of the preview-control / minimum-lap-time
    literature: the vehicle model supplies a feasible speed envelope, the
    driver sees a finite preview window, and throttle/brake commands track a
    target fraction of that envelope.
    """

    target_speed_ratio: float = 0.94
    preview_time_s: float = 1.2
    min_preview_segments: int = 2
    max_preview_segments: int = 16
    throttle_gain: float = 2.4
    brake_gain: float = 1.2
    throttle_margin_ms: float = 0.4
    brake_margin_ms: float = 0.3
    max_throttle: float = 1.0
    max_brake: float = 1.0


class PreviewDriverStrategy(DriverStrategy):
    """Predictive driver model using finite preview and physics speed targets.

    Unlike ``CalibratedStrategy``, this model does not store per-segment
    observed speed caps.  Telemetry calibration only estimates global driver
    parameters; runtime braking/throttle decisions come from the track,
    speed envelope, and current state.
    """

    name = "preview_driver"
    uses_observed_speed_caps = False

    def __init__(
        self,
        track: Track,
        dynamics: VehicleDynamics | None = None,
        *,
        params: PreviewDriverParams | None = None,
        name: str = "preview_driver",
    ) -> None:
        self.name = name
        self._track = track
        self._dynamics = dynamics
        self.params = params or PreviewDriverParams()
        self._num_segments = track.num_segments
        self._envelope_ms: np.ndarray | None = None
        self._target_ms: np.ndarray | None = None
        self._preview_target_cache: np.ndarray | None = None
        self._preview_distance_cache: np.ndarray | None = None

    def set_envelope(self, envelope_ms: np.ndarray) -> None:
        """Receive the engine's forward/backward speed envelope."""
        env = np.asarray(envelope_ms, dtype=np.float64)
        if len(env) != self._num_segments:
            raise ValueError(
                f"Envelope length {len(env)} does not match track segments "
                f"{self._num_segments}."
            )
        ratio = float(np.clip(self.params.target_speed_ratio, 0.2, 1.0))
        self._envelope_ms = env.copy()
        self._target_ms = env * ratio
        self._build_preview_cache()

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        idx = state.segment_idx % self._num_segments
        target_ms, preview_distance_m = self._preview_target(idx, state.speed)
        speed = max(float(state.speed), 0.0)

        if math_isfinite(target_ms) and speed > target_ms + self.params.brake_margin_ms:
            max_decel = self._max_brake_decel(speed)
            desired_decel = (
                (speed * speed - target_ms * target_ms)
                / (2.0 * max(preview_distance_m, 1.0))
            )
            brake = self.params.brake_gain * desired_decel / max(max_decel, 1e-6)
            brake = max(0.0, min(float(self.params.max_brake), float(brake)))
            if brake > 1e-3:
                return ControlCommand(
                    ControlAction.BRAKE,
                    throttle_pct=0.0,
                    brake_pct=brake,
                )

        if not math_isfinite(target_ms):
            return ControlCommand(
                ControlAction.THROTTLE,
                throttle_pct=max(0.0, min(1.0, float(self.params.max_throttle))),
                brake_pct=0.0,
            )

        if speed < target_ms - self.params.throttle_margin_ms:
            err = (target_ms - speed) / max(target_ms, 1.0)
            throttle = self.params.throttle_gain * err
            throttle = max(0.0, min(float(self.params.max_throttle), float(throttle)))
            if throttle > 1e-3:
                return ControlCommand(
                    ControlAction.THROTTLE,
                    throttle_pct=throttle,
                    brake_pct=0.0,
                )

        return ControlCommand(ControlAction.COAST, throttle_pct=0.0, brake_pct=0.0)

    def with_params(self, **kwargs) -> PreviewDriverStrategy:
        return PreviewDriverStrategy(
            self._track,
            self._dynamics,
            params=replace(self.params, **kwargs),
            name=self.name,
        )

    @classmethod
    def from_telemetry(
        cls,
        aim_df: pd.DataFrame,
        track: Track,
        dynamics: VehicleDynamics | None = None,
        *,
        laps: list[int] | None = None,
        brake_max_pressure_bar: float = 60.0,
        name: str = "preview_driver",
    ) -> PreviewDriverStrategy:
        """Fit global preview-driver parameters from telemetry.

        The fit deliberately avoids per-segment speed caps.  It estimates:
        - target speed ratio from observed corner speed vs physics corner limit
        - max throttle from high-percentile throttle position
        - max brake from high-percentile brake pressure
        """
        speed_col = _telemetry_speed_col(aim_df)
        selected = cls._selected_lap_slices(aim_df, laps)
        speed_ratios: list[float] = []

        if dynamics is not None:
            for start_idx, end_idx, _ in selected:
                lap_df = aim_df.iloc[start_idx:end_idx]
                lap_dist_raw = lap_df["Distance on GPS Speed"].values
                if len(lap_dist_raw) < 2:
                    continue
                lap_span = float(lap_dist_raw[-1] - lap_dist_raw[0])
                if lap_span <= 0.0:
                    continue
                lap_d = (lap_dist_raw - lap_dist_raw[0]) * (
                    track.total_distance_m / lap_span
                )
                lap_speed_ms = lap_df[speed_col].values / 3.6
                for seg in track.segments:
                    if abs(seg.curvature) < 1e-6:
                        continue
                    limit = dynamics.max_cornering_speed(
                        seg.curvature,
                        seg.grip_factor,
                    )
                    if not math_isfinite(limit) or limit <= 1.0:
                        continue
                    mid = seg.distance_start_m + seg.length_m / 2.0
                    nearest = int(np.argmin(np.abs(lap_d - mid)))
                    ratio = float(lap_speed_ms[nearest] / limit)
                    if math_isfinite(ratio):
                        speed_ratios.append(ratio)

        if speed_ratios:
            # Use a high percentile rather than the median/mean.  The preview
            # model's target speed ratio represents the driver's intended pace
            # near the physics envelope; slower samples include traffic,
            # driver-change laps, mistakes, and non-limit corner entries.
            target_speed_ratio = float(
                np.clip(np.nanpercentile(speed_ratios, 95), 0.90, 1.0)
            )
        else:
            target_speed_ratio = PreviewDriverParams.target_speed_ratio

        throttle = np.clip(aim_df["Throttle Pos"].values / 100.0, 0.0, 1.0)
        max_throttle = float(np.clip(np.nanpercentile(throttle, 95), 0.35, 1.0))

        brake_raw = np.maximum(
            aim_df["FBrakePressure"].values,
            aim_df["RBrakePressure"].values,
        )
        brake = np.clip(brake_raw / max(float(brake_max_pressure_bar), 1.0), 0.0, 1.0)
        max_brake = float(np.clip(np.nanpercentile(brake, 95), 0.25, 1.0))

        params = PreviewDriverParams(
            target_speed_ratio=target_speed_ratio,
            max_throttle=max_throttle,
            max_brake=max_brake,
        )
        return cls(track, dynamics, params=params, name=name)

    @staticmethod
    def _selected_lap_slices(
        aim_df: pd.DataFrame,
        laps: list[int] | None,
    ) -> list[tuple[int, int, float]]:
        boundaries = _detect_lap_boundaries_safe(aim_df)
        if boundaries:
            if laps is not None:
                selected = [boundaries[i] for i in laps if 0 <= i < len(boundaries)]
                return selected or boundaries
            try:
                from fsae_sim.analysis.telemetry_analysis import _auto_select_laps

                selected = _auto_select_laps(aim_df, boundaries)
                return selected or boundaries
            except Exception:
                return boundaries
        total_dist = aim_df["Distance on GPS Speed"].values
        return [(0, len(aim_df), float(total_dist[-1] - total_dist[0]))]

    def _preview_target(
        self,
        segment_idx: int,
        speed_ms: float,
    ) -> tuple[float, float]:
        preview_segments = self._preview_segment_count(segment_idx, speed_ms)
        if (
            self._preview_target_cache is not None
            and self._preview_distance_cache is not None
        ):
            cache_idx = min(
                preview_segments,
                self._preview_target_cache.shape[0],
            ) - 1
            return (
                float(self._preview_target_cache[cache_idx, segment_idx]),
                float(self._preview_distance_cache[cache_idx, segment_idx]),
            )

        targets: list[float] = []
        distances: list[float] = []
        cumulative = 0.0
        target_array = self._target_ms

        for offset in range(preview_segments):
            idx = (segment_idx + offset) % self._num_segments
            seg = self._track.segments[idx]
            cumulative += seg.length_m
            if target_array is not None:
                target = float(target_array[idx])
            elif self._dynamics is not None:
                target = self._dynamics.max_cornering_speed(
                    seg.curvature,
                    seg.grip_factor,
                )
                if math_isfinite(target):
                    target *= float(np.clip(self.params.target_speed_ratio, 0.2, 1.0))
            else:
                target = float("inf")
            targets.append(float(target))
            distances.append(cumulative)

        finite = [
            (target, dist) for target, dist in zip(targets, distances)
            if math_isfinite(target)
        ]
        if not finite:
            return float("inf"), cumulative
        return min(finite, key=lambda item: item[0])

    def _preview_segment_count(self, segment_idx: int, speed_ms: float) -> int:
        target_distance = max(
            self._track.segments[segment_idx].length_m * self.params.min_preview_segments,
            max(speed_ms, 1.0) * self.params.preview_time_s,
        )
        total = 0.0
        count = 0
        max_count = max(
            int(self.params.min_preview_segments),
            int(self.params.max_preview_segments),
        )
        while count < min(max_count, self._num_segments):
            idx = (segment_idx + count) % self._num_segments
            total += self._track.segments[idx].length_m
            count += 1
            if count >= self.params.min_preview_segments and total >= target_distance:
                break
        return max(1, count)

    def _max_brake_decel(self, speed_ms: float) -> float:
        if self._dynamics is None:
            return 0.55 * 9.81
        return (
            self._dynamics.mechanical_brake_force(1.0, speed_ms)
            / max(self._dynamics.m_effective, 1e-6)
        )

    def _build_preview_cache(self) -> None:
        target_array = self._target_ms
        if target_array is None or self._num_segments <= 0:
            self._preview_target_cache = None
            self._preview_distance_cache = None
            return

        max_count = min(
            max(
                int(self.params.min_preview_segments),
                int(self.params.max_preview_segments),
            ),
            self._num_segments,
        )
        target_cache = np.empty((max_count, self._num_segments), dtype=np.float64)
        distance_cache = np.empty((max_count, self._num_segments), dtype=np.float64)

        for start_idx in range(self._num_segments):
            best_target = float("inf")
            best_distance = 0.0
            cumulative = 0.0
            for offset in range(max_count):
                idx = (start_idx + offset) % self._num_segments
                cumulative += self._track.segments[idx].length_m
                target = float(target_array[idx])
                if target < best_target:
                    best_target = target
                    best_distance = cumulative
                target_cache[offset, start_idx] = best_target
                distance_cache[offset, start_idx] = best_distance

        self._preview_target_cache = target_cache
        self._preview_distance_cache = distance_cache


def math_isfinite(value: float) -> bool:
    return math.isfinite(float(value))


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
            # D-03: coast_throttle was a dead knob — the engine's coast
            # path uses the back-EMF-aware electrical_power model (D-17)
            # which depends on motor state, not on any driver "throttle"
            # during COAST. throttle_pct is forced to 0.0 so no downstream
            # caller can accidentally depend on it.
            return ControlCommand(
                ControlAction.COAST, throttle_pct=0.0, brake_pct=0.0,
            )

    def with_params(self, **kwargs) -> PedalProfileStrategy:
        """Return a new strategy with modified DriverParams.

        Shares the underlying profile arrays (numpy views).
        Only the DriverParams are replaced.
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

    @classmethod
    def from_telemetry(
        cls,
        aim_df: pd.DataFrame,
        track: Track,
        *,
        laps: list[int] | None = None,
        throttle_threshold: float = 5.0,
        brake_threshold: float = 2.0,
        brake_max_pressure_bar: float | None = None,
        name: str = "pedal_profile",
    ) -> PedalProfileStrategy:
        """Calibrate from AiM telemetry.

        Samples raw throttle position and brake pressure at each track
        segment midpoint, classifies actions per-lap, then aggregates
        across representative laps using median for pedal/brake values.
        """
        num_segments = track.num_segments
        lap_boundaries = _detect_lap_boundaries_safe(aim_df)

        if lap_boundaries and len(lap_boundaries) >= 2:
            if laps is not None:
                selected = [lap_boundaries[i] for i in laps if i < len(lap_boundaries)]
            else:
                # D-18: distance + time + mean_speed filter (shared
                # helper with extract_per_segment_actions).
                from fsae_sim.analysis.telemetry_analysis import _auto_select_laps
                selected = _auto_select_laps(aim_df, lap_boundaries)
            if not selected:
                selected = lap_boundaries
        else:
            total_dist = aim_df["Distance on GPS Speed"].values
            selected = [(0, len(aim_df), float(total_dist[-1] - total_dist[0]))]

        # D-08: brake normalization is now data-independent (DSS-derived
        # max pressure, not 99th percentile of the data at hand).
        if brake_max_pressure_bar is not None:
            brake_norm = float(brake_max_pressure_bar)
        else:
            brake_norm = 60.0
        brake_norm = max(brake_norm, 1.0)

        has_torque = "LVCU Torque Req" in aim_df.columns
        _INVERTER_TORQUE_LIMIT = 85.0

        throttle_matrix = []
        pedal_matrix = []
        brake_matrix = []
        speed_matrix = []
        speed_col = _telemetry_speed_col(aim_df)

        for start_idx, end_idx, _ in selected:
            lap_df = aim_df.iloc[start_idx:end_idx]
            lap_dist_raw = lap_df["Distance on GPS Speed"].values
            # D-07: rescale each lap onto the track total distance so
            # segment-midpoint lookups land at the correct physical
            # location across all laps (see telemetry_analysis.py for
            # rationale).
            lap_span = float(lap_dist_raw[-1] - lap_dist_raw[0])
            if lap_span > 0.0:
                lap_d = (lap_dist_raw - lap_dist_raw[0]) * (
                    track.total_distance_m / lap_span
                )
            else:
                lap_d = lap_dist_raw - lap_dist_raw[0]
            lap_throttle_raw = lap_df["Throttle Pos"].values
            lap_speed = lap_df[speed_col].values
            lap_brake = np.maximum(
                lap_df["FBrakePressure"].values,
                lap_df["RBrakePressure"].values,
            )
            if has_torque:
                lap_torque = lap_df["LVCU Torque Req"].values
            else:
                lap_torque = None

            lap_throttles = np.zeros(num_segments)
            lap_pedals = np.zeros(num_segments)
            lap_brakes = np.zeros(num_segments)
            lap_speeds = np.zeros(num_segments)

            for seg in track.segments:
                mid = seg.distance_start_m + seg.length_m / 2.0
                half_bin = seg.length_m / 2.0

                mask = (lap_d >= mid - half_bin) & (lap_d < mid + half_bin)
                if not np.any(mask):
                    nearest_idx = np.argmin(np.abs(lap_d - mid))
                    mask = np.zeros(len(lap_d), dtype=bool)
                    mask[nearest_idx] = True

                seg_pedal = float(np.median(lap_throttle_raw[mask]))
                seg_brake = float(np.median(lap_brake[mask]))
                seg_speed = float(np.mean(lap_speed[mask]))

                if lap_torque is not None:
                    seg_torque = float(np.median(np.clip(lap_torque[mask], 0, None)))
                    lap_throttles[seg.index] = float(np.clip(
                        seg_torque / _INVERTER_TORQUE_LIMIT, 0.0, 1.0,
                    ))
                else:
                    lap_throttles[seg.index] = float(np.clip(seg_pedal / 100.0, 0.0, 1.0))

                lap_pedals[seg.index] = seg_pedal
                lap_brakes[seg.index] = float(np.clip(
                    max(0.0, seg_brake) / brake_norm, 0.0, 1.0,
                ))
                lap_speeds[seg.index] = seg_speed / 3.6

            throttle_matrix.append(lap_throttles)
            pedal_matrix.append(lap_pedals)
            brake_matrix.append(lap_brakes)
            speed_matrix.append(lap_speeds)

        throttle_arr = np.array(throttle_matrix)
        pedal_arr = np.array(pedal_matrix)
        brake_arr = np.array(brake_matrix)
        speed_arr = np.array(speed_matrix)

        final_throttle = np.median(throttle_arr, axis=0)
        final_pedal = np.median(pedal_arr, axis=0)
        final_brake = np.median(brake_arr, axis=0)
        final_speed = np.mean(speed_arr, axis=0)

        final_actions = np.zeros(num_segments, dtype=int)
        brake_threshold_norm = brake_threshold / brake_norm
        # D-01: classify on torque fraction, not raw pedal %.
        # `final_throttle` is the median LVCU-torque / inverter_cap per
        # segment (or pedal/100 fallback). Using it directly keeps the
        # classifier aligned with the intensity the driver will actually
        # command, so low-pedal-but-high-torque segments (back-EMF-limited
        # region) no longer drop to COAST. Threshold is 3% of inverter cap.
        TORQUE_FRACTION_THROTTLE_THRESHOLD = 0.03
        for i in range(num_segments):
            if final_brake[i] > brake_threshold_norm:
                final_actions[i] = 2
                final_throttle[i] = 0.0
            elif final_throttle[i] > TORQUE_FRACTION_THROTTLE_THRESHOLD:
                final_actions[i] = 1
                final_brake[i] = 0.0
            else:
                final_actions[i] = 0
                final_throttle[i] = 0.0
                final_brake[i] = 0.0

        return cls(
            throttle_pct=final_throttle,
            brake_pct=final_brake,
            actions=final_actions,
            ref_speed_ms=final_speed,
            num_segments=num_segments,
        )
