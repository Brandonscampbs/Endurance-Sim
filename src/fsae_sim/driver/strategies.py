"""Driver strategy implementations.

Three strategies:

- ``ReplayStrategy``: reproduces recorded driver inputs (torque, throttle,
  brake) from AiM telemetry through the engine's force model. Used by the
  CLI ``sim_compare.py`` for sim-vs-telemetry comparison and by the engine
  for replay-mode runs.
- ``CalibratedStrategy`` (name="driver"): zone-based driver model fitted
  from per-segment lap-mean telemetry. The production strategy used by
  the backend baseline and the Simulate-page override path.
- ``AdaptiveStrategy`` (name="adaptive"): envelope-following driver (Wave
  4a — pure feedforward). Wraps an :class:`fsae_sim.driver.adaptive.
  AdaptiveDriver` so the rest of the engine can address it through the
  same ``DriverStrategy`` protocol. Wave 4b will add the PI corrector,
  energy shaper, and end-to-end engine integration.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from fsae_sim.driver.adaptive import AdaptiveDriver, AdaptiveDriverParams
    from fsae_sim.vehicle import VehicleConfig
    from fsae_sim.vehicle.powertrain_model import PowertrainModel


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

    name = "driver"

    def __init__(
        self,
        zones: list[DriverZone],
        num_segments: int,
        name: str = "driver",
        use_observed_speed_caps: bool = True,
        segment_actions_df: pd.DataFrame | None = None,
    ) -> None:
        self.name = name
        self._zones = list(zones)
        self._num_segments = num_segments
        self._use_observed_speed_caps = bool(use_observed_speed_caps)
        # Per-segment telemetry table (action + intensity + lap fraction).
        # When provided, runtime decisions read per-segment values so the
        # natural lift-then-throttle taper survives instead of getting
        # flatlined by zone-level intensity averaging. Saved here so
        # ``with_zone_override`` / ``without_observed_speed_caps`` can
        # thread it through derived strategies.
        self._segment_actions_df: pd.DataFrame | None = segment_actions_df

        # Build flat lookup: segment_idx -> (action, intensity, max_speed_ms).
        # ``intensity`` here is the LEGACY single-action intensity used by
        # the zone-only path (back-compat). The lap-mean force model below
        # carries throttle and brake separately and supersedes this.
        self._segment_actions: list[tuple[ControlAction, float, float]] = [
            (ControlAction.COAST, 0.0, 0.0)
        ] * num_segments
        # Lap-mean force commands per segment. ``_segment_throttle_pct[i]``
        # is the time-averaged throttle the calibrated driver applies in
        # segment ``i`` across the lap distribution; ``_segment_brake_pct``
        # is the analogous lap-mean brake. Both can be non-zero in the
        # same segment when the driver mixes throttle and brake across
        # different laps. The engine sums forces from both at runtime
        # (matching how it already handles replay's recorded inputs).
        self._segment_throttle_pct: np.ndarray = np.zeros(num_segments, dtype=np.float64)
        self._segment_brake_pct: np.ndarray = np.zeros(num_segments, dtype=np.float64)
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
                    # Mirror zone intensity into the per-segment force
                    # arrays so zone-only construction (no df) keeps
                    # matching its previous behavior.
                    if zone.action == ControlAction.THROTTLE:
                        self._segment_throttle_pct[seg_idx] = zone.intensity
                    elif zone.action == ControlAction.BRAKE:
                        self._segment_brake_pct[seg_idx] = zone.intensity

        # When per-segment telemetry is provided, override the zone-derived
        # action+intensity for each segment AND populate the lap-mean force
        # arrays. The action label is set to whichever of throttle / brake
        # is dominant (with intensity equal to that dominant value) so the
        # engine's coaching-style branching still produces sane behavior;
        # the lap-mean arrays are what the engine actually integrates.
        if segment_actions_df is not None:
            seg_idx_arr = segment_actions_df["segment_idx"].to_numpy()
            has_throttle_mean = "throttle_lap_mean" in segment_actions_df.columns
            has_brake_mean = "brake_lap_mean" in segment_actions_df.columns
            throttle_mean_arr = (
                segment_actions_df["throttle_lap_mean"].to_numpy()
                if has_throttle_mean else None
            )
            brake_mean_arr = (
                segment_actions_df["brake_lap_mean"].to_numpy()
                if has_brake_mean else None
            )
            # Fall-back column for the action-label intensity when lap-mean
            # columns aren't present (older callers / unit-test fixtures).
            intensity_col = (
                "effective_intensity"
                if "effective_intensity" in segment_actions_df.columns
                else "intensity"
            )
            action_arr = segment_actions_df["action"].to_numpy()
            intensity_arr = segment_actions_df[intensity_col].to_numpy()
            for k in range(len(seg_idx_arr)):
                idx = int(seg_idx_arr[k])
                if not (0 <= idx < num_segments):
                    continue
                _, _, max_speed_ms = self._segment_actions[idx]
                if has_throttle_mean and has_brake_mean:
                    t_mean = float(throttle_mean_arr[k])
                    b_mean = float(brake_mean_arr[k])
                    self._segment_throttle_pct[idx] = max(0.0, t_mean)
                    self._segment_brake_pct[idx] = max(0.0, b_mean)
                    if t_mean >= b_mean and t_mean > 0.0:
                        action = ControlAction.THROTTLE
                        intensity = t_mean
                    elif b_mean > 0.0:
                        action = ControlAction.BRAKE
                        intensity = b_mean
                    else:
                        action = ControlAction.COAST
                        intensity = 0.0
                    self._segment_actions[idx] = (
                        action, intensity, max_speed_ms,
                    )
                else:
                    # Legacy path: set action and lap-mean force from the
                    # single intensity column.
                    action = action_arr[k]
                    intensity = float(intensity_arr[k])
                    self._segment_actions[idx] = (
                        action, intensity, max_speed_ms,
                    )
                    if action == ControlAction.THROTTLE:
                        self._segment_throttle_pct[idx] = intensity
                        self._segment_brake_pct[idx] = 0.0
                    elif action == ControlAction.BRAKE:
                        self._segment_throttle_pct[idx] = 0.0
                        self._segment_brake_pct[idx] = intensity
                    else:
                        self._segment_throttle_pct[idx] = 0.0
                        self._segment_brake_pct[idx] = 0.0

    @property
    def zones(self) -> list[DriverZone]:
        return list(self._zones)

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        idx = state.segment_idx % self._num_segments
        action, _label_intensity, max_speed_ms = self._segment_actions[idx]

        # D-09: pass the zone's observed peak speed through metadata so
        # the engine can cap exit_speed. Only populated when a real cap
        # is available (>0) to leave straight-line zones unconstrained.
        meta: dict | None = None
        if self._use_observed_speed_caps and max_speed_ms > 0.0:
            meta = {"max_speed_ms": float(max_speed_ms)}

        # Lap-mean force model: emit BOTH throttle and brake commands
        # from the per-segment lap-averaged forces. The action label is
        # kept for diagnostics / coaching brief, but the engine uses the
        # pct fields for force balance (matching how it handles replay).
        throttle_value = float(self._segment_throttle_pct[idx])
        brake_value = float(self._segment_brake_pct[idx])

        throttle_value = max(0.0, min(1.0, throttle_value))
        brake_value = max(0.0, min(1.0, brake_value))

        # Choose the descriptive action label from whichever pedal is
        # dominant. Engine no longer routes on this; it's coaching info.
        if throttle_value > brake_value and throttle_value > 0.0:
            cmd_action = ControlAction.THROTTLE
        elif brake_value > 0.0:
            cmd_action = ControlAction.BRAKE
        else:
            cmd_action = ControlAction.COAST

        return ControlCommand(
            cmd_action,
            throttle_pct=throttle_value,
            brake_pct=brake_value,
            metadata=meta,
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
        """Return a new strategy with one zone's action/intensity changed.

        When per-segment telemetry is in use, the override broadcasts the
        new ``action`` + ``intensity`` across every segment in the zone so
        the runtime behavior matches the user's intent.
        """
        new_zones = []
        target_zone: DriverZone | None = None
        for z in self._zones:
            if z.zone_id == zone_id:
                target_zone = z
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

        df_copy: pd.DataFrame | None = None
        if self._segment_actions_df is not None and target_zone is not None:
            df_copy = self._segment_actions_df.copy()
            seg_mask = (
                (df_copy["segment_idx"] >= target_zone.segment_start)
                & (df_copy["segment_idx"] <= target_zone.segment_end)
            )
            df_copy.loc[seg_mask, "action"] = action
            df_copy.loc[seg_mask, "intensity"] = float(intensity)
            if "effective_intensity" in df_copy.columns:
                df_copy.loc[seg_mask, "effective_intensity"] = float(intensity)
            # Override the lap-mean force columns the runtime reads.
            if "throttle_lap_mean" in df_copy.columns:
                df_copy.loc[seg_mask, "throttle_lap_mean"] = (
                    float(intensity) if action == ControlAction.THROTTLE else 0.0
                )
            if "brake_lap_mean" in df_copy.columns:
                df_copy.loc[seg_mask, "brake_lap_mean"] = (
                    float(intensity) if action == ControlAction.BRAKE else 0.0
                )
            if "lap_fraction" in df_copy.columns:
                df_copy.loc[seg_mask, "lap_fraction"] = (
                    0.0 if action == ControlAction.COAST else 1.0
                )

        return CalibratedStrategy(
            new_zones,
            self._num_segments,
            name=self.name,
            use_observed_speed_caps=self._use_observed_speed_caps,
            segment_actions_df=df_copy,
        )

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
            use_observed_speed_caps=False,
            segment_actions_df=self._segment_actions_df,
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
        name: str = "driver",
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

        # Per-segment data is what the runtime reads. Zones are a coaching
        # artifact for plots / driver brief but no longer constrain
        # runtime behavior. ``effective_intensity`` is duty-cycle weighted
        # so ambiguous "barely throttle" segments command less torque than
        # the median across throttle-laps suggests.
        return cls(
            zones,
            track.num_segments,
            name=name,
            use_observed_speed_caps=use_observed_speed_caps,
            segment_actions_df=seg_actions,
        )

    @classmethod
    def from_zone_list(
        cls,
        zones: list[dict],
        track: Track,
        name: str = "driver",
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


class AdaptiveStrategy(DriverStrategy):
    """Envelope-following adaptive driver (Wave 4a — feedforward core).

    Wraps an :class:`fsae_sim.driver.adaptive.AdaptiveDriver` so the
    engine can address it through the same :class:`DriverStrategy`
    protocol used by :class:`ReplayStrategy` and :class:`CalibratedStrategy`.

    Wave 4a behavior: pure feedforward envelope tracking. Inverse force-
    balance solve on the firmware-faithful pedal -> wheel-force chain.
    No PI corrector, no energy shaper, no telemetry — the only inputs
    are the pre-computed speed envelope and the live BMS current limit.

    PREDICTION-mode compatible: ``uses_observed_speed_caps == False`` so
    :class:`fsae_sim.sim.engine.SimulationEngine` accepts this strategy
    when ``mode == SimulationMode.PREDICTION``.

    Wave 4b will add: PI velocity-error corrector, LBP/LS stint-energy
    shaping, end-to-end engine swap-in, and Michigan endurance regression.
    """

    name = "adaptive"

    def __init__(
        self,
        dynamics: VehicleDynamics,
        powertrain: "PowertrainModel",
        track: Track,
        params: "AdaptiveDriverParams | None" = None,
    ) -> None:
        # Local import: AdaptiveDriver imports several heavy types and
        # is not a hot path at module load time.
        from fsae_sim.driver.adaptive import AdaptiveDriver, AdaptiveDriverParams

        self._track = track
        self._params = params if params is not None else AdaptiveDriverParams()
        self._driver = AdaptiveDriver(
            dynamics=dynamics, powertrain=powertrain, params=self._params,
        )

    @classmethod
    def from_config(
        cls,
        vehicle_config: "VehicleConfig",
        track: Track,
        *,
        params: "AdaptiveDriverParams | None" = None,
    ) -> "AdaptiveStrategy":
        """Convenience constructor that builds dynamics + powertrain
        directly from a :class:`VehicleConfig`. Used by the engine
        dispatch and the validation harness.
        """
        from fsae_sim.vehicle.powertrain_model import PowertrainModel
        pt = PowertrainModel(vehicle_config.powertrain)
        dyn = VehicleDynamics(
            vehicle_config.vehicle,
            powertrain_config=vehicle_config.powertrain,
        )
        return cls(dynamics=dyn, powertrain=pt, track=track, params=params)

    # ------------------------------------------------------------------
    # Configuration plumbing (mirrors the CalibratedStrategy/Replay API)
    # ------------------------------------------------------------------

    @property
    def driver(self) -> "AdaptiveDriver":
        return self._driver

    @property
    def params(self) -> "AdaptiveDriverParams":
        return self._params

    @property
    def uses_observed_speed_caps(self) -> bool:
        """Adaptive strategy is PREDICTION-safe: no telemetry caps."""
        return False

    def set_envelope(self, v_max: np.ndarray) -> None:
        """Hook called by the engine after :class:`SpeedEnvelope.compute`.

        Mirrors the same method on calibrated/replay strategies that
        consume the pre-computed envelope.
        """
        self._driver.set_envelope(v_max)

    def set_bms_limit(self, bms_current_limit_a: float) -> None:
        """Hook for the engine's per-lap BMS-refresh path."""
        self._driver.set_bms_limit(bms_current_limit_a)

    def reset(self) -> None:
        """Reset the wrapped driver's internal state (PI, energy shaper).

        Called by the engine at the start of each run so back-to-back
        sims with the same strategy instance are deterministic.
        """
        self._driver.reset()

    def set_energy_state(
        self,
        *,
        energy_used_kwh: float,
        lap_index: int,
        segment_progress: float,
    ) -> None:
        """Forward energy-shaper state from the engine to the driver."""
        self._driver.set_energy_state(
            energy_used_kwh=energy_used_kwh,
            lap_index=lap_index,
            segment_progress=segment_progress,
        )

    # ------------------------------------------------------------------
    # DriverStrategy protocol
    # ------------------------------------------------------------------

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        return self._driver.decide(state, upcoming)
