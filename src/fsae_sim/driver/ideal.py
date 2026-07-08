"""Idealized no-brake coast-planning driver.

This strategy is intentionally not a human driver model.  It represents a
physics-limited efficiency driver:

* use no mechanical brake and no regen request;
* corner at the lateral speed limit;
* accelerate only up to the lesser of powertrain and tire traction capacity;
* release throttle early enough that passive coast losses carry the car down
  to the next corner speed limit.

The strategy consumes the engine's speed envelope as a hard upper bound, then
builds a stricter "coast-to-corner" envelope by integrating passive
resistance backward around the lap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fsae_sim.driver.strategy import (
    ControlAction,
    ControlCommand,
    DriverStrategy,
    SimState,
)
from fsae_sim.track.track import Segment, Track
from fsae_sim.physics_constants import GRAVITY_M_S2

if TYPE_CHECKING:
    from fsae_sim.vehicle import VehicleConfig
    from fsae_sim.vehicle.dynamics import VehicleDynamics
    from fsae_sim.vehicle.powertrain_model import PowertrainModel


@dataclass(frozen=True)
class CoastOptimalParams:
    """Tuning constants for :class:`CoastOptimalStrategy`.

    The defaults keep a small numerical margin under lateral and coast
    envelopes so the engine's hard speed limiter does not need to add brake
    force for floating-point overshoots.
    """

    corner_speed_margin: float = 0.995
    """Multiplier on the analytical lateral speed limit."""

    traction_margin: float = 0.98
    """Multiplier on available longitudinal tire force."""

    speed_deadband_ms: float = 0.08
    """Do not chase envelope errors smaller than this speed delta."""

    min_speed_ms: float = 0.5
    """Speed floor used for force/powertrain calculations."""

    max_unbounded_speed_ms: float = 90.0
    """Fallback finite speed when a synthetic track has no corners."""

    coast_wrap_iterations: int = 10
    """Backward coast-pass iterations around the closed lap."""

    coast_force_iterations: int = 4
    """Inner iterations for speed-dependent passive resistance."""

    pedal_iterations: int = 24
    """Binary-search iterations for pedal-to-wheel-force inversion."""

    default_bms_current_limit_a: float = 200.0
    """BMS limit used before the engine pushes the live per-lap value."""


class CoastOptimalStrategy(DriverStrategy):
    """Idealized no-brake, traction-limited, coast-to-corner strategy."""

    name = "ideal_coast"

    def __init__(
        self,
        track: Track,
        *,
        dynamics: "VehicleDynamics | None" = None,
        powertrain: "PowertrainModel | None" = None,
        params: CoastOptimalParams | None = None,
    ) -> None:
        self._track = track
        self._dyn = dynamics
        self._pt = powertrain
        self._params = params if params is not None else CoastOptimalParams()
        self._physical_envelope: np.ndarray | None = None
        self._coast_envelope: np.ndarray | None = None
        self._bms_current_limit_a = self._params.default_bms_current_limit_a

    @classmethod
    def from_config(
        cls,
        vehicle_config: "VehicleConfig",
        track: Track,
        *,
        params: CoastOptimalParams | None = None,
    ) -> "CoastOptimalStrategy":
        """Construct from a vehicle config.

        The simulation engine will later call :meth:`bind_models` with its
        exact dynamics and powertrain instances, including tire/load-transfer
        and inverter-delivery maps.  The lightweight objects here make the
        strategy usable in direct unit tests before engine binding.
        """
        from fsae_sim.vehicle.dynamics import VehicleDynamics
        from fsae_sim.vehicle.powertrain_model import PowertrainModel

        return cls(
            track,
            dynamics=VehicleDynamics(
                vehicle_config.vehicle,
                powertrain_config=vehicle_config.powertrain,
            ),
            powertrain=PowertrainModel(vehicle_config.powertrain),
            params=params,
        )

    @property
    def uses_observed_speed_caps(self) -> bool:
        return False

    @property
    def coast_envelope(self) -> np.ndarray:
        if self._coast_envelope is None:
            self._rebuild_coast_envelope()
        assert self._coast_envelope is not None
        return self._coast_envelope.copy()

    def bind_models(
        self,
        dynamics: "VehicleDynamics",
        powertrain: "PowertrainModel",
    ) -> None:
        """Receive the engine's exact force models."""
        self._dyn = dynamics
        self._pt = powertrain
        self._rebuild_coast_envelope()

    def set_envelope(self, v_max: np.ndarray) -> None:
        """Receive the engine's hard physical speed envelope."""
        self._physical_envelope = np.asarray(v_max, dtype=np.float64).copy()
        self._rebuild_coast_envelope()

    def set_bms_limit(self, bms_current_limit_a: float) -> None:
        self._bms_current_limit_a = float(bms_current_limit_a)

    def reset(self) -> None:
        """Stateless, present for engine API symmetry."""
        return None

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        if self._dyn is None or self._pt is None:
            raise RuntimeError("CoastOptimalStrategy needs dynamics and powertrain")
        if not upcoming:
            raise ValueError("CoastOptimalStrategy requires an upcoming segment")
        if self._coast_envelope is None:
            self._rebuild_coast_envelope()

        idx = state.segment_idx % self._track.num_segments
        next_idx = (idx + 1) % self._track.num_segments
        seg = upcoming[0]

        target_exit = float(self._coast_envelope[next_idx])
        if not math.isfinite(target_exit):
            target_exit = self._params.max_unbounded_speed_ms
        target_exit = max(self._params.min_speed_ms, target_exit)

        v_entry = max(float(state.speed), self._params.min_speed_ms)
        if v_entry >= target_exit - self._params.speed_deadband_ms:
            return self._coast()

        length = max(float(seg.length_m), 1e-6)
        v_op = max(self._params.min_speed_ms, 0.5 * (v_entry + target_exit))
        a_target = (target_exit * target_exit - v_entry * v_entry) / (
            2.0 * length
        )
        f_resist = self._resistance(v_op, seg.grade, seg.curvature)
        f_required = self._dyn.m_effective * a_target + f_resist
        if f_required <= 0.0:
            return self._coast()

        f_available = self._available_drive_force(
            v_op,
            seg.curvature,
            seg.grip_factor,
        )
        f_target = min(f_required, f_available)
        if f_target <= 1e-6:
            return self._coast()

        rpm = self._pt.motor_rpm_from_speed(v_op)
        pedal = self._pedal_for_force(f_target, rpm)
        if pedal <= 1e-5:
            return self._coast()
        return ControlCommand(
            action=ControlAction.THROTTLE,
            throttle_pct=pedal,
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

    def _rebuild_coast_envelope(self) -> None:
        if self._dyn is None or self._track.num_segments == 0:
            return

        base = self._corner_limits()
        if self._physical_envelope is not None:
            physical = np.asarray(self._physical_envelope, dtype=np.float64)
            if len(physical) == len(base):
                base = np.minimum(base, physical)

        finite = np.isfinite(base)
        if not finite.any():
            base = np.full_like(base, self._params.max_unbounded_speed_ms)
        else:
            base = np.where(finite, base, self._params.max_unbounded_speed_ms)
        base = np.maximum(base, self._params.min_speed_ms)

        limits = base.copy()
        n = len(limits)
        for _ in range(max(1, self._params.coast_wrap_iterations)):
            previous = limits.copy()
            for i in range(n - 1, -1, -1):
                exit_limit = limits[(i + 1) % n]
                entry_limit = self._coast_entry_speed_for_exit(i, exit_limit)
                limits[i] = min(base[i], limits[i], entry_limit)
            if np.max(np.abs(previous - limits)) < 1e-4:
                break

        self._coast_envelope = np.maximum(limits, self._params.min_speed_ms)

    def _corner_limits(self) -> np.ndarray:
        assert self._dyn is not None
        out = []
        margin = self._params.corner_speed_margin
        for seg in self._track.segments:
            v = self._dyn.max_cornering_speed(
                seg.curvature,
                seg.grip_factor,
            )
            out.append(float(v) * margin if math.isfinite(v) else float("inf"))
        return np.asarray(out, dtype=np.float64)

    def _coast_entry_speed_for_exit(
        self,
        segment_idx: int,
        exit_speed_ms: float,
    ) -> float:
        assert self._dyn is not None
        seg = self._track.segments[segment_idx]
        length = max(float(seg.length_m), 0.0)
        exit_speed = max(float(exit_speed_ms), self._params.min_speed_ms)
        entry_speed = exit_speed
        for _ in range(max(1, self._params.coast_force_iterations)):
            v_op = max(
                self._params.min_speed_ms,
                0.5 * (entry_speed + exit_speed),
            )
            f_resist = self._resistance(v_op, seg.grade, seg.curvature)
            v_sq = exit_speed * exit_speed + (
                2.0 * f_resist * length / self._dyn.m_effective
            )
            entry_speed = math.sqrt(max(self._params.min_speed_ms ** 2, v_sq))
            entry_speed = min(entry_speed, self._params.max_unbounded_speed_ms)
        return entry_speed

    def _resistance(self, speed: float, grade: float, curvature: float) -> float:
        assert self._dyn is not None
        try:
            return float(
                self._dyn.total_resistance(
                    speed,
                    grade=grade,
                    curvature=curvature,
                )
            )
        except TypeError:
            try:
                return float(self._dyn.total_resistance(speed, grade=grade))
            except TypeError:
                return float(self._dyn.total_resistance(speed))

    def _available_drive_force(
        self,
        speed_ms: float,
        curvature: float,
        grip_factor: float,
    ) -> float:
        assert self._dyn is not None and self._pt is not None
        rpm = self._pt.motor_rpm_from_speed(speed_ms)
        full_pedal_force = max(0.0, self._wheel_force_at_pedal(1.0, rpm))
        tire_force = self._finite_or(
            self._dyn.max_traction_force(speed_ms),
            full_pedal_force,
        )
        tire_force *= self._params.traction_margin
        combined_force = self._combined_drive_force_cap(
            speed_ms,
            curvature,
            grip_factor,
            tire_force,
        )
        return max(0.0, min(full_pedal_force, tire_force, combined_force))

    def _combined_drive_force_cap(
        self,
        speed_ms: float,
        curvature: float,
        grip_factor: float,
        tire_force_n: float,
    ) -> float:
        assert self._dyn is not None
        if abs(curvature) < 1e-6 or tire_force_n <= 0.0:
            return tire_force_n

        try:
            zero_long_speed = self._dyn.max_cornering_speed(
                curvature,
                grip_factor,
                longitudinal_g=0.0,
            )
            if speed_ms >= zero_long_speed * self._params.corner_speed_margin:
                return 0.0
        except TypeError:
            return tire_force_n

        m = max(float(self._dyn.vehicle.mass_kg), 1e-9)
        hi_g = max(0.0, tire_force_n / (m * GRAVITY_M_S2))
        lo_g = 0.0
        for _ in range(18):
            mid_g = 0.5 * (lo_g + hi_g)
            try:
                v_allowed = self._dyn.max_cornering_speed(
                    curvature,
                    grip_factor,
                    longitudinal_g=mid_g,
                )
            except TypeError:
                return tire_force_n
            if math.isfinite(v_allowed) and speed_ms > (
                v_allowed * self._params.corner_speed_margin
            ):
                hi_g = mid_g
            else:
                lo_g = mid_g
        return lo_g * m * GRAVITY_M_S2

    def _pedal_for_force(self, force_n: float, rpm: float) -> float:
        assert self._pt is not None
        target = max(0.0, float(force_n))
        lo = 0.0
        hi = 1.0
        f_lo = self._wheel_force_at_pedal(lo, rpm)
        f_hi = self._wheel_force_at_pedal(hi, rpm)
        if target <= f_lo:
            return 0.0
        if f_hi <= target:
            return 1.0
        for _ in range(max(1, self._params.pedal_iterations)):
            mid = 0.5 * (lo + hi)
            f_mid = self._wheel_force_at_pedal(mid, rpm)
            if f_mid < target:
                lo = mid
            else:
                hi = mid
        return max(0.0, min(1.0, hi))

    def _wheel_force_at_pedal(self, pedal_pct: float, rpm: float) -> float:
        assert self._pt is not None
        if rpm >= self._pt.config.motor_speed_max_rpm:
            return 0.0
        lvcu = self._pt.lvcu_torque_command(
            float(pedal_pct),
            float(rpm),
            self._bms_current_limit_a,
        )
        torque = float(lvcu)
        if hasattr(self._pt, "apply_inverter_delivery"):
            torque = float(self._pt.apply_inverter_delivery(rpm, torque))
        return float(self._pt.wheel_force(torque))

    @staticmethod
    def _finite_or(value: float, fallback: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return fallback
        return out if math.isfinite(out) else fallback
