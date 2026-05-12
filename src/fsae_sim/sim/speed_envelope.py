"""Forward-backward speed envelope for quasi-static simulation.

Computes the fastest physically achievable speed at every track segment,
respecting cornering limits, powertrain acceleration, and braking
deceleration.  The result is a speed ceiling that no synthetic driver
strategy can exceed.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fsae_sim.track.track import Track
    from fsae_sim.vehicle.dynamics import VehicleDynamics
    from fsae_sim.vehicle.powertrain_model import PowertrainModel


class SpeedEnvelope:
    """Forward-backward speed envelope solver with corner speed caching.

    Args:
        dynamics: Vehicle dynamics model (for corner speeds and resistance).
        powertrain: Powertrain model (for max drive/regen force).
        track: Track geometry.
    """

    _MIN_SPEED: float = 0.5

    def __init__(
        self,
        dynamics: VehicleDynamics,
        powertrain: PowertrainModel,
        track: Track,
    ) -> None:
        self._dynamics = dynamics
        self._powertrain = powertrain
        self._track = track
        self._corner_speed_cache: dict[tuple, np.ndarray] = {}
        # BMS refresh bookkeeping. ``last_built_bms_limit_a`` is the
        # BMS current limit the most recent envelope was built with;
        # the engine compares this against the live (temp, soc)-derived
        # limit at lap boundaries and recomputes when the drift exceeds
        # ``BMS_REFRESH_DELTA_A``. ``None`` until the first compute call.
        self.last_built_bms_limit_a: float | None = None

    def compute(
        self,
        initial_speed: float = 0.5,
        bms_current_limit_a: float | None = None,
    ) -> np.ndarray:
        """Compute the speed envelope for the full track.

        Records the BMS current limit used in this build under
        ``self.last_built_bms_limit_a`` so callers can decide whether
        to recompute (e.g. at lap boundaries when thermal derating has
        moved the limit).

        Args:
            initial_speed: Vehicle speed at segment 0 (m/s).
            bms_current_limit_a: BMS discharge current limit (A) used
                by the forward-pass acceleration ceiling.

        Returns:
            1-D array of maximum feasible speed (m/s) per segment.
        """
        # Track the BMS limit the envelope was built with so the engine
        # can compare against later (temp, soc)-derived limits.
        self.last_built_bms_limit_a = bms_current_limit_a

        segments = self._track.segments
        n = len(segments)
        m_eff = self._dynamics.m_effective

        # Pass 1: corner speeds (cached)
        v_corner = self._get_corner_speeds()

        # Pass 2: backward pass (braking feasibility)
        # C7: every total_resistance call carries the local segment's
        # curvature so cornering drag is accounted for on the planning
        # side as well as the engine-side force balance.
        v_back = np.empty(n, dtype=np.float64)
        v_back[n - 1] = v_corner[n - 1]

        for i in range(n - 2, -1, -1):
            v = v_back[i + 1]
            seg = segments[i]
            # Max braking force = passive resistance + active mechanical
            # braking, bounded by tire grip. CT-16EV mechanical pads can
            # exceed motor regen (~0.5 g) and reach the tire limit (~1 g),
            # so brake capacity is the tire-grip ceiling, not the regen
            # ceiling. Adding resistance on top keeps the math honest:
            # every newton of drag also reduces speed.
            f_resist = self._resistance(v, seg.grade, seg.curvature)
            # Use mechanical_brake_force(1.0, v) as the active braking
            # ceiling. This honours the calibrated peak-brake-decel
            # (~0.55 g for CT-16EV) rather than the tire-grip ceiling.
            # Adding resist on top is honest: drag and rolling resistance
            # also slow the car independently of brake-pad output.
            f_brake_active = self._brake_force(v)
            f_brake = f_brake_active + f_resist
            a_brake = f_brake / m_eff

            # v_entry^2 = v_exit^2 + 2 * a_brake * d
            v_entry_sq = v * v + 2.0 * a_brake * seg.length_m
            v_back[i] = min(v_corner[i], math.sqrt(max(0.0, v_entry_sq)))

        # Lap-wrap: iterate until fixed point.  A reduction propagated
        # into segment 0 can change what the last segment must brake
        # for; one pass with an early break can miss the feedback.
        # Cap iterations as a safety — typical convergence in ≤ 3 rounds.
        for _wrap_iter in range(5):
            last_seg = segments[n - 1]
            v_last = v_back[n - 1]
            f_resist = self._resistance(
                v_back[0], last_seg.grade, last_seg.curvature
            )
            f_brake_active = self._brake_force(v_back[0])
            f_brake = f_brake_active + f_resist
            a_brake = f_brake / m_eff
            v_wrap_sq = (
                v_back[0] * v_back[0] + 2.0 * a_brake * last_seg.length_m
            )
            v_wrap = math.sqrt(max(0.0, v_wrap_sq))

            if v_last <= v_wrap:
                break  # fixed point: last segment already satisfies wrap

            v_back[n - 1] = min(v_back[n - 1], v_wrap)
            changed = False
            for i in range(n - 2, -1, -1):
                v = v_back[i + 1]
                seg = segments[i]
                f_resist = self._resistance(v, seg.grade, seg.curvature)
                f_brake_active = self._brake_force(v)
                f_brake = f_brake_active + f_resist
                a_brake = f_brake / m_eff
                v_entry_sq = v * v + 2.0 * a_brake * seg.length_m
                new_limit = min(v_corner[i], math.sqrt(max(0.0, v_entry_sq)))
                if new_limit < v_back[i]:
                    v_back[i] = new_limit
                    changed = True
            if not changed:
                break

        # Pass 3: forward pass (acceleration feasibility)
        # Use lap-wrapped backward-pass limit as initial speed.  The backward
        # pass already handles circuit wrap-around, so v_back[0] is the fastest
        # the car can enter segment 0 while still braking for upcoming corners.
        # Using initial_speed here created an artificial acceleration ramp that
        # penalised every lap on straight segments.
        v_fwd = np.empty(n, dtype=np.float64)
        # If there are no cornering constraints anywhere on the track,
        # v_back[0] will be infinite and the forward pass has no finite
        # starting point.  Fall back to ``initial_speed`` so acceleration
        # is bounded by powertrain physics instead of producing NaN.
        if math.isinf(v_back[0]):
            v_fwd[0] = max(initial_speed, self._MIN_SPEED)
        else:
            v_fwd[0] = v_back[0]

        for i in range(1, n):
            v = v_fwd[i - 1]
            prev_seg = segments[i - 1]
            f_drive = self._drive_force(v, bms_current_limit_a)
            f_resist = self._resistance(v, prev_seg.grade, prev_seg.curvature)
            f_net = f_drive - f_resist
            a_accel = f_net / m_eff

            v_exit_sq = v * v + 2.0 * a_accel * prev_seg.length_m
            v_exit = math.sqrt(max(0.0, v_exit_sq))
            v_fwd[i] = min(v_back[i], v_exit)

        # Pass 4: combined slip correction
        # Where the envelope shows acceleration or braking near corners,
        # re-check corner speeds with longitudinal_g to account for
        # friction ellipse reduction.
        #
        # Guard: only proceed if max_cornering_speed accepts longitudinal_g.
        # We inspect the underlying callable (side_effect for mocks, the bound
        # method otherwise) so we never make extra calls against a legacy
        # dynamics object that doesn't support the parameter.
        _cs_callable = getattr(
            self._dynamics.max_cornering_speed, "side_effect", None
        ) or self._dynamics.max_cornering_speed
        try:
            _sig = inspect.signature(_cs_callable)
            _supports_long_g = "longitudinal_g" in _sig.parameters
        except (ValueError, TypeError):
            _supports_long_g = False

        v_corrected = v_fwd.copy()
        needs_repropagation = False

        if _supports_long_g:
            for i in range(n):
                seg = segments[i]
                if abs(seg.curvature) < 1e-6:
                    continue  # only correct at corners

                # Estimate longitudinal_g from the speed change BETWEEN the
                # previous segment's exit and this segment's exit.  dv^2
                # accumulated across the PREVIOUS segment's length
                # (v_fwd[i] came from forward-integrating over segments[i-1]),
                # so divide by that segment's length, not the current one.
                if i > 0:
                    dv_sq = v_fwd[i] ** 2 - v_fwd[i - 1] ** 2
                    prev_length = segments[i - 1].length_m
                    a_long = dv_sq / (2.0 * prev_length) if prev_length > 0 else 0.0
                    long_g = a_long / 9.81
                else:
                    long_g = 0.0

                if abs(long_g) < 0.01:
                    continue

                # Re-query corner speed with longitudinal demand
                try:
                    v_corrected_corner = self._dynamics.max_cornering_speed(
                        seg.curvature, seg.grip_factor, longitudinal_g=long_g,
                    )
                except TypeError:
                    # Dynamics doesn't support longitudinal_g (e.g., legacy mode)
                    continue

                if v_corrected_corner < v_corrected[i]:
                    v_corrected[i] = v_corrected_corner
                    needs_repropagation = True

        if needs_repropagation:
            # Re-run backward pass from corrected values
            for i in range(n - 2, -1, -1):
                v = v_corrected[i + 1]
                seg = segments[i]
                f_resist = self._resistance(v, seg.grade, seg.curvature)
                f_brake_active = self._brake_force(v)
                f_brake = f_brake_active + f_resist
                a_brake = f_brake / m_eff
                v_entry_sq = v * v + 2.0 * a_brake * seg.length_m
                new_limit = min(v_corrected[i], math.sqrt(max(0.0, v_entry_sq)))
                if new_limit >= v_corrected[i]:
                    continue
                v_corrected[i] = new_limit

            # Re-run forward pass
            v_corrected[0] = min(v_corrected[0], v_back[0])
            for i in range(1, n):
                v = v_corrected[i - 1]
                prev_seg = segments[i - 1]
                f_drive = self._drive_force(v, bms_current_limit_a)
                f_resist = self._resistance(v, prev_seg.grade, prev_seg.curvature)
                f_net = f_drive - f_resist
                a_accel = f_net / m_eff
                v_exit_sq = v * v + 2.0 * a_accel * prev_seg.length_m
                v_exit = math.sqrt(max(0.0, v_exit_sq))
                v_corrected[i] = min(v_corrected[i], v_exit)

            return v_corrected

        return v_fwd

    # ------------------------------------------------------------------
    # Signature-safe resistance call (C7)
    # ------------------------------------------------------------------

    def _resistance(self, speed: float, grade: float, curvature: float) -> float:
        """Call ``total_resistance`` with whichever kwargs it accepts.

        The production ``VehicleDynamics.total_resistance`` accepts
        ``(speed, grade, curvature)`` so we can pass all three.  Test
        doubles and legacy callables may only accept ``(speed, grade)``
        or ``(speed,)``; introspect once and cache the result.
        """
        if not hasattr(self, "_resist_call_mode"):
            fn = getattr(
                self._dynamics.total_resistance, "side_effect", None,
            ) or self._dynamics.total_resistance
            try:
                params = inspect.signature(fn).parameters
            except (ValueError, TypeError):
                params = {}
            has_grade = "grade" in params
            has_curvature = "curvature" in params
            if has_grade and has_curvature:
                self._resist_call_mode = "grade_curvature"
            elif has_grade:
                self._resist_call_mode = "grade"
            else:
                self._resist_call_mode = "speed"
        if self._resist_call_mode == "grade_curvature":
            return self._dynamics.total_resistance(
                speed, grade=grade, curvature=curvature,
            )
        if self._resist_call_mode == "grade":
            return self._dynamics.total_resistance(speed, grade=grade)
        return self._dynamics.total_resistance(speed)

    def _as_finite_float(self, value, fallback: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(out):
            return fallback
        return out

    def _brake_force(self, speed: float) -> float:
        """Runtime-consistent active mechanical braking ceiling."""
        fallback = self._as_finite_float(
            self._dynamics.max_braking_force(speed), 0.0,
        )
        fn = getattr(self._dynamics, "mechanical_brake_force", None)
        if fn is None:
            return fallback
        active = self._as_finite_float(fn(1.0, speed), fallback)
        if fallback > 10.0 and active < 10.0:
            return fallback
        return active

    def _drive_force(
        self,
        speed: float,
        bms_current_limit_a: float | None,
    ) -> float:
        """Runtime-consistent full-throttle force, optionally BMS limited."""
        if bms_current_limit_a is not None and hasattr(
            self._powertrain, "lvcu_torque_ceiling"
        ):
            rpm = self._powertrain.motor_rpm_from_speed(speed)
            torque = self._powertrain.lvcu_torque_ceiling(
                rpm, bms_current_limit_a,
            )
            if hasattr(self._powertrain, "apply_inverter_delivery"):
                torque = self._powertrain.apply_inverter_delivery(rpm, torque)
            f_drive = self._powertrain.wheel_force(torque)
        else:
            f_drive = self._powertrain.drive_force(1.0, speed)
        f_traction = self._dynamics.max_traction_force(speed)
        return min(
            self._as_finite_float(f_drive, 0.0),
            self._as_finite_float(f_traction, float("inf")),
        )

    # ------------------------------------------------------------------
    # Corner speed caching
    # ------------------------------------------------------------------

    def _cache_key(self) -> tuple:
        grip_factors = tuple(s.grip_factor for s in self._track.segments)
        return (
            self._track.name,
            len(self._track.segments),
            self._dynamics.vehicle.mass_kg,
            grip_factors,
        )

    def _get_corner_speeds(self) -> np.ndarray:
        key = self._cache_key()
        if key in self._corner_speed_cache:
            return self._corner_speed_cache[key].copy()

        v_corner = np.array([
            self._dynamics.max_cornering_speed(seg.curvature, seg.grip_factor)
            for seg in self._track.segments
        ])
        self._corner_speed_cache[key] = v_corner
        return v_corner.copy()
