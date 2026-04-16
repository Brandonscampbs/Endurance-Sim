"""Forward-backward speed envelope for quasi-static simulation.

Computes the fastest physically achievable speed at every track segment,
respecting cornering limits, powertrain acceleration, and braking
deceleration.  The result is a speed ceiling that no synthetic driver
strategy can exceed.
"""

from __future__ import annotations

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

    def compute(self, initial_speed: float = 0.5) -> np.ndarray:
        """Compute the speed envelope for the full track.

        Args:
            initial_speed: Vehicle speed at segment 0 (m/s).

        Returns:
            1-D array of maximum feasible speed (m/s) per segment.
        """
        segments = self._track.segments
        n = len(segments)
        m_eff = self._dynamics.m_effective

        # Pass 1: corner speeds (cached)
        v_corner = self._get_corner_speeds()

        # Pass 2: backward pass (braking feasibility)
        # C7: every total_resistance call must carry the local segment's
        # curvature so cornering drag is accounted for on both sides
        # (envelope planning and engine-side force balance).
        v_back = np.empty(n, dtype=np.float64)
        v_back[n - 1] = v_corner[n - 1]

        for i in range(n - 2, -1, -1):
            v = v_back[i + 1]
            seg = segments[i]
            # Max braking force at the speed we need to reach
            f_resist = self._dynamics.total_resistance(
                v, seg.grade, seg.curvature,
            )
            f_regen = abs(self._powertrain.regen_force(1.0, v))
            f_tire_limit = self._dynamics.max_braking_force(v)
            f_brake = min(f_resist + f_regen, f_tire_limit)
            a_brake = f_brake / m_eff

            # v_entry^2 = v_exit^2 + 2 * a_brake * d
            v_entry_sq = v * v + 2.0 * a_brake * seg.length_m
            v_back[i] = min(v_corner[i], math.sqrt(max(0.0, v_entry_sq)))

        # Lap-wrap: check if the last segment can feed into the first
        last_seg = segments[n - 1]
        v_last = v_back[n - 1]
        f_resist = self._dynamics.total_resistance(
            v_back[0], last_seg.grade, last_seg.curvature,
        )
        f_regen = abs(self._powertrain.regen_force(1.0, v_back[0]))
        f_tire_limit = self._dynamics.max_braking_force(v_back[0])
        f_brake = min(f_resist + f_regen, f_tire_limit)
        a_brake = f_brake / m_eff
        v_wrap_sq = v_back[0] * v_back[0] + 2.0 * a_brake * last_seg.length_m
        v_wrap = math.sqrt(max(0.0, v_wrap_sq))

        if v_last > v_wrap:
            v_back[n - 1] = min(v_back[n - 1], v_wrap)
            for i in range(n - 2, -1, -1):
                v = v_back[i + 1]
                seg = segments[i]
                f_resist = self._dynamics.total_resistance(
                    v, seg.grade, seg.curvature,
                )
                f_regen = abs(self._powertrain.regen_force(1.0, v))
                f_tire_limit = self._dynamics.max_braking_force(v)
                f_brake = min(f_resist + f_regen, f_tire_limit)
                a_brake = f_brake / m_eff
                v_entry_sq = v * v + 2.0 * a_brake * seg.length_m
                new_limit = min(v_corner[i], math.sqrt(max(0.0, v_entry_sq)))
                if new_limit >= v_back[i]:
                    break
                v_back[i] = new_limit

        # Pass 3: forward pass (acceleration feasibility)
        # Use lap-wrapped backward-pass limit as initial speed.  The backward
        # pass already handles circuit wrap-around, so v_back[0] is the fastest
        # the car can enter segment 0 while still braking for upcoming corners.
        # Using initial_speed here created an artificial acceleration ramp that
        # penalised every lap on straight segments.
        v_fwd = np.empty(n, dtype=np.float64)
        v_fwd[0] = v_back[0]

        for i in range(1, n):
            v = v_fwd[i - 1]
            prev_seg = segments[i - 1]
            f_drive = self._powertrain.drive_force(1.0, v)
            f_traction = self._dynamics.max_traction_force(v)
            f_drive = min(f_drive, f_traction)
            f_resist = self._dynamics.total_resistance(
                v, prev_seg.grade, prev_seg.curvature,
            )
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
        # S7: Agent 4's VehicleDynamics.max_cornering_speed accepts
        # longitudinal_g.  Pass it directly; callers that don't (legacy
        # or mocked) will raise TypeError, which we trap and fall through.
        v_corrected = v_fwd.copy()
        needs_repropagation = False

        for i in range(n):
            seg = segments[i]
            if abs(seg.curvature) < 1e-6:
                continue  # only correct at corners

            # Estimate longitudinal_g from speed change across this segment
            if i > 0:
                dv_sq = v_fwd[i] ** 2 - v_fwd[i - 1] ** 2
                a_long = dv_sq / (2.0 * seg.length_m)
                long_g = a_long / 9.81
            else:
                long_g = 0.0

            if abs(long_g) < 0.01:
                continue

            try:
                v_corrected_corner = self._dynamics.max_cornering_speed(
                    seg.curvature, seg.grip_factor, longitudinal_g=long_g,
                )
            except TypeError:
                # Dynamics doesn't yet support longitudinal_g (legacy or
                # a mock that isn't updated).  Skip combined-slip pass.
                break

            if v_corrected_corner < v_corrected[i]:
                v_corrected[i] = v_corrected_corner
                needs_repropagation = True

        if needs_repropagation:
            # Re-run backward pass from corrected values
            for i in range(n - 2, -1, -1):
                v = v_corrected[i + 1]
                seg = segments[i]
                f_resist = self._dynamics.total_resistance(
                    v, seg.grade, seg.curvature,
                )
                f_regen = abs(self._powertrain.regen_force(1.0, v))
                f_tire_limit = self._dynamics.max_braking_force(v)
                f_brake = min(f_resist + f_regen, f_tire_limit)
                a_brake = f_brake / m_eff
                v_entry_sq = v * v + 2.0 * a_brake * seg.length_m
                new_limit = min(
                    v_corrected[i], math.sqrt(max(0.0, v_entry_sq)),
                )
                if new_limit >= v_corrected[i]:
                    continue
                v_corrected[i] = new_limit

            # Re-run forward pass
            v_corrected[0] = min(v_corrected[0], v_back[0])
            for i in range(1, n):
                v = v_corrected[i - 1]
                prev_seg = segments[i - 1]
                f_drive = self._powertrain.drive_force(1.0, v)
                f_traction = self._dynamics.max_traction_force(v)
                f_drive = min(f_drive, f_traction)
                f_resist = self._dynamics.total_resistance(
                    v, prev_seg.grade, prev_seg.curvature,
                )
                f_net = f_drive - f_resist
                a_accel = f_net / m_eff
                v_exit_sq = v * v + 2.0 * a_accel * prev_seg.length_m
                v_exit = math.sqrt(max(0.0, v_exit_sq))
                v_corrected[i] = min(v_corrected[i], v_exit)

            return v_corrected

        return v_fwd

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
