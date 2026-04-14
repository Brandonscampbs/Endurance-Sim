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
        v_back = np.empty(n, dtype=np.float64)
        v_back[n - 1] = v_corner[n - 1]

        for i in range(n - 2, -1, -1):
            v = v_back[i + 1]
            # Max braking force at the speed we need to reach
            f_resist = self._dynamics.total_resistance(v)
            f_regen = abs(self._powertrain.regen_force(1.0, v))
            f_tire_limit = self._dynamics.max_braking_force(v)
            f_brake = min(f_resist + f_regen, f_tire_limit)
            a_brake = f_brake / m_eff

            # v_entry^2 = v_exit^2 + 2 * a_brake * d
            v_entry_sq = v * v + 2.0 * a_brake * segments[i].length_m
            v_back[i] = min(v_corner[i], math.sqrt(v_entry_sq))

        # Lap-wrap: check if the last segment can feed into the first
        v_last = v_back[n - 1]
        f_resist = self._dynamics.total_resistance(v_back[0])
        f_regen = abs(self._powertrain.regen_force(1.0, v_back[0]))
        f_tire_limit = self._dynamics.max_braking_force(v_back[0])
        f_brake = min(f_resist + f_regen, f_tire_limit)
        a_brake = f_brake / m_eff
        v_wrap_sq = v_back[0] * v_back[0] + 2.0 * a_brake * segments[n - 1].length_m
        v_wrap = math.sqrt(v_wrap_sq)

        if v_last > v_wrap:
            v_back[n - 1] = min(v_back[n - 1], v_wrap)
            for i in range(n - 2, -1, -1):
                v = v_back[i + 1]
                f_resist = self._dynamics.total_resistance(v)
                f_regen = abs(self._powertrain.regen_force(1.0, v))
                f_tire_limit = self._dynamics.max_braking_force(v)
                f_brake = min(f_resist + f_regen, f_tire_limit)
                a_brake = f_brake / m_eff
                v_entry_sq = v * v + 2.0 * a_brake * segments[i].length_m
                new_limit = min(v_corner[i], math.sqrt(v_entry_sq))
                if new_limit >= v_back[i]:
                    break
                v_back[i] = new_limit

        # Pass 3: forward pass (acceleration feasibility)
        v_fwd = np.empty(n, dtype=np.float64)
        v_fwd[0] = min(v_back[0], max(initial_speed, self._MIN_SPEED))

        for i in range(1, n):
            v = v_fwd[i - 1]
            f_drive = self._powertrain.drive_force(1.0, v)
            f_traction = self._dynamics.max_traction_force(v)
            f_drive = min(f_drive, f_traction)
            f_resist = self._dynamics.total_resistance(v, segments[i - 1].grade)
            f_net = f_drive - f_resist
            a_accel = f_net / m_eff

            v_exit_sq = v * v + 2.0 * a_accel * segments[i - 1].length_m
            v_exit = math.sqrt(max(0.0, v_exit_sq))
            v_fwd[i] = min(v_back[i], v_exit)

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
