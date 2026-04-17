"""Cornering speed solver for FSAE vehicle dynamics.

Computes the maximum sustainable cornering speed for a given path
curvature by bisecting over speed and checking whether the four tires
can collectively produce the required centripetal force.  Roll-induced
camber changes are modelled so that degressive tire models and load
transfer effects are captured accurately.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsae_sim.vehicle.load_transfer import LoadTransferModel
    from fsae_sim.vehicle.tire_model import PacejkaTireModel


class CorneringSolver:
    """Find the maximum steady-state cornering speed for a given curvature.

    Uses bisection search over speed to find the highest speed at which
    the four tires can produce enough lateral force to sustain the
    required centripetal acceleration.

    Tire ordering convention follows LoadTransferModel: (FL, FR, RL, RR).
    Positive curvature = right turn.

    Args:
        tire_model: Pacejka tire model providing ``peak_lateral_force``.
        load_transfer: Load transfer model providing ``tire_loads`` and
            roll stiffness values.
        mass_kg: Total vehicle mass including driver (kg).
        static_camber_front_rad: Static camber angle for front tires (rad).
        static_camber_rear_rad: Static camber angle for rear tires (rad).
        roll_camber_front: Roll-camber coefficient for front (deg camber
            per deg roll).  Typically negative for conventional geometry.
        roll_camber_rear: Roll-camber coefficient for rear (deg camber
            per deg roll).
    """

    GRAVITY: float = 9.81
    _V_LOW: float = 0.5
    _V_HIGH: float = 50.0
    _ITERATIONS: int = 30  # hard cap; loop exits earlier on convergence
    _V_TOL_ABS: float = 0.01  # absolute m/s tolerance
    _V_TOL_REL: float = 1e-3  # relative tolerance
    _CURVATURE_THRESHOLD: float = 1e-6

    def __init__(
        self,
        tire_model: PacejkaTireModel,
        load_transfer: LoadTransferModel,
        mass_kg: float,
        static_camber_front_rad: float,
        static_camber_rear_rad: float,
        roll_camber_front: float,
        roll_camber_rear: float,
    ) -> None:
        self._tire = tire_model
        self._load_transfer = load_transfer
        self._mass_kg = mass_kg
        self._static_camber_front_rad = static_camber_front_rad
        self._static_camber_rear_rad = static_camber_rear_rad
        self._roll_camber_front = roll_camber_front
        self._roll_camber_rear = roll_camber_rear

    def max_cornering_speed(
        self,
        curvature: float,
        mu_scale: float = 1.0,
        longitudinal_g: float = 0.0,
    ) -> float:
        """Find the maximum speed sustainable through a corner.

        For straight segments (|curvature| < threshold), returns infinity.
        Otherwise performs bisection search between ``_V_LOW`` and ``_V_HIGH``
        for 30 iterations, converging to < 0.1 m/s tolerance.

        Args:
            curvature: Path curvature in 1/m.  Positive = right turn.
                Only the magnitude matters; sign is ignored.
            mu_scale: Friction scaling factor.  1.0 = nominal grip,
                < 1.0 = reduced grip (e.g. wet), > 1.0 = extra grip.
            longitudinal_g: Simultaneous longitudinal acceleration in g-units.
                Positive = accelerating, negative = braking.  Values with
                ``|longitudinal_g| < 0.01`` are treated as zero (pure
                cornering).  Non-zero values invoke the friction-ellipse
                model, reducing available lateral capacity.

        Returns:
            Maximum sustainable speed in m/s.  ``math.inf`` for straights.
        """
        if abs(curvature) < self._CURVATURE_THRESHOLD:
            return math.inf

        abs_curvature = abs(curvature)

        v_low = self._V_LOW
        v_high = self._V_HIGH

        # Convergence-gated bisection (NF-34): exit when bracket width drops
        # below max(abs_tol, rel_tol * v_low).  The iteration counter is a
        # hard safety cap; typical convergence is < 15 halvings.
        for _ in range(self._ITERATIONS):
            tol = max(self._V_TOL_ABS, self._V_TOL_REL * v_low)
            if (v_high - v_low) <= tol:
                break
            v_mid = (v_low + v_high) / 2.0
            if self._can_sustain(v_mid, abs_curvature, mu_scale, longitudinal_g):
                v_low = v_mid
            else:
                v_high = v_mid

        # Return the lower bracket (known to sustain) to avoid reporting a
        # speed that fails the sustain check due to tire-model regularizer
        # oscillation at the boundary.
        return v_low

    def _can_sustain(
        self,
        speed: float,
        curvature: float,
        mu_scale: float,
        longitudinal_g: float = 0.0,
    ) -> bool:
        """Check whether the vehicle can sustain cornering at the given speed.

        Computes the required centripetal acceleration, obtains per-tire
        normal loads from the load transfer model, estimates roll angle
        and roll-induced camber changes, then sums peak lateral force
        from all four tires (scaled by ``mu_scale``).

        When ``longitudinal_g`` is non-zero (above the 0.01 g threshold),
        the friction-ellipse model is applied: longitudinal force demand is
        distributed across the appropriate tires (rear only for acceleration,
        proportional to normal load for braking), and each tire's available
        lateral capacity is reduced by ``sqrt(1 - (Fx/Fx_peak)^2)``.

        Args:
            speed: Vehicle speed in m/s.
            curvature: Absolute path curvature in 1/m.
            mu_scale: Friction scaling factor.
            longitudinal_g: Simultaneous longitudinal acceleration in g-units.
                Positive = accelerating (rear tires), negative = braking
                (all tires, proportional to load).

        Returns:
            True if total tire lateral capacity >= required centripetal force.
        """
        # Required lateral acceleration
        a_lat = speed * speed * curvature  # m/s^2
        a_lat_g = a_lat / self.GRAVITY  # in g-units

        # Per-tire normal loads: (FL, FR, RL, RR)
        fl, fr, rl, rr = self._load_transfer.tire_loads(speed, a_lat_g, 0.0)

        # Roll angle estimation
        # Total lateral force = mass * a_lat
        total_lateral_force = self._mass_kg * a_lat
        k_total = (
            self._load_transfer.roll_stiffness_front
            + self._load_transfer.roll_stiffness_rear
        )
        roll_moment_arm = (
            self._load_transfer._cg_height_m
            - self._load_transfer._rc_at_cg
        )
        roll_angle_rad = (
            total_lateral_force * roll_moment_arm / k_total
            if k_total > 0 else 0.0
        )

        # Camber per tire due to roll
        # Roll-camber coefficients are in deg/deg; since we compute
        # roll_angle in rad and want camber change in rad, the deg/deg
        # ratio means the units cancel: camber_rad = roll_rad * (deg/deg).
        camber_change_front = roll_angle_rad * self._roll_camber_front
        camber_change_rear = roll_angle_rad * self._roll_camber_rear

        # For a right turn (positive curvature):
        #   Outside tires (FL, RL) gain load
        #   Inside tires (FR, RR) lose load
        # Outside camber = static + roll * roll_camber
        # Inside camber  = static - roll * roll_camber
        camber_fl = self._static_camber_front_rad + camber_change_front  # outside
        camber_fr = self._static_camber_front_rad - camber_change_front  # inside
        camber_rl = self._static_camber_rear_rad + camber_change_rear    # outside
        camber_rr = self._static_camber_rear_rad - camber_change_rear    # inside

        # Sum lateral capacity from all four tires
        loads = [fl, fr, rl, rr]
        cambers = [camber_fl, camber_fr, camber_rl, camber_rr]

        if abs(longitudinal_g) > 0.01:
            # Distribute longitudinal demand to tires
            f_x_total = self._mass_kg * abs(longitudinal_g) * self.GRAVITY

            if longitudinal_g > 0:
                # Traction: rear tires only
                fx_per_tire = [0.0, 0.0, f_x_total / 2.0, f_x_total / 2.0]
            else:
                # Braking: proportional to normal load
                total_fz = sum(loads)
                if total_fz > 0:
                    fx_per_tire = [f_x_total * (fz / total_fz) for fz in loads]
                else:
                    fx_per_tire = [0.0, 0.0, 0.0, 0.0]

            total_capacity = 0.0
            for fz, camber, fx in zip(loads, cambers, fx_per_tire):
                fx_peak = self._tire.peak_longitudinal_force(fz, camber)
                # Physical friction ellipse (NF-33): no 0.99 cap — when the
                # tire is at or past longitudinal peak it has zero lateral
                # capacity, not 14%.  Divide-by-zero is protected by fx_peak
                # check; overdrive (fx > fx_peak) is clipped to 1.0.
                if fx_peak > 1e-6:
                    fx_ratio = min(abs(fx) / fx_peak, 1.0)
                else:
                    fx_ratio = 1.0
                fy_peak = self._tire.peak_lateral_force(fz, camber)
                arg = max(0.0, 1.0 - fx_ratio * fx_ratio)
                fy_available = fy_peak * math.sqrt(arg)
                total_capacity += fy_available * mu_scale
        else:
            # Pure cornering (no longitudinal demand)
            total_capacity = sum(
                self._tire.peak_lateral_force(fz, cam) * mu_scale
                for fz, cam in zip(loads, cambers)
            )

        # Required centripetal force (inertial only, not including downforce)
        required_force = self._mass_kg * a_lat

        return total_capacity >= required_force
