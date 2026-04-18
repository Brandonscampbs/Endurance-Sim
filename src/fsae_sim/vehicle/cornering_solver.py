"""Cornering speed solver for FSAE vehicle dynamics.

Computes the maximum sustainable cornering speed for a given path
curvature by bisecting over speed and checking whether the four tires
can collectively produce the required centripetal force.  Roll-induced
camber changes are modelled so that degressive tire models and load
transfer effects are captured accurately.

At max cornering speed the car is locally at steady speed, but the
drive tires still carry a non-zero longitudinal force to balance
aerodynamic drag + rolling resistance.  That longitudinal demand
consumes part of each drive tire's friction circle and reduces the
available lateral grip via the friction ellipse.  Ignoring it (as
an earlier version of this solver did when ``longitudinal_g=0``)
produces over-optimistic apex speeds; no single grip_scale can
simultaneously match straight-line top speed AND corner apex speeds
against telemetry when the solver does not close this loop.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fsae_sim.physics_constants import AIR_DENSITY_KG_M3, GRAVITY_M_S2

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

    GRAVITY: float = GRAVITY_M_S2  # NF-22: single source of truth
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
        cd_a_m2: float = 0.0,
        cl_a_m2: float = 0.0,
        rolling_resistance: float = 0.0,
        rear_wheel_drive: bool = True,
    ) -> None:
        self._tire = tire_model
        self._load_transfer = load_transfer
        self._mass_kg = mass_kg
        self._static_camber_front_rad = static_camber_front_rad
        self._static_camber_rear_rad = static_camber_rear_rad
        self._roll_camber_front = roll_camber_front
        self._roll_camber_rear = roll_camber_rear
        # Used to compute the self-consistent drive-tire Fx demand at max
        # cornering speed. Pass 0.0 for all three to fall back to the
        # pre-close-the-loop pure-lateral solver (over-optimistic).
        self._cd_a_m2 = float(cd_a_m2)
        self._cl_a_m2 = float(cl_a_m2)
        self._rolling_resistance = float(rolling_resistance)
        self._rear_wheel_drive = bool(rear_wheel_drive)

    def _longitudinal_g_at_speed(self, speed_ms: float) -> float:
        """Return the self-consistent longitudinal acceleration (g) required
        for the drive axle to hold ``speed_ms`` against drag + rolling.

        At max cornering speed the car is locally at steady speed so the
        drivetrain must produce exactly enough Fx to balance resistance.
        That Fx is routed through the drive tires (rear-only on CT-16EV),
        consuming part of their friction-circle and reducing available Fy.
        """
        v = max(abs(speed_ms), 0.0)
        # Aero drag
        drag_n = 0.5 * AIR_DENSITY_KG_M3 * self._cd_a_m2 * v * v
        # Rolling resistance (scales with normal load = weight + downforce)
        downforce_n = 0.5 * AIR_DENSITY_KG_M3 * self._cl_a_m2 * v * v
        normal_n = self._mass_kg * self.GRAVITY + downforce_n
        rolling_n = self._rolling_resistance * normal_n
        f_demand_n = drag_n + rolling_n
        mg = self._mass_kg * self.GRAVITY
        return f_demand_n / mg if mg > 0.0 else 0.0

    def max_cornering_speed(
        self,
        curvature: float,
        mu_scale: float = 1.0,
        longitudinal_g: float | None = None,
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
            longitudinal_g: Optional override for the longitudinal
                acceleration (g-units) to impose during the ellipse check.
                When ``None`` (default), the solver computes the
                self-consistent steady-cornering demand from aero drag and
                rolling resistance at each candidate speed — that is, the
                drive axle must produce the Fx that balances resistance at
                max cornering speed. This is the physically correct default:
                a car holding max apex speed is dragging air and its drive
                tires cannot also deliver full lateral. Pass ``0.0``
                explicitly to revert to the old "pure lateral" behavior
                (over-optimistic); pass a non-zero value to model entry
                braking or corner-exit acceleration.

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
            # Self-consistent longitudinal_g at v_mid unless explicitly
            # overridden by the caller.
            long_g = (
                longitudinal_g
                if longitudinal_g is not None
                else self._longitudinal_g_at_speed(v_mid)
            )
            if self._can_sustain(v_mid, abs_curvature, mu_scale, long_g):
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

        # Per-tire normal loads: (FL, FR, RL, RR).
        # Pass longitudinal_g through so the load transfer reflects
        # simultaneous accel/brake weight shift (front-to-rear), not
        # just pure lateral transfer.  Without this, fx/fy capacity in
        # the ellipse is computed on pure-cornering loads even under
        # trail-brake / corner-exit accel.
        fl, fr, rl, rr = self._load_transfer.tire_loads(
            speed, a_lat_g, longitudinal_g,
        )

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
                # Apply mu_scale to BOTH peaks so the ellipse axes scale
                # consistently in low-grip conditions.  Earlier code
                # scaled only fy_peak (after the sqrt), which meant
                # fx_ratio was measured against an unscaled fx_peak and
                # under-reported saturation when mu_scale < 1.
                fx_peak = (
                    self._tire.peak_longitudinal_force(fz, camber) * mu_scale
                )
                fy_peak = (
                    self._tire.peak_lateral_force(fz, camber) * mu_scale
                )
                # Physical friction ellipse (NF-33): no 0.99 cap — when the
                # tire is at or past longitudinal peak it has zero lateral
                # capacity, not 14%.  Divide-by-zero is protected by fx_peak
                # check; overdrive (fx > fx_peak) is clipped to 1.0.
                if fx_peak > 1e-6:
                    fx_ratio = min(abs(fx) / fx_peak, 1.0)
                else:
                    fx_ratio = 1.0
                arg = max(0.0, 1.0 - fx_ratio * fx_ratio)
                fy_available = fy_peak * math.sqrt(arg)
                total_capacity += fy_available
        else:
            # Pure cornering (no longitudinal demand)
            total_capacity = sum(
                self._tire.peak_lateral_force(fz, cam) * mu_scale
                for fz, cam in zip(loads, cambers)
            )

        # Required centripetal force (inertial only, not including downforce)
        required_force = self._mass_kg * a_lat

        return total_capacity >= required_force
