"""Load transfer model for FSAE vehicle dynamics.

Computes tire normal loads from static weight distribution, aerodynamic
downforce, and longitudinal/lateral load transfer using geometric and
elastic components based on roll stiffness distribution.
"""

from __future__ import annotations

import math

from fsae_sim.physics_constants import AIR_DENSITY_KG_M3 as AIR_DENSITY
from fsae_sim.physics_constants import GRAVITY_M_S2 as GRAVITY
from fsae_sim.vehicle.vehicle import SuspensionConfig, VehicleParams


def _redistribute_same_axle(
    left_load: float, right_load: float
) -> tuple[float, float]:
    """Clamp per-tire loads to >= 0 and push any negative share to the
    opposite tire on the same axle.

    The sum ``left_load + right_load`` is preserved (up to the case where
    both tires would go negative — physically unreachable under any
    finite lateral / longitudinal acceleration that can actually happen
    on a real car, but handled gracefully by clamping both to zero).

    Returns:
        A ``(left, right)`` pair where both values are >= 0 and
        ``left + right == max(0, left_load + right_load)``.
    """
    axle_total = left_load + right_load
    if axle_total <= 0.0:
        return (0.0, 0.0)
    if left_load < 0.0:
        return (0.0, axle_total)
    if right_load < 0.0:
        return (axle_total, 0.0)
    return (left_load, right_load)


class LoadTransferModel:
    """Calculates per-tire normal loads under combined loading.

    Combines static weight, aerodynamic downforce, longitudinal load transfer
    (acceleration/braking), and lateral load transfer (cornering) decomposed
    into geometric and elastic components.

    Tire ordering convention: (FL, FR, RL, RR).
    Sign conventions:
        - Positive longitudinal_g = forward acceleration (rear loads increase)
        - Positive lateral_g = rightward turn (left tires gain load)

    Args:
        vehicle: Vehicle parameters (mass, wheelbase, downforce coeff).
        suspension: Suspension geometry config (track widths, roll centers,
            roll stiffness).
        cg_height_m: Centre of gravity height in metres.
        weight_dist_front: Static front weight distribution as fraction (0-1).
        downforce_dist_front: Aero downforce front distribution as fraction.
    """

    def __init__(
        self,
        vehicle: VehicleParams,
        suspension: SuspensionConfig,
        cg_height_m: float = 0.2794,
        weight_dist_front: float = 0.53,
        downforce_dist_front: float = 0.61,
    ) -> None:
        self._vehicle = vehicle
        self._suspension = suspension
        self._cg_height_m = cg_height_m
        self._weight_dist_front = weight_dist_front
        self._downforce_dist_front = downforce_dist_front

        # Convert track widths from mm to m
        self.front_track: float = suspension.front_track_mm / 1000.0
        self.rear_track: float = suspension.rear_track_mm / 1000.0
        # NF-40: validate track widths before first division
        if self.front_track <= 0.001 or self.rear_track <= 0.001:
            raise ValueError(
                f"SuspensionConfig track widths must be > 0.001 m "
                f"(got front_track={self.front_track:.4f} m, "
                f"rear_track={self.rear_track:.4f} m)."
            )

        # Convert roll centre heights from mm to m
        self.rc_front: float = suspension.roll_center_height_front_mm / 1000.0
        self.rc_rear: float = suspension.roll_center_height_rear_mm / 1000.0

        # Convert roll stiffness from Nm/deg to Nm/rad (exposed for cornering solver)
        self.roll_stiffness_front: float = (
            suspension.roll_stiffness_front_nm_per_deg * 180.0 / math.pi
        )
        self.roll_stiffness_rear: float = (
            suspension.roll_stiffness_rear_nm_per_deg * 180.0 / math.pi
        )
        self._k_roll_total: float = self.roll_stiffness_front + self.roll_stiffness_rear
        if (
            self.roll_stiffness_front < 0.0
            or self.roll_stiffness_rear < 0.0
            or self._k_roll_total <= 0.0
        ):
            raise ValueError(
                "SuspensionConfig roll stiffness values must be non-negative "
                "and sum to a positive value "
                f"(front={suspension.roll_stiffness_front_nm_per_deg!r} "
                f"Nm/deg, rear={suspension.roll_stiffness_rear_nm_per_deg!r} "
                "Nm/deg)."
            )

        # Roll axis height at CG (linear interpolation along wheelbase)
        # CG position from front axle = (1 - weight_dist_front) * wheelbase
        dist_cg_from_front = (1.0 - weight_dist_front) * vehicle.wheelbase_m
        self._rc_at_cg: float = (
            self.rc_front
            + (self.rc_rear - self.rc_front) * dist_cg_from_front / vehicle.wheelbase_m
        )

        # Hot-path constants. ``tire_loads`` is called >4M times on a
        # 22-lap replay, so collapsing the per-call arithmetic to a few
        # multiplies removes meaningful overhead without touching physics.
        # Air density is read once (env config is immutable) and the
        # full coefficient chain for each output term is folded.
        env = getattr(vehicle, "environment", None)
        rho = AIR_DENSITY if env is None else env.air_density_kg_m3
        self._air_density_kg_m3: float = rho

        mass = float(vehicle.mass_kg)
        mg = mass * GRAVITY
        self._mg = mg
        self._mass_kg = mass

        # Static loads — fully constant, never changes during a sim.
        front_axle = mg * weight_dist_front
        rear_axle = mg * (1.0 - weight_dist_front)
        fl_s = fr_s = front_axle / 2.0
        rl_s = rr_s = rear_axle / 2.0
        self._static_loads_tuple: tuple[float, float, float, float] = (
            fl_s, fr_s, rl_s, rr_s,
        )

        # Aero downforce per axle.
        # ``delta_axle = 0.5 * rho * cl_a * dist * v²``. Fold the
        # constants so per-call cost is one multiply + (v*v).
        half_rho_cla = 0.5 * rho * vehicle.downforce_coefficient
        self._half_rho_cla = half_rho_cla
        self._half_rho_cla_front = half_rho_cla * downforce_dist_front
        self._half_rho_cla_rear = half_rho_cla * (1.0 - downforce_dist_front)
        # Per-tire (split 50/50 left-right per axle) — used directly by
        # ``tire_loads`` so we save a divide-by-two per per-tire add.
        self._half_rho_cla_front_per_tire = self._half_rho_cla_front / 2.0
        self._half_rho_cla_rear_per_tire = self._half_rho_cla_rear / 2.0

        # Longitudinal transfer: ``delta_long = mass * g * cg_h / wheelbase * accel_g``.
        # Pre-fold the constant prefix; per-call this is a single multiply.
        self._long_transfer_coeff = (
            mg * cg_height_m / vehicle.wheelbase_m
        )
        # Halved version (half goes to each tire on an axle).
        self._long_transfer_coeff_per_tire = self._long_transfer_coeff / 2.0

        # Lateral transfer: geometric + elastic = K_axle × |lateral_g|.
        mass_front = mass * weight_dist_front
        mass_rear = mass * (1.0 - weight_dist_front)
        roll_arm = cg_height_m - self._rc_at_cg
        roll_moment_per_g = mg * roll_arm  # = mass * g * roll_arm

        geo_front_coeff = mass_front * GRAVITY * self.rc_front / self.front_track
        geo_rear_coeff = mass_rear * GRAVITY * self.rc_rear / self.rear_track
        elastic_front_coeff = (
            roll_moment_per_g
            * self.roll_stiffness_front
            / self._k_roll_total
            / self.front_track
        )
        elastic_rear_coeff = (
            roll_moment_per_g
            * self.roll_stiffness_rear
            / self._k_roll_total
            / self.rear_track
        )
        self._lat_transfer_coeff_front = geo_front_coeff + elastic_front_coeff
        self._lat_transfer_coeff_rear = geo_rear_coeff + elastic_rear_coeff

    def static_loads(self) -> tuple[float, float, float, float]:
        """Return static tire loads from weight distribution.

        Assumes 50/50 left-right split on each axle (level ground).

        Returns:
            (FL, FR, RL, RR) normal loads in Newtons.
        """
        return self._static_loads_tuple

    def _air_density(self) -> float:
        """Air density (kg/m^3) cached at construction.

        Reads ``vehicle.environment`` once on init (frozen dataclass) and
        falls back to ``physics_constants.AIR_DENSITY`` when absent.
        """
        return self._air_density_kg_m3

    def aero_loads(self, speed_ms: float) -> tuple[float, float]:
        """Return aerodynamic downforce per axle.

        Uses ClA (downforce coefficient * area) from vehicle config and
        the front/rear distribution fraction.

        Args:
            speed_ms: Vehicle speed in metres per second.

        Returns:
            (delta_front, delta_rear) downforce in Newtons per axle.
        """
        v_sq = speed_ms * speed_ms
        return (
            self._half_rho_cla_front * v_sq,
            self._half_rho_cla_rear * v_sq,
        )

    def longitudinal_transfer(self, accel_g: float) -> float:
        """Return longitudinal load transfer for a given acceleration.

        Positive accel_g means forward acceleration, transferring load to
        the rear axle. The returned value is the delta applied: positive
        means rear gains, front loses.

        Args:
            accel_g: Longitudinal acceleration in g-units.

        Returns:
            Load transfer in Newtons (added to rear axle, subtracted from front).
        """
        return self._long_transfer_coeff * accel_g

    def lateral_transfer(
        self, lateral_g: float, speed_ms: float
    ) -> tuple[float, float]:
        """Return lateral load transfer per axle.

        Decomposes into geometric (direct) and elastic (roll stiffness)
        components. Uses absolute value of lateral_g; sign handling is done
        in tire_loads() to assign left/right correctly.

        Args:
            lateral_g: Lateral acceleration in g-units (positive = right turn).
            speed_ms: Vehicle speed in m/s (unused here, reserved for
                future aero-dependent weight distribution).

        Returns:
            (delta_front, delta_rear) lateral load transfer magnitudes
            in Newtons.
        """
        abs_lat_g = -lateral_g if lateral_g < 0.0 else lateral_g
        if abs_lat_g < 1e-12:
            return (0.0, 0.0)
        return (
            self._lat_transfer_coeff_front * abs_lat_g,
            self._lat_transfer_coeff_rear * abs_lat_g,
        )

    def tire_loads(
        self,
        speed_ms: float,
        lateral_g: float,
        longitudinal_g: float,
    ) -> tuple[float, float, float, float]:
        """Compute combined per-tire normal loads.

        Combines static weight, aerodynamic downforce, longitudinal transfer,
        and lateral transfer into per-tire loads. Loads are clamped to >= 0
        (a tire cannot push the ground).

        Sign conventions:
            - Positive lateral_g = right turn => left tires gain load
            - Positive longitudinal_g = forward accel => rear tires gain load

        Args:
            speed_ms: Vehicle speed in metres per second.
            lateral_g: Lateral acceleration in g-units.
            longitudinal_g: Longitudinal acceleration in g-units.

        Returns:
            (FL, FR, RL, RR) normal loads in Newtons, each >= 0.
        """
        fl_s, fr_s, rl_s, rr_s = self._static_loads_tuple

        # Aero downforce (split 50/50 left-right per axle).
        v_sq = speed_ms * speed_ms
        aero_f_per_tire = self._half_rho_cla_front_per_tire * v_sq
        aero_r_per_tire = self._half_rho_cla_rear_per_tire * v_sq
        fl = fl_s + aero_f_per_tire
        fr = fr_s + aero_f_per_tire
        rl = rl_s + aero_r_per_tire
        rr = rr_s + aero_r_per_tire

        # Longitudinal transfer (positive = rear gains).  Per-tire
        # coefficient already halved at construction.
        delta_long_per_tire = self._long_transfer_coeff_per_tire * longitudinal_g
        fl -= delta_long_per_tire
        fr -= delta_long_per_tire
        rl += delta_long_per_tire
        rr += delta_long_per_tire

        # Lateral transfer.  Inline the abs() / coefficient lookup so
        # ``lateral_transfer`` only has to be called when an external
        # caller asks for the per-axle delta.
        abs_lat_g = -lateral_g if lateral_g < 0.0 else lateral_g
        if not abs_lat_g < 1e-12:
            sign_lat = 1.0 if lateral_g > 0.0 else -1.0
            delta_lat_f = self._lat_transfer_coeff_front * abs_lat_g * sign_lat
            delta_lat_r = self._lat_transfer_coeff_rear * abs_lat_g * sign_lat
            fl += delta_lat_f
            fr -= delta_lat_f
            rl += delta_lat_r
            rr -= delta_lat_r

        # Clamp to non-negative (tire cannot pull the ground) while
        # preserving vertical equilibrium: any negative portion is
        # redistributed to the same-axle opposite tire.  When an inside
        # wheel lifts, its share of vertical load has to go somewhere
        # and the only place it *can* go is the outside wheel on the
        # same axle.  Clamping without redistribution would under-predict
        # outside-tire grip and bias cornering capacity downward.
        #
        # Conservation target (up to floating-point noise):
        #     fl + fr + rl + rr == m*g + downforce
        # which must hold for any input of lateral_g / longitudinal_g,
        # even when one or more wheels would otherwise go negative.
        fl, fr = _redistribute_same_axle(fl, fr)
        rl, rr = _redistribute_same_axle(rl, rr)

        return (fl, fr, rl, rr)
