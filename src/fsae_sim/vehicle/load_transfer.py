"""Load transfer model for FSAE vehicle dynamics.

Computes tire normal loads from static weight distribution, aerodynamic
downforce, and longitudinal/lateral load transfer using geometric and
elastic components based on roll stiffness distribution.
"""

from __future__ import annotations

import math

from fsae_sim.vehicle.vehicle import SuspensionConfig, VehicleParams

GRAVITY: float = 9.81
AIR_DENSITY: float = 1.225


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
        weight_dist_front: float = 0.45,
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

        # Roll axis height at CG (linear interpolation along wheelbase)
        # CG position from front axle = (1 - weight_dist_front) * wheelbase
        dist_cg_from_front = (1.0 - weight_dist_front) * vehicle.wheelbase_m
        self._rc_at_cg: float = (
            self.rc_front
            + (self.rc_rear - self.rc_front) * dist_cg_from_front / vehicle.wheelbase_m
        )

    def static_loads(self) -> tuple[float, float, float, float]:
        """Return static tire loads from weight distribution.

        Assumes 50/50 left-right split on each axle (level ground).

        Returns:
            (FL, FR, RL, RR) normal loads in Newtons.
        """
        weight = self._vehicle.mass_kg * GRAVITY
        front_axle = weight * self._weight_dist_front
        rear_axle = weight * (1.0 - self._weight_dist_front)
        fl = fr = front_axle / 2.0
        rl = rr = rear_axle / 2.0
        return (fl, fr, rl, rr)

    def aero_loads(self, speed_ms: float) -> tuple[float, float]:
        """Return aerodynamic downforce per axle.

        Uses ClA (downforce coefficient * area) from vehicle config and
        the front/rear distribution fraction.

        Args:
            speed_ms: Vehicle speed in metres per second.

        Returns:
            (delta_front, delta_rear) downforce in Newtons per axle.
        """
        dynamic_pressure = 0.5 * AIR_DENSITY * speed_ms * speed_ms
        total_downforce = dynamic_pressure * self._vehicle.downforce_coefficient
        delta_front = total_downforce * self._downforce_dist_front
        delta_rear = total_downforce * (1.0 - self._downforce_dist_front)
        return (delta_front, delta_rear)

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
        return (
            self._vehicle.mass_kg
            * accel_g
            * GRAVITY
            * self._cg_height_m
            / self._vehicle.wheelbase_m
        )

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
        abs_lat_g = abs(lateral_g)
        if abs_lat_g < 1e-12:
            return (0.0, 0.0)

        mass_front = self._vehicle.mass_kg * self._weight_dist_front
        mass_rear = self._vehicle.mass_kg * (1.0 - self._weight_dist_front)

        # Geometric (direct) transfer through roll centre
        geo_front = mass_front * abs_lat_g * GRAVITY * self.rc_front / self.front_track
        geo_rear = mass_rear * abs_lat_g * GRAVITY * self.rc_rear / self.rear_track

        # Elastic transfer through roll stiffness distribution
        total_lateral_force = self._vehicle.mass_kg * abs_lat_g * GRAVITY
        roll_arm = self._cg_height_m - self._rc_at_cg
        roll_moment = total_lateral_force * roll_arm

        elastic_front = (
            roll_moment * self.roll_stiffness_front / self._k_roll_total / self.front_track
        )
        elastic_rear = (
            roll_moment * self.roll_stiffness_rear / self._k_roll_total / self.rear_track
        )

        return (geo_front + elastic_front, geo_rear + elastic_rear)

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
        fl_s, fr_s, rl_s, rr_s = self.static_loads()

        # Aero downforce (split 50/50 left-right per axle)
        aero_f, aero_r = self.aero_loads(speed_ms)
        fl = fl_s + aero_f / 2.0
        fr = fr_s + aero_f / 2.0
        rl = rl_s + aero_r / 2.0
        rr = rr_s + aero_r / 2.0

        # Longitudinal transfer (positive = rear gains)
        delta_long = self.longitudinal_transfer(longitudinal_g)
        fl -= delta_long / 2.0
        fr -= delta_long / 2.0
        rl += delta_long / 2.0
        rr += delta_long / 2.0

        # Lateral transfer
        delta_lat_f, delta_lat_r = self.lateral_transfer(lateral_g, speed_ms)
        if lateral_g > 0:
            # Right turn: left tires gain, right tires lose
            fl += delta_lat_f
            fr -= delta_lat_f
            rl += delta_lat_r
            rr -= delta_lat_r
        else:
            # Left turn: right tires gain, left tires lose
            fl -= delta_lat_f
            fr += delta_lat_f
            rl -= delta_lat_r
            rr += delta_lat_r

        # Clamp to non-negative (tire cannot pull the ground)
        fl = max(fl, 0.0)
        fr = max(fr, 0.0)
        rl = max(rl, 0.0)
        rr = max(rr, 0.0)

        return (fl, fr, rl, rr)
