# src/fsae_sim/dynamics6dof/params.py
from __future__ import annotations

from dataclasses import dataclass

GRAVITY = 9.81  # m/s^2


@dataclass(frozen=True)
class Dynamics6DofParams:
    """Parameters consumed by the kart-6dof port.

    Values not present in DSS are estimated conservatively and flagged in
    docstrings. Override via YAML by authoring configs/ct16ev_dynamics6dof.yaml
    (added in a later task).
    """

    mass_kg: float
    wheelbase_m: float
    cg_height_m: float
    weight_dist_front: float  # 0..1, fraction of static mass on front axle
    track_front_m: float
    track_rear_m: float

    # Inertia tensor (kg m^2) about CG, body frame axes x,y,z
    ixx_kgm2: float
    iyy_kgm2: float
    izz_kgm2: float
    ixz_kgm2: float

    # Aerodynamics — CdA and ClA absorb the area; keep cd/cl with area=1 m^2
    rho_air_kgpm3: float
    cd_a_m2: float  # drag coefficient * area
    cl_a_m2: float  # lift coefficient * area (positive = downforce when applied with proper sign)

    # Suspension stiffnesses (N/m)
    k_chassis_front_npm: float
    k_chassis_rear_npm: float
    k_antiroll_front_npm: float
    k_antiroll_rear_npm: float
    c_chassis_front_nspm: float
    c_chassis_rear_nspm: float

    # Tire vertical stiffness/damping (from .tir + estimate)
    k_tire_radial_npm: float
    c_tire_radial_nspm: float
    tire_unloaded_radius_m: float

    # Drivetrain
    rear_axle_inertia_kgm2: float
    final_drive: float

    @property
    def static_sym_displ_front_m(self) -> float:
        """Axle-level symmetric suspension displacement that produces front
        static Fz at z=0. Derived from the series-spring balance:

            k_tire * w_static = weight_front_per_corner
            w_static = k_chassis/(k_chassis + k_tire) * displ_sym
            => displ_sym = weight_front_per_corner * (k_chassis + k_tire)
                           / (k_tire * k_chassis)
        """
        w_front_per_corner = 0.5 * self.mass_kg * GRAVITY * self.weight_dist_front
        return w_front_per_corner * (self.k_chassis_front_npm + self.k_tire_radial_npm) \
            / (self.k_tire_radial_npm * self.k_chassis_front_npm)

    @property
    def static_sym_displ_rear_m(self) -> float:
        w_rear_per_corner = 0.5 * self.mass_kg * GRAVITY * (1.0 - self.weight_dist_front)
        return w_rear_per_corner * (self.k_chassis_rear_npm + self.k_tire_radial_npm) \
            / (self.k_tire_radial_npm * self.k_chassis_rear_npm)

    @classmethod
    def ct16ev_defaults(cls) -> "Dynamics6DofParams":
        # CT-16EV baseline. Values from DSS where available, otherwise estimates
        # consistent with an FSAE-class car. Explicit numbers live here so the
        # port can be tested before YAML integration lands in Task 13.
        return cls(
            mass_kg=288.0,
            wheelbase_m=1.549,
            cg_height_m=0.2794,
            weight_dist_front=0.47,
            track_front_m=1.194,
            track_rear_m=1.168,
            ixx_kgm2=35.0,
            iyy_kgm2=80.0,
            izz_kgm2=95.0,
            ixz_kgm2=0.0,
            rho_air_kgpm3=1.225,
            cd_a_m2=1.50,
            cl_a_m2=2.18,
            k_chassis_front_npm=40000.0,
            k_chassis_rear_npm=40000.0,
            k_antiroll_front_npm=13636.0,  # 238 N m/deg = ~13.6 kN/m equivalent wheel-rate
            k_antiroll_rear_npm=14780.0,
            c_chassis_front_nspm=1500.0,
            c_chassis_rear_nspm=1500.0,
            k_tire_radial_npm=150000.0,
            c_tire_radial_nspm=150.0,
            tire_unloaded_radius_m=0.2042,
            rear_axle_inertia_kgm2=0.3,
            final_drive=3.6363,
        )
