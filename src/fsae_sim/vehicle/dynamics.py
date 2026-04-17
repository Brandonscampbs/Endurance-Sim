"""Vehicle dynamics force-balance model.

Computes resistive forces, cornering speed limits, and longitudinal
acceleration for a quasi-static endurance simulation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq, minimize_scalar

from fsae_sim.physics_constants import AIR_DENSITY_KG_M3, GRAVITY_M_S2
from fsae_sim.vehicle.vehicle import VehicleParams

if TYPE_CHECKING:
    from fsae_sim.vehicle.cornering_solver import CorneringSolver
    from fsae_sim.vehicle.load_transfer import LoadTransferModel
    from fsae_sim.vehicle.tire_model import PacejkaTireModel
    from fsae_sim.vehicle.powertrain import PowertrainConfig


class VehicleDynamics:
    """Longitudinal and lateral force-balance model.

    All forces follow the convention: positive = in the direction of travel
    (or resistance to it, returned as positive magnitude).

    When ``tire_model``, ``load_transfer``, and ``cornering_solver`` are
    provided, the model delegates cornering speed limits and
    traction/braking force limits to the physics-based subsystems.
    When they are ``None`` (the default), legacy analytical formulas are
    used, preserving backward compatibility.
    """

    # Lateral grip limit for FSAE on dry asphalt (Hoosier R25B / LC0)
    _LEGACY_MAX_LATERAL_G: float = 1.3

    def __init__(
        self,
        vehicle: VehicleParams,
        tire_model: "PacejkaTireModel | None" = None,
        load_transfer: "LoadTransferModel | None" = None,
        cornering_solver: "CorneringSolver | None" = None,
        powertrain_config: "PowertrainConfig | None" = None,
        cornering_stiffness_scale: float = 1.0,
    ) -> None:
        """Construct a VehicleDynamics model.

        Args:
            vehicle: Vehicle parameter bundle.
            tire_model: Optional Pacejka tire model.  Required for the
                physics-based cornering-drag path; falls back to the
                small-angle analytical model when None.
            load_transfer: Optional LoadTransferModel; paired with
                ``tire_model`` for per-tire cornering drag and
                traction/braking limits.
            cornering_solver: Optional solver used for
                ``max_cornering_speed``.
            powertrain_config: Optional powertrain config used to compute
                ``m_effective`` (bare mass + rotational inertia of motor,
                gears and wheels).
            cornering_stiffness_scale: Multiplicative factor applied to
                the Pacejka cornering stiffness when computing cornering
                drag.  Captures the gap between TTC lab cornering
                stiffness and real-world on-car stiffness (surface
                texture, temperature, combined loading, compliance
                steer).  ``1.0`` is TTC nominal; values < 1 mean the
                on-car tire takes more slip angle for a given lateral
                force, producing more cornering drag.  This is the
                correct home for the TTC-vs-track calibration: applied
                at the slip-angle level, not downstream on the drag
                force.
        """
        self.vehicle = vehicle
        self.tire_model = tire_model
        self.load_transfer = load_transfer
        self.cornering_solver = cornering_solver
        self.cornering_stiffness_scale = float(cornering_stiffness_scale)

        # Effective mass: bare mass + rotational inertia of spinning components
        if powertrain_config is not None:
            tire_radius = 0.2042  # m, Hoosier 16x7.5-10 UNLOADED_RADIUS from .tir
            G = powertrain_config.gear_ratio
            eta = powertrain_config.drivetrain_efficiency
            j_eff = (
                vehicle.rotor_inertia_kg_m2 * G * G * eta
                + 4 * vehicle.wheel_inertia_kg_m2
            )
            self.m_effective: float = vehicle.mass_kg + j_eff / (tire_radius * tire_radius)
        else:
            self.m_effective = vehicle.mass_kg

    # ------------------------------------------------------------------
    # Resistance forces  (all return positive magnitudes)
    # ------------------------------------------------------------------

    def drag_force(self, speed_ms: float) -> float:
        """Aerodynamic drag (N).  F = 0.5 * rho * Cd * A * v^2."""
        v = abs(speed_ms)
        return (
            0.5
            * AIR_DENSITY_KG_M3
            * self.vehicle.drag_coefficient
            * self.vehicle.frontal_area_m2
            * v * v
        )

    def downforce(self, speed_ms: float) -> float:
        """Aerodynamic downforce (N). F = 0.5 * rho * ClA * v^2."""
        v = abs(speed_ms)
        return (
            0.5
            * AIR_DENSITY_KG_M3
            * self.vehicle.downforce_coefficient
            * v * v
        )

    def rolling_resistance_force(self, speed_ms: float = 0.0) -> float:
        """Rolling resistance (N).  Increases with downforce."""
        normal_force = (
            self.vehicle.mass_kg * GRAVITY_M_S2
            + self.downforce(speed_ms)
        )
        return normal_force * self.vehicle.rolling_resistance

    def grade_force(self, grade: float) -> float:
        """Grade resistance (N).  Positive grade = uphill = positive force opposing motion.

        ``grade`` is rise/run (dimensionless).
        """
        # sin(atan(grade)) for small grades ≈ grade, but exact is better
        angle = math.atan(grade)
        return self.vehicle.mass_kg * GRAVITY_M_S2 * math.sin(angle)

    def cornering_drag(self, speed_ms: float, curvature: float) -> float:
        """Drag force (N) from tire slip angles during cornering.

        When the car corners, tires operate at slip angles that create a
        longitudinal drag component. Uses the Pacejka tire model when
        available, otherwise falls back to a small-angle analytical
        approximation.

        Args:
            speed_ms: Vehicle speed (m/s).
            curvature: Path curvature (1/m). 0 = straight.

        Returns:
            Cornering drag force (N), always >= 0.
        """
        if abs(curvature) < 1e-6 or speed_ms < 0.5:
            return 0.0

        # Total lateral force needed for the turn
        f_lat_total = self.vehicle.mass_kg * speed_ms ** 2 * abs(curvature)

        if (
            self.tire_model is not None
            and self.load_transfer is not None
        ):
            return self._cornering_drag_pacejka(speed_ms, curvature, f_lat_total)

        return self._cornering_drag_analytical(f_lat_total)

    def _cornering_drag_analytical(self, f_lat_total: float) -> float:
        """Analytical cornering drag using small-angle approximation.

        Assumes linear tire: Fy = C_alpha * alpha, so alpha = Fy/C_alpha,
        and drag = Fy * sin(alpha) ~ Fy * alpha = Fy^2 / C_alpha.

        C_alpha estimated from peak grip (mu=1.5) and typical FSAE peak
        slip angle (~0.15 rad).
        """
        mu_peak = 1.5
        alpha_peak = 0.15  # rad, typical FSAE tire
        c_alpha_total = (
            self.vehicle.mass_kg * GRAVITY_M_S2 * mu_peak / alpha_peak
        )
        return f_lat_total ** 2 / c_alpha_total

    def _find_slip_angle(
        self, f_lat_needed: float, normal_load: float,
    ) -> float:
        """Find slip angle (rad) that produces the needed lateral force.

        Uses brentq root-finding on the Pacejka lateral_force function,
        which is monotonic below peak slip angle. If demanded force
        exceeds the tire's peak, returns the peak slip angle (tire
        saturated).

        Args:
            f_lat_needed: Required lateral force magnitude (N).
            normal_load: Tire normal load (N).

        Returns:
            Slip angle in radians (always >= 0).
        """
        if normal_load < 1.0 or f_lat_needed < 1.0:
            return 0.0

        # Pacejka Fy is non-monotonic past its peak: it rises, hits
        # peak_Fy at alpha_peak, then decays.  Brentq over a non-monotonic
        # interval can pick either the pre-peak or post-peak root; we
        # always want the pre-peak (smaller) solution.  Approach:
        #   1. Locate alpha_peak numerically within [0.001, 0.5 rad].
        #   2. If demand exceeds peak_Fy, tire is saturated — return
        #      alpha_peak (the physically correct "stuck" slip angle).
        #   3. Otherwise Fy is monotonic in [0, alpha_peak]; brentq finds
        #      the unique pre-peak root there.
        # This replaces the earlier check `f_lat_needed >= Fy(pi/2)`,
        # which mistakenly classified demands between Fy(pi/2) (post-peak
        # degraded) and peak_Fy as saturated.
        _ALPHA_SEARCH_MAX = 0.5  # ~29 deg — past any physical peak

        # Pacejka residual at alpha=0 (from SVy / camber); demands below
        # this are already satisfied at zero slip.
        fy_at_zero = abs(self.tire_model.lateral_force(0.0, normal_load))
        if f_lat_needed <= fy_at_zero:
            return 0.0

        result = minimize_scalar(
            lambda a: -abs(self.tire_model.lateral_force(a, normal_load)),
            bounds=(0.001, _ALPHA_SEARCH_MAX),
            method="bounded",
        )
        alpha_peak = abs(result.x)
        peak_fy = abs(self.tire_model.lateral_force(alpha_peak, normal_load))

        if f_lat_needed >= peak_fy:
            return alpha_peak  # saturated — no pre-peak solution exists

        return brentq(
            lambda a: abs(
                self.tire_model.lateral_force(a, normal_load)
            ) - f_lat_needed,
            0.0,
            alpha_peak,
            xtol=1e-4,
        )

    def _cornering_drag_pacejka(
        self, speed_ms: float, curvature: float, f_lat_total: float,
    ) -> float:
        """Cornering drag from per-tire Pacejka slip angles.

        For each wheel:
            1. Query the load-transfer model for the tire normal load at
               the current lateral demand ``a_lat_g`` and zero
               longitudinal demand.
            2. Distribute the required lateral force across the four
               tires proportional to normal load (simple steady-state
               share; combined-slip is handled separately).
            3. Solve for the slip angle that produces the per-tire
               lateral-force share.
            4. Project the resulting lateral force onto the velocity
               direction:  drag = |Fy| * sin(alpha).

        The per-tire slip angle is scaled by
        ``1.0 / cornering_stiffness_scale`` to model the gap between
        TTC lab stiffness and real-world on-car stiffness.  When
        ``cornering_stiffness_scale == 1.0`` the stock Pacejka result is
        returned unchanged.  This is the physics-faithful location for
        that calibration: it changes the operating point of every tire,
        every corner, rather than multiplying the final drag by a
        constant.  No power-law or scalar fudge factor exists anywhere
        in the path.
        """
        if f_lat_total < 1.0:
            return 0.0

        # Lateral acceleration for load transfer
        a_lat_g = speed_ms ** 2 * abs(curvature) / GRAVITY_M_S2

        # Per-tire normal loads under cornering
        fl, fr, rl, rr = self.load_transfer.tire_loads(
            speed_ms, a_lat_g, 0.0,
        )
        loads = [fl, fr, rl, rr]
        total_load = sum(loads)
        if total_load < 1.0:
            return 0.0

        # Stiffness-scale factor (<= 1 means tire needs more slip than TTC)
        scale = self.cornering_stiffness_scale
        if scale <= 0.0:
            scale = 1.0
        alpha_scale = 1.0 / scale

        total_drag = 0.0
        for fz in loads:
            if fz < 1.0:
                continue
            # This tire's share of lateral force, proportional to load
            f_lat_tire = f_lat_total * (fz / total_load)
            # Find the slip angle that produces this lateral force in
            # the TTC-calibrated Pacejka model.
            alpha_ttc = self._find_slip_angle(f_lat_tire, fz)
            # Real-world effective slip angle: lower stiffness => more
            # slip to produce the same lateral force.
            alpha_eff = min(alpha_ttc * alpha_scale, math.pi / 2.0)
            # Drag component: lateral force projected onto velocity direction.
            # Fy magnitude is preserved — only the operating slip angle
            # grows — so the extra cornering drag comes from sin(alpha).
            total_drag += f_lat_tire * math.sin(alpha_eff)

        return total_drag

    # Constant mechanical parasitic drag (N): drivetrain bearings, chain
    # friction, brake pad drag, motor cogging/windage.  Back-derived from
    # direction-averaged straight-line coasting telemetry at Michigan 2025.
    _PARASITIC_DRAG_N: float = 70.0

    def parasitic_drag(self) -> float:
        """Mechanical parasitic drag (N) from drivetrain and bearings."""
        return self._PARASITIC_DRAG_N

    def total_resistance(
        self, speed_ms: float, grade: float = 0.0, curvature: float = 0.0,
    ) -> float:
        """Sum of all resistance forces (N) at given speed, grade, and curvature."""
        return (
            self.drag_force(speed_ms)
            + self.rolling_resistance_force(speed_ms)
            + self.grade_force(grade)
            + self.cornering_drag(speed_ms, curvature)
            + self.parasitic_drag()
        )

    # ------------------------------------------------------------------
    # Cornering speed limit
    # ------------------------------------------------------------------

    def max_cornering_speed(
        self, curvature: float, grip_factor: float = 1.0,
    ) -> float:
        """Maximum speed (m/s) through a corner of given curvature.

        When a ``CorneringSolver`` is available, delegates to it for a
        physics-based result that accounts for load transfer, tire model,
        and roll-induced camber.

        Otherwise falls back to the legacy analytical formula that uses
        ``_LEGACY_MAX_LATERAL_G`` with a downforce correction.

        For a straight segment (curvature ~ 0), returns infinity.
        """
        kappa = abs(curvature)
        if kappa < 1e-6:
            return float("inf")

        # Delegate to physics-based solver when available
        if self.cornering_solver is not None:
            return self.cornering_solver.max_cornering_speed(
                kappa, mu_scale=grip_factor,
            )

        # Legacy analytical formula
        mu = self._LEGACY_MAX_LATERAL_G * grip_factor
        m = self.vehicle.mass_kg
        g = GRAVITY_M_S2
        cl_a = self.vehicle.downforce_coefficient

        if cl_a < 1e-6:
            # No downforce: simple formula
            return math.sqrt(mu * g / kappa)

        # With downforce: (m*g + 0.5*rho*ClA*v^2)*mu = m*v^2*kappa
        # v^2 * (m*kappa - 0.5*rho*ClA*mu) = m*g*mu
        rho = AIR_DENSITY_KG_M3
        denom = m * kappa - 0.5 * rho * cl_a * mu
        if denom <= 0:
            # Downforce dominates: effectively unlimited speed for this curvature
            return float("inf")
        v_sq = m * g * mu / denom
        return math.sqrt(v_sq)

    # ------------------------------------------------------------------
    # Tire-limited traction / braking
    # ------------------------------------------------------------------

    def max_traction_force(self, speed_ms: float) -> float:
        """Maximum drive force (N) from rear tires.

        NF-6: solve the traction-limit / load-transfer fixed point
        self-consistently.  Rear-tire peak Fx depends on rear load,
        which depends on longitudinal acceleration, which depends on
        the drive force — i.e. a = F_drive / (m·g).  Iterate until
        the acceleration implied by the returned force matches the
        acceleration used to compute the load transfer.

        In legacy mode (no tire/load-transfer models), returns infinity
        so that the powertrain limit is the only constraint.
        """
        if self.tire_model is None or self.load_transfer is None:
            return float("inf")

        mg = self.vehicle.mass_kg * GRAVITY_M_S2
        long_g = 0.3  # initial guess
        for _ in range(8):  # converges in 2-3 iters for physical values
            _, _, rl, rr = self.load_transfer.tire_loads(speed_ms, 0.0, long_g)
            f_drive = (
                self.tire_model.peak_longitudinal_force(rl)
                + self.tire_model.peak_longitudinal_force(rr)
            )
            long_g_new = f_drive / mg if mg > 0 else long_g
            if abs(long_g_new - long_g) < 1e-3:
                break
            long_g = long_g_new
        return f_drive

    def max_braking_force(self, speed_ms: float) -> float:
        """Maximum braking force (N) from all four tires.

        NF-6: self-consistent fixed point on decel-induced load transfer,
        mirroring max_traction_force but over all four tires.  Decel is
        always negative in the load-transfer sign convention (weight
        shifts forward under braking).

        In legacy mode (no tire/load-transfer models), returns infinity
        so that there is no tire-limited braking constraint.
        """
        if self.tire_model is None or self.load_transfer is None:
            return float("inf")

        mg = self.vehicle.mass_kg * GRAVITY_M_S2
        long_g = -1.0  # initial guess (hard braking)
        for _ in range(8):
            fl, fr, rl, rr = self.load_transfer.tire_loads(speed_ms, 0.0, long_g)
            f_brake = sum(
                self.tire_model.peak_longitudinal_force(f)
                for f in [fl, fr, rl, rr]
            )
            long_g_new = -f_brake / mg if mg > 0 else long_g
            if abs(long_g_new - long_g) < 1e-3:
                break
            long_g = long_g_new
        return f_brake

    # ------------------------------------------------------------------
    # Longitudinal acceleration
    # ------------------------------------------------------------------

    def acceleration(
        self, net_force_n: float,
    ) -> float:
        """Longitudinal acceleration (m/s^2) from net force.

        Uses ``m_effective`` which includes rotational inertia of motor,
        gears, and wheels when a powertrain config is provided.

        ``net_force_n`` is drive_force - total_resistance (positive = accelerating).
        """
        return net_force_n / self.m_effective

    def resolve_exit_speed(
        self,
        entry_speed_ms: float,
        segment_length_m: float,
        net_force_n: float,
        corner_speed_limit_ms: float,
    ) -> tuple[float, float]:
        """Compute segment exit speed and time from entry conditions.

        Uses constant-acceleration kinematics over the segment:
            v_exit^2 = v_entry^2 + 2 * a * d

        The exit speed is clamped to the corner speed limit.

        Returns:
            (exit_speed_ms, segment_time_s)
        """
        a = self.acceleration(net_force_n)

        # v^2 = v0^2 + 2*a*d
        v_sq = entry_speed_ms ** 2 + 2.0 * a * segment_length_m
        if v_sq < 0:
            # Car cannot make it through the segment (stalls)
            v_sq = 0.0

        exit_speed = math.sqrt(v_sq)

        # Clamp to cornering limit
        exit_speed = min(exit_speed, corner_speed_limit_ms)

        # Segment time: use average speed for the segment
        avg_speed = (entry_speed_ms + exit_speed) / 2.0
        if avg_speed < 0.1:
            # Near-zero speed: avoid division by zero, use small speed
            avg_speed = 0.1
        segment_time = segment_length_m / avg_speed

        return exit_speed, segment_time
