"""Vehicle dynamics force-balance model.

Computes resistive forces, cornering speed limits, and longitudinal
acceleration for a quasi-static endurance simulation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq, minimize_scalar

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

    AIR_DENSITY_KG_M3: float = 1.225  # ISA sea level, 15 C
    GRAVITY_M_S2: float = 9.81
    # Lateral grip limit for FSAE on dry asphalt (Hoosier R25B / LC0)
    _LEGACY_MAX_LATERAL_G: float = 1.3
    # Single-owner stall floor shared with sim/engine.py _MIN_SPEED_MS.
    _MIN_AVG_SPEED_MS: float = 0.5

    def __init__(
        self,
        vehicle: VehicleParams,
        tire_model: "PacejkaTireModel | None" = None,
        load_transfer: "LoadTransferModel | None" = None,
        cornering_solver: "CorneringSolver | None" = None,
        powertrain_config: "PowertrainConfig | None" = None,
        cornering_stiffness_scale: float = 1.0,
    ) -> None:
        self.vehicle = vehicle
        self.tire_model = tire_model
        self.load_transfer = load_transfer
        self.cornering_solver = cornering_solver
        # C1: scalar applied to Pacejka-derived slip angle so the TTC-on-an-8"-
        # wheel data can be bridged to real-world 10"-wheel stiffness without
        # a downstream fudge factor.  Default 1.0 preserves raw tire model.
        if cornering_stiffness_scale <= 0.0:
            raise ValueError(
                f"cornering_stiffness_scale must be > 0; got {cornering_stiffness_scale}"
            )
        self.cornering_stiffness_scale: float = cornering_stiffness_scale

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
            * self.AIR_DENSITY_KG_M3
            * self.vehicle.drag_coefficient
            * self.vehicle.frontal_area_m2
            * v * v
        )

    def downforce(self, speed_ms: float) -> float:
        """Aerodynamic downforce (N). F = 0.5 * rho * ClA * v^2."""
        v = abs(speed_ms)
        return (
            0.5
            * self.AIR_DENSITY_KG_M3
            * self.vehicle.downforce_coefficient
            * v * v
        )

    def rolling_resistance_force(self, speed_ms: float = 0.0) -> float:
        """Rolling resistance (N).  Increases with downforce."""
        normal_force = (
            self.vehicle.mass_kg * self.GRAVITY_M_S2
            + self.downforce(speed_ms)
        )
        return normal_force * self.vehicle.rolling_resistance

    def grade_force(self, grade: float) -> float:
        """Grade resistance (N).  Positive grade = uphill = positive force opposing motion.

        ``grade`` is rise/run (dimensionless).

        Returns ``m_effective * g * sin(atan(grade))`` so that dividing by
        ``m_effective`` in :meth:`acceleration` reproduces the textbook
        ``a = g * sin(theta)`` identity and energy is conserved on a closed
        loop (NF-19).  With no powertrain config, ``m_effective == mass_kg``
        and the result is identical to the bare-mass formulation.
        """
        angle = math.atan(grade)
        return self.m_effective * self.GRAVITY_M_S2 * math.sin(angle)

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
            self.vehicle.mass_kg * self.GRAVITY_M_S2 * mu_peak / alpha_peak
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

        # Use pi/2 as the upper search bound, consistent with peak_lateral_force.
        _ALPHA_MAX = math.pi / 2.0

        # Pacejka models can produce a small non-zero Fy at alpha=0 due to
        # residual camber/alignment effects.  When f_lat_needed is less than
        # that residual, both bracket endpoints have the same sign and brentq
        # would raise.  The demanded lateral force is already met at zero slip.
        fy_at_zero = abs(self.tire_model.lateral_force(0.0, normal_load))
        if f_lat_needed <= fy_at_zero:
            return 0.0

        # When demand exceeds what the tire can produce within [0, _ALPHA_MAX],
        # the tire is saturated — return the slip angle at the bracket ceiling.
        fy_at_max = abs(self.tire_model.lateral_force(_ALPHA_MAX, normal_load))
        if f_lat_needed >= fy_at_max:
            result = minimize_scalar(
                lambda a: -abs(
                    self.tire_model.lateral_force(a, normal_load)
                ),
                bounds=(0.001, _ALPHA_MAX),
                method="bounded",
            )
            return abs(result.x)

        # Fy is monotonic below peak — brentq finds the unique root
        return brentq(
            lambda a: abs(
                self.tire_model.lateral_force(a, normal_load)
            ) - f_lat_needed,
            0.0,
            _ALPHA_MAX,
            xtol=1e-4,
        )

    def _cornering_drag_pacejka(
        self, speed_ms: float, curvature: float, f_lat_total: float,
    ) -> float:
        """Cornering drag using Pacejka tire model with load transfer.

        Computes per-tire slip angles from lateral force demand distributed
        by normal load, then sums the drag component (Fy * sin(alpha))
        across all four tires.
        """
        # Lateral acceleration for load transfer
        a_lat_g = speed_ms ** 2 * abs(curvature) / self.GRAVITY_M_S2

        # Per-tire normal loads under cornering
        fl, fr, rl, rr = self.load_transfer.tire_loads(
            speed_ms, a_lat_g, 0.0,
        )
        loads = [fl, fr, rl, rr]
        total_load = sum(loads)
        if total_load < 1.0:
            return 0.0

        total_drag = 0.0
        for fz in loads:
            if fz < 1.0:
                continue
            # This tire's share of lateral force, proportional to load
            f_lat_tire = f_lat_total * (fz / total_load)
            # Find slip angle that produces this lateral force on the raw
            # Pacejka model.  Cornering-stiffness scaling (TTC-donor vs
            # real-world) applies at the tire model level: more stiffness
            # means the same Fy is produced at a smaller slip angle.
            alpha_raw = self._find_slip_angle(f_lat_tire, fz)
            alpha = alpha_raw / self.cornering_stiffness_scale
            # Drag component: lateral force projected onto velocity direction.
            # Fy is held at the demanded magnitude (the tire is assumed to
            # produce f_lat_tire regardless of the effective stiffness);
            # only the slip angle — and thus the sin(alpha) drag arm —
            # changes with the stiffness scale.
            total_drag += f_lat_tire * math.sin(alpha)

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
        longitudinal_g: float = 0.0,
    ) -> float:
        """Maximum speed (m/s) through a corner of given curvature.

        When a ``CorneringSolver`` is available, delegates to it for a
        physics-based result that accounts for load transfer, tire model,
        and roll-induced camber.  ``longitudinal_g`` is forwarded to the
        solver for friction-ellipse coupling.  When ``longitudinal_g`` is
        0.0 (the default, pure cornering), a fixed-point iteration
        estimates the cornering-drag-induced longitudinal load and feeds
        it back into the solver so the reported speed is sustainable in
        the presence of tire-sourced cornering drag (audit C6).

        Otherwise falls back to the legacy analytical formula that uses
        ``_LEGACY_MAX_LATERAL_G`` with a downforce correction.

        For a straight segment (curvature ~ 0), returns infinity.
        """
        kappa = abs(curvature)
        if kappa < 1e-6:
            return float("inf")

        # Delegate to physics-based solver when available
        if self.cornering_solver is not None:
            if longitudinal_g != 0.0:
                return self.cornering_solver.max_cornering_speed(
                    kappa, mu_scale=grip_factor, longitudinal_g=longitudinal_g,
                )
            v = self.cornering_solver.max_cornering_speed(
                kappa, mu_scale=grip_factor,
            )
            # Fixed-point iterate (C6): estimate the longitudinal load
            # imposed by cornering drag at the current candidate speed,
            # feed it back into the solver, and repeat until converged.
            # Requires tire + load-transfer models so cornering_drag is
            # physically computable.
            if self.tire_model is None or self.load_transfer is None:
                return v
            for _ in range(5):
                drag = self.cornering_drag(v, kappa)
                long_g = drag / (self.vehicle.mass_kg * self.GRAVITY_M_S2)
                if long_g < 1e-3:
                    break
                v_next = self.cornering_solver.max_cornering_speed(
                    kappa, mu_scale=grip_factor, longitudinal_g=long_g,
                )
                if abs(v_next - v) < 0.05:
                    v = v_next
                    break
                v = v_next
            return v

        # Legacy analytical formula
        mu = self._LEGACY_MAX_LATERAL_G * grip_factor
        m = self.vehicle.mass_kg
        g = self.GRAVITY_M_S2
        cl_a = self.vehicle.downforce_coefficient

        if cl_a < 1e-6:
            # No downforce: simple formula
            return math.sqrt(mu * g / kappa)

        # With downforce: (m*g + 0.5*rho*ClA*v^2)*mu = m*v^2*kappa
        # v^2 * (m*kappa - 0.5*rho*ClA*mu) = m*g*mu
        rho = self.AIR_DENSITY_KG_M3
        denom = m * kappa - 0.5 * rho * cl_a * mu
        # NF-36: a tiny positive ``denom`` produces absurd 1/eps speeds that
        # propagate silently into the envelope.  Require ``denom`` to be at
        # least 0.1% of the lateral-force coefficient ``m*kappa`` so the
        # resulting speed stays well-conditioned.
        denom_floor = max(1e-3 * m * kappa, 1e-9)
        if denom <= denom_floor:
            # Downforce dominates: effectively unlimited speed for this curvature
            return float("inf")
        v_sq = m * g * mu / denom
        return math.sqrt(v_sq)

    # ------------------------------------------------------------------
    # Tire-limited traction / braking
    # ------------------------------------------------------------------

    def max_traction_force(self, speed_ms: float) -> float:
        """Maximum drive force (N) from rear tires.

        When tire and load-transfer models are available, iterates
        ``a -> tire_loads -> F_x -> a`` to a fixed point (NF-6): the load
        transfer that creates rear downforce is itself driven by the
        longitudinal acceleration the rear tires can sustain, so a single
        guess of 0.3 g is self-referential.

        In legacy mode (no tire/load-transfer models), returns infinity
        so that the powertrain limit is the only constraint.
        """
        if self.tire_model is None or self.load_transfer is None:
            return float("inf")
        mg = self.vehicle.mass_kg * self.GRAVITY_M_S2
        a_g = 0.3  # initial guess
        f_last = 0.0
        for _ in range(6):
            _, _, rl, rr = self.load_transfer.tire_loads(speed_ms, 0.0, a_g)
            f_x = (
                self.tire_model.peak_longitudinal_force(rl)
                + self.tire_model.peak_longitudinal_force(rr)
            )
            a_next = f_x / mg
            if abs(a_next - a_g) < 0.005:
                return f_x
            a_g = a_next
            f_last = f_x
        return f_last

    def max_braking_force(self, speed_ms: float) -> float:
        """Maximum braking force (N) from all four tires.

        When tire and load-transfer models are available, iterates
        ``a -> tire_loads -> F_x -> a`` to a fixed point (NF-6).  The
        hardcoded -1.0 g assumption in the previous implementation
        under-predicted cap at mild brake and over-predicted at hard
        brake.

        In legacy mode (no tire/load-transfer models), returns infinity
        so that there is no tire-limited braking constraint.
        """
        if self.tire_model is None or self.load_transfer is None:
            return float("inf")
        mg = self.vehicle.mass_kg * self.GRAVITY_M_S2
        a_g = -1.0  # initial guess
        f_last = 0.0
        for _ in range(6):
            fl, fr, rl, rr = self.load_transfer.tire_loads(speed_ms, 0.0, a_g)
            f_x = sum(
                self.tire_model.peak_longitudinal_force(f)
                for f in (fl, fr, rl, rr)
            )
            a_next = -f_x / mg
            if abs(a_next - a_g) < 0.005:
                return f_x
            a_g = a_next
            f_last = f_x
        return f_last

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

        # Segment time: use average speed for the segment.
        # Floor at 0.5 m/s so this stays consistent with the engine's
        # _MIN_SPEED_MS and the invariant
        # ``segment_time * avg_speed == segment_length`` holds with a single
        # owner of the stall floor (audit C9).
        avg_speed = (entry_speed_ms + exit_speed) / 2.0
        if avg_speed < self._MIN_AVG_SPEED_MS:
            avg_speed = self._MIN_AVG_SPEED_MS
        segment_time = segment_length_m / avg_speed

        return exit_speed, segment_time
