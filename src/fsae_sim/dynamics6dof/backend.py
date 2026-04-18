"""Backend adapter exposing the engine's VehicleDynamics API using the
dynamics6dof-ported physics (aero, static suspension, PAC02 via ISO adapter).

The engine (``fsae_sim.sim.engine``) runs a quasi-static per-segment force
balance. It only needs STEADY-STATE answers to five questions:

    - max_traction_force(v)     — peak rear-tire Fx with aero-speed Fz lift
    - max_braking_force(v)      — peak Fx over all 4 wheels during hard decel
    - total_resistance(v, ...)  — aero drag + rolling + grade + cornering drag
    - max_cornering_speed(κ)    — largest v with m·v²·κ ≤ available lateral grip
    - resolve_exit_speed(...)   — kinematic integration over one segment

This adapter computes all five from the dynamics6dof modules so the engine
can switch between the legacy ``VehicleDynamics`` backend and the ported
backend by flag, with identical call-site contracts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .aero import aero_force
from .params import GRAVITY, Dynamics6DofParams
from .tire_pac02 import PAC02Corner

if TYPE_CHECKING:
    from fsae_sim.vehicle.vehicle import VehicleConfig


@dataclass
class FastestLapDynamicsBackend:
    """Steady-state dynamics backend built from dynamics6dof physics.

    The backend keeps a PAC02 tire adapter (ISO convention) and a
    ``Dynamics6DofParams`` bundle; all force-balance quantities are
    computed analytically without running the ODE.
    """

    params: Dynamics6DofParams
    corner: PAC02Corner
    rolling_resistance: float = 0.015
    parasitic_drag_n: float = 5.0
    max_lateral_g_limit: float = 3.0  # safety cap on the solver iterations

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_vehicle_config(cls, cfg: "VehicleConfig") -> "FastestLapDynamicsBackend":
        """Build a backend from the project's ``VehicleConfig`` YAML bundle.

        Tire coefficients come from ``cfg.tire.tir_file``; vehicle mass,
        CdA, ClA, tracks, wheelbase from ``cfg.vehicle`` and suspension
        stiffnesses from ``cfg.suspension``. Parameters not in the YAML
        (per-axis inertia, radial tire/damping) fall back to the CT-16EV
        defaults in ``Dynamics6DofParams.ct16ev_defaults()``.
        """
        from pathlib import Path

        from fsae_sim.vehicle.tire_model import PacejkaTireModel

        tir_path = Path(cfg.tire.tir_file)
        model = PacejkaTireModel(tir_path)
        model.apply_grip_scale(cfg.tire.grip_scale)
        corner = PAC02Corner(model)

        defaults = Dynamics6DofParams.ct16ev_defaults()
        from dataclasses import replace

        params = replace(
            defaults,
            mass_kg=cfg.vehicle.mass_kg,
            wheelbase_m=cfg.vehicle.wheelbase_m,
            cd_a_m2=cfg.vehicle.drag_coefficient * cfg.vehicle.frontal_area_m2,
            cl_a_m2=cfg.vehicle.downforce_coefficient,
            track_front_m=cfg.suspension.front_track_mm / 1000.0,
            track_rear_m=cfg.suspension.rear_track_mm / 1000.0,
        )
        return cls(
            params=params,
            corner=corner,
            rolling_resistance=cfg.vehicle.rolling_resistance,
        )

    # ------------------------------------------------------------------
    # Vertical-load helpers
    # ------------------------------------------------------------------
    def _downforce_per_corner(self, speed_ms: float) -> float:
        """Downforce (N) on one corner from aero lift."""
        v_body = np.array([speed_ms, 0.0, 0.0])
        _, lift = aero_force(v_body, np.zeros(3), self.params)
        # lift[2] is negative in ISO for downforce (pulls car to -z); return positive.
        return float(-lift[2]) / 4.0

    def _static_corner_loads(self, speed_ms: float) -> tuple[float, float]:
        """Return (front_corner_fz, rear_corner_fz) under straight-line cruise."""
        p = self.params
        df_per_corner = self._downforce_per_corner(speed_ms)
        front = 0.5 * p.mass_kg * GRAVITY * p.weight_dist_front + df_per_corner
        rear = 0.5 * p.mass_kg * GRAVITY * (1.0 - p.weight_dist_front) + df_per_corner
        return front, rear

    def _long_accel_loads(
        self, speed_ms: float, long_g: float,
    ) -> tuple[float, float, float, float]:
        """Return (FL, FR, RL, RR) tire Fz under longitudinal acceleration long_g.

        ``long_g`` > 0 = forward acceleration (weight shifts rearward).
        ``long_g`` < 0 = braking (weight shifts forward).
        """
        p = self.params
        df_per_corner = self._downforce_per_corner(speed_ms)
        mg = p.mass_kg * GRAVITY
        # Weight transfer magnitude (split evenly left/right): ΔFz = m·a·h / L
        dfz = p.mass_kg * long_g * GRAVITY * p.cg_height_m / p.wheelbase_m
        front_corner = 0.5 * mg * p.weight_dist_front + df_per_corner - 0.5 * dfz
        rear_corner = 0.5 * mg * (1.0 - p.weight_dist_front) + df_per_corner + 0.5 * dfz
        return front_corner, front_corner, rear_corner, rear_corner

    # ------------------------------------------------------------------
    # Resistance forces
    # ------------------------------------------------------------------
    def drag_force(self, speed_ms: float) -> float:
        v = abs(speed_ms)
        v_body = np.array([v, 0.0, 0.0])
        drag, _ = aero_force(v_body, np.zeros(3), self.params)
        return float(np.linalg.norm(drag[:2]))

    def downforce(self, speed_ms: float) -> float:
        return 4.0 * self._downforce_per_corner(speed_ms)

    def rolling_resistance_force(self, speed_ms: float = 0.0) -> float:
        normal = self.params.mass_kg * GRAVITY + self.downforce(speed_ms)
        return normal * self.rolling_resistance

    def grade_force(self, grade: float) -> float:
        angle = math.atan(grade)
        return self.params.mass_kg * GRAVITY * math.sin(angle)

    def parasitic_drag(self) -> float:
        return self.parasitic_drag_n

    def cornering_drag(self, speed_ms: float, curvature: float) -> float:
        if abs(curvature) < 1e-6 or speed_ms < 0.5:
            return 0.0
        p = self.params
        # Total lateral force needed for the turn
        f_lat_required = p.mass_kg * speed_ms ** 2 * abs(curvature)
        # Static Fz per corner under speed-dependent downforce
        front_fz, rear_fz = self._static_corner_loads(speed_ms)
        # Solve for slip angle α such that 4 * Fy(α, Fz) ≈ f_lat_required,
        # then sum the longitudinal drag |Fy·sin(α)| per corner.
        alpha = self._solve_slip_angle_for_lateral_force(
            f_lat_required, front_fz, rear_fz,
        )
        # Cornering drag = sum over corners of |Fy · sin(α)|, approximately
        # |F_lat · tan(α)| since F_lat_total sums axle-wise Fy and the slip is
        # approximately the same across corners. Keep it simple and physically
        # correct to leading order:
        return abs(f_lat_required) * math.tan(max(alpha, 0.0))

    def _solve_slip_angle_for_lateral_force(
        self, f_lat_required: float, front_fz: float, rear_fz: float,
    ) -> float:
        """Iterative solver: find α such that Σ Fy(α, Fz) = f_lat_required.

        Bisection on α in [0, 0.3 rad] is fine for quasi-static cornering.
        """
        def total_fy(alpha: float) -> float:
            _, fy_f = self.corner.forces(
                slip_angle_rad=alpha, slip_ratio=0.0, fz_n=front_fz,
            )
            _, fy_r = self.corner.forces(
                slip_angle_rad=alpha, slip_ratio=0.0, fz_n=rear_fz,
            )
            return 2.0 * fy_f + 2.0 * fy_r  # two tires per axle

        lo, hi = 0.0, 0.3
        if total_fy(hi) < f_lat_required:
            return hi  # saturated — return the max slip
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if total_fy(mid) < f_lat_required:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-5:
                break
        return 0.5 * (lo + hi)

    def total_resistance(
        self, speed_ms: float, grade: float = 0.0, curvature: float = 0.0,
    ) -> float:
        return (
            self.drag_force(speed_ms)
            + self.rolling_resistance_force(speed_ms)
            + self.grade_force(grade)
            + self.cornering_drag(speed_ms, curvature)
            + self.parasitic_drag()
        )

    # ------------------------------------------------------------------
    # Tire-limited traction / braking
    # ------------------------------------------------------------------
    def max_traction_force(self, speed_ms: float) -> float:
        """Peak rear-axle Fx with self-consistent long-weight transfer."""
        mg = self.params.mass_kg * GRAVITY
        long_g = 0.3
        fx_total = 0.0
        for _ in range(8):
            _, _, rl, rr = self._long_accel_loads(speed_ms, long_g)
            fx_rl = float(self.corner.model.peak_longitudinal_force(rl))
            fx_rr = float(self.corner.model.peak_longitudinal_force(rr))
            fx_total = fx_rl + fx_rr
            new_long_g = fx_total / mg if mg > 0 else long_g
            if abs(new_long_g - long_g) < 1e-3:
                break
            long_g = new_long_g
        return fx_total

    def max_braking_force(self, speed_ms: float) -> float:
        """Peak 4-wheel Fx during braking with self-consistent decel weight transfer."""
        mg = self.params.mass_kg * GRAVITY
        long_g = -1.0
        fx_total = 0.0
        for _ in range(8):
            fl, fr, rl, rr = self._long_accel_loads(speed_ms, long_g)
            fx_total = sum(
                float(self.corner.model.peak_longitudinal_force(fz))
                for fz in (fl, fr, rl, rr)
            )
            new_long_g = -fx_total / mg if mg > 0 else long_g
            if abs(new_long_g - long_g) < 1e-3:
                break
            long_g = new_long_g
        return fx_total

    # ------------------------------------------------------------------
    # Cornering speed
    # ------------------------------------------------------------------
    def max_cornering_speed(
        self, curvature: float, grip_factor: float = 1.0,
    ) -> float:
        """Maximum sustainable cornering speed through ``curvature`` (1/m).

        Physics: at max corner speed the car holds steady v, so the drive
        axle must produce Fx equal to aero drag + rolling resistance. That
        drive-tire Fx consumes part of the rear-tire friction circle; we
        apply the standard elliptic reduction

            Fy_available = Fy_peak · sqrt(1 - (Fx/Fx_peak)²)

        per drive tire. Front (non-drive) tires retain full lateral peak.
        Without this reduction the solver is over-optimistic and no single
        grip_factor can simultaneously match straight-line and apex speeds
        against telemetry.
        """
        kappa = abs(curvature)
        if kappa < 1e-6:
            return float("inf")

        def lateral_excess(v: float) -> float:
            front_fz, rear_fz = self._static_corner_loads(v)
            # Non-drive (front) tires: pure lateral peak.
            fy_front = 2.0 * float(
                self.corner.model.peak_lateral_force(front_fz)
            )
            # Drive (rear) tires: elliptic reduction by Fx demand.
            fx_demand_total = self.drag_force(v) + self.rolling_resistance_force(v)
            fx_per_rear = 0.5 * fx_demand_total  # RWD, split evenly
            fx_peak_rear = float(
                self.corner.model.peak_longitudinal_force(rear_fz)
            )
            fy_peak_rear = float(
                self.corner.model.peak_lateral_force(rear_fz)
            )
            if fx_peak_rear > 1e-6:
                fx_ratio = min(fx_per_rear / fx_peak_rear, 1.0)
            else:
                fx_ratio = 1.0
            fy_per_rear = fy_peak_rear * math.sqrt(max(0.0, 1.0 - fx_ratio * fx_ratio))
            fy_rear = 2.0 * fy_per_rear
            peak_fy_total = (fy_front + fy_rear) * grip_factor
            required = self.params.mass_kg * v * v * kappa
            return peak_fy_total - required

        lo, hi = 0.1, math.sqrt(self.max_lateral_g_limit * GRAVITY / kappa) * 2.0
        if lateral_excess(lo) < 0:
            return 0.0
        if lateral_excess(hi) > 0:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if lateral_excess(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.01:
                break
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def acceleration(self, net_force_n: float) -> float:
        return net_force_n / self.params.mass_kg

    def resolve_exit_speed(
        self,
        entry_speed_ms: float,
        segment_length_m: float,
        net_force_n: float,
        corner_speed_limit_ms: float,
    ) -> tuple[float, float]:
        a = self.acceleration(net_force_n)
        v_sq = entry_speed_ms ** 2 + 2.0 * a * segment_length_m
        if v_sq < 0:
            v_sq = 0.0
        exit_speed = math.sqrt(v_sq)
        exit_speed = min(exit_speed, corner_speed_limit_ms)
        avg_speed = max(0.5 * (entry_speed_ms + exit_speed), 0.1)
        seg_time = segment_length_m / avg_speed
        return exit_speed, seg_time

    # ------------------------------------------------------------------
    # Shims for engine fallback paths that peek at attributes
    # ------------------------------------------------------------------
    @property
    def vehicle(self):
        # engine.py + envelope modules read .vehicle to peek at mass etc.
        # Provide a minimal proxy sufficient for downstream reads.
        from types import SimpleNamespace
        p = self.params
        return SimpleNamespace(
            mass_kg=p.mass_kg,
            drag_coefficient=p.cd_a_m2,
            frontal_area_m2=1.0,
            downforce_coefficient=p.cl_a_m2,
            rolling_resistance=self.rolling_resistance,
            wheelbase_m=p.wheelbase_m,
        )

    @property
    def m_effective(self) -> float:
        return self.params.mass_kg

    @property
    def tire_model(self):
        return self.corner.model

    @property
    def load_transfer(self):
        return None  # engine branches on this; None means "physics-based answers via me"

    @property
    def cornering_solver(self):
        return self  # our max_cornering_speed is the solver

    def mechanical_brake_force(self, brake_pct: float, speed_ms: float) -> float:
        return max(0.0, min(1.0, brake_pct)) * self.max_braking_force(speed_ms)
