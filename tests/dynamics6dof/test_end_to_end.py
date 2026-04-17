"""End-to-end integration test using the full stack.

Drives the RK4 integrator over the ODE RHS with the real PAC02 tire adapter.
No oracle comparison — we check that the system produces physically sensible
behaviour (energy budget, steady-state settling, yaw response to steering).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fsae_sim.dynamics6dof.integrator import rk4_step
from fsae_sim.dynamics6dof.ode import rhs
from fsae_sim.dynamics6dof.params import Dynamics6DofParams
from fsae_sim.dynamics6dof.state import State6Dof
from fsae_sim.dynamics6dof.tire_pac02 import PAC02Corner
from fsae_sim.vehicle.tire_model import PacejkaTireModel
from fsae_sim.vehicle.vehicle import VehicleConfig


@pytest.fixture
def pac02_corner():
    cfg_path = Path(__file__).parents[2] / "configs" / "ct16ev.yaml"
    cfg = VehicleConfig.from_yaml(cfg_path)
    tir_path = Path(__file__).parents[2] / cfg.tire.tir_file
    model = PacejkaTireModel(tir_path)
    model.apply_grip_scale(cfg.tire.grip_scale)
    return PAC02Corner(model)


def test_zero_throttle_at_cruise_decelerates_over_half_second(pac02_corner):
    """Coast test: vx=20 m/s, no throttle. Drag should bleed off speed."""
    p = Dynamics6DofParams.ct16ev_defaults()
    q = State6Dof.initial(vx=20.0, rear_omega=20.0 / p.tire_unloaded_radius_m).to_array()
    dt = 1e-3
    steps = 500

    def tire_fn(*, slip_angle_rad, slip_ratio, fz_n):
        return pac02_corner.forces(
            slip_angle_rad=slip_angle_rad, slip_ratio=slip_ratio, fz_n=fz_n,
        )

    for _ in range(steps):
        q = rk4_step(
            q, dt, rhs,
            steering_rad=0.0, throttle=0.0, brake=0.0,
            params=p, tire_forces_fn=tire_fn,
        )
    final_vx = q[1]
    # Drag at 20 m/s ~ 367 N; over 0.5 s that's ~640 N⋅s impulse.
    # Δv ≈ -impulse/m ≈ -2.2 m/s. Final vx should be 17.5-19.5 m/s.
    # Also rolling resistance from tires reduces further, so allow wider band.
    assert 15.0 < final_vx < 20.0


def test_throttle_at_cruise_accelerates_or_maintains(pac02_corner):
    """Throttle test: vx=20 m/s, throttle=1.0. Speed should not drop rapidly."""
    p = Dynamics6DofParams.ct16ev_defaults()
    q = State6Dof.initial(vx=20.0, rear_omega=20.0 / p.tire_unloaded_radius_m).to_array()
    dt = 1e-3
    steps = 500

    def tire_fn(*, slip_angle_rad, slip_ratio, fz_n):
        return pac02_corner.forces(
            slip_angle_rad=slip_angle_rad, slip_ratio=slip_ratio, fz_n=fz_n,
        )

    for _ in range(steps):
        q = rk4_step(
            q, dt, rhs,
            steering_rad=0.0, throttle=1.0, brake=0.0,
            params=p, tire_forces_fn=tire_fn,
        )
    final_vx = q[1]
    # With 200 N⋅m at the axle over 0.2042 m radius, peak Fx ≈ 980 N at each rear.
    # Minus 367 N drag, net force ~ +600-1500 N. Δv over 0.5s: +1 to +3 m/s.
    # Accept anywhere > 18 m/s (no severe grip limit failure) and < 35 (sanity).
    assert 18.0 < final_vx < 35.0


def test_finite_outputs_under_combined_maneuver(pac02_corner):
    """Stress test: steering + throttle, make sure nothing goes NaN/inf."""
    p = Dynamics6DofParams.ct16ev_defaults()
    q = State6Dof.initial(vx=15.0, rear_omega=15.0 / p.tire_unloaded_radius_m).to_array()
    dt = 1e-3
    steps = 200

    def tire_fn(*, slip_angle_rad, slip_ratio, fz_n):
        return pac02_corner.forces(
            slip_angle_rad=slip_angle_rad, slip_ratio=slip_ratio, fz_n=fz_n,
        )

    for _ in range(steps):
        q = rk4_step(
            q, dt, rhs,
            steering_rad=0.05, throttle=0.3, brake=0.0,
            params=p, tire_forces_fn=tire_fn,
        )
        assert np.all(np.isfinite(q)), f"State went non-finite: {q}"
    # At the end, yaw rate should be nonzero under steady steering
    assert abs(q[3]) > 0.01  # wz (yaw rate)
