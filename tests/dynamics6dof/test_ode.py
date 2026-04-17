"""Unit tests for the ODE RHS.

These tests target the ASSEMBLY logic — force and torque summation, rotating-
frame Coriolis, Euler about CG, drivetrain coupling — by injecting known
inputs and checking analytical outputs. We deliberately do NOT compare to
fastest-lap's DLL here because the tire models differ (the DLL uses its
simplified MF, we use PAC02 with Hoosier LC0 coefficients), so numerical
agreement on tire forces is not expected. Sub-components are verified against
the oracle where it makes sense (aero) in their own test files.
"""
from __future__ import annotations

import numpy as np
import pytest

from fsae_sim.dynamics6dof.ode import rhs
from fsae_sim.dynamics6dof.params import Dynamics6DofParams
from fsae_sim.dynamics6dof.state import State6Dof


@pytest.fixture
def params() -> Dynamics6DofParams:
    return Dynamics6DofParams.ct16ev_defaults()


def test_zero_state_zero_tires_is_static_equilibrium(params):
    # State = all zeros -> chassis sits at its static equilibrium, suspension
    # deformation produces Fz per corner that balances gravity. Expected:
    # all derivatives ≈ 0 (no tire forces injected, but suspension Fz reacts
    # to the static offset baked into params).
    state = State6Dof.from_array(np.zeros(10))
    dstate = rhs(state, steering_rad=0.0, throttle=0.0, brake=0.0, params=params)
    # d(rear_omega) = 0, dvx = 0, dvy = 0
    assert dstate[0] == pytest.approx(0.0, abs=1e-6)
    assert dstate[1] == pytest.approx(0.0, abs=1e-6)
    assert dstate[2] == pytest.approx(0.0, abs=1e-6)
    # d(dz)/dt should be approximately zero (gravity balanced by suspension Fz)
    assert dstate[7] == pytest.approx(0.0, abs=1e-3)


def test_derivative_ordering_matches_state_ordering(params):
    # d(z)/dt = state.dz, d(phi)/dt = state.dphi, d(mu)/dt = state.dmu
    # Load the state with distinct values so we can't miss a swap.
    arr = np.array([100.0, 20.0, 1.0, 0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    state = State6Dof.from_array(arr)
    dstate = rhs(state, steering_rad=0.0, throttle=0.0, brake=0.0, params=params)
    assert dstate[4] == pytest.approx(state.dz)
    assert dstate[5] == pytest.approx(state.dphi)
    assert dstate[6] == pytest.approx(state.dmu)


def test_aero_drag_decelerates_car_at_cruise(params):
    state = State6Dof.initial(vx=20.0, rear_omega=20.0 / params.tire_unloaded_radius_m)
    dstate = rhs(state, steering_rad=0.0, throttle=0.0, brake=0.0, params=params)
    # With CdA=1.50, rho=1.225, speed=20 m/s:
    #   drag_mag = 0.5 * 1.225 * 1.50 * 20 * 20 = 367.5 N  (acts in -x, opposing +vx)
    # dvx = (-367.5) / 288 ≈ -1.28 m/s^2 (gravity-vertical doesn't reach x)
    assert dstate[1] < 0.0
    assert dstate[1] == pytest.approx(-367.5 / params.mass_kg, rel=5e-2)


def test_applied_forward_tire_fx_accelerates_car(params):
    # Inject 250 N Fx at each rear tire, front tires get 0 — car should accelerate forward.
    def tire_fn(*, slip_angle_rad, slip_ratio, fz_n):
        return 250.0, 0.0

    state = State6Dof.initial(vx=20.0, rear_omega=20.0 / params.tire_unloaded_radius_m)
    dstate = rhs(
        state, steering_rad=0.0, throttle=0.0, brake=0.0,
        params=params, tire_forces_fn=tire_fn,
    )
    # All 4 corners get 250 N forward (free-rolling fronts also "produce" Fx per the
    # injected function, which is the test contract). Net force = 1000 N forward,
    # minus drag ~367 N. Expected dvx = (1000 - 367.5) / 288 ≈ 2.2 m/s^2.
    assert dstate[1] > 0.0
    assert dstate[1] == pytest.approx((4 * 250.0 - 367.5) / params.mass_kg, rel=0.1)


def test_forward_tire_fx_decelerates_wheel(params):
    # With Fx_rear > 0 on car and no engine torque, the rear axle reaction -Fx*R
    # slows omega.
    def tire_fn(*, slip_angle_rad, slip_ratio, fz_n):
        return 100.0, 0.0

    state = State6Dof.initial(vx=20.0, rear_omega=20.0 / params.tire_unloaded_radius_m)
    dstate = rhs(
        state, steering_rad=0.0, throttle=0.0, brake=0.0,
        params=params, tire_forces_fn=tire_fn,
    )
    # d(rear_omega) = (0 - 0 - (100 + 100) * R) / I_axle = -200 * 0.2042 / 0.3 ≈ -136
    expected = -(2 * 100.0 * params.tire_unloaded_radius_m) / params.rear_axle_inertia_kgm2
    assert dstate[0] == pytest.approx(expected, rel=1e-6)


def test_throttle_accelerates_rear_axle(params):
    state = State6Dof.initial(vx=20.0, rear_omega=20.0 / params.tire_unloaded_radius_m)
    # With zero tire forces, T_eng = throttle * engine_torque_scale_nm only.
    dstate = rhs(
        state, steering_rad=0.0, throttle=0.5, brake=0.0,
        params=params, engine_torque_scale_nm=200.0,
    )
    # d(rear_omega) = 0.5 * 200 / 0.3 ≈ 333 rad/s^2
    assert dstate[0] == pytest.approx(0.5 * 200.0 / params.rear_axle_inertia_kgm2, rel=1e-9)


def test_lateral_force_at_front_creates_positive_yaw_moment(params):
    # Front-axle Fy in +y direction should yaw the car CCW (positive dwz)
    def tire_fn(*, slip_angle_rad, slip_ratio, fz_n):
        return 0.0, 500.0  # pure +Fy on every corner (for tests it's fine)

    state = State6Dof.initial(vx=20.0, rear_omega=20.0 / params.tire_unloaded_radius_m)
    dstate = rhs(
        state, steering_rad=0.0, throttle=0.0, brake=0.0,
        params=params, tire_forces_fn=tire_fn,
    )
    # At state wz=0, the Euler equation reduces to I*dw = τ. The lateral force
    # at front axle (x=+a) contributes +a*Fy_total to Mz. At rear (x=-b) it
    # contributes -b*Fy_total. Net Mz = (a - b) * (Fy_front + Fy_rear). With
    # CT-16EV (47% front), a > b, so sum > 0 -> dwz > 0.
    assert dstate[3] > 0.0


def test_yaw_rate_coriolis_turns_forward_velocity_into_lateral(params):
    # Pure yaw rate at forward velocity, no forces, should give dvy = -wz*vx
    # (coriolis). vx should not change from coriolis alone.
    state = State6Dof.from_array(np.array([100.0, 20.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    dstate = rhs(state, steering_rad=0.0, throttle=0.0, brake=0.0, params=params)
    # Observed dvy includes both tire-contribution (here ~0, no slip/fz) and
    # coriolis. Gravity/aero don't reach y. So dvy ≈ -wz*vx = -0.5*20 = -10 m/s^2,
    # plus whatever drag-x effect reaches y (none, so 0). Aero lift doesn't enter vy.
    # Because tires have zero forces (tire_forces_fn None defaults to zero), the
    # only y contributions are coriolis and aero drag (which has 0 y component
    # at vy=0). So dvy ≈ -10.
    assert dstate[2] == pytest.approx(-0.5 * 20.0, rel=5e-2)


def test_return_shape_is_10(params):
    state = State6Dof.initial(vx=10.0)
    dstate = rhs(state, steering_rad=0.1, throttle=0.2, brake=0.0, params=params)
    assert dstate.shape == (10,)
    assert np.all(np.isfinite(dstate))
