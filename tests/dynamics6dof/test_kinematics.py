import numpy as np
import pytest

from fsae_sim.dynamics6dof.kinematics import corner_contact_velocity, slip_quantities


def test_zero_slip_at_cruise():
    # Straight cruise: wheel speed matches vx exactly
    vx = 20.0
    R0 = 0.2042
    omega = vx / R0
    v = np.array([vx, 0.0, 0.0])
    kappa, lam = slip_quantities(v, omega, R0, steering_rad=0.0)
    assert kappa == pytest.approx(0.0, abs=1e-12)
    assert lam == pytest.approx(0.0, abs=1e-12)


def test_positive_kappa_when_wheel_faster():
    R0 = 0.2042
    v = np.array([20.0, 0.0, 0.0])
    omega = (20.0 * 1.1) / R0  # 10% overspeed → kappa=+0.1
    kappa, lam = slip_quantities(v, omega, R0, steering_rad=0.0)
    assert kappa == pytest.approx(0.1, rel=1e-9)
    assert lam == pytest.approx(0.0, abs=1e-12)


def test_slip_angle_from_lateral_velocity():
    R0 = 0.2042
    v = np.array([20.0, -2.0, 0.0])
    omega = 20.0 / R0
    kappa, lam = slip_quantities(v, omega, R0, steering_rad=0.0)
    assert lam == pytest.approx(0.1, rel=1e-9)  # lambda = -vy/vx


def test_steering_rotates_velocity_into_tire_frame():
    # If chassis has pure forward velocity and the wheel is steered by delta,
    # in the *tire* frame the velocity has components (cos d, -sin d)*vx.
    v_body = np.array([20.0, 0.0, 0.0])
    delta = 0.1
    v_tire = corner_contact_velocity(v_body, steering_rad=delta)
    assert v_tire[0] == pytest.approx(20.0 * np.cos(delta), rel=1e-12)
    assert v_tire[1] == pytest.approx(-20.0 * np.sin(delta), rel=1e-12)
