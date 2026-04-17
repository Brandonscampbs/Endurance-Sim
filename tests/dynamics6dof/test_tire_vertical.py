import pytest

from fsae_sim.dynamics6dof.tire_vertical import fz_from_deformation, smooth_pos


def test_smooth_pos_is_identity_far_above_zero():
    # For x >> sqrt(eps2), smooth_pos(x, eps2) -> x
    assert smooth_pos(1000.0, 1.0) == pytest.approx(1000.0, rel=1e-5)


def test_smooth_pos_is_near_zero_far_below_zero():
    assert smooth_pos(-1000.0, 1.0) == pytest.approx(0.0, abs=1e-3)


def test_smooth_pos_is_positive_and_bounded_at_zero():
    assert smooth_pos(0.0, 1.0) == pytest.approx(0.5, rel=1e-12)
    assert smooth_pos(0.0, 0.25) == pytest.approx(0.25, rel=1e-12)


def test_smooth_pos_slope_at_zero():
    # dsmooth_pos/dx at x=0 equals 0.5 (C^1 through zero)
    eps2 = 1e-6
    h = 1e-4
    dfdx = (smooth_pos(h, eps2) - smooth_pos(-h, eps2)) / (2 * h)
    assert dfdx == pytest.approx(0.5, abs=0.05)


def test_fz_scales_with_deformation():
    kt, ct = 150_000.0, 150.0
    # k*w = 1500 N, eps2=1 N^2 -> overshoot ~ eps2/(4*1500) ≈ 1.7e-4
    fz = fz_from_deformation(w=0.01, dw=0.0, k_tire=kt, c_tire=ct)
    assert fz == pytest.approx(1500.0, rel=1e-3)


def test_fz_damping_term_contributes():
    kt, ct = 150_000.0, 150.0
    fz_static = fz_from_deformation(w=0.01, dw=0.0, k_tire=kt, c_tire=ct)
    fz_moving = fz_from_deformation(w=0.01, dw=1.0, k_tire=kt, c_tire=ct)
    assert fz_moving > fz_static
    assert (fz_moving - fz_static) == pytest.approx(150.0, rel=1e-3)


def test_fz_never_negative():
    # Tension: w<0, dw=0 -> Fz clamps near 0
    fz = fz_from_deformation(w=-0.01, dw=0.0, k_tire=150_000.0, c_tire=150.0)
    assert fz >= 0.0
    assert fz < 1.0
