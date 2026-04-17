import numpy as np
import pytest

from fsae_sim.dynamics6dof.geometry import corner_positions, corner_velocities
from fsae_sim.dynamics6dof.params import Dynamics6DofParams


def test_corner_positions_left_is_positive_y():
    p = Dynamics6DofParams.ct16ev_defaults()
    pos = corner_positions(p)
    fl, fr, rl, rr = pos["FL"], pos["FR"], pos["RL"], pos["RR"]
    assert fl[1] > 0 and fr[1] < 0
    assert rl[1] > 0 and rr[1] < 0
    assert fl[0] > 0 and rl[0] < 0
    assert all(c[2] == pytest.approx(-p.cg_height_m) for c in pos.values())


def test_pure_forward_velocity_yields_pure_forward_at_all_corners():
    p = Dynamics6DofParams.ct16ev_defaults()
    v = np.array([20.0, 0.0, 0.0])
    omega = np.zeros(3)
    vels = corner_velocities(v, omega, p)
    for c in ("FL", "FR", "RL", "RR"):
        np.testing.assert_allclose(vels[c], [20.0, 0.0, 0.0], atol=1e-12)


def test_pure_yaw_produces_lateral_velocities_at_front_and_rear():
    p = Dynamics6DofParams.ct16ev_defaults()
    v = np.zeros(3)
    omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s yaw
    vels = corner_velocities(v, omega, p)
    # omega cross r: at front axle (x>0), cross z with x gives +y on left/right
    # front-left: r=(a, +t/2, -h), omega x r = (-t/2, a, 0)
    a = p.wheelbase_m * (1.0 - p.weight_dist_front)
    t = p.track_front_m
    expected_fl = np.array([-0.5 * t, a, 0.0])
    np.testing.assert_allclose(vels["FL"], expected_fl, atol=1e-12)
