# tests/dynamics6dof/test_integrator.py
import numpy as np
import pytest

from fsae_sim.dynamics6dof.integrator import rk4_step
from fsae_sim.dynamics6dof.state import State6Dof


def test_rk4_matches_euler_for_linear_system():
    # dx/dt = -x. Closed-form solution: x(t) = x0 * exp(-t).
    def f(s, **kwargs):
        q = s.to_array()
        return -q

    s0 = State6Dof.from_array(np.array([1.0] * 10))
    s = s0
    dt = 0.01
    for _ in range(100):
        s = State6Dof.from_array(rk4_step(s.to_array(), dt, f))
    np.testing.assert_allclose(s.to_array(), np.exp(-1.0) * s0.to_array(), rtol=1e-6)
