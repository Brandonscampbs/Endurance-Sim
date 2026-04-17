# src/fsae_sim/dynamics6dof/integrator.py
from __future__ import annotations

import numpy as np

from .state import State6Dof


def rk4_step(q: np.ndarray, dt: float, rhs_fn, **kwargs) -> np.ndarray:
    """Classical RK4 single step. `rhs_fn(state_obj, **kwargs) -> dq/dt`."""
    def f_wrapper(q_vec):
        return rhs_fn(State6Dof.from_array(q_vec), **kwargs)

    k1 = f_wrapper(q)
    k2 = f_wrapper(q + 0.5 * dt * k1)
    k3 = f_wrapper(q + 0.5 * dt * k2)
    k4 = f_wrapper(q + dt * k3)
    return q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
