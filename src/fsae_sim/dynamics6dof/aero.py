# src/fsae_sim/dynamics6dof/aero.py
from __future__ import annotations

import numpy as np

from .params import Dynamics6DofParams


def aero_force(
    vel_body_mps: np.ndarray,
    wind_body_mps: np.ndarray,
    params: Dynamics6DofParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (drag_vector_N, lift_vector_N) in body frame.

    Mirrors fastest-lap's chassis.hpp:264-288: drag scales with the relative
    airspeed vector times its magnitude; lift uses only the forward component
    squared and acts on +z (or -z if downforce) as a pure vertical force.
    """
    v_air = wind_body_mps - vel_body_mps  # aerodynamic velocity seen by car
    speed = float(np.linalg.norm(v_air[:2]))  # planar airspeed
    qbar = 0.5 * params.rho_air_kgpm3 * speed
    drag = qbar * params.cd_a_m2 * v_air  # full 3-vector, direction = v_air
    # Lift uses only forward component squared, vertical axis only. Sign follows
    # fastest-lap: positive cl adds +z; our convention treats downforce as
    # positive cl_a, so we flip the sign on the z component.
    vx_air = v_air[0]
    lift_z = 0.5 * params.rho_air_kgpm3 * params.cl_a_m2 * vx_air * vx_air
    lift = np.array([0.0, 0.0, -lift_z])  # downforce pulls car into ground (-z)
    return drag, lift
