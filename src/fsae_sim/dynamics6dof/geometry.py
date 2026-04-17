from __future__ import annotations

import numpy as np

from .params import Dynamics6DofParams


def corner_positions(params: Dynamics6DofParams) -> dict[str, np.ndarray]:
    """Return body-frame 3D positions of the four contact patches relative to CG.

    Conventions: x forward, y left, z up. Contact patch sits at -cg_height below CG.
    """
    a = params.wheelbase_m * (1.0 - params.weight_dist_front)  # CG to front axle
    b = params.wheelbase_m * params.weight_dist_front           # CG to rear axle
    tf = 0.5 * params.track_front_m
    tr = 0.5 * params.track_rear_m
    h = params.cg_height_m
    return {
        "FL": np.array([+a, +tf, -h]),
        "FR": np.array([+a, -tf, -h]),
        "RL": np.array([-b, +tr, -h]),
        "RR": np.array([-b, -tr, -h]),
    }


def corner_velocities(
    v_cg_body_mps: np.ndarray,
    omega_body_radps: np.ndarray,
    params: Dynamics6DofParams,
) -> dict[str, np.ndarray]:
    """Body-frame velocity at each contact patch, accounting for rotation.

    v_corner = v_cg + omega x r_corner.
    """
    positions = corner_positions(params)
    return {k: v_cg_body_mps + np.cross(omega_body_radps, r) for k, r in positions.items()}
