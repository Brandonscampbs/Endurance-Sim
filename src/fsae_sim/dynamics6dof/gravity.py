from __future__ import annotations

import numpy as np

from .params import GRAVITY, Dynamics6DofParams


def gravity_body_force(roll_rad: float, pitch_rad: float, params: Dynamics6DofParams) -> np.ndarray:
    """Project gravity (world -z) into chassis body frame.

    Body frame is reached from world via yaw (irrelevant for gravity) then
    pitch-then-roll. For small angles this is the exact rotation applied to
    the world-frame gravity vector (0, 0, -g).
    """
    c_phi, s_phi = np.cos(roll_rad), np.sin(roll_rad)
    c_mu, s_mu = np.cos(pitch_rad), np.sin(pitch_rad)
    g = GRAVITY
    # Applying R_roll * R_pitch to (0, 0, -g):
    # R_pitch: (x cos_mu + z sin_mu, y, -x sin_mu + z cos_mu)
    # Start with v_world = (0, 0, -g); after pitch: (-g sin_mu, 0, -g cos_mu)
    # After roll: (-g sin_mu, -g cos_mu * sin_phi, -g cos_mu * cos_phi) -- but sign of
    # y term depends on roll convention. We follow ISO 8855: positive roll =
    # left side up, so a positive roll tilts gravity toward +y.
    fx = -g * s_mu
    fy = +g * c_mu * s_phi
    fz = -g * c_mu * c_phi
    return params.mass_kg * np.array([fx, fy, fz])
