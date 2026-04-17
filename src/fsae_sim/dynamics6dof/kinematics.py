from __future__ import annotations

import numpy as np


def corner_contact_velocity(v_corner_body_mps: np.ndarray, steering_rad: float = 0.0) -> np.ndarray:
    """Rotate chassis-frame velocity at a corner into tire frame.

    Steering rotation is about z (yaw). Positive steering → left turn.
    """
    c, s = np.cos(steering_rad), np.sin(steering_rad)
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ v_corner_body_mps


def slip_quantities(
    v_corner_body_mps: np.ndarray,
    wheel_omega_radps: float,
    wheel_radius_m: float,
    steering_rad: float = 0.0,
    eps: float = 0.1,
) -> tuple[float, float]:
    """Return (slip_ratio kappa, slip_angle lambda) for a single corner.

    Mirrors fastest-lap's `kappa = (omega*R0 - vx)/vx` and `lambda = -vy/vx`
    (both linear-approximation forms). We floor vx to `eps` m/s to avoid
    division by zero at standstill; the outer sim must not rely on results
    when vx < eps.
    """
    v_tire = corner_contact_velocity(v_corner_body_mps, steering_rad)
    vx = v_tire[0] if abs(v_tire[0]) >= eps else (eps if v_tire[0] >= 0 else -eps)
    kappa = (wheel_omega_radps * wheel_radius_m - vx) / vx
    lam = -v_tire[1] / vx
    return float(kappa), float(lam)
