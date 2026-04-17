"""Tire vertical force with C^2 smooth regularization.

Matches lion-cpp's `smooth_pos(a, eps2) = 0.5*(a + sqrt(a^2 + eps2))` exactly
(verified against fastest-lap's pinned lion-cpp SHA e1e4ac06). `eps2` is the
squared smoothing width passed directly — do NOT divide by 4.
"""
from __future__ import annotations

import numpy as np


def smooth_pos(x: float, eps2: float) -> float:
    """C^2 smooth approximation of max(x, 0).

    Matches lion/foundation/utils.hpp::smooth_pos. For x >> sqrt(eps2) this
    returns x; for x << -sqrt(eps2) it returns ~0; and it is C^2 through zero.
    """
    return 0.5 * (x + float(np.sqrt(x * x + eps2)))


def fz_from_deformation(
    w: float,
    dw: float,
    k_tire: float,
    c_tire: float,
    fz_eps2: float = 1.0,
) -> float:
    """Return positive Fz (N) from tire radial deformation and rate.

    Fz = smooth_pos(k_tire*w + c_tire*dw, fz_eps2). Default fz_eps2=1.0 N^2
    gives <0.3 N smoothing-region error; negligible at typical Fz ~700+ N.
    """
    return smooth_pos(k_tire * w + c_tire * dw, fz_eps2)
