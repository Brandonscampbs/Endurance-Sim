# src/fsae_sim/dynamics6dof/suspension.py
"""Per-axle suspension load-transfer via sym/asym decomposition.

Ported from fastest-lap ``axle_car_6dof.hpp:119-157``. For one axle we
decompose the chassis-to-wheel kinematic into a symmetric (heave) mode and
an asymmetric (roll) mode. Each mode has its own effective series stiffness
between the chassis spring and the tire radial spring; the anti-roll bar
appears only in the asymmetric path.

Sign convention: positive ``w`` means the tire is compressed vertically, so
Fz (computed downstream in ``tire_vertical.py``) is non-negative.
"""
from __future__ import annotations


def axle_tire_deformation(
    *,
    z: float,
    phi: float,
    dz: float,
    dphi: float,
    track_m: float,
    k_chassis: float,
    k_antiroll: float,
    k_tire: float,
    steering_rad: float = 0.0,
    beta_left: float = 0.0,
    beta_right: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return ``(w_left, w_right, dw_left, dw_right)`` for one axle.

    Parameters
    ----------
    z, phi:
        Chassis heave (m, positive down) and roll angle (rad, right-hand
        rule about the body x-axis) at the axle.
    dz, dphi:
        Time derivatives of ``z`` and ``phi``.
    track_m:
        Axle track width (distance between left and right contact patches).
    k_chassis, k_antiroll, k_tire:
        Chassis wheel-rate, anti-roll bar wheel-rate, and tire radial
        stiffness, all in N/m.
    steering_rad, beta_left, beta_right:
        Optional bump-steer-style coupling from steering to per-wheel
        asymmetric displacement. Zero for the rear axle and for the front
        when bump-steer is neglected. Kept in the signature so the caller
        does not need to know whether an axle steers.

    Notes
    -----
    Effective stiffnesses (series combination of chassis + tire, with the
    anti-roll bar contributing only to the asymmetric mode)::

        sym_stiffness  = 1 / (k_chassis + k_tire)
        asym_stiffness = 2 * k_antiroll * k_tire * sym_stiffness
                         / (2 * k_antiroll + k_chassis + k_tire)

    Displacements::

        displ_sym  = z
        displ_asym = 0.5 * track * phi + beta_right * steering_rad

    Per-wheel tire deformation::

        wl = k_chassis * sym_stiffness * (displ_sym - displ_asym)
             - displ_asym * asym_stiffness
        wr = k_chassis * sym_stiffness * (displ_sym + displ_asym)
             + displ_asym * asym_stiffness

    Rates follow by differentiating; ``ddispl_asym = 0.5 * track * dphi``
    since the steering term is quasi-static on the timescale of the chassis
    modes.
    """
    sym_stiffness = 1.0 / (k_chassis + k_tire)
    asym_stiffness = (
        2.0 * k_antiroll * k_tire * sym_stiffness
        / (2.0 * k_antiroll + k_chassis + k_tire)
    )

    displ_sym = z
    ddispl_sym = dz
    displ_asym = 0.5 * track_m * phi + beta_right * steering_rad
    ddispl_asym = 0.5 * track_m * dphi

    wl = (
        k_chassis * sym_stiffness * (displ_sym - displ_asym)
        - displ_asym * asym_stiffness
    )
    wr = (
        k_chassis * sym_stiffness * (displ_sym + displ_asym)
        + displ_asym * asym_stiffness
    )
    dwl = (
        k_chassis * sym_stiffness * (ddispl_sym - ddispl_asym)
        - ddispl_asym * asym_stiffness
    )
    dwr = (
        k_chassis * sym_stiffness * (ddispl_sym + ddispl_asym)
        + ddispl_asym * asym_stiffness
    )
    # ``beta_left`` is accepted for API symmetry with the C++ signature but
    # is not required for the current CT-16EV axle definitions (bump-steer
    # neglected). Referenced here so a future extension can couple it
    # without changing the call sites.
    _ = beta_left
    return wl, wr, dwl, dwr
