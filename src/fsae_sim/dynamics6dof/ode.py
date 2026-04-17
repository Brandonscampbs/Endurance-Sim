"""ODE right-hand side for the kart-6dof port.

Assembles gravity, aero, per-corner tire forces (using an injectable
`tire_forces_fn` so callers can use PAC02), then solves Newton's 2nd law
in the rotating body frame and Euler's rotational equations about CG.

Frame convention: x forward, y left, z up. Angles are small-rotation
roll-pitch-yaw; gravity body-frame projection uses the exact rotation.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .aero import aero_force
from .geometry import corner_positions, corner_velocities
from .gravity import gravity_body_force
from .kinematics import corner_contact_velocity, slip_quantities
from .params import Dynamics6DofParams
from .state import STATE_SIZE, State6Dof
from .suspension import axle_tire_deformation
from .tire_vertical import fz_from_deformation

TireForcesFn = Callable[..., tuple[float, float]]
"""Signature: tire_forces_fn(*, slip_angle_rad, slip_ratio, fz_n) -> (Fx_tire, Fy_tire)."""


def _zero_tire_forces(*, slip_angle_rad: float, slip_ratio: float, fz_n: float) -> tuple[float, float]:
    return 0.0, 0.0


def rhs(
    state: State6Dof,
    *,
    steering_rad: float,
    throttle: float,
    brake: float,
    params: Dynamics6DofParams,
    tire_forces_fn: Optional[TireForcesFn] = None,
    engine_torque_scale_nm: float = 200.0,
    brake_torque_scale_nm: float = 500.0,
) -> np.ndarray:
    """Return d(state)/dt as a 10-vector matching ``State6Dof.to_array()`` order.

    Parameters
    ----------
    state:
        Current 10-state ``State6Dof``.
    steering_rad, throttle, brake:
        Control inputs. Throttle and brake are 0..1; final scaling to N⋅m at
        the rear axle is controlled by ``engine_torque_scale_nm`` /
        ``brake_torque_scale_nm`` (the real LVCU torque chain lives in the
        powertrain module and should be plumbed through in Task 13).
    params:
        ``Dynamics6DofParams``.
    tire_forces_fn:
        Per-corner lateral/longitudinal tire-force callable. If ``None``,
        tire forces are zero (useful for gravity+aero+drivetrain-only tests).

    Returns
    -------
    np.ndarray of shape (10,)
        Derivative in the order
        ``[d(rear_omega), d(vx), d(vy), d(wz), d(z), d(phi), d(mu),
            d(dz), d(dphi), d(dmu)]``.
    """
    if tire_forces_fn is None:
        tire_forces_fn = _zero_tire_forces

    p = params

    # Kinematic assembly -----------------------------------------------------
    v_body = np.array([state.vx, state.vy, state.dz])
    omega = np.array([state.dphi, state.dmu, state.wz])  # roll, pitch, yaw rates

    positions = corner_positions(p)
    v_corners = corner_velocities(v_body, omega, p)

    # Per-axle suspension deformation. `state.z` is measured as a DEVIATION
    # from the per-axle static equilibrium, so we add the static offset before
    # passing to the axle solver. This keeps the equilibrium at z=0 (useful
    # for the outer sim) while the suspension module stays pure-geometric.
    wl_f, wr_f, dwl_f, dwr_f = axle_tire_deformation(
        z=state.z + p.static_sym_displ_front_m,
        phi=state.phi, dz=state.dz, dphi=state.dphi,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_antiroll=p.k_antiroll_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=steering_rad, beta_left=0.0, beta_right=0.0,
    )
    wl_r, wr_r, dwl_r, dwr_r = axle_tire_deformation(
        z=state.z + p.static_sym_displ_rear_m,
        phi=state.phi, dz=state.dz, dphi=state.dphi,
        track_m=p.track_rear_m,
        k_chassis=p.k_chassis_rear_npm,
        k_antiroll=p.k_antiroll_rear_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0, beta_left=0.0, beta_right=0.0,
    )
    fz = {
        "FL": fz_from_deformation(wl_f, dwl_f, p.k_tire_radial_npm, p.c_tire_radial_nspm),
        "FR": fz_from_deformation(wr_f, dwr_f, p.k_tire_radial_npm, p.c_tire_radial_nspm),
        "RL": fz_from_deformation(wl_r, dwl_r, p.k_tire_radial_npm, p.c_tire_radial_nspm),
        "RR": fz_from_deformation(wr_r, dwr_r, p.k_tire_radial_npm, p.c_tire_radial_nspm),
    }

    # Per-corner tire forces -------------------------------------------------
    tire_f: dict[str, np.ndarray] = {}
    for c in ("FL", "FR", "RL", "RR"):
        delta = steering_rad if c.startswith("F") else 0.0
        v_corner = v_corners[c]
        if c in ("RL", "RR"):
            wheel_omega = state.rear_omega
        else:
            # Free-rolling front: omega = v_forward_at_tire / R0
            v_tire_frame = corner_contact_velocity(v_corner, steering_rad=delta)
            wheel_omega = max(v_tire_frame[0], 1e-3) / p.tire_unloaded_radius_m

        if fz[c] < 1e-6:
            fx_tire = fy_tire = 0.0
        else:
            _, lam = slip_quantities(
                v_corner, wheel_omega, p.tire_unloaded_radius_m, steering_rad=delta,
            )
            kappa, _ = slip_quantities(
                v_corner, wheel_omega, p.tire_unloaded_radius_m, steering_rad=delta,
            )
            fx_tire, fy_tire = tire_forces_fn(
                slip_angle_rad=lam, slip_ratio=kappa, fz_n=fz[c],
            )

        # Rotate tire-frame (Fx, Fy) into body frame. Steering is about z.
        c_d, s_d = float(np.cos(delta)), float(np.sin(delta))
        fx_body = fx_tire * c_d - fy_tire * s_d
        fy_body = fx_tire * s_d + fy_tire * c_d
        # Fz acts UP on the car in our +z-up convention.
        tire_f[c] = np.array([fx_body, fy_body, fz[c]])

    # Total force and torque about CG ---------------------------------------
    total_f = np.zeros(3)
    total_t = np.zeros(3)
    for c, f_vec in tire_f.items():
        total_f += f_vec
        total_t += np.cross(positions[c], f_vec)

    # Aero at CG (no moment arm)
    drag, lift = aero_force(v_body, np.zeros(3), p)
    total_f += drag + lift

    # Gravity in body frame
    total_f += gravity_body_force(state.phi, state.mu, p)

    # Newton in rotating body frame: m*(dv + ω × v) = F
    dv = total_f / p.mass_kg - np.cross(omega, v_body)

    # Euler about CG: I*dω + ω × (I*ω) = M
    I = np.array([
        [p.ixx_kgm2, 0.0, p.ixz_kgm2],
        [0.0, p.iyy_kgm2, 0.0],
        [p.ixz_kgm2, 0.0, p.izz_kgm2],
    ])
    dw = np.linalg.solve(I, total_t - np.cross(omega, I @ omega))

    # Rear axle shaft ODE (locked axle, matches fastest-lap kart):
    # I_axle * d(omega)/dt = T_engine - T_brake - (Fx_RL + Fx_RR) * R
    T_eng = throttle * engine_torque_scale_nm
    T_brk = brake * brake_torque_scale_nm
    fx_rl = tire_f["RL"][0]
    fx_rr = tire_f["RR"][0]
    R = p.tire_unloaded_radius_m
    d_rear_omega = (T_eng - T_brk - (fx_rl + fx_rr) * R) / p.rear_axle_inertia_kgm2

    # Pack in State6Dof order: [rear_omega, vx, vy, wz, z, phi, mu, dz, dphi, dmu]
    dstate = np.empty(STATE_SIZE)
    dstate[0] = d_rear_omega
    dstate[1] = dv[0]
    dstate[2] = dv[1]
    dstate[3] = dw[2]
    dstate[4] = state.dz
    dstate[5] = state.dphi
    dstate[6] = state.dmu
    dstate[7] = dv[2]
    dstate[8] = dw[0]
    dstate[9] = dw[1]
    return dstate
