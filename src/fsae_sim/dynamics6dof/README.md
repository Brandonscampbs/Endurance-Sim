# dynamics6dof — 4-wheel vehicle dynamics (port of fastest-lap kart-6dof)

This package is a Python port of the chassis / load-transfer / yaw dynamics
from [fastest-lap](https://github.com/juanmanzanero/fastest-lap) (MIT-licensed,
F1-engineer-authored). It exists so the project has a human-authored,
independently-written 4-wheel dynamics backbone to pair with the project's
existing PAC02 tire model.

## Scope

- Chassis 6-DOF: heave + pitch + roll + yaw + planar translation (10-state)
- Per-corner Fz from tire deformation via sym/asym suspension decomposition
- Rotating-frame Newton's 2nd law + Euler's rotational equations about CG
- Locked rear-axle drivetrain ODE
- Aero drag + downforce
- Gravity projection onto tilted chassis frame

## Out of scope (kept in existing modules)

- Tire lateral/longitudinal force generation — routed to
  `fsae_sim.vehicle.tire_model.PacejkaTireModel` via `PAC02Corner` adapter
- LVCU torque command chain — stays in `fsae_sim.vehicle.powertrain_model`
- Battery, motor efficiency, inverter — all unchanged

## Conventions

- ISO 8855 body frame: x forward, y LEFT, z up
- Angles in radians unless suffixed `_deg`
- Positive slip angle `α` means velocity is to the right of the wheel plane
- Positive slip ratio `κ = (ωR − v)/v` means wheel driving faster than car
- The `state.z` coordinate is DEVIATION from static-equilibrium ride height.
  At z=0, each corner has Fz = m·g / 4 × weight distribution factor.

The PAC02 `.tir` files in `Real-Car-Data-And-Stats/` are fit in the SAE tire
axis system (y-right, z-down). `PAC02Corner.forces` flips the Fy sign at the
seam so the rest of the dynamics6dof package can work in ISO throughout.

## Module map

| File | Purpose |
|---|---|
| `state.py` | 10-state dataclass: `[rear_omega, vx, vy, wz, z, phi, mu, dz, dphi, dmu]` |
| `params.py` | `Dynamics6DofParams` + CT-16EV defaults + static-equilibrium displacement |
| `aero.py` | Drag + downforce, body frame. Oracle-verified against fastest-lap DLL |
| `gravity.py` | Gravity projection onto tilted chassis frame |
| `geometry.py` | Per-corner positions and body-frame corner velocities (`v + ω × r`) |
| `kinematics.py` | Per-corner slip ratio κ and slip angle λ in tire frame |
| `suspension.py` | Symmetric/asymmetric decomposition → per-wheel tire deformation |
| `tire_vertical.py` | Fz from deformation with lion-cpp `smooth_pos` regularization |
| `tire_pac02.py` | Adapter that routes (α, κ, Fz) to `PacejkaTireModel` + ISO/SAE Fy flip |
| `ode.py` | Full RHS: Newton + Euler + drivetrain, composes every module |
| `integrator.py` | Classical RK4 fixed-step integrator |
| `oracle.py` | Optional fastest-lap DLL ctypes wrapper for regression oracle tests |

## Running tests

Regular unit tests (no DLL required):

```bash
pytest tests/dynamics6dof -v
```

Oracle-verified tests (requires fastest-lap Windows DLL):

```bash
# Default DLL location (or set FASTEST_LAP_ROOT env var)
# C:/Users/brand/AppData/Local/Temp/fl-w10/v0.5
pytest tests/dynamics6dof -m oracle -v
```

Oracle tests are skipped cleanly if the DLL is unavailable.

## Known deviations from fastest-lap's kart-6dof

- **Open differential vs locked axle.** CT-16EV has an open differential.
  The port currently uses fastest-lap's `POWERED_WITHOUT_DIFFERENTIAL`
  (locked rear axle, single ω shared by both rear wheels). Matching
  behaviour with a real open diff is a follow-up.
- **Tire model swap.** fastest-lap's simplified 12-parameter MF is replaced
  with the project's PAC02 (Hoosier LC0 Round-8 fit). Numerical agreement
  on tire-force-driven quantities with the fastest-lap DLL is NOT expected.
  Aero, gravity, inertia, and kinematic quantities do match.
- **Cartesian road state.** The `t, n, chi` curvilinear road states present
  in fastest-lap's kart XML are not modeled here — the outer lap sim
  (`fsae_sim.sim.engine`) owns track position independently.

## How it plugs into the rest of the sim

This package exports a self-contained ODE (`ode.rhs`) plus an RK4 stepper
(`integrator.rk4_step`) that can be driven from any orchestration layer. A
thin backend adapter exposing the old `VehicleDynamics` interface, plus the
`dynamics_backend` flag in `engine.py`, are the remaining integration tasks
in the plan and are not yet wired.

Until that integration lands, the existing legacy backend continues to be
the engine's default.
