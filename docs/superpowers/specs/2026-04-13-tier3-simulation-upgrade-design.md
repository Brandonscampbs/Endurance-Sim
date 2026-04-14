# Tier 3 Simulation Upgrade: 4-Wheel Model with Pacejka Tires

**Date**: 2026-04-13
**Status**: Approved
**Goal**: Upgrade the point-mass quasi-static sim to a 4-wheel model with real tire data, load transfer, and steady-state cornering solver for improved lap time prediction accuracy (~2-3% vs telemetry).

## Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| Primary use case | Lap time prediction accuracy | Confidence in sim when sweeping parameters (RPM, torque limits) |
| Missing Fx data | Symmetric assumption, validated against telemetry | Peak lateral mu (~2.66) mirrored for longitudinal, then checked against AiM braking G |
| Load transfer model | 4-wheel with roll stiffness distribution | DSS has full suspension data; captures axle saturation that bicycle model misses |
| Cornering model | Steady-state solve (not MMM) | Right fidelity for lap time; MMM documented as future phase |
| Migration strategy | Replace in-place | Keeping both models adds confusion; old tests updated to new physics |
| Build order | Bottom-up component replacement | Each layer validated before the next depends on it |

## Component 1: Pacejka Tire Model

**File**: `src/fsae_sim/vehicle/tire_model.py`
**Class**: `PacejkaTireModel`

### Responsibility

Load a PAC2002 `.tir` file and compute tire forces as a function of slip, load, and camber.

### Construction

Takes a path to a `.tir` file. Parses coefficients from `[LATERAL_COEFFICIENTS]`, `[LONGITUDINAL_COEFFICIENTS]`, `[ALIGNING_COEFFICIENTS]`, `[VERTICAL]`, `[DIMENSION]`, and `[SCALING_COEFFICIENTS]` sections.

### Methods

- `lateral_force(slip_angle, normal_load, camber) -> float` — PAC2002 Fy using fitted PCY1, PDY1-3, PEY1-4, PKY1-3, PHY1-3, PVY1-4 coefficients.
- `longitudinal_force(slip_ratio, normal_load, camber) -> float` — PAC2002 Fx. Since the .tir files have all-zero Fx coefficients, this constructs a **symmetric model**: copies the lateral Pacejka structure (PCY1→PCX1, PDY1→PDX1, PDY2→PDX2, PEY→PEX, PKY1→PKX1) to produce an Fx curve with the same peak mu (~2.66) and load sensitivity. Slip ratio input maps to the same range as slip angle in the lateral formula.
- `combined_forces(slip_angle, slip_ratio, normal_load, camber) -> (Fx, Fy)` — Friction circle scaling. Combined slip coefficients (RBx, RCx, etc.) are zeroed, so this uses vector-magnitude friction circle: each force component is scaled so the resultant doesn't exceed the friction circle envelope.
- `peak_lateral_force(normal_load, camber) -> float` — Sweeps slip angle to find maximum Fy at given load. Used by the cornering solver.
- `peak_longitudinal_force(normal_load, camber) -> float` — Same for Fx.
- `loaded_radius(normal_load, speed) -> float` — From QV1/QV2/QFZ coefficients for speed-from-RPM conversion.

### Data files

Four .tir files at `Real-Car-Data-And-Stats/Tire Models from TTC/`:
- 8, 10, 12, 14 psi variants of Hoosier LC0 16x7.5-10 on 8" rim
- PAC2002 format, Round 8 TTC data, fitted by Stackpole Engineering Services
- USE_MODE = 2 (Fy, Mx, Mz only — Fx coefficients all zero)

### Validation

- Generate Fy vs. slip angle at 3 loads (200N, 657N, 1000N) for each pressure file
- Verify peak mu matches PDY1 (~2.66)
- Verify load sensitivity: grip/load ratio decreases at higher loads
- Verify camber sensitivity direction matches sign of PDY3
- Symmetric Fx model: verify same peak mu as Fy
- **Telemetry Fx check**: extract peak braking G from AiM data, compare against model prediction at corresponding loads

## Component 2: Load Transfer Model

**File**: `src/fsae_sim/vehicle/load_transfer.py`
**Class**: `LoadTransferModel`

### Responsibility

Compute the vertical load on each of the four tires given speed, lateral acceleration, and longitudinal acceleration.

### Construction

All parameters from DSS, loaded via config:
- Mass: 278 kg with driver (210 kg car + 68 kg driver)
- CG height: 279.4 mm (physically confirmed)
- Wheelbase: 1549 mm
- Front track: 1194 mm, rear track: 1168 mm
- Static mass distribution: 92.4 / 117.6 kg (front/rear, car only); 45% front with driver
- Roll stiffness: 238 Nm/deg front, 258 Nm/deg rear
- Roll center heights: 88.9 mm front, 63.5 mm rear (static)
- Aero: 625 N downforce at 80 kph, 61% front, 431 N drag at 80 kph

### Methods

- `static_loads() -> (FL, FR, RL, RR)` — Corner weights from mass distribution.
- `aero_loads(speed) -> (delta_F_front, delta_F_rear)` — Downforce per axle, v² scaled from 80 kph reference, 61/39 split.
- `longitudinal_transfer(accel_g) -> float` — `delta_Fz = m * a * h_cg / wheelbase`. Positive accel shifts load rearward.
- `lateral_transfer(lateral_g, speed) -> (delta_Fz_front, delta_Fz_rear)` — 4-wheel lateral load transfer:
  - Geometric component: `delta_Fz_geo_f = m_f * a_lat * h_rc_f / track_f` (same for rear)
  - Elastic component: remaining lateral moment distributed by roll stiffness ratio `K_f / (K_f + K_r)`
  - Per-axle total: geometric + elastic, divided by track width
- `tire_loads(speed, lateral_g, longitudinal_g) -> (FL, FR, RL, RR)` — Combines static + aero + longitudinal + lateral into four normal loads.

### Key physics

Roll stiffness distribution (238/258 = 48%/52%) differs from static weight distribution (45%/55%). This difference determines the understeer/oversteer balance under lateral load transfer — a bicycle model would miss this.

### Validation

- Hand-calculate: 1g lateral at 80 kph — verify loads sum to weight + downforce
- Hand-calculate: 1.5g braking — verify no tire goes negative
- Combined 1g lateral + 0.5g longitudinal — verify diagonal loading makes physical sense

## Component 3: Steady-State Cornering Solver

**File**: `src/fsae_sim/vehicle/cornering_solver.py`
**Class**: `CorneringSolver`

### Responsibility

Given a segment's curvature, find the maximum speed at which all four tires remain within their grip limits simultaneously. Replaces the fixed 1.3G assumption.

### Construction

Takes a `PacejkaTireModel`, `LoadTransferModel`, and vehicle params (static camber front/rear, mass, roll camber coefficients from DSS: -0.5 front, -0.554 rear deg/deg).

### Core method

`max_cornering_speed(curvature, camber_front, camber_rear) -> float`

### Algorithm (iterative bisection)

1. Speed bounds: `v_low = 0`, `v_high = 50 m/s`
2. At candidate speed `v`:
   a. Required lateral acceleration: `a_lat = v² * curvature`
   b. Tire normal loads: `LoadTransferModel.tire_loads(v, a_lat, 0)`
   c. Roll angle: `total_lateral_force / total_roll_stiffness`
   d. Effective camber per wheel: static + roll_angle * roll_camber_coefficient
   e. Peak lateral force per tire: `PacejkaTireModel.peak_lateral_force(Fz, camber)`
   f. Total capacity: `Fy_FL + Fy_FR + Fy_RL + Fy_RR`
   g. Required force: `m * a_lat` (inertial mass, not downforce-augmented)
   h. If capacity >= required: feasible, try higher. Else: too high, try lower.
3. Converge to 0.1 m/s tolerance.

### Physics captured

- **Tire load sensitivity**: Degressive Pacejka curve means outside tire gains less grip than inside tire loses under load transfer.
- **Aero benefit**: Higher speed → more downforce → more grip, competing against increased load transfer.
- **Camber from roll**: Roll changes camber, affecting grip through PDY3/PKY3 coefficients.

### Validation

- Hairpin (~5m radius): ~30-40 kph predicted
- Long sweeper: approaches aero-limited speed
- Compare against AiM GPS speed at each Michigan corner — sim should predict >= actual (driver doesn't hit 100% of limit)

## Component 4: Updated VehicleDynamics

**File**: `src/fsae_sim/vehicle/dynamics.py` (replace in-place)

### Changes

**Constructor**: Adds `PacejkaTireModel`, `LoadTransferModel`, `CorneringSolver` alongside existing mass, CdA, ClA, Crr.

**Unchanged methods**: `drag_force`, `rolling_resistance_force`, `grade_force`, `total_resistance`, `acceleration`.

**Modified methods**:
- `max_cornering_speed(curvature, grip_factor)` — Delegates to `CorneringSolver`. The `grip_factor` is applied as a scaling factor on tire peak mu via Pacejka scaling coefficient LMUY.
- `resolve_exit_speed(entry_speed, segment_length, net_force, corner_limit)` — Same kinematics, `corner_limit` now from cornering solver.

**New methods**:
- `max_traction_force(speed) -> float` — Maximum drive force rear tires can transmit. Uses load transfer model for rear tire loads under acceleration, queries tire model for peak Fx, sums both rear tires.
- `max_braking_force(speed) -> float` — Maximum braking force from all four tires under braking load transfer.

**Removed**:
- `MAX_LATERAL_G = 1.3` constant
- Internal downforce iteration loop in old `max_cornering_speed`

## Component 5: Sim Engine Updates

**File**: `src/fsae_sim/sim/engine.py` (modify in-place)

### Changes

**Minimal**. The engine's per-segment loop structure stays the same.

**Traction limiting** (new):
```python
drive_force = min(powertrain_drive_force, dynamics.max_traction_force(speed))
```

**Braking force limiting** (new):
```python
brake_force = min(requested_brake_force, dynamics.max_braking_force(speed))
```

**Corner speed limits**: Same call site `dynamics.max_cornering_speed(curvature, grip_factor)` — implementation behind it changes, engine doesn't know.

**Unchanged**: Battery stepping, SOC/temp termination, state recording, strategy interface, replay mode.

## Config Changes

### New sections in `configs/ct16ev.yaml`

```yaml
tire:
  tir_file: "Real-Car-Data-And-Stats/Tire Models from TTC/Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
  static_camber_front_deg: -1.25
  static_camber_rear_deg: -1.25

suspension:
  roll_stiffness_front_nm_per_deg: 238
  roll_stiffness_rear_nm_per_deg: 258
  roll_center_height_front_mm: 88.9
  roll_center_height_rear_mm: 63.5
  roll_camber_front_deg_per_deg: -0.5
  roll_camber_rear_deg_per_deg: -0.554
  front_track_mm: 1194
  rear_track_mm: 1168
```

### New dataclasses in `vehicle.py`

- `TireConfig`: tir_file path, static camber front/rear
- `SuspensionConfig`: roll stiffnesses, roll center heights, roll camber coefficients, track widths
- `VehicleConfig` gains these two fields

## Validation Strategy

### Phase 1 — Component-level (during development)

Each component validated in isolation before integration:
- Tire: Fy curves at multiple loads, peak mu check, load sensitivity shape
- Load transfer: hand-calculated scenarios match model output
- Cornering solver: physically reasonable speeds at known curvatures

### Phase 2 — Telemetry comparison (after integration)

Using Michigan 2025 AiM data:
- **Corner speeds**: Predicted max vs. actual GPS speed at each corner. Sim should predict >= actual.
- **Fx validation**: Peak braking G from telemetry vs. symmetric tire model prediction. Flag discrepancies.
- **Full endurance**: 22-lap ReplayStrategy run. Compare total time, energy, final SOC. Target: match or improve on current ~5% accuracy.
- **Force-based strategies**: CoastOnly and ThresholdBraking lap times should be more realistic with real tire data.

### Phase 3 — Parameter sweep confidence

Small sweep: inverter torque 60-120 Nm. Verify:
- Higher torque → faster straights, same corner speeds (until traction-limited)
- Energy increases with torque
- No discontinuities

## Future Phase: MMM Diagrams (Not Built)

**Milliken Moment Method** — compute a full lateral force vs. yaw moment envelope at each speed. Provides:
- Understeer/oversteer characterization
- Stability margins
- Trim state analysis (how much steering input needed at each lateral G)

This is a vehicle dynamics analysis tool, not a lap time tool. Build after Tier 3 is validated and when the team wants to use the sim for setup/balance tuning beyond parameter sweeps.

## Build Order

Bottom-up component replacement:
1. Tire model (independent, testable against .tir file)
2. Load transfer model (independent, testable with hand calcs)
3. Cornering solver (depends on 1 & 2)
4. Update VehicleDynamics (depends on 1, 2, 3)
5. Update sim engine (depends on 4)
6. Full validation against telemetry (depends on 5)
