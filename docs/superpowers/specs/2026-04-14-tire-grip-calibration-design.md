# Tire Grip Calibration from Telemetry

## Problem

The Pacejka tire model fitted to TTC flat-belt rig data gives peak mu = 2.66. Real on-car telemetry shows peak lateral G of 1.22 (95th percentile). This 2.2x overprediction is the root cause of the simulation's accuracy problems:

- Corner speed limits are ~2x too high (70 km/h through 14m-radius hairpins vs 35 km/h real)
- The sim never decelerates for corners, producing a flat ~60 km/h speed profile
- Energy consumption is 37% too low because there are no acceleration/deceleration cycles
- Driving time error is 10.3%

The TTC flat-belt rig is known to over-predict on-car grip. The .tir file's scaling factors (LMUY, LMUX) exist to calibrate rig data to real-world performance.

## Solution

Calibrate the Pacejka model's LMUY scaling factor from endurance telemetry. Extract the car's real effective peak friction coefficient from lateral acceleration data, compare to the Pacejka model's peak, and apply the ratio as LMUY.

## Extraction Method

From AiM endurance telemetry, for each sample where the car is cornering:

```
Filter: |GPS_LatAcc| > 0.3g AND GPS_Speed > 15 km/h

lateral_force = mass * |a_lat_g| * g
downforce     = 0.5 * rho * ClA * v^2
total_normal  = mass * g + downforce
effective_mu  = lateral_force / total_normal
```

Take the 95th percentile of `effective_mu` as the car's peak grip. The drivers pushed through corners near the traction limit during endurance (confirmed from telemetry analysis), so this represents the real vehicle grip capability at endurance operating conditions (tire temperature, pressure, wear).

Compare to Pacejka peak mu at representative normal load:

```
pacejka_mu = peak_lateral_force(Fz_representative) / Fz_representative
grip_scale = effective_mu_95th / pacejka_mu
```

Where `Fz_representative` is the average per-tire normal load during cornering (static weight / 4 + average downforce / 4).

## Changes

### 1. PacejkaTireModel: `apply_grip_scale(scale)` method

New method that multiplies the loaded LMUY and LMUX scaling factors by the given scale. Called once after construction. Affects all force calculations through the existing Magic Formula machinery:

- `lateral_force()` — peak force (D = mu * Fz) scales down
- `longitudinal_force()` — same, via mirrored LMUY
- `peak_lateral_force()` — scales down (used by cornering solver, cornering drag, traction limits)
- `peak_longitudinal_force()` — scales down (used by traction limits, friction circle)
- Cornering stiffness Kya is preserved (B = Kya / (C*D) compensates), so initial tire response is unchanged; the tire saturates earlier at lower peak force

File: `src/fsae_sim/vehicle/tire_model.py`

### 2. TireConfig: `grip_scale` field

New optional field, default 1.0. Stored in vehicle YAML.

File: `src/fsae_sim/vehicle/vehicle.py` (TireConfig dataclass)

### 3. SimulationEngine: apply grip_scale on construction

After constructing `PacejkaTireModel`, call `apply_grip_scale(tire_cfg.grip_scale)`.

File: `src/fsae_sim/sim/engine.py`

### 4. Extraction function: `extract_tire_grip_scale()`

New function that takes AiM telemetry DataFrame, vehicle params (mass, ClA), and a PacejkaTireModel, and returns the computed grip_scale value.

File: `src/fsae_sim/analysis/telemetry_analysis.py`

### 5. Config update: ct16ev.yaml

Add calibrated `grip_scale` value to the tire section.

File: `configs/ct16ev.yaml`

### 6. Validation script update

Update `scripts/validate_tier3.py` to report real vs Pacejka grip and the calibrated scale factor.

## Validation Criteria

After applying the calibrated grip_scale:

1. `max_cornering_speed()` for tight corners (curvature 0.05-0.07) should give speeds within 20% of telemetry corner speeds
2. Re-running the CalibratedStrategy 22-lap sim should show a speed profile that varies between 30-60 km/h (not flat at 60)
3. Energy consumption should move toward the telemetry value (currently 37% low)
4. Driving time error should improve from 10.3%

## Not in scope

- Load-dependent grip correction (mu vs Fz curve recalibration) — single scale factor is sufficient given the data quality
- Temperature-dependent grip — no thermal tire model yet
- Separate longitudinal grip calibration — no Fx data available; same scale applied to both
- Autocross data calibration — data not available; endurance peak lateral G is sufficient since drivers pushed corners near the limit
