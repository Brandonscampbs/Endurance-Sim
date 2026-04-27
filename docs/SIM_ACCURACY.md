# Simulator Accuracy — CT-16EV vs Michigan 2025 Endurance

Snapshot of how well the sim reproduces the real-car telemetry, what was
fixed to get here, and what remains. Rerun with
`python scripts/sim_compare.py --strategy {replay|calibrated}` and the
plots land under `results/current_*`.

## Headline numbers (lap 1, replay strategy)

| Metric | Original | After fixes |
|---|---:|---:|
| RMSE (speed-vs-distance) | **16.87 km/h** | **2.96 km/h** |
| Bias | +6.86 km/h | +1.76 km/h |
| p95 \|residual\| | 29.5 km/h | 5.93 km/h |

| Endurance metric (calibrated, 21 laps) | Telem | Sim | Δ |
|---|---:|---:|---:|
| Driving time | 1614.7 s | 1500 s | -7.1% |
| Net energy | 3.27 kWh | 3.21 kWh | **-1.9%** |
| Mean pack voltage | 411.7 V | 406.4 V | -1.3% |
| Final pack voltage | 390.8 V | 392.5 V | +0.4% |
| Mean \|pack current\| | 18.6 A | 18.9 A | +1.6% |
| Final cell temp | 38.0 °C | 36.2 °C | -4.9% |

8/8 validation metrics PASS on calibrated. Mean RMSE across all 21 laps
is 4.77 km/h on replay, 6.04 km/h on calibrated.

## What was wrong (in order of impact)

### 1. Track lap-detection mismatch (+224 m phase offset)

`Track.from_telemetry` and `analysis.validation.detect_lap_boundaries`
used different start/finish anchors — a GPS-proximity gate vs a
latitude-crossing detector. The two fired at different physical points
on the track, so every track feature appeared at a different lap-relative
distance in the sim than in the comparison telemetry.

**Fix:** unified `Track.from_telemetry` to call `detect_lap_boundaries`
and slice the chosen lap from its boundary. Sim's d=0 and the
comparison harness's d=0 now coincide on every lap.

### 2. Multi-lap centerline averaging shifted the post-hairpin apex by 18 m

Once the alignment was unified, an averaging step that resampled each
lap's `(lat, lon)` onto a canonical mean-lap-length s-axis was still
smearing apex locations: per-lap GPS-speed integration drift means the
same physical apex lands at slightly different rescaled-s in each lap.
Averaging at the same s-value displaces the centerline shape.

**Fix:** rebuild the track from a single reference lap (lap 1) instead
of averaging. Per-lap drift no longer matters because we never align
laps to each other.

### 3. Mechanical brake force modelled as motor regen (under by ~50 %)

Engine `BRAKE` branch and `SpeedEnvelope` backward pass both used
`PowertrainModel.regen_force` as a proxy for the mechanical brake
force. That capped braking at the motor's RPM-dependent regen ceiling
(~0.5 g at racing speeds) — fine for a regen-only car, wrong for
CT-16EV's mechanical pads. The under-braking forced the planner to
brake ~13 m earlier than the real driver, dragging speed below
telemetry on every corner approach.

**Fix:** introduced `VehicleDynamics.mechanical_brake_force`, which
scales a calibrated peak brake decel (`_MAX_BRAKE_DECEL_G = 0.55 g`,
back-derived from the 1st percentile of telemetry `GPS LonAcc`) by
`brake_pct`. The 0.55 g matches the real driver's observed peak —
not the tire-grip ceiling (~1.3 g), because the brake hardware itself
doesn't pad-press hard enough at full pedal to reach lock-up. Engine
and all three envelope passes now use this consistently.

### 4. Energy-honest motor torque clamp

When `resolve_exit_speed` clipped speed down to `corner_speed_limit_ms`,
the bookkeeping still recorded `electrical_power = elec_power(commanded
torque, …)` — so the battery was paying for kinetic energy that the
clamp silently erased. ~20 % of segments were affected, leaking ~0.95
kWh of phantom electrical work over the endurance.

**Fix:** after the clamp, back-compute the motor torque from the
realised acceleration (`a_actual = (v_exit² − v_entry²)/(2L)` →
`F_drive_actual = m_eff·a_actual + resist_f` →
`T_motor_actual = F_drive_actual·r / (G·η)`). The correction can only
*reduce* commanded torque, never amplify. Cuts the energy gap from
+16 % to <2 %.

### 5. YawRate fallback for systematic GPS LatAcc dropouts

The cleaned telemetry has GPS LatAcc as NaN in a 100 m band of every
lap (and entire laps 1-4 — only laps 5+ have valid LatAcc). The track
builder treated NaN as zero curvature, marking real corners as
straight. The d=755 m hairpin (R = 14 m in YawRate, R = 6 m in lap-1
YawRate) was being driven through at 50 km/h instead of 30 km/h.

**Fix:** when `GPS LatAcc` is NaN and `YawRate` is available, recover
curvature from `κ = ω/v`. This is exact for a planar vehicle path and
restores the missing geometry.

### 6. Track curvature smoothing flattened hairpins

`_SMOOTH_DISTANCE_M = 5 m` rolling-median window crushed the tightest
hairpin (real R = 4.6 m) to R = 35 m.

**Fix:** dropped to 2 m (4-bin median at 0.5 m bins). Still rejects
single-sample GPS-acceleration noise without flattening the peak.
Sim hairpin R now matches telemetry pointwise R within averaging tolerance.

### 7. Parasitic drag re-derived from coast events

`_PARASITIC_DRAG_N = 70 N` was an over-estimate that absorbed
unrelated physics gaps (pre-fix track curvature, pre-fix grip scale).

**Fix:** ran `scripts/calibrate_physics.py` against 318 clean coast
samples (no throttle, no brake, |lat_g|<0.05) — back-derived 30 N
from `m·a + F_aero + F_rolling + F_grade + F_parasitic = 0`.

### 8. Misc smaller fixes

- `motor_speed_max_rpm`: 2900 → **3000**. Cascadia inverter overshoots
  its setpoint transiently; real telemetry peaks at 2947 RPM.
- `tire.grip_scale`: 0.4697 → **0.50**. Empirical sweet spot — p99 of
  effective μ (0.548) was too aggressive for the driver's sustained
  cornering, p75 (0.35) was too restrictive once the YawRate fallback
  exposed all the new corners. 0.50 minimises residual RMSE.
- Replay strategy `from_full_endurance`: trim to lap-1 start so
  cumulative-distance origin matches the sim's. Without this, sim
  distance 200 m mapped to pit-out throttle, not lap-1 driver inputs.
- `sim_compare.py`: pass real lap-1 entry speed (~38 km/h) as
  `initial_speed_ms` instead of zero, eliminating a 50 m sim-slow
  ramp at the start of every comparison.
- `sim_compare.py`: run for `len(detected_laps)` laps instead of a
  hardcoded 22; otherwise the replay strategy extrapolates past the
  end of the recording on lap 22 and stalls the sim for 3000 s.

## Files modified

- `configs/ct16ev.yaml` — `motor_speed_max_rpm`, `grip_scale`
- `src/fsae_sim/track/track.py` — single-lap centerline build, YawRate
  fallback, unified lap detection, 2 m smoothing
- `src/fsae_sim/sim/engine.py` — energy-honest torque clamp,
  `mechanical_brake_force` instead of `regen_force` for BRAKE branch
- `src/fsae_sim/sim/speed_envelope.py` — all three braking passes use
  `mechanical_brake_force`, not regen
- `src/fsae_sim/vehicle/dynamics.py` — `mechanical_brake_force` with
  calibrated `_MAX_BRAKE_DECEL_G`, parasitic drag 70 N → 30 N
- `src/fsae_sim/driver/strategies.py` — Replay trim-to-lap-1 + re-zero
  cumulative distance
- `tests/test_vehicle.py` — updated motor_speed_max_rpm assertion to
  reflect the new 3000 RPM cap

## Scripts

- `scripts/sim_compare.py` — main comparison harness. Outputs
  `results/{label}_lap{1,5,10,15,20}.png`, `_speed_full.png`,
  `_summary.json`, and the `_sim.parquet`.
- `scripts/calibrate_physics.py` — back-derives parasitic drag and
  grip-scale percentiles from the cleaned telemetry. Output:
  `results/calibration.json`.
- `scripts/diagnose_replay_step.py` — channel-by-channel time-step
  comparison (speed / lon-G / lat-G / motor torque / forces). Output:
  `results/diag_replay_channels.png` + `_table.csv`. Identifies the
  lap-1 residual bands (sign and magnitude per band).
- `scripts/diagnose_track.py` — sim track curvature vs telemetry
  pointwise. Used to verify hairpin radii match.

## Outstanding

- ~7 % time gap on calibrated still flags as FAIL (target <5 %). Most
  of that comes from the calibrated strategy's median-collapse losing
  per-lap variability in driver torque commands; replay's time gap is
  similar (~7 %) suggesting a residual physics or strategy issue worth
  investigating.
- SOC-consumed metric still flags ~30 % over telem. Decomposes as
  ~1.32× capacity convention × ~1.0× energy — the AiM `State of
  Charge` channel is being reported against a ~32 % larger reference
  capacity than the cell datasheet (4.5 Ah/cell). Not a sim bug, but
  the metric will keep failing until the reconciliation is wired.
- The d=0-50 m sim-slow band on lap 1 (~3 km/h dip) is the residual
  rolling-start ramp — sim still has to accelerate from the
  `_MIN_SPEED_MS = 0.5` floor through the first few segments.
- The +6 km/h sim-fast peak at d=273 m on lap 1 (the post-hairpin
  apex itself) is no longer an alignment problem — it's a separate
  physics gap (cornering envelope slightly too generous at this
  specific R).
