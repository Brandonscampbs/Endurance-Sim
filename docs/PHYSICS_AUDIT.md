# EV FSAE Endurance Simulator Physics Audit

This document is a physics and validation audit of the current endurance simulator. It is intentionally focused on whether the simulation is true, predictive, and internally consistent. It is not a code-style review, and it does not propose small refactors as a substitute for fixing the model.

The short version: the repository is currently closer to a telemetry-calibrated replay and visualization tool than a validated EV FSAE endurance simulator. Several outputs can look plausible because the same real-car data is used to construct the track, tune the driver, tune grip, calibrate the battery, and validate the final result. That is circular validation, not proof.

## Scope

Reviewed areas:

- Telemetry loading and preprocessing.
- Track generation from GPS and lap detection.
- Driver strategy generation and replay.
- Speed envelope construction.
- Segment integration and vehicle dynamics.
- Tire, load transfer, aero, braking, and powertrain models.
- Battery, current, voltage, charge, and thermal model.
- Validation metrics and existing result files.
- Backend simulation runner and visualization export path.
- Configs for CT16EV and CT17EV.
- Existing tests and documented issue list.

Out of scope:

- Frontend/UI review except where the backend visualization computes misleading physics values.
- Full derivation of a replacement model.
- Verification against raw original car logs beyond what is available in this repo.

## Overall Verdict

This simulator should not yet be used to answer engineering questions like:

- How much faster will CT17EV be than CT16EV?
- How much energy will an endurance run consume?
- How many net pack amp-hours and how much energy margin do we need?
- What is the benefit of a different aero package, tire, torque map, battery, or driver strategy?
- What is the fastest feasible lap time?

It can currently be used for:

- Exploring data structures and plotting telemetry.
- Producing approximate replay-style animations.
- Running rough sensitivity experiments if the results are treated as qualitative only.
- Finding which parts of the pipeline need physical validation.

The main reason is not one isolated bug. The full pipeline leaks measured data into the sim at multiple points, and the physics integrator is not energy-conservative. That combination can hide large physical errors while still producing believable plots.

## Evidence Snapshot

Existing result files indicate that the tuned models still miss key endurance metrics:

- Current calibrated summary: sim total time about 1495.3 s vs telemetry driving time about 1608.8 s. The sim is roughly 7 percent too fast.
- Current replay summary: sim total time about 1516.0 s vs telemetry driving time about 1608.8 s. Even replay-style inputs are roughly 6 percent too fast.
- Calibrated energy: sim net energy about 3.221 kWh vs telemetry about 3.271 kWh, which looks close.
- Current calibrated charge: sim net charge about 7.79 Ah vs telemetry about 8.04 Ah, roughly 3.2 percent low.
- Current replay charge: sim net charge about 8.11 Ah vs telemetry about 8.04 Ah, roughly 0.8 percent high.
- Calibrated lap behavior repeats nearly identical laps while the real endurance data has substantial lap-to-lap variation.

The charge and energy agreement are encouraging, but still not enough by themselves to trust the model as a tune optimizer. A pack model can match integrated Ah and kWh while still producing wrong current timing, voltage sag, temperature, and limiting behavior.

A `pytest -q` run produced real failures in effective inertia, load transfer, and configuration expectations, plus sandbox-related temp directory errors. The repo is not internally self-consistent at the test level either.

## Pipeline Map

The current pipeline is approximately:

1. Load cleaned endurance telemetry from CSV.
2. Alias or derive speed and distance fields.
3. Detect laps from GPS latitude crossing.
4. Build a track from averaged GPS traces.
5. Build a driver strategy from the same telemetry.
6. Derive grip scale and battery calibration from the same telemetry.
7. Run the segment-based simulator.
8. Compare the result against the same telemetry.
9. Export data for visualization with another simplified force model.

That is not an independent validation loop. It is a calibration loop with many hidden constraints.

## Highest-Severity Findings

### 1. Validation Is Circular

The same dataset is used for too many roles:

- Track geometry is built from the endurance telemetry.
- Driver zones are generated from the endurance telemetry.
- Segment speed caps come from the endurance telemetry.
- Tire grip scale is derived from the endurance telemetry.
- Battery pack behavior is calibrated from the endurance telemetry.
- The final result is validated against the endurance telemetry.

Relevant files:

- `src/fsae_sim/track/track.py`
- `src/fsae_sim/driver/strategies.py`
- `src/fsae_sim/analysis/telemetry_analysis.py`
- `src/fsae_sim/vehicle/battery_model.py`
- `backend/services/sim_runner.py`
- `src/fsae_sim/analysis/validation.py`

Why this is wrong:

If you use a lap to build the track, infer the driver, tune the grip, fit the battery, cap local speeds, and then validate on that same lap, the validation result cannot prove predictive accuracy. It mostly proves the software can reprocess the same dataset.

What this breaks:

- CT16EV validation credibility.
- CT17EV extrapolation.
- Any setup comparison.
- Any claim about endurance energy margin.
- Any lap-time prediction under different driver or strategy assumptions.

Minimum fix:

- Split telemetry into calibration and holdout sets.
- Build track geometry from independent survey, cone map, lidar, RTK GPS, or a separately validated racing-line map.
- Calibrate tire and battery models on separate tests, not the same endurance run used for validation.
- Remove telemetry-derived speed caps from prediction mode.

### 2. Segment Integration Is Not Physically Conservative

The engine resolves each segment using simplified entry-state forces and then directly clamps speed to corner limits. If the vehicle arrives too fast, speed can be reduced without simulating braking force, tire work, time loss, or energy dissipation.

Relevant files:

- `src/fsae_sim/sim/engine.py`
- `src/fsae_sim/vehicle/dynamics.py`

Specific problems:

- Entry speed can be clamped to the corner limit.
- Exit speed can be clamped after a constant-acceleration solve.
- Power is computed later at a different operating point than the force balance.
- Segment resistance is evaluated at entry speed for some calculations and average speed for others.
- There is no robust predictor-corrector integration for speed, force, power, and battery state.

Why this is wrong:

A vehicle cannot simply lose kinetic energy because a segment has a lower speed cap. It must brake, coast, scrub speed through tire slip, hit a barrier, or leave the track. Those choices have time and energy consequences.

What this breaks:

- Lap time.
- Brake energy and regen energy.
- Tire utilization.
- Current draw.
- Thermal state.
- Any corner-entry or corner-exit comparison.

Minimum fix:

- Replace clamp-first segment logic with a time-domain or distance-domain integrator that solves force balance continuously.
- Apply braking/coasting commands before the speed constraint is reached.
- Account for the work done when kinetic energy changes.
- Use a consistent operating point or predictor-corrector method for force, power, and battery state.

### 3. Speed Envelope and Runtime Engine Use Different Physics

The speed envelope is supposed to describe feasible speed. The runtime engine then uses it as a constraint. But the envelope and the runtime engine disagree about braking and power limits.

Relevant files:

- `src/fsae_sim/sim/speed_envelope.py`
- `src/fsae_sim/sim/engine.py`
- `src/fsae_sim/vehicle/dynamics.py`
- `src/fsae_sim/vehicle/powertrain_model.py`

Specific problems:

- The backward pass uses a regen-force proxy for braking feasibility.
- The runtime engine uses mechanical brake force, not that same regen braking model.
- The forward pass ignores BMS current limits.
- The envelope can assume acceleration that the battery cannot actually supply later.
- Combined-slip correction is effectively dead in the production path because the vehicle dynamics call does not pass longitudinal demand into the cornering solver.

Why this is wrong:

The car's feasible speed envelope depends on the same tires, brakes, torque limits, current limits, aero, and vehicle state that the runtime sim uses. If the envelope and runtime model use different physics, the cap can be either too optimistic or too conservative.

What this breaks:

- Fastest-lap calculation.
- Corner entry speed.
- Braking distance.
- Acceleration zones.
- Any strategy that relies on the envelope to decide throttle or brake.

Minimum fix:

- Use one shared acceleration/braking feasibility model for both envelope and runtime integration.
- Include BMS current limits and voltage limits in the envelope.
- Include mechanical brake capacity in the backward pass.
- Make combined-slip demand active when longitudinal force and lateral force coexist.

### 4. Tire Model Is Not Ground Truth

The current tire model is not a validated representation of the specific tire in the actual operating conditions.

Relevant files:

- `src/fsae_sim/vehicle/tire_model.py`
- `src/fsae_sim/vehicle/cornering_solver.py`
- `src/fsae_sim/vehicle/dynamics.py`
- `src/fsae_sim/analysis/telemetry_analysis.py`

Specific problems:

- The code notes that longitudinal coefficients are transplanted from a different tire/test.
- Grip is scaled empirically from telemetry.
- Peak force is often taken from simplified closed-form or peak approximations.
- Production traction and braking paths mostly use pure-slip peak force, not full combined-slip Pacejka.
- Tire temperature, pressure, wear, track surface, and transient relaxation are not modeled.
- The turn-direction/camber-sign convention is not robustly validated.

Why this is wrong:

FSAE lap time and energy are highly sensitive to tire force. If tire limits are fitted to observed telemetry from the validation run, the model is not predicting grip. It is inferring what grip must have been for that run.

What this breaks:

- Cornering speed.
- Braking distance.
- Exit traction.
- Effect of downforce.
- Effect of mass.
- Effect of tire selection or setup.

Minimum fix:

- Use a consistent tire dataset for the actual tire.
- Validate pure lateral, pure longitudinal, and combined-slip behavior separately.
- Use tire model outputs in the main force balance, not just in helper functions.
- Keep telemetry-derived grip scale as an explicit calibration mode, not default physics truth.

### 5. Driver Model Is a Telemetry Compressor, Not a Driver

The calibrated strategy maps track segments to observed behavior. That can help replay one event, but it cannot predict how a driver would behave under changed vehicle performance.

Relevant files:

- `src/fsae_sim/driver/strategies.py`
- `src/fsae_sim/analysis/telemetry_analysis.py`
- `src/fsae_sim/sim/engine.py`

Specific problems:

- Segment zones are derived from observed telemetry.
- Speed caps can be derived from observed speed percentiles.
- The same lap behavior repeats across simulated laps.
- There is little real feedback control.
- It does not model driver adaptation to extra grip, less power, thermal derating, traffic, cones, fatigue, or caution periods.
- Replay mode mixes measured torque/brake inputs with simulated speed dynamics, which is neither pure replay nor pure prediction.

Why this is wrong:

A predictive driver model should respond to vehicle state and upcoming track constraints. A telemetry compressor only repeats what happened. If the car changes, the repeated commands may be infeasible, suboptimal, or nonsensical.

What this breaks:

- CT17EV predictions.
- Strategy comparisons.
- Driver improvement estimates.
- Thermal derating consequences.
- Energy-saving driving modes.

Minimum fix:

- Separate replay mode from predictive mode.
- In replay mode, replay measured speed, voltage, current, torque, and time directly when the goal is data viewing.
- In predictive mode, use a controller with lookahead, braking points, throttle limits, lateral margin, and energy objectives.
- Do not allow observed speed caps in predictive mode.

### 6. Track Model Is Not a Reliable Track Model

The track is generated from real-car GPS and speed signals. That is not equivalent to a measured track centerline or racing line.

Relevant files:

- `src/fsae_sim/data/loader.py`
- `src/fsae_sim/track/track.py`
- `src/fsae_sim/analysis/validation.py`

Specific problems:

- Cleaned data aliases `GPS Speed` to `LFspeed`.
- Distance is recomputed from speed integration.
- Lap detection uses a simple GPS latitude crossing.
- GPS traces from laps are averaged into a centerline-like path.
- The result is closer to an averaged driven path than a physical track definition.
- Curvature comes from smoothed GPS derivatives, which are very sensitive to noise and smoothing choices.
- Grade is interpolated from GPS slope and is not strongly validated.

Why this is wrong:

Track curvature drives lateral acceleration, tire load, downforce usefulness, drag, and braking points. A noisy or over-smoothed curvature profile can make the car appear too fast or too slow in corners.

What this breaks:

- Corner speed.
- Distance traveled.
- Lap time.
- Brake points.
- Local energy consumption.
- Any comparison across tracks or layouts.

Minimum fix:

- Define the track from a cone map, surveyed path, RTK GPS, or hand-validated centerline/racing line.
- Treat telemetry-derived racing line as data, not ground truth.
- Validate lap length against known event distance.
- Validate curvature against video, map, or lateral acceleration independent of the same speed signal used in the sim.

### 7. Battery Charge, Voltage, and Thermal Model Need More Validation

The battery model now validates against net pack amp-hours and net V*I energy, not the displayed AiM/BMS SOC channel. That is the right scoring-oriented target, but the model still needs stronger validation of current timing, voltage sag, and thermal behavior.

Relevant files:

- `src/fsae_sim/vehicle/battery_model.py`
- `backend/services/sim_runner.py`
- `src/fsae_sim/analysis/validation.py`

Specific problems:

- Pack model is calibrated from the same telemetry used for validation.
- Internal resistance is not fully temperature-dependent.
- OCV extrapolation can extend beyond calibration range.
- Parallel cell sharing is assumed ideal.
- OCV hysteresis is not modeled.
- The thermal model is lumped and does not represent module gradients or airflow.
- Displayed BMS/AiM SOC is an estimator and should not be used as a sim-vs-telemetry energy metric.

Why this is wrong:

Endurance viability depends on amp-hours used, current limits, voltage sag, and thermal derating. Matching net kWh alone does not validate those states.

What this breaks:

- Endurance finish margin.
- Thermal limit prediction.
- Voltage sag prediction.
- Current limit prediction.
- Regen acceptance.
- CT17EV battery extrapolation.

Minimum fix:

- Validate the battery model on independent current/voltage/temperature logs.
- Treat displayed BMS/AiM SOC as diagnostic only; use integrated current for scored charge usage.
- Add temperature dependence for resistance and capacity where needed.
- Validate pack current limits against actual BMS behavior.
- Track energy, charge, voltage, current, and temperature errors separately.

### 8. Powertrain Model Has Mixed Physical and Firmware-Derived Logic

The powertrain model combines physical torque/power relationships with firmware-like constants and empirical caps. Some of this may be useful, but the boundaries are unclear.

Relevant files:

- `src/fsae_sim/vehicle/powertrain_model.py`
- `configs/ct16ev.yaml`
- `configs/ct17ev.yaml`

Specific problems:

- Tire radius for wheel speed and force conversion is a hardcoded unloaded value.
- Vehicle dynamics may use loaded radius elsewhere, creating inconsistent wheel speed and force conversion.
- The speed envelope uses max motor torque without BMS current limits.
- Regen force is modeled even though the CT16EV runtime path uses mechanical braking and forces motor torque to zero while braking.
- Coast electrical behavior includes a simplified back-EMF rectification branch.
- Config `drivetrain_efficiency` and hardcoded gearbox efficiency conventions are easy to misuse.
- Firmware constants such as `lvcu_power_constant` are not clearly separated from physics constants.

Why this is wrong:

Wheel torque, motor RPM, voltage, current, and power limits must be solved consistently. Mixing unloaded tire radius, loaded tire radius, firmware caps, and physical torque curves can create plausible but wrong force and power.

What this breaks:

- Acceleration prediction.
- Current draw.
- Field weakening behavior.
- Regen prediction.
- Gear ratio sensitivity.
- CT17EV extrapolation.

Minimum fix:

- Use one rolling radius convention consistently.
- Make firmware-command model separate from physical actuator capability.
- Validate torque command, motor torque, inverter current, DC current, and wheel force against logs.
- Include pack voltage and current limits in every place where available power matters.

### 9. Braking Model Is Too Simplified

The braking path does not model the actual hydraulic system or brake balance.

Relevant files:

- `src/fsae_sim/vehicle/dynamics.py`
- `src/fsae_sim/sim/engine.py`
- `src/fsae_sim/sim/speed_envelope.py`

Specific problems:

- Brake force is a linear scale of tire capacity.
- No pedal pressure to line pressure mapping.
- No brake bias model.
- No rotor/caliper/pad model.
- No wheel lock or ABS-like behavior.
- Speed envelope braking uses regen-like force rather than the same mechanical brake model.
- Brake thermal state is absent.

Why this is wrong:

FSAE braking performance is not simply "brake pedal percent times tire limit." Mechanical bias and load transfer determine which axle locks first and how much deceleration is possible.

What this breaks:

- Braking distance.
- Corner entry speed.
- Tire utilization.
- Brake energy.
- Replay of brake traces.

Minimum fix:

- Model pedal pressure, master cylinder, caliper piston area, rotor radius, pad coefficient, and bias.
- Apply wheel lock constraints per axle or per wheel.
- Use the same braking model in the envelope and runtime engine.
- Keep regen braking separate from mechanical braking and enforce battery acceptance limits.

### 10. Load Transfer and Aero Are Under-Validated

The sim has a load transfer model, but important parameters are hardcoded or uncertain.

Relevant files:

- `src/fsae_sim/vehicle/load_transfer.py`
- `src/fsae_sim/vehicle/dynamics.py`
- `configs/ct16ev.yaml`
- `configs/ct17ev.yaml`

Specific problems:

- CG height and distribution are defaults unless explicitly configured.
- Downforce distribution is defaulted, not clearly measured.
- Anti-dive and anti-squat are absent.
- Transient pitch/roll dynamics are absent.
- Damping and compliance are absent.
- Aero coefficients use ambiguous naming: `drag_coefficient` appears to be CdA when frontal area is set to 1.0.
- Downforce treatment is not consistently validated across normal load, rolling resistance, and cornering calculations.

Why this is wrong:

Load transfer changes tire capacity. Aero changes both drag and tire normal load. Small errors here can move lap time significantly, especially for an FSAE car with aero.

What this breaks:

- Cornering speed.
- Braking capacity.
- Traction on exit.
- Aero sensitivity.
- CT16EV vs CT17EV comparisons.

Minimum fix:

- Put measured CG, wheelbase, track widths, roll stiffness, aero balance, and CdA/ClA conventions in config.
- Validate normal load distribution against known static weights and expected aero maps.
- Validate load transfer against hand calculations before using it in the lap sim.

### 11. CT17EV Configuration Is Not Ready for Prediction

The CT17EV config appears incomplete relative to the CT16EV physics path.

Relevant files:

- `configs/ct17ev.yaml`
- `src/fsae_sim/sim/engine.py`
- `src/fsae_sim/vehicle/vehicle.py`

Specific problems:

- Missing tire section means no Pacejka model for CT17EV.
- Missing suspension section means no load-transfer/cornering solver path.
- Missing or unclear downforce coefficient means the fallback model may run with no downforce.
- Torque and firmware constants appear stale or guessed.

Why this is wrong:

A CT17EV prediction without tire, suspension, aero, battery, and powertrain validation is not a CT17EV prediction. It is a legacy fallback vehicle with some changed numbers.

What this breaks:

- All CT17EV lap-time predictions.
- CT17EV energy predictions.
- Design trade studies.
- Claims about improvement over CT16EV.

Minimum fix:

- Build a complete CT17EV config with measured or explicitly assumed tire, suspension, aero, powertrain, and battery parameters.
- Fail loudly if a config lacks required predictive-model sections.
- Do not silently fall back to legacy dynamics for design comparison runs.

### 12. Visualization Export Has Separate Incorrect Physics

The visualization export path computes additional forces and load transfer instead of faithfully displaying the sim's physics state.

Relevant file:

- `backend/services/visualization_export.py`

Specific problems:

- Driver mass is added again even though CT16EV config mass already includes driver.
- Regen force sign handling can make braking appear like forward force.
- Visualization load transfer uses simplified constants separate from the sim.
- Real telemetry force estimates use their own torque-to-force conversion.

Why this is wrong:

Visualization can make users believe the sim is physically explaining something when it is actually showing a second, different, simplified model.

What this breaks:

- Debugging.
- Engineering interpretation.
- Comparisons between real and simulated forces.
- Trust in displayed load transfer and tire utilization.

Minimum fix:

- Visualization should display fields computed by the sim, not recompute physics differently.
- If it must estimate missing real-car quantities, label them as estimates and use the same conventions as the main model.
- Fix mass and sign conventions.

## Subsystem Audit

### Telemetry Loading

Primary concern: the cleaned data pipeline changes the meaning of signals.

Issues:

- `GPS Speed` is aliased to `LFspeed`.
- Distance is recomputed from speed integration.
- The distance channel name implies GPS distance, but the source can be wheel speed.
- Front-left wheel speed can differ from vehicle speed due to slip, wheel radius, turning path, locking, sensor scaling, and filtering.

Impact:

- Validation speed and track distance are not fully independent of the vehicle model assumptions.
- GPS-based and wheel-speed-based quantities are mixed.
- Lap distance and curvature validation can be biased.

### Lap Detection

Primary concern: simple median-latitude crossing is not robust.

Issues:

- The start/finish line is inferred from GPS latitude behavior.
- No explicit start line geometry is used.
- This method is track-specific and can fail on layouts with different orientation.
- Detection errors affect lap selection, track generation, driver model, and validation.

Impact:

- One bad boundary corrupts every downstream lap-based calculation.

### Track Generation

Primary concern: averaged GPS lap path is not a physical track model.

Issues:

- Averaging driven laps can blur different racing lines.
- Smoothing can erase tight curvature.
- GPS noise can create fake curvature.
- Interpolation by distance assumes the distance channel is valid.
- Grade from GPS slope is likely noisy.

Impact:

- The corner speed envelope is only as good as curvature.
- The model can be too fast if real tight corners are smoothed.
- The model can waste energy if fake curvature or grade is introduced.

### Driver Strategy

Primary concern: telemetry-derived segment commands are not a predictive driver.

Issues:

- Per-segment medians remove transient behavior.
- Observed caps leak validation data.
- Segment lookup has no real anticipation except what is baked into telemetry.
- Repeated-lap behavior misses real endurance variability.
- Replay torque/brake semantics can conflict with simulated speed and powertrain state.

Impact:

- Good-looking plots can be produced without the model understanding why the driver did something.

### Vehicle Dynamics

Primary concern: the dynamics are a collection of useful approximations, not one coherent solver.

Issues:

- Entry-state force evaluation.
- Constant acceleration per segment.
- Post-solve speed clamping.
- Pure tire peak forces used in places where combined slip matters.
- Load transfer simplified and under-validated.
- Brake force simplified.
- Effective mass uses a questionable drivetrain-efficiency treatment on reflected rotor inertia.

Impact:

- Errors can cancel in one dataset and explode under a vehicle change.

### Tire and Cornering

Primary concern: tire force is the dominant lap-time model, and it is empirical here.

Issues:

- Mixed tire coefficient sources.
- Grip scale fitted from car data.
- Temperature and pressure absent.
- Full combined slip mostly inactive.
- No tire relaxation or transient behavior.
- No yaw moment equilibrium for realistic understeer/oversteer behavior.

Impact:

- The sim cannot confidently predict changes in aero, mass, tire, or setup.

### Aero

Primary concern: coefficients and reference area conventions are unclear.

Issues:

- `drag_coefficient` appears to be used like CdA when `frontal_area_m2` is 1.0.
- Same risk exists for downforce coefficient conventions.
- Downforce balance is defaulted or not measured in config.
- Air density is fixed ISA standard, not event conditions.

Impact:

- Drag and downforce sensitivity can be wrong even if the numeric result looks reasonable for one config.

### Brakes and Regen

Primary concern: mechanical braking, regen braking, and envelope braking are not one consistent system.

Issues:

- Runtime CT16EV braking is mechanical.
- Envelope braking uses regen-like force.
- Regen electrical power does not feed back into brake feasibility in the runtime path.
- Battery acceptance limits are not strongly represented in regen.
- Mechanical brake model lacks actual hardware parameters.

Impact:

- Corner entry, energy recovery, and brake force are not physically linked.

### Powertrain

Primary concern: physical capability and firmware command behavior are entangled.

Issues:

- Hardcoded tire radius for motor RPM and wheel force.
- LVCU torque command uses firmware-derived constants that are not clearly unit-checked.
- Speed envelope ignores current-limited torque.
- Motor/inverter efficiency treatment is not validated against logs.
- Field weakening and voltage/current limits need a single authoritative model.

Impact:

- Current draw, torque availability, and acceleration can be wrong in exactly the regions that matter for endurance.

### Battery

Primary concern: current, voltage, charge, and temperature are not jointly validated strongly enough.

Issues:

- Same telemetry used for calibration and validation.
- Displayed SOC disagreement is not a scoring metric; net Ah and net kWh are the validation targets.
- Lumped thermal model.
- Ideal parallel-cell sharing.
- No hysteresis.
- Limited temperature dependence.
- OCV extrapolation risk.

Impact:

- Endurance feasibility conclusions are unsafe.

### Validation

Primary concern: many validation metrics are not independent, and pass/fail thresholds are loose.

Issues:

- Data used for calibration is reused for validation.
- Mean speed, distance, and time are algebraically related.
- Total distance comparison can be misleading if sim states omit final segment endpoint or lap count differs.
- Full-endurance tests appear more diagnostic than proof-oriented.
- Some tests encode stale expected constants.

Impact:

- Passing tests do not imply physical truth.

## Why the Existing Results Are Misleading

The current summaries show a typical symptom of an overfit simulator:

- Net kWh looks close.
- Displayed SOC is not a trusted energy estimator and is ignored for validation.
- Lap time is badly fast.
- Replay is still fast.
- Speed RMSE remains material.
- Lap variability is not reproduced.

This means the model is probably matching one integrated quantity through calibration while missing the underlying state evolution.

For endurance, the underlying state evolution matters more than one integrated number. A car can consume the right total energy in the model for the wrong reasons:

- Too much time at too low drag.
- Too little braking loss.
- Wrong voltage sag.
- Wrong current profile.
- Wrong motor efficiency.
- Wrong pack resistance.
- Wrong tire drag.
- Wrong coast loss.
- Wrong speed profile.

Those errors can cancel on CT16EV and then fail badly on CT17EV.

## Recommended Fix Order

### Phase 1: Stop Hidden Data Leakage

Do this before adding more model detail.

- Add explicit modes: `replay`, `calibration`, `prediction`, and `validation`.
- In prediction mode, forbid telemetry-derived speed caps.
- In validation mode, require a holdout dataset that was not used for calibration.
- Make pack calibration optional and store calibration provenance.
- Make grip scale provenance explicit.
- Fail if validation and calibration point to the same run unless explicitly allowed.

### Phase 2: Fix Conservation in the Engine

This is the most important physics correction.

- Replace speed clamping with physical braking/coasting integration.
- Use predictor-corrector or smaller time/distance integration.
- Use one operating point convention for force, torque, power, and battery current.
- Track kinetic energy changes explicitly.
- Make mechanical braking, regen, drag, rolling resistance, and tire losses appear in energy accounting.

### Phase 3: Unify Envelope and Runtime Physics

- Use the same acceleration function in forward envelope and runtime.
- Use the same braking function in backward envelope and runtime.
- Include BMS limits in the envelope.
- Include mechanical brake limits in the envelope.
- Activate combined-slip constraints when longitudinal and lateral demand overlap.

### Phase 4: Validate Track and Driver Separately

- Build or import a known track map.
- Validate lap length and curvature independently.
- Separate replay from predictive driver behavior.
- Build a driver with lookahead braking, throttle, lateral margin, and energy saving.
- Validate one-lap speed trace on a holdout lap.

### Phase 5: Validate Tire, Aero, Brakes, and Powertrain

- Tire: pure lateral, pure longitudinal, combined slip, load sensitivity.
- Aero: CdA, ClA, balance, speed dependence, air density.
- Brakes: pressure, bias, torque, lockup.
- Powertrain: torque command, motor torque, DC current, voltage, efficiency.

### Phase 6: Validate Battery and Thermal Model

- Fit battery on one dataset.
- Validate voltage, current, charge, and temperature on another dataset.
- Keep internal coulomb-counted SOC separate from displayed BMS/AiM SOC and do not score against displayed SOC.
- Add temperature-dependent resistance if needed.
- Add module/cell gradients only after the lumped model is proven insufficient.

### Phase 7: Fix CT17EV Config and Prediction Path

- Require complete CT17EV tire, suspension, aero, powertrain, and battery parameters.
- Disable silent fallback to legacy dynamics for prediction.
- Create a config validation test that fails on missing physics sections.
- Compare CT17EV only after CT16EV holdout validation is acceptable.

## Suggested Acceptance Criteria

Before calling the simulator predictive, require something like:

- Track lap length error under 1 percent against independent measurement.
- Holdout lap time error under 2 percent without observed speed caps.
- Speed trace RMSE under 3 to 4 km/h on holdout laps.
- Net charge error under 5 percent on holdout endurance data, measured in Ah.
- Net energy error under 5 percent on holdout endurance data.
- Pack voltage RMSE and bias within validated electrical-model limits.
- Peak current and current-limit timing aligned with logs.
- Tire-limited corner speeds validated on at least several corners not used for grip fitting.
- Braking zones validated with measured brake pressure or a credible proxy.
- CT17EV prediction blocked unless required CT17EV physical parameters are present.

These numbers are suggestions. The important part is that they must be checked on holdout data and must not depend on telemetry-derived speed caps.

## Risk Ranking

Critical:

- Circular validation and calibration leakage.
- Non-conservative speed clamping and segment integration.
- Envelope/runtime braking and power-limit mismatch.
- Tire model fitted to the same telemetry it validates against.
- CT17EV fallback dynamics.

High:

- Driver strategy is not predictive.
- Track curvature and distance are derived from weak telemetry preprocessing.
- Current, voltage, and thermal states still need stronger holdout validation despite close Ah/kWh.
- Combined-slip is mostly absent from runtime force limits.
- Mechanical brake model is not hardware-based.

Medium:

- Aero coefficient convention ambiguity.
- Effective mass/inertia treatment.
- Visualization recomputes incorrect physics.
- Air density fixed to standard conditions.
- Grade and curvature smoothing choices not validated.

Low but still important:

- Test expectations stale relative to config/code.
- Performance limitations from Python loop.
- Plot scaling and visual diagnostics can hide bias.
- Documentation and naming inconsistencies.

## Practical Interpretation for the Team

Treat current outputs as "debuggable estimates," not engineering truth.

If the sim says CT17EV is faster, uses less charge, or can finish endurance with a certain Ah/energy margin, that result should be considered unproven until:

- CT16EV can validate on holdout data.
- The engine conserves energy and time through braking/cornering events.
- Tire and battery models are validated independently.
- CT17EV config has complete measured assumptions.

The most dangerous failure mode is false confidence. The simulator has enough detail to look sophisticated, but many of those details are either calibrated from the answer, bypassed in the runtime path, or inconsistent across subsystems.

## File-Level Notes

`src/fsae_sim/sim/engine.py`

- Central integration logic.
- Contains speed clamping and segment force/power operating-point mismatch.
- Applies telemetry-derived speed caps through command metadata.
- Uses mechanical braking path that disagrees with speed envelope braking.

`src/fsae_sim/sim/speed_envelope.py`

- Builds forward/backward speed limits.
- Ignores BMS current in forward pass.
- Uses regen-like braking in backward pass.
- Contains combined-slip logic that is not effectively activated by current production calls.

`src/fsae_sim/vehicle/dynamics.py`

- Contains drag, rolling resistance, cornering drag, traction, braking, and exit speed logic.
- Uses simplified tire limits in places where combined-slip behavior is needed.
- Mechanical brake force is a simplified tire-capacity scale.
- Effective mass treatment needs review.

`src/fsae_sim/vehicle/tire_model.py`

- Tire model source and coefficient consistency are major concerns.
- Grip scaling from telemetry makes it calibration-dependent.
- Combined-slip implementation is not fully wired into runtime dynamics.

`src/fsae_sim/vehicle/cornering_solver.py`

- Provides a quasi-static cornering capacity check.
- Does not solve full vehicle yaw equilibrium.
- Combined-slip support exists but is mostly bypassed by the current call path.

`src/fsae_sim/vehicle/load_transfer.py`

- Useful structure but under-validated parameters.
- Missing important transient and geometry effects.
- Existing tests indicate changed or inconsistent expected values.

`src/fsae_sim/vehicle/powertrain_model.py`

- Mixes physical equations and firmware-derived constants.
- Uses hardcoded tire radius.
- Current limits are not consistently represented in all force-capability paths.

`src/fsae_sim/vehicle/battery_model.py`

- Needs independent validation for charge, voltage, current, and temperature.
- Same-run calibration makes current validation weak.

`src/fsae_sim/track/track.py`

- Builds track from averaged GPS telemetry.
- Curvature and grade are therefore data-processing artifacts unless independently validated.

`src/fsae_sim/data/loader.py`

- Cleaned data aliases and derived distance fields should be treated carefully.
- Names imply GPS quantities that may actually come from wheel speed.

`src/fsae_sim/driver/strategies.py`

- Replay and calibrated strategies are useful but should not be confused with predictive driver modeling.
- Observed speed caps are a major validation leak.

`src/fsae_sim/analysis/telemetry_analysis.py`

- Collapses telemetry into zones and derives grip scale.
- This is calibration logic, not physics truth.

`src/fsae_sim/analysis/validation.py`

- Needs stronger independence checks and stricter physical metrics.
- Current metrics can pass while important state variables are wrong.

`backend/services/sim_runner.py`

- Wires together same-telemetry calibration and validation.
- Hardcoded lap count and initial state choices need review.

`backend/services/visualization_export.py`

- Recomputes physics for display.
- Has mass and force-sign issues.

`configs/ct16ev.yaml`

- Contains empirically tuned values.
- CdA/ClA naming conventions need clarification.
- Current dirty state differs from some tests.

`configs/ct17ev.yaml`

- Incomplete for predictive use.
- Missing tire/suspension/aero detail makes CT17EV comparisons unsafe.

## Final Recommendation

Do not spend the next effort polishing plots or adding more knobs. The correct next step is to make the simulator falsifiable.

The minimum credible path is:

1. Remove leakage between calibration and validation.
2. Fix the non-conservative integrator.
3. Make speed envelope and runtime physics use the same force model.
4. Validate track, tire, powertrain, and battery independently.
5. Only then run CT17EV predictions.

Until then, all endurance conclusions should be labeled as unvalidated estimates.
