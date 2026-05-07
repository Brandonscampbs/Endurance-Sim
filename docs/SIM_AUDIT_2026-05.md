# Simulator Audit and Improvement Plan — 2026-05-06

A current-state physics audit of the FSAE EV endurance simulator, graded against
industry-standard quasi-steady-state (QSS) lap simulators, with a prioritized
improvement checklist.

This is the canonical physics-audit document. The earlier
`docs/PHYSICS_AUDIT.md` (a critique of a much earlier version of the
codebase) was retired in commit `ef0da94`; the historical text remains
accessible at git blob `ef0da94^:docs/PHYSICS_AUDIT.md` for anyone who
wants to read the original critique.

---

## TL;DR

**Grade for ranking torque / RPM / current-limit tunes against each other on
the Michigan track: B+ → A− achievable.**

**Grade for predicting absolute endurance time or score: C−.**

Use this sim to sort tune options. Do not use it to forecast wall-clock
endurance time or absolute score points. The physics is faithful when given
correct driver inputs (replay validates within 0.1 % on time, 0.8 % on net
Ah). The 7.3 % calibrated-mode time error is essentially all driver-model
deficiency, and that error largely cancels in deltas across sweeps as long as
the swept change does not unlock new driver behavior.

---

## Industry context

Lap simulation lives on a spectrum:

| Tier            | Tool                              | Approach                | FSAE accuracy on a tuned car  |
|-----------------|-----------------------------------|-------------------------|-------------------------------|
| Lightweight     | OptimumLap                        | Point-mass QSS          | ~10 % lap-time                |
| Academic        | TUMFTM `laptime-simulation`       | QSS, electrified PT     | "differentiable", no abs. tgt |
| Professional    | IPG CarMaker                      | Multi-body + driver     | 1–2 % on validated track      |
| Top-tier        | AVL VSM, Adams/Car, VI-Grade      | Full transient + DiL    | <1 % with rig-validated tires |

This sim is QSS with a forward-backward speed envelope, GGV cornering envelope,
full Pacejka PAC2002 + combined slip, EMRAX 228 efficiency map, LVCU-firmware-
faithful torque chain, calibrated inverter delivery map, and Voltt-calibrated
equivalent-circuit battery. It is **at or above the TUMFTM open-source
baseline** and clearly above OptimumLap.

QSS is the right architecture for FSAE EV tune sweeps. The Duke FSAE
post on QSS limitations and the QUB Belfast EV LTS thesis both confirm: QSS is
explicitly a parameter-sweep tool, not an absolute-prediction tool. Lap time is
a continuous, differentiable function of vehicle parameters in this regime, so
relative deltas are believable even when absolutes are off.

---

## Subsystem scorecard

| Subsystem                                  | Grade | One-line verdict |
|--------------------------------------------|:-----:|------------------|
| Powertrain (`vehicle/powertrain_model.py`) | A−    | LVCU-firmware-faithful, EMRAX map, inverter delivery map. Strongest subsystem. |
| Speed envelope (`sim/speed_envelope.py`)   | A−    | Forward-backward with BMS limit, lap-wrap fixed point, combined-slip pass. |
| Tire (`vehicle/tire_model.py`)             | B+    | Real PAC2002 lateral + combined slip; longitudinal coefficients transplanted from R25B. |
| Vehicle dynamics (`vehicle/dynamics.py`)   | B+    | Pacejka cornering drag, self-consistent traction/braking fixed point. |
| Engine (`sim/engine.py`)                   | B+    | Force-balance speed enforcement + predictor-corrector. No clamping. |
| Battery (`vehicle/battery_model.py`)       | B     | Voltt-cal cell + AiM-refined pack, voltage-floor enforcement, lumped thermal. |
| Track (`track/track.py`)                   | B−    | Multi-lap weighted GPS centerline + Hampel filter. Still telemetry-derived. |
| Driver model (`driver/strategies.py`)      | C+    | Calibrated/replay only. **No adaptive driver.** Biggest single weakness. |

---

## What is no longer broken (vs. the retired earlier audit)

The earlier audit's "highest-severity findings" are largely resolved. Verified
against current source:

- "Validation is circular" — `assert_calibration_validation_split` raises on
  overlap; `BatteryModel.calibrate_pack_from_telemetry` requires `holdout_laps`
  or an explicit opt-in; `CalibratedStrategy.from_telemetry` accepts
  `holdout_laps`; `SimulationMode.PREDICTION` forbids telemetry-derived speed
  caps and tracks (`engine.py:120-141`).
- "Segment integration is non-conservative" — `enforce_speed_limit`
  (`engine.py:417-478`) does proper force balance to land at the cap, not
  speed deletion.
- "Envelope and engine use different physics" — both now call
  `dynamics.mechanical_brake_force(1.0, v)` for active braking and
  `lvcu_torque_ceiling` + `apply_inverter_delivery` for drive
  (`speed_envelope.py:294-329`).
- "Combined slip is dead code" — wired into `CorneringSolver._can_sustain` and
  `speed_envelope.py:161-247` Pass 4.
- "Powertrain mixes physical and firmware logic ambiguously" — `lvcu_torque_command`
  (firmware), `max_motor_torque` (physical PMSM envelope), and
  `apply_inverter_delivery` (calibrated map) are now distinct.
- "Battery scoring against AiM SOC" — net Ah and net kWh are the scored
  metrics; AiM SOC is diagnostic only (`SIM_ACCURACY.md`).
- "Track from naive median-lat crossing + single-lap" — replaced with multi-lap
  weighted-mean centerline, periodic Gaussian smoothing, Hampel outlier
  filter, dynamic-curvature reference for outlier replacement, and 2-D
  start/finish gate.

Do not waste cycles re-investigating those findings — they are closed.

---

## Improvement checklist (priority order)

Each item: what to change, where, and why it moves the grade.

### P0 — biggest grade lifts

- [ ] **Adaptive driver model.** Implement a controller that targets a
  longitudinal-g profile rather than replaying a fixed pedal trace.
  - File: new `src/fsae_sim/driver/adaptive.py`.
  - Inputs: speed envelope (already pre-computed), upcoming segments, current
    state.
  - Outputs: throttle / brake commands sized so the next-segment exit speed
    matches the envelope.
  - Why: the calibrated driver does not change brake or throttle points when
    you raise `motor_speed_max_rpm` or `torque_limit_inverter_nm`. Sweep
    benefits that require late-braking or earlier throttle pickup are
    invisible today. This is the single change that pushes the sim from
    B+ to A− for tune-sweep work.
  - Acceptance: replay-mode lap time error < 1 % on Michigan endurance with
    the new driver substituted (i.e. it can drive as well as the real driver
    when given the same envelope).

- [ ] **Sweep harness with delta-vs-baseline and noise-floor reporting.**
  - File: new `scripts/sim_sweep.py`.
  - Behavior: takes a baseline config + a parameter grid, runs each combination,
    reports outputs as `delta_vs_baseline ± noise_floor`. Noise floor =
    `|calibrated_baseline - replay_baseline|` per metric.
  - Outputs: a CSV ranking sweep candidates, with a flag column "below noise
    floor" so you do not chase signal that is not there.
  - Why: prevents chasing 0.2 s lap-time deltas that are smaller than the
    model's own self-consistency error.

### P1 — accuracy fixes that move sweep deltas

- [ ] **Refresh BMS current limit lap-by-lap in the speed envelope.**
  - File: `src/fsae_sim/sim/engine.py:323-327`. Currently the envelope uses
    `initial_bms_limit` once before the lap loop.
  - Fix: re-`compute()` the envelope at the start of each lap (or when temp
    crosses a threshold) using the current `(soc, temp)` BMS limit.
  - Cost: ~few ms per lap on a 21-lap endurance sim. Acceptable.
  - Why: late-stint thermal derating is currently underestimated. Endurance
    energy and time both bias optimistic in the last 4-5 laps — the laps
    that matter most for the efficiency score.

- [x] **`m_effective` should not contain η (issue M13).** RESOLVED 2026-05-07.
  - File: `src/fsae_sim/vehicle/dynamics.py:96-98`.
  - Diagnosis: η is a power-flow / force property; it cannot appear in
    kinetic energy or equivalent inertia. KE is a state function of speed
    alone (Genta §5.2; Krause/Wasynczuk/Sudhoff §3.5). The earlier audit
    note proposing `× G²/η` for regen was a misdiagnosis — direction-
    dependent inertia violates conservation.
  - Fix: drop η from the rotor-inertia term entirely.
    `m_eff = m + (J_motor·G² + 4·J_wheel) / r²` — symmetric, no η.
  - The directional asymmetry between accel and regen lives correctly in
    `drive_force` (× η) and `regen_force` (× 1/η, S12). It does not need
    to be re-applied in the inertia term.
  - Cross-check: TUMFTM `laptime-simulation` `__compute_m_eq` matches.

- [ ] **Coast electrical-power gap (issue 22, ~45 Wh / stint).**
  - File: `src/fsae_sim/vehicle/powertrain_model.py:545-617` (the `coast`
    branch in `electrical_power`).
  - Investigate: when motor torque is below `_COAST_TORQUE_THRESHOLD_NM` and
    `V_bemf < V_pack`, the model returns 0 W. Telemetry shows non-zero
    coast power. Likely sources: inverter switching losses, motor cogging
    drag converted to heat in the inverter, low-side body-diode leakage.
  - Why: 45 Wh/stint is small (~0.6 % of 7 kWh) but it is a known systematic
    bias and may matter for tight efficiency-score sweeps.

### P2 — completeness fixes

- [ ] **Use `tire_model.loaded_radius()` for `motor_rpm_from_speed` and
  `wheel_force` (issue 18 / PARTIAL).**
  - Files: `src/fsae_sim/vehicle/powertrain_model.py:67, 162, 459`.
  - Currently uses constant `TIRE_RADIUS_M = 0.2042` (unloaded). Loaded
    radius is ~3 % smaller under load — biases motor RPM and wheel force
    by the same.
  - Fix: pass an `Fz` reference (from load transfer, per-wheel) and call
    `tire_model.loaded_radius(fz)`. For a single rolling radius, take the
    mean of the four-wheel loads.

- [ ] **Pack thermal — air-flow / speed dependence on heat-out (issue 4).**
  - File: `src/fsae_sim/vehicle/battery_model.py:920-933`. `heat_out_w` uses
    a constant `thermal_conductance_w_per_k`.
  - Fix: scale convective conductance with vehicle speed (or pack-fan
    duty if known). A simple `h_conv = h_static + k_v · v` is enough to
    capture the first-order effect.

- [ ] **OCV temperature dependence (issue 8).**
  - File: same.
  - Currently SOC-only. Cell discharge curves shift modestly with temperature.
  - Fix: optional 2-D OCV(SOC, T) interpolator if the Voltt cell sim has
    multiple temperatures; otherwise document as a known omission.

- [ ] **Air density vs ISA (issue 19).**
  - File: `src/fsae_sim/physics_constants.py`.
  - Fix: make `AIR_DENSITY_KG_M3` configurable per event from temp/pressure.
    Michigan endurance day should be pulled from local METAR or a config knob.

### P3 — bigger projects (defer until P0–P1 done)

- [ ] **LC0 longitudinal tire data.** Run TTC Round 8 longitudinal slip data
  for the actual tire if available, replace transplanted R25B PDX/PKX/PCX.
  Highest impact on torque sweeps that push rear-tire utilization.
- [ ] **Independent track model.** Build the centerline from cone map / RTK
  GPS / surveyed data instead of averaged driven laps. Required for
  predicting events other than Michigan 2025.
- [ ] **Per-cell or per-module pack thermal.** Current lumped model with
  Newton cooling is sufficient for sweeps on similar duty cycles. Per-module
  becomes useful when comparing cooling-package changes.
- [ ] **Heun / RK2 integrator across the segment.** The current
  predictor-corrector at average operating point is good at 0.5 m
  segments; this would mainly let you increase segment length without
  losing accuracy.

---

## How to actually run sweeps responsibly

1. **Calibrate the noise floor.** Run baseline once with `CalibratedStrategy`
   and once with `ReplayStrategy`. The metric-by-metric difference is your
   noise floor.
   ```bash
   python scripts/sim_compare.py --strategy calibrated --no-plots
   python scripts/sim_compare.py --strategy replay --no-plots
   ```

2. **Drop telemetry-derived speed caps for sweeps that raise capability.**
   Otherwise the calibrated zone caps clip the envelope and you see no
   benefit from a higher torque or RPM ceiling.
   ```python
   strategy = CalibratedStrategy.from_telemetry(...).without_observed_speed_caps()
   ```
   Or run in `SimulationMode.PREDICTION`, which enforces this.

3. **Sweep with the same strategy across the whole grid.** Driver-model
   error largely cancels in deltas only if the driver is identical run-to-run.

4. **Report deltas, not absolutes.** A swept candidate that beats baseline by
   more than the noise floor is signal. Anything within noise is not.

5. **Use net Ah and net kWh as primary metrics.** They validate within 3-4 %
   on calibrated mode (vs 7 % for time) and they map directly to the
   efficiency-score side of the FSAE EV endurance score.

6. **Cross-check interesting candidates in replay-equivalent mode.**
   Compute the equivalent torque trace the swept tune would have produced
   under the calibrated pedal profile, then run replay. If both agree,
   trust the candidate. If they diverge, the driver model is hiding
   something (most likely a brake-point shift the calibrated driver did
   not make).

---

## Sweep-by-sweep confidence summary

For the user's stated parameters:

| Sweep parameter                        | Direction | Confidence  | Caveat |
|----------------------------------------|-----------|-------------|--------|
| `torque_limit_inverter_nm`             | down      | High        | LVCU ceiling shrinks proportionally; calibrated throttle scales with it. |
| `torque_limit_inverter_nm`             | up        | Medium-high | Driver does not adapt brake points; gain may be understated. |
| `motor_speed_max_rpm`                  | up        | Medium      | Top-speed gain is real; corner-exit gain depends on driver brake-point adaptation. |
| `brake_speed_rpm` (FW corner)          | either    | Medium      | Affects field-weakening shape directly; driver doesn't adapt. |
| BMS `discharge_limits` table           | either    | Medium-high | Threads through `lvcu_torque_ceiling`. Only the *initial* limit feeds the envelope today (P1 fix). |
| `lvcu_power_constant`                  | either    | High        | Direct knob on the LVCU power-limit shape. |
| Mass / aero / tire choice              | any       | Lower       | Each requires a recalibrated `grip_scale` and may invalidate driver profile. Defer until adaptive driver. |

---

## References

- Duke FSAE — "All models are wrong, but…" — https://www.dukefsae.com/single-post/all-models-are-wrong-but
- TUMFTM `laptime-simulation` — https://github.com/TUMFTM/laptime-simulation
- "A Quasi-Steady-State Lap Time Simulation for Electrified Race Cars" (IEEE EVER 2019) — https://ieeexplore.ieee.org/document/8813646/
- "Lap Time Simulation Tool for an Electric Formula Student Car" (QUB Belfast)
  — https://pure.qub.ac.uk/files/164458711/Lap_Time_Simulation_Tool_for_the_Development_of_an_Electric_Formula_Student_Car.pdf
- "Performance Optimization of a Formula Student Racing Car Using IPG CarMaker, Part 1" (MDPI 2024)
  — https://www.mdpi.com/2673-4591/79/1/86
- "Optimizing Torque Delivery for an Energy-Limited Electric Race Car Using MPC" (MDPI 2022)
  — https://www.mdpi.com/2032-6653/13/12/224
- "Lap Time Simulation of FSAE Vehicle With Quasi-Steady-State Model" (SAE 2016-36-0164)
  — https://www.sae.org/publications/technical-papers/content/2016-36-0164/
- Wisconsin Racing WR-217e Powertrain Architecture & LapSim
  — https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf
- Chalmers Formula Student EV Powertrain Design
  — https://publications.lib.chalmers.se/records/fulltext/191837/191837.pdf
- OptimumLap product page — https://optimumg.com/product/optimumlap/
- FSAE EV Handbook — Endurance & Energy Efficiency
  — https://www.fsaeonline.com/Page.aspx?pageid=fa25ca79-3a7a-4b3f-803f-7bdddb3b2c57
- FSAE Electric 2025 Overall Results (Michigan)
  — https://www.fsaeonline.com/CompResources/2025/8f030a58-d9e4-49b8-bc83-6ca16c7ce715/FSAE_2025_MI6_results.pdf
