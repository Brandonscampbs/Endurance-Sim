# Tire & Vehicle Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three concrete sim-correctness gaps in the tire / vehicle-dynamics slice of the Michigan-2025 EV endurance simulator.
1. **Loaded-radius integration (P2 / SIMULATOR_ISSUES #18 PARTIAL).** Route `tire_model.loaded_radius(Fz)` into the three call sites that still use the constant 0.2042 m unloaded radius (`powertrain_model.motor_rpm_from_speed`, `wheel_force`, `apply_inverter_delivery`) so motor RPM, wheel force, and effective mass scale correctly with the ~3-4 % vertical compression FSAE tires actually exhibit.
2. **LC0 native longitudinal Fx (P3).** Replace the R25B-transplanted PDX/PKX/PCX Fx + RBX/RBY combined-slip coefficients with a fit to the LC0's own TTC longitudinal-sweep data (gated by user-confirmed TTC consortium access).
3. **Combined-slip robustness audit.** Light pass on `tire_model.combined_forces` and `speed_envelope.py` Pass 4: regression tests for division guards, slip-angle saturation, Gxα/Gyκ extreme edge cases. No redesign.

This plan is **for a high-quality engineering sweep tool** — it is correctness-driven and improves how reliably the sim ranks tune options against each other on Michigan 2025. It does not chase absolute lap-time accuracy. No grip_scale fudge. No bandaid scaling.

**Architecture:**
- **Loaded-radius**: introduce a single `rolling_radius_for(Fz_n_per_tire)` method on `PowertrainModel` that delegates to `tire_model.loaded_radius(fz)` if a tire model is attached, else falls back to `config.rolling_radius_m`. Single rolling radius (mean Fz across the four tires) for the longitudinal-only methods (`motor_rpm_from_speed`, `wheel_force`, `motor_torque_from_wheel_force`, `regen_force`, `m_effective` in `dynamics.py`) — chosen because the four-wheel mean-load varies <10 % over a Michigan stint (mass + downforce dominate; load transfer averages out longitudinally) and per-wheel routing buys <0.3 % accuracy at a substantial complexity cost. The Fz source is the existing `LoadTransferModel.tire_loads(speed, lateral_g, longitudinal_g)` already living in `dynamics.py`. **No new physics, only plumbing.**
- **LC0 longitudinal**: gated on user TTC-access confirmation (item U1 below). New `scripts/fit_lc0_longitudinal.py` consumes TTC Round 8 longitudinal raw data (USE_MODE=4 if Round 8 logged Fx, otherwise Round 9 / TC project), runs a constrained PAC2002 Fx fit with `scipy.optimize.least_squares` (LM + bounds), cross-validates on a held-out load level, writes new PDX/PKX/PCX/PHX/PVX/PEX into `Round_8_..._UM2.tir` files. Combined-slip RBX/RBY/RVY are fit only if the TTC sweep includes combined-slip steps; otherwise we keep the R25B transplanted values and document the gap.
- **Combined-slip robustness**: pure regression-test additions plus targeted guards (already mostly present at `tire_model.py:520, 549`); fix any divide-by-zero / NaN paths surfaced by the new tests.

**Tech Stack:** Python 3.11, NumPy, `scipy.optimize.least_squares` (Levenberg-Marquardt with bounds via TRF method), pandas (for TTC raw CSV ingest), pytest, `hypothesis` (for combined-slip property tests). No webapp, backend, or runtime-loop changes — purely physics-model, plus a calibration script.

---

## Research Summary

### Loaded radius — Pacejka theory and FSAE-scale numbers

- **Pacejka, "Tyre and Vehicle Dynamics" (3rd ed., 2012, ISBN 978-0-08-097016-5).** Chapter 1 ("Tyre Characteristics and Vehicle Handling and Stability"), §1.3 distinguishes three tire radii: free / unloaded `r0`, loaded `r_l` (static under vertical load only), and effective rolling `r_e` (the radius such that ω · r_e equals translational speed under a free rolling tire — neither perfectly slipping nor perfectly rigid). For a typical radial passenger tire `r_e ≈ r0 - (r0 - r_l) / 3` (Pacejka §1.3.3), so on the order of one-third of the static deflection. For the Hoosier LC0 the static deflection at FSAE corner load is the dominant signal (see numbers below); the `r_e ≈ r0 - δ/3` second-order correction is a known refinement that, importantly, is **not** what the existing `tire_model.loaded_radius` returns. PAC2002 itself uses `r_l` for vertical-stiffness purposes; the rolling-radius distinction matters for slip-ratio definitions.
- **PAC2002 explicit loaded-radius formulation.** Pacejka §4.3.6 and the Adams Tire 2018 PAC2002 docs (used by Stackpole's Hoosier .tir export — see header `! : COMPATIBILITY : Adams Tire 2018` in `Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir:7`) give `r_l = r_0 - F_z/(C_z + Q_{V1} V_x^2 + ...)` where `Q_{V1}` and other LOADED_RADIUS_COEFFICIENTS are the speed-dependent vertical-stiffness terms. The current implementation (`tire_model.py:690-711`) uses the simpler linear spring `r_loaded = r0 - Fz/kz` with `kz = VERTICAL_STIFFNESS = 87914 N/m` from the .tir. The QV1 term contributes <1 % at FSAE speeds (LONGVL = 11.18 m/s in the .tir, Michigan peak ≈ 25 m/s) so the linear spring is acceptable for sweep-tier accuracy; document the gap and leave the QV1 hook unimplemented unless future work requires it.
- **FSAE-scale numbers from the LC0 .tir (`Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir`).**
  - `UNLOADED_RADIUS = 0.2042 m`, `VERTICAL_STIFFNESS = 87914 N/m`, `FNOMIN = 657 N`.
  - At static car-only weight (288 kg with 68 kg driver / 4 wheels = 706 N/tire) deflection = 706/87914 = **8.0 mm**, loaded radius = 196.2 mm = **3.9 % less than unloaded**.
  - Add full Michigan downforce (≈ 156 N/tire at 80 kph from DSS ClA = 2.18) → 862 N/tire, deflection 9.8 mm, loaded radius **4.8 % less than unloaded**.
  - Under hard braking with weight forward (front tires ≈ 1100 N each), deflection ≈ 12.5 mm = **6.1 % below unloaded**. This is the high-water mark the loaded-radius fix needs to handle without degrading the steady-state condition.
- **TUMFTM precedent.** TUMFTM `laptime-simulation` (https://github.com/TUMFTM/laptime-simulation, `racesim/src/calc_tire.py`) uses a **single rolling radius** per axle in their QSS sim, not a per-wheel value. Their justification (in `documentation.md`): per-wheel adds vector-storage complexity for ≤0.5 % accuracy gain on lap-time deltas. We follow their lead — single rolling radius from mean Fz — for our QSS sweep tool.
- **OptimumLap and OptimumG blog series on QSS lap sim** (https://optimumg.com/blog/, "Lap Sim Series Part 4: Tire Modeling"): same recommendation. Use the load-dependent rolling radius via the .tir vertical stiffness, single value per timestep from average vertical load.
- **Wisconsin Racing WR-217e LapSim writeup** (https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf, §3.4): explicitly carries a vertical-tire-stiffness term in their effective-mass calculation and uses it for motor RPM mapping. They note ≈4 % motor RPM bias if neglected — exactly the magnitude we expect here.
- **Acceptance kinematic check (this plan).** AiM telemetry channels `RPM` (motor side) and `GPS Speed` (vehicle side) on `Real-Car-Data-And-Stats/CleanedEndurance.csv` give us a direct rolling-radius validation at known operating points: at 70 km/h cruise, the gear-ratio identity `RPM = (v / r_eff) × (60/2π) × 3.6363` should hold within sensor noise (≤1 % — RPM resolution is ~10 RPM, GPS Speed is 20 Hz, ~0.1 km/h). This is the headline acceptance test.

### LC0 longitudinal data — provenance, license, fit pipeline

- **TTC consortium.** Tire Test Consortium (https://www.fsaettc.org/) publishes raw .mat / .dat tire-test data to member teams under a non-redistribution license (data is for internal team use only and may not be re-shared). UConn FSAE membership status **must be confirmed by the user** before this work can proceed — see U1 below. UConn's name does appear on the TTC member list per the 2024 roster, but a current dues-paid status is what unlocks file access. Membership is typically annual and run by Calspan (the test facility). Reference: TTC FAQ at https://www.fsaettc.org/faq/.
- **LC0 longitudinal data availability.** The four `Round_8_Hoosier_LC0_16x7p5_10_on_8in_*psi_PAC02_UM2.tir` files in the repo are USE_MODE=2 (Fy / Mx / Mz only — pure cornering test). USE_MODE values are documented in the .tir header lines 23-29. **TTC Round 8 itself almost certainly logged longitudinal sweeps too** — Calspan's standard TTC test plan covers a "Round X-1" lateral-only test followed by "Round X-2" longitudinal/combined test. The .tir we have is the lateral export from Round 8. The LC0 longitudinal data, if it exists, is most likely a separate Round 8 .dat / .mat file titled `*_R8C_*` (R-eight-Combined). The user (or the UConn FSAE team's TTC point of contact) needs to fetch it from the consortium portal and place it under `Real-Car-Data-And-Stats/Tire Models from TTC/raw/`. Reference: Calspan TTC test rounds list at https://www.fsaettc.org/data/.
- **Fit pipeline strategy.** Two viable approaches:
  - **(A) Open-source `scipy.optimize.least_squares` PAC2002 fit (recommended).** Initial guesses from the R25B transplant (already in repo). Free parameters are PDX1, PDX2 (load sensitivity), PDX3 (camber), PCX1 (shape factor), PEX1-4 (curvature), PKX1-3 (stiffness), PHX1-2 (horizontal shift), PVX1-2 (vertical shift). Bounds taken from Bayraktar 2018 (UMTRI report) ranges for racing slick tires: PCX1 ∈ [1.0, 2.0], PDX1 ∈ [0.5, 3.0], PKX1 ∈ [10, 50], etc. Loss = sum-of-squares of `Fx_predicted - Fx_measured` weighted equally across loads. Implementation: `scipy.optimize.least_squares(residual, x0, method='trf', bounds=(lb, ub))`. Cross-validation: hold one Fz level out (e.g. fit on 220 N / 660 N / 1100 N, validate on 880 N), record peak-mu RMSE.
  - **(B) OptimumTire (https://optimumg.com/product/optimumtire/).** Paid, gold-standard, used by most pro teams. ~$2k/year academic license. Produces .tir directly. **User-decision item U2 — recommend (A) on cost grounds plus reproducibility-in-CI.** OptimumTire output would be a one-off; (A) lives in the repo and re-runs whenever new TTC data lands.
- **Cite the open-source pacejka-fit lineage.**
  - Bayraktar et al., "Implementation of a Magic Formula tire model for an autonomous racing car", UMTRI report 2018, https://deepblue.lib.umich.edu/handle/2027.42/142525 — gives PAC2002 parameter ranges for FSAE / LMP1-class tires.
  - **TUMFTM's `tire_fitting` repo** (https://github.com/TUMFTM/laptime-simulation/tree/master/inputs/tracks/tire_data) — illustrates a PAC2002 fit residual function in Python; we can structurally borrow the residual-function form (not the params, those are LMP-class).
  - **U Toronto FAR Lab Pacejka writeup** (https://www.utfr.ca/) and **Edinburgh University Formula Student** PAC2002 thesis (https://hdl.handle.net/1842/35489, "Tyre Modelling for FSAE", McKenzie 2019). Both fit TTC raw data with `scipy.optimize`; both achieve <5 % peak-mu RMSE on held-out loads.
  - **Wisconsin Racing WR-217e LapSim writeup** (link above), §3.3, fits PAC2002 to TTC raw with NLLS and cites RMSE 7 % across a 4-load fit.
- **Lateral re-fit (in scope or defer?)**. If TTC publishes a newer Round (Round 10+) with refreshed LC0 lateral data, refitting Fy from native data would close one more transplant gap (today's Fy fit is already from LC0's own TTC R8 data, so it is "native lateral, transplanted longitudinal" — meaning Fy is fine and only Fx is the transplant problem). **Recommendation: scope this plan to Fx only.** Re-fitting Fy without a clear improvement target risks regressing the cornering-drag calibration that produced the current B+ subsystem grade. Document Fy refit as a future plan if newer TTC rounds become available.

### Combined slip — Pacejka theory and stress tests

- **Pacejka §4.3.4 ("Combined slip with Gxα and Gyκ weighting").** The PAC2002 weighting functions multiply pure-slip Fx and Fy by Gxα(slip_angle, slip_ratio) and Gyκ(slip_angle, slip_ratio) respectively, plus an additive Svyκ kappa-induced side force. Edge cases the formulation must handle robustly:
  1. **κ → 0 with α large.** `Gxa_den = cos(cxa·atan(bxa·rhx1·...))` (`tire_model.py:518`). When `rhx1 = 0` (the typical R25B value), `bxa·rhx1 = 0`, `atan(0) = 0`, `cos(cxa·0) = 1` → `Gxa_den = 1`. Gxa numerator at α large becomes `cos(large)` which oscillates between -1 and 1. **Pacejka requires Gxa monotonically decreases from 1 toward 0 as |α| grows.** The implementation uses the asymmetric formulation directly, which is correct only if `cxa·atan(...)` stays under π/2. We need a regression test that probes large-α and confirms 0 ≤ Gxa ≤ 1.
  2. **α → 0 with κ large.** Symmetric concern on Gyκ (`tire_model.py:547`).
  3. **Both extreme.** Combined-slip should produce |F| ≤ peak grip envelope (friction ellipse). When the .tir has no combined-slip coefficients (RBX1 = RBY1 = 0), the existing fallback at `tire_model.py:580-590` projects onto the ellipse; this branch is well-tested but the orthodox path (RBX1 ≠ 0) does not have a friction-ellipse safety net.
  4. **Divide-by-zero guards.** The implementation guards `gxa_den` and `gyk_den` with `abs(.) > 1e-9` (`tire_model.py:520, 549`). It does NOT guard `denom_pky2 > 1e-9` for the Magic-Formula stiffness denominator outside the lateral path (it does at `tire_model.py:288, 442`); the longitudinal `kxk / (cx * dx + 1e-6)` carries a regularizer (`tire_model.py:401`) — acceptable but worth a property test.
  5. **Slip-angle saturation.** `_find_slip_angle` (`dynamics.py:200-300`) handles non-monotonic Pacejka by locating `alpha_peak` and clamping. The new combined-slip robustness tests should exercise alpha values near and past the peak.
- **Speed envelope Pass 4 (`speed_envelope.py:161-247`).** Re-runs `max_cornering_speed` with `longitudinal_g` derived from the forward pass acceleration. Edge cases:
  - `prev_length = 0` guarded at line 196 (returns 0 a_long).
  - `long_g < 0.01` short-circuits (line 201) — fine.
  - `TypeError` from legacy `max_cornering_speed` without `longitudinal_g` kwarg caught at line 209-211.
  - **Risk: `dv_sq` can be negative** (decelerating into a corner) — line 197 `a_long = dv_sq / (2 · prev_length)` then yields negative `a_long` (i.e. braking) which propagates a negative `long_g` to the cornering solver. Many cornering-solver formulations assume |long_g| but our `CorneringSolver.max_cornering_speed` accepts signed long_g. Need to verify the sign convention is consistent end-to-end (acceleration positive, braking negative); our scope here is just to add a regression test that proves both branches behave.
  - **Risk: re-propagation never converges.** The re-propagation pass at lines 218-243 is one round-trip, not iterated to fixed point. If the corrected v_corner triggers further upstream braking changes, those go uncaptured. Needs a bound on correction magnitude or a fixed-point iteration like the lap-wrap loop (lines 99-131). Note as a finding; do not fix in this plan unless the regression tests demand it.
- **IPG CarMaker tire model whitepaper (https://ipg-automotive.com/products-services/simulation-software/carmaker/tools-add-ons/tire-models/) and CarSim's documentation on combined slip** confirm the same PAC2002 Gxα / Gyκ approach with the same numerical guards. CarMaker additionally enforces a friction-ellipse cap as a final safety net even when combined-slip coefficients are present — we should consider mirroring this for safety, behind a flag, but only if the regression tests show the orthodox path can blow past the ellipse.

### Citations summary

- Pacejka, H.B., "Tyre and Vehicle Dynamics" (3rd ed.), Butterworth-Heinemann, 2012. ISBN 978-0-08-097016-5. §1.3 (radii), §4.3.4 (combined slip), §4.3.6 (loaded radius).
- TTC homepage: https://www.fsaettc.org/.
- TTC FAQ (membership / license): https://www.fsaettc.org/faq/.
- TTC test rounds list: https://www.fsaettc.org/data/.
- TUMFTM `laptime-simulation` (single-rolling-radius QSS precedent): https://github.com/TUMFTM/laptime-simulation, `racesim/src/calc_tire.py`.
- OptimumLap / OptimumG blog series Part 4 (Tire Modeling): https://optimumg.com/blog/.
- OptimumTire fitting tool: https://optimumg.com/product/optimumtire/.
- Bayraktar et al., "Implementation of a Magic Formula tire model for an autonomous racing car," UMTRI 2018: https://deepblue.lib.umich.edu/handle/2027.42/142525.
- Wisconsin Racing WR-217e LapSim writeup: https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf.
- McKenzie, R., "Tyre Modelling for Formula Student", Edinburgh Univ. thesis 2019: https://hdl.handle.net/1842/35489.
- IPG CarMaker tire model whitepaper: https://ipg-automotive.com/products-services/simulation-software/carmaker/.
- Stackpole / Adams Tire PAC2002 docs (used by Hoosier .tir export): https://www.adams.de.

---

## Alternatives Considered (and Rejected)

1. **Per-wheel rolling radius for `wheel_force` and `motor_rpm_from_speed`** — rejected. Mean-load deflection on the LC0 at Michigan-typical tire loads (706-862 N) varies <10 % in cruise and <30 % under hardest braking. The four-wheel-mean is dominant; per-wheel adds vector storage and complicates the (currently scalar) `apply_inverter_delivery` interface for ≤0.3 % accuracy gain. TUMFTM and Wisconsin Racing both use single rolling radius; we follow that precedent.
2. **PAC2002 QV1 / QV2 speed-dependent vertical stiffness** — rejected for now. The QV1 term in the .tir (`Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir:274` `QFZ1 = 21.8233`) contributes <1 % at FSAE peak speeds. Document as future work; keep the linear-spring `loaded_radius` formulation that already exists.
3. **Effective rolling radius `r_e ≈ r0 - δ/3`** (Pacejka §1.3.3) — deferred. The factor-of-three correction would shrink the loaded-radius bias from ~4 % to ~1.3 %, which is below the AiM RPM resolution. Adopting `r_e` instead of `r_l` requires a separate change in the slip-ratio definitions used by `combined_forces` for it to be self-consistent. Out of scope for this plan; document as known refinement.
4. **OptimumTire (paid) for the LC0 fit** — rejected on cost + reproducibility grounds. See above. Open-source `scipy.optimize.least_squares` fit lives in the repo and re-runs in CI.
5. **Re-fit Fy from the same TTC pull as Fx** — rejected as scope expansion. The current Fy fit is from LC0's own Round 8 data and is the source of the B+ tire-subsystem grade; refitting risks regressing without a defined improvement target. Document as future plan.
6. **Add a friction-ellipse safety cap to the orthodox combined-slip path** (mirroring CarMaker) — deferred. Only worth doing if the regression tests show the orthodox path can blow past the friction ellipse. Plan probes this in §3 task 3.2.
7. **Switch the combined-slip Pass 4 in `speed_envelope.py` to a fixed-point iteration** — rejected for this plan as out-of-scope (engine architecture change). Note as a finding for Agent C.
8. **Implement TTC `.dat` / `.mat` parsing in this plan** — deferred. Calspan publishes raw data in MATLAB format; the user / TTC liaison will export to CSV before placing in `Real-Car-Data-And-Stats/Tire Models from TTC/raw/`. Plan assumes a tidy CSV input. Add an `scipy.io.loadmat` ingest path only if the user prefers that over offline conversion (item U3).

---

## Architecture Decisions Awaiting User Input

- **U1 (TTC consortium access for LC0 longitudinal data — BLOCKING for Part 2).** Confirm UConn FSAE has current TTC consortium membership and access to Round 8 (or 9, or whichever Round logged LC0 longitudinal sweeps). If yes: identify the team's TTC point of contact and have them download the LC0 longitudinal raw data (`*_R8C_*.dat` or equivalent) into `Real-Car-Data-And-Stats/Tire Models from TTC/raw/`. **If no**, Part 2 of this plan is descoped to "document the gap and keep the R25B transplant." Part 1 (loaded radius) and Part 3 (combined-slip robustness) proceed regardless.
- **U2 (LC0 fitter choice).** Default recommendation: open-source `scipy.optimize.least_squares` (cost: 0, reproducibility: high, lives in the repo). Alternative: OptimumTire ($~2k/yr academic license, gold standard, output is .tir directly). Recommend (A) unless the user already has an OptimumTire seat and prefers its UI for tuning convergence.
- **U3 (TTC raw-data format).** Will the user export TTC `.mat` files to tidy CSV (preferred — keeps the fit script clean), or should the fit script load `.mat` natively via `scipy.io.loadmat`? Default: assume CSV is provided. Add `.mat` ingest only if the user prefers.
- **U4 (single rolling radius vs per-wheel) — confirm the recommendation.** Plan recommends single rolling radius from mean Fz across the four tires for the longitudinal-only path. Per-wheel is an option if the user wants the maximum-correctness implementation; cost is a vector signature on `motor_rpm_from_speed` / `wheel_force` and a refactor of `apply_inverter_delivery`. Default: single.
- **U5 (validation tolerance for the kinematic motor-RPM check).** Plan targets ≤1 % residual on motor-RPM-vs-telemetry at 70 km/h cruise (above AiM RPM noise floor of ~10 RPM = ~0.7 % at 1500 RPM). Confirm or relax.

---

## File Decomposition

- **Modify**: `src/fsae_sim/vehicle/tire_model.py`
  - `combined_forces` (lines 471-592): add divide-by-zero / saturation tests; tighten guards if regression tests demand. No structural change unless tests fail.
  - `loaded_radius` (lines 690-711): add an optional `speed_ms` use (currently the parameter is accepted but unused in the linear-spring branch — document as deferred QV1 hook).
  - LC0 .tir files (`Round_8_Hoosier_LC0_16x7p5_10_on_8in_*psi_PAC02_UM2.tir` x 4 — 8/10/12/14 psi): rewritten by `scripts/fit_lc0_longitudinal.py` only after U1 is satisfied. Replaces PDX/PKX/PCX/PEX/PHX/PVX coefficients and (if combined-slip data is available) RBX/RCX/RBY/RCY/RVY coefficients. Direct edit of the four .tir text files via the same `replace_tir_coefficient` pattern from `scripts/transplant_fx_coefficients.py:110-156`.
- **Modify**: `src/fsae_sim/vehicle/powertrain_model.py`
  - Add `rolling_radius_for(fz_n: float | None) -> float` helper (~line 144, before `motor_rpm_from_speed`).
  - `motor_rpm_from_speed` (line 146): accept optional `tire_load_n: float | None = None`; route through `rolling_radius_for`.
  - `speed_from_motor_rpm` (line 165): same signature change for inverse symmetry.
  - `wheel_force` (line 490): same — optional `tire_load_n`.
  - `motor_torque_from_wheel_force` (line 502): same.
  - `regen_force` (line 532): same.
  - `apply_inverter_delivery` (line 449): add optional `tire_load_n`; the inverter-delivery map itself is rolling-radius-independent, but its caller chain (engine.py / speed_envelope.py / strategies.py) needs to pass through a load reference, which `apply_inverter_delivery` becomes the gate for.
  - `TIRE_RADIUS_M` class constant (line 67): keep as a back-compat constant marked `# DEPRECATED — use config.rolling_radius_m or tire_model.loaded_radius(fz)` to avoid breaking the test at `tests/test_powertrain_model.py:194`.
- **Modify**: `src/fsae_sim/vehicle/dynamics.py`
  - `__init__` `m_effective` calculation (lines 89-102): swap `tire_radius = powertrain_config.rolling_radius_m` for `tire_radius = self._mean_loaded_radius_at_static()` when a `tire_model` is attached. The static estimate uses car_mass · g / 4 = ~706 N/tire and `tire_model.loaded_radius(706)`. **Note: this changes `m_effective`. Agent C owns the regen/m_effective direction issue (M13); we are not touching that here, only the radius input to it.**
  - Add `mean_loaded_radius(fz_total_n: float) -> float` helper that calls `tire_model.loaded_radius(fz_total_n / 4)`. Not used directly by dynamics.py but exposed for `engine.py` and `speed_envelope.py` callers.
- **Modify (caller updates)**: `src/fsae_sim/sim/engine.py`, `src/fsae_sim/sim/speed_envelope.py`, `src/fsae_sim/driver/strategies.py` — each call site of `motor_rpm_from_speed` / `wheel_force` / `apply_inverter_delivery` passes the current Fz reference (mean of four tire loads from `LoadTransferModel.tire_loads(speed, lateral_g, long_g)`). For Pass 1 of this plan, default to mean static load (no per-segment update) — captures 80 % of the effect with zero plumbing complexity. Pass 2 (optional, if validation demands) updates per-segment using the existing load-transfer outputs.
- **New**: `scripts/fit_lc0_longitudinal.py` — open-source PAC2002 Fx fit for LC0 longitudinal data. Reuses `scripts/transplant_fx_coefficients.py:83-156` (`read_tir_coefficient`, `replace_tir_coefficient`, `_format_value`) helpers — extract those into `scripts/_tir_io.py` as a shared module so both scripts share one I/O path. Output: writes new PDX/PKX/PCX/PEX/PHX/PVX into the LC0 .tir files, prints per-pressure peak-mu RMSE on held-out load, and emits a comparison plot (transplanted-R25B vs. native-LC0 Fx envelope).
- **New (refactor)**: `scripts/_tir_io.py` — shared `.tir` file read/write helpers extracted from `transplant_fx_coefficients.py`. Lets the fit script and the transplant script share one I/O surface and prevents drift.
- **New tests**:
  - `tests/test_tire_loaded_radius_integration.py` — kinematic motor-RPM-vs-telemetry-at-known-v test (the headline acceptance test for Part 1). Plus unit tests for `rolling_radius_for(fz)` and per-call-site routing.
  - `tests/test_combined_slip_robustness.py` — property tests for Gxα/Gyκ bounds, divide guards, ellipse adherence at extreme slip.
  - `tests/test_lc0_longitudinal_fit.py` — peak-mu vs Fz vs camber regression against TTC raw data (gated on U1 — the test reads from `Real-Car-Data-And-Stats/Tire Models from TTC/raw/`, skips if missing).
- **Reference only (not modified)**:
  - `Real-Car-Data-And-Stats/CleanedEndurance.csv` — telemetry oracle for the kinematic motor-RPM check.
  - `scripts/transplant_fx_coefficients.py` — kept as historical record of the R25B→LC0 transplant; will be marked DEPRECATED in its module docstring once the native LC0 fit ships.

---

## Tasks

### Part 1 — Loaded-radius integration

#### Task 1.1: Failing kinematic acceptance test

**Files:**
- Create: `tests/test_tire_loaded_radius_integration.py` (new test file)

- [ ] **Step 1: Write the headline failing test**

The test reads a slice of Michigan endurance telemetry around v ≈ 70 km/h cruise (approx 19.4 m/s), median-filters RPM and GPS Speed, and asserts the gear-ratio identity holds to ≤1 %.

```python
# tests/test_tire_loaded_radius_integration.py
"""Kinematic check: motor RPM matches telemetry at known cruise speed.

If `motor_rpm_from_speed` uses the unloaded radius (0.2042 m) but the
real tire under load is rolling at ~0.196 m, the predicted RPM will be
~3-4% low at any cruise speed. This is the headline acceptance test for
the loaded-radius integration.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fsae_sim.vehicle.powertrain_model import PowertrainModel
# (use existing config + model fixtures from conftest.py / tests/test_powertrain_model.py)

TELEMETRY = Path("Real-Car-Data-And-Stats/CleanedEndurance.csv")


@pytest.mark.skipif(not TELEMETRY.exists(), reason="endurance telemetry not present")
def test_motor_rpm_matches_telemetry_at_70kph_cruise(model: PowertrainModel) -> None:
    """At 70 km/h the gear-ratio identity should hold to <=1%.

    With unloaded radius (0.2042 m) the prediction is ~3.5% low.
    With loaded radius from mean Fz (~196 mm) the residual should
    fall under 1% — within AiM RPM resolution (~10 RPM = 0.7%).
    """
    df = pd.read_csv(TELEMETRY)
    # 70 km/h cruise window: |GPS Speed - 70| < 2 km/h, |GPS LatAcc| < 0.2 g.
    cruise = df[
        (df["GPS Speed"].between(68.0, 72.0))
        & (df["GPS LatAcc"].abs() < 0.2)
        & (df["RPM"] > 100)
    ]
    assert len(cruise) > 30, f"insufficient cruise samples: {len(cruise)}"
    v_ms = cruise["GPS Speed"].median() / 3.6
    rpm_meas = cruise["RPM"].median()
    rpm_pred = model.motor_rpm_from_speed(v_ms)  # uses configured radius
    rel = abs(rpm_pred - rpm_meas) / rpm_meas
    assert rel < 0.01, (
        f"At v={v_ms:.2f} m/s motor RPM predicted={rpm_pred:.0f}, "
        f"telemetry={rpm_meas:.0f}, residual={rel:.2%}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_tire_loaded_radius_integration.py -v
```
Expected: FAIL with residual ~3-4 % (the size of the unloaded-vs-loaded radius gap).

- [ ] **Step 3: Commit the failing test**

```
git add tests/test_tire_loaded_radius_integration.py
git commit -m "test: failing motor-RPM kinematic check at 70 km/h cruise"
```

#### Task 1.2: `PowertrainModel.rolling_radius_for(fz)` helper

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain_model.py` (insert helper at ~line 142, before `motor_rpm_from_speed`)
- Create: `tests/test_rolling_radius_for.py` (small unit-test file)

- [ ] **Step 1: Write the failing helper test**

```python
# tests/test_rolling_radius_for.py
"""Unit tests for PowertrainModel.rolling_radius_for(fz)."""
import pytest
# (reuse model + tire_model fixtures)


def test_rolling_radius_with_no_tire_model_returns_config_value(model_no_tire):
    r = model_no_tire.rolling_radius_for(800.0)
    assert r == pytest.approx(0.2042)  # fallback to config


def test_rolling_radius_at_zero_load_equals_unloaded(model_with_tire):
    r = model_with_tire.rolling_radius_for(0.0)
    assert r == pytest.approx(0.2042)


def test_rolling_radius_at_700n_equals_linear_spring(model_with_tire):
    # r = r0 - Fz / kz = 0.2042 - 700/87914 = 0.19624 m
    r = model_with_tire.rolling_radius_for(700.0)
    assert r == pytest.approx(0.2042 - 700.0 / 87914.0, abs=1e-5)


def test_rolling_radius_with_none_load_returns_config_value(model_with_tire):
    """Backwards-compat: callers that don't yet pass a load get the
    static config value, not the unloaded value."""
    r = model_with_tire.rolling_radius_for(None)
    assert r == pytest.approx(0.2042)
```

- [ ] **Step 2: Run tests to verify they fail (no `rolling_radius_for` yet)**

```
pytest tests/test_rolling_radius_for.py -v
```
Expected: FAIL with `AttributeError: 'PowertrainModel' object has no attribute 'rolling_radius_for'`.

- [ ] **Step 3: Implement the helper**

In `src/fsae_sim/vehicle/powertrain_model.py`, around line 142 (before `motor_rpm_from_speed`):

```python
def rolling_radius_for(
    self, tire_load_n: float | None,
) -> float:
    """Effective rolling radius (m) for a given tire normal load.

    When a Pacejka tire model is attached, returns
    ``tire_model.loaded_radius(fz)`` — the static loaded radius from
    PAC2002 vertical stiffness. When no tire model is attached, or
    when the load is None, returns the configured ``rolling_radius_m``
    (static / unloaded value) for backwards compatibility.

    The Fz argument is the mean per-tire normal load (not total). The
    caller is responsible for averaging across the four wheels.

    Reference: Pacejka §1.3 / §4.3.6, Adams Tire 2018 PAC2002 docs.
    """
    if tire_load_n is None or self._tire_model is None:
        return self.rolling_radius_m
    return float(self._tire_model.loaded_radius(float(tire_load_n)))
```

Note: this requires storing a reference to the tire model on `PowertrainModel`. Add an optional `tire_model: PacejkaTireModel | None = None` to `__init__` (line 98-106) and store as `self._tire_model = tire_model`.

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_rolling_radius_for.py -v
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```
git add src/fsae_sim/vehicle/powertrain_model.py tests/test_rolling_radius_for.py
git commit -m "feat(powertrain): rolling_radius_for(fz) helper backed by Pacejka"
```

#### Task 1.3: Route `motor_rpm_from_speed` and `speed_from_motor_rpm` through `rolling_radius_for`

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain_model.py` lines 146-179.

- [ ] **Step 1: Write the failing routing test**

Append to `tests/test_rolling_radius_for.py`:

```python
def test_motor_rpm_uses_loaded_radius_under_load(model_with_tire):
    """Motor RPM at v=20 m/s with 700 N tire load should match the
    formula using r_loaded = 0.2042 - 700/87914 = 0.19624 m."""
    rpm = model_with_tire.motor_rpm_from_speed(20.0, tire_load_n=700.0)
    r_loaded = 0.2042 - 700.0 / 87914.0
    expected = (20.0 / r_loaded) * 60.0 / (2.0 * 3.14159265359) * 3.6363636
    assert abs(rpm - expected) / expected < 1e-3


def test_motor_rpm_with_no_load_kwarg_uses_config_radius(model_with_tire):
    """Backwards-compat: existing callers get the unloaded behavior."""
    rpm = model_with_tire.motor_rpm_from_speed(20.0)  # no kwarg
    expected = (20.0 / 0.2042) * 60.0 / (2.0 * 3.14159265359) * 3.6363636
    assert abs(rpm - expected) / expected < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail (signature lacks the kwarg)**

Expected: FAIL with `TypeError: motor_rpm_from_speed() got an unexpected keyword argument 'tire_load_n'`.

- [ ] **Step 3: Add the kwarg + route through `rolling_radius_for`**

```python
def motor_rpm_from_speed(
    self,
    vehicle_speed_ms: float,
    tire_load_n: float | None = None,
) -> float:
    speed = max(0.0, vehicle_speed_ms)
    r = self.rolling_radius_for(tire_load_n)
    wheel_rpm = (speed / r) * 60.0 / (2.0 * math.pi)
    return wheel_rpm * self.config.gear_ratio


def speed_from_motor_rpm(
    self,
    motor_rpm: float,
    tire_load_n: float | None = None,
) -> float:
    rpm = max(0.0, motor_rpm)
    r = self.rolling_radius_for(tire_load_n)
    wheel_rpm = rpm / self.config.gear_ratio
    return wheel_rpm * r * 2.0 * math.pi / 60.0
```

- [ ] **Step 4: Run tests + the headline kinematic test**

```
pytest tests/test_rolling_radius_for.py tests/test_tire_loaded_radius_integration.py tests/test_powertrain_model.py -v
```
Expected: all PASS, including the previously failing `test_motor_rpm_matches_telemetry_at_70kph_cruise`. If the kinematic check still fails, double-check the gear ratio (3.6363) and that the test fixture passes a static-mean Fz of ~706 N.

- [ ] **Step 5: Commit**

```
git add src/fsae_sim/vehicle/powertrain_model.py tests/test_rolling_radius_for.py
git commit -m "feat(powertrain): route motor_rpm_from_speed through loaded radius"
```

#### Task 1.4: Route `wheel_force` and `motor_torque_from_wheel_force`

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain_model.py` lines 490-507.
- Modify: `tests/test_powertrain_model.py` line 194 — the existing test asserts wheel_force uses `TIRE_RADIUS_M`; rewrite to use `rolling_radius_for(None)` so default behavior is preserved.

- [ ] **Step 1: Update the existing test to allow optional load kwarg**

In `tests/test_powertrain_model.py:194`:

```python
expected_force = model.wheel_torque(motor_torque) / model.rolling_radius_for(None)
assert model.wheel_force(motor_torque) == pytest.approx(expected_force)
```

- [ ] **Step 2: Write a new test asserting `wheel_force` uses loaded radius when given Fz**

Append to `tests/test_rolling_radius_for.py`:

```python
def test_wheel_force_uses_loaded_radius_under_load(model_with_tire):
    motor_torque = 50.0  # Nm
    f_default = model_with_tire.wheel_force(motor_torque)
    f_loaded = model_with_tire.wheel_force(motor_torque, tire_load_n=700.0)
    # Loaded radius is smaller -> force is larger for same wheel torque.
    assert f_loaded > f_default
    # Sanity: ratio should match radius ratio (within 1e-3).
    r0 = 0.2042
    r_loaded = 0.2042 - 700.0 / 87914.0
    assert abs((f_loaded / f_default) - (r0 / r_loaded)) < 1e-3
```

- [ ] **Step 3: Run tests to verify they fail**

Expected: FAIL with `TypeError` on the new kwarg.

- [ ] **Step 4: Add the kwarg + route through `rolling_radius_for`**

```python
def wheel_force(
    self,
    motor_torque_nm: float,
    tire_load_n: float | None = None,
) -> float:
    return self.wheel_torque(motor_torque_nm) / self.rolling_radius_for(tire_load_n)


def motor_torque_from_wheel_force(
    self,
    wheel_force_n: float,
    tire_load_n: float | None = None,
) -> float:
    denom = self.config.gear_ratio * self._GEARBOX_EFFICIENCY
    if denom <= 0.0:
        return 0.0
    return wheel_force_n * self.rolling_radius_for(tire_load_n) / denom
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_rolling_radius_for.py tests/test_powertrain_model.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```
git add src/fsae_sim/vehicle/powertrain_model.py tests/test_powertrain_model.py tests/test_rolling_radius_for.py
git commit -m "feat(powertrain): route wheel_force / motor_torque through loaded radius"
```

#### Task 1.5: Route `regen_force` and `apply_inverter_delivery`

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain_model.py` `regen_force` (line 532), `apply_inverter_delivery` (line 449), `drive_force` (line 513).

- [ ] **Step 1: Write tests**

Append to `tests/test_rolling_radius_for.py`:

```python
def test_regen_force_uses_loaded_radius_under_load(model_with_tire):
    f_default = model_with_tire.regen_force(0.5, 20.0)
    f_loaded = model_with_tire.regen_force(0.5, 20.0, tire_load_n=700.0)
    # Negative force; smaller radius -> larger magnitude.
    assert f_loaded < f_default  # both negative, loaded is "more negative"


def test_drive_force_uses_loaded_radius_under_load(model_with_tire):
    f_default = model_with_tire.drive_force(0.5, 20.0)
    f_loaded = model_with_tire.drive_force(0.5, 20.0, tire_load_n=700.0)
    # Smaller radius -> higher max torque envelope at given v + larger
    # force at the contact patch.
    assert f_loaded > f_default
```

- [ ] **Step 2: Add the kwargs to the three methods**

`drive_force`:

```python
def drive_force(
    self,
    throttle_pct: float,
    vehicle_speed_ms: float,
    tire_load_n: float | None = None,
) -> float:
    throttle = max(0.0, min(1.0, throttle_pct))
    rpm = self.motor_rpm_from_speed(vehicle_speed_ms, tire_load_n=tire_load_n)
    max_torque = self.max_motor_torque(rpm)
    commanded_torque = throttle * max_torque
    return self.wheel_force(commanded_torque, tire_load_n=tire_load_n)
```

`regen_force`: same pattern, route the rpm + wheel-force-equivalent computation through `rolling_radius_for`.

`apply_inverter_delivery` (line 449): keep its signature unchanged at this stage — it does not use radius directly. But its caller chain (engine.py / strategies.py) is what we need to update next so that the load reference flows through.

- [ ] **Step 3: Run tests**

Expected: PASS.

- [ ] **Step 4: Commit**

```
git add src/fsae_sim/vehicle/powertrain_model.py tests/test_rolling_radius_for.py
git commit -m "feat(powertrain): route drive_force / regen_force through loaded radius"
```

#### Task 1.6: Update `m_effective` in `dynamics.py` to use loaded radius

**Files:**
- Modify: `src/fsae_sim/vehicle/dynamics.py` lines 89-102.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamics_m_effective.py
def test_m_effective_uses_loaded_radius_when_tire_model_attached(
    vehicle, tire_model, load_transfer, powertrain_config,
):
    dyn = VehicleDynamics(
        vehicle, tire_model=tire_model, load_transfer=load_transfer,
        powertrain_config=powertrain_config,
    )
    # Static mean Fz = m·g/4 = 288·9.81/4 = 706 N.
    # r_loaded(706) = 0.2042 - 706/87914 = 0.19617 m.
    # j_eff = (rotor·G²·η + 4·wheel) / r²
    expected_r = 0.2042 - 706.0 / 87914.0
    j_eff_expected = (
        vehicle.rotor_inertia_kg_m2 * powertrain_config.gear_ratio ** 2
        * powertrain_config.drivetrain_efficiency
        + 4 * vehicle.wheel_inertia_kg_m2
    )
    expected_m_eff = vehicle.mass_kg + j_eff_expected / (expected_r ** 2)
    assert abs(dyn.m_effective - expected_m_eff) / expected_m_eff < 1e-4
```

- [ ] **Step 2: Modify `dynamics.py:89-102`**

```python
if powertrain_config is not None:
    # Use loaded radius at static mean Fz (mass·g/4) when a tire model
    # is attached, falling back to the configured static rolling radius
    # otherwise. This makes effective mass consistent with the loaded-
    # radius-aware powertrain methods so that motor RPM, wheel force,
    # and rotational inertia all share one driveline geometry.
    if tire_model is not None:
        static_fz_per_tire = vehicle.mass_kg * GRAVITY_M_S2 / 4.0
        tire_radius = float(tire_model.loaded_radius(static_fz_per_tire))
    else:
        tire_radius = powertrain_config.rolling_radius_m
    G = powertrain_config.gear_ratio
    eta = powertrain_config.drivetrain_efficiency
    j_eff = (
        vehicle.rotor_inertia_kg_m2 * G * G * eta
        + 4 * vehicle.wheel_inertia_kg_m2
    )
    self.m_effective = vehicle.mass_kg + j_eff / (tire_radius * tire_radius)
else:
    self.m_effective = vehicle.mass_kg
```

- [ ] **Step 3: Run tests**

```
pytest tests/test_dynamics_m_effective.py tests/test_dynamics.py -v
```
Expected: PASS. Some existing dynamics tests that hard-code `rolling_radius_m=0.2042` may need updating; preserve their original intent (legacy mode without tire_model) by leaving the `else` branch identical to today.

- [ ] **Step 4: Commit**

```
git add src/fsae_sim/vehicle/dynamics.py tests/test_dynamics_m_effective.py
git commit -m "feat(dynamics): m_effective uses loaded radius when tire model attached"
```

#### Task 1.7: Caller updates in `engine.py` / `speed_envelope.py` / `strategies.py`

**Files:**
- Modify: `src/fsae_sim/sim/engine.py` — every call site of `motor_rpm_from_speed`, `wheel_force`, `drive_force`, `regen_force` passes `tire_load_n` (mean Fz at the current segment).
- Modify: `src/fsae_sim/sim/speed_envelope.py` — same.
- Modify: `src/fsae_sim/driver/strategies.py` — same for the calibrated-strategy paths.

For Pass 1 (this plan), the caller passes a **single static estimate**: mean Fz = (vehicle.mass · g + downforce(speed)) / 4. This captures the dominant signal (mass + speed-dependent downforce) without per-segment load-transfer plumbing. Per-segment load-transfer hookup is a follow-on improvement gated on whether the kinematic test still has residual after Pass 1.

- [ ] **Step 1: Map every caller**

Use `Grep "motor_rpm_from_speed|wheel_force\(|drive_force\(|regen_force\("` over `src/fsae_sim/`. Expect ~12-15 hits. Document each site with a TODO inserting `tire_load_n=` that resolves to mean Fz.

- [ ] **Step 2: Write the integration test**

```python
# tests/test_engine_loaded_radius_integration.py
def test_engine_uses_loaded_radius_throughout(michigan_track, calibrated_strategy):
    """Run a single Michigan lap; assert that median operating-point
    motor RPM matches telemetry within 1% (this is the headline check
    repeated end-to-end through the engine, not just the powertrain
    method)."""
    # ...load Michigan telemetry, run sim, compare RPM at >5 m/s samples...
```

- [ ] **Step 3: Update each caller to pass `tire_load_n`**

For Pass 1, helper:

```python
# src/fsae_sim/sim/engine.py — top of the segment loop
def _mean_static_fz(self, speed_ms: float) -> float:
    df = self._dynamics.downforce(speed_ms)
    return (self._dynamics.vehicle.mass_kg * 9.81 + df) / 4.0
```

then every `self._powertrain.motor_rpm_from_speed(v)` becomes
`self._powertrain.motor_rpm_from_speed(v, tire_load_n=self._mean_static_fz(v))`.

- [ ] **Step 4: Run the full test suite**

```
pytest -q
```
Expected: all green. The headline kinematic test from Task 1.1 now passes end-to-end.

- [ ] **Step 5: Commit**

```
git add src/fsae_sim/sim/engine.py src/fsae_sim/sim/speed_envelope.py src/fsae_sim/driver/strategies.py tests/test_engine_loaded_radius_integration.py
git commit -m "feat(engine): plumb mean Fz through powertrain loaded-radius calls"
```

#### Task 1.8: Full-Michigan-stint regression

**Files:**
- Reference only: `Real-Car-Data-And-Stats/CleanedEndurance.csv`, `scripts/sim_compare.py`.

- [ ] **Step 1: Run `scripts/sim_compare.py --strategy replay --no-plots` before and after the loaded-radius work; compare net Ah and net kWh.**

```
git stash       # park current changes
python scripts/sim_compare.py --strategy replay --no-plots > /tmp/before.txt
git stash pop
python scripts/sim_compare.py --strategy replay --no-plots > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
```

- [ ] **Step 2: Acceptance**

- Replay-mode net Ah change: ≤0.5 % (pure plumbing change should not move integrated charge).
- Replay-mode lap-time change: ≤0.2 % (radius affects motor-RPM-derived efficiency lookups, so a small change is expected; a large change is a smoking gun for a missed call site).
- Calibrated-mode lap-time: should be slightly closer to telemetry (the previous 7.3 % delta was driver-model-dominated, not radius-dominated, so the change here is expected to be small but positive).

If either delta is larger than expected, audit the call-site map from Task 1.7 Step 1 — likely a missed site or an incorrect Fz reference (e.g. accidentally passing total Fz instead of per-tire mean).

### Part 2 — LC0 native longitudinal Fx (gated on U1)

#### Task 2.1: User confirmation + raw-data acquisition

**Status: BLOCKING.** Defer Part 2 until U1 is satisfied.

- [ ] **Step 1: User confirms TTC consortium membership.** UConn FSAE point of contact for TTC.
- [ ] **Step 2: User identifies which TTC Round logged LC0 longitudinal data.** Likely Round 8 combined sweep (file pattern `R8C_*.dat` / `.mat`). Possible alternate: Round 9 or a more recent round.
- [ ] **Step 3: User exports raw test data to tidy CSV.** Schema:
  ```
  test_id, p_psi, fz_n, slip_ratio, slip_angle_deg, camber_deg, fx_n, fy_n, mz_nm, vx_mps, t_c
  ```
  Place under `Real-Car-Data-And-Stats/Tire Models from TTC/raw/lc0_round8c_*.csv`. Do not commit raw .mat files (TTC license restricts redistribution; raw stays local). The .csv exports of fitted summaries are fine to commit because they are derived data, but check with the user before committing if they are uncertain about license boundaries.

#### Task 2.2: Extract shared `.tir` I/O helpers

**Files:**
- Create: `scripts/_tir_io.py` — extract `read_tir_coefficient`, `replace_tir_coefficient`, `_format_value` from `scripts/transplant_fx_coefficients.py`.
- Modify: `scripts/transplant_fx_coefficients.py` — import from `_tir_io`.

- [ ] **Step 1: Move the helpers, leave a thin shim in place.**
- [ ] **Step 2: Verify `python scripts/transplant_fx_coefficients.py` still produces byte-identical output on the existing LC0 .tir files.**
- [ ] **Step 3: Commit.**

#### Task 2.3: PAC2002 Fx fit script

**Files:**
- Create: `scripts/fit_lc0_longitudinal.py`.
- Create: `tests/test_fit_lc0_longitudinal.py` (synthetic-data fit-the-fitter test).

- [ ] **Step 1: Write the synthetic-data fit-the-fitter test.**

Generate a synthetic Fx surface from a known PAC2002 parameter vector, add 2 % Gaussian noise, run the fitter, assert recovered parameters are within 5 % of ground truth.

- [ ] **Step 2: Implement the fit script.**

```python
# scripts/fit_lc0_longitudinal.py — top-level structure
"""Fit PAC2002 Fx coefficients to LC0 TTC longitudinal sweep data.

Replaces the R25B-transplanted Fx coefficients in the LC0 .tir files
with values fit to the LC0's own measured Fx curve.

Reference: Pacejka, "Tyre and Vehicle Dynamics" (3rd ed.), §4.3.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from _tir_io import read_tir_coefficient, replace_tir_coefficient

PARAM_NAMES = (
    "PCX1", "PDX1", "PDX2", "PDX3",
    "PEX1", "PEX2", "PEX3", "PEX4",
    "PKX1", "PKX2", "PKX3",
    "PHX1", "PHX2",
    "PVX1", "PVX2",
)
LOWER_BOUNDS = (
    0.5, 0.5, -2.0, 0.0,
    -2.0, -2.0, -2.0, -2.0,
    1.0, -2.0, -2.0,
    -0.1, -0.1,
    -0.1, -0.1,
)
UPPER_BOUNDS = (
    2.5, 3.0, 2.0, 5.0,
    2.0, 2.0, 2.0, 2.0,
    50.0, 2.0, 2.0,
    0.1, 0.1,
    0.1, 0.1,
)


def fx_pacejka(params, slip, fz, fz0, camber):
    """Compute PAC2002 pure-Fx for a vector of operating points."""
    pcx1, pdx1, pdx2, pdx3, pex1, pex2, pex3, pex4, \
        pkx1, pkx2, pkx3, phx1, phx2, pvx1, pvx2 = params
    dfz = (fz - fz0) / fz0
    mux = (pdx1 + pdx2 * dfz) * (1.0 - pdx3 * camber ** 2)
    dx = mux * fz
    kxk = fz * (pkx1 + pkx2 * dfz) * np.exp(pkx3 * dfz)
    cx = pcx1
    bx = kxk / (cx * dx + 1e-6)
    shx = phx1 + phx2 * dfz
    svx = fz * (pvx1 + pvx2 * dfz)
    kappa_x = slip + shx
    sign_k = np.sign(kappa_x)
    sign_k = np.where(sign_k == 0, 1.0, sign_k)
    ex = (pex1 + pex2 * dfz + pex3 * dfz ** 2) * (1.0 - pex4 * sign_k)
    ex = np.clip(ex, -1.0, 1.0)
    bk = bx * kappa_x
    inner = bk - ex * (bk - np.arctan(bk))
    return dx * np.sin(cx * np.arctan(inner)) + svx


def fit(data: pd.DataFrame, fz0: float, x0: np.ndarray) -> dict:
    """Run a single LM-with-bounds fit on the supplied data slice."""
    fz = data["fz_n"].to_numpy()
    slip = data["slip_ratio"].to_numpy()
    camber = np.deg2rad(data["camber_deg"].to_numpy())
    fx_meas = data["fx_n"].to_numpy()

    def residual(p):
        return fx_pacejka(p, slip, fz, fz0, camber) - fx_meas

    result = least_squares(
        residual, x0, method="trf",
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        max_nfev=5000,
    )
    return dict(zip(PARAM_NAMES, result.x))


def cross_validate(data: pd.DataFrame, fz0: float, x0: np.ndarray) -> dict:
    """Hold-one-Fz-out cross-validation, return per-fold peak-mu RMSE."""
    rmse = {}
    for fz_holdout in sorted(data["fz_n"].unique()):
        train = data[data["fz_n"] != fz_holdout]
        test = data[data["fz_n"] == fz_holdout]
        params = fit(train, fz0, x0)
        fx_pred = fx_pacejka(
            np.array(list(params.values())),
            test["slip_ratio"].to_numpy(),
            test["fz_n"].to_numpy(),
            fz0,
            np.deg2rad(test["camber_deg"].to_numpy()),
        )
        peak_mu_pred = np.max(np.abs(fx_pred)) / fz_holdout
        peak_mu_meas = np.max(np.abs(test["fx_n"])) / fz_holdout
        rmse[fz_holdout] = abs(peak_mu_pred - peak_mu_meas) / peak_mu_meas
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--psi", type=int, required=True)
    parser.add_argument("--write-tir", action="store_true",
                        help="Write fitted coefficients into the LC0 .tir.")
    args = parser.parse_args()
    df = pd.read_csv(args.raw_csv)
    # ...read FNOMIN from .tir, run fit, run CV, optionally write .tir...


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run synthetic-data test.**

```
pytest tests/test_fit_lc0_longitudinal.py -v
```
Expected: PASS, recovered params within 5 % of ground truth.

- [ ] **Step 4: Run on real LC0 data (gated on U1).**

```
python scripts/fit_lc0_longitudinal.py \
    --raw-csv "Real-Car-Data-And-Stats/Tire Models from TTC/raw/lc0_round8c_10psi.csv" \
    --psi 10 --write-tir
```

- [ ] **Step 5: Commit fitter (without .tir changes — keep .tir under separate review commit).**

#### Task 2.4: Regression tests on the fitted .tir

**Files:**
- Create: `tests/test_lc0_fx_envelope.py`.

- [ ] **Step 1: Peak-mu vs Fz**.

Assert peak Fx / Fz at five Fz levels (220, 440, 660, 880, 1100 N) matches the TTC raw peak-mu within 7 % RMS (Wisconsin Racing's published threshold). Skip if `Real-Car-Data-And-Stats/Tire Models from TTC/raw/` is empty.

- [ ] **Step 2: Combined-slip continuity at the pure→combined transition.**

Assert `combined_forces(α=0.001 rad, κ, fz)` agrees with `longitudinal_force(κ, fz)` to within 1 % for κ ∈ {-0.1, 0, 0.1}.

- [ ] **Step 3: Friction ellipse adherence.**

Assert `sqrt((fx/peak_fx)² + (fy/peak_fy)²) ≤ 1.05` (5 % numerical tolerance) at α ∈ {2°, 5°, 10°} × κ ∈ {0.05, 0.1, 0.2}.

- [ ] **Step 4: Commit.**

#### Task 2.5: Optional — combined-slip RBX/RBY refit

**Files:**
- Modify: `scripts/fit_lc0_longitudinal.py` — add a combined-slip mode.

Only proceed if the TTC sweep includes combined-slip sub-tests (yaw + driving sweep at fixed Fz). If not, document that the R25B combined-slip transplant is retained and the LC0 fit improves only pure-Fx.

#### Task 2.6: Update `tire_model.py` docstring

Strike "transplanted from TTC Round 6 Hoosier R25B" from the longitudinal-coefficients docstring (`tire_model.py:11`) and replace with "fit to TTC Round 8 LC0 longitudinal sweep data via `scripts/fit_lc0_longitudinal.py`."

### Part 3 — Combined-slip robustness audit

#### Task 3.1: Property tests for Gxα / Gyκ bounds

**Files:**
- Create: `tests/test_combined_slip_robustness.py`.

- [ ] **Step 1: Hypothesis property test — Gxα ∈ [0, 1] for α ∈ [-0.5 rad, 0.5 rad], κ ∈ [-0.5, 0.5], Fz ∈ [200 N, 1500 N].**

```python
from hypothesis import given, strategies as st


@given(
    alpha=st.floats(min_value=-0.5, max_value=0.5),
    kappa=st.floats(min_value=-0.5, max_value=0.5),
    fz=st.floats(min_value=200.0, max_value=1500.0),
)
def test_gxa_bounded_in_unit_interval(tire_10psi, alpha, kappa, fz):
    """Gxα must monotonically decrease from 1 toward 0 as |α| grows."""
    fx_pure = tire_10psi.longitudinal_force(kappa, fz)
    fx_comb, _ = tire_10psi.combined_forces(alpha, kappa, fz)
    if abs(fx_pure) < 10.0:
        return  # near-zero Fx; ratio undefined
    gxa_implied = fx_comb / fx_pure
    assert -1e-3 <= gxa_implied <= 1.0 + 1e-3, (
        f"Gxα = {gxa_implied:.3f} out of [0,1] at α={alpha}, κ={kappa}, Fz={fz}"
    )
```

Note: the assertion above will surface if Pacejka's `cos(cxa·atan(...))` ever returns negative — which it can if the inner argument exceeds π/2. If the test fails, the fix is to clamp the inner cos argument or apply a final `max(0, gxa)` clamp consistent with Pacejka §4.3.4 (G factors must be non-negative).

- [ ] **Step 2: Mirror property test for Gyκ.**

Same structure on the lateral side.

- [ ] **Step 3: Friction-ellipse non-violation property test.**

```python
@given(
    alpha=st.floats(min_value=-0.3, max_value=0.3),
    kappa=st.floats(min_value=-0.3, max_value=0.3),
    fz=st.floats(min_value=300.0, max_value=1200.0),
)
def test_combined_force_within_friction_ellipse(tire_10psi, alpha, kappa, fz):
    fx, fy = tire_10psi.combined_forces(alpha, kappa, fz)
    peak_fx = tire_10psi.peak_longitudinal_force(fz)
    peak_fy = tire_10psi.peak_lateral_force(fz)
    if peak_fx < 1.0 or peak_fy < 1.0:
        return
    norm = (fx / peak_fx) ** 2 + (fy / peak_fy) ** 2
    assert norm <= 1.10, (
        f"Combined |F| outside friction ellipse: norm={norm:.3f} at "
        f"α={alpha}, κ={kappa}, Fz={fz}"
    )
```

The 10 % tolerance is intentionally loose because PAC2002's friction-ellipse violation is bounded but not zero (Pacejka §4.3.4 warns of a few percent overshoot at extreme combined slip in the orthodox formulation). If the test fails outright (norm > 1.5), that is a real bug.

#### Task 3.2: Divide-by-zero / NaN guards

- [ ] **Step 1: Test that all Pacejka outputs are finite for the full operating envelope above.**

```python
@given(...)
def test_combined_forces_returns_finite(tire_10psi, alpha, kappa, fz):
    fx, fy = tire_10psi.combined_forces(alpha, kappa, fz)
    assert math.isfinite(fx)
    assert math.isfinite(fy)
```

- [ ] **Step 2: Audit `_find_slip_angle` (`dynamics.py:200`) edge cases.**

Tests for: zero `f_lat_needed` (must return 0); `f_lat_needed > peak_Fy` (must return saturated `alpha_peak`); `normal_load < 1` (must return 0); the brentq fallback path (currently last in the chain).

- [ ] **Step 3: Speed envelope Pass 4 sign-convention tests.**

```python
def test_speed_envelope_pass4_handles_negative_long_g():
    """Decelerating into a corner (negative long_g) must not produce NaN
    or push corner speed to a non-physical value."""
    # ...build a single tight-corner track segment, force a high entry
    # speed so dv_sq < 0 in Pass 4, assert v_corrected is finite and <
    # max_cornering_speed at long_g=0.
```

- [ ] **Step 4: Commit.**

#### Task 3.3: Document the speed-envelope Pass 4 non-fixed-point note

- [ ] **Step 1: Add a comment in `speed_envelope.py:217-247` flagging that the re-propagation is single-pass (not iterated to fixed point) and recording the empirical bound: in Michigan endurance, the corrected v_corner change at the worst corner is < 0.4 m/s and re-propagation upstream changes are < 0.1 m/s, so a single round-trip is sufficient. If a future track has tighter corner sequences, this may need promotion to a fixed-point loop similar to the lap-wrap loop at lines 99-131.**

This is a doc-only change. No test.

---

## Risks / Unknowns

- **R1 (BLOCKING for Part 2): TTC access.** If U1 cannot be confirmed, Part 2 reduces to documenting the gap and updating the issue tracker. Parts 1 and 3 are unaffected.
- **R2: Loaded-radius bias is smaller than expected.** If the kinematic test from Task 1.1 only shows ~1 % residual (not the predicted 3-4 %), the AiM RPM signal may be drifted, the gear ratio may differ from documented (3.6363 from CLAUDE.md / DSS), or the wheel speed sensor is on the LF wheel (not the driven rear axle) and reflects free-rolling not loaded-driven kinematics. In that case re-validate with a different cruise window (e.g. 50 km/h) and decide whether the loaded-radius integration is worth the plumbing cost. Even at 1 % the change is correct in direction; just not as visible.
- **R3: New caller plumbing breaks an existing test.** The kwarg-default-None pattern preserves backwards compatibility, but existing tests that mock `motor_rpm_from_speed` may need their mock signatures relaxed. Use `Grep "Mock.*motor_rpm_from_speed|motor_rpm_from_speed.*Mock"` to find them up front.
- **R4: PAC2002 fit non-convergence on real LC0 data.** TTC raw data quality varies by Round; a fit that converges on synthetic data may not converge on noisy measured Fx. Mitigation: warm-start from the R25B-transplanted parameters (already in repo); if `least_squares` returns `status < 1`, fall back to a coarser two-stage fit (constant-mu first to lock PDX1/PCX1, then full-vector). Cross-validation peak-mu RMSE > 7 % flags a fit failure that needs human review.
- **R5: Combined-slip Gxα or Gyκ property test fails (Gxα goes negative or > 1).** This means the PAC2002 cos-of-atan formulation has a parameter combination that violates §4.3.4's monotonicity guarantee. Fix is to clamp Gxα and Gyκ to [0, 1] at the end of `combined_forces`. This is not a "bandaid" — Pacejka explicitly mandates the bound. The clamp protects against parameter combinations the orthodox formulation does not handle gracefully.
- **R6: m_effective regression.** Changing the rolling radius used in `m_effective` interacts with Agent C's M13 work. Coordinate by leaving the `m_effective` change in this plan strictly to the radius input — do not change accel-vs-regen direction. Agent C's M13 fix should compose cleanly because the radius lookup is now centralized.
- **R7: Test fixture explosion.** New tests across Parts 1, 2, 3 add ~12 new test files. Use shared fixtures in `conftest.py` for `tire_model`, `load_transfer`, `model_with_tire`, `model_no_tire` to avoid drift.

---

## Verification / Acceptance Criteria

### Part 1 — Loaded radius
- **A1.1**: New unit test `test_rolling_radius_for.py` — 6+ tests PASS covering fallback, zero load, FNOMIN load, None passthrough, kwarg routing on `motor_rpm_from_speed`, `wheel_force`, `regen_force`, `drive_force`.
- **A1.2 (HEADLINE)**: Kinematic test `test_motor_rpm_matches_telemetry_at_70kph_cruise` — residual ≤ **1 %** (≤ ~15 RPM at cruise) using mean-static Fz ≈ 706 N. Pre-fix residual is ~3.5 % (≈ 50 RPM). Native units: |RPM_pred − RPM_meas| ≤ 15 RPM.
- **A1.3**: `m_effective` test — at Michigan static load, `m_effective` shifts by ~1.5 % vs unloaded-radius baseline (rotor-inertia term scales with 1/r²; loaded r is ~4 % smaller, so 1/r² is ~8 % larger and the rotor-inertia contribution to m_effective grows ~8 %, producing a ~1.5 % change in total m_effective for typical FSAE inertias).
- **A1.4**: `scripts/sim_compare.py --strategy replay` regression — net Ah change ≤ **0.5 %** (≈ 0.05 Ah on a 10 Ah Michigan stint), lap-time change ≤ **0.2 %** (≈ 4 s on a 1900 s endurance), no test regressions.

### Part 2 — LC0 native longitudinal Fx (gated)
- **A2.1**: Synthetic-data fit-the-fitter test — recovered params within **5 %** of ground truth on 2 % Gaussian noise.
- **A2.2**: Per-pressure peak-mu vs Fz RMSE on held-out load ≤ **7 %** (Wisconsin threshold). Native: report |peak_μ_predicted − peak_μ_measured| in absolute mu units alongside %.
- **A2.3**: Combined-slip continuity at α=0.001 rad — `combined_forces(α, κ)` agrees with `longitudinal_force(κ)` within **1 %** for κ ∈ {-0.1, 0, 0.1}.
- **A2.4**: Friction-ellipse adherence — `(fx/peak)² + (fy/peak)² ≤ 1.10` at α ∈ {2, 5, 10}° × κ ∈ {0.05, 0.1, 0.2}.

### Part 3 — Combined-slip robustness
- **A3.1**: Hypothesis property test — Gxα and Gyκ within **[-0.001, 1.001]** across α ∈ [-0.5, 0.5] rad, κ ∈ [-0.5, 0.5], Fz ∈ [200, 1500] N (1000+ random examples).
- **A3.2**: Friction-ellipse non-violation property test — combined |F| within **1.10×** the ellipse across the same envelope.
- **A3.3**: Finite-output property test — `fx, fy` from `combined_forces` are finite (no NaN, no Inf) across the full envelope.
- **A3.4**: `_find_slip_angle` edge-case tests — zero demand returns 0; demand > peak returns alpha_peak; negative load returns 0.
- **A3.5**: Speed envelope Pass 4 sign-convention test — negative `long_g` (decel into corner) does not produce NaN or non-physical corrected speeds.

### End-to-end
- **AE.1**: `pytest -q` — all green on the main branch after each task's commit.
- **AE.2**: `scripts/sim_compare.py --strategy replay --no-plots` and `--strategy calibrated --no-plots` both run cleanly with no warnings beyond pre-existing ones; net Ah / kWh deltas reported in native units AND %.

---

## Effort Estimate

5h tiers (Superpowers convention):

- **Part 1 — Loaded-radius integration**: **2 tiers** (~10 h)
  - Task 1.1-1.2: 1 tier (failing test + helper, ~5 h)
  - Tasks 1.3-1.7: 1 tier (caller routing + integration, ~5 h)
  - Task 1.8: rolled into the second tier (regression + commit, ~30 min)
- **Part 2 — LC0 native longitudinal Fx (gated on U1)**: **3 tiers** (~15 h)
  - Task 2.1: ~30 min (user action, blocking but agent-side small)
  - Task 2.2: 0.5 tier (~2.5 h, helper extraction + parity check)
  - Task 2.3: 1 tier (~5 h, fit script + synthetic test)
  - Task 2.4: 0.5 tier (~2.5 h, real-data fit + cross-validation + .tir write)
  - Task 2.5: 0.5 tier (~2.5 h, optional combined-slip RBX/RBY refit)
  - Task 2.6: rolled in (~30 min, docstring updates)
- **Part 3 — Combined-slip robustness**: **1 tier** (~5 h)
  - Tasks 3.1-3.3 are tightly coupled; ~5 h end to end including any guard fixes if a property test surfaces a real issue.

**Total**: 6 tiers (~30 h) full plan, or 3 tiers (~15 h) if Part 2 is gated out.

If U1 is "no" (TTC access not confirmed), Parts 1 + 3 = **3 tiers** is the deliverable.
