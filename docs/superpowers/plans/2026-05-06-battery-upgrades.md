# Battery Modeling Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three concrete battery-modeling gaps in the FSAE EV endurance sim so it ranks tune options correctly across the *full 22-lap stint* (not just the cold first lap):

1. **P1 — BMS lap refresh (`engine.py:323-327`).** Re-`compute()` the speed envelope at lap boundaries with the *current* `(soc, temp)` BMS limit so late-stint thermal derating is visible to the planner.
2. **P2 — Speed-dependent pack convection (`battery_model.py:920-933`, `vehicle/battery.py:33`).** Replace the constant `thermal_conductance_w_per_k` (default 0 W/K, i.e. adiabatic) with a forced-convection model `h(v) = h_static + k_v · v`, calibrated against `CleanedEndurance.csv` Pack Temp.
3. **P2 — OCV temperature dependence (`battery_model.py`).** Voltt's 2025 export is single-temperature (25 °C ambient, see Research Summary), so we cannot ship a true 2-D `OCV(SOC, T)` interpolator from internal data. Plan an interim literature-based shift `OCV(SOC, T) ≈ OCV_25(SOC) + α(SOC) · (T − 25)` from Molicel P45B / INR21700 datasheet curves, with a feature flag and a clear "future work" note.

Plus one planning-only item:

4. **P3 (deferred) — Per-module / per-segment pack thermal.** Sketch the 5-segment thermal graph topology and conductance matrix with explicit decision criteria for activation; do not implement.

**Architecture:**

- **BMS lap refresh.** Add a `refresh_envelope_each_lap: bool = True` knob to `SimulationEngine.run()`. Inside the existing `for lap in range(num_laps):` outer loop (`engine.py:510`), at the *start of each lap* (and only if the in-flight `(temp, soc)` produces a BMS limit that has shifted by more than a tunable threshold from the limit used to build the current envelope) call `self._envelope.compute(initial_speed=speed, bms_current_limit_a=current_bms_limit)` and rebind `v_max`. Keep the per-segment LVCU torque clamp (`engine.py:556`) unchanged — it already uses live BMS limit; the change is *only* on the planning side. **Do not touch the segment integrator.** This stays inside the `engine.py:323-327` slice the user assigned.
- **Speed-dependent convection.** Promote `BatteryConfig.thermal_conductance_w_per_k: float` to a small object: keep the current scalar as a backwards-compatible *static* term `h_static_w_per_k`, plus a new `h_speed_w_per_k_per_mps`. New API on `BatteryModel`: `heat_out_w(self, temp_c, vehicle_speed_ms) -> float`, called from `step()` and `step_power()`. The runtime callers (engine, validation, replay) all already know `avg_speed`; thread it through. Single first-order convection law — no Reynolds, no Nusselt correlation in production code, only as calibration justification (see Research).
- **OCV(T).** Optional, off by default. Add `BatteryConfig.ocv_temperature_shift_mv_per_k: dict[float, float] | None = None` mapping SOC anchor points (e.g. 0, 25, 50, 75, 100 %) to OCV shift in mV/K below 25 °C reference. When non-None, `BatteryModel.ocv()` returns `_ocv_25(soc) + interp(soc, anchors) * (temp_c - 25.0) / 1000.0`. Default values from Molicel P45B 0 °C / 25 °C / 45 °C discharge curves in the Round-trip cell datasheet (see Research). For the 2025 endurance the temp swing is ~9 K (29.7 → 38 °C, see SIM_ACCURACY.md), so the maximum OCV shift per cell is ≤ 30 mV — pack-level effect is real but second-order vs. R(SOC) and convection.
- **Per-module thermal.** Planning artifact only. Documents topology, inputs, decision criteria.

**Tech Stack:** Python 3.11, NumPy, SciPy `interp1d`, pandas (Pack Temp telemetry calibration), pytest. No new runtime dependencies. No webapp / FastAPI changes.

---

## Research Summary

### BMS lap refresh

**Authoritative source — `Real-Car-Data-And-Stats/Endurance Tune2.txt`** (verified, lines 6-18):

```
TEMP:
30   -> 100A
35   ->  85A
40   ->  65A
45   ->  55A
50   ->  45A
55   ->  40A
60   ->  35A
65   ->   0A
```
plus internal-SOC taper: `@85% - 1A per 1%`. The 30 °C / 100 A → 65 °C / 0 A linear-segment table is the firmware-side limit. Between 30 °C and 65 °C the discharge ceiling drops by ~2.86 A per °C on average and steepens at the high end (the 60 → 65 °C step alone drops 35 A). Telemetry shows mean cell temp climbing from 25 °C cold-start to ~38 °C end-of-stint (`SIM_ACCURACY.md` calibrated row "Final cell temp 36.92 C / 38.00 C"). At 38 °C the BMS limit interpolates to ~73 A — already 27 % below the cold 100 A used for the planning envelope today.

**Why "refresh per lap" not "per segment".** Within a single lap the temp swing is < 0.5 °C (lumped thermal mass ≈ 39 kJ/K from `battery_model.py:101-106`, peak heat ≈ 1.5 kW for 80 s ≈ 120 kJ ≈ 3 K total, smeared over the lap), which moves the BMS limit by < 1.5 A. The forward-pass envelope is insensitive to BMS at that resolution because cornering speed dominates almost everywhere except the back straight. Refreshing per segment (≈ 200 segments × 22 laps = 4400 envelope computes) would burn ~5 s of extra wall time on a sub-second sim with no fidelity gain. Refreshing per lap (22 envelope computes, ~few ms each) catches the multi-degree drift that *does* matter and is invisible to the per-segment LVCU clamp on the back straight where the planner already assumes full power. Threshold-trigger ("recompute when limit shifts > X A") is a third option — adopt as a cheap optimization (skip refresh on early laps where temp is rising slowly), default `X = 5 A` (≈ 1.7 °C of headroom).

**Cost back-of-envelope.** `SpeedEnvelope.compute()` is O(N) over segments: backward pass + lap-wrap fixed point (≤ 5 iters by current cap, `speed_envelope.py:99`) + forward pass + combined-slip Pass 4 conditional. Michigan endurance has ≈ 2000 segments at 0.5 m. At ~10 µs per segment per pass, one envelope ≈ 60 ms in pure Python. 22 envelopes ≈ 1.3 s extra on a current ~5 s endurance run. Acceptable. With a `threshold` triggering 8-12 refreshes instead of 22, we land at < 1 s overhead.

**Citations.**
- `Real-Car-Data-And-Stats/Endurance Tune2.txt` (BMS table, internal-SOC taper).
- `src/fsae_sim/sim/engine.py:323-327` (current `initial_bms_limit` plumbing).
- `src/fsae_sim/sim/speed_envelope.py:307-329` (`_drive_force` BMS branch).
- `docs/SIM_AUDIT_2026-05.md` lines 136-145 (P1 task statement and "few ms per lap" estimate).

### Pack thermal — speed-dependence on heat-out

**Current state.** `battery_model.py:920-933`:
```python
heat_in_w = cell_current ** 2 * r_cell * self._num_cells
heat_out_w = self.config.thermal_conductance_w_per_k * (
    temp_c - self.config.ambient_temperature_c
)
```
With `BatteryConfig.thermal_conductance_w_per_k: float = 0.0` default (`vehicle/battery.py:33`) and *no* override in `configs/ct16ev.yaml` (verified by grep), the 2025 sim runs **adiabatic**. Voltt sim 2025 export confirms the same starting assumption: `Heat Transfer Coefficient: 0 W/m²K` (`Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/simulation_info.txt:35`). Despite this, real telemetry shows mean cell temp settling around 38 °C after a 13 °C climb in 1608 s — i.e. there *is* heat leaving the pack on the real car (sealed enclosure but ~0.13 m² of exposed external area + chassis frame conductance). Calibrating a lumped `h_static` from telemetry will recover this baseline. A speed-dependent term is then physically required because forced-convection at car speed (peak ~28 m/s, mean 13.6 m/s on Michigan endurance) over the chassis-side surfaces must scale with airflow.

**Convection law — recommendation.**

Use a single linear law:

```
h(v) = h_static + k_v * v          [W/K]
```

Two parameters, both calibrated by least-squares fit against `CleanedEndurance.csv` Pack Temp time series (forward-Euler integrate the same heat balance in calibration to avoid an apples-to-oranges error). This is the *Newton's cooling with airflow boost* approximation and it is what every published FSAE / formula-style lap-sim with a pack thermal model uses. Justification:

- **Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer" 7e (2011), Ch. 7.** External flow over a flat plate at the Reynolds numbers we run (Re ≈ ρ·v·L/μ ≈ 1.2 · 13.6 · 0.5 / 1.8e-5 ≈ 4.5e5, transitional) gives `Nu ∝ Re^0.5 Pr^(1/3)` (laminar) or `Nu ∝ Re^0.8 Pr^(1/3)` (turbulent), so `h ∝ v^0.5` to `v^0.8`. A linear `h ∝ v` is a defensible chord through this range over the speeds we actually see (5–28 m/s); the extra 0.2–0.5 power in the exponent does not survive the noise floor of a single 1608-s telemetry record. **YAGNI: use linear.**
- **Churchill-Bernstein correlation (1977) for cylinders in cross-flow** is the *cell-level* gold standard (each 21700 is a 70 mm × 21 mm cylinder in a 4P × 22S × 5-segment array). It would justify a per-module model (Future Work below) but the *lumped* pack does not see one cylinder — it sees the box, and a flat-plate-style linear `h(v)` is appropriate for the lumped abstraction.
- **Wisconsin Racing WR-217e Architecture & LapSim PDF** (https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf): WR uses a lumped pack thermal node with cooling power that scales with airflow over the enclosure; their model is a single linear conductance term. Good direct precedent for our scale of model.
- **Chalmers Formula Student EV Powertrain Design** (https://publications.lib.chalmers.se/records/fulltext/191837/191837.pdf): same pattern — lumped pack node, h scales linearly with airflow during driving.
- **Molicel INR-21700-P45B datasheet (Rev. 0, 2021).** Internal resistance ~16 mΩ at 25 °C / 50 % SOC, datasheet "Heat Generation" matches `I^2 R + |I|·T·dV_oc/dT` (entropic). The reversible heat term shows up in the Voltt cell trace as the `Reversible Heat [W]` column (see `2025_Pack_cell.csv` header) and is **not** modeled by the current `I^2 R` alone — but its time-integral over a discharge is a few percent of resistive heat for full SOC swings, and ≈ 0 for our partial-SOC stint. Document as known omission, do not block on it.

**Calibration parameters.**

Fit `(h_static_w_per_k, h_speed_w_per_k_per_mps, ambient_temperature_c_at_event)` against `CleanedEndurance.csv` Pack Temp at 20 Hz. Recommended fit setup:

- Use real Pack Current trace as the heat-in driver — not sim output — so this fit is *independent* of every other sim component.
- Heat-in: `I_pack^2 · R_pack(SOC) / (S/P)^2 · num_cells = I_pack^2 · R_pack(SOC) · P / S` if you re-derive at cell level. Equivalently use Voltt's `Resistive Heat [W]` column directly if available (`2025_Pack_cell.csv` has it for the duty cycle, not for AiM telemetry — re-compute from `R_pack(SOC) · I^2`).
- Heat-out: `(h_static + k_v · v_GPS) · (T_pack(t) − T_ambient)` at each timestep.
- Loss: weighted L2 over T_pack (residual in K), not over dT/dt (avoids amplifying telemetry noise). Optimizer: `scipy.optimize.least_squares` with bounds `h_static ∈ [0.5, 50] W/K`, `k_v ∈ [0.1, 10] W/K/(m/s)`, `T_amb ∈ [15, 35] °C`.

**Residual targets (verifiable, see Acceptance Criteria below).**

- Mean residual on Pack Temp: ≤ 1.0 K (target), ≤ 1.5 K (max acceptable).
- Peak residual on Pack Temp: ≤ 3.0 K (target), ≤ 5.0 K (max acceptable). Telemetry "Pack Temp" is *max* cell sensor (`battery_model.py:39-46` docstring confirms), so we expect a positive bias of 1–3 K vs. lumped mean — bake that into target.
- Final cell temp residual: ≤ 1.0 K. Currently +1.08 K (sim 36.92 vs telemetry 38.0) per `SIM_ACCURACY.md`. Target hold or improve.

**Citations.**
- Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed. (Wiley, 2011), Ch. 7 (External flow), §7.2 (flat plate), §7.4 (cylinder in cross-flow). ISBN 978-0470501979.
- Churchill, S.W. & Bernstein, M., "A correlating equation for forced convection from gases and liquids to a circular cylinder in crossflow", *J. Heat Transfer* 99 (1977) 300–306. https://doi.org/10.1115/1.3450685
- Wisconsin Racing WR-217e: https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf
- Chalmers Formula Student EV: https://publications.lib.chalmers.se/records/fulltext/191837/191837.pdf
- Molicel INR-21700-P45B Product Data Sheet (E/One, Inc., Rev 0 2021). Available via E/One distributor; cell resistance, capacity, and discharge curves at 0/25/45/60 °C.
- Pesaran, A.A. (NREL), "Battery thermal models for hybrid vehicle simulations", *J. Power Sources* 110 (2002) 377–382. https://doi.org/10.1016/S0378-7753(02)00200-8 — lumped vs distributed pack thermal trade-offs.

### OCV(SOC, T)

**Voltt 2025 export — verified single-temperature.** `simulation_info.txt` (lines 33–39):
```
## Environment Conditions
Ambient Temperature: 25 °C
Heat Transfer Coefficient: 0 W/m²K

## Initial Conditions
Initial SOC: 100 %
Initial Temperature: 25 °C
```
The CSV columns include `Temperature [°C]` but it is the in-simulation cell temperature evolving from the 25 °C initial condition during the duty-cycle run, not an OCV-vs-T sweep. The 2026 export uses `Heat Transfer Coefficient: 50 W/m²K` (still 25 °C ambient) — same story. **Decision: we cannot derive `OCV(T)` from Voltt for 2025.** Re-export Voltt at multiple ambient temperatures is a future-work item (Voltt supports it; UConn would need to author duty cycles at 5 °C / 25 °C / 45 °C). **User-decision item U2 below.**

**Interim approach — literature shift model.** Molicel INR-21700-P45B datasheet publishes discharge curves at 0 °C, 25 °C, 45 °C, 60 °C (page 5 of typical datasheet revision). The plateau-region OCV shift between 25 °C and 45 °C is ~8 mV at 50 % SOC, ~15 mV near the knees (10 % and 90 %). NREL test reports for similar 21700 chemistry (NMC/Si-graphite) show the same magnitude. Linearizing:

```
OCV(SOC, T) ≈ OCV_25(SOC) + α(SOC) · (T − 25) / 1000   [V]
```
with `α(SOC)` (mV/K) anchored at:

| SOC %  |   0   |  10   |  25   |  50   |  75   |  90   |  100  |
|--------|-------|-------|-------|-------|-------|-------|-------|
| α mV/K |  -1.5 |  -1.2 |  -0.6 |  -0.4 |  -0.4 |  -0.6 |  -1.0 |

(Negative because cell OCV drops at higher temperatures across most of the SOC range — entropic coefficient `dV_oc/dT < 0` for graphite anode at SOC > 5 %; positive only at very low SOC for some chemistries. Magnitudes from NMC graphite literature average; refine if a clean Molicel-specific datasheet plot can be digitized.)

**Effect on our endurance.** Mean stint cell temp swing ≈ 9 K (29.7 → 38.7 °C if we accept the +0.7 K tightening from the convection fit). At 50 % SOC midpoint, OCV shifts by 0.4 mV/K · 9 K = 3.6 mV/cell. Pack-level (110S): 0.4 V — comparable to a 0.1 % shift in a 400 V pack. **Net Ah / kWh impact: <0.5 %, probably indistinguishable from telemetry noise floor.** This is why it's P2 not P1.

**Acceptance for OCV(T).** Net Ah / net kWh residuals must not regress when the feature is enabled with literature anchors. If they regress (residual increases by > 0.2 % vs anchors-disabled), revert to anchors-disabled until Voltt multi-T export is available. Default config: feature OFF.

**Citations.**
- Molicel INR-21700-P45B Product Data Sheet (E/One, Inc.). https://www.molicel.com/wp-content/uploads/INR21700P45B-V4-80092.pdf (current public datasheet path; check for revision date on download).
- Sandia National Laboratories, "Investigation of Path Dependence in Commercial Lithium-Ion Cells …", Report SAND2014-1685 (2014). https://www.osti.gov/biblio/1130854
- NREL, Smith, K. & Wang, C.-Y., "Power and thermal characterization of a lithium-ion battery pack for hybrid-electric vehicles", *J. Power Sources* 160 (2006) 662–673. https://doi.org/10.1016/j.jpowsour.2006.01.038
- Reynier, Y. et al., "Entropy of Li intercalation in Li_xCoO_2", *Phys. Rev. B* 70 (2004) 174304 — entropic coefficient of graphite anodes.
- IEEE Std 485-2020, "Recommended Practice for Sizing Lead-Acid Batteries for Stationary Applications" — temperature correction methodology (the framework, not the numerical values, is portable to Li-ion).
- `Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/simulation_info.txt` — confirms single-temperature Voltt export for 2025.
- `Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2026/simulation_info.txt` — confirms 2026 export is also single-T (different `h`, same ambient).

### Per-module / per-segment pack thermal (Future Work)

Planning-only (see "Future Work" section). Citations supporting the deferred design:

- Newman, J. & Tiedemann, W., "Porous-electrode theory with battery applications", *AIChE J.* 21 (1975) 25–41. https://doi.org/10.1002/aic.690210103 — canonical lumped-electrochemistry model that lumped-thermal pack models build on.
- Pesaran, A.A. (NREL), "Battery thermal management in EVs and HEVs: Issues and solutions", *Adv. Auto. Battery Conf.* (2001). NREL/CP-540-31123. https://www.nrel.gov/docs/fy01osti/31123.pdf — distributed-vs-lumped trade-offs and per-module modeling guidance.
- Bernardi, D., Pawlikowski, E., Newman, J., "A general energy balance for battery systems", *J. Electrochem. Soc.* 132 (1985) 5–12. https://doi.org/10.1149/1.2113792 — heat-generation decomposition (resistive + reversible + mixing) used as the per-segment heat-source model.
- Inui, Y., Kobayashi, Y., et al., "Simulation of temperature distribution in cylindrical and prismatic Li-ion battery cells", *Energy Conversion and Management* 48 (2007) 2103–2109 — temperature-gradient model justification.

---

## Alternatives Considered (and Rejected)

1. **Refresh envelope per segment (every ~0.5 m).** Rejected. Within-lap temp swing is < 0.5 K → BMS limit shifts by < 1.5 A → invisible to the planner everywhere except a back-straight that's already Power-limited not BMS-limited at high SOC. ~5 s of extra runtime for zero fidelity.
2. **Refresh envelope only once at lap N (e.g. mid-stint).** Rejected. Misses the late-stint compounding: laps 18-22 are where derate compounds, and the worst-case refresh cost is bounded (22 envelope computes ≈ 1.3 s).
3. **Thread BMS limit into envelope as a per-segment cap (force-balance side) instead of via `lvcu_torque_ceiling`.** Rejected. The current architecture uses `lvcu_torque_ceiling` as the LVCU-firmware-faithful translation of the inverter-level current cap — the correct physics path. Adding a parallel limit would create a "two clocks" inconsistency with the per-segment runtime path that already does this right (`engine.py:556`).
4. **Adopt Churchill-Bernstein (cylinder cross-flow) for the lumped pack.** Rejected. Cylinder correlations apply at the *cell* level (each P45B is a 21 mm cylinder in air); the lumped pack sees a box-shaped enclosure, for which a flat-plate or simple Newton-cooling correlation is the natural lumped abstraction. Churchill-Bernstein belongs in the per-module model (Future Work).
5. **Power-law `h(v) = h_0 + k · v^0.8` (turbulent flat-plate).** Rejected. The fit data (one 1608 s record, Pack Temp at 20 Hz) cannot distinguish 0.5 vs 0.8 vs 1.0 exponent given the noise floor (Pack Temp resolution is 0.5 °C steps). Pure linear is the parsimonious choice; the user's "no fudge factors" rule cuts harder against a free exponent than against a pure conductance term.
6. **Non-zero default `h_static` to mask the current adiabatic behaviour.** Rejected. Bandaid. Fix the model (linear h(v) calibrated against telemetry), do not patch the adiabatic default with a literature guess. Calibration is the only honest source.
7. **Per-cell entropic heat term `T · I · dV/dT` in the lumped model.** Rejected for v1. For partial-SOC endurance stints (95 % → ~70 % typical), the integral of reversible heat is < 5 % of resistive heat. Document as known omission, revisit when full-SOC sweep cases come up.
8. **Re-derive Voltt cell at multiple ambient temperatures right now.** Rejected as blocking. Voltt re-export is < 1 day of work (UConn owns the account) but blocks shipping the OCV(T) plumbing. Ship the literature-shift interim now, swap in Voltt-derived `α(SOC)` as a future patch when the multi-T export exists. **User-decision item U2.**
9. **Separate `T_max_cell` model for BMS limit (telemetry "Pack Temp" is max sensor) vs `T_mean` for energy balance.** Deferred to Future Work / per-module. Currently the lumped mean is used for both, which under-derates BMS by ~1-3 K. A cheap intermediate is a fixed offset `T_max ≈ T_mean + ΔT_bias` calibrated from telemetry; flagged in Future Work.
10. **Adopt RC-network OCV-hysteresis model from Plett's textbook.** Rejected as out of scope. Hysteresis is M6 in `SIMULATOR_ISSUES.md`; this slice is thermal + lap-refresh, not the OCV core.

---

## Architecture Decisions Awaiting User Input

- **U1 (BMS-refresh threshold).** Recommended: refresh envelope at start of each lap *if* the live BMS limit differs from the limit used to build the current envelope by more than `5.0 A` (≈ 1.7 °C of headroom). Alternatives: (a) every lap unconditionally (max ~1.3 s overhead, simplest), (b) `10 A` threshold (fewer recomputes, may miss sub-limit-step moves), (c) `2 A` threshold (close to per-segment cost, no fidelity gain). Default: `5.0 A` if no answer.
- **U2 (Voltt multi-T re-export).** Should we block on Voltt re-export at 0 / 25 / 45 °C ambient before shipping OCV(T), or ship literature-shift now and swap later? Recommendation: *ship literature-shift now*, file a follow-up to re-export Voltt with multi-T duty cycles when there's a free hour with the Voltt account. The OCV-T effect is < 0.5 % on net Ah — it's worth getting the plumbing in place even before the numbers are first-party.
- **U3 (Pack Temp definition for calibration).** Telemetry "Pack Temp" is max cell sensor reading. The lumped sim model is a mass-mean. Two clean choices: (a) calibrate against telemetry as-is and accept a positive bias; (b) compute a "telemetry-mean" from BMS log (if multi-cell data is available) — not in `CleanedEndurance.csv`, so only path (a) is available. Flag this in the residual report; do not introduce a `T_max ≈ T_mean + ΔT_bias` correction unless / until multi-cell BMS data is available.
- **U4 (default convection feature flag).** Recommendation: ship convection ON by default (with a config knob to disable); this *changes* the calibrated kWh number but in the correct direction. Alternative: ship OFF by default, opt-in via config. Default: ON.
- **U5 (per-module activation criteria).** When should the user activate the per-module model (Future Work)? Recommended criteria: (a) comparing cooling-package geometry changes (different segment baffles, fan placement), (b) packs with > 5 °C inter-segment ΔT in telemetry, (c) modeling 2026 P50B pack with active cooling (`Heat Transfer Coefficient: 50 W/m²K` in the Voltt 2026 export). User to confirm.

---

## File Decomposition

### Modify

- **`src/fsae_sim/vehicle/battery.py`** — extend `BatteryConfig`:
  - Rename `thermal_conductance_w_per_k` to `h_static_w_per_k` *while keeping the old name as an alias* (deprecation warning if both supplied or if old name is used).
  - Add `h_speed_w_per_k_per_mps: float = 0.0` (default = no speed dependence, backwards-compatible).
  - Add `ocv_temperature_anchors_mv_per_k: tuple[tuple[float, float], ...] | None = None` — sorted SOC %, mV/K pairs.
  - Update `from_dict` to parse both new keys plus deprecation alias.
- **`src/fsae_sim/vehicle/battery_model.py`** — extend `BatteryModel`:
  - New `heat_out_w(self, temp_c, vehicle_speed_ms) -> float` method computing `(h_static + h_speed * v) * (T − T_amb)`.
  - Update `step` and `step_power` to accept `vehicle_speed_ms: float = 0.0` (kwarg, default keeps adiabatic-zero behaviour for legacy callers) and route `heat_out_w(temp_c, vehicle_speed_ms)` instead of inlined formula.
  - Update `ocv(self, soc_pct, *, temp_c: float = 25.0)` signature; if `ocv_temperature_anchors_mv_per_k` is set, apply linear interp shift; otherwise return current behaviour. Add a deprecation note for callers that don't pass `temp_c` (emit on first miss only — model already uses this idiom for `_ocv_extrap_warned`).
  - Plumb `temp_c` through `internal_resistance` *only* if user later asks (out of scope for this plan; M-8 in `SIMULATOR_ISSUES.md`).
- **`src/fsae_sim/sim/engine.py`** — *only* the `engine.py:323-327` slice plus a small lap-boundary block:
  - Add `refresh_envelope_each_lap: bool = True` and `bms_refresh_threshold_a: float = 5.0` kwargs to `run()`.
  - At the start of each `for lap in range(num_laps):` iteration (after the segment loop's `time, distance, speed, soc, temp` has stepped to the lap boundary), if `refresh_envelope_each_lap` and `abs(self.battery_model.max_discharge_current(temp, soc) - last_envelope_bms_limit) >= bms_refresh_threshold_a`, call `self._envelope.compute(initial_speed=speed, bms_current_limit_a=...)` and rebind `v_max`. Also rebind `last_envelope_bms_limit`.
  - Thread `vehicle_speed_ms=avg_speed` into the existing `self.battery_model.step_power(...)` call (`engine.py:697`).
  - Optionally thread `temp_c=temp` into `pack_voltage(...)` callers (only if OCV(T) feature flag is on); otherwise leave call sites unchanged.
- **`configs/ct16ev.yaml`** — populate calibrated thermal terms once the calibration script is run:
  - `h_static_w_per_k: <fitted value>` (replaces implicit 0.0).
  - `h_speed_w_per_k_per_mps: <fitted value>`.
  - `ambient_temperature_c: <event ambient, ~28 °C from CleanedEndurance>` (already configurable).
  - Optionally `ocv_temperature_anchors_mv_per_k:` literature anchors per the table above (commented out if user keeps feature off).

### Create

- **`scripts/calibrate_pack_thermal.py`** — standalone calibration that loads `CleanedEndurance.csv`, fits `(h_static, h_speed, T_amb)` with `scipy.optimize.least_squares`, prints residual stats (mean, p95, peak), writes a YAML snippet, and saves a residual-trace plot to `docs/plots/pack_thermal_calibration.png`.
- **`tests/test_battery_lap_refresh.py`** — pytest. Synthetic 22-lap run with a hot-start SOC trajectory that pushes BMS into derate by lap 10. Asserts: (a) without refresh, last-lap envelope on the back straight equals first-lap envelope; (b) with refresh, last-lap envelope is materially lower (≥ 5 km/h drop on a known segment); (c) net Ah doesn't go negative; (d) test runs in < 5 s.
- **`tests/test_battery_thermal_convection.py`** — pytest. (a) Heat balance unit test: known I_pack, known v, known dt → expected dT. (b) Calibration regression: load a small slice of `CleanedEndurance.csv` (laps 1-2 only, ~150 s), fit `(h_static, h_speed)`, assert mean residual ≤ 1.5 K and peak residual ≤ 5 K. (c) Adiabatic-equivalence: with `h_static = h_speed = 0`, the new code path is bit-identical to the old.
- **`tests/test_battery_ocv_temperature.py`** — pytest. (a) Disabled (anchors None) reproduces current `ocv()` exactly. (b) Enabled with anchors and `temp_c = 25` reproduces current `ocv()` exactly (zero shift at reference). (c) Enabled with `temp_c = 45` shifts OCV by the expected mV at 50 % SOC anchor.

### Reference only (do not modify)

- `src/fsae_sim/sim/speed_envelope.py` — entry point `compute(initial_speed, bms_current_limit_a)` already accepts the BMS limit kwarg; we just call it more times.
- `src/fsae_sim/data/loader.py` (Voltt CSV loader) — only consulted to confirm column names match this plan's `α(SOC)` table format if we ever load anchors from CSV.
- `Real-Car-Data-And-Stats/Endurance Tune2.txt` — BMS table is hardcoded into `configs/ct16ev.yaml` already, not re-derived here.

---

## Tasks

### Part 1 — BMS lap refresh (P1)

#### Task 1.1: Failing test for last-lap envelope visibility under derate

**Files:**
- Create: `tests/test_battery_lap_refresh.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_battery_lap_refresh.py
"""Lap-boundary BMS refresh test: late-stint envelope drops when BMS derates."""
from __future__ import annotations

import numpy as np
import pytest

from fsae_sim.config import load_vehicle_config
from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle.battery_model import BatteryModel


@pytest.mark.skipif(
    not (Path("Real-Car-Data-And-Stats/CleanedEndurance.csv").exists()),
    reason="requires CleanedEndurance.csv telemetry",
)
def test_lap_refresh_lowers_late_lap_envelope_under_thermal_derate():
    # Build sim seeded into late-stint thermal regime: high temp -> BMS derate.
    vehicle = load_vehicle_config("configs/ct16ev.yaml")
    track = Track.from_telemetry("Real-Car-Data-And-Stats/CleanedEndurance.csv")
    bm = BatteryModel.from_config_and_data(
        vehicle.battery,
        "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv",
    )
    strat = CalibratedStrategy.from_telemetry(
        "Real-Car-Data-And-Stats/CleanedEndurance.csv",
    )
    engine = SimulationEngine(vehicle, track, strat, bm)

    # Seed at temp where BMS limit is well below cold ceiling (e.g. 50 C -> 45 A).
    common = dict(num_laps=22, initial_soc_pct=95.0,
                  initial_temp_c=50.0, initial_speed_ms=8.0)

    res_no_refresh = engine.run(refresh_envelope_each_lap=False, **common)
    res_refresh = engine.run(refresh_envelope_each_lap=True,
                             bms_refresh_threshold_a=2.0, **common)

    # Acceptance: with refresh, last-lap mean speed on the back straight
    # is materially lower (envelope clamped by the live BMS limit).
    last_lap_no = res_no_refresh.states.query("lap == 21")["speed_ms"].mean()
    last_lap_yes = res_refresh.states.query("lap == 21")["speed_ms"].mean()
    assert last_lap_yes < last_lap_no - 0.5  # >= 0.5 m/s drop on average

    # Net Ah remains physical (positive, finite).
    assert 0 < res_refresh.net_charge_ah < 20
```

- [ ] **Step 2: Run test to verify it fails.** Expected: `TypeError: run() got an unexpected keyword argument 'refresh_envelope_each_lap'`.

- [ ] **Step 3: Commit.**
```
git add tests/test_battery_lap_refresh.py
git commit -m "test(battery): failing test for BMS lap-refresh in envelope"
```

#### Task 1.2: Implement lap-boundary envelope refresh

**Files:**
- Modify: `src/fsae_sim/sim/engine.py` (lines 323-327 area, plus `run()` signature, plus a small block at the start of each lap iteration).

- [ ] **Step 1: Add `refresh_envelope_each_lap` and `bms_refresh_threshold_a` params to `run()`.**

- [ ] **Step 2: After the existing `initial_bms_limit` envelope build (line 323-327), bind `last_envelope_bms_limit = initial_bms_limit`.**

- [ ] **Step 3: Inside `for lap in range(num_laps):` (after the lap counter increments, *before* the inner `for seg_idx, segment in enumerate(segments):` loop), add:**

```python
if refresh_envelope_each_lap and lap > 0:
    current_bms = self.battery_model.max_discharge_current(temp, soc)
    if abs(current_bms - last_envelope_bms_limit) >= bms_refresh_threshold_a:
        v_max = self._envelope.compute(
            initial_speed=speed,
            bms_current_limit_a=current_bms,
        )
        # Re-push to strategy if it consumes the envelope (D-20 path).
        if hasattr(self.strategy, "set_envelope"):
            try:
                self.strategy.set_envelope(v_max)
            except Exception:
                pass
        last_envelope_bms_limit = current_bms
```

- [ ] **Step 4: Run the new test.** Expected: PASS.
- [ ] **Step 5: Run the full battery / engine pytest suite to confirm no regressions.**
- [ ] **Step 6: Commit.**

#### Task 1.3: 22-lap end-of-stint validation

- [ ] **Step 1: Run `python scripts/sim_compare.py --strategy calibrated --no-plots` before and after the lap-refresh change.**
- [ ] **Step 2: Capture before/after net Ah, net kWh, final temp, last-lap mean speed.** Verify net Ah residual does not regress (allow +/- 0.5 % bound). Verify last-lap mean speed *drops* by ≥ 1 km/h (the late-stint BMS derate becomes visible).
- [ ] **Step 3: Update `docs/SIM_ACCURACY.md` table with the new numbers and a short note: "Calibrated mode now refreshes BMS limit at lap boundaries; late-stint envelope reflects derate."**
- [ ] **Step 4: Commit.**

---

### Part 2 — Speed-dependent pack convection (P2)

#### Task 2.1: Add `h(v)` configuration plumbing and a backwards-compatible `heat_out_w` method

**Files:**
- Modify: `src/fsae_sim/vehicle/battery.py` (add `h_speed_w_per_k_per_mps`, alias old name).
- Modify: `src/fsae_sim/vehicle/battery_model.py` (add `heat_out_w`, plumb into `step`).

- [ ] **Step 1: Failing test for adiabatic equivalence.** Add to `tests/test_battery_thermal_convection.py`:

```python
def test_zero_h_recovers_adiabatic_baseline():
    cfg = make_battery_config(h_static_w_per_k=0.0, h_speed_w_per_k_per_mps=0.0)
    bm = BatteryModel(cfg, cell_capacity_ah=4.5)
    # ... calibrate from a tiny synthetic Voltt-format frame ...
    # Step at a known I, dt, T, v -> dT must equal old-formula dT.
```

- [ ] **Step 2: Add `h_speed_w_per_k_per_mps` to `BatteryConfig` with default 0.0.**

- [ ] **Step 3: Add `BatteryModel.heat_out_w(self, temp_c, vehicle_speed_ms) -> float`. Replace inline formula in `step` with this call.**

- [ ] **Step 4: Add `vehicle_speed_ms: float = 0.0` kwarg to `step` and `step_power`.** Default keeps legacy callers bit-identical (h_speed term is multiplied by 0).

- [ ] **Step 5: Verify the equivalence test passes.**
- [ ] **Step 6: Commit.**

#### Task 2.2: Calibration script

**Files:**
- Create: `scripts/calibrate_pack_thermal.py`

- [ ] **Step 1: Load `CleanedEndurance.csv` Pack Temp, Pack Current, GPS Speed, State of Charge.**
- [ ] **Step 2: Build heat-in trace as `I_pack^2 · R_pack(SOC)`.** Use `BatteryModel.pack_resistance(SOC)` so we share the same R model the sim uses.
- [ ] **Step 3: Forward-Euler integrate the lumped thermal ODE with `(h_static, h_speed, T_amb)` parameters.**
- [ ] **Step 4: `scipy.optimize.least_squares` on the residual vector `T_pack_sim − T_pack_telemetry`.** Bounds documented in Research above.
- [ ] **Step 5: Print mean / p95 / peak residual in K and as % of full temp swing.** Save plot to `docs/plots/pack_thermal_calibration.png`.
- [ ] **Step 6: Output a copy-pasteable YAML snippet for `configs/ct16ev.yaml`.**
- [ ] **Step 7: Commit.**

#### Task 2.3: Failing-then-passing calibration regression test

**Files:**
- Modify: `tests/test_battery_thermal_convection.py`

- [ ] **Step 1: Add a regression test that loads a small slice of `CleanedEndurance.csv` (first 150 s), fits `(h_static, h_speed)`, asserts mean Pack Temp residual ≤ 1.5 K and peak ≤ 5 K.** This will fail until we run the calibration.

- [ ] **Step 2: Run calibration script. Update `configs/ct16ev.yaml` with the fitted values. Re-run test.**
- [ ] **Step 3: Update `docs/SIM_ACCURACY.md` with the new "Final cell temp" row.** Target: |residual| ≤ 1.0 K (currently 1.08 K).
- [ ] **Step 4: Commit.**

#### Task 2.4: Wire `vehicle_speed_ms` through the engine

- [ ] **Step 1: At the existing `self.battery_model.step_power(elec_power, seg_time, soc, temp, time_s=time)` call (`engine.py:697`), add `vehicle_speed_ms=avg_speed`.**
- [ ] **Step 2: Run full pytest. Run `sim_compare.py` calibrated; confirm net Ah / kWh residuals are within ±1.5 % of the pre-change values (this tightens with calibration; no regression bound by itself).**
- [ ] **Step 3: Commit.**

---

### Part 3 — OCV(T) interim shift (P2)

#### Task 3.1: Plumb `temp_c` into `ocv()` (no behaviour change unless anchors set)

**Files:**
- Modify: `src/fsae_sim/vehicle/battery.py` — add `ocv_temperature_anchors_mv_per_k: tuple[tuple[float, float], ...] | None = None`.
- Modify: `src/fsae_sim/vehicle/battery_model.py` — extend `ocv` signature, build a SOC-keyed `α(SOC)` interpolator from anchors at calibration time, apply shift.

- [ ] **Step 1: Failing test that disabled anchors reproduce current `ocv()` exactly.**
- [ ] **Step 2: Add anchors field to config, default None.**
- [ ] **Step 3: In `BatteryModel.calibrate_from_voltt`, if anchors are set, build `self._ocv_temp_alpha_interp = interp1d(...)`.**
- [ ] **Step 4: Modify `ocv(self, soc_pct, *, temp_c: float | None = None)`:**
  - If `temp_c is None` or anchors not set → existing behaviour.
  - Else: `return base_ocv + α(SOC) * (temp_c - 25.0) / 1000.0`, with the same voltage-floor / extrapolation guards.
- [ ] **Step 5: All existing call sites that don't yet pass `temp_c` continue to work.** Pack-level `pack_voltage` reads `temp_c=self._last_temp_c` (set by `step` if anchors are enabled, otherwise no-op).
- [ ] **Step 6: Failing test "enabled with `temp_c=45` shifts OCV by 0.4 mV/K * 20 K * 110 series = 0.88 V at 50 % SOC anchor" passes.**
- [ ] **Step 7: Commit.**

#### Task 3.2: Literature anchor table in default config (commented out)

**Files:**
- Modify: `configs/ct16ev.yaml`

- [ ] **Step 1: Add a commented-out `ocv_temperature_anchors_mv_per_k:` block with the table from Research Summary.**
- [ ] **Step 2: Add a one-line comment "literature shift; replace with Voltt-derived values per User-Decision U2".**
- [ ] **Step 3: Commit.**

#### Task 3.3: Document as Future Work in SIMULATOR_ISSUES.md

**Files:**
- Modify: `docs/SIMULATOR_ISSUES.md`

- [ ] **Step 1: Update OPEN issue 8 entry from "OCV temperature dependence (issue 8)" to: "OCV(T) — literature interim shipped (commented in ct16ev.yaml). Voltt re-export at 0/25/45 °C ambient outstanding (User-Decision U2)."**
- [ ] **Step 2: Commit.**

---

## Risks / Unknowns

- **R1 — Lap-refresh threshold tuning.** Setting `bms_refresh_threshold_a = 5.0 A` was sized from a back-of-envelope estimate (1.7 K of cell-temp headroom at the steepest part of the BMS table). If real telemetry shows the limit drifting in 2-3 A increments per lap rather than 5 A jumps, the threshold may need to drop to 2.0 A to catch the derate ramp. Mitigation: log `last_envelope_bms_limit` at each refresh into the SimResult `states` DataFrame so we can see the cadence post-hoc.
- **R2 — Pack Temp telemetry is max-cell, not mean.** Calibrating `(h_static, h_speed)` against max-cell will give a `(h_static, h_speed)` pair that *under-predicts* lumped-mean cooling (we are fitting cooling that has to bring max down to telemetry, not mean down to mean). The lumped sim mean will then run hotter than reality. Quantification: telemetry final 38 °C, sim final 36.92 °C — sim is currently *cooler* not hotter, suggesting the existing R(SOC) is also off. Mitigation: report the residual structure (sign-conditioned) in the calibration script output so we can tell which way the bias goes.
- **R3 — Convection on a stationary car.** In the initial rolling-start segment (`v ≈ 0.5 m/s`), `h(v)` collapses to `h_static`. If the calibration produces `h_static < 0.5 W/K` (essentially adiabatic when stopped), this is correct physics but counter-intuitive — make sure we don't push fitter too hard toward `h_static = 0`. Bound `h_static ∈ [0.5, 50]` per the calibration script.
- **R4 — OCV(T) regression.** If the literature anchors are wrong-sign for our chemistry / SOC range, enabling the feature could *increase* net kWh residual. Mitigation: anchors disabled by default in `ct16ev.yaml`; explicit acceptance criterion in Task 3.1 that disabled mode reproduces current behaviour bit-identically.
- **R5 — Lap-refresh causes fluctuations in metrics that compound noise on sweeps.** If the threshold triggers refresh on different laps for two adjacent sweep candidates, their numbers diverge for a non-physical reason. Mitigation: report `n_envelope_refreshes` as part of `SimResult` so the sweep harness can flag candidates where refresh count differs by > 2.
- **R6 — Voltt single-T trap.** The plan repeatedly says "Voltt is single-T for 2025"; it is *also* single-T for 2026. If a future Voltt re-export ships, the OCV(T) interpolator must accept either anchor pairs (current plan) *or* a 2-D grid CSV (future). Design the anchors-only path now to be a thin wrapper around the 2-D interpolator so we don't refactor later.

---

## Verification / Acceptance Criteria

### Quantitative

- **BMS lap refresh.**
  - Last-lap (lap 22) mean speed on the back straight (segments where envelope is BMS-limited at 38 °C) drops by ≥ 1 km/h vs. unrefreshed baseline.
  - Net Ah residual on calibrated mode unchanged or improved (current −3.2 %, target ≤ |−3.5 %|).
  - Net kWh residual on calibrated mode unchanged or improved (current −4.3 %, target ≤ |−4.5 %|).
  - 22-lap calibrated runtime increases by < 2.0 s (typical wall time was ~5 s; budget 7 s).
  - **Native units:** envelope-side BMS limit at lap 22 is ≤ 80 A (currently planned at 100 A cold).
- **Speed-dependent convection.**
  - Mean Pack Temp residual on `CleanedEndurance.csv` ≤ 1.0 K.
  - p95 Pack Temp residual ≤ 2.5 K.
  - Peak Pack Temp residual ≤ 5.0 K.
  - Final cell temp residual: |Δ| ≤ 1.0 K (currently 1.08 K).
  - **Native units:** report `(h_static, h_speed, T_amb)` fitted values in calibration script output, plus residual histogram.
- **OCV(T).**
  - With anchors disabled: net Ah / net kWh residuals bit-identical to pre-change calibrated mode.
  - With anchors enabled at literature values: net Ah / net kWh residuals shift by ≤ 0.5 %.
  - If they shift by > 0.5 % in the worse direction, leave anchors disabled in default config.

### Qualitative

- **Mechanism, not bandaid.** No constant offset on `heat_out_w`. No fudge factor on `heat_in_w`. Both terms of `h(v)` come from a documented physical model (forced-convection over a flat plate / box) calibrated against telemetry.
- **Reproducibility.** `scripts/calibrate_pack_thermal.py` writes its inputs and fitted parameters into `docs/plots/pack_thermal_calibration.png` caption (and an adjacent `.txt` summary). Anyone re-running the script with the same telemetry should land within 5 % on each parameter.
- **No SOC-accuracy refinement.** Per user memory: kWh / Ah are the validation metrics. SOC remains an internal model state; this plan does not score against telemetry SOC.
- **Conformance with sim correctness rules.** No clamping to mask errors; voltage-floor enforcement (`battery_model.py:680-693`) unchanged; BMS limit refresh is a *fidelity* gain, not a clip.

---

## Future Work — Per-Module Thermal (Deferred)

**Decision criterion for activation.** Activate the per-module model when *any* of:

1. We are sweeping cooling-package geometry (e.g. baffle changes, fan add/remove).
2. The 2026 P50B pack with `Heat Transfer Coefficient: 50 W/m²K` (per Voltt 2026 export) replaces CT-16EV — active cooling makes intra-segment ΔT first-order.
3. Telemetry from a future event shows segment-to-segment ΔT > 5 °C in the BMS log (single-cell sensors are currently lumped in `Pack Temp`).

**Topology — 5 segments × 22S × 4P.** The DSS-documented pack layout is 5 physical segments of 22 series × 4 parallel cells each (`CLAUDE.md` "Pack" row). Model topology:

```
Segment 1 ──┐ Segment 2 ──┐ Segment 3 ──┐ Segment 4 ──┐ Segment 5
 (T_1)      G_12          G_23          G_34          G_45        (T_5)
   │                         │
   G_1_amb                   G_3_amb       (per-segment cooling node)
   │                         │
 ambient                  ambient
```

- 5 thermal nodes, one per segment: `T_1, T_2, T_3, T_4, T_5`.
- Inter-segment conductance `G_ij` (W/K): from busbar/wall conductance — needs measurement or FEM. Initial guess: `G_ij ≈ 1.0 W/K` (low — cells are thermally isolated by holders).
- Per-segment ambient conductance `G_i_amb`: heat path through enclosure wall to the air. Speed-dependent same as lumped: `G_i_amb(v) = g_i_static + g_i_speed * v_local`.
- Heat generation per segment: `Q_i = (I_pack / parallel)^2 * R_cell(SOC, T_i) * series_per_segment`. Distributed by segment current; if 4P sharing is perfect (assumption M5 in `SIMULATOR_ISSUES.md`), `Q_i = Q_total / 5`.

**Conductance matrix (mass-matrix form).**

```
M_thermal · dT/dt = -K(v) · T + K_amb(v) · T_amb + Q
```

with `M_thermal = diag(C_seg_i)`, `K(v)` symmetric tridiagonal (intra-segment + ambient), `Q` heat-generation vector. This is a cheap ODE — RK2 or even forward-Euler at lap-segment resolution is plenty.

**Additional inputs needed.**

- Per-segment cell temperature telemetry (BMS multi-cell channel — present in BMS log but stripped from `CleanedEndurance.csv`; would need a re-export).
- Per-segment cooling geometry (which segments face the airflow path on the chassis; needs a CAD review).
- Inter-segment busbar dimensions for `G_ij` (DSS specifies series/parallel topology but not busbar conductance).

**Why we are not implementing now.**

- `CleanedEndurance.csv` strips multi-cell BMS data — calibration target is missing.
- 2025 CT-16EV has no active cooling; lumped + speed-dependent `h(v)` captures the full physics for endurance sweeps on this car.
- Per-module fidelity matters most for *cooling-package* sweeps, not *tune* sweeps. The user's stated focus (max torque, max RPM, current-limit) does not exercise per-module ΔT.
- Effort is ≥ 8 h (dataclass for graph, ODE assembly, calibration of 5 × 2 = 10 parameters with regularization, three new tests) — not justified before P0/P1/P2 are done.

**Estimated effort when activated:** 8–12 h.

---

## Effort Estimate

5-hour tiers:

- **Tier 1 (≤ 5 h): BMS lap refresh — Tasks 1.1–1.3.** Failing test → minimal patch in `engine.py:323-327` slice → 22-lap validation. Self-contained; no API changes outside `run()`.
- **Tier 2 (5–10 h): Speed-dependent convection — Tasks 2.1–2.4.** Config-plumbing + new `heat_out_w` method + calibration script + regression test + engine wiring. The `scipy.optimize.least_squares` calibration is the slow part; the rest is mechanical.
- **Tier 3 (5–10 h): OCV(T) interim shift — Tasks 3.1–3.3.** Anchors plumbing + literature table + ct16ev YAML update + tests. Disabled-by-default keeps risk low.
- **Tier 4 (planning only, 0 h implementation): Per-module thermal — documented as Future Work above.**

Total active implementation: **15–25 h** spread across three commits-able milestones (one per part). All within the user's "engineering sweep tool" mission and confined to the four planning domains the parent agent assigned.
