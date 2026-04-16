# Driver Model Fix Campaign — Design

**Date.** 2026-04-16
**Author.** Brandon + Claude
**Source catalog.** `docs/DRIVER_MODEL_ISSUES_2026-04-16.md` (28 issues: 6 Critical, 11 Significant, 6 Moderate, 5 Minor)

---

## 1. Goal

Close every driver-model-layer issue in the 2026-04-16 catalog so that the Verification, Visualization, and Simulate webapp pages can trust the driver model. The driver model must remain adaptive (not just replay) because the Simulate page changes the car's performance envelope, which a pure-replay driver cannot respond to.

## 2. Scope

- **In.** All 28 issues in the catalog (D-01 through D-28), including items flagged `[sweeps-only]`. Under Scope A, even sweeps-only fixes are worth doing because `CalibratedStrategy` and `PedalProfileStrategy` are load-bearing for the Simulate page, which perturbs the car envelope and needs a driver that reacts.
- **Not deleted.** `CalibratedStrategy`, `PedalProfileStrategy`, `DriverParams` stay in the repo. They will be cleaned up but not removed.
- **Out.** Webapp page changes beyond the driver-channel validation harness (D-23). Documentation rewrites outside of `docs/REMAINING_ISSUES.md` updates and a post-mortem. Physics issues not listed in the driver-model catalog.

## 3. Key Architectural Decisions

### 3.1 Segment bin size (D-14)

Commit the uncommitted `5.0 m → 0.5 m` change on `src/fsae_sim/track/track.py` together with the smoothing-window rescale (fixed 25 m physical distance, odd window). Re-run `scripts/validate_tier3.py` immediately and record the new baseline. Every subsequent fix is measured against that baseline. The old "8/8 metrics pass" claim at 5.0 m is retired.

### 3.2 Electrical model (D-17)

Delete `PowertrainModel.coast_electrical_power`. Extend `PowertrainModel.electrical_power(motor_torque, rpm, pack_voltage)` with a back-EMF rectifier term:

- When `K_e · ω_electrical > V_pack` (free-wheeling at high RPM, zero commanded torque), motor acts as a generator and current flows into the pack through the inverter body diodes.
- `K_e` is fit from the EMRAX 228 MV LC datasheet (or DSS, or inferred from Voltt data as a fallback) — **not** fit to Michigan telemetry's `-456 W @ 2299 RPM` point. That point is a *validation target*, not a calibration target.
- Engine dispatch is by physical state (motor torque magnitude), not `cmd.action`. A `COAST` command with zero commanded torque and an `ACCEL` command with 0 Nm torque produce the same electrical power through the same code path.

### 3.3 Replay electrical power (D-13)

Restore the `if is_replay and self.strategy.has_electrical_power: elec_power = self.strategy.measured_electrical_power(...)` branch in `src/fsae_sim/sim/engine.py`. Rationale: V×I at pack terminals is the definitional ground truth for energy; using CAN torque feedback instead introduces the documented ~10% positive bias. The ~10% bias is a separate issue (CAN torque estimation artifact) that does not need to be solved to make replay faithful. Document the asymmetry in a code comment so it is not flagged as a bug later.

### 3.4 Driver-channel validation gate (D-23)

Build this early — as item 3 in the campaign, before any driver logic change lands. Without it, there is no regression test that catches driver-model divergence from telemetry. Harness compares sim per-sample throttle %, brake %, and action classification against telemetry on a held-out lap subset (stint 2, laps 13-21). Outputs RMSE, R², Pearson correlation per channel. Integrates with existing `validate_full_endurance`.

### 3.5 Train/test split (D-04)

- **Battery:** pack R and capacity fit from Voltt simulation data only (`About-Energy-Volt-Simulations/`). `calibrate_pack_from_telemetry` either deleted or made opt-in with a loud warning and excluded from default validation.
- **Driver:** calibration uses laps 1-10 of Michigan 2025 endurance. Laps 13-21 (stint 2) are held out for validation. Driver-change lap (the one spanning 10-13) excluded from both.
- Validation report identifies which parameters were fit from which source; any metric whose parameters were fit to telemetry is marked as a consistency check, not a predictive claim.

### 3.6 Interface unification (D-28)

If a clean merge of `DriverParams` (scalar multipliers for `PedalProfileStrategy`) and `with_zone_override` (zone-keyed action/intensity override for `CalibratedStrategy`) emerges during implementation, unify. If not, leave both in place with a deprecation note identifying which to use for which strategy. Not a correctness bug — design debt. This is the only item we commit to possibly punting on.

### 3.7 ControlCommand extension (D-09, D-27)

Add `metadata: dict[str, float] | None = None` to `ControlCommand`. First consumer: `metadata["max_speed_ms"]` for zone speed caps. Future consumers (if needed) piggyback without further dataclass surgery.

## 4. Fix Order

Revised from the catalog's recommended order to front-load (a) the baseline re-establishment and (b) the validation gate.

| # | Item | Reason for position |
|---|------|----|
| 1 | D-14 | Establish new baseline first; everything downstream measured against it |
| 2 | D-05 | Regen accounting; prerequisite for "net energy" meaning anything |
| 3 | D-23 | Driver-channel validation harness; gate for all remaining items |
| 4 | D-06 | Restore `ReplayStrategy` torque clip range; fixes Replay baseline |
| 5 | D-13 | Restore V×I branch for replay electrical power |
| 6 | D-17 | Back-EMF rectifier model; delete `coast_electrical_power`; route by motor state |
| 7 | D-01 | Classify on torque fraction, not raw pedal % |
| 8 | D-07 | Rescale per-lap distance to `track.total_distance_m` |
| 9 | D-04 | Train/test split |
| 10 | D-02 | Drop two-pass intensity overwrite in `CalibratedStrategy.__init__` |
| 11 | D-03 | Delete dead `coast_throttle` (back-EMF model obsoletes it) |
| 12 | D-15 | Unify field-weakening (one model, applied once) |
| 13 | D-16 | Derive motor-torque-for-power from clipped wheel force under regen |
| 14 | D-22 | Fix `regen_force` eta direction + derating |
| 15 | D-10 | Fix `max_speed_ms` to peak, not mean |
| 16 | D-27 | Add `metadata` field to `ControlCommand` |
| 17 | D-09 | Wire `max_speed_ms` through `metadata`; engine honors zone cap |
| 18 | D-11 | Populate `SimState.pack_current` with last loop iteration value |
| 19 | D-08 | Normalize brake to DSS max, not data-percentile |
| 20 | D-18 | Filter calibration laps by distance AND time AND mean_speed |
| 21 | D-19 | Plumb column-name kwargs through `CalibratedStrategy.from_telemetry` |
| 22 | D-20 | `CoastOnly`/`ThresholdBraking` honor `SpeedEnvelope` or are re-documented |
| 23 | D-24 | Unify `ReplayStrategy` thresholds with calibration thresholds |
| 24 | D-25 | `_segment_to_zone` O(1) index array |
| 25 | D-26 | Warn on `initial_speed_ms` clamp; add `rolling_start=True` flag |
| 26 | D-12 + D-21 | Fix `compare_driver_stints` to actually compare two stints, zone-aligned |
| 27 | D-28 | Unify `DriverParams` + `with_zone_override` if clean merge exists |

## 5. Testing Plan

- **Items 1-2** regression-tested against the existing 8-metric `validate_tier3.py` at the new 0.5 m baseline.
- **Item 3** is the new validation harness itself. Unit tests verify it correctly detects fabricated divergences (corrupt throttle trace → RMSE > threshold). Current-state numbers recorded as the pre-fix baseline before any driver logic changes.
- **Items 4-27** each get: (a) a unit test written first (TDD) against the specific function changed, and (b) a re-run of D-23 and `validate_tier3.py` afterward. Any regression blocks the commit.
- **Existing tests** (`tests/test_calibrated_strategy.py`, `tests/test_pedal_profile_strategy.py`, `tests/test_dynamics.py`) kept green throughout. If a fix legitimately changes an expected value, update the test and explain in the commit message.

## 6. Commit Protocol

- Work directly on `main` (solo development; no PR review gate needed).
- One commit per D-XX item. Commit message format: `fix(driver): D-XX — <short description>`.
- Body includes: what changed, what test verifies it, any baseline-number delta.
- Run `pytest` + `validate_tier3.py` before every commit. Failure blocks the commit.
- If a fix changes a validation metric, record the before/after number in the commit body.

## 7. Risks and Open Questions

1. **Re-baselining at 0.5 m (D-14) may reveal previously-hidden physics errors.** If the new baseline shows fewer than 8/8 metrics passing, pause and decide whether to fix inline or catalogue and proceed.
2. **EMRAX 228 MV LC `K_e` fit (D-17).** If the DSS / datasheet does not give a usable value, fit from Voltt data (coast power vs RPM at known pack voltage). Worst case: fit from the -456 W data point (defeats half the rigor but unblocks the campaign). Flag if hit.
3. **Replay electrical asymmetry (D-13).** Replay uses V×I; non-replay uses modeled power. Acceptable but must be documented in-code.
4. **D-28 interface unification.** Underspecified in the catalog. If no clean merge emerges, punt with a deprecation note. Only item we commit to possibly not fully solving.
5. **Scope creep temptation.** Several items touch physics outside the driver layer (D-15, D-16, D-22 in `powertrain_model.py`; D-17 too). If a fix reveals a related physics bug not in the driver catalog, we catalog it in `docs/REMAINING_ISSUES.md` and keep going — do not expand scope mid-campaign.

## 8. Out of Scope

- Webapp page surface updates beyond exposing D-23 results on the Verification page (can come later in a separate pass).
- Documentation rewrites except updates to `docs/REMAINING_ISSUES.md` and a post-mortem doc at the end of the campaign.
- Physics issues not listed in `docs/DRIVER_MODEL_ISSUES_2026-04-16.md`.
- Sweep infrastructure. Still out of this repo per the webapp refocus.

## 9. Definition of Done

- All 28 items have landed commits on `main` (with the allowed partial-punt on D-28).
- `pytest` is green.
- `validate_tier3.py` passes at the new 0.5 m baseline (target: at least as many metrics pass as the pre-fix baseline; ideally more).
- Driver-channel validation harness (D-23) records RMSE / R² / correlation for throttle, brake, action per sample on the held-out lap subset. Numbers recorded in the commit body of D-23 and again at end of campaign.
- `docs/REMAINING_ISSUES.md` updated to reflect closed driver-model issues.
- Post-mortem doc (`docs/DRIVER_MODEL_FIXES_POSTMORTEM_2026-04-XX.md`) written at end of campaign: before/after metric table, any punted items, any newly-discovered issues catalogued elsewhere.
