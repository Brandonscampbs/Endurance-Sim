# FSAE EV Simulator — Known Issues Tracker

*Authoritative consolidation as of 2026-04-16. Replaces
`REMAINING_ISSUES.md`, `SIMULATOR_AUDIT_2026-04-16.md`,
`SIMULATOR_AUDIT_NEW_FINDINGS_2026-04-16.md`, and
`DRIVER_MODEL_ISSUES_2026-04-16.md`. IDs (C*, S*, M*, m*, NF-*, D-*)
are preserved so existing references still resolve. Detail lives in
git — this file is a pointer index.*

## Status summary

| Status   | Count | Notes |
|----------|-------|-------|
| FIXED    | 71    | Landed in R2 merge wave + driver campaign (2026-04-16) |
| PARTIAL  | 2     | #3 regen double-count, #18 tire radius |
| DEFERRED | 28    | Intentionally skipped this session — see Deferred |
| OPEN     | ~50   | Moderate/Minor buckets not yet touched |
| REFUTED  | 2     | #21, NF-38 |

Legend: REMAINING_ISSUES uses bare numbers (1–22). Audit uses C/S/M/m.
New-findings uses NF-*. Driver doc uses D-*.

---

## FIXED

### Data loading & encoding
- **NF-1** `data/loader.py` — Latin-1 decoding for AiM CSV. [b58b78e]
- **NF-2** `scripts/clean_endurance_data.py` — Latin-1 decoding; drop blanket float dtype. [99b1b1e]
- **NF-12** `vehicle/vehicle.py` — explicit UTF-8 YAML read; wrap TypeError with file context. [4685ac2]
- **NF-27** `data/loader.py` — required-column validation in `load_cleaned_csv`. [b58b78e]
- **NF-28** `data/loader.py` — encoding + schema + empty-file check in `load_voltt_csv`. [b58b78e]
- Loader test coverage added. [1cf5676]

### Tire model
- **6** Longitudinal PAC2002 model (R25B transplant). [pre-session: ddf2968]
- **C4** / **M11** `tire_model.py` — closed-form `|μ·Fz|` peaks; replaces `minimize_scalar`. [d30b707, a7aabfd]
- **NF-4** `.tir` parser — warn/assert on swallowed non-numeric values. [d30b707]
- **NF-18** `tire_model.py:222` — PKY2 sign-flip fix in cornering-stiffness atan. [d30b707]
- **NF-35** `tire_model.py:265, 353` — symmetric `ey` clamp. [d30b707]

### Vehicle dynamics / cornering / load transfer
- **1** Roll angle moment-arm fix. [pre-session: 4ae070a]
- **11** Cornering drag — Pacejka-per-tire with fallback. [pre-session]
- **C1** `_cornering_drag_pacejka` rewritten off power-law fudge. [29a9b53]
- **C5** `load_transfer.py` — redistribute lifted-tire loads to same axle; vertical equilibrium preserved. [5435702]
- **C6** `cornering_solver.py` — includes longitudinal demand in lateral solve. [29a9b53, 7fbbb3e]
- **C9** `dynamics.py` — single speed floor; `seg_time × avg_speed = length` honored. [29a9b53]
- **NF-6** `dynamics.py` — `max_traction_force` / `max_braking_force` fixed-point iterate, drops 0.3g/-1.0g magic. [29a9b53]
- **NF-19** `dynamics.py` — grade force uses `m_effective` consistently. [29a9b53]
- **NF-33** `cornering_solver.py` — remove `min(…, 0.99)` friction-ellipse cap. [5ad7e11]
- **NF-34** `cornering_solver.py` — convergence-gated bisection (not fixed 30 iters). [5ad7e11]
- **NF-36** `dynamics.py` — legacy `max_cornering_speed` denominator sign guard. [29a9b53]
- **NF-40** `load_transfer.py:158` — front/rear track zero-guard + validation. [5435702]

### Battery
- **2** BMS current limit feeds back through `lvcu_torque_ceiling()`. [pre-session]
- **C15** Battery calibration from Voltt only; AiM stint 2 held out. [cae072d]
- **NF-9** Drop hardcoded `cell_capacity_ah=4.5` from tests/scripts; CT-17EV config fixed. [9391ab4, cae072d]
- **NF-10** `validation.py` — align sim lumped vs telemetry max-cell temp comparison. [cae072d]
- **NF-16** `battery_model.py:313` — OCV extrapolation floor + warn (no silent throttle truncation). [cae072d]
- **NF-26** `calibrate_pack_from_telemetry` returns new model (no in-place mutation). [cae072d]
- **S15** Battery thermal mass accounts for busbars + enclosure. [cae072d]
- **S16** Voltage floor no longer silently clamped. [cae072d]
- **S17** Pack resistance SOC/T dependence added. [cae072d]

### Powertrain
- **C2** Back-EMF rectifier coast model replaces `coast_electrical_power` telemetry fit. [f2804ea, 3cec553]
- **C3** Regen electrical path no longer double-counts inverter losses. [3cec553, 978e9c5]
- **C17** `torque_limit_lvcu_nm` 150 → 220 Nm (firmware). [caf73d8]
- **NF-31** CT-17EV `lvcu_power_constant` scaled for 100S pack. [65d4c90]
- **NF-41** `powertrain_model.py` pedal dead-zone zero-width guard. [978e9c5]
- **S12** `regen_force` — `F / η_gearbox` sign correction for generator mode. [978e9c5]
- **S13** BSE / APPS / startup gate modeled. [978e9c5]
- **S14** BMS current limit -3 A safety offset added. [978e9c5]

### Simulation engine / speed envelope
- **C7** `speed_envelope` passes curvature to `total_resistance`. [27e970c]
- **C8** Dual energy counters (discharge + regen); applied to sim and validation symmetrically. [2db92b8, f290e3b]
- **NF-3** `engine.py` — motor-map path resolved, warn on fallback. [2db92b8]
- **NF-5** `engine.py` BRAKE branch — brake torque from clipped regen for consistency. [2db92b8]

### Driver strategies
- **C13** `CalibratedStrategy` zone intensity no longer overwritten by raw per-segment values. [8dc372f]
- **NF-13** `_detect_lap_boundaries_safe` — narrow exceptions, no blanket swallow. [9d40a7e]
- **NF-25** `telemetry_analysis.py:329` — NaN guard before median/intensity. [9d40a7e]
- **NF-30** `strategies.py` / `telemetry_analysis.py` — `brake_pct` unit convention unified. [8dc372f]
- **S9** `PedalProfileStrategy.from_telemetry` — train/test split. [8dc372f]
- **S10** Zone `max_speed_ms` stored as peak, not mean-of-means. [9d40a7e]
- **S18** `ReplayStrategy` clips LVCU torque symmetric; regen routed through electrical path. [8dc372f]
- **D-05** Regen tracked separately in energy accounting. [95d40ff]
- **D-06** `ReplayStrategy` preserves negative LVCU torque. [4b60486]
- **D-13** Replay V×I electrical path restored (reapplied 4d087f9). [e00e2cf, 4d087f9]
- **D-14** Segment bin 5.0 → 0.5 re-baselined. [c5ed66f]
- **D-15** Field-weakening unified on constant-power shape (reapplied 4d087f9). [cf9486e, 4d087f9]
- **D-16** Regen motor torque derived from clipped wheel force. [c3d97e9]
- **D-17** `electrical_power()` back-EMF rectifier model (coast gap logged, not fudged). [f2804ea]
- **D-22** Regen efficiency derate applied in `regen_force` envelope (test side reverted 4d087f9). [d79a56a, 4d087f9]
- **D-23** Driver-channel validation harness added. [788e89b]

### Track
- **C12** Curvature smoother rescaled (no longer erases hairpin peaks). [9ee4506]
- **NF-7** `track/track.py` — low-speed curvature no longer forced to zero. [9ee4506]
- **NF-20** `track/track.py:231` — `floor()` fractional-bin handling. [9ee4506]
- **S19** Start/finish detection — 2D gate, laps outside 500–2000 m no longer dropped. [9ee4506]

### Validation & scoring
- **C8 (val side)** Stint-aware validation; honest energy accounting. [5424ec6, f290e3b]
- **C16** Driver-change pause (210 s) no longer collapsed. [5424ec6]
- **NF-43** `scoring.py` — `total_time_s` includes driver-change time. [f290e3b]
- **NF-44** Driver-change bonus gated by `driver_change_completed` flag. [f290e3b]
- **NF-45** `validation.py:220` — speed-filter scope fixed. [f290e3b]
- **NF-59** `score_sim_result(track_distance_km)` propagated through `score()`. [f290e3b]
- **S11** Efficiency score uses raw time, not cone-corrected. [f290e3b]

### Constants, dead code, hygiene
- **NF-22** `GRAVITY` / `AIR_DENSITY` single source of truth in `physics_constants`. [ae6b597]
- **NF-42** Hot-termination temp pulled from battery config (no `65.0` magic). [ed2632e]
- **NF-55** `analysis/__init__.py` — drop stub functions from `__all__`. [cb9ed3d]
- **NF-56** / **m7** Dead `src/fsae_sim/scoring/` package deleted. [f3dd485]

### Tire model (earlier)
- **15** Motor efficiency 2D map via `MotorEfficiencyMap`. [pre-session]

---

## PARTIAL

- **3** (powertrain) `regen_force` generator-mode used `F×η` instead of `F/η` — generator-side addressed by S12 [978e9c5], but earlier history flagged this as partial; confirm with validation run.
- **18** Tire radius constant updated to 0.2042 m, still not dynamic `loaded_radius()` (~3% under load).

---

## OPEN

### Critical (physics / correctness blockers)
- **C10** `sim/engine.py` — Engine integrates speed with entry-speed forces; no corrector step (Heun). Deferred.
- **C11** Mechanical vs electrical torque use different operating points. Deferred pending integrator.
- **C14** `PedalProfileStrategy` classifier discards torque-based intensity. Deferred (driver-side).

### Significant
- **S1** Regen tire-saturation doesn't feed back to electrical power. Absorbed by C11 when landed.
- **S2** / **S3** Multiple field-weakening models; replay double-counts — partly addressed by D-15 but deeper engine rewrite deferred.
- **S4** `resolve_exit_speed` clamps without charging energy. Revisit once C7/C9 settle.
- **S5** Driver decision sees stale `pack_current = 0` every segment. Engine-arch deferred.
- **S6** Speed envelope computed once, ignores BMS current cap. Deferred.
- **S7** Combined-slip (Pass 4) dead code. Deferred.
- **S8** `ReplayStrategy` V×I path — landed-reverted-relanded; keep eye on regression.

### Moderate
- **4** `battery_model.py:362` — Battery thermal model has no cooling term.
- **5** / superseded by C4 for peak-force; residual scipy optimizer calls in cornering solver may remain.
- **7** Linear field-weakening taper — addressed for replay (D-15); audit physics path post-merge.
- **8** `battery_model.py:257` — Internal resistance temperature-independent.
- **9** `analysis/scoring.py:214` — `EFmin = 0.0` inflates efficiency scores.
- **10** / **D-12** `compare_driver_stints` compares same data to itself.
- **12** Python sim loop limits throughput.
- **13** / **D-09** `CalibratedStrategy.decide()` never returns `max_speed_ms`.
- **14** Effective mass includes drivetrain efficiency on rotor inertia.
- **16** Track curvature from GPS acceleration (noisy at low speed).
- **17** 5 m segment resolution — now 0.5 m default, but nominal config entry stale.
- **19** ISA air density vs Michigan conditions.
- **20** Downforce front distribution is default, not DSS-measured.
- **22** Back-EMF alone doesn't explain coast power (45 Wh/stint gap; mechanisms listed).
- **M1** No anti-squat / anti-dive geometry.
- **M2** Friction ellipse uses peak forces, not combined-slip Pacejka.
- **M3** OCV extrapolated linearly below calibration range.
- **M4** Coulomb counting has no coulombic efficiency.
- **M5** 4P cell current sharing assumed perfect.
- **M6** No OCV hysteresis.
- **M7** Downforce treated inconsistently across resistance functions.
- **M8** Camber sign convention undocumented.
- **M10** Brake distribution load-proportional, not mechanical-bias.
- **M12** PAC2002 Svy missing LKYG on camber term.
- **M13** `m_effective` doesn't distinguish accel vs regen direction.
- **M14** LONGVL speed correction ignored in Fx.
- **M15** `max_traction_force` hardcodes 0.3g load-transfer (check after NF-6).
- **M16** Forward-Euler lag between pack_voltage and pack_current.
- **M17** Regen active at arbitrarily low RPM (no back-EMF cutoff) — may be covered by D-17.
- **M18** Driver-change lap not filtered from default calibration.
- **M19** `from_telemetry` ignores user-provided column names.
- **M20** `CoastOnly` / `ThresholdBraking` ignore forward-propagated envelope.

### Minor
- **m1** `compare_driver_stints` feature gap (duplicate of #10/D-12).
- **m2** Plots use `LFspeed`; metrics use `GPS Speed`.
- **m3** Validation plot auto-scales, hiding bias.
- **m4** Pass/fail thresholds loose (15–20 %).
- **m5** Mean-speed / distance / time are algebraic identities.
- **m6** `speed > 5 km/h` filter no-op on cleaned data.
- **m8** Driver-change bonus gating (duplicate of NF-44 — verify closed).
- **m9** `total_time_s` missing DC overhead (duplicate of NF-43 — verify closed).
- **m10** Empty-bin carry-forward corrupts curvature across lap wrap.
- **m11** Grade is not smoothed (curvature is).
- **m12** `ReplayStrategy.decide` 0.05 thresholds inconsistent with calibration units.
- **m13** Cosmetic: linear-scan `zone_for_segment`, standing-start clamp, `initial_speed` dead API, FW constants Michigan-fit, stale docstring 5 m → 0.5 m, lap-wrap speed clamp, `laps_completed` off-by-one on mid-lap termination.

### Driver-model (open)
- **D-01** / **C14** `PedalProfileStrategy.from_telemetry` classifies on raw pedal %. Deferred.
- **D-02** already closed as C13.
- **D-03** Dead `coast_throttle` parameter in `DriverParams`.
- **D-04** / **C4(val)** Calibration and validation share telemetry. Partly addressed by C15/C16 — audit.
- **D-07** Per-lap distance misalignment at segment sampling.
- **D-08** Brake normalization depends on calibration laps.
- **D-10** `DriverZone.max_speed_ms` stored as mean (also S10 — verify closed).
- **D-11** Driver decision sees stale `pack_current = 0` (same as S5).
- **D-18** Driver-change lap contaminates default calibration (= M18).
- **D-19** `from_telemetry` ignores user column names (= M19).
- **D-20** `CoastOnly` / `ThresholdBraking` ignore envelope (= M20).
- **D-21** `compare_driver_stints` returns segment-level diffs despite zone collapse.
- **D-24** `ReplayStrategy.decide` thresholds inconsistent (= m12).
- **D-25** `zone_for_segment` linear scan (= m13).
- **D-26** `initial_speed_ms` clamped 0.5 (= m13).
- **D-27** `ControlCommand` dataclass frozen but extension awkward.
- **D-28** `DriverParams` only used by `PedalProfileStrategy`.

### New-findings — moderate/minor buckets (not triaged)
The NF moderate/minor summary sections (~50 items across Numerical,
Unit, Conservation, Data-loading, Hidden-state, Architecture
categories) are catalogued in git history at
`docs/SIMULATOR_AUDIT_NEW_FINDINGS_2026-04-16.md` (pre-deletion); they
remain OPEN with no further triage. Representative items:
- Numerical: `+1e-6` regularizers, `math.fsum` accumulator, `iterrows()` motor map.
- Unit/dimensional: SOC fraction-vs-percent latent, `lvcu_power_constant` firmware-fit units.
- Conservation: cornering drag ignores load redistribution, distance-accumulator drift.
- Data-loading: CT-17EV YAML stale, `CdA` reference-area convention undocumented.
- Hidden state: module-level track constants not configurable; `.tir` not cached.
- Architecture: empty `src/fsae_sim/__init__.py`, `strategy.py` vs `strategies.py` footgun.

---

## DEFERRED (intentionally skipped this session)

- **NF-11** DSS sync script — non-trivial openpyxl work, design pass needed.
- **NF-24** `bms_limit` uses entry SOC/temp vs avg-segment torque — Heun-dependent.
- **NF-38** `minimize_scalar.success` checks — moot after C4 closed-form.
- **NF-58** `isinstance` on concrete strategies — arch refactor.
- **NF-60** `pyproject.toml` / Dockerfile dep + Python version pinning.
- **NF-61** Freeze result/config dataclasses.
- **NF-62** Split `analysis/telemetry_analysis.py` (603 lines, 4 concerns).
- **NF-63** Split `driver/strategies.py` (776 lines).
- **NF-64** `analysis` ↔ `driver.strategies` import cycle.
- **NF-65** Capability-flag try/except fail-silent at module level.
- **C10** Heun integrator / predictor-corrector.
- **C11** Mechanical-vs-electrical operating-point unification (depends on C10).
- **S2**, **S3**, **S5**, **S6**, **S7**, **S8** — engine-arch deep rewrites.
- Moderate/Minor items not explicitly listed in FIXED.

---

## REFUTED

- **#21** `_REGEN_EFFICIENCY_FACTOR` undefined — verified defined as class constant after D-17. [commit trail in f2804ea]
- **NF-38** `minimize_scalar.success` — no longer called after C4 closed-form peaks landed. Moot.

---

## Pointer index

Grep a legacy ID (e.g. `C4`, `NF-18`, `D-15`) against the git log for
the actual commit that closed it. Subsections above cite commit SHAs;
the first 7 chars are unique within this repo. If an ID appears in
both OPEN and FIXED lists, treat the FIXED entry as authoritative —
the OPEN listing is a duplicate cross-reference from the other source
document.
