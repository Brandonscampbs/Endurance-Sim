# FSAE EV Simulator — Known Issues

Current open physics/code gaps. Detail and fix history live in git; this file is a working list of what's still wrong, intended to stay small enough to live in CLAUDE.md context.

## Status

| Status   | Count | Notes |
|----------|-------|-------|
| PARTIAL  | 2     | Regen double-count residual; tire radius not dynamic |
| DEFERRED | ~17   | Engine-arch rewrites, test/config hygiene |
| OPEN     | ~38   | Moderate/Minor buckets untriaged |

Legend: `C*` critical, `S*` significant, `M*` moderate, `m*` minor, `NF-*` new-finding, `D-*` driver-model.

---

## PARTIAL

- **3** `regen_force` generator-mode sign — S12 addressed; latent under CT-16EV (no regen commanded per commit 591d79e), hot for CT-17EV if regen is enabled.
- **18** Tire radius 0.2042 m constant, not dynamic `loaded_radius()` (~3% under load).

## OPEN

### Critical
- **C10** Engine integrates speed with entry-speed forces; no Heun corrector.
- **C11** Mechanical vs electrical torque use different operating points.

### Significant
- **S1** Regen tire-saturation doesn't feed back to electrical power (absorbs into C11).
- **S2/S3** Multiple field-weakening models; replay double-counts.
- **S4** `resolve_exit_speed` clamps without charging energy.
- **S5** Driver decision sees stale `pack_current = 0` per segment.
- **S6** Speed envelope ignores BMS current cap.
- **S7** Combined-slip (Pass 4) dead code.
- **S8** `ReplayStrategy` V×I path — watch for regression.

### Moderate
- **4** `battery_model.py` thermal model is lumped with constant `thermal_conductance_w_per_k`; no airflow/speed dependence and no per-module/cell gradient.
- **5** Residual scipy optimizer calls may remain in cornering solver.
- **7** Linear field-weakening taper — audit physics path.
- **8** `battery_model.py` internal resistance temperature-independent (SOC dependence added by S17; T dependence still pending).
- **9** `analysis/scoring.py` EFmin falls back to 0 when track distance is unknown (inflates efficiency score on that path).
- **12** Python sim loop limits throughput.
- **14** Effective mass includes drivetrain efficiency on rotor inertia.
- **16** Track curvature from GPS acceleration (noisy at low speed).
- **17** 5 m segment resolution — nominal config entry stale (default now 0.5 m).
- **19** ISA air density vs Michigan conditions.
- **20** Downforce front distribution is default, not DSS-measured.
- **22** Back-EMF alone doesn't explain coast power (~45 Wh/stint gap).
- **M1** No anti-squat / anti-dive geometry.
- **M2** Friction ellipse uses peak forces, not combined-slip Pacejka.
- **M3** OCV extrapolated linearly below calibration range.
- **M5** 4P cell current sharing assumed perfect.
- **M6** No OCV hysteresis.
- **M7** Downforce treated inconsistently across resistance functions.
- **M8** Camber sign convention undocumented.
- **M10** Brake distribution load-proportional, not mechanical-bias.
- **M12** PAC2002 Svy missing LKYG on camber term.
- **M13** `m_effective` doesn't distinguish accel vs regen direction.
- **M14** LONGVL speed correction ignored in Fx.
- **M15** `max_traction_force` hardcodes 0.3g load-transfer (NF-6 dropped the 0.3g/-1.0g magic from force iter; audit whether this duplicate claim still applies).
- **M16** Forward-Euler lag between `pack_voltage` and `pack_current`.
- **M17** Regen active at arbitrarily low RPM (no back-EMF cutoff).

### Minor
- **m2** Plots use `LFspeed`; metrics use `GPS Speed`.
- **m3** Validation plot auto-scales, hiding bias.
- **m4** Pass/fail thresholds loose (15–20 %).
- **m5** Mean-speed / distance / time are algebraic identities.
- **m6** `speed > 5 km/h` filter no-op on cleaned data.
- **m10** Empty-bin carry-forward corrupts curvature across lap wrap.
- **m11** Grade is not smoothed.
- **m12** `ReplayStrategy.decide` 0.05 thresholds inconsistent with calibration units.
- **m13** Cosmetic cluster: linear-scan `zone_for_segment`, standing-start clamp, dead `initial_speed` API, FW constants Michigan-fit, lap-wrap speed clamp, `laps_completed` off-by-one.

### Other / untriaged
- Numerical regularizers (`+1e-6`), `math.fsum` accumulator, `iterrows()` motor map.
- Unit latent: SOC fraction-vs-percent; `lvcu_power_constant` firmware-fit units.
- Conservation: cornering drag ignores load redistribution; distance-accumulator drift.
- Data-loading: CT-17EV YAML stale (NF-31 addressed `lvcu_power_constant`; rest unaudited); `CdA` reference-area convention undocumented.
- Hidden state: module-level track constants not configurable; `.tir` not cached.

## Xfailed tests (deferred)

- `tests/test_engine_envelope.py::test_synthetic_strategy_uses_envelope` — engine exceeds envelope ~1 m/s at tightest corner; needs engine-side fix.
- `tests/test_tire_model.py::test_closed_form_peak_longitudinal_matches_optimizer` — closed-form Fx diverges from optimizer baseline 14–90 % at Fz ≥ 1500 N; needs tire-model audit.

## DEFERRED (intentionally skipped; subset of OPEN above)

- **NF-11** DSS sync script (openpyxl).
- **NF-24** `bms_limit` uses entry SOC/temp vs avg-segment torque — Heun-dependent.
- **NF-58** `isinstance` on concrete strategies — arch refactor.
- **NF-60** `pyproject.toml` / Dockerfile dep + Python version pinning.
- **NF-61** Freeze result/config dataclasses.
- **NF-62** Split `analysis/telemetry_analysis.py`.
- **NF-63** Split `driver/strategies.py`.
- **NF-64** `analysis` ↔ `driver.strategies` import cycle.
- **NF-65** Capability-flag try/except fail-silent at module level.
- **C10** Heun integrator / predictor-corrector.
- **C11** Mechanical-vs-electrical operating-point unification (depends on C10).
- **S2, S3, S5, S6, S7, S8** — engine-arch deep rewrites.

---

Pointer: grep a legacy ID (`C4`, `NF-18`, `D-15`, etc.) against `git log` to find the commit that closed it.
