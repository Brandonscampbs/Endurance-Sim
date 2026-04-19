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

## CLOSED

Full-audit pass against Michigan lap 16. With these fixes the legacy
backend now matches telemetry at ``grip_scale=0.65`` on **both** lap
time (70.92 s sim vs 71.05 s telem) **and** mean bias (+0.01 m/s) —
previously no single grip_scale could match both, which was the
smoking gun for missing physics.

- **Track curvature from IMU LatAcc** (2026-04-17) — `Track.from_telemetry`
  now computes signed curvature directly from GPS lat/lon path geometry
  via `_curvature_from_position` (second derivative of the smoothed
  x(s), y(s) projection). Previously used `GPS LatAcc × 9.81 / GPS
  Speed²`, which silently returned zero whenever LatAcc was NaN (the
  first 3-4 laps of Michigan 2025 had all-NaN LatAcc during IMU
  warm-up), producing a nearly-straight track with no corners.
- **Cornering solver pure-lateral default** (2026-04-19) — `CorneringSolver.max_cornering_speed`
  now closes the drag loop: at each candidate speed the drive-tire Fx
  demand is computed from aero drag + rolling resistance, and the
  friction-ellipse reduction is applied to the rear tires via the
  existing `_can_sustain` path.
- **Mechanical brake force via motor-torque cap** (2026-04-19) —
  `engine.py` now routes brake commands through
  `dynamics.mechanical_brake_force`, which caps at the tire limit (~4-5 kN,
  ~1.5-2g). Previously used `powertrain.regen_force`, capping at motor
  torque capability (~1.5 kN, ~0.55g) — wrong for CT-16EV's mechanical-
  only brakes.
- **ReplayStrategy brake_pct percentile normalization** (2026-04-19) —
  `from_aim_data` / `from_full_endurance` now normalize by physical
  `brake_max_pressure_bar=60`, not by 99th-percentile of observed
  pressure. Previously 8.5 bar peak on a 60-bar system became
  `brake_pct=1.0`, inflating replay brake demand 7×.
- **`m_effective` spurious gearbox-efficiency multiplier** (2026-04-19) —
  `VehicleDynamics.m_effective` now uses `J_reflected = J_rotor · G²`
  (pure kinematics). Previously multiplied by `η_gearbox` which silently
  made sim ~0.4% fast in acceleration.
- **Cornering drag Fy-proportional-to-Fz distribution** (2026-04-19) —
  `_cornering_drag_pacejka` now uses yaw-equilibrium axle share and
  per-axle common-slip-angle solve. Previously distributed lateral force
  proportional to each tire's Fz and solved per-tire, which under-
  counted drag from lightly-loaded inside tires.
- **Lateral load transfer total-mass + no-aero** (2026-04-19) —
  `LoadTransferModel.lateral_transfer` now uses sprung mass only for the
  elastic (roll-stiffness) path and includes aero downforce in the
  per-axle Fz that drives the geometric path. Adds
  `unsprung_mass_front_kg`/`unsprung_mass_rear_kg` to `VehicleParams`.
- **Segment resistance evaluated at entry speed only** (2026-04-19) —
  `engine.py` now evaluates `total_resistance` at the segment mid-point
  speed via one Picard iteration. Previously held entry-speed resistance
  constant across the segment, under-counting drag by up to 2× on
  accelerating straights where v can double.

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
