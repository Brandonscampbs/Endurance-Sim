# Driver Model Fix Campaign — Post-Mortem (2026-04-16)

## Scope

28 items from `docs/DRIVER_MODEL_ISSUES_2026-04-16.md` (6 Critical, 11
Significant, 6 Moderate, 5 Minor), landed across four agent waves
(Foundation, Electrical, Calibration, Control+Cleanup) on `main`.

**Delivered: 28/28. Nothing punted.** D-28 was the one item flagged
as possibly punt-able in the design spec; landed cleanly as
`CalibratedStrategy.with_params(DriverParams)`, mirroring
`PedalProfileStrategy`. `with_zone_override` and `DriverParams`
operate at different granularities (zone-scope vs global scalar) so
both survive and compose rather than merge.

## Before / after

**D-23 driver-channel validation (laps 13–21, n=12 801 samples):**

| Channel | Pre | Post |
|---|---|---|
| Throttle RMSE / R² / corr | 0.30 / — / — | 0.30 / −1.03 / 0.01 |
| Brake RMSE / R² / corr    | 0.12 / — / — | 0.13 / −7.09 / −0.06 |
| Action accuracy           | 66 %         | 66.7 % |

The D-23 block only exercises ReplayStrategy, which already sampled
telemetry directly, so these numbers barely moved. A second D-23 pass
against `CalibratedStrategy` would be more meaningful and is listed
below as follow-up work.

**validate_tier3 Replay metrics: 5/8 → 6/8 passing.** Mean pack
voltage moved from ~6 % to 2.7 %. Final pack voltage (5.8 %) and mean
|pack current| (25.9 %) remain above threshold — battery-model
territory, not driver-model.

## Key decisions

See `docs/superpowers/specs/2026-04-16-driver-model-fixes-design.md`.

- **Back-EMF electrical model (D-17)** — rectifier term in
  `electrical_power`; `coast_electrical_power` deleted; dispatch by
  physical state, not `cmd.action`.
- **ControlCommand.metadata (D-27)** — extension hook; first consumer
  `max_speed_ms` (D-09).
- **Train/test split (D-04)** — calibration on laps 1–10, validation
  on laps 13–21; driver-change lap excluded from both.
- **Replay vs modeled power asymmetry (D-13)** — Replay uses measured
  V×I; non-replay uses modeled. Documented in-code.
- **Envelope-aware synthetic strategies (D-20)** — `CoastOnly` /
  `ThresholdBraking` accept an optional `SpeedEnvelope`; the engine
  auto-wires it via duck-typed `set_envelope`.

## R2 merge mid-campaign (`f6a1452`)

R2 merges (tire, dynamics, battery, powertrain, envelope, driver,
track, validation) landed while D-XX work was in flight. `f6a1452`
reconciled per-subsystem: took R2 physics where clearly superior
(closed-form tire peaks, cornering-solver convergence, battery
thermal mass) and kept D-XX driver-layer changes everywhere. Suite
went from 12 failing to green during the reconcile.

## New issues surfaced

Added to `docs/SIMULATOR_ISSUES.md`:

- D-23 harness only covers ReplayStrategy; calibrated/pedal-profile
  drivers have no per-sample regression gate.
- Final pack voltage (5.8 %) and mean |pack current| (25.9 %) fail
  vs telemetry — likely SOC-vs-V or pack-R(SOC, T) interpolation.
- `MotorEfficiencyMap` fallback now warns; test callers should
  either supply a stub or silence the warning.

## Status

`pytest`: **561 passed, 41 skipped, 0 failed, 0 xfailed.**
`validate_tier3.py`: **6/8 Replay metrics**, worst 25.9 %.
Design-spec §9 Definition of Done met.
