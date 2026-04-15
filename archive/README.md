# Archive

Investigation and debugging scripts from the Phase 2 physics alignment work (March-April 2026). These scripts were used to diagnose simulation-vs-telemetry discrepancies and are preserved for reference. Their conclusions have been absorbed into the codebase.

**Do not use these scripts for current work.** They may reference old APIs, wrong parameter values, or fixed bugs.

## What was investigated

| Directory | Scripts | Key findings |
|---|---|---|
| `analysis/` | 8 thermal/force/SOC scripts | Battery resistance characterization, SOC tracking behavior (BMS saturation limits), regen torque discrepancy quantified |
| `scripts/` | 18 force/speed/torque scripts | Efficiency double-dip bug found and fixed, tire radius was wrong (0.228 -> 0.2042m), parasitic drag was missing (~70N), LVCU torque chain validated against firmware |

## Keeper scripts (still in `scripts/`)

The following scripts remain in `scripts/` because they are part of permanent workflows:

- `clean_endurance_data.py` -- produces `CleanedEndurance.csv` from raw AiM export
- `fix_gps_data.py` -- reconstructs corrupted GPS data for laps 1-5
- `analyze_gps_laps.py` -- detects start/finish crossings, computes lap boundaries
- `transplant_fx_coefficients.py` -- copies longitudinal Pacejka coefficients from R25B donor .tir files to LC0
- `validate_driver_model.py` -- end-to-end CalibratedStrategy validation runner
- `validate_tier3.py` -- Tier 3 validation (replay sim, Pacejka vs legacy, force-based strategies)
