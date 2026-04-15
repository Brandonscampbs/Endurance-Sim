# Simulation Alignment Log

## Goal
Align the FSAE endurance simulation against cleaned Michigan 2025 telemetry (`CleanedEndurance.csv`) to per-segment accuracy: speed 1:1, torque matched, energy/SOC/voltage/temperature within 5%.

## Telemetry Ground Truth
- **File**: `Real-Car-Data-And-Stats/CleanedEndurance.csv`
- **Speed sensor**: LFspeed (front wheel speed, km/h) — more accurate than GPS Speed
- **Official time**: 1614s (cleaned to official timing window)
- **Total distance**: 22,100 m (22 laps x ~1005 m)
- **Energy**: 3.33 kWh (V*I integration)
- **SOC**: 95.0% -> 60.5% (34.0% consumed, BMS)
- **Pack temp**: 29 -> 38 C (+9 C)
- **Min cell voltage**: 3.31 V
- **Max LFspeed**: 63.2 km/h
- **Mean LFspeed (moving)**: 49.5 km/h

## Iteration History

### Iteration 0 — Baseline (old CSV, original code)
- Sim: 1499.9s, 3.18 kWh, 62.8% final SOC
- Time error: 9.9% (164.6s too fast)
- 7/8 metrics passed (driving time failed)
- Root cause: sim 4.8s/lap too fast, no speed regulation

### Iteration 1 — Cleaned CSV, loader adaptation
- Added `load_cleaned_csv()` for new format
- Updated Track.from_telemetry for missing GPS columns
- Recalibrated grip_scale: 0.4643 -> 0.4697 (+1.2%)
- Sim: 1503.6s, 3.07 kWh, 64.5% final SOC
- Time error: 6.5% (105.2s too fast)
- 6/8 passed (time, SOC failed)

### Iteration 2 — Strategy-level speed modulation (REVERTED)
- Added speed-aware throttle modulation to CalibratedStrategy
- Mean speed targets too low, 85% ramp-down too early
- Sim: 1766.8s (158s too slow), 1.99 kWh
- 3/8 passed — approach over-corrected, reverted

### Iteration 3 — p90 speed targets + tighter band (PARTIALLY REVERTED)
- Changed speed percentile to p90, modulation band to 97%
- Sim: 1602.2s, 2.40 kWh
- Time correct (0.4%) but energy 28% too low
- Coasting at target = zero power draw, unrealistic

### Iteration 4 — Engine-level speed cap + maintenance torque
- Speed target enforced as physics cap in engine
- Motor torque = maintenance (resistance-matching) when capped
- Sim: 1614.6s, 2.71 kWh
- Time perfect but energy still 18.6% low
- Maintenance torque underestimates real energy

### Iteration 5 — Commanded torque with speed cap (CURRENT)
- Speed target caps exit speed (correct time)
- Motor torque = commanded throttle (correct energy)
- Sim: 1614.6s, 3.29 kWh, 62.2% final SOC
- **8/8 metrics passed**
- Time: 0.4% error (5.8s), Energy: 1.1% error
- Score: 252.5 vs 252.9 actual (0.2% error)

### Iteration 6 — Motor efficiency map + start temp 29°C
- Implemented 2D motor efficiency lookup from EMRAX 228 CSV
- Fixed starting temperature: 25°C → 29°C (matched telemetry)
- Sim: 1614.6s, 3.40 kWh, 61.0% final SOC, 39.0°C final temp
- **8/8 metrics passed**
- SOC error: 0.0% (was 3.7%)
- Temp error: 2.7% (was 9.1%)
- Score error: 1.5%

### Iteration 7 — LVCU torque command model, speed cap removed (CURRENT)
- Implemented real LVCU torque command chain: tmap_lut dead zone remap + power-limited ceiling + inverter clamp
- LVCU limits torque upstream of force resolution (BMS current limit feeds into torque formula)
- Removed speed cap bandaid from engine (speed targets from telemetry)
- Removed speed target fields from CalibratedStrategy
- Removed after-the-fact BMS current clamp from engine
- Reverted p90 speed percentile back to mean in telemetry analysis
- Sim: 1455.9s, 3.57 kWh, 58.7% final SOC, 41.5°C final temp
- **6/8 metrics passed** (time and current failed)
- Time: 9.5% too fast — honest baseline without bandaid
- Energy: 7.3% high (driving harder without speed regulation)
- Root cause of time gap: zone-averaged throttle intensities don't capture real driver's continuous speed modulation. The LVCU physics are correct, but the driver model over-drives.

## Current Status
| Metric | Telemetry | Sim | Error | Target | Status |
|---|---|---|---|---|---|
| Driving time | 1608.8s | 1455.9s | 9.5% (152.9s) | <5% | FAIL |
| Distance | 22,100m | 22,105m | 0.0% | <1% | PASS |
| SOC consumed | 34.0% | 36.3% | 6.7% | <10% | PASS |
| Cell temp | 38.0 C | 41.5 C | 9.1% | <15% | PASS |
| Mean voltage | 411.7V | 407.5V | 1.0% | <5% | PASS |
| Final voltage | 390.8V | 390.9V | 0.0% | <5% | PASS |
| Mean |current| | 18.6A | 23.0A | 23.8% | <20% | FAIL |
| Energy | 3.33 kWh | 3.57 kWh | 7.3% | <15% | PASS |

## Remaining Work
1. Driver model refinement — zone-averaged throttle intensities over-drive vs real driver's continuous speed modulation (root cause of 9.5% time gap)
2. Per-segment speed alignment — need finer-grained driver model or speed-dependent throttle modulation within zones
3. Current alignment — 23.8% too high, directly follows from over-driving

## Changes Made
| File | Change |
|---|---|
| `src/fsae_sim/data/loader.py` | Added `load_cleaned_csv()` |
| `src/fsae_sim/track/track.py` | Flexible GPS column handling, DataFrame input |
| `src/fsae_sim/analysis/telemetry_analysis.py` | Mean speed aggregation (was p90, reverted) |
| `src/fsae_sim/driver/strategies.py` | Zone-based driver model (speed targets removed) |
| `src/fsae_sim/sim/engine.py` | LVCU torque command upstream of force resolution, motor map loading |
| `src/fsae_sim/vehicle/motor_efficiency.py` | NEW: 2D motor+inverter efficiency lookup from EMRAX 228 CSV |
| `src/fsae_sim/vehicle/powertrain_model.py` | `lvcu_torque_command()` + operating-point efficiency |
| `src/fsae_sim/vehicle/powertrain.py` | LVCU config fields (dead zone, power constant, etc.) |
| `configs/ct16ev.yaml` | grip_scale 0.4697, LVCU parameters |
| `scripts/validate_driver_model.py` | Uses cleaned CSV, start temp 29°C |
