# Pedal-Profile Driver Model Design

**Date:** 2026-04-15
**Status:** Draft
**Replaces:** CalibratedStrategy as the primary driver model for validation and sweeps

## Problem

The current `CalibratedStrategy` produces 10.5% time error (1439s sim vs 1608s telemetry) because it collapses real driver behavior into 50 coarse zones with averaged torque fractions. It also bypasses the LVCU firmware model by storing post-LVCU torque fractions directly, which couples the driver model to the car tune and prevents independent car-tune sweeps.

Meanwhile, `ReplayStrategy` achieves 0.1% time error by replaying raw driver inputs through the physics. The gap is entirely driver model quality.

## Design

### Core Concept

Store what the driver's feet actually do: **per-segment throttle position (%) and brake pressure**, extracted from telemetry. At runtime, feed these raw pedal inputs through `lvcu_torque_command()` — the same firmware chain the real car uses. The driver model knows nothing about torque; it only knows pedal positions.

This gives clean separation:
- **Driver model** = pedal/brake profiles + sweep modifiers (what the human does)
- **LVCU model** = pedal-to-torque conversion (what the car's firmware does)
- **Car tune** = PowertrainConfig parameters (torque limit, RPM limit, power constant)

When you change the car tune, the same pedal inputs produce different torque, different speeds, different lap times. When you change driver behavior, different pedal inputs go through the same LVCU. Both sweep independently.

### Class: `PedalProfileStrategy`

New class in `src/fsae_sim/driver/strategies.py`, subclassing `DriverStrategy`.

#### Calibration Data (per-segment arrays, one value per track segment)

Extracted from telemetry via median across representative laps:

| Array | Source | Description |
|---|---|---|
| `throttle_pct[N]` | Median of `Throttle Pos / 100` per segment | Raw pedal position, 0-1 |
| `brake_pct[N]` | Median of `max(FBrake, RBrake) / brake_norm` per segment | Normalized brake pressure, 0-1 |
| `action[N]` | Majority vote per segment | THROTTLE / COAST / BRAKE classification |
| `ref_speed_ms[N]` | Median of `GPS Speed / 3.6` per segment | Reference speed (for driver sweep scaling, not for control) |

`N` = `track.num_segments` (default ~2015 at 0.5m resolution, or ~201 at 5m).

`brake_norm` = 99th percentile of nonzero brake pressure across all moving samples (same normalization as existing code).

#### Sweep Parameters

```python
@dataclass
class DriverParams:
    """Tunable driver behavior parameters for sweeps."""
    throttle_scale: float = 1.0       # Multiplier on all throttle inputs
    brake_scale: float = 1.0          # Multiplier on all brake inputs
    coast_throttle: float = 0.0       # Throttle applied during coast segments (0 = pure coast)
    max_throttle: float = 1.0         # Hard cap on throttle output
    max_brake: float = 1.0            # Hard cap on brake output
```

These are intentionally simple scalar multipliers. A `throttle_scale` of 1.1 means "driver pushes 10% harder everywhere." For Phase 3 sweeps, these are the knobs to turn. More granular control (per-zone overrides) can be added later but isn't needed now.

#### `decide()` Logic

```python
def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
    seg_idx = state.segment_idx % self._num_segments
    action = self._actions[seg_idx]

    if action == THROTTLE:
        throttle = self._throttle_pct[seg_idx] * self.params.throttle_scale
        throttle = min(throttle, self.params.max_throttle)
        throttle = clip(throttle, 0.0, 1.0)
        return ControlCommand(THROTTLE, throttle_pct=throttle)

    elif action == BRAKE:
        brake = self._brake_pct[seg_idx] * self.params.brake_scale
        brake = min(brake, self.params.max_brake)
        brake = clip(brake, 0.0, 1.0)
        return ControlCommand(BRAKE, brake_pct=brake)

    else:  # COAST
        return ControlCommand(COAST, throttle_pct=self.params.coast_throttle)
```

#### `from_telemetry()` Classmethod

Same lap detection and segment sampling as existing `extract_per_segment_actions()`, but:
- Stores raw `Throttle Pos / 100` as intensity (not LVCU Torque Req / 85)
- Stores raw normalized brake pressure (not torque-converted)
- No zone collapsing — keeps per-segment resolution

Reuses the existing per-lap-then-aggregate pipeline (`_extract_per_lap_then_aggregate` pattern) for median across representative laps.

#### `with_params()` Method

Returns a new strategy instance with modified `DriverParams`:
```python
def with_params(self, **kwargs) -> PedalProfileStrategy:
    new_params = replace(self.params, **kwargs)
    return PedalProfileStrategy(..., params=new_params)
```

### Engine Integration

The engine must process `PedalProfileStrategy` commands through `lvcu_torque_command()` — the full LVCU firmware chain. This is the **default path** for any strategy that isn't ReplayStrategy or CalibratedStrategy. So **no engine changes are needed** — the existing `else` branch already does the right thing:

```python
# Existing engine code (no changes needed):
else:
    motor_torque = self.powertrain.lvcu_torque_command(
        cmd.throttle_pct, motor_rpm, bms_current_limit,
    )
```

The `throttle_pct` from `PedalProfileStrategy` is the raw pedal position (0-1), which goes through dead zone remap, power-limited ceiling, and inverter clamp — exactly like the real car.

### Car Tune Sweeps

Car tune parameters live in `PowertrainConfig` (already exists in `configs/ct16ev.yaml`):
- `torque_limit_inverter_nm`: Inverter torque cap (currently 85 Nm)
- `torque_limit_lvcu_nm`: LVCU software torque cap (currently 150 Nm)
- `motor_speed_max_rpm`: Max RPM (currently 2900)
- `brake_speed_rpm`: Field-weakening onset RPM (currently 2400)
- `lvcu_power_constant`: Power limiting constant

To sweep car tune: modify the YAML or construct `PowertrainConfig` with different values. Same `PedalProfileStrategy`, different `PowertrainConfig` → different `lvcu_torque_command()` output → different performance. No driver model changes needed.

### What This Replaces

- `CalibratedStrategy` remains in the codebase (backwards compat) but `PedalProfileStrategy` becomes the primary model for validation and sweeps
- `ReplayStrategy` remains as the physics validation baseline (distance-indexed replay)
- The telemetry extraction pipeline (`extract_per_segment_actions`, `collapse_to_zones`) remains for zone-based analysis but is not used by `PedalProfileStrategy`

### Validation Criteria

Run `PedalProfileStrategy` with default params (all 1.0) on 22-lap Michigan endurance:

| Metric | Target | Rationale |
|---|---|---|
| Driving time | < 5% error vs telemetry (1608s) | Must be much closer than CalibratedStrategy's 10.5% |
| Energy consumed | < 10% error vs telemetry (3.33 kWh) | Current: 11.5% |
| SOC consumed | < 10% error vs telemetry (34%) | Current: 10.8% |
| Mean pack current | < 20% error vs telemetry (18.6A) | Current: 26.8% |
| All 8 metrics | Pass | CalibratedStrategy passes 5/8 |

Stretch goal: match replay's 8/8 pass rate. The pedal-profile approach should be very close to replay since it uses the same raw inputs, just segment-averaged instead of distance-interpolated.

### File Changes

| File | Change |
|---|---|
| `src/fsae_sim/driver/strategies.py` | Add `PedalProfileStrategy` class and `DriverParams` dataclass |
| `scripts/validate_driver_model.py` | Switch from `CalibratedStrategy` to `PedalProfileStrategy` |
| `tests/test_pedal_profile_strategy.py` | New test file for the strategy |

No changes to: `engine.py`, `powertrain_model.py`, `strategy.py`, `telemetry_analysis.py`.

### Sweep Usage (Phase 3 Preview)

```python
# Calibrate baseline from telemetry
baseline = PedalProfileStrategy.from_telemetry(aim_df, track)

# Driver sweep: more aggressive
aggressive = baseline.with_params(throttle_scale=1.15)

# Driver sweep: conservative with regen
conservative = baseline.with_params(throttle_scale=0.85, brake_scale=1.2)

# Car tune sweep: higher torque limit
config_high_torque = VehicleConfig.from_yaml("configs/ct16ev.yaml")
config_high_torque.powertrain.torque_limit_inverter_nm = 100.0
engine = SimulationEngine(config_high_torque, track, baseline, battery)
# Same driver, more power available → faster

# Combined sweep: aggressive driver + high power
engine = SimulationEngine(config_high_torque, track, aggressive, battery)
```
