# LVCU Torque Command Model — Design Spec

**Date**: 2026-04-15
**Status**: Approved
**Goal**: Replace the sim's simplified torque model with the real LVCU torque command chain, remove the speed cap bandaid, and move current limiting upstream of force resolution.

## Problem

The sim computes drive force from `pedal × max_motor_torque(rpm)` with no regard for BMS current limits, then clamps pack current after exit speed is already resolved. The real car's LVCU limits torque *before* commanding the inverter, using a power-limited formula that incorporates the BMS discharge current limit. This causes the sim to be ~10% too fast because it applies more force than the real car can produce, especially in the second half of endurance when temperature rises and SOC drops.

An overnight session added per-segment speed caps from telemetry to mask this — a bandaid that makes the sim unable to predict anything new. This design replaces that with the real physics.

## LVCU Code (from Real-Car-Data-And-Stats/LVCU Code.txt)

The real torque command chain is:

```
pedal (0-1) -> tmap_lut() -> torque_lut() -> CAN command -> inverter clamp
```

### tmap_lut: Pedal Dead Zone Remap

```c
double tmap_lut(double tps) {
    double V_MIN = 0.1;
    double V_MAX = 0.9;
    double tps_local = (fmax(V_MIN, fmin(tps, V_MAX)) - V_MIN) * (1 / (V_MAX - V_MIN));
    return tps_local;
}
```

Remaps pedal [0.1, 0.9] -> [0.0, 1.0]. Bottom 10% and top 10% are dead zones.

### torque_lut: Power-Limited Torque Ceiling

```c
int torque_lut(double tps) {
    uint32_t torque_limit_local = torque_limit;
    torque_limit_local = fmin(torque_limit,
        (double)(4200 * current_limit) * 1.0 / fmax(230.4, (double)motor_speed * 0.1076));
    if (motor_speed >= 6000) {
        torque_limit_local = 300;
    }
    return tps * torque_limit_local;
}
```

All values in Cascadia 0.1 Nm CAN units. Endurance settings:
- `torque_limit` = 1500 (150 Nm LVCU limit)
- `current_limit` = BMS discharge limit (starts ~100A, dynamic via CAN)
- Inverter independently clamps to 85 Nm (IQ=170A setting)

### Reverse-Engineered Constants

The formula `4200 * I / max(230.4, RPM * 0.1076)` implements a constant-power torque ceiling:

- **Effective power**: P ≈ 409 × I_bms watts ≈ V_nominal × I_bms (110S pack at ~407V)
- **Below ~2141 RPM**: floor of 230.4 dominates, flat torque cap
- **Above ~2141 RPM**: torque drops as 1/RPM (constant power hyperbola)
- **6000 RPM override**: 30 Nm limp mode (irrelevant for endurance at 2900 max)

### When the Power Limit Gates Torque (below 85 Nm inverter limit)

| BMS Current | Torque @ 2900 RPM | Gates? |
|---|---|---|
| 100A (30C) | 134 Nm | No |
| 65A (40C) | 87 Nm | Barely |
| 55A (45C) | 74 Nm | Yes |
| 45A (50C) | 60 Nm | Yes |
| 50A (70% SOC + 40C) | 67 Nm | Yes |

The power limit increasingly constrains torque as the endurance run progresses.

## Design

### 1. PowertrainModel — New `lvcu_torque_command` Method

**Method signature:**
```python
def lvcu_torque_command(
    self, pedal_pct: float, motor_rpm: float, bms_current_limit_a: float
) -> float:
```

**Logic (faithful to C code):**
1. `tmap_lut`: remap pedal [deadzone_low, deadzone_high] -> [0.0, 1.0], clamped
2. Power-limited ceiling: `min(lvcu_limit_nm, K_power * I_bms / max(omega_floor, RPM * K_rpm))`
3. Inverter clamp: `min(result, inverter_torque_limit)`
4. Over-speed override: if RPM >= 6000, ceiling = 30 Nm
5. Return: `remapped_pedal * clamped_ceiling`

**Constants in Nm (converted from 0.1 Nm CAN units):**
- `K_power` = 420.0 (4200 / 10)
- `K_rpm` = 0.1076
- `omega_floor` = 23.04 (230.4 / 10)
- `overspeed_rpm` = 6000
- `overspeed_torque_nm` = 30.0 (300 / 10)

### 2. PowertrainConfig — New Fields

```yaml
powertrain:
  # ... existing fields ...
  lvcu_power_constant: 420.0       # K_power in Nm (4200 in 0.1Nm CAN units / 10)
  lvcu_rpm_scale: 0.1076           # RPM to angular velocity scale factor
  lvcu_omega_floor: 23.04          # min denominator (230.4 in CAN units / 10)
  lvcu_pedal_deadzone_low: 0.1     # tmap_lut V_MIN
  lvcu_pedal_deadzone_high: 0.9    # tmap_lut V_MAX
  lvcu_overspeed_rpm: 6000         # hard torque override threshold
  lvcu_overspeed_torque_nm: 30.0   # torque at overspeed
```

### 3. SimulationEngine — Upstream Torque Limiting

**New force-based resolution flow:**
1. Get BMS current limit: `bms_I = battery_model.max_discharge_current(temp, soc)`
2. Compute motor RPM from current speed
3. Get LVCU-limited torque: `motor_torque = powertrain.lvcu_torque_command(cmd.throttle_pct, rpm, bms_I)`
4. Drive force: `drive_f = powertrain.wheel_force(motor_torque)`, capped by traction
5. Resolve exit speed with correctly-limited force
6. Compute electrical power from same `motor_torque`

**Remove:**
- Speed cap / `speed_target_ms` code (engine lines 236-244)
- After-the-fact BMS current clamp (engine lines 270-275)
- `isinstance(self.strategy, CalibratedStrategy)` check in engine

### 4. CalibratedStrategy Cleanup

**Remove:**
- `_segment_speed_targets_ms` field
- `speed_target_ms()` method
- Speed target extraction in `from_telemetry()`
- `segment_speed_targets_ms` passthrough in `with_zone_override()`

### 5. telemetry_analysis.py

- Revert p90 speed percentile back to `np.mean` (speed targets no longer feed into sim)

### 6. What We Keep From Overnight Work

- `motor_efficiency.py` + EMRAX motor map — real physics
- `PowertrainModel._get_efficiency()` — operating-point efficiency
- `load_cleaned_csv()` — better telemetry data
- `Track.from_telemetry()` DataFrame input + optional GPS columns
- `grip_scale: 0.4697` — recalibrated for cleaned data
- Starting temp 29C — matches reality

### 7. What We Discard

- Speed cap in engine
- Speed targets in strategy
- p90 speed percentile in telemetry analysis

## Testing

### Unit Tests for `lvcu_torque_command`

| Test Case | Pedal | RPM | BMS I | Expected |
|---|---|---|---|---|
| Low RPM, full power | 1.0 | 1000 | 100 | 85 Nm (inverter-limited) |
| Mid RPM, low current | 1.0 | 2900 | 50 | ~67 Nm (power-limited) |
| Half pedal | 0.5 | 1000 | 100 | 42.5 Nm |
| Below dead zone | 0.05 | 1000 | 100 | 0 Nm |
| Above dead zone | 0.95 | 1000 | 100 | 85 Nm |
| Zero current | 1.0 | 1000 | 0 | 0 Nm |
| Overspeed | 1.0 | 6500 | 100 | 30 Nm |

### Integration Validation

Run 22-lap endurance sim vs cleaned telemetry. Compare:
- Driving time, energy, SOC, temperature, voltage, current

If time accuracy worsens (expected without speed cap), that reveals real gaps to fix (driver intensity, cornering model, etc.) — not something to patch over.
