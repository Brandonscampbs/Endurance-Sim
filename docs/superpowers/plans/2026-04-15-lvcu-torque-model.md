# LVCU Torque Command Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sim's simplified `pedal × max_torque(rpm)` with the real LVCU torque command chain (dead zone remap → power-limited ceiling → inverter clamp), move current limiting upstream of force resolution, and remove the speed cap bandaid.

**Architecture:** Add `lvcu_torque_command()` to `PowertrainModel` replicating the real C code. Add LVCU config fields to `PowertrainConfig` and YAML. Rewire `SimulationEngine` to compute torque via LVCU before resolving forces. Remove speed cap from engine and speed targets from `CalibratedStrategy`.

**Tech Stack:** Python, pytest, NumPy, dataclasses, YAML config

**Spec:** `docs/superpowers/specs/2026-04-15-lvcu-torque-model-design.md`

---

### Task 1: Add LVCU Config Fields to PowertrainConfig

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain.py`
- Modify: `configs/ct16ev.yaml`

- [ ] **Step 1: Add fields to PowertrainConfig dataclass**

In `src/fsae_sim/vehicle/powertrain.py`, add LVCU fields with defaults so existing code doesn't break:

```python
@dataclass(frozen=True)
class PowertrainConfig:
    """Motor, inverter, and drivetrain parameters."""
    motor_speed_max_rpm: float
    brake_speed_rpm: float
    torque_limit_inverter_nm: float
    torque_limit_lvcu_nm: float
    iq_limit_a: float
    id_limit_a: float
    gear_ratio: float
    drivetrain_efficiency: float
    # LVCU torque command parameters (from real LVCU Code.txt)
    lvcu_power_constant: float = 420.0        # 4200 in 0.1Nm CAN units / 10
    lvcu_rpm_scale: float = 0.1076            # RPM to angular velocity scale
    lvcu_omega_floor: float = 23.04           # 230.4 in CAN units / 10
    lvcu_pedal_deadzone_low: float = 0.1      # tmap_lut V_MIN
    lvcu_pedal_deadzone_high: float = 0.9     # tmap_lut V_MAX
    lvcu_overspeed_rpm: float = 6000.0        # hard torque override threshold
    lvcu_overspeed_torque_nm: float = 30.0    # torque at overspeed (300/10)
```

- [ ] **Step 2: Add LVCU fields to ct16ev.yaml**

In `configs/ct16ev.yaml`, add under the `powertrain:` section:

```yaml
powertrain:
  # ... existing fields ...
  lvcu_power_constant: 420.0
  lvcu_rpm_scale: 0.1076
  lvcu_omega_floor: 23.04
  lvcu_pedal_deadzone_low: 0.1
  lvcu_pedal_deadzone_high: 0.9
  lvcu_overspeed_rpm: 6000.0
  lvcu_overspeed_torque_nm: 30.0
```

- [ ] **Step 3: Verify config loads**

Run: `python -c "from fsae_sim.vehicle import VehicleConfig; c = VehicleConfig.from_yaml('configs/ct16ev.yaml'); print(c.powertrain.lvcu_power_constant)"`

Expected: `420.0`

- [ ] **Step 4: Commit**

```bash
git add src/fsae_sim/vehicle/powertrain.py configs/ct16ev.yaml
git commit -m "feat: add LVCU torque command config fields to PowertrainConfig"
```

---

### Task 2: Write Failing Tests for lvcu_torque_command

**Files:**
- Modify: `tests/test_powertrain_model.py`

- [ ] **Step 1: Write test class for LVCU torque command**

Add to `tests/test_powertrain_model.py`:

```python
class TestLVCUTorqueCommand:
    """Tests for lvcu_torque_command — replicates real LVCU C code."""

    def test_full_pedal_low_rpm_inverter_limited(self, model: PowertrainModel) -> None:
        """At 100A and low RPM, inverter 85 Nm limit should bind."""
        torque = model.lvcu_torque_command(1.0, 1000.0, 100.0)
        assert torque == pytest.approx(85.0)

    def test_full_pedal_high_rpm_power_limited(self, model: PowertrainModel) -> None:
        """At 50A and 2900 RPM, the power limit should bind below 85 Nm.

        Power ceiling: 420 * 50 / max(23.04, 2900 * 0.1076)
                     = 21000 / max(23.04, 312.04)
                     = 21000 / 312.04
                     ≈ 67.3 Nm
        """
        torque = model.lvcu_torque_command(1.0, 2900.0, 50.0)
        assert torque == pytest.approx(420.0 * 50.0 / (2900.0 * 0.1076), rel=0.01)
        assert torque < 85.0

    def test_half_pedal_scales_linearly(self, model: PowertrainModel) -> None:
        """Half pedal (after dead zone remap) gives half the torque ceiling."""
        # pedal=0.5 remaps to (0.5-0.1)/(0.9-0.1) = 0.5
        full = model.lvcu_torque_command(1.0, 1000.0, 100.0)
        half = model.lvcu_torque_command(0.5, 1000.0, 100.0)
        assert half == pytest.approx(full * 0.5)

    def test_pedal_below_deadzone_gives_zero(self, model: PowertrainModel) -> None:
        """Pedal at 0.05 is below V_MIN=0.1, should produce 0 torque."""
        torque = model.lvcu_torque_command(0.05, 1000.0, 100.0)
        assert torque == 0.0

    def test_pedal_above_deadzone_gives_full(self, model: PowertrainModel) -> None:
        """Pedal at 0.95 is above V_MAX=0.9, should produce full torque."""
        torque = model.lvcu_torque_command(0.95, 1000.0, 100.0)
        assert torque == pytest.approx(85.0)

    def test_zero_pedal_gives_zero(self, model: PowertrainModel) -> None:
        torque = model.lvcu_torque_command(0.0, 1000.0, 100.0)
        assert torque == 0.0

    def test_zero_current_gives_zero(self, model: PowertrainModel) -> None:
        """If BMS current limit is 0, no torque can be produced."""
        torque = model.lvcu_torque_command(1.0, 1000.0, 0.0)
        assert torque == 0.0

    def test_overspeed_caps_torque(self, model: PowertrainModel) -> None:
        """At >= 6000 RPM, torque ceiling drops to 30 Nm."""
        torque = model.lvcu_torque_command(1.0, 6500.0, 100.0)
        assert torque == pytest.approx(30.0)

    def test_power_limit_at_low_rpm_uses_floor(self, model: PowertrainModel) -> None:
        """Below ~2141 RPM, the omega floor dominates.

        At 1000 RPM: max(23.04, 1000*0.1076) = max(23.04, 107.6) = 107.6
        So floor is NOT active at 1000 RPM. Try 200 RPM:
        max(23.04, 200*0.1076) = max(23.04, 21.52) = 23.04 (floor active)
        Power ceiling: 420 * 100 / 23.04 = 1822.9 Nm -> clamped to 85 Nm
        """
        torque = model.lvcu_torque_command(1.0, 200.0, 100.0)
        assert torque == pytest.approx(85.0)  # inverter limit still binds

    def test_power_limit_becomes_binding_with_low_current(self, model: PowertrainModel) -> None:
        """At 45A and 2400 RPM, power limit should be near inverter limit.

        Power ceiling: 420 * 45 / max(23.04, 2400 * 0.1076)
                     = 18900 / max(23.04, 258.24)
                     = 18900 / 258.24
                     ≈ 73.2 Nm  (below 85 Nm — power limit binds)
        """
        torque = model.lvcu_torque_command(1.0, 2400.0, 45.0)
        expected = 420.0 * 45.0 / (2400.0 * 0.1076)
        assert torque == pytest.approx(expected, rel=0.01)
        assert torque < 85.0

    def test_lvcu_limit_caps_before_inverter(self, model: PowertrainModel) -> None:
        """With very high current, LVCU 150 Nm limit should bind before
        the power formula would give more, but inverter 85 Nm still wins.

        At 500A (unrealistic, but tests the min chain):
        Power: 420*500/max(23.04, 1000*0.1076) = 210000/107.6 = 1951 Nm
        LVCU limit: 150 Nm
        Inverter limit: 85 Nm
        Result: 85 Nm
        """
        torque = model.lvcu_torque_command(1.0, 1000.0, 500.0)
        assert torque == pytest.approx(85.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_powertrain_model.py::TestLVCUTorqueCommand -v`

Expected: All fail with `AttributeError: 'PowertrainModel' object has no attribute 'lvcu_torque_command'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_powertrain_model.py
git commit -m "test: add failing tests for LVCU torque command model"
```

---

### Task 3: Implement lvcu_torque_command in PowertrainModel

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain_model.py:121` (insert after `max_motor_torque`)

- [ ] **Step 1: Add the lvcu_torque_command method**

Insert after line 161 (after `max_motor_torque` method, before the `# Torque and force through drivetrain` section) in `src/fsae_sim/vehicle/powertrain_model.py`:

```python
    def lvcu_torque_command(
        self, pedal_pct: float, motor_rpm: float, bms_current_limit_a: float,
    ) -> float:
        """Motor torque command replicating the real LVCU firmware.

        Faithfully implements the torque command chain from LVCU Code.txt:
        pedal -> tmap_lut (dead zone remap) -> torque_lut (power-limited
        ceiling) -> inverter clamp.

        The power-limited ceiling prevents the motor from drawing more
        current than the BMS allows, expressed as a torque limit that
        decreases with RPM (constant-power hyperbola).

        Args:
            pedal_pct: Raw pedal position in [0.0, 1.0].
            motor_rpm: Motor shaft speed in RPM.
            bms_current_limit_a: BMS discharge current limit in A (from
                temp + SOC taper).

        Returns:
            Commanded motor torque in Nm (>= 0).
        """
        cfg = self.config

        # 1. tmap_lut: dead zone remap [V_MIN, V_MAX] -> [0, 1]
        pedal_clamped = max(cfg.lvcu_pedal_deadzone_low,
                           min(pedal_pct, cfg.lvcu_pedal_deadzone_high))
        pedal_remapped = (
            (pedal_clamped - cfg.lvcu_pedal_deadzone_low)
            / (cfg.lvcu_pedal_deadzone_high - cfg.lvcu_pedal_deadzone_low)
        )

        # 2. torque_lut: power-limited torque ceiling
        omega_term = max(cfg.lvcu_omega_floor, motor_rpm * cfg.lvcu_rpm_scale)
        power_ceiling_nm = cfg.lvcu_power_constant * bms_current_limit_a / omega_term

        # LVCU torque limit (software cap)
        torque_ceiling_nm = min(cfg.torque_limit_lvcu_nm, power_ceiling_nm)

        # Overspeed override
        if motor_rpm >= cfg.lvcu_overspeed_rpm:
            torque_ceiling_nm = cfg.lvcu_overspeed_torque_nm

        # Inverter hardware limit (independent clamp)
        torque_ceiling_nm = min(torque_ceiling_nm, cfg.torque_limit_inverter_nm)

        # 3. Final command: remapped pedal × clamped ceiling
        return pedal_remapped * torque_ceiling_nm
```

- [ ] **Step 2: Run LVCU tests to verify they pass**

Run: `pytest tests/test_powertrain_model.py::TestLVCUTorqueCommand -v`

Expected: All 11 tests PASS

- [ ] **Step 3: Run all powertrain tests to check for regressions**

Run: `pytest tests/test_powertrain_model.py -v`

Expected: All existing tests still pass (lvcu_torque_command is additive, doesn't change existing methods)

- [ ] **Step 4: Commit**

```bash
git add src/fsae_sim/vehicle/powertrain_model.py
git commit -m "feat: implement lvcu_torque_command replicating real LVCU firmware"
```

---

### Task 4: Rewire SimulationEngine to Use LVCU Torque Command

**Files:**
- Modify: `src/fsae_sim/sim/engine.py`

- [ ] **Step 1: Remove speed cap and rewire force-based resolution**

In `src/fsae_sim/sim/engine.py`, replace the force-based resolution block (lines 208-261) with:

```python
                else:
                    # --- Force-based resolution mode ---
                    # Used for synthetic strategies (coast, threshold braking, etc.)

                    # 2. Speed limit from pre-computed envelope
                    corner_limit = float(v_max[seg_idx])

                    # 2b. BMS current limit for LVCU torque command
                    bms_current_limit = self.battery_model.max_discharge_current(temp, soc)
                    motor_rpm = self.powertrain.motor_rpm_from_speed(speed)

                    # 3. Compute forces based on driver action
                    if cmd.action == ControlAction.THROTTLE:
                        motor_torque = self.powertrain.lvcu_torque_command(
                            cmd.throttle_pct, motor_rpm, bms_current_limit,
                        )
                        drive_f = self.powertrain.wheel_force(motor_torque)
                        drive_f = min(drive_f, self.dynamics.max_traction_force(speed))
                        regen_f = 0.0
                    elif cmd.action == ControlAction.BRAKE:
                        drive_f = 0.0
                        regen_f = self.powertrain.regen_force(cmd.brake_pct, speed)
                        max_brake = self.dynamics.max_braking_force(speed)
                        if abs(regen_f) > max_brake:
                            regen_f = -max_brake
                    else:  # COAST
                        drive_f = 0.0
                        regen_f = 0.0

                    # 4. Resistive forces
                    resist_f = self.dynamics.total_resistance(speed, segment.grade, segment.curvature)

                    # 5. Net force and speed resolution
                    net_force = drive_f + regen_f - resist_f

                    exit_speed, seg_time = self.dynamics.resolve_exit_speed(
                        speed, segment.length_m, net_force, corner_limit,
                    )
                    exit_speed = max(exit_speed, self._MIN_SPEED_MS)

                    avg_speed = (speed + exit_speed) / 2.0
                    motor_rpm = self.powertrain.motor_rpm_from_speed(avg_speed)

                    # Recompute torque at resolved avg speed for accurate power calc
                    if cmd.action == ControlAction.THROTTLE:
                        motor_torque = self.powertrain.lvcu_torque_command(
                            cmd.throttle_pct, motor_rpm, bms_current_limit,
                        )
                    elif cmd.action == ControlAction.BRAKE:
                        max_torque = self.powertrain.max_motor_torque(motor_rpm)
                        motor_torque = -cmd.brake_pct * max_torque
                    else:
                        motor_torque = 0.0
```

- [ ] **Step 2: Remove the after-the-fact BMS current clamp**

Replace lines 270-275 (the current clamp block):

```python
                # 8. Enforce BMS current limit
                max_current = self.battery_model.max_discharge_current(temp, soc)
                if pack_current > max_current:
                    pack_current = max_current
                    # Recalculate power with limited current
                    elec_power = pack_current * pack_voltage
```

With a comment explaining why it's no longer needed:

```python
                # BMS current limit is now enforced upstream via
                # lvcu_torque_command — no after-the-fact clamp needed.
```

- [ ] **Step 3: Remove CalibratedStrategy import (no longer needed in engine)**

At the top of `engine.py`, change:

```python
from fsae_sim.driver.strategies import CalibratedStrategy, ReplayStrategy
```

To:

```python
from fsae_sim.driver.strategies import ReplayStrategy
```

- [ ] **Step 4: Remove unused `m_effective_fallback` line**

Delete line 114:

```python
        # Cache effective mass for maintenance torque calculation
        self.m_effective_fallback = self.dynamics.m_effective
```

- [ ] **Step 5: Run engine tests**

Run: `pytest tests/test_engine.py -v`

Expected: All pass (engine tests use simple strategies, not CalibratedStrategy speed targets)

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`

Expected: All pass. If `test_calibrated_strategy.py` fails due to speed target removal (Task 5), that's expected and will be fixed next.

- [ ] **Step 7: Commit**

```bash
git add src/fsae_sim/sim/engine.py
git commit -m "feat: rewire engine to use LVCU torque command upstream of force resolution

Removes speed cap bandaid and after-the-fact BMS current clamp.
Torque is now limited by the real LVCU firmware logic before
force/speed resolution."
```

---

### Task 5: Remove Speed Cap From CalibratedStrategy

**Files:**
- Modify: `src/fsae_sim/driver/strategies.py`
- Modify: `tests/test_calibrated_strategy.py` (if tests reference speed targets)

- [ ] **Step 1: Remove speed target fields and methods from CalibratedStrategy**

In `src/fsae_sim/driver/strategies.py`:

Remove `segment_speed_targets_ms` parameter from `__init__` (line 285). Remove the `_segment_speed_targets_ms` field and its docstring (lines 291-295). Remove the `speed_target_ms` method (lines 312-322). Result for `__init__`:

```python
    def __init__(
        self,
        zones: list[DriverZone],
        num_segments: int,
        name: str = "calibrated",
    ) -> None:
        self.name = name
        self._zones = list(zones)
        self._num_segments = num_segments

        # Build flat lookup: segment_idx -> (action, intensity, max_speed_ms)
        self._segment_actions: list[tuple[ControlAction, float, float]] = [
            (ControlAction.COAST, 0.0, 0.0)
        ] * num_segments
        for zone in zones:
            for seg_idx in range(zone.segment_start, zone.segment_end + 1):
                if 0 <= seg_idx < num_segments:
                    self._segment_actions[seg_idx] = (
                        zone.action, zone.intensity, zone.max_speed_ms,
                    )
```

- [ ] **Step 2: Remove speed targets from with_zone_override**

Change `with_zone_override` return (around line 397-400) from:

```python
        return CalibratedStrategy(
            new_zones, self._num_segments, name=self.name,
            segment_speed_targets_ms=self._segment_speed_targets_ms,
        )
```

To:

```python
        return CalibratedStrategy(new_zones, self._num_segments, name=self.name)
```

- [ ] **Step 3: Remove speed target extraction from from_telemetry**

In `from_telemetry` (around lines 432-443), replace:

```python
        # Extract per-segment speed targets from telemetry for speed-aware
        # throttle modulation.  Uses the mean observed speed at each segment.
        segment_speed_targets_ms = None
        if "mean_speed_kmh" in seg_actions.columns:
            segment_speed_targets_ms = (
                seg_actions["mean_speed_kmh"].values / 3.6
            ).copy()

        return cls(
            zones, track.num_segments, name=name,
            segment_speed_targets_ms=segment_speed_targets_ms,
        )
```

With:

```python
        return cls(zones, track.num_segments, name=name)
```

- [ ] **Step 4: Fix any tests referencing speed targets**

Run: `pytest tests/test_calibrated_strategy.py -v`

If any tests reference `speed_target_ms` or `segment_speed_targets_ms`, remove those tests. They tested bandaid functionality.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`

Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/fsae_sim/driver/strategies.py tests/test_calibrated_strategy.py
git commit -m "refactor: remove speed cap from CalibratedStrategy

Speed regulation now emerges from LVCU torque limiting physics,
not telemetry-derived speed targets."
```

---

### Task 6: Revert p90 Speed Percentile in Telemetry Analysis

**Files:**
- Modify: `src/fsae_sim/analysis/telemetry_analysis.py`

- [ ] **Step 1: Revert p90 to mean**

In `src/fsae_sim/analysis/telemetry_analysis.py`, around line 340-342, change:

```python
        # Use 90th percentile speed across laps — represents the speed
        # on a representative hot lap, not dragged down by slow/warm-up laps.
        mean_speed = float(np.percentile(speed_matrix[:, i], 90))
```

To:

```python
        mean_speed = float(np.mean(speed_matrix[:, i]))
```

- [ ] **Step 2: Run telemetry tests**

Run: `pytest tests/test_telemetry_analysis.py -v`

Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/fsae_sim/analysis/telemetry_analysis.py
git commit -m "revert: restore mean speed aggregation in telemetry analysis

The p90 percentile was added to support speed cap targets, which
have been removed in favor of LVCU torque limiting."
```

---

### Task 7: Integration Validation

**Files:**
- Modify: `scripts/validate_driver_model.py` (if needed to remove speed-target-related output)

- [ ] **Step 1: Run the validation script**

Run: `python scripts/validate_driver_model.py`

This runs the 22-lap endurance sim against cleaned telemetry. Record the output metrics:
- Driving time (target: <5% error)
- Energy (target: <5% error)
- SOC consumed (target: <5% error)
- Temperature (target: <5% error)

- [ ] **Step 2: Analyze results**

Compare against the pre-LVCU results from `docs/simulation_alignment_log.md`:
- If time accuracy worsened: this is expected and honest. The sim is no longer being force-fed real speeds. Document the new baseline.
- If energy accuracy improved: the LVCU power limiting is correctly reducing energy draw in later laps.
- If any metric is wildly wrong (>20% error): investigate root cause before proceeding.

- [ ] **Step 3: Update simulation alignment log**

Add a new iteration entry to `docs/simulation_alignment_log.md` documenting:
- What changed (LVCU torque model, speed cap removed)
- New metric values
- What the remaining gaps point to (driver model calibration, cornering, etc.)

- [ ] **Step 4: Commit**

```bash
git add docs/simulation_alignment_log.md scripts/validate_driver_model.py
git commit -m "docs: add LVCU torque model validation results to alignment log"
```
