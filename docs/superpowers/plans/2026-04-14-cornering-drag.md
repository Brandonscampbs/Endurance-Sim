# Cornering Drag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tire cornering drag to the vehicle dynamics force model so the simulation accurately accounts for energy lost through tire slip angles during cornering, closing the 37% energy error and 12% time error vs Michigan 2025 telemetry.

**Architecture:** Add a `cornering_drag(speed_ms, curvature)` method to `VehicleDynamics` with two code paths: a Pacejka-based per-tire calculation (when tire_model + load_transfer are available) and an analytical small-angle fallback (legacy mode). Update `total_resistance()` to accept a `curvature` parameter (default 0.0 for backward compat). Update `SimulationEngine` to pass `segment.curvature` at both call sites.

**Tech Stack:** Python, scipy.optimize.brentq/minimize_scalar, Pacejka tire model, pytest

**Spec:** `docs/superpowers/specs/2026-04-14-cornering-drag-design.md`

---

## File Structure

| File | Role | Change |
|---|---|---|
| `src/fsae_sim/vehicle/dynamics.py` | Vehicle force model | Add `cornering_drag()`, `_find_slip_angle()`, update `total_resistance()` |
| `src/fsae_sim/sim/engine.py` | Simulation loop | Pass `segment.curvature` to `total_resistance()` at 2 call sites |
| `tests/test_dynamics.py` | Dynamics unit tests | Add `TestCorneringDrag` and `TestCorneringDragPacejka` classes |

---

### Task 1: Write failing tests for cornering_drag — legacy (analytical) mode

**Files:**
- Modify: `tests/test_dynamics.py` (append after line 236)

These tests use the existing `dynamics` fixture which has no tire model, exercising the analytical fallback path.

- [ ] **Step 1: Add TestCorneringDrag test class**

Append this to the end of `tests/test_dynamics.py`:

```python
class TestCorneringDrag:
    """Test cornering_drag() in legacy mode (no tire model, analytical fallback)."""

    def test_zero_curvature_returns_zero(self, dynamics):
        assert dynamics.cornering_drag(11.0, 0.0) == 0.0

    def test_near_zero_curvature_returns_zero(self, dynamics):
        assert dynamics.cornering_drag(11.0, 1e-8) == 0.0

    def test_low_speed_returns_zero(self, dynamics):
        assert dynamics.cornering_drag(0.3, 0.05) == 0.0

    def test_positive_in_corner(self, dynamics):
        """Cornering at 40 km/h through κ=0.02 should produce positive drag."""
        drag = dynamics.cornering_drag(11.1, 0.02)
        assert drag > 0.0

    def test_increases_with_curvature(self, dynamics):
        drag_gentle = dynamics.cornering_drag(11.1, 0.01)
        drag_tight = dynamics.cornering_drag(11.1, 0.05)
        assert drag_tight > drag_gentle

    def test_increases_with_speed(self, dynamics):
        drag_slow = dynamics.cornering_drag(5.0, 0.02)
        drag_fast = dynamics.cornering_drag(15.0, 0.02)
        assert drag_fast > drag_slow

    def test_analytical_known_value(self, dynamics):
        """Hand calculation for analytical fallback.

        mass=278, v=11.1 m/s, κ=0.02
        F_lat = 278 * 11.1^2 * 0.02 = 684.8 N
        C_α_total = 278 * 9.81 * 1.5 / 0.15 = 27,271 N/rad
        drag = 684.8^2 / 27,271 = 17.2 N
        """
        drag = dynamics.cornering_drag(11.1, 0.02)
        assert 10.0 < drag < 30.0

    def test_total_resistance_with_curvature(self, dynamics):
        """total_resistance with curvature > without."""
        r_straight = dynamics.total_resistance(11.1, 0.0, 0.0)
        r_corner = dynamics.total_resistance(11.1, 0.0, 0.05)
        assert r_corner > r_straight

    def test_total_resistance_backward_compat(self, dynamics):
        """Calling with 2 args still works (curvature defaults to 0)."""
        r = dynamics.total_resistance(11.1, 0.0)
        assert r > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dynamics.py::TestCorneringDrag -v`
Expected: FAIL — `AttributeError: 'VehicleDynamics' object has no attribute 'cornering_drag'`

---

### Task 2: Implement cornering_drag — analytical fallback path

**Files:**
- Modify: `src/fsae_sim/vehicle/dynamics.py:92-98` (total_resistance method + new method above it)

- [ ] **Step 1: Add cornering_drag method to VehicleDynamics**

Insert after the `grade_force` method (after line 91) and before `total_resistance` (line 92) in `dynamics.py`:

```python
    def cornering_drag(self, speed_ms: float, curvature: float) -> float:
        """Drag force (N) from tire slip angles during cornering.

        When the car corners, tires operate at slip angles that create a
        longitudinal drag component. Uses the Pacejka tire model when
        available, otherwise falls back to a small-angle analytical
        approximation.

        Args:
            speed_ms: Vehicle speed (m/s).
            curvature: Path curvature (1/m). 0 = straight.

        Returns:
            Cornering drag force (N), always >= 0.
        """
        if abs(curvature) < 1e-6 or speed_ms < 0.5:
            return 0.0

        # Total lateral force needed for the turn
        f_lat_total = self.vehicle.mass_kg * speed_ms ** 2 * abs(curvature)

        if (
            self.tire_model is not None
            and self.load_transfer is not None
        ):
            return self._cornering_drag_pacejka(speed_ms, curvature, f_lat_total)

        return self._cornering_drag_analytical(f_lat_total)

    def _cornering_drag_analytical(self, f_lat_total: float) -> float:
        """Analytical cornering drag using small-angle approximation.

        Assumes linear tire: Fy = C_α * α, so α = Fy/C_α,
        and drag = Fy * sin(α) ≈ Fy * α = Fy²/C_α.

        C_α estimated from peak grip (mu=1.5) and typical FSAE peak
        slip angle (~0.15 rad).
        """
        mu_peak = 1.5
        alpha_peak = 0.15  # rad, typical FSAE tire
        c_alpha_total = (
            self.vehicle.mass_kg * self.GRAVITY_M_S2 * mu_peak / alpha_peak
        )
        return f_lat_total ** 2 / c_alpha_total

    def _cornering_drag_pacejka(
        self, speed_ms: float, curvature: float, f_lat_total: float,
    ) -> float:
        """Pacejka-based cornering drag — stub, implemented in Task 4."""
        return self._cornering_drag_analytical(f_lat_total)
```

Note: The `_cornering_drag_pacejka` stub delegates to the analytical path so the code doesn't crash if a Pacejka-equipped VehicleDynamics is used before Task 4. Task 4 replaces this stub with the real implementation.

- [ ] **Step 2: Update total_resistance signature to accept curvature**

Replace the existing `total_resistance` method (lines 92-98) with:

```python
    def total_resistance(
        self, speed_ms: float, grade: float = 0.0, curvature: float = 0.0,
    ) -> float:
        """Sum of all resistance forces (N) at given speed, grade, and curvature."""
        return (
            self.drag_force(speed_ms)
            + self.rolling_resistance_force(speed_ms)
            + self.grade_force(grade)
            + self.cornering_drag(speed_ms, curvature)
        )
```

- [ ] **Step 3: Run tests to verify legacy tests pass**

Run: `pytest tests/test_dynamics.py::TestCorneringDrag -v`
Expected: All 9 tests PASS

- [ ] **Step 4: Run full existing test suite to verify backward compat**

Run: `pytest tests/test_dynamics.py -v`
Expected: All existing tests PASS (curvature defaults to 0.0)

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/vehicle/dynamics.py tests/test_dynamics.py
git commit -m "feat: add cornering_drag with analytical fallback to VehicleDynamics

Adds cornering_drag(speed_ms, curvature) method that computes tire slip
angle drag during cornering. Analytical fallback uses small-angle
approximation (Fy^2/C_alpha). Updates total_resistance() to accept
curvature parameter (default 0.0 preserves backward compat).

Pacejka tire model path is stubbed — next commit implements it."
```

---

### Task 3: Write failing tests for cornering_drag — Pacejka mode

**Files:**
- Modify: `tests/test_dynamics.py` (append after TestCorneringDrag)

These tests create a VehicleDynamics with mocked tire_model and load_transfer to exercise the Pacejka code path.

- [ ] **Step 1: Add TestCorneringDragPacejka test class**

Append to the end of `tests/test_dynamics.py`:

```python
class TestCorneringDragPacejka:
    """Test cornering_drag() with Pacejka tire model (mocked)."""

    @pytest.fixture
    def mock_tire(self):
        """Mock tire model with simple linear response.

        lateral_force(alpha, Fz) = -Fz * 15.0 * alpha  (negative per Pacejka convention)
        peak_lateral_force(Fz) = Fz * 1.5  (mu = 1.5)
        """
        tire = MagicMock()
        tire.lateral_force.side_effect = (
            lambda alpha, fz, camber=0.0: -fz * 15.0 * alpha
        )
        tire.peak_lateral_force.side_effect = (
            lambda fz, camber=0.0: fz * 1.5
        )
        return tire

    @pytest.fixture
    def mock_lt(self, ct16ev_params):
        """Mock load transfer that returns equal loads."""
        lt = MagicMock()
        total_weight = ct16ev_params.mass_kg * 9.81
        per_tire = total_weight / 4.0
        lt.tire_loads.return_value = (per_tire, per_tire, per_tire, per_tire)
        return lt

    @pytest.fixture
    def dynamics_pacejka(self, ct16ev_params, mock_tire, mock_lt):
        return VehicleDynamics(
            ct16ev_params, tire_model=mock_tire, load_transfer=mock_lt,
        )

    def test_positive_drag_in_corner(self, dynamics_pacejka):
        drag = dynamics_pacejka.cornering_drag(11.1, 0.02)
        assert drag > 0.0

    def test_zero_on_straight(self, dynamics_pacejka):
        assert dynamics_pacejka.cornering_drag(11.1, 0.0) == 0.0

    def test_increases_with_curvature(self, dynamics_pacejka):
        drag_gentle = dynamics_pacejka.cornering_drag(11.1, 0.01)
        drag_tight = dynamics_pacejka.cornering_drag(11.1, 0.05)
        assert drag_tight > drag_gentle

    def test_calls_load_transfer(self, dynamics_pacejka, mock_lt):
        dynamics_pacejka.cornering_drag(11.1, 0.02)
        mock_lt.tire_loads.assert_called_once()
        # Should pass lateral g, not longitudinal
        args = mock_lt.tire_loads.call_args
        speed_arg = args[0][0]
        lat_g_arg = args[0][1]
        long_g_arg = args[0][2]
        assert abs(speed_arg - 11.1) < 0.01
        assert lat_g_arg > 0  # positive lateral g
        assert long_g_arg == 0.0  # no longitudinal accel during steady cornering

    def test_saturated_tire_no_crash(self, ct16ev_params):
        """When demanded Fy exceeds peak, should not crash or return NaN."""
        tire = MagicMock()
        # Very weak tire: peak at 100 N, but we demand much more
        tire.lateral_force.side_effect = (
            lambda alpha, fz, camber=0.0: -min(100.0, fz * 5.0 * alpha)
        )
        tire.peak_lateral_force.side_effect = lambda fz, camber=0.0: 100.0
        lt = MagicMock()
        lt.tire_loads.return_value = (200, 200, 200, 200)
        dyn = VehicleDynamics(ct16ev_params, tire_model=tire, load_transfer=lt)
        drag = dyn.cornering_drag(15.0, 0.10)  # high speed, tight corner
        assert drag > 0.0
        assert math.isfinite(drag)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dynamics.py::TestCorneringDragPacejka -v`
Expected: `test_calls_load_transfer` FAILS because the Task 2 stub delegates to `_cornering_drag_analytical` without calling `load_transfer.tire_loads()`. Other tests may pass since the analytical fallback produces positive values, but `test_saturated_tire_no_crash` behavior will differ once the real Pacejka path is in place.

---

### Task 4: Implement cornering_drag — Pacejka path

**Files:**
- Modify: `src/fsae_sim/vehicle/dynamics.py`

- [ ] **Step 1: Add scipy imports to dynamics.py**

At the top of `dynamics.py`, after the existing `import math` line (line 9), add:

```python
from scipy.optimize import brentq, minimize_scalar
```

- [ ] **Step 2: Add _find_slip_angle helper method**

Add this method to the `VehicleDynamics` class, after the `_cornering_drag_analytical` method:

```python
    def _find_slip_angle(
        self, f_lat_needed: float, normal_load: float,
    ) -> float:
        """Find slip angle (rad) that produces the needed lateral force.

        Uses brentq root-finding on the Pacejka lateral_force function,
        which is monotonic below peak slip angle. If demanded force
        exceeds the tire's peak, returns the peak slip angle (tire
        saturated).

        Args:
            f_lat_needed: Required lateral force magnitude (N).
            normal_load: Tire normal load (N).

        Returns:
            Slip angle in radians (always >= 0).
        """
        if normal_load < 1.0 or f_lat_needed < 1.0:
            return 0.0

        peak_fy = self.tire_model.peak_lateral_force(normal_load)
        if f_lat_needed >= peak_fy:
            # Tire is saturated — find the slip angle at peak Fy
            result = minimize_scalar(
                lambda a: -abs(
                    self.tire_model.lateral_force(a, normal_load)
                ),
                bounds=(0.001, math.pi / 4.0),
                method="bounded",
            )
            return abs(result.x)

        # Fy is monotonic below peak — brentq finds the unique root
        return brentq(
            lambda a: abs(
                self.tire_model.lateral_force(a, normal_load)
            ) - f_lat_needed,
            0.0,
            math.pi / 4.0,
            xtol=1e-4,
        )
```

- [ ] **Step 3: Replace _cornering_drag_pacejka stub with real implementation**

Replace the `_cornering_drag_pacejka` placeholder with:

```python
    def _cornering_drag_pacejka(
        self, speed_ms: float, curvature: float, f_lat_total: float,
    ) -> float:
        """Cornering drag using Pacejka tire model with load transfer.

        Computes per-tire slip angles from lateral force demand distributed
        by normal load, then sums the drag component (Fy * sin(alpha))
        across all four tires.
        """
        # Lateral acceleration for load transfer
        a_lat_g = speed_ms ** 2 * abs(curvature) / self.GRAVITY_M_S2

        # Per-tire normal loads under cornering
        fl, fr, rl, rr = self.load_transfer.tire_loads(
            speed_ms, a_lat_g, 0.0,
        )
        loads = [fl, fr, rl, rr]
        total_load = sum(loads)
        if total_load < 1.0:
            return 0.0

        total_drag = 0.0
        for fz in loads:
            if fz < 1.0:
                continue
            # This tire's share of lateral force, proportional to load
            f_lat_tire = f_lat_total * (fz / total_load)
            # Find slip angle that produces this lateral force
            alpha = self._find_slip_angle(f_lat_tire, fz)
            # Drag component: lateral force projected onto velocity direction
            fy_actual = abs(
                self.tire_model.lateral_force(alpha, fz)
            )
            total_drag += fy_actual * math.sin(alpha)

        return total_drag
```

- [ ] **Step 4: Run Pacejka tests**

Run: `pytest tests/test_dynamics.py::TestCorneringDragPacejka -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run all dynamics tests**

Run: `pytest tests/test_dynamics.py -v`
Expected: All tests PASS (legacy + Pacejka + backward compat)

- [ ] **Step 6: Commit**

```bash
git add src/fsae_sim/vehicle/dynamics.py tests/test_dynamics.py
git commit -m "feat: implement Pacejka-based cornering drag with per-tire slip angles

Uses load transfer model for per-tire normal loads under cornering,
inverts Pacejka Fy to find slip angles via brentq, computes drag as
Fy*sin(alpha) per tire. Handles saturated tires by clamping to peak
slip angle."
```

---

### Task 5: Write failing test for SimulationEngine curvature passing

**Files:**
- Modify: `tests/test_engine.py` (append after TestTractionClamping)

- [ ] **Step 1: Add test verifying engine passes curvature to total_resistance**

Append to the end of `tests/test_engine.py`:

```python
class TestCorneringDragIntegration:
    """Verify engine passes curvature to total_resistance."""

    def test_corner_segments_have_higher_resistance(
        self, vehicle_config, battery_model,
    ):
        """A track with corners should produce higher resistance than all-straight."""
        straight_track = Track(
            name="straight",
            segments=[
                Segment(i, i * 50.0, 50.0, curvature=0.0, grade=0.0)
                for i in range(10)
            ],
        )
        curvy_track = Track(
            name="curvy",
            segments=[
                Segment(i, i * 50.0, 50.0, curvature=0.04, grade=0.0)
                for i in range(10)
            ],
        )
        strategy = FullThrottleStrategy()
        engine_straight = SimulationEngine(
            vehicle_config, straight_track, strategy, battery_model,
        )
        engine_curvy = SimulationEngine(
            vehicle_config, curvy_track, strategy, battery_model,
        )
        r_straight = engine_straight.run(num_laps=1)
        r_curvy = engine_curvy.run(num_laps=1)

        # Curvy track should consume more energy (higher resistance)
        assert r_curvy.total_energy_kwh > r_straight.total_energy_kwh
        # Curvy track resistance column should show higher values
        assert (
            r_curvy.states["resistance_force_n"].mean()
            > r_straight.states["resistance_force_n"].mean()
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::TestCorneringDragIntegration -v`
Expected: FAIL — engine doesn't pass curvature yet, so resistance is identical.

---

### Task 6: Update SimulationEngine to pass curvature

**Files:**
- Modify: `src/fsae_sim/sim/engine.py:172,203`

- [ ] **Step 1: Update replay mode call site (line 172)**

In `engine.py`, replace line 172:

```python
                    resist_f = self.dynamics.total_resistance(avg_speed, segment.grade)
```

with:

```python
                    resist_f = self.dynamics.total_resistance(avg_speed, segment.grade, segment.curvature)
```

- [ ] **Step 2: Update force-based mode call site (line 203)**

In `engine.py`, replace line 203:

```python
                    resist_f = self.dynamics.total_resistance(speed, segment.grade)
```

with:

```python
                    resist_f = self.dynamics.total_resistance(speed, segment.grade, segment.curvature)
```

- [ ] **Step 3: Run integration test**

Run: `pytest tests/test_engine.py::TestCorneringDragIntegration -v`
Expected: PASS

- [ ] **Step 4: Run full engine test suite**

Run: `pytest tests/test_engine.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/sim/engine.py tests/test_engine.py
git commit -m "feat: pass segment curvature to total_resistance in SimulationEngine

Both replay and force-based code paths now pass segment.curvature to
total_resistance(), enabling cornering drag in all simulation modes."
```

---

### Task 7: Full validation and iteration

**Files:**
- Run: `scripts/validate_driver_model.py`

- [ ] **Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (existing 383 + new cornering drag tests)

- [ ] **Step 2: Run validation script**

Run: `python scripts/validate_driver_model.py`
Expected: Observe error metrics. Target:
- Driving time error: < 5% (stretch: < 3%)
- Energy consumed error: < 5% (stretch: < 3%)
- Mean pack current error: < 8% (stretch: < 5%)

- [ ] **Step 3: Analyze results and iterate**

If errors exceed targets:
1. Check cornering drag magnitude — is it in the right ballpark (~100-200 N average)?
2. Compare per-segment resistance to telemetry-derived resistance
3. Investigate secondary physics if cornering drag alone is insufficient
4. Document any remaining error with physics justification

Do NOT apply bandaid fixes (fudge factors, clamping, speed governors).

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: cornering drag complete — validation results

[Include actual validation numbers in commit message body]"
```
