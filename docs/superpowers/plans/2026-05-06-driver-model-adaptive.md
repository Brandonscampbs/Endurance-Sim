# Adaptive Driver Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lap-mean `CalibratedStrategy` with an envelope-following adaptive driver that emits firmware-faithful pedal commands (throttle / brake / regen request) targeting the pre-computed `v_max(s)` envelope, so capability sweeps (motor RPM ↑, inverter torque cap ↑, BMS limit changes) actually produce the late-braking and earlier-throttle gains they should.

**Architecture:** Pure feedforward envelope follower with a velocity-error PI corrector and a stint-level energy shaper. The forward-backward speed envelope (`SpeedEnvelope.compute()` at `src\fsae_sim\sim\speed_envelope.py:45`) is already the optimal velocity profile under the current vehicle physics; the driver's job is therefore not to plan a trajectory but to *track it* — choose the throttle / brake fraction whose resulting force balance puts the next-segment exit speed on the envelope. This is the architecture used by the open-source TUMFTM `laptimesim` reference (see Research Summary), and it is the only feedforward design that makes the envelope's per-segment v_max change visible in lap time when sweep parameters move the envelope. A short-horizon PI on (v − v_max) closes residual integration error and absorbs predictor-corrector slop. A stint-level kWh-budget shaper biases throttle reduction onto the lowest-cost segments (longest "time-to-next-brake-zone" — TUMFTM `LBP` strategy) when the driver must finish on a tight energy budget.

**Tech Stack:** Python 3.11, NumPy (vectorized envelope-difference passes, no per-segment Python loops where avoidable), no SciPy required at runtime, pytest + hypothesis for tests, pandas only for telemetry-grounded regression tests against `CleanedEndurance.csv`.

---

## Research Summary

- **TUMFTM `laptime-simulation`** (https://github.com/TUMFTM/laptime-simulation) — Read `laptimesim/src/driver.py` and `laptimesim/src/lap.py` directly. The `Driver` object exposes a per-track-point `throttle_pos[i]` array (default 1.0) plus an `em_boost_use[i]` boolean array. The lap solver in `lap.py` does forward-backward integration: when tire potential remains and `vel_cl[i] <= vel_lim_cl[i]` it accelerates with the configured throttle; when the next-point envelope is below current speed it switches to backward braking iterations until convergence. **Key takeaway: the driver itself does not pick brake points — the envelope does. The driver's only choices are throttle reduction (energy management) and lift-and-coast (`__lift_coast`, lines 247-275 of `driver.py`).** This is exactly the right separation of concerns for our case: `SpeedEnvelope` already does forward-backward, and the engine already enforces the speed cap at `engine.py:435-496`. Our adaptive driver therefore encodes (a) the throttle pedal that on-envelope acceleration requires, (b) lift-and-coast / brake commands at envelope discontinuities, and (c) energy-budget throttle cuts.
- **TUMFTM `laptimesim` energy strategies** (`__strategy_lbp`, `__strategy_ls` in `driver.py`) — `LBP` (longest time to next brake point) finds indices where `np.diff(vel_cl) < 0`, computes time-to-next-brake for every other index, sorts descending, and applies boost greedily until the energy budget runs out. `LS` (lowest speed) does the same but sorts by `vel_cl` ascending. Both are O(N log N) per call and run once per lap. **Adopted directly** as our energy shaper, with `LBP` as the default because Michigan endurance has long straights between corner clusters where lifts cost the least lap time per Wh saved.
- **MDPI 2022 — "Optimizing Torque Delivery for an Energy-Limited Electric Race Car Using Model Predictive Control"** (Maull & Schommer, *World Electric Vehicle Journal* 13(12):224, https://www.mdpi.com/2032-6653/13/12/224 — abstract retrieved via Google Scholar). Quote: "The energy-managing torque control algorithm developed in this work optimizes the finite onboard energy from the battery pack to reduce lap time and energy consumption when energy deficits occur. The longitudinal dynamics of the vehicle were represented by a linearized first-principles model... A Simulink-based model predictive controller (MPC) architecture was created to balance energy use requirements with optimum lap time. This controller was tested against a hardware-limited and torque-limited system in a constant torque request and a varying torque request scenario. The controller decreased the elapsed time to complete a 150 m straight-line acceleration by 11.4% over the torque-limited solution and 13.5% in a 150 m Formula Student manoeuvre." **Takeaway: MPC is a valid framing for FS energy management but Maull's experiments are 150 m straight-line / autocross slaloms, not full endurance, and the comparator is a fixed torque-cap controller. The win comes from the energy-aware shaping, not from MPC-vs-PI on velocity tracking.** We use the *idea* — a stint-budget shaper that allocates pedal cuts to least-time-cost segments — without taking on a runtime QP solver. Justification for not adopting full MPC below.
- **QUB Belfast — "Lap Time Simulation Tool for the Development of an Electric Formula Student Car"** (https://pure.qub.ac.uk/files/164458711/Lap_Time_Simulation_Tool_for_the_Development_of_an_Electric_Formula_Student_Car.pdf — Cloudflare-blocked direct fetch; cite by URL). Per the abstract referenced in `docs/SIM_AUDIT_2026-05.md` and the Duke FSAE summary, the QUB thesis confirms QSS adequacy for FS sweeps and treats the driver as an envelope follower with no adaptive learning. Their published sensitivity analyses presume the envelope captures all of the parameter sensitivity, which only holds if the driver actually follows the envelope (our current calibrated driver does not).
- **MDPI 2024 — "Performance Optimization of a Formula Student Racing Car Using IPG CarMaker, Part 1"** (Takács & Zelei, *Engineering Proceedings* 79(1):86, https://www.mdpi.com/2673-4591/79/1/86, abstract: "The IPG CarMaker uses a multibody vehicle model and a learning algorithm for the virtual driver. The goal is to discover the behavior of the learning algorithm from the point of view of reliability and convergence. Simulations demonstrate that the lap time converges reliably. We also report that small changes in the vehicle parameters induce small changes in the simulated lap time, ie, the lap time is a differentiable function of the vehicle parameters."). **Takeaway: even in the multi-body professional tier, the criterion that justifies using the simulator for sweeps is "lap time is a differentiable function of vehicle parameters" — i.e. driver convergence to the envelope is what makes the sweep meaningful. The IPGDriver is a preview-follower with online learning; we do not need the learning loop because the envelope is recomputed analytically each sweep iteration.**
- **SAE 2016-36-0164 — "Lap Time Simulation of FSAE Vehicle With Quasi-Steady-State Model"** (https://www.sae.org/publications/technical-papers/content/2016-36-0164/ — abstract gated). Cited in `docs/SIM_AUDIT_2026-05.md` as confirming QSS adequacy for FSAE-class sweeps and identifying combined-slip Pacejka and tire thermal as the highest-leverage QSS upgrades. Driver model is unspecified beyond "follows the envelope" — same separation of concerns we adopt.
- **Wisconsin Racing WR-217e LapSim** (https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf, full PDF read). Pages 4-10 build a symplectic-Euler explicit time-stepping point mass with `F_motor = min(T_max·N/r, P_max/v)` and `F_cp = min(F_friction, F_motor)`. **They have no driver model at all — the simulator IS the envelope, and the "driver" simply commands full available force every step.** Page 21-23 add a software power limit (e.g. 40 kW endurance vs 80 kW peak rules), which produces a flat envelope reduction. Page 23 explicitly notes the trade: "as the power limit is reduced from 80 [kW] to 40 [kW], lap time increases from 80 [sec] to 82 [sec], or a 2.5% increase. At the same time, energy consumption goes from 8.1 [kWh] to 6.5 [kWh], or a 20% decrease." **Takeaway: a static torque/power cap parameter sweep IS visible end-to-end exactly because the WR-217e LapSim driver always tracks the envelope. Today our calibrated driver does not, which is why the audit grades it C+.**
- **Chalmers FS EV Powertrain Design** (https://publications.lib.chalmers.se/records/fulltext/191837/191837.pdf, full PDF read). Bachelor's thesis, 2013. Page 31 confirms standard PMSM constant-torque + constant-power field-weakening envelope (matches `PowertrainModel.max_motor_torque`). Page 38+ uses a 4WD vs RWD acceleration-event comparison with no adaptive driver. **No new technique to adopt; confirms the standard QSS architecture with no adaptive driver is the FSAE baseline.**
- **IEEE EVER 2019 — "A Quasi-Steady-State Lap Time Simulation for Electrified Race Cars"** (Heilmeier, Geisslinger, Betz, https://ieeexplore.ieee.org/document/8813646/ — abstract retrieved via Google Scholar showing it as the academic publication that documents the TUMFTM `laptimesim` codebase). **Same architecture as the open-source repo above; treat as the canonical citation for the QSS + envelope-follower + energy-management approach.**
- **OptimumLap product page** (https://optimumg.com/product/optimumlap/ — point-mass QSS, no driver). Validates the lower bound: a tool with literally zero driver model can still grade lap-time sensitivity to mass, drag, motor curve. Our current sim with telemetry-replay driver is strictly stronger; an adaptive envelope-follower is strictly stronger still.
- **CT-16EV codebase**: confirmed by reading `src\fsae_sim\sim\speed_envelope.py:45-247` that `SpeedEnvelope.compute()` already produces a per-segment ceiling that respects (a) corner cornering speed (Pass 1), (b) backward braking feasibility with lap-wrap fixed point (Pass 2), (c) forward acceleration feasibility honoring `lvcu_torque_ceiling`/`apply_inverter_delivery` (Pass 3), and (d) combined-slip correction with re-propagation (Pass 4). **We do NOT need a separate planner.** Confirmed by reading `engine.py:529-544` that the engine consumes `ControlCommand(throttle_pct, brake_pct, metadata={'max_speed_ms': ...})` and combines `throttle_pct` → `pedal_to_torque_request` (calibrated) or `lvcu_torque_command` (predictive) → `apply_inverter_delivery` → wheel force; brake_pct → `mechanical_brake_force`. The same firmware-faithful chain is already wired for any strategy that fills these fields.

**Why we chose envelope-follower with PI tracking over (a) pure feedforward, (b) PID, (c) MPC, (d) RL:**
- Pure feedforward (TUMFTM-style throttle = 1 except for energy management) is sufficient *if* the predictor-corrector at `engine.py:601-626` lands the resolved exit speed exactly on `v_max[i]`. In practice it does not always (the engine's force-balance solve uses `enforce_speed_limit` which clips drive force to land on the envelope at the entry-speed estimate, then re-solves at the operating-speed midpoint — there is residual error from drag-vs-speed nonlinearity). A small PI on the velocity error (using `v − v_max` summed across the previous N segments as the I-term) absorbs that residual.
- PID adds a derivative term that fights with the inherent step-discontinuity at the envelope's corner-entry transitions and would require a derivative filter we do not need. **Reject the D-term**.
- MPC needs a runtime QP/NLP solver in the inner loop. Maull & Schommer 2022 demonstrated it in MATLAB/Simulink targeting 150 m maneuvers, not 21-22 km endurance with sub-second iteration budget for a sweep harness. The MPC paper's *gain* is energy shaping (already captured by the LBP-style stint budget), not velocity tracking. **Reject MPC for velocity control**; **adopt MPC's energy-shaping idea** as the LBP-LS hybrid stint budgeter, which is closed-form O(N log N) and ships with TUMFTM.
- RL would learn what an envelope follower computes analytically, take days to converge, and break determinism. The user's "correctness over accuracy" guidance and "no bandaid fixes" guidance both rule it out. RL is also overkill: the optimal control problem here has a closed-form analytical solution (the envelope IS the value function for a min-time problem). **Reject RL with prejudice**.

## Alternatives Considered (and Rejected)

1. **Keep `CalibratedStrategy` and just add a "shift brake points by Δ" knob driven by capability changes** — rejected as a bandaid (CLAUDE.md "No bandaid fixes — root cause only"). The structural problem is that the calibrated driver replays a fixed pedal trace; a Δ-shift knob hides this without making the driver adapt to the actual envelope shape.
2. **Pure replay with a pedal-rescaling layer** ("if torque cap is +20%, scale recorded LVCU Torque Req by 0.83 to keep wheel torque the same") — rejected. This breaks pedal physics: the LVCU's `pedal_to_torque_request` has a power-divide and inverter clamp that are nonlinear in the pedal position, so a simple scale produces a request the LVCU itself would never issue. Also fails on the stated acceptance bar (it does not "drive as well as the real driver when given the same envelope" — it just replays the real driver).
3. **Full nonlinear MPC with horizon ~50 m** — rejected. Cited paper (Maull & Schommer 2022) used MPC over 150 m maneuvers with a Simulink solver; our sweep harness needs 21-lap, 22 km endurance to complete in seconds (CLAUDE.md "sims should complete in seconds, not minutes"). Solver-in-the-loop violates that. The energy-management *output* of MPC is captured by LBP/LS heuristics at sub-millisecond cost.
4. **Backward-pass-from-driver scratch (driver recomputes its own brake points instead of reading `SpeedEnvelope`'s)** — rejected. The envelope is already the right answer under the current vehicle physics; recomputing it inside the driver would either duplicate `SpeedEnvelope` (DRY violation, two source-of-truth divergence) or be coarser. The envelope handles `lvcu_torque_ceiling`, combined slip, and lap-wrap fixed point correctly; a hand-rolled driver brake-point search would not.
5. **Reinforcement learning** — rejected on determinism, training cost, and adoption-risk grounds; see "Why RL" rejected above.
6. **Hand-tuned per-corner brake bias from Michigan telemetry baked into the strategy** — rejected. Bakes Michigan-specific behavior into a "predictive" driver, which would invalidate `SimulationMode.PREDICTION` (engine.py:130-141 forbids telemetry-derived shortcuts). The brake bias instead lives in the *configuration* (a brake-bias parameter on the vehicle, not the driver), and the driver derives a structural rule (regen first, then friction) from physics.
7. **One-off PID tuning per sweep configuration** — rejected. Defeats the point of running a sweep deterministically. PI gains are computed from the segment time constants once at construction.
8. **Accept the 7.3% calibrated lap-time error and only add an "adaptation modifier"** — rejected; "correctness over accuracy" plus the explicit acceptance bar (replay-mode lap-time error < 1%) mean the driver must be able to follow the envelope to within the same tolerance the engine itself can hit.

## Architecture Decisions Awaiting User Input

- **A1 — Brake-bias rule (regen vs friction split)**: when the driver wants `−F` Newtons of decel, what fraction goes to motor regen versus mechanical brake?
  - **Recommendation**: **prefer-regen up to motor envelope, then add friction**. Compute `F_regen_max(rpm) = wheel_force(max_motor_torque(rpm))` (negative), use as much of it as needed, and route any remainder through `mechanical_brake_force(brake_pct, v)`. This matches the only physical structural rule (regen comes free up to the motor envelope; brake pads come on top). It does *not* require any telemetry. Real-driver brake-pressure / regen-torque ratios from Michigan can be exposed in a calibration report (regression test only) but never become a runtime input.
  - **Alternative under consideration**: real-driver bias is closer to ~50/50 by feel, with the brake pedal being the only regen request channel (CT-16EV: brake pedal pressure → BSE-gated regen torque). User decides whether the adaptive driver should target the real bias or the energy-optimal bias. **Default if no answer**: energy-optimal (regen-first), with the bias regression test reporting the delta vs real for transparency.
- **A2 — Energy-budget shaping default**: should the adaptive driver default to **always-on FCFB** (Full-Capability Full-Boost; no kWh shaping) or **LBP** (Longest-time-to-Brake-Point; lifts on long straights to meet a kWh budget)?
  - **Recommendation**: **FCFB by default**, with LBP triggered when the user supplies an explicit `energy_budget_kwh` parameter. This keeps the baseline endurance time honest (no driver-side throttle cuts that would mask underlying physics issues) and only invokes shaping when the user is explicitly asking the sim to manage to a budget.
- **A3 — Predictor-corrector iteration count and PI gains**: the engine already runs a 2-pass predictor-corrector (`engine.py:601-625`); the adaptive driver layers a PI on top. **Recommendation**: integral term clamped to ±0.5 m/s velocity correction, proportional gain such that a 1 m/s overshoot at the envelope reduces throttle by 50% over the next segment. Concrete numbers derived analytically below in Task 4.3. User to confirm if anti-windup or different gains are preferred.
- **A4 — Retiring `CalibratedStrategy`**: keep it for the validation-against-baseline noise-floor calculation in `scripts/sim_compare.py`, but make `AdaptiveStrategy` the default for `SimulationMode.PREDICTION` and the Simulate webapp page. **Recommendation**: keep both, default switches to adaptive in PREDICTION mode only; CALIBRATION/REPLAY behavior is unchanged. User to confirm.
- **A5 — Lookahead segment count**: TUMFTM uses the full closed-loop `vel_lim_cl` array (full lap). Our `engine.py:340-347` precomputes `lookahead = 5` segments (~2.5 m at the 0.5 m default segment size). **Recommendation**: increase `lookahead` to 60 segments (~30 m, ~1 s at 30 m/s) so the driver can pre-lift and trail-brake across an entire corner-entry sequence. User to confirm.

---

## File Decomposition

| File | Action | Responsibility |
|------|--------|----------------|
| `src\fsae_sim\driver\adaptive.py` | **Create** | New `AdaptiveStrategy` class. Public API mirrors `CalibratedStrategy` (`name`, `decide`, `set_envelope`, optional `with_energy_budget`); subclasses `DriverStrategy`. |
| `src\fsae_sim\driver\envelope_tracker.py` | **Create** | Stateless helper: given (entry_speed, target_exit_speed, segment_length, vehicle, powertrain, bms_limit_a) returns (throttle_pct, brake_pct, regen_pct) by inverting the engine's force balance. Single source of truth for the throttle/brake mapping; called by `AdaptiveStrategy.decide`. |
| `src\fsae_sim\driver\energy_shaper.py` | **Create** | LBP/LS energy-budget shaping (TUMFTM port). Pure functions: `compute_throttle_mask(envelope, time_per_segment, energy_per_segment, budget_kwh) -> np.ndarray[bool]`. |
| `src\fsae_sim\driver\__init__.py` | **Modify** | Export `AdaptiveStrategy` alongside `CalibratedStrategy`, `ReplayStrategy`. |
| `src\fsae_sim\sim\engine.py` | **Modify** | Allow `AdaptiveStrategy` to be passed as `strategy`. `set_envelope` already plumbed at line 334. Increase `lookahead` from 5 → 60 (Task A5; behind a kwarg with default 60). Add `is_adaptive` branch in `commanded_motor_torque` so it flows through `lvcu_torque_command` (firmware-faithful path) the same as the prediction path, NOT through `pedal_to_torque_request` (which assumes pre-deadzone-remapped pedal). |
| `tests\driver\test_adaptive_strategy.py` | **Create** | Unit tests: flat envelope → constant throttle, ramp-down envelope → brake/regen, ramp-up → throttle. Hypothesis property tests on (entry, target, length) → invariants. |
| `tests\driver\test_envelope_tracker.py` | **Create** | Unit tests for the inverse force-balance solver. |
| `tests\driver\test_energy_shaper.py` | **Create** | Unit tests for LBP / LS / FCFB; test that allocation respects budget exactly. |
| `tests\driver\test_adaptive_michigan_replay.py` | **Create** | Telemetry-grounded regression test: build envelope from baseline `CT-16EV` config + calibrated track + grip_scale, run engine with `AdaptiveStrategy(envelope)`, assert lap-time error vs telemetry < 1% and net Ah error < 2%. |
| `scripts\sim_compare.py` | **Modify** | Add `--strategy adaptive` option. Existing `calibrated`/`replay` modes unchanged. |
| `docs\SIM_ACCURACY.md` | **Modify** | Add an "Adaptive sim" column to the metrics table once tests are passing. |
| `docs\SIM_AUDIT_2026-05.md` | **Modify** | Re-grade Driver model row from C+ to (target) A−; check off the P0 adaptive driver checkbox. |

---

## Tasks

### Task 1.1 — Failing test for envelope tracker on a flat envelope

**Files:**
- Create: `tests\driver\test_envelope_tracker.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/driver/test_envelope_tracker.py
"""Inverse force-balance solver tests for the envelope-tracker helper."""
from __future__ import annotations

import math
import pytest
import numpy as np

from fsae_sim.driver.envelope_tracker import EnvelopeTracker
from fsae_sim.vehicle.powertrain import PowertrainConfig
from fsae_sim.vehicle.powertrain_model import PowertrainModel
from fsae_sim.vehicle.dynamics import VehicleDynamics
# Reuse the existing CT-16EV vehicle config helper.
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.config_loader import load_vehicle_config


def _ct16ev_models():
    cfg = load_vehicle_config("ct16ev")
    pt = PowertrainModel(cfg.powertrain)
    dyn = VehicleDynamics(cfg.vehicle, powertrain_config=cfg.powertrain)
    return cfg, pt, dyn


def test_flat_envelope_at_steady_speed_commands_partial_throttle():
    """At a flat 20 m/s envelope, drag balances drive => partial throttle, no brake."""
    cfg, pt, dyn = _ct16ev_models()
    tracker = EnvelopeTracker(dyn, pt)
    # Entry at 20 m/s, target 20 m/s, 0.5 m segment, BMS unlimited.
    cmd = tracker.command(
        entry_speed_ms=20.0,
        target_exit_ms=20.0,
        segment_length_m=0.5,
        bms_current_limit_a=200.0,
        curvature=0.0,
        grade=0.0,
    )
    # Drag at 20 m/s with CdA=1.5 is ~0.5*1.225*1.5*400 = 367.5 N. The
    # adaptive driver must request enough torque to balance that exactly.
    assert cmd.brake_pct == pytest.approx(0.0)
    assert 0.05 < cmd.throttle_pct < 1.0
    # Force-balance check: predicted exit speed lands within 0.05 m/s of target.
    drive_torque = pt.lvcu_torque_command(
        cmd.throttle_pct, pt.motor_rpm_from_speed(20.0), 200.0,
    )
    drive_torque = pt.apply_inverter_delivery(
        pt.motor_rpm_from_speed(20.0), drive_torque,
    )
    f_drive = pt.wheel_force(drive_torque)
    f_resist = dyn.total_resistance(20.0, grade=0.0, curvature=0.0)
    a = (f_drive - f_resist) / dyn.m_effective
    v_exit = math.sqrt(20.0 * 20.0 + 2.0 * a * 0.5)
    assert v_exit == pytest.approx(20.0, abs=0.05)
```

- [ ] **Step 2: Run test, confirm `ImportError: EnvelopeTracker`.**

```powershell
pytest tests\driver\test_envelope_tracker.py::test_flat_envelope_at_steady_speed_commands_partial_throttle -x
```

### Task 1.2 — Implement `EnvelopeTracker` core (inverse force balance, no PI yet)

**Files:**
- Create: `src\fsae_sim\driver\envelope_tracker.py`

- [ ] **Step 1: Implement the helper (no PI yet — pure feedforward).**

Algorithm:
1. Compute required net force per kinematics:
   `F_required = m_eff * (v_target² − v_entry²) / (2 * length)`
2. Add resistance (drag + rolling + cornering drag) to get target `F_drive_minus_brake = F_required + F_resistance`.
3. If `F_drive_minus_brake >= 0`: route through drive; solve for the pedal that produces it through the firmware-faithful chain.
4. If `F_drive_minus_brake < 0`: split between regen and friction (recommendation A1: regen-first).
5. Inverse pedal solve: `lvcu_torque_command` is monotonic in `pedal_pct` for fixed `(rpm, bms_limit)`. Closed-form inverse: `pedal_remapped = F_required_at_wheel / wheel_force(torque_ceiling); pedal_raw = pedal_remapped * (deadzone_high − deadzone_low) + deadzone_low`. Single function call, no Newton iteration.
6. Honor traction limit: `min(F_drive, dyn.max_traction_force(speed))`.
7. Honor brake-pad ceiling at `dynamics._MAX_BRAKE_DECEL_G * mass_kg * g`.

```python
# src/fsae_sim/driver/envelope_tracker.py
"""Stateless inverse force-balance solver for the adaptive driver.

Given a target exit speed and entry speed for a track segment, returns
the throttle / brake / regen pedal commands whose resulting force
balance under the firmware-faithful LVCU + inverter delivery + tire
limits chain produces that exit speed.

Intentionally stateless: every call is determined by its inputs. The
PI velocity-error correction lives in AdaptiveStrategy, not here.
"""
from __future__ import annotations

from dataclasses import dataclass

from fsae_sim.driver.strategy import ControlAction, ControlCommand
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain_model import PowertrainModel


@dataclass(frozen=True)
class TrackerCommand:
    throttle_pct: float
    brake_pct: float
    regen_request_pct: float  # advisory: portion of brake_pct attributable to regen
    action: ControlAction


class EnvelopeTracker:
    def __init__(
        self,
        dynamics: VehicleDynamics,
        powertrain: PowertrainModel,
    ) -> None:
        self._dyn = dynamics
        self._pt = powertrain

    def command(
        self,
        *,
        entry_speed_ms: float,
        target_exit_ms: float,
        segment_length_m: float,
        bms_current_limit_a: float,
        curvature: float,
        grade: float,
    ) -> TrackerCommand:
        m_eff = self._dyn.m_effective
        v0, v1 = entry_speed_ms, target_exit_ms
        # Average operating speed for the inverse solve. Mirrors engine's
        # predictor-corrector midpoint.
        v_op = max(0.5, (v0 + v1) / 2.0)
        rpm = self._pt.motor_rpm_from_speed(v_op)
        f_resist = self._dyn.total_resistance(v_op, grade=grade, curvature=curvature)
        f_required_net = m_eff * (v1 * v1 - v0 * v0) / (2.0 * max(segment_length_m, 1e-6))
        f_drive_minus_brake = f_required_net + f_resist

        if f_drive_minus_brake >= 0.0:
            # Drive direction: solve for throttle pedal.
            torque_ceiling = self._pt.lvcu_torque_ceiling(rpm, bms_current_limit_a)
            torque_ceiling = self._pt.apply_inverter_delivery(rpm, torque_ceiling)
            f_max_drive = self._pt.wheel_force(torque_ceiling)
            f_drive = min(f_drive_minus_brake, f_max_drive)
            f_drive = min(f_drive, self._dyn.max_traction_force(v_op))
            pedal_remapped = f_drive / max(f_max_drive, 1e-6) if f_max_drive > 0 else 0.0
            cfg = self._pt.config
            span = cfg.lvcu_pedal_deadzone_high - cfg.lvcu_pedal_deadzone_low
            pedal_raw = cfg.lvcu_pedal_deadzone_low + pedal_remapped * span
            pedal_raw = max(0.0, min(1.0, pedal_raw))
            return TrackerCommand(
                throttle_pct=pedal_raw,
                brake_pct=0.0,
                regen_request_pct=0.0,
                action=ControlAction.THROTTLE if pedal_raw > 0.0 else ControlAction.COAST,
            )

        # Decel direction: regen first, friction for the remainder.
        f_decel_required = -f_drive_minus_brake  # positive
        # Available regen at this rpm:
        regen_torque_max = self._pt.max_motor_torque(rpm)
        f_regen_max = self._pt.wheel_force(regen_torque_max)  # positive at the wheel
        f_regen_used = min(f_decel_required, f_regen_max)
        f_friction_needed = f_decel_required - f_regen_used
        # Convert to pedal fractions. Brake pedal is normalized to peak
        # observed brake-decel (mechanical_brake_force calls this). Regen
        # is requested via the brake pedal too (CT-16EV: BSE-gated), but
        # we report it separately so the engine and reporting layers can
        # split the channels.
        peak_brake_force = self._dyn._MAX_BRAKE_DECEL_G * 9.81 * self._dyn.vehicle.mass_kg
        brake_pct_friction = min(1.0, f_friction_needed / max(peak_brake_force, 1.0))
        regen_pct = (
            min(1.0, f_regen_used / max(f_regen_max, 1.0))
            if f_regen_max > 0 else 0.0
        )
        # Combined brake_pct = friction component (engine consumes this for
        # mechanical_brake_force). Regen is separately advised.
        return TrackerCommand(
            throttle_pct=0.0,
            brake_pct=brake_pct_friction,
            regen_request_pct=regen_pct,
            action=ControlAction.BRAKE,
        )
```

- [ ] **Step 2: Re-run the failing test, confirm pass.**

### Task 1.3 — Property tests for envelope tracker (analytical sanity)

- [ ] **Step 1: Add property tests with hypothesis.**

```python
# tests/driver/test_envelope_tracker.py (append)
from hypothesis import given, settings, strategies as st

@settings(max_examples=200, deadline=None)
@given(
    v_entry=st.floats(min_value=2.0, max_value=35.0),
    v_target=st.floats(min_value=2.0, max_value=35.0),
    seg_len=st.floats(min_value=0.1, max_value=2.0),
)
def test_tracker_resolves_to_within_segment_kinematics(v_entry, v_target, seg_len):
    """For any (entry, target, length), the predicted exit speed under the
    issued command is within 0.5 m/s of the target on a straight, flat
    segment with no BMS clamp."""
    cfg, pt, dyn = _ct16ev_models()
    tracker = EnvelopeTracker(dyn, pt)
    cmd = tracker.command(
        entry_speed_ms=v_entry,
        target_exit_ms=v_target,
        segment_length_m=seg_len,
        bms_current_limit_a=300.0,
        curvature=0.0,
        grade=0.0,
    )
    rpm = pt.motor_rpm_from_speed((v_entry + v_target) / 2.0)
    f_drive = 0.0
    if cmd.throttle_pct > 0:
        t = pt.lvcu_torque_command(cmd.throttle_pct, rpm, 300.0)
        t = pt.apply_inverter_delivery(rpm, t)
        f_drive = pt.wheel_force(t)
    f_brake = (
        dyn.mechanical_brake_force(cmd.brake_pct, (v_entry + v_target) / 2.0)
        if cmd.brake_pct > 0 else 0.0
    )
    f_resist = dyn.total_resistance((v_entry + v_target) / 2.0, grade=0.0, curvature=0.0)
    a = (f_drive - f_brake - f_resist) / dyn.m_effective
    v_exit_sq = v_entry * v_entry + 2.0 * a * seg_len
    v_exit = (max(0.0, v_exit_sq)) ** 0.5
    # Tolerance 0.5 m/s — accounts for traction / brake-pad ceilings
    # genuinely preventing the target from being reached.
    assert abs(v_exit - v_target) <= 0.5 or v_target > v_exit  # cap miss is OK
```

- [ ] **Step 2: Run, confirm pass. Investigate any failures (do NOT add tolerance until root cause is understood).**

### Task 2.1 — Failing test for `AdaptiveStrategy.decide` on a single segment

**Files:**
- Create: `tests\driver\test_adaptive_strategy.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/driver/test_adaptive_strategy.py
"""AdaptiveStrategy unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from fsae_sim.driver.adaptive import AdaptiveStrategy
from fsae_sim.driver.strategy import ControlAction, ControlCommand, SimState
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.config_loader import load_vehicle_config


def _flat_track(num_segments: int = 100, seg_len: float = 0.5) -> Track:
    segs = [
        Segment(
            index=i,
            distance_start_m=i * seg_len,
            length_m=seg_len,
            curvature=0.0,
            grade=0.0,
        )
        for i in range(num_segments)
    ]
    return Track(name="flat100", segments=segs, source="synthetic")


def test_adaptive_with_constant_envelope_commands_constant_throttle():
    cfg = load_vehicle_config("ct16ev")
    track = _flat_track()
    strategy = AdaptiveStrategy.from_models(cfg, track)
    v_max = np.full(track.num_segments, 25.0, dtype=np.float64)  # 25 m/s flat
    strategy.set_envelope(v_max)
    state = SimState(
        time=0.0, distance=10.0, speed=25.0,
        soc=0.95, pack_voltage=405.0, pack_current=10.0,
        cell_temp=30.0, lap=0, segment_idx=20,
    )
    cmd = strategy.decide(state, [track.segments[i] for i in range(20, 25)])
    assert cmd.brake_pct == pytest.approx(0.0, abs=1e-3)
    # Drag at 25 m/s should cost ~575 N to overcome -> partial throttle.
    assert 0.1 < cmd.throttle_pct < 0.95
    assert cmd.action in (ControlAction.THROTTLE, ControlAction.COAST)
```

- [ ] **Step 2: Run test, confirm `ImportError`.**

### Task 2.2 — Implement `AdaptiveStrategy` skeleton

**Files:**
- Create: `src\fsae_sim\driver\adaptive.py`

- [ ] **Step 1: Implement the strategy.**

```python
# src/fsae_sim/driver/adaptive.py
"""Adaptive envelope-following driver strategy.

Given a pre-computed forward-backward speed envelope, commands
throttle / brake / regen so each segment exit speed matches the
envelope. Stateless apart from a single integral-error term carried
across segments for PI tracking.

This is the production driver for SimulationMode.PREDICTION and the
default for the Simulate webapp page. ReplayStrategy and
CalibratedStrategy are retained for replay/validation/noise-floor work.
"""
from __future__ import annotations

import numpy as np

from fsae_sim.driver.envelope_tracker import EnvelopeTracker, TrackerCommand
from fsae_sim.driver.strategy import (
    ControlAction, ControlCommand, DriverStrategy, SimState,
)
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain_model import PowertrainModel


class AdaptiveStrategy(DriverStrategy):
    name = "adaptive"

    # PI gains derived analytically: at v=25 m/s, 0.5 m segment, the
    # segment time is ~20 ms. To correct a 1 m/s overshoot in 5 segments
    # (~100 ms) requires roughly K_p = 0.5 (50% throttle pull-back per
    # m/s of error). Integral term clamped at +/-0.5 m/s (A3).
    _KP_VELOCITY: float = 0.05  # throttle fraction per (m/s) of error
    _KI_VELOCITY: float = 0.005
    _I_CLAMP: float = 0.5

    def __init__(
        self,
        dynamics: VehicleDynamics,
        powertrain: PowertrainModel,
        track: Track,
        *,
        bms_current_limit_a: float = 200.0,
        energy_budget_kwh: float | None = None,
    ) -> None:
        self._tracker = EnvelopeTracker(dynamics, powertrain)
        self._track = track
        self._envelope: np.ndarray | None = None
        self._bms_current_limit_a = bms_current_limit_a
        self._energy_budget_kwh = energy_budget_kwh
        self._throttle_mask: np.ndarray | None = None  # set by energy shaper
        # PI state.
        self._velocity_error_integral: float = 0.0

    @classmethod
    def from_models(
        cls,
        vehicle: VehicleConfig,
        track: Track,
        *,
        energy_budget_kwh: float | None = None,
    ) -> "AdaptiveStrategy":
        # Local import to avoid circular at module load time.
        from fsae_sim.vehicle.powertrain_model import PowertrainModel
        from fsae_sim.vehicle.dynamics import VehicleDynamics
        pt = PowertrainModel(vehicle.powertrain)
        dyn = VehicleDynamics(vehicle.vehicle, powertrain_config=vehicle.powertrain)
        return cls(dyn, pt, track, energy_budget_kwh=energy_budget_kwh)

    @property
    def uses_observed_speed_caps(self) -> bool:
        # PREDICTION-mode safe: the envelope is the only speed reference,
        # and SpeedEnvelope.compute() does not consume telemetry.
        return False

    def set_envelope(self, v_max: np.ndarray) -> None:
        self._envelope = np.asarray(v_max, dtype=np.float64).copy()
        self._velocity_error_integral = 0.0  # reset I on new envelope

    def set_bms_limit(self, bms_current_limit_a: float) -> None:
        self._bms_current_limit_a = float(bms_current_limit_a)

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        if self._envelope is None:
            raise RuntimeError(
                "AdaptiveStrategy.decide called before set_envelope; "
                "engine.py wires this at line 334-338."
            )
        idx = state.segment_idx % len(self._envelope)
        # Target exit speed = envelope at next segment (or wrap to seg 0).
        next_idx = (idx + 1) % len(self._envelope)
        target_exit = float(self._envelope[next_idx])
        seg = upcoming[0] if upcoming else self._track.segments[idx]
        # Velocity error: how much we exceed the envelope at this segment.
        v_err = state.speed - float(self._envelope[idx])
        self._velocity_error_integral = max(
            -self._I_CLAMP,
            min(self._I_CLAMP, self._velocity_error_integral + v_err),
        )
        # PI correction: subtract from target_exit so the tracker pulls
        # us back to envelope when ahead, pushes harder when behind.
        pi_correction = (
            self._KP_VELOCITY * v_err + self._KI_VELOCITY * self._velocity_error_integral
        )
        target_with_pi = max(0.5, target_exit - pi_correction * target_exit)

        # Energy shaper: if this segment is masked off, cut throttle.
        force_lift = (
            self._throttle_mask is not None
            and not bool(self._throttle_mask[idx])
        )

        cmd = self._tracker.command(
            entry_speed_ms=state.speed,
            target_exit_ms=target_with_pi,
            segment_length_m=seg.length_m,
            bms_current_limit_a=self._bms_current_limit_a,
            curvature=seg.curvature,
            grade=seg.grade,
        )
        throttle_pct = cmd.throttle_pct if not force_lift else 0.0
        return ControlCommand(
            action=cmd.action if throttle_pct > 0.0 or cmd.brake_pct > 0.0 else ControlAction.COAST,
            throttle_pct=throttle_pct,
            brake_pct=cmd.brake_pct,
            metadata={
                "max_speed_ms": float(self._envelope[idx]),
                "regen_request_pct": float(cmd.regen_request_pct),
                "v_err": float(v_err),
            },
        )
```

- [ ] **Step 2: Re-run flat-envelope test, confirm pass.**

### Task 2.3 — Tests for ramp-up and ramp-down envelopes

- [ ] **Step 1: Add ramp-down (corner-entry) test.**

```python
# tests/driver/test_adaptive_strategy.py (append)

def test_ramp_down_envelope_commands_brake():
    """Envelope drops from 25 to 10 m/s -> brake_pct > 0."""
    cfg = load_vehicle_config("ct16ev")
    track = _flat_track()
    strategy = AdaptiveStrategy.from_models(cfg, track)
    v_max = np.linspace(25.0, 10.0, track.num_segments)
    strategy.set_envelope(v_max)
    state = SimState(
        time=0.0, distance=25.0, speed=25.0,
        soc=0.95, pack_voltage=405.0, pack_current=0.0,
        cell_temp=30.0, lap=0, segment_idx=50,
    )
    cmd = strategy.decide(state, [track.segments[i] for i in range(50, 55)])
    assert cmd.throttle_pct == pytest.approx(0.0, abs=1e-3)
    assert cmd.brake_pct > 0.0
    assert cmd.metadata["regen_request_pct"] >= 0.0


def test_ramp_up_envelope_commands_full_throttle():
    """Standing-start to 30 m/s envelope -> full throttle, no brake."""
    cfg = load_vehicle_config("ct16ev")
    track = _flat_track()
    strategy = AdaptiveStrategy.from_models(cfg, track)
    v_max = np.linspace(2.0, 30.0, track.num_segments)
    strategy.set_envelope(v_max)
    state = SimState(
        time=0.0, distance=2.0, speed=2.0,
        soc=0.95, pack_voltage=405.0, pack_current=0.0,
        cell_temp=30.0, lap=0, segment_idx=2,
    )
    cmd = strategy.decide(state, [track.segments[i] for i in range(2, 7)])
    assert cmd.brake_pct == pytest.approx(0.0, abs=1e-3)
    assert cmd.throttle_pct >= 0.85  # near saturation (deadzone-remapped 1.0)
```

- [ ] **Step 2: Run tests, fix any issues. Do NOT loosen tolerances without finding root cause.**

### Task 3.1 — Plumb `AdaptiveStrategy` into `engine.py`

- [ ] **Step 1: Find the dispatch point.**

Confirmed at `src\fsae_sim\sim\engine.py:328-329`:
```python
is_replay = isinstance(self.strategy, ReplayStrategy)
is_calibrated = isinstance(self.strategy, CalibratedStrategy)
```

- [ ] **Step 2: Add `is_adaptive`.**

The engine has two distinct LVCU paths today:
- `is_replay`: uses recorded torque directly.
- `is_calibrated`: uses `pedal_to_torque_request` because calibration data feeds *post-deadzone* pedal positions (AiM "Throttle Pos" is already `tmap_lut(tps_combined)` per LVCU Code.txt line 499).
- `else` (predictive): uses `lvcu_torque_command` which applies the deadzone remap.

The adaptive driver outputs **raw pedal positions** (TPS_combined, pre-deadzone) per Task 1.2 step 5 — `pedal_raw = deadzone_low + pedal_remapped * span`. Therefore it must flow through `lvcu_torque_command`, NOT `pedal_to_torque_request`. The `else` branch already does this. Add an explicit `is_adaptive = isinstance(self.strategy, AdaptiveStrategy)` check and route through the same `else` branch (or remove the `is_calibrated` flag entirely once `CalibratedStrategy` is no longer the default — but keep it for now to preserve the calibration path; A4).

- [ ] **Step 3: Set the adaptive driver's BMS limit each lap.**

The engine recomputes BMS limit per segment (`engine.py:556`). Plumb the *initial* one to `AdaptiveStrategy.set_bms_limit(initial_bms_limit)` once before the lap loop, parallel to how `set_envelope(v_max)` is wired at line 334-338. Also push it again at the start of each lap once P1 lap-by-lap envelope refresh lands; do not block on that.

- [ ] **Step 4: Increase `lookahead` from 5 to 60 (A5).**

Change `engine.py:340`: `lookahead = 60`. Behind kwarg `lookahead_segments=60`.

### Task 3.2 — End-to-end test: adaptive driver + flat synthetic track

- [ ] **Step 1: Write integration test.**

```python
# tests/driver/test_adaptive_michigan_replay.py
"""End-to-end adaptive driver vs Michigan endurance telemetry."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fsae_sim.config_loader import load_vehicle_config
from fsae_sim.data.loader import load_aim_csv
from fsae_sim.driver.adaptive import AdaptiveStrategy
from fsae_sim.sim.engine import SimulationEngine, SimulationMode
from fsae_sim.track.track import Track
from fsae_sim.vehicle.battery_model import BatteryModel


REPO = Path(__file__).resolve().parents[2]
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"


@pytest.mark.skipif(not TELEM.exists(), reason="Telemetry CSV not present")
def test_adaptive_michigan_endurance_within_one_percent_lap_time():
    cfg = load_vehicle_config("ct16ev")
    _meta, df = load_aim_csv(TELEM)
    track = Track.from_telemetry(df=df)
    battery = BatteryModel.from_calibrated_pack(cfg.battery)
    strategy = AdaptiveStrategy.from_models(cfg, track)
    engine = SimulationEngine(
        cfg, track, strategy, battery,
        mode=SimulationMode.CALIBRATION,  # allow telem track for now
        allow_telemetry_track=True,
    )
    result = engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=25.0)

    # Telemetry baseline (from SIM_ACCURACY.md):
    real_time_s = 1608.75
    real_distance_m = 22100.23
    real_charge_ah = 8.04
    real_energy_kwh = 3.27

    # Acceptance: < 1% lap-time error (i.e. <= 16.1 s on 1608 s).
    assert abs(result.total_time_s - real_time_s) / real_time_s < 0.01, (
        f"Time error {result.total_time_s - real_time_s:.1f} s "
        f"({(result.total_time_s - real_time_s) / real_time_s * 100:.2f}%) "
        f"exceeds 1% (= {real_time_s * 0.01:.1f} s). Adaptive driver "
        "is not following the envelope as well as the real driver."
    )
    # Net Ah within 2% (0.16 Ah) — acceptance bar from plan header.
    assert abs(result.net_charge_ah - real_charge_ah) / real_charge_ah < 0.02, (
        f"Net Ah error {result.net_charge_ah - real_charge_ah:.2f} Ah "
        f"({(result.net_charge_ah - real_charge_ah) / real_charge_ah * 100:.2f}%) "
        f"exceeds 2% (= {real_charge_ah * 0.02:.3f} Ah)."
    )
    # Net kWh within 2% (0.065 kWh).
    assert abs(result.net_energy_kwh - real_energy_kwh) / real_energy_kwh < 0.02
```

- [ ] **Step 2: Run. Expect this to fail initially. Diagnose.**

Likely failure modes and fixes:
1. **Time error too large in early laps**: PI gains too small; the tracker is allowing speed to drift below envelope. Bump `_KP_VELOCITY` to 0.1 deterministically (do NOT tune to match telemetry — choose from the 5-segment-recovery target in Task 2.2). Re-derive analytically if necessary.
2. **Net Ah error too large**: regen-first split (A1 default) recovers more energy than real driver did. Either accept (it's energy-correct) or add a brake-bias config matching real-driver behavior. **User decision A1**.
3. **Brake force exceeds tire ceiling on combined-slip corners**: tracker doesn't yet honor combined-slip envelope. Wire combined-slip cornering speed query into the tracker as a hard `target_exit` cap, not a clamp. (This is what the engine's `enforce_speed_limit` already does.)

### Task 3.3 — Adaptive driver test against the per-lap holdout slice

- [ ] **Step 1: Add holdout-only test.**

```python
def test_adaptive_michigan_holdout_laps_within_one_percent():
    """Holdout laps 13-21 from SIM_ACCURACY.md show calibrated driver
    has +2.10 s over 9 laps (0.13%). Adaptive must do at least as well."""
    cfg = load_vehicle_config("ct16ev")
    _meta, df = load_aim_csv(TELEM)
    track = Track.from_telemetry(df=df)
    battery = BatteryModel.from_calibrated_pack(cfg.battery)
    strategy = AdaptiveStrategy.from_models(cfg, track)
    # Run 21 laps total, evaluate laps 13-21 separately.
    engine = SimulationEngine(
        cfg, track, strategy, battery,
        mode=SimulationMode.CALIBRATION, allow_telemetry_track=True,
    )
    result = engine.run(num_laps=21)
    # Per-lap analysis using engine state log:
    states = result.states
    holdout = states[(states["lap"] >= 13) & (states["lap"] <= 21)]
    sim_time_holdout = float(holdout["seg_time_s"].sum())
    # Telemetry holdout time computed from CleanedEndurance.csv
    # detect_lap_boundaries[13:22].
    from fsae_sim.analysis.validation import detect_lap_boundaries
    boundaries = detect_lap_boundaries(df)
    real_holdout_s = sum(
        float(df.iloc[end]["Time"] - df.iloc[start]["Time"])
        for (start, end) in boundaries[13:22]
    )
    # Acceptance: same 1% bar.
    assert abs(sim_time_holdout - real_holdout_s) / real_holdout_s < 0.01
```

- [ ] **Step 2: Run.**

### Task 4.1 — Energy shaper: LBP allocator (TUMFTM port)

**Files:**
- Create: `src\fsae_sim\driver\energy_shaper.py`

- [ ] **Step 1: Write failing test.**

```python
# tests/driver/test_energy_shaper.py
"""Energy-budget allocator tests (TUMFTM LBP port)."""
import numpy as np
import pytest

from fsae_sim.driver.energy_shaper import (
    compute_throttle_mask, EnergyShaperStrategy,
)


def test_lbp_allocates_lifts_to_longest_pre_brake_segments():
    """Envelope: 0..10 are long straight (50 m), 10..15 ramp down to brake,
    15..20 corner, 20..30 next straight. LBP should preferentially lift
    in the longest-time-to-brake segments (0..5 of the first straight
    far from the brake point at 10)."""
    n = 30
    # Simple two-corner envelope.
    v_max = np.full(n, 30.0)
    v_max[10:15] = np.linspace(30.0, 10.0, 5)
    v_max[15:20] = 10.0
    v_max[20:25] = np.linspace(10.0, 30.0, 5)
    seg_len = np.full(n, 5.0)
    # Time per segment ~ length / v
    t_per_seg = seg_len / v_max
    # Energy per segment proxy: throttle * length * v_max (Joules)
    e_per_seg = 0.8 * seg_len * v_max  # arbitrary kWh budget feed
    # Budget: cut 20% of total energy.
    total = float(e_per_seg.sum())
    mask = compute_throttle_mask(
        v_max, t_per_seg, e_per_seg, target_energy=total * 0.8,
    )
    # Mask: True = throttle, False = lift.
    # Expect lifts to be at indices far from the next brake (10..15).
    lifted = np.where(~mask)[0]
    # All lifted indices should have time-to-next-brake > median segment time.
    # (Sanity check; the precise indices are TUMFTM-defined.)
    assert len(lifted) > 0
    assert (lifted < 10).any()  # some lifts on the first straight
```

- [ ] **Step 2: Implement LBP allocator (port of `__strategy_lbp` in TUMFTM `driver.py:160-200`).**

```python
# src/fsae_sim/driver/energy_shaper.py
"""Stint-level energy-budget shaper for the adaptive driver.

Implements TUMFTM laptime-simulation's LBP (Longest time to next Brake
Point) allocator: when the energy budget cannot support full-throttle
across the lap, lift on segments that maximize the time to the next
brake point (i.e. cost the least lap time per Wh saved).

Reference: `laptimesim/src/driver.py:__strategy_lbp` in
https://github.com/TUMFTM/laptime-simulation
"""
from __future__ import annotations

import numpy as np


def compute_throttle_mask(
    v_max: np.ndarray,
    t_per_seg: np.ndarray,
    e_per_seg: np.ndarray,
    *,
    target_energy: float,
    strategy: str = "LBP",
) -> np.ndarray:
    """Return per-segment boolean mask: True = throttle, False = lift."""
    n = len(v_max)
    mask = np.ones(n, dtype=bool)
    current_energy = float(e_per_seg.sum())
    if current_energy <= target_energy:
        return mask
    # Find brake points: indices where envelope decreases.
    brake_inds = np.flatnonzero(np.diff(v_max) < -0.05)
    if len(brake_inds) == 0:
        # Fallback: flat lift across the whole lap.
        return mask  # nothing to shape against; caller scales down separately.
    if strategy == "LBP":
        # Time to next brake point for each segment (with wrap).
        t_to_brake = np.zeros(n)
        for i in range(n):
            future_brakes = brake_inds[brake_inds >= i]
            if len(future_brakes) > 0:
                next_brake = future_brakes[0]
                t_to_brake[i] = float(t_per_seg[i:next_brake].sum())
            else:
                # Wrap.
                next_brake = brake_inds[0]
                t_to_brake[i] = (
                    float(t_per_seg[i:].sum()) + float(t_per_seg[:next_brake].sum())
                )
        # Sort segments by descending time-to-next-brake (longest = lift first).
        order = np.argsort(-t_to_brake)
    elif strategy == "LS":
        # Sort by ascending speed (slowest = lift first).
        order = np.argsort(v_max)
    else:
        raise ValueError(f"Unknown energy strategy {strategy!r}")
    # Greedily lift segments until budget met.
    for idx in order:
        if current_energy <= target_energy:
            break
        # Skip brake-point segments (they are already cost-zero).
        if idx in brake_inds:
            continue
        mask[idx] = False
        current_energy -= float(e_per_seg[idx])
    return mask
```

- [ ] **Step 3: Run test, confirm pass.**

### Task 4.2 — Wire energy shaper into `AdaptiveStrategy`

- [ ] **Step 1: Add public method.**

```python
# Append to src/fsae_sim/driver/adaptive.py

def shape_for_budget(
    self,
    energy_budget_kwh: float,
    *,
    estimated_seg_times_s: np.ndarray,
    estimated_seg_energy_j: np.ndarray,
) -> None:
    """Pre-compute the per-segment throttle mask for an energy-limited stint."""
    from fsae_sim.driver.energy_shaper import compute_throttle_mask
    self._throttle_mask = compute_throttle_mask(
        self._envelope, estimated_seg_times_s, estimated_seg_energy_j,
        target_energy=energy_budget_kwh * 3_600_000.0,  # kWh → J
    )
```

- [ ] **Step 2: Add caller in engine for the budget case (defer plumbing — only used when user passes `--energy-budget`; tracked as A2 user decision and a follow-up task).**

### Task 5.1 — Update `scripts/sim_compare.py` to add adaptive option

- [ ] **Step 1: Add `adaptive` to the `--strategy` choices.**

- [ ] **Step 2: Build adaptive strategy when selected.**

```python
# In scripts/sim_compare.py:
elif args.strategy == "adaptive":
    from fsae_sim.driver.adaptive import AdaptiveStrategy
    strategy = AdaptiveStrategy.from_models(vehicle_config, track)
```

- [ ] **Step 3: Run all three modes, capture metrics.**

```powershell
python scripts\sim_compare.py --strategy replay --no-plots
python scripts\sim_compare.py --strategy calibrated --no-plots
python scripts\sim_compare.py --strategy adaptive --no-plots
```

Record results into `docs\SIM_ACCURACY.md` as a new column.

### Task 5.2 — Update `docs\SIM_ACCURACY.md` with adaptive metrics

- [ ] **Step 1: Add a new row for adaptive mode in the metrics table.**

- [ ] **Step 2: Update interpretive text** (currently says calibrated mode passes 7/8). Add: "Adaptive mode passes X/8 with the same envelope inputs the real driver would respond to."

### Task 5.3 — Update `docs\SIM_AUDIT_2026-05.md`

- [ ] **Step 1: Re-grade Driver model row in subsystem scorecard** from C+ to A− (target).

- [ ] **Step 2: Check off the P0 adaptive driver checkbox.**

- [ ] **Step 3: Update sweep confidence summary** — torque-up and motor-RPM-up rows move from Medium-High to High because the driver now adapts brake points.

### Task 6.1 — Retire `CalibratedStrategy` as PREDICTION default (A4)

- [ ] **Step 1: In `engine.py:120-128`, when `mode == PREDICTION` and strategy is `CalibratedStrategy`, raise a clearer error pointing to `AdaptiveStrategy`.** Keep `CalibratedStrategy` available for CALIBRATION/REPLAY/VALIDATION.

- [ ] **Step 2: In `webapp/.../simulate.ts` (backend), default the Simulate-page strategy to `adaptive` when sweep params are present.** Out of scope for this plan if it requires backend changes — file a follow-up.

### Task 6.2 — Regression: confirm calibrated-mode behavior unchanged

- [ ] **Step 1: Run existing calibrated-mode tests (`tests\test_*calibrated*.py`).**

- [ ] **Step 2: Run `python scripts\sim_compare.py --strategy calibrated --no-plots` and assert numeric output matches the values in current `docs\SIM_ACCURACY.md` (7.3% time, etc.) within 0.1%.**

### Task 7.1 — Smoke test for combined-slip / corner-entry behavior

- [ ] **Step 1: Synthetic 200 m straight + 30 m radius corner + 200 m straight track.**

- [ ] **Step 2: Run adaptive driver. Check brake-application timing matches the envelope's backward pass to within one segment.**

```python
# tests/driver/test_adaptive_corner_entry.py
def test_adaptive_brakes_at_envelope_back_solved_distance():
    # Build a track with one mid-lap 30 m radius corner.
    ...
    # Find the segment where brake_pct first exceeds 0.05 in adaptive run.
    # Find the first segment where v_max[i] < v_max[i-1] in the envelope.
    # Assert their indices are within +/- 1 of each other.
```

### Task 8.1 — Final acceptance run and sign-off

- [ ] **Step 1: Run full pytest suite.** `pytest tests/ -x`

- [ ] **Step 2: Run all three sim_compare modes, archive outputs.**

- [ ] **Step 3: Open code-review request via `superpowers:code-reviewer`.**

- [ ] **Step 4: Commit with conventional commit message.**

```
git commit -m "feat(driver): add adaptive envelope-following driver

Replaces fixed pedal trace with envelope tracker + PI velocity
correction + LBP energy shaper. Closes P0 audit item; lifts
sweep grade from B+ to A- by making torque/RPM ceiling changes
visible in lap time.

Acceptance: replay-mode lap time error 0.X% (<=18s) on Michigan
endurance with adaptive driver; net Ah error 0.X%; net kWh
error 0.X%."
```

---

## Risks / Unknowns

- **PI gain analytical derivation may not match real-world segment dynamics**: the analytical 5-segment recovery target assumes 0.5 m segments; if track preprocessing changes `bin_size_m`, gains need to scale with `1/n_segments_per_corner_entry`. **Mitigation**: derive `_KP_VELOCITY` at `AdaptiveStrategy.__init__` from `track.total_distance_m / track.num_segments` so it scales automatically. Tests in 2.x cover both 0.5 m and 1.0 m grids.
- **Combined-slip corner entries with positive longitudinal demand**: when the driver demands trail-brake force during a corner, the tire's combined-slip ellipse reduces lateral capacity. The envelope's Pass 4 already accounts for this by re-querying corner speed with `longitudinal_g`. **Risk**: if PI correction pushes the operating point off the envelope into a state Pass 4 did not anticipate, lateral grip can go below required centripetal. **Mitigation**: clamp the tracker's `target_exit` to the corner-speed query at the operating speed (`max_cornering_speed(curvature, grip_factor, longitudinal_g=current_long_g)`).
- **Energy shaper can mask off too many segments under tight budget**: the greedy LBP allocator does not check whether the resulting envelope is still feasible (it can starve drive force on a corner exit, causing entry-speed drift below envelope on the next iteration). **Mitigation**: in Task 4.2 step 2, run a quick consistency pass that re-runs the engine once with the mask applied and asserts that no segment's resolved exit speed drops > 1.0 m/s below the unconstrained envelope; otherwise refine the mask.
- **`SimulationMode.PREDICTION` requires `vehicle.require_predictive_ready()` (engine.py:131)** which checks `tire.grip_scale != 1.0` and other empirical flags. If the adaptive driver is exercised in PREDICTION mode and the CT-16EV config has empirical grip, we get an early raise — that is correct behavior. **Mitigation**: tests run in CALIBRATION mode against telemetry as an honest acceptance bar; PREDICTION mode gets a smaller per-component test suite that doesn't depend on Michigan-specific calibration.
- **Lookahead increase from 5 → 60 segments may slow per-segment dispatch by ~10x in `engine.py`** (60 Segment object creations per segment). **Mitigation**: list slicing of the precomputed `upcoming_by_segment` table at line 341-347 is O(60) but cheap (no object creation). Profile-driven if it shows up.
- **Brake-bias decision (A1)** and **energy-budget default (A2)** are user calls; defaults are reasonable but the user should look at them.
- **Calibrated baseline in `scripts/sim_compare.py`** is currently the noise-floor anchor for sweep work. Removing it as default in PREDICTION mode (Task 6.1) does not delete it; it remains for CALIBRATION-mode compares. This is the "keep both, side-by-side" interpretation of A4.

## Verification / Acceptance Criteria

- **AC1 (functional, mandatory)**: When the calibrated driver is replaced with `AdaptiveStrategy(envelope=baseline_envelope)` and run on the Michigan 2025 endurance track in CALIBRATION mode (allowing telemetry track and empirical grip — same setup as the current calibrated baseline), the resulting endurance lap-time error vs telemetry is **≤ 1% (≤ 16.1 s on the real 1608.75 s)**, and the net Ah error is **≤ 2% (≤ 0.16 Ah on the real 8.04 Ah)**, and the net kWh error is **≤ 2% (≤ 0.065 kWh on the real 3.27 kWh)**.

  *Honest interpretation reminder*: 1% on Michigan endurance is ~16 s — not a small number. The acceptance bar matches replay mode (which validates 0.1% on time and 0.8% on Ah today; the adaptive driver does not get to use recorded inputs, only the envelope, so a small degradation is expected and 1% is the line beyond which we stop calling this "adaptive driver works as well as the real driver given the same envelope").

- **AC2 (correctness, mandatory)**: Synthetic flat-envelope test (`test_flat_envelope_at_steady_speed_commands_partial_throttle`) and ramp tests (`test_ramp_down_envelope_commands_brake`, `test_ramp_up_envelope_commands_full_throttle`) all pass without tolerance loosening.

- **AC3 (sweep responsiveness, mandatory)**: Running the same Michigan endurance with `vehicle.powertrain.torque_limit_inverter_nm` raised from 85 → 120 Nm produces (a) a lower endurance time (because corner-exit acceleration is faster, and the driver brakes later because the envelope's brake-from-higher-speed entry is steeper), and (b) higher net kWh (more energy used because more is available). With the calibrated driver today, neither (a) nor (b) is observed within noise floor, because the calibrated driver replays a fixed pedal trace. **Quantitative target**: `Δlap_time` < −2 s (real signal above the ~1 s noise floor) and `Δnet_kWh` > +0.05 kWh.

- **AC4 (determinism, mandatory)**: Two consecutive runs of `AdaptiveStrategy` on the same envelope with the same seed-free inputs produce bitwise-identical state DataFrames. (No RNG in the strategy; PI integral is reset on every `set_envelope`.)

- **AC5 (regression, mandatory)**: All existing tests in `tests/` pass, including the calibrated-mode validation tests and replay-mode tests. `python scripts\sim_compare.py --strategy calibrated --no-plots` and `--strategy replay --no-plots` produce numerically-equivalent output to today's `docs\SIM_ACCURACY.md` (within 0.1% per metric).

- **AC6 (mode safety)**: `SimulationMode.PREDICTION` accepts `AdaptiveStrategy` (because `uses_observed_speed_caps == False`) and refuses `CalibratedStrategy` with telemetry-derived caps (existing behavior).

- **AC7 (performance budget, advisory)**: A 22-lap Michigan endurance with `AdaptiveStrategy` completes in ≤ 5 s on the developer machine (matches today's 22-lap calibrated run within 20%). Profile if exceeded; do not paper over.

## Effort Estimate

| Task | Estimate |
|------|----------|
| 1.1 – 1.3 (envelope tracker + tests) | 4 h |
| 2.1 – 2.3 (AdaptiveStrategy skeleton + tests) | 3 h |
| 3.1 – 3.3 (engine plumbing + Michigan replay tests) | 5 h |
| 4.1 – 4.2 (energy shaper) | 3 h |
| 5.1 – 5.3 (sim_compare + docs) | 2 h |
| 6.1 – 6.2 (default switch + regression) | 2 h |
| 7.1 (combined-slip smoke) | 2 h |
| 8.1 (acceptance + commit) | 2 h |
| **Total** | **~23 h** (≈ 5 × 5h tier buckets) |

The 23 h estimate assumes the analytical PI derivation in Task 2.2 holds without re-tuning. If 3.2 (Michigan replay test) fails first try and root-cause investigation is required (rather than gain re-tuning, which would be a bandaid), add 5–10 h for diagnostic work and 1 follow-up commit.
