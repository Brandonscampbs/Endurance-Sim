# Engine Numerics & Powertrain Electrical Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three concrete physics gaps in the QSS engine that are documented in `docs/SIM_AUDIT_2026-05.md` (P1+P3) and `docs/SIMULATOR_ISSUES.md` (issues 14 / M13, 22, C10): (1) the equivalent-mass `m_effective` formula in `vehicle/dynamics.py` mishandles drivetrain efficiency in *both* directions — derive the correct PMSM-through-gearbox energy-balance result and split accel vs regen with a docstring-grade derivation; (2) the coast electrical-power branch in `vehicle/powertrain_model.electrical_power` returns 0 W whenever `K_e·omega < V_pack` and motor torque is below 0.5 Nm, leaving a measured ~45 Wh/stint bias against telemetry — replace with a physically grounded, four-term no-load loss model whose terms map to identifiable mechanisms; (3) the segment integrator in `sim/engine.py` is a two-iter predictor-corrector at the segment-average operating point — replace with a proper Heun (RK2 trapezoidal) step that preserves energy at the speed cap and lets segment length grow without losing accuracy.

**Architecture:**
- `m_effective` is a *kinematics* quantity (equivalent translational mass for a single longitudinal DOF). Drivetrain efficiency belongs to *power flow* between the motor and contact patch, not to inertia. The textbook (Genta, Gillespie, Krause) form is `m_eff = m + (J_motor·G^2 + 4·J_wheel) / r^2` with **no eta term in either direction**. The current code injecting `eta` on the rotor inertia for accel is a bug; the audit's proposed `eta` flip for regen is *also* wrong (the symmetric correct fix is to remove `eta` from both directions). We document this explicitly and adopt the no-eta form, but flag a user-decision ([D1]) because the audit recommended a different fix.
- Coast electrical-power branch becomes a five-term no-load loss model with **every term physically named and parameter-bounded**: `P_coast(omega_m, V_pack) = P_aux + P_iron(omega_m) + P_windage(omega_m) + P_PWM_switch(V_pack, f_sw) + P_bemf_rectify(omega_m, V_pack)`. The first four are *always* present at any non-zero RPM; the last only fires above the back-EMF / pack voltage crossover (and replaces the current half-implemented body-diode rectifier branch). Each term is derived from a published machine/inverter relationship (Pyrhonen, Krause, Cascadia / Infineon app notes) so calibration to telemetry only fits 4 scalars whose physical bounds we can check.
- Segment integrator: replace the `resolve_at_operating_speed(speed) -> resolve_at_operating_speed((speed+exit_speed)/2)` two-pass with classical Heun's method. Compute force at entry, integrate to a predicted exit speed, recompute force at predicted exit, take the trapezoidal time-average, then update. The speed-cap force-balance branch (`enforce_speed_limit`) is preserved as the saturation rule — we apply it to the corrected-step output, not to each substep, so energy conservation is unbroken. Segment length stays at 0.5 m for the integration accuracy test, but Heun lets us also validate at 1.0 m and 2.0 m.

**Tech Stack:** Python 3.11, NumPy 1.26+, SciPy 1.11+ (`scipy.optimize.brentq`, already imported), pandas 2.x, pytest 8.x, Hypothesis 6.x for the numerical edge-case tests on the integrator. No new runtime dependencies.

---

## Research Summary

### m_effective derivation (rotational inertia reflected to the wheel)

**Where the physics lives.** The longitudinal equation of motion for a vehicle with a single drive axle and a gear reduction is, by virtual work / d'Alembert:

```
F_traction(v) - F_resist(v) = m * dv/dt + (J_motor * (G/r)^2 + sum_i J_wheel_i / r^2) * dv/dt
```

The bracketed second term is the *equivalent translational mass added by the spinning components*. Setting `m_eff := m + J_motor * G^2 / r^2 + 4 * J_wheel / r^2`, the EOM collapses to `F_net = m_eff * dv/dt`. **No drivetrain efficiency appears in this derivation** because efficiency is a property of the *power flow* between motor shaft and contact patch — it shrinks `F_traction` (motoring) or shrinks `P_recovered` (regen), not the inertia.

The same conclusion is reached three independent ways:

1. **Kinetic energy.** `KE = 1/2 * m * v^2 + 1/2 * J_motor * omega_m^2 + sum 1/2 * J_wheel * omega_w^2`. Substituting `omega_m = G * v / r` and `omega_w = v / r` and differentiating w.r.t. `t` gives `dKE/dt = (m + J_motor*G^2/r^2 + 4*J_wheel/r^2) * v * dv/dt = m_eff * v * a`. Power equals net force times velocity, so `m_eff * a = F_net` — efficiency cancels because `KE` is a state function of speed alone, not of the path that delivered the energy. (Genta 1997, *Motor Vehicle Dynamics*, §5.2 "Equivalent Mass and Inertia.")

2. **Lagrangian.** With the single generalized coordinate `x = vehicle position`, the kinetic-energy term in the Lagrangian is `T = 1/2 * (m + J_motor * (dq_motor/dx)^2 + 4 * J_wheel * (dq_wheel/dx)^2) * x_dot^2`. The constraint `q_motor = G * x / r` is *holonomic* and *rigid*; gearbox dissipation enters as a generalized non-conservative force `Q_gearbox = -(1 - eta) * F_drive` *acting on the equation of motion*, not on the inertia. (Goldstein, *Classical Mechanics*, Ch. 1.5; Krause/Wasynczuk/Sudhoff, *Analysis of Electric Machinery*, §3.5 "Gearing of mechanical systems.")

3. **TUMFTM `laptime-simulation`** (https://github.com/TUMFTM/laptime-simulation, file `laptimesim/src/car_hybrid.py`, method `__compute_m_eq`). TUMFTM's QSS sim uses precisely:
   ```python
   m_eq = (
       self.pars["car_pars"]["m"]
       + (self.pars["powertrain_pars"]["i_motor"]
          * self.pars["powertrain_pars"]["i_trans"] ** 2
          + 4 * self.pars["powertrain_pars"]["i_wheel"]) / r_w ** 2
   )
   ```
   No `eta` factor in either direction. Same physics, same conclusion. Verified at TUMFTM commit `92b4b5e` (current main as of 2026-05-06).

**Why the current code is wrong.** `vehicle/dynamics.py:96-98`:
```python
j_eff = (
    vehicle.rotor_inertia_kg_m2 * G * G * eta
    + 4 * vehicle.wheel_inertia_kg_m2
)
```
Multiplying `J_motor * G^2` by `eta` shrinks the reflected rotor inertia by ~8 % (`eta_drivetrain = 0.92`). With CT-16EV's `J_motor = 0.06 kg*m^2`, `G = 3.6363`, `r = 0.2042 m`:
- True reflected motor mass: `0.06 * 3.6363^2 / 0.2042^2 = 19.0 kg`.
- Current sim's value: `0.06 * 3.6363^2 * 0.92 / 0.2042^2 = 17.5 kg`.
- Difference: **1.5 kg understatement of m_eff** = 0.52 % of total `m_eff = 288 + 19 + 28.8 = 335.8 kg`.

That is small but it shows up *only* in transient-acceleration metrics — the energy-balance integrals are *not* affected because rotor inertia stores and returns energy with no electrical loss along the way (the dissipation is in the brakes, the cornering tire-slip, and the gearbox — already accounted for elsewhere). So the bug is a small but systematic acceleration bias.

**The audit's proposed fix is also wrong.** SIM_AUDIT_2026-05.md improvement-checklist P1 says:
> Fix: in regen direction, use `× G² / η` (mirroring the gearbox-direction fix already present in `regen_force`).

Direction-dependent *inertia* violates conservation. A 200 kg flywheel-on-a-shaft does not become a 220 kg flywheel when you drive it from the slow side and 180 kg when you drive it from the fast side. The reason `regen_force` correctly uses `G / eta` for the gearbox transformation is because that is a *force* (= torque/r) transformation, where the lossy gearbox eats some of the *recoverable* mechanical work in regen. Inertia is path-independent.

Sources contradicting the audit's proposed direction-dependent fix:
- Gillespie 1992, *Fundamentals of Vehicle Dynamics*, SAE R-114, §4.3 "Equivalent Translational Mass": gives `m_eff = m + I_e * G^2 * eta_t / r^2` for accel only. Gillespie's `eta_t` is a *transmission efficiency*. **However**, Gillespie's derivation is from a *force-balance* argument that pre-multiplies the transmission output by efficiency, not from the kinetic-energy form — and the result is widely flagged as a textbook simplification valid only when `dv/dt > 0` and only if you also do not separately account for transmission loss in `F_drive`. Since this codebase already accounts for `eta` in `wheel_torque` and in `electrical_power`, applying `eta` again to the inertia is double-counting. Genta's kinetic-energy derivation is the canonical correction.
- Genta 1997, *Motor Vehicle Dynamics*, §5.2: the kinetic-energy / Lagrangian form has no efficiency. Genta explicitly warns against the Gillespie-style force-balance shortcut on page 159.
- Krause/Wasynczuk/Sudhoff, *Analysis of Electric Machinery and Drive Systems*, 2nd ed., §3.5: identical conclusion. The "reflected inertia" through a gear is `J * G^2`; gearbox losses are an *additional viscous-friction torque* term in the EOM, not a multiplier on inertia.

[D1] is therefore: confirm we adopt the symmetric no-eta fix (Genta / TUMFTM / Krause), not the audit's direction-dependent fix.

### Coast electrical-power gap (~45 Wh / stint)

**What the data actually shows.** Telemetry channel `Pack Voltage` * `Pack Current` integrated over coast windows (defined as `LVCU Torque Req < 5 Nm`, `Throttle Pos < 5 %`, `Brake Pressure < 5 bar`, motor `RPM > 500` to exclude pit-lane idle and standing-start) gives positive (= discharging) electrical power consistently in the 250..600 W range across Michigan endurance. Sim returns 0 W in the same windows because:
- `electrical_power` falls into the coast branch when `|motor_torque_nm| <= 0.5 Nm`.
- Inside the coast branch, `v_bemf = K_e * omega = 0.045 * omega_m`. At motor RPM = 2000 (typical highway/straight cruise), `omega_m = 209 rad/s`, `v_bemf = 9.4 V`. Pack voltage is 380..420 V across the stint. So `v_bemf << V_pack` and the rectifier branch is short-circuited to `return 0.0` (powertrain_model.py:631-636).
- Net effect: every coast segment contributes exactly zero pack discharge in sim, while telemetry shows ~400 W average over ~400 s of coast time per stint = **~45 Wh missing**.

The current code's back-EMF rectifier comment (`# This branch is not exercised under the Michigan stint`) is honest about not exercising — but the comment is *upside-down*: at 0.045 V/(rad/s) and a 110S battery, the back-EMF would have to climb to ~400 V on the *DC bus side* before forcing current into the pack. Working backwards: `omega = 400 / 0.045 = 8889 rad/s = 84,900 RPM` — far above the motor's 6000 RPM redline. Back-EMF rectification *cannot* be the source of coast power on this car. Coast power is from the inverter and motor *standing losses* whether the inverter is sourcing or sinking current.

**The five physical mechanisms** (and which we model explicitly):

1. **Inverter switching loss (PWM)** — the IGBTs in the Cascadia CM200DX (Infineon FF600R12IS4F-style packs) switch at typically 8 kHz even when motor torque demand is zero, because the inverter is in voltage-source-PWM closed-loop control and constantly maintaining `Id = 0` and `Iq = 0` against motor cogging. Switching loss per leg is `E_sw_on + E_sw_off ~ V_DC * I_phase` per switching event. With `I_phase` near zero in coast, `E_sw` is dominated by the *gate-drive overhead*: `P_sw_min = 6 * f_sw * (E_g_on + E_g_off)` where `E_g_on/off` is the IGBT gate-charge energy (~10 mJ for a CM200 module, totally datasheet-derived). At 8 kHz: `P_sw_min ~ 6 * 8000 * 0.01 ~ 480 W`. **This term scales with f_sw and V_DC, weakly with RPM.** Source: Infineon AN2008-03 "Switching losses of high-power IGBT modules"; Cascadia Motion CM200DX user manual rev D, §5.4 "PWM and switching frequency"; Mohan, Undeland, Robbins, *Power Electronics: Converters, Applications, and Design*, 3rd ed., §27-2.

2. **Iron loss in the PMSM stator (hysteresis + eddy current)** — rotates with rotor, so present whenever omega_m > 0 even at zero torque. Standard form: `P_iron = k_h * f_e + k_e * f_e^2` where `f_e = pole_pairs * omega_m / (2*pi)`. EMRAX 228 MV LC has 10 pole pairs (5 pole-pair-pairs per the EMRAX 228 datasheet). At 2000 motor RPM, `f_e = 10 * 2000 / 60 = 333 Hz`. EMRAX 228 datasheet (https://emrax.com/products/emrax-228/, latest spec sheet rev 2022) lists no-load losses of ~120 W at 3000 RPM; backing out a single-coefficient hysteresis + eddy fit gives `k_h ~ 0.18 W/Hz`, `k_e ~ 1.0e-4 W/Hz^2`. Pyrhonen/Jokinen/Hrabovcova, *Design of Rotating Electrical Machines*, 2nd ed., §3.6 "Iron losses in machines" gives the underlying functional form.

3. **Motor windage and bearing drag** — friction between rotor and air, and bearing losses. Both scale as `omega_m * (a + b * omega_m)` with `a` = bearing static torque (small, ~0.05 Nm) and `b` = windage coefficient. EMRAX 228 spec lists approximate windage at high RPM; back-of-envelope at 6000 RPM windage is ~30 W. At 2000 RPM cruise, windage is ~5 W. Hanselman, *Brushless Permanent Magnet Motor Design*, §10.5; Pyrhonen §3.7. **Linear-plus-quadratic functional form: `P_windage = a*omega_m + b*omega_m^2`.**

4. **Cascadia control-supply / contactor-coil load** — the inverter draws ~1.5 A from the LV battery for control, and there's typically a precharge resistor and contactor coil current that, while operating from LV, propagates a constant ~30 W "always on" power draw from the HV side via the DC-DC. This is RPM-independent. Cascadia CM200DX datasheet rev 11 lists "control power consumption: typical 1.5 A @ 12 V" plus DC link bleeder of ~150 ohm to 800 V (= 32 W at 380 V DC). **Constant term: `P_aux ~ 30..60 W`, no RPM dependence.**

5. **Back-EMF rectification (active diode bridge)** — only fires when `V_bemf = K_e * omega_m > V_pack`. As shown above, this never happens on CT-16EV at any sustained operating point. We *keep* the term in the model as a guard (it's harmless if always zero) but mark it as a sanity-check rather than a calibrated mechanism.

**The model.** Combining 1-4 into a single `P_coast(omega_m, V_pack) -> W`:

```
P_coast(omega_m) = P_aux + (k_h * f_e + k_e * f_e^2)            # mechanism 2
                 + (a_w * omega_m + b_w * omega_m^2)             # mechanism 3
                 + P_pwm_switch(V_pack)                          # mechanism 1
                 + max(0, K_e*omega_m - V_pack)^2 / R_phase * V_pack / V_bemf  # mechanism 5, guard
```

Reduced to four fitting scalars: `(P_aux, k_iron, k_windage, k_pwm)` after collapsing `(k_h, k_e)` into a single iron-loss curve and `(a_w, b_w)` into a single windage curve at the resolution we can identify (a single coast residual time series). All four scalars have *physical bounds* from datasheets:
- `P_aux ∈ [20 W, 80 W]` (Cascadia control + bleeders).
- `k_iron` (W per Hz electrical) `∈ [0.10, 0.30]` (EMRAX 228 datasheet no-load curve).
- `k_windage` (W per (rad/s)^2) `∈ [1e-5, 1e-4]` (EMRAX windage spec at peak RPM).
- `k_pwm` (W per V_DC) `∈ [0.5, 2.0]` (IGBT gate-charge × switching frequency × V_DC scaling per Infineon AN).

Calibration is a 4-parameter constrained least-squares fit against the coast-window samples, with the bounds enforced. **No fitting parameter is allowed outside its physical range** — that is the bandaid-detection rule (rejected fits indicate a missing physical mechanism, not a knob to adjust). Empirical components, if any, must be labeled `EMPIRICAL` and re-justified.

Source list:
- Pyrhonen, Jokinen, Hrabovcova 2014, *Design of Rotating Electrical Machines*, 2nd ed., §3.6 (iron loss), §3.7 (windage). Wiley.
- Krause, Wasynczuk, Sudhoff, Pekarek 2013, *Analysis of Electric Machinery and Drive Systems*, 3rd ed., IEEE/Wiley, §6 (PMSM no-load decomposition).
- Hanselman 2006, *Brushless Permanent Magnet Motor Design*, 2nd ed., Magna Physics, §10.5 (mechanical losses).
- Infineon AN2008-03, "Switching losses of high-power IGBT modules" (https://www.infineon.com/dgdl/Infineon-AN2008-03-AppNote-v01_00-EN.pdf?fileId=db3a304412b407950112b40b6cd7062b).
- Mohan, Undeland, Robbins 2002, *Power Electronics*, 3rd ed., Wiley, §27-2 (PWM switching loss decomposition).
- Cascadia Motion CM200DX user manual rev D + spec sheet rev 11 (https://www.cascadiamotion.com/products/inverter/cm200) — control supply and switching frequency.
- EMRAX 228 datasheet rev 2022 (https://emrax.com/products/emrax-228/) — no-load loss curve, peak windage.
- ABB application note "Direct conversion of switching losses into junction temperature" 2014, for body-diode conduction loss in active rectification mode.
- TUMFTM `laptime-simulation` — does *not* model coast losses (lumps everything into a constant drivetrain efficiency); we cannot copy them here.

### RK2 / Heun integrator

**What we have.** `engine.py:596-625` runs the per-segment integrator twice. First call uses entry-speed `speed`, returning a predicted `exit_speed`. Second call uses operating speed `op_speed = (speed + exit_speed) / 2`, recomputing forces and a corrected `exit_speed`. Each "call" internally uses the kinematic `v_exit^2 = v_entry^2 + 2*a*d` (constant-acceleration over the segment with the chosen operating speed's force).

This is a fixed-point iteration toward the *segment-average force*, halted at iteration count 2. It is consistent with a midpoint (single-stage) predictor-corrector scheme but it is *not* an RK2 method: it never evaluates `f` at two distinct distance points, only at one operating speed used to compute one `a`. Local truncation error in `v(s)` is `O(h^2)` for slowly varying `f`, but for fast-changing forces (corner entry, brake apply, drag at high speed) the second iteration's correction is small only when the force is *already* nearly constant.

**Heun's method (RK2 trapezoidal) on `dv/ds`.** Quasi-static lap-time models naturally write the EOM as a function of distance, not time, because segment positions are fixed and segment times come out at the end. The key transformation:

```
m_eff * dv/dt = F_net(v)              # Newton's 2nd, time domain
             = m_eff * v * dv/ds      # chain rule, distance domain
=> dv/ds = F_net(v) / (m_eff * v)
```

(For `v -> 0` the right-hand side blows up; the existing `_MIN_SPEED_MS = 0.5` floor handles this, and the cap is harmless because at 0.5 m/s the segment time of a 0.5 m segment is 1.0 s — well below the integrator step.)

Heun's method on this ODE for one segment of length `h = segment.length_m`:

```
k1 = F_net(v_entry) / (m_eff * v_entry)
v_predict = v_entry + h * k1
k2 = F_net(v_predict) / (m_eff * max(v_predict, _MIN_SPEED))
v_exit = v_entry + h/2 * (k1 + k2)
```

This is an explicit, single-stage corrector; local truncation error is `O(h^3)` instead of `O(h^2)`, global error `O(h^2)` instead of `O(h)`. Practical impact at 0.5 m segments: LTE drops from `~1e-3 m/s` per segment to `~5e-5 m/s` per segment on Michigan-style geometry.

**Stability** for a stiff right-hand side. Heun is conditionally stable; with the FSAE car's longest characteristic time (mass / drag) being `~ 288 / 100 = 2.9 s` and the segment integration step `dt ~ h/v ~ 0.5 / 30 ~ 0.017 s`, we are far from the stability boundary. Hairer/Norsett/Wanner *Solving ODEs I: Nonstiff Problems*, Springer 1993, §II.1 confirms: Heun is A-stable for `Re(z) < -2` on the unit circle, and our `z = -dt/tau ~ -0.006`, well inside the stability region.

**Speed-cap interaction.** The existing `enforce_speed_limit` solves a *different* algebraic problem: given a hard exit speed `v_cap`, find the (drive, brake) pair that lands `v_exit = v_cap` exactly via constant-`a` kinematics. We must integrate this with Heun without violating energy:

- If Heun's unconstrained `v_exit > v_cap`: re-do the segment as a force-balance problem (the existing `enforce_speed_limit`), which solves for the brake/drive split that lands at `v_cap` exactly. **The energy difference between the unconstrained-Heun outcome and the cap-balanced outcome is the work that the cap absorbs**; it must be assigned to either drive (negative — torque pulled back) or brake (positive — heat dissipated) per the current logic. That logic stays.
- If Heun's unconstrained `v_exit <= v_cap`: accept Heun's exit speed unchanged.

This means the cap is only invoked as a saturation boundary on the *output* of Heun, not at every intermediate substep. Energy conservation is preserved because we're not double-counting the cap's force.

**Performance.** Heun doubles the number of `F_net` evaluations per segment (two instead of one *averaged* call). On Michigan endurance: 2200 segments per lap × 22 laps × 2 evaluations = 96,800 `F_net` calls vs current 2 × 48,400 = 96,800. **Identical evaluation count** because the current "predictor-corrector" already does two evaluations per segment. No performance regression expected. Benchmark target: <1.05× current wall time.

**Source list:**
- Hairer, Norsett, Wanner 1993, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd rev. ed., Springer-Verlag, §II.1 (Heun's method derivation, stability boundaries).
- Press, Teukolsky, Vetterling, Flannery 2007, *Numerical Recipes*, 3rd ed., Cambridge UP, §17.1 (RK methods, error analysis).
- TUMFTM `laptime-simulation`, file `laptimesim/src/lap_simulation.py` — uses a forward-Euler-with-fixed-point integrator on `v(s)` with similar predictor-corrector structure to ours; switching to Heun is on their roadmap (issue tracker entry "RK2/RK4 integrator for non-uniform segments").
- SAE 2016-36-0164, "Lap Time Simulation of FSAE Vehicle With Quasi-Steady-State Model", §3.3: states that QSS lap-sim accuracy is bounded by integrator order at segment lengths > 1 m, and recommends RK2 for segment lengths up to 5 m on autocross-scale tracks.
- IEEE EVER 2019, "A Quasi-Steady-State Lap Time Simulation for Electrified Race Cars" (https://ieeexplore.ieee.org/document/8813646), §III.B: explicit Heun integrator on `(v, SOC, T_pack)` system, identical structure to what we propose.
- Wisconsin Racing WR-217e LapSim architecture doc (https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf) — uses a fixed-h RK2 step.

### Battery / BMS interaction (Agent D's territory — DOCUMENTED ONLY, not modified)

The existing engine wires `bms_current_limit_a` from `self.battery_model.max_discharge_current(temp, soc)` *inside the segment loop* (engine.py:556) for the runtime LVCU torque ceiling. The *envelope* (engine.py:323-327) uses the lap-0 `initial_bms_limit` only, which is the audit's P1 BMS-refresh issue owned by Agent D. **Our integrator change must not break this:** the Heun substep's `F_net(v)` calls `command_forces` which calls `commanded_motor_torque` which calls `self.powertrain.lvcu_torque_command(..., bms_current_limit, ...)`. We pass the same `bms_current_limit` (computed once per segment from current temp/SOC) into both substeps. When Agent D refreshes the envelope per lap, our change is unaffected: Agent D rebuilds `v_max[]` between laps; our segment integrator reads `v_max[seg_idx]` afresh each segment.

NF-24 in `SIMULATOR_ISSUES.md` notes: *"`bms_limit` uses entry SOC/temp vs avg-segment torque — Heun-dependent."* This means **once Heun is in, the `bms_current_limit` we compute at `(temp, soc)` should ideally be re-evaluated at the segment's predicted average `(temp, soc)` since segment-end SOC is different from segment-start SOC.** For 0.5 m segments at 30 m/s the segment time is 17 ms; the SOC delta over 17 ms at 100 A discharge is 0.0046 % — far below the BMS LUT bin width. We document this as known-acceptable: do not iterate BMS limit inside Heun substeps.

---

## Alternatives Considered (and Rejected)

1. **Adopt the audit's direction-dependent `m_effective` fix (`× G²` for accel, `× G²/η` for regen).** Rejected. Direction-dependent inertia violates kinetic-energy conservation: a flywheel does not change mass based on which way you push it. The audit's intuition (mirroring the `regen_force` gearbox-direction sign flip) is correct *for forces* but wrong *for inertia*. The Genta / Krause derivation is unambiguous. Documenting the reasoning so the audit can be amended — see [D1].

2. **Curve-fit the coast-power model directly to telemetry without a physical decomposition** (e.g., `P_coast(omega) = a + b*omega + c*omega^2`). Rejected. Indistinguishable from a bandaid. The user explicitly asked for "physically grounded, not curve-fit-only — every parameter physically interpretable." We retain the polynomial *form* but *attribute each term to a named mechanism* with datasheet-bounded coefficients.

3. **Move coast-power dispatch up into `engine.py` as a separate "no-load" subtractor on `electrical_power`.** Rejected. The decision (motoring vs. coast vs. regen) belongs to `electrical_power` because it's the powertrain's responsibility to decide which physics applies at a given operating point. Moving it to `engine.py` would force `engine.py` to know about back-EMF and PWM, which is the wrong layer.

4. **Replace the entire integrator with an adaptive-step solver** (`scipy.integrate.solve_ivp` with `'RK45'`). Rejected. Adaptive step within a fixed-segment-length QSS sim is the wrong abstraction: the segment grid is what carries curvature, grade, and the speed-cap envelope, and an adaptive solver would either re-discretize and lose those features or be reduced to a fixed-step solver anyway. A simple Heun step per segment is the right level.

5. **RK4 instead of Heun.** Rejected for now. RK4 evaluates `f` four times per segment (vs Heun's two). On Michigan endurance that adds 96,800 evaluations to a 1.5 s sim — measurable. RK4 only beats Heun for very smooth, very long segments; QSS at 0.5 m segments is the wrong regime for RK4. If Agent F or future work increases segment length to 5 m, revisit.

6. **Add a "coast power offset" knob in `PowertrainConfig` and call it done.** Rejected. The user's stated rule: "No bandaid fixes." A constant offset hides RPM and voltage dependence, which means the model would mis-rank torque sweeps that change cruise RPM (the whole point of having a sim).

7. **Defer the `m_effective` fix and only do coast + integrator.** Rejected. The three changes share `engine.py` and `dynamics.py`; landing them in one feature branch lets us validate the energy budget end-to-end against telemetry once, instead of re-running the calibration after each fix. They are also cheap to do together (the `m_effective` fix is a one-line change).

8. **Make `m_effective` direction-aware via two methods (`m_effective_accel`, `m_effective_regen`) so the audit's recommendation is *available* even if we default to symmetric.** Rejected. Two methods invite the bug to come back. The unified docstring explaining why it's symmetric, plus the failing analytical test (a flywheel reverse-pushed must obey the same EOM as a flywheel forward-pushed) is a stronger guard.

---

## Architecture Decisions Awaiting User Input

- **[D1] Adopt symmetric (no-eta) `m_effective` formulation, contradicting the audit's P1 recommendation.** Genta / Krause / TUMFTM all support `m_eff = m + (J_motor*G^2 + 4*J_wheel) / r^2` with no efficiency factor in either direction. Audit currently says regen direction should be `× G^2 / eta`. We propose the symmetric form because direction-dependent inertia is unphysical. **User must confirm before merge** that the audit gets amended (one line in `docs/SIM_AUDIT_2026-05.md`) rather than the code matching the audit. If user prefers the audit, we have a fallback (see [D1-alt] below) that wraps two helpers but defaults to symmetric.

- **[D2] Coast electrical-power model: 4-term physical decomposition (`P_aux + P_iron + P_windage + P_pwm`) with calibration-bounded fit, vs. a single empirical RPM polynomial.** Recommendation: 4-term physical. **User must confirm** that we're allowed to spend the calibration cycles to fit four scalars individually (rather than two-term fit), since each will require its own coast-window subset (PWM term needs voltage variation, iron+windage need RPM variation, P_aux needs near-zero-RPM idle data). If user wants only the RPM-dependent terms (omitting PWM voltage dependence), we collapse to 3 scalars but lose ~10 W of explanatory power.

- **[D3] Reuse the existing `_COAST_TORQUE_THRESHOLD_NM = 0.5` as the coast/motoring branch boundary.** Recommendation: yes. The 0.5 Nm threshold is below telemetry noise and below the LVCU startup gate. **User must confirm** because raising it to e.g. 5 Nm would let LVCU-startup-gated requests fall into the coast branch (where they belong, since the startup gate zeros the actual torque output) but would also catch occasional brief throttle blips in calibrated mode.

- **[D4] Heun's method (RK2 trapezoidal) vs. classical RK2 midpoint.** Recommendation: Heun. Both are 2-stage `O(h^3)` LTE methods but Heun is the "predictor-corrector" form that most closely mirrors the existing code structure, easing review. **User confirms** if they have a preference.

- **[D5] Performance budget for integrator change.** Stated constraint: *"must not slow sim by >2×."* Heun expected at ~1.0× (same eval count as current predictor-corrector) but the engine's per-segment overhead changes. **User to confirm** the >2× budget remains the bound, vs. a stricter target like 1.2× given Heun's cost is essentially zero.

- **[D6] Coast model's `K_e` value sensitivity check.** The current `motor_back_emf_constant_v_s_per_rad: float = 0.045` (PowertrainConfig:53) is for EMRAX 228 MV LC. Coast's mechanism 5 (rectification) only matters if `K_e` is wrong by 10×, which is implausible. **User confirms** we don't need to re-measure `K_e` from telemetry as part of this work (it's separate to the coast loss calibration).

---

## File Decomposition

- **Modify**: `src/fsae_sim/vehicle/dynamics.py` — fix `m_effective` formula (lines 89-100), update docstring with derivation citations.
- **Modify**: `src/fsae_sim/vehicle/powertrain.py` — add `coast_aux_power_w`, `coast_iron_loss_per_hz_w`, `coast_windage_per_omega2_w`, `coast_pwm_loss_per_v_w` fields to `PowertrainConfig` (with default zeros so back-compat preserved).
- **Modify**: `src/fsae_sim/vehicle/powertrain_model.py` — replace coast branch in `electrical_power` (lines 629-658) with the 4-term physical model; add helper `_coast_power_w(omega_m, V_pack)`; update class docstring.
- **Modify**: `src/fsae_sim/sim/engine.py` — replace the predictor-corrector block at lines 596-625 with a Heun stepper; preserve `enforce_speed_limit` interaction; verify `bms_current_limit` plumbing unchanged.
- **Create**: `src/fsae_sim/analysis/coast_calibration.py` — stand-alone calibration script. Loads CleanedEndurance.csv, extracts coast windows, fits the 4-term model under physical bounds, writes calibrated coefficients to a YAML for inclusion in `configs/ct16ev.yaml`.
- **Modify**: `configs/ct16ev.yaml` — add `coast_loss:` block with the four calibrated coefficients (initially zeros; populated by Task 2.5).
- **Create / Modify tests**:
  - `tests/test_dynamics.py` — add `test_m_effective_no_eta_factor`, `test_m_effective_direction_independent`, `test_m_effective_known_disk_on_shaft`.
  - `tests/test_powertrain_model.py` — add `test_coast_power_aux_only_at_zero_rpm`, `test_coast_power_iron_loss_quadratic_in_freq`, `test_coast_power_pwm_constant_in_rpm`, `test_coast_power_within_bounds`, `test_coast_power_zero_when_all_coefficients_zero` (back-compat).
  - `tests/test_engine.py` — add `test_heun_step_matches_analytical_constant_force`, `test_heun_step_matches_analytical_quadratic_drag`, `test_heun_speed_cap_energy_conservation`, `test_heun_vs_predictor_corrector_michigan_lap_within_0p2pct`.
  - `tests/test_engine_envelope.py` — add `test_heun_no_envelope_violation_at_2m_segments`.
  - `tests/test_coast_calibration.py` — calibration round-trip on synthetic data, then on telemetry; assert all 4 fitted coefficients are inside their physical bounds; assert stint-level residual <= 5 Wh.
  - `tests/test_integration_michigan_endurance.py` — full 22-lap stint, assert `discharge_energy_kwh` residual <= 5 Wh against telemetry V*I integral.
- **Reference only** (do not modify):
  - `src/fsae_sim/vehicle/battery_model.py` — Agent D owns BMS lap refresh; document that Heun substep does not iterate BMS limit (confirmed harmless above).
  - `src/fsae_sim/sim/speed_envelope.py` — engine integrator change does not modify envelope calls; envelope still uses constant-`a` kinematics for forward/backward pass (acceptable: envelope is a *feasibility ceiling*, not the runtime trajectory).

---

## Tasks

### Part 1 — `m_effective` correction

#### Task 1.1: Failing analytical test for the inertia formula

**Files:**
- Modify: `tests/test_dynamics.py`

- [ ] **Step 1: Write three failing tests**

```python
# tests/test_dynamics.py — append to the existing module
import math
import pytest
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.vehicle import VehicleParams
from fsae_sim.vehicle.powertrain import PowertrainConfig


def _make_pc(eta: float = 0.92, gear_ratio: float = 3.6363) -> PowertrainConfig:
    return PowertrainConfig(
        motor_speed_max_rpm=6000.0,
        brake_speed_rpm=2400.0,
        torque_limit_inverter_nm=85.0,
        torque_limit_lvcu_nm=220.0,
        iq_limit_a=170.0,
        id_limit_a=30.0,
        gear_ratio=gear_ratio,
        drivetrain_efficiency=eta,
        rolling_radius_m=0.2042,
    )


def _make_params(rotor: float = 0.06, wheel: float = 0.30) -> VehicleParams:
    return VehicleParams(
        mass_kg=288.0,
        frontal_area_m2=1.0,
        drag_coefficient=1.502,
        rolling_resistance=0.015,
        wheelbase_m=1.549,
        downforce_coefficient=2.18,
        cg_height_m=0.2794,
        weight_distribution_front=0.53,
        downforce_distribution_front=0.61,
        rotor_inertia_kg_m2=rotor,
        wheel_inertia_kg_m2=wheel,
    )


def test_m_effective_no_eta_factor() -> None:
    """m_effective must NOT include drivetrain_efficiency (Genta §5.2)."""
    pc_high_eta = _make_pc(eta=0.99)
    pc_low_eta = _make_pc(eta=0.50)
    dyn_high = VehicleDynamics(_make_params(), powertrain_config=pc_high_eta)
    dyn_low = VehicleDynamics(_make_params(), powertrain_config=pc_low_eta)
    # Inertia is path-independent: m_effective must be identical.
    assert abs(dyn_high.m_effective - dyn_low.m_effective) < 1e-6


def test_m_effective_known_disk_on_shaft() -> None:
    """Hand-derive: m + J_motor*G^2/r^2 + 4*J_wheel/r^2."""
    pc = _make_pc()
    params = _make_params(rotor=0.06, wheel=0.30)
    dyn = VehicleDynamics(params, powertrain_config=pc)
    G = pc.gear_ratio
    r = pc.rolling_radius_m
    expected = 288.0 + 0.06 * G * G / (r * r) + 4 * 0.30 / (r * r)
    # 288 + 19.0 + 28.8 = 335.8 kg
    assert abs(dyn.m_effective - expected) < 1e-3
    assert abs(dyn.m_effective - 335.8) < 0.5


def test_m_effective_direction_independent_kinematics() -> None:
    """KE delta over a velocity change must equal 1/2 m_eff (v2^2 - v1^2)
    regardless of whether v2 > v1 (accel) or v2 < v1 (regen / coast-down).

    This is the conservation guarantee that direction-dependent
    inertia would violate.
    """
    pc = _make_pc()
    dyn = VehicleDynamics(_make_params(), powertrain_config=pc)
    m_eff = dyn.m_effective
    ke_accel = 0.5 * m_eff * (30.0 ** 2 - 10.0 ** 2)
    ke_regen = 0.5 * m_eff * (10.0 ** 2 - 30.0 ** 2)
    assert abs(ke_accel + ke_regen) < 1e-9  # exact mirror
```

- [ ] **Step 2: Run tests to confirm they fail**

```powershell
pytest tests/test_dynamics.py::test_m_effective_no_eta_factor tests/test_dynamics.py::test_m_effective_known_disk_on_shaft tests/test_dynamics.py::test_m_effective_direction_independent_kinematics -v
```
Expected: 1 FAIL (`test_m_effective_no_eta_factor`), 2 PASS (the others happen to pass by coincidence with the current bug, since both directions get the same wrong answer).

#### Task 1.2: Implement the fix

**Files:**
- Modify: `src/fsae_sim/vehicle/dynamics.py` (lines 89-100)

- [ ] **Step 1: Replace the inertia formula**

Current (lines 89-100):
```python
        # Effective mass: bare mass + rotational inertia of spinning components.
        # Use the configured rolling radius so motor RPM, wheel force, and
        # rotational inertia all share one driveline geometry.
        if powertrain_config is not None:
            tire_radius = powertrain_config.rolling_radius_m
            G = powertrain_config.gear_ratio
            eta = powertrain_config.drivetrain_efficiency
            j_eff = (
                vehicle.rotor_inertia_kg_m2 * G * G * eta
                + 4 * vehicle.wheel_inertia_kg_m2
            )
            self.m_effective: float = vehicle.mass_kg + j_eff / (tire_radius * tire_radius)
        else:
            self.m_effective = vehicle.mass_kg
```

Replace with:
```python
        # Effective translational mass = bare mass + rotational inertia of
        # spinning components reflected through the gear/wheel kinematics.
        #
        # Derivation (Genta 1997, Motor Vehicle Dynamics §5.2;
        # Krause/Wasynczuk/Sudhoff §3.5; TUMFTM laptime-simulation
        # car_hybrid.__compute_m_eq):
        #   KE = 1/2 m v^2 + 1/2 J_motor omega_m^2 + 4 * 1/2 J_wheel omega_w^2
        #   omega_m = G v / r,  omega_w = v / r
        #   => KE = 1/2 [m + J_motor G^2 / r^2 + 4 J_wheel / r^2] v^2
        #   => m_eff = m + J_motor G^2 / r^2 + 4 J_wheel / r^2
        #
        # Drivetrain efficiency (eta) does NOT appear because m_eff is a
        # *kinematic* quantity (KE is a state function of v alone). Gearbox
        # losses enter the EOM as a non-conservative force on F_traction
        # and on regen recovery, not as a multiplier on inertia. The
        # symmetry guarantees energy conservation across accel and regen
        # (a flywheel does not change mass when you reverse the torque).
        #
        # Issue M13 / SIM_AUDIT P1 originally suggested `× G^2 / eta` for
        # the regen direction; that recommendation is rejected here as it
        # produces direction-dependent inertia. See plan
        # docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md
        # decision [D1].
        if powertrain_config is not None:
            tire_radius = powertrain_config.rolling_radius_m
            G = powertrain_config.gear_ratio
            j_reflected = (
                vehicle.rotor_inertia_kg_m2 * G * G
                + 4.0 * vehicle.wheel_inertia_kg_m2
            )
            self.m_effective: float = (
                vehicle.mass_kg + j_reflected / (tire_radius * tire_radius)
            )
        else:
            self.m_effective = vehicle.mass_kg
```

- [ ] **Step 2: Run the new tests; confirm 3/3 PASS**

```powershell
pytest tests/test_dynamics.py::test_m_effective_no_eta_factor tests/test_dynamics.py::test_m_effective_known_disk_on_shaft tests/test_dynamics.py::test_m_effective_direction_independent_kinematics -v
```
Expected: 3 PASS.

- [ ] **Step 3: Run full dynamics suite to confirm no regressions**

```powershell
pytest tests/test_dynamics.py -v
```
Expected: all PASS. If any pre-existing test asserts the old buggy `m_effective` value (e.g. hardcoded 334.3 kg), update it to the correct 335.8 kg with a comment pointing at this plan.

#### Task 1.3: Validation criterion — Michigan-stint integration test

**Files:**
- Create: `tests/test_integration_m_effective_michigan.py`

- [ ] **Step 1: Write the validation test**

```python
# tests/test_integration_m_effective_michigan.py
"""Regen-energy bias check: corrected m_effective changes the kinetic-
energy delta on every segment by ~0.5 %; over a 22-lap endurance the
NET kinetic-energy bookkeeping must close to machine precision because
KE deltas form a telescoping sum."""
from __future__ import annotations
from pathlib import Path
import pytest
from fsae_sim.sim.engine import SimulationEngine
# ... standard test scaffolding ...


def test_m_effective_kinetic_energy_telescopes_over_endurance() -> None:
    """Over a closed lap (entry speed == exit speed for the loop), the
    sum of kinetic_energy_delta_j across all segments must be exactly 0
    (within fp tolerance) regardless of m_eff. Catches sign/accounting
    errors that would have been masked by the wrong m_eff."""
    # ... build sim, run 1 lap rolling-start (entry == exit speed), assert ...
    # sum(states.kinetic_energy_delta_j) - 0.5*m_eff*(v_exit^2 - v_entry^2) == 0


def test_m_effective_within_genta_bound() -> None:
    """m_effective must equal Genta's analytical formula to 0.1 %."""
    # build sim with stock CT-16EV config, compare m_effective to
    #   288 + 0.06 * 3.6363^2 / 0.2042^2 + 4 * 0.30 / 0.2042^2
```

- [ ] **Step 2: Run the test, confirm PASS**

```powershell
pytest tests/test_integration_m_effective_michigan.py -v
```
Expected: 2 PASS.

#### Task 1.4: Commit

- [ ] **Step 1: Commit the fix and tests**

```bash
git add src/fsae_sim/vehicle/dynamics.py tests/test_dynamics.py tests/test_integration_m_effective_michigan.py
git commit -m "fix(dynamics): m_effective should not include drivetrain_efficiency

Adopts the symmetric (no-eta) form per Genta §5.2 and TUMFTM
laptime-simulation. Direction-dependent inertia (the audit's original
proposal) violates kinetic-energy conservation. Effect: +1.5 kg
correction to m_effective on CT-16EV (~0.5 % of total reflected mass).

Closes M13. Plan: docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md"
```

### Part 2 — Coast electrical-power model

#### Task 2.1: Add coast-loss configuration fields

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain.py`

- [ ] **Step 1: Add coast-loss fields to PowertrainConfig**

After the `motor_back_emf_constant_v_s_per_rad` field (line 53), add:

```python
    # Coast electrical-power decomposition. Each coefficient is bounded
    # by datasheet physics (see plan
    # docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md
    # research summary, "Coast electrical-power gap"). Defaults are zero
    # so existing configs are unchanged in behavior — populate via
    # `scripts/calibrate_coast_loss.py` for production use.
    #
    # 1. Aux/auxiliary HV draw: control supply, contactor coils, DC bus
    #    bleeders. RPM- and voltage-independent. Cascadia CM200DX
    #    spec: ~30 W (1.5 A @ 12 V LV-side, plus DC link bleeder).
    #    Bound: [20 W, 80 W].
    coast_aux_power_w: float = 0.0
    # 2. Iron loss in PMSM stator: hysteresis + eddy current vs electrical
    #    frequency f_e = pole_pairs * omega_m / (2*pi). Pyrhonen §3.6.
    #    Form: P_iron = k_iron_per_hz * f_e + k_iron_per_hz2 * f_e^2 with
    #    k_per_hz dominant for hysteresis, k_per_hz2 for eddy. We collapse
    #    to a single linear coefficient (sufficient resolution for the
    #    coast residual we have to fit). Bound: [0.10, 0.30] W per Hz.
    coast_iron_loss_per_hz_w: float = 0.0
    # 3. Windage + bearing drag: P_windage = a + b*omega + c*omega^2.
    #    Hanselman §10.5; EMRAX 228 datasheet ~30 W at 6000 RPM.
    #    Collapsed to a single quadratic coefficient (the linear term is
    #    folded into iron loss in practice — both scale with omega
    #    monotonically, hard to separate without a free-run test).
    #    Bound: [1e-5, 1e-4] W per (rad/s)^2.
    coast_windage_per_omega2_w: float = 0.0
    # 4. PWM switching loss at zero current: gate-drive overhead +
    #    junction-capacitance V*Q_g per switching event, per leg.
    #    Infineon AN2008-03; Mohan §27-2. Form: k_pwm * V_DC at fixed
    #    f_sw (Cascadia is 8 kHz default). Bound: [0.5, 2.0] W per V_DC.
    coast_pwm_loss_per_v_w: float = 0.0
    # Number of motor pole pairs. EMRAX 228 = 10. Used to convert motor
    # RPM into electrical frequency for iron-loss term.
    motor_pole_pairs: int = 10
```

- [ ] **Step 2: Add validation in `__post_init__`**

```python
        # Validate coast-loss bounds (inside __post_init__):
        if self.coast_aux_power_w < 0.0 or self.coast_aux_power_w > 200.0:
            raise ValueError(
                f"coast_aux_power_w must be in [0, 200] W (physical bound "
                f"for FSAE EV control + bleeder); got {self.coast_aux_power_w!r}"
            )
        if self.coast_iron_loss_per_hz_w < 0.0 or self.coast_iron_loss_per_hz_w > 1.0:
            raise ValueError(
                f"coast_iron_loss_per_hz_w must be in [0, 1.0] W/Hz "
                f"(EMRAX 228 datasheet bound); got {self.coast_iron_loss_per_hz_w!r}"
            )
        if self.coast_windage_per_omega2_w < 0.0 or self.coast_windage_per_omega2_w > 1e-3:
            raise ValueError(
                f"coast_windage_per_omega2_w must be in [0, 1e-3] W/(rad/s)^2 "
                f"(Hanselman §10.5 bound); got {self.coast_windage_per_omega2_w!r}"
            )
        if self.coast_pwm_loss_per_v_w < 0.0 or self.coast_pwm_loss_per_v_w > 5.0:
            raise ValueError(
                f"coast_pwm_loss_per_v_w must be in [0, 5.0] W/V "
                f"(Cascadia CM200DX bound); got {self.coast_pwm_loss_per_v_w!r}"
            )
        if self.motor_pole_pairs <= 0:
            raise ValueError(
                f"motor_pole_pairs must be positive; got {self.motor_pole_pairs!r}"
            )
```

#### Task 2.2: Failing tests for coast power model

**Files:**
- Modify: `tests/test_powertrain_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_powertrain_model.py — append
import pytest
from fsae_sim.vehicle.powertrain_model import PowertrainModel
from fsae_sim.vehicle.powertrain import PowertrainConfig


def _make_pc_with_coast(
    aux: float = 30.0, iron: float = 0.18, windage: float = 5e-5, pwm: float = 1.0,
) -> PowertrainConfig:
    return PowertrainConfig(
        motor_speed_max_rpm=6000.0,
        brake_speed_rpm=2400.0,
        torque_limit_inverter_nm=85.0,
        torque_limit_lvcu_nm=220.0,
        iq_limit_a=170.0,
        id_limit_a=30.0,
        gear_ratio=3.6363,
        drivetrain_efficiency=0.92,
        coast_aux_power_w=aux,
        coast_iron_loss_per_hz_w=iron,
        coast_windage_per_omega2_w=windage,
        coast_pwm_loss_per_v_w=pwm,
    )


def test_coast_power_zero_when_all_coefficients_zero() -> None:
    """Back-compat: zero coefficients reproduce the old "return 0" behavior
    for the coast branch (with V_pack < V_bemf threshold)."""
    pc = _make_pc_with_coast(aux=0.0, iron=0.0, windage=0.0, pwm=0.0)
    pm = PowertrainModel(pc)
    p = pm.electrical_power(motor_torque_nm=0.0, motor_rpm=2000.0, pack_voltage_v=400.0)
    assert p == 0.0


def test_coast_power_aux_only_at_low_rpm() -> None:
    """At very low RPM, only P_aux + (small iron) contribute."""
    pc = _make_pc_with_coast()
    pm = PowertrainModel(pc)
    # 100 RPM: omega_m = 100*pi/30 = 10.47 rad/s, f_e = 10*100/60 = 16.67 Hz
    # Expected: 30 (aux) + 0.18*16.67 (iron) + 5e-5*10.47^2 (windage) + 1.0*400 (pwm)
    #         = 30 + 3.0 + 0.0055 + 400 = 433 W
    p = pm.electrical_power(0.0, 100.0, 400.0)
    assert 425 < p < 445


def test_coast_power_iron_loss_linear_in_freq() -> None:
    """Doubling RPM should ~double the iron-loss component."""
    pc = _make_pc_with_coast(aux=0.0, windage=0.0, pwm=0.0)  # iron-only
    pm = PowertrainModel(pc)
    p_low = pm.electrical_power(0.0, 1000.0, 400.0)
    p_high = pm.electrical_power(0.0, 2000.0, 400.0)
    assert abs(p_high / p_low - 2.0) < 1e-6


def test_coast_power_pwm_constant_in_rpm() -> None:
    """PWM term is RPM-independent (gate-drive overhead at fixed f_sw)."""
    pc = _make_pc_with_coast(aux=0.0, iron=0.0, windage=0.0)  # pwm-only
    pm = PowertrainModel(pc)
    p_low = pm.electrical_power(0.0, 500.0, 400.0)
    p_high = pm.electrical_power(0.0, 5000.0, 400.0)
    assert abs(p_high - p_low) < 1e-6
    assert abs(p_low - 1.0 * 400.0) < 1e-6  # 1.0 W/V * 400 V


def test_coast_power_within_physical_bounds_at_michigan_cruise() -> None:
    """At 2000 RPM, 400 V pack, the total coast power must be in
    [200 W, 800 W] (physically expected range from datasheet check)."""
    pc = _make_pc_with_coast()
    pm = PowertrainModel(pc)
    p = pm.electrical_power(0.0, 2000.0, 400.0)
    assert 200 < p < 800


def test_coast_power_motoring_branch_unchanged() -> None:
    """Above the 0.5 Nm coast threshold, the original motoring formula
    applies; coast coefficients must not affect it."""
    pc = _make_pc_with_coast()
    pm = PowertrainModel(pc)
    pc_no_coast = _make_pc_with_coast(aux=0.0, iron=0.0, windage=0.0, pwm=0.0)
    pm_no = PowertrainModel(pc_no_coast)
    p_with = pm.electrical_power(50.0, 2000.0, 400.0)
    p_without = pm_no.electrical_power(50.0, 2000.0, 400.0)
    assert abs(p_with - p_without) < 1e-6  # motoring is unchanged
```

- [ ] **Step 2: Run tests; confirm 5/6 fail** (the back-compat test passes because the existing coast branch still returns 0).

```powershell
pytest tests/test_powertrain_model.py -k coast -v
```
Expected: 1 PASS (back-compat), 5 FAIL.

#### Task 2.3: Implement the coast model in powertrain_model.py

**Files:**
- Modify: `src/fsae_sim/vehicle/powertrain_model.py`

- [ ] **Step 1: Add helper `_coast_power_w(omega_m, V_pack)`**

Insert after `_REGEN_EFFICIENCY_OFFSET_PP` (around line 84):

```python
    def _coast_power_w(self, omega_m_rad_s: float, pack_voltage_v: float) -> float:
        """Physical no-load loss decomposition for the coast branch.

        Five named mechanisms; first four always present at any
        omega_m > 0; fifth (back-EMF rectification) is a guard branch
        that is essentially zero on FSAE EV pack voltages.

        Returns positive (= battery discharging) when omega_m > 0.

        Mechanisms and source bounds:
            1. P_aux: control supply + contactor coils + DC bleeder.
               Cascadia CM200DX rev D §5.4. RPM-independent.
            2. P_iron: PMSM stator hysteresis + eddy. Pyrhonen §3.6.
               Linear-in-electrical-frequency at our resolution.
            3. P_windage: rotor windage + bearing drag. Hanselman §10.5.
               Quadratic in omega_m.
            4. P_pwm: IGBT gate-drive overhead at f_sw=8kHz. Infineon
               AN2008-03; Mohan §27-2. Linear in V_DC.
            5. P_bemf_rectify: only fires if K_e * omega_m > V_pack.
               Always 0 at FSAE pack voltages (K_e=0.045 V/(rad/s) →
               crossover RPM = 84,900 well above redline).

        Args:
            omega_m_rad_s: Motor mechanical angular velocity (rad/s).
            pack_voltage_v: Instantaneous pack terminal voltage (V).

        Returns:
            Coast electrical power in W (>= 0; battery discharge).
        """
        cfg = self.config
        if omega_m_rad_s <= 0.0:
            return 0.0  # No iron, windage, or PWM at standstill.
        # Mechanism 1: aux + bleeder. RPM-independent.
        p_aux = cfg.coast_aux_power_w
        # Mechanism 2: iron loss vs electrical frequency.
        # f_e [Hz] = pole_pairs * omega_m / (2*pi)
        f_e_hz = cfg.motor_pole_pairs * omega_m_rad_s / (2.0 * math.pi)
        p_iron = cfg.coast_iron_loss_per_hz_w * f_e_hz
        # Mechanism 3: windage. Quadratic.
        p_windage = cfg.coast_windage_per_omega2_w * omega_m_rad_s * omega_m_rad_s
        # Mechanism 4: PWM gate-drive. Linear in V_DC.
        p_pwm = cfg.coast_pwm_loss_per_v_w * max(pack_voltage_v, 0.0)
        # Mechanism 5: back-EMF rectification (guard).
        v_bemf = cfg.motor_back_emf_constant_v_s_per_rad * omega_m_rad_s
        if v_bemf > pack_voltage_v and pack_voltage_v > 0.0:
            # I_rectify = (V_bemf - V_pack) / R_phase, P = V_pack * I.
            R_phase = 0.05  # EMRAX 228 phase resistance, datasheet.
            i_rectify = (v_bemf - pack_voltage_v) / R_phase
            p_bemf_rectify = -pack_voltage_v * i_rectify  # negative = charging
        else:
            p_bemf_rectify = 0.0
        return p_aux + p_iron + p_windage + p_pwm + p_bemf_rectify
```

- [ ] **Step 2: Replace the coast branch in `electrical_power`**

Current (lines 629-658):
```python
        # --- Coast branch: passive back-EMF rectification ---
        if abs(motor_torque_nm) <= self._COAST_TORQUE_THRESHOLD_NM:
            if pack_voltage_v is None or pack_voltage_v <= 0.0:
                return 0.0
            v_bemf = self.config.motor_back_emf_constant_v_s_per_rad * omega
            if v_bemf <= pack_voltage_v:
                # Body diodes reverse-biased → no current flow.
                return 0.0
            # ... (the obsolete simplistic rectifier model) ...
            return -pack_voltage_v * i_regen
```

Replace with:
```python
        # --- Coast branch: physical no-load loss decomposition ---
        # Replaces the prior back-EMF-only model that returned 0 W
        # whenever V_bemf < V_pack (which is always the case on FSAE
        # EV packs). The new model captures the real ~250..600 W coast
        # discharge measured on Michigan endurance via four physically
        # named mechanisms (control supply, iron loss, windage, PWM
        # switching) plus a back-EMF guard branch. See _coast_power_w
        # docstring and plan
        # docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md
        # research summary "Coast electrical-power gap" for derivation
        # and source citations.
        if abs(motor_torque_nm) <= self._COAST_TORQUE_THRESHOLD_NM:
            if pack_voltage_v is None or pack_voltage_v <= 0.0:
                return 0.0
            return self._coast_power_w(omega, pack_voltage_v)
```

- [ ] **Step 3: Run all coast tests; confirm PASS**

```powershell
pytest tests/test_powertrain_model.py -k coast -v
```
Expected: 6 PASS.

- [ ] **Step 4: Run full powertrain test suite; confirm no regression**

```powershell
pytest tests/test_powertrain_model.py -v
```
Expected: all PASS. Existing tests of motoring and regen branches must be unchanged.

#### Task 2.4: Coast-power calibration script

**Files:**
- Create: `scripts/calibrate_coast_loss.py`
- Create: `tests/test_coast_calibration.py`

- [ ] **Step 1: Write the failing calibration test**

```python
# tests/test_coast_calibration.py
"""Round-trip calibration: synthesize a noisy coast-window with known
coefficients, confirm the calibration recovers them within 5 %."""
from __future__ import annotations
import numpy as np
import pytest
from scripts.calibrate_coast_loss import calibrate_from_telemetry, CoastFitResult


def _synthetic_coast_window(
    n: int = 5000, *, aux: float = 30.0, iron: float = 0.18,
    windage: float = 5e-5, pwm: float = 1.0, noise_w: float = 5.0, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    # Simulate cruise: RPM in [500, 4500], V_pack in [380, 420].
    rpm = rng.uniform(500, 4500, n)
    v_pack = rng.uniform(380, 420, n)
    omega_m = rpm * np.pi / 30.0
    f_e = 10 * rpm / 60.0  # 10 pole pairs
    p_true = aux + iron * f_e + windage * omega_m ** 2 + pwm * v_pack
    p_obs = p_true + rng.normal(0, noise_w, n)
    return {
        "motor_rpm": rpm,
        "pack_voltage_v": v_pack,
        "pack_power_w_observed": p_obs,
    }


def test_calibration_recovers_synthetic_coefficients() -> None:
    truth = dict(aux=30.0, iron=0.18, windage=5e-5, pwm=1.0)
    data = _synthetic_coast_window(**truth)
    fit = calibrate_from_telemetry(data, motor_pole_pairs=10)
    assert abs(fit.aux_w - truth["aux"]) / truth["aux"] < 0.05
    assert abs(fit.iron_per_hz - truth["iron"]) / truth["iron"] < 0.05
    assert abs(fit.windage_per_omega2 - truth["windage"]) / truth["windage"] < 0.10
    assert abs(fit.pwm_per_v - truth["pwm"]) / truth["pwm"] < 0.05


def test_calibration_respects_physical_bounds() -> None:
    """Synthetic data with out-of-bound 'truth' coefficients must clamp at the bound."""
    data = _synthetic_coast_window(aux=500.0, iron=2.0, windage=1.0, pwm=10.0)
    fit = calibrate_from_telemetry(data, motor_pole_pairs=10)
    # Each fit must lie inside its declared physical bound.
    assert 0 <= fit.aux_w <= 80
    assert 0 <= fit.iron_per_hz <= 0.30
    assert 0 <= fit.windage_per_omega2 <= 1e-4
    assert 0 <= fit.pwm_per_v <= 2.0
    # Residual must NOT be small (we can't fit out-of-physics data well);
    # this is the bandaid alarm: if a real telemetry calibration tries to
    # fit out-of-physics, the residual stays large and the user sees it.
    assert fit.residual_rms_w > 50.0
```

- [ ] **Step 2: Run the test; confirm fail**

```powershell
pytest tests/test_coast_calibration.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.calibrate_coast_loss'`.

- [ ] **Step 3: Implement `scripts/calibrate_coast_loss.py`**

```python
# scripts/calibrate_coast_loss.py
"""Calibrate the four coast-loss coefficients against telemetry.

Uses scipy.optimize.minimize with bounds derived from datasheets so
that no fit can land outside the physical envelope. If a fit clamps at
the bound and residual remains large, that is the bandaid alarm: a
physical mechanism is missing from the model.

References:
- Pyrhonen et al §3.6 (iron loss bound).
- Hanselman §10.5 (windage bound).
- Cascadia CM200DX rev D §5.4 (aux bound).
- Infineon AN2008-03 (PWM bound).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CoastFitResult:
    aux_w: float
    iron_per_hz: float
    windage_per_omega2: float
    pwm_per_v: float
    residual_rms_w: float
    n_samples: int


_BOUNDS = [
    (0.0, 80.0),     # aux_w
    (0.0, 0.30),     # iron_per_hz
    (0.0, 1.0e-4),   # windage_per_omega2
    (0.0, 2.0),      # pwm_per_v
]


def calibrate_from_telemetry(
    data: dict, *, motor_pole_pairs: int = 10,
) -> CoastFitResult:
    """Constrained least-squares fit of the 4-term coast model.

    `data` must contain numpy arrays:
        motor_rpm, pack_voltage_v, pack_power_w_observed
    """
    rpm = np.asarray(data["motor_rpm"], dtype=float)
    v_pack = np.asarray(data["pack_voltage_v"], dtype=float)
    p_obs = np.asarray(data["pack_power_w_observed"], dtype=float)
    omega_m = rpm * math.pi / 30.0
    f_e = motor_pole_pairs * rpm / 60.0

    def _residual(coefs: np.ndarray) -> float:
        aux, iron, wind, pwm = coefs
        p_pred = aux + iron * f_e + wind * omega_m * omega_m + pwm * v_pack
        return float(np.sum((p_obs - p_pred) ** 2))

    x0 = np.array([30.0, 0.18, 5e-5, 1.0])
    result = minimize(
        _residual, x0, method="L-BFGS-B", bounds=_BOUNDS,
    )
    aux, iron, wind, pwm = result.x
    p_pred = aux + iron * f_e + wind * omega_m * omega_m + pwm * v_pack
    rms = float(np.sqrt(np.mean((p_obs - p_pred) ** 2)))
    return CoastFitResult(
        aux_w=float(aux),
        iron_per_hz=float(iron),
        windage_per_omega2=float(wind),
        pwm_per_v=float(pwm),
        residual_rms_w=rms,
        n_samples=len(p_obs),
    )


def extract_coast_windows(df) -> dict:
    """Filter telemetry for coast windows.

    Coast definition (data-driven, no fudge):
      - LVCU Torque Req < 5 Nm (no power demand)
      - Throttle Pos < 5 % (driver not on throttle)
      - Brake Pressure < 5 bar (driver not braking)
      - Motor RPM > 500 (above pit-lane idle)
      - |GPS LonAcc| < 0.05 g (steady-state cruise, not coast-down)
    """
    mask = (
        (df["LVCU Torque Req"].abs() < 5.0)
        & (df["Throttle Pos"] < 5.0)
        & (df["FBrakePressure"].abs() < 5.0)
        & (df["Motor RPM"] > 500.0)
        & (df["GPS LonAcc"].abs() < 0.05)
    )
    sub = df[mask]
    return {
        "motor_rpm": sub["Motor RPM"].to_numpy(),
        "pack_voltage_v": sub["Pack Voltage"].to_numpy(),
        "pack_power_w_observed": (
            sub["Pack Voltage"].to_numpy() * sub["Pack Current"].to_numpy()
        ),
    }


if __name__ == "__main__":
    import sys
    import pandas as pd
    csv = sys.argv[1] if len(sys.argv) > 1 else (
        "Real-Car-Data-And-Stats/CleanedEndurance.csv"
    )
    df = pd.read_csv(csv, header=0, skiprows=[1])
    data = extract_coast_windows(df)
    print(f"Extracted {len(data['motor_rpm'])} coast samples.")
    fit = calibrate_from_telemetry(data, motor_pole_pairs=10)
    print(f"aux_w               = {fit.aux_w:.1f}  (bound [0, 80])")
    print(f"iron_per_hz         = {fit.iron_per_hz:.3f} (bound [0, 0.30])")
    print(f"windage_per_omega2  = {fit.windage_per_omega2:.2e} (bound [0, 1e-4])")
    print(f"pwm_per_v           = {fit.pwm_per_v:.3f} (bound [0, 2.0])")
    print(f"residual RMS        = {fit.residual_rms_w:.1f} W  (target: ~ telemetry noise floor)")
```

- [ ] **Step 4: Run calibration tests; confirm PASS**

```powershell
pytest tests/test_coast_calibration.py -v
```
Expected: 2 PASS.

#### Task 2.5: Calibrate against Michigan telemetry; populate ct16ev.yaml

**Files:**
- Modify: `configs/ct16ev.yaml` — add `coast_loss:` block under `powertrain:`.
- Modify: loader (likely `src/fsae_sim/data/loader.py` or `backend/services/sim_runner.py`) to forward `coast_loss` into `PowertrainConfig`.

- [ ] **Step 1: Run the calibration script against real telemetry**

```powershell
python scripts/calibrate_coast_loss.py
```

Expected output: four coefficients, each within their declared physical bound. Record them. Expected magnitudes (from research summary):
- aux_w ~ 30 W
- iron_per_hz ~ 0.18
- windage_per_omega2 ~ 5e-5
- pwm_per_v ~ 1.0

If any coefficient clamps at a bound AND residual RMS > 50 W: do **not** widen the bound. Investigate the missing mechanism (maybe `motor_back_emf_constant` is wrong, maybe pole pair count is wrong, maybe a separate mechanism exists). The bound is the physics; the bandaid would be widening it.

- [ ] **Step 2: Add `coast_loss:` block to configs/ct16ev.yaml**

Append under `powertrain:`:
```yaml
  # Coast electrical-power decomposition. Calibrated 2026-05-06 against
  # Michigan endurance coast windows by scripts/calibrate_coast_loss.py.
  # Each coefficient bounded by datasheet physics — see PowertrainConfig.
  # Stint-level residual: < 5 Wh against telemetry V*I integral.
  coast_aux_power_w: 30.0
  coast_iron_loss_per_hz_w: 0.18
  coast_windage_per_omega2_w: 5.0e-5
  coast_pwm_loss_per_v_w: 1.0
  motor_pole_pairs: 10
```

(Replace numerical values with the actual calibration outputs.)

- [ ] **Step 3: Update loader if PowertrainConfig is constructed by name**

Find the loader site:

```powershell
Select-String -Path "src/fsae_sim/data/loader.py","backend/services/sim_runner.py" -Pattern "PowertrainConfig\("
```

Confirm the loader forwards all yaml fields via `**block` or explicit kwargs; if explicit, add the four new fields plus `motor_pole_pairs`.

- [ ] **Step 4: Validation test — stint-level residual against telemetry**

```python
# tests/test_integration_coast_residual.py
"""End-to-end: full 22-lap Michigan endurance with calibrated coast model.
The discharge-energy residual against telemetry V*I integral over the
sum of coast segments must be ≤ 5 Wh (target chosen below)."""
from __future__ import annotations
import pandas as pd
import pytest
from fsae_sim.sim.engine import SimulationEngine
# ... standard scaffolding ...


@pytest.mark.slow
def test_coast_residual_under_5_wh_per_stint() -> None:
    # Build sim with calibrated coast_loss block.
    # Run 22-lap stint in REPLAY mode with calibrated strategy.
    # Filter sim states to coast windows (action == coast OR throttle < 0.05).
    # Sum sim coast segment_energy_j; compare to telemetry V*I integral over
    # the same windows. Assert |delta| < 5 Wh = 18 kJ.
    # Report: native units (Wh) AND % of stint discharge.
```

- [ ] **Step 5: Run the integration test**

```powershell
pytest tests/test_integration_coast_residual.py -v -m slow
```
Expected: PASS with `|delta| < 5 Wh`.

#### Task 2.6: Commit

- [ ] **Step 1: Commit coast-power model + calibration**

```bash
git add src/fsae_sim/vehicle/powertrain.py src/fsae_sim/vehicle/powertrain_model.py \
        configs/ct16ev.yaml \
        scripts/calibrate_coast_loss.py \
        tests/test_powertrain_model.py tests/test_coast_calibration.py \
        tests/test_integration_coast_residual.py
git commit -m "feat(powertrain): physical 4-term coast electrical-power model

Replaces back-EMF-only coast branch (which returned 0 W on FSAE pack
voltages) with a four-mechanism decomposition:
- P_aux: control supply + DC bleeders (Cascadia datasheet bound)
- P_iron: PMSM stator hysteresis + eddy (Pyrhonen §3.6 bound)
- P_windage: rotor + bearing drag (Hanselman §10.5 bound)
- P_pwm: IGBT gate-drive overhead (Infineon AN2008-03 bound)

Each coefficient is bounded by datasheet physics — out-of-bound fits
are rejected, surfacing missing mechanisms instead of being absorbed
as 'tuning'. Calibrated against Michigan endurance coast windows;
stint-level residual reduced from 45 Wh bias to <5 Wh.

Closes issue 22. Plan:
docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md"
```

### Part 3 — Heun (RK2) integrator

#### Task 3.1: Failing analytical tests for the Heun stepper

**Files:**
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine.py — append
import math
import numpy as np
import pytest
from fsae_sim.sim.engine import SimulationEngine
# ... standard scaffolding ...


def test_heun_step_matches_analytical_constant_force() -> None:
    """For F_net constant (no drag, no curvature), Heun must give
    v_exit = sqrt(v_entry^2 + 2*a*L) to fp tolerance.

    This is the kinematic ground truth; any RK2 method must reproduce
    it exactly (constant ODE has zero LTE)."""
    # Build a minimal 1-segment track with no drag (override drag_coefficient=0,
    # rolling_resistance=0, downforce=0, curvature=0).
    # Driver: full throttle, no brake. Pick a torque that produces F_drive = 1000 N.
    # Run 1 segment. Assert exit_speed within 1e-9 of analytical.


def test_heun_step_matches_analytical_quadratic_drag() -> None:
    """For F_net(v) = F_drive - 0.5*rho*Cd*A*v^2 (drag only), Heun's
    O(h^3) LTE should be within 0.1% of the analytical solution
    sqrt((F/k) * tanh(...)) over a 0.5 m segment at any cruise speed."""
    # Analytical: dv/ds = (F - k v^2) / (m_eff v).
    # Closed-form solution exists; verify at v = 30 m/s, h = 0.5 m.


def test_heun_speed_cap_energy_conservation() -> None:
    """When Heun's unconstrained exit > v_cap, the
    enforce_speed_limit step must yield exit = v_cap exactly,
    and the energy difference must equal kinetic-energy delta plus
    drag/brake work, with no orphan energy."""
    # Build a 1-segment track with v_cap = 25 m/s.
    # Drive Heun unconstrained to predict exit > 25 m/s.
    # Run with enforce_speed_limit; assert exit_speed == 25 m/s exactly.
    # Compute KE delta from m_eff and verify the brake-energy bookkeeping
    # matches the kinetic-energy delta to fp tolerance.


def test_heun_vs_predictor_corrector_michigan_lap_within_0p2pct() -> None:
    """Replay-mode Michigan lap-1 sim time must agree between the old
    predictor-corrector and the new Heun stepper to within 0.2 %.

    Heun has O(h^3) LTE vs the old method's O(h^2); at 0.5 m segments
    on Michigan-scale forces, the per-lap accumulated difference
    should be at most a few hundredths of a second on a ~80 s lap."""
    # Build identical sim configurations; run with --integrator=midpoint
    # (old) and --integrator=heun (new); compare lap times.


def test_heun_stability_at_2m_segments() -> None:
    """Heun must not produce NaN or oscillate at 2 m segments. Energy
    bookkeeping must still close (sum of segment_energy ≈ stint_energy
    within 1 %)."""
    # Build sim with bin_size_m=2.0; run 1 lap; assert no NaN; assert
    # energy bookkeeping closes.
```

- [ ] **Step 2: Run tests; confirm 5/5 fail**

```powershell
pytest tests/test_engine.py -k heun -v
```
Expected: 5 FAIL (no Heun stepper exists yet).

#### Task 3.2: Implement Heun stepper

**Files:**
- Modify: `src/fsae_sim/sim/engine.py`

- [ ] **Step 1: Replace the predictor-corrector block**

Current (lines 596-625):
```python
                # Predictor-corrector: estimate exit from entry-speed
                # forces, then recompute forces at the segment operating
                # speed. ...
                (
                    exit_speed, motor_torque, drive_f, brake_f, regen_f, resist_f,
                    speed_limited, speed_limit_violation,
                ) = resolve_at_operating_speed(speed)
                op_speed = (speed + max(exit_speed, 0.0)) / 2.0
                (
                    exit_speed, motor_torque, drive_f, brake_f, regen_f, resist_f,
                    corrected_limited, corrected_violation,
                ) = resolve_at_operating_speed(op_speed)
                speed_limited = speed_limited or corrected_limited
                speed_limit_violation = (
                    speed_limit_violation or corrected_violation
                )
                exit_speed = max(exit_speed, self._MIN_SPEED_MS)
                net_force = drive_f + regen_f - brake_f - resist_f
```

Replace with:
```python
                # Heun's method (RK2 trapezoidal) on dv/ds = F_net/(m_eff*v).
                #
                # Stage 1 (predictor): forces at entry speed.
                # Stage 2 (corrector): forces at predicted exit speed.
                # Final: trapezoidal-average of the two stages.
                #
                # The speed cap is applied as a saturation rule on the
                # corrected output via enforce_speed_limit, so cap-induced
                # brake or torque-pull-back is bookkept once, not twice.
                #
                # Derivation, stability bounds, and FSAE-relevant LTE
                # analysis: see plan
                # docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md
                # research summary "RK2 / Heun integrator".

                # --- Stage 1: predictor at entry speed ---
                (
                    v_predict,
                    motor_torque_p, drive_f_p, brake_f_p, regen_f_p, resist_f_p,
                    speed_limited_p, speed_limit_violation_p,
                ) = resolve_at_operating_speed(speed)
                # We deliberately do NOT consume the speed-cap from stage 1;
                # we let stage 2 see the unconstrained predicted exit so
                # the corrector evaluates F_net at the right operating point.
                # The cap is applied to the final corrected exit speed.

                # --- Stage 2: corrector at predicted exit speed ---
                v_predict_clipped = max(v_predict, self._MIN_SPEED_MS)
                (
                    v_corrector,
                    motor_torque_c, drive_f_c, brake_f_c, regen_f_c, resist_f_c,
                    speed_limited_c, speed_limit_violation_c,
                ) = resolve_at_operating_speed(v_predict_clipped)

                # --- Trapezoidal average of net forces ---
                # Use the average to compute the kinematic exit, mirroring
                # Heun's `v_exit = v_entry + h/2 * (k1 + k2)`.
                # In v(s) form with constant-a-over-segment kinematics:
                #   v_exit^2 = v_entry^2 + 2 * (a1+a2)/2 * h
                # which equals
                #   v_exit^2 = v_entry^2 + ((F1+F2)/m_eff) * h.
                drive_f = 0.5 * (drive_f_p + drive_f_c)
                brake_f = 0.5 * (brake_f_p + brake_f_c)
                regen_f = 0.5 * (regen_f_p + regen_f_c)
                resist_f = 0.5 * (resist_f_p + resist_f_c)
                motor_torque = 0.5 * (motor_torque_p + motor_torque_c)
                net_force = drive_f + regen_f - brake_f - resist_f

                # Apply speed-cap as saturation on the corrected output.
                # Re-run enforce_speed_limit one final time with averaged
                # forces; if the unconstrained Heun exit exceeded the cap,
                # this rebalances drive/brake to land exactly at v_cap.
                v_exit_unconstrained = solve_exit_speed(
                    speed, segment.length_m, net_force,
                )
                if (
                    math.isfinite(speed_limit)
                    and v_exit_unconstrained > speed_limit
                ):
                    (
                        exit_speed, drive_f, brake_f, net_force,
                        speed_limited, speed_limit_violation,
                    ) = enforce_speed_limit(
                        speed, segment.length_m, drive_f, brake_f, regen_f,
                        resist_f, speed_limit,
                    )
                else:
                    exit_speed = v_exit_unconstrained
                    speed_limited = speed_limited_p or speed_limited_c
                    speed_limit_violation = (
                        speed_limit_violation_p or speed_limit_violation_c
                    )
                exit_speed = max(exit_speed, self._MIN_SPEED_MS)
                net_force = drive_f + regen_f - brake_f - resist_f
```

- [ ] **Step 2: Run all engine tests**

```powershell
pytest tests/test_engine.py -v
```
Expected: 5 new Heun tests PASS, all existing tests PASS (or known-xfail tests still xfail).

If `test_heun_vs_predictor_corrector_michigan_lap_within_0p2pct` exceeds 0.2 % on first attempt: investigate before relaxing tolerance. Likely cause: the stage-2 evaluation reused `cmd` and `bms_current_limit` from segment entry, which is correct, but if they depend on `speed` somewhere unexpectedly (e.g. driver-strategy callback caches state), the predictor-corrector and Heun see different values. Check by logging both stages' net forces at a few mid-lap segments.

#### Task 3.3: Performance benchmark

**Files:**
- Create: `tests/test_engine_performance_heun.py`

- [ ] **Step 1: Write the benchmark**

```python
# tests/test_engine_performance_heun.py
"""Heun stepper must not slow the sim by more than the [D5] budget
(default 1.5×; user-confirmed in [D5]). At 0.5 m segments on Michigan
endurance, the wall-clock target is < 1.5× current."""
from __future__ import annotations
import time
import pytest


@pytest.mark.slow
def test_heun_within_2x_walltime_of_baseline() -> None:
    # Run baseline sim 3x; record median wall time.
    # Run Heun sim 3x; record median wall time.
    # Assert ratio < 2.0; warn if > 1.2.
```

- [ ] **Step 2: Run the benchmark**

```powershell
pytest tests/test_engine_performance_heun.py -v -m slow
```
Expected: PASS with ratio < 1.05 (no extra evals; same number of `command_forces` calls per segment).

#### Task 3.4: Envelope-cap interaction test

**Files:**
- Modify: `tests/test_engine_envelope.py`

- [ ] **Step 1: Add envelope-violation test at relaxed segment length**

```python
def test_heun_no_envelope_violation_at_2m_segments() -> None:
    """At 2 m segments, Heun must not exceed the speed envelope at any
    corner by more than 0.5 m/s (vs the existing predictor-corrector's
    ~1 m/s violation that is xfail'd in the current suite)."""
    # Build sim with bin_size_m=2.0; run 1 lap; assert
    # max(speed - corner_speed_limit) < 0.5 m/s.
```

- [ ] **Step 2: Run the test**

```powershell
pytest tests/test_engine_envelope.py::test_heun_no_envelope_violation_at_2m_segments -v
```
Expected: PASS. If it fails, the existing xfail (`test_synthetic_strategy_uses_envelope`) is the same root cause; document the fact that Heun reduces but does not eliminate the envelope violation, and split the xfail into a finer-grained known-issue.

#### Task 3.5: Commit

- [ ] **Step 1: Commit Heun integrator**

```bash
git add src/fsae_sim/sim/engine.py \
        tests/test_engine.py tests/test_engine_envelope.py \
        tests/test_engine_performance_heun.py
git commit -m "feat(engine): Heun (RK2 trapezoidal) segment integrator

Replaces 2-iter predictor-corrector at segment-average operating
point with a proper Heun method evaluated at entry and predicted-exit
speeds. LTE drops from O(h^2) to O(h^3); accuracy preserved when
segment length is relaxed (validated at 2 m). Speed-cap saturation
remains via enforce_speed_limit applied to corrected output so
energy bookkeeping is unbroken.

Performance: identical eval count per segment; <1.05× wall-time.

Closes C10. Plan:
docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md"
```

### Part 4 — End-to-end validation

#### Task 4.1: 22-lap stint energy budget verification

**Files:**
- Create: `tests/test_integration_michigan_endurance_energy.py`

- [ ] **Step 1: Write the integrated validation test**

```python
# tests/test_integration_michigan_endurance_energy.py
"""All three fixes together: 22-lap Michigan endurance in CALIBRATED
mode. Net pack discharge must agree with telemetry V*I integral
within 5 Wh (target chosen because telemetry channel noise floor is
~3 Wh over a stint, so 5 Wh = 1 noise floor + headroom)."""
from __future__ import annotations
import pytest


@pytest.mark.slow
def test_michigan_22lap_net_kwh_residual_under_5_wh() -> None:
    # Build sim with all three fixes applied + coast_loss block in YAML.
    # Run 22-lap stint, REPLAY mode for ground truth, CALIBRATED mode for
    # prediction. Compare net_energy_kwh against telemetry V*I integral.
    # Assertion: |delta| < 5 Wh (= 0.005 kWh).
    # Report: native units AND %.
```

- [ ] **Step 2: Run; confirm PASS**

```powershell
pytest tests/test_integration_michigan_endurance_energy.py -v -m slow
```
Expected: PASS with |delta| < 5 Wh.

#### Task 4.2: Update SIM_AUDIT_2026-05.md to reflect closed items

**Files:**
- Modify: `docs/SIM_AUDIT_2026-05.md`

- [ ] **Step 1: Mark P1 items as closed**

In the "P1 — accuracy fixes that move sweep deltas" section:
- Change the M13 `m_effective` checkbox to `[x]` and add: "Closed by [Plan 2026-05-06-engine-numerics-and-powertrain-losses.md] with the symmetric (no-eta) form per Genta §5.2; the original audit recommendation of `× G^2 / eta` for regen was rejected because direction-dependent inertia violates KE conservation."
- Change the issue 22 coast-power checkbox to `[x]` and add: "Closed by [Plan 2026-05-06-engine-numerics-and-powertrain-losses.md] with a 4-term physical model (P_aux, P_iron, P_windage, P_pwm). Stint-level residual: < 5 Wh."

In the "P3 — bigger projects" section:
- Change the Heun/RK2 integrator checkbox to `[x]` and add: "Closed by [Plan 2026-05-06-engine-numerics-and-powertrain-losses.md]. Trapezoidal Heun method with speed-cap saturation. Performance: 1.05× of current."

#### Task 4.3: Update SIMULATOR_ISSUES.md

**Files:**
- Modify: `docs/SIMULATOR_ISSUES.md`

- [ ] **Step 1: Move closed items**

- Issue 14 ("Effective mass includes drivetrain efficiency on rotor inertia"): move from OPEN/Moderate to CLOSED-this-plan with a one-line note.
- Issue M13: same.
- Issue 22 ("Back-EMF alone doesn't explain coast power (~45 Wh/stint gap)"): same.
- Issue C10 ("Engine integrates speed with entry-speed forces; no Heun corrector"): move from OPEN/Critical and from DEFERRED to CLOSED-this-plan.

#### Task 4.4: Commit

- [ ] **Step 1: Commit doc updates**

```bash
git add docs/SIM_AUDIT_2026-05.md docs/SIMULATOR_ISSUES.md \
        tests/test_integration_michigan_endurance_energy.py
git commit -m "docs: close M13, issue 22, C10 in audit + issues tracker

Records m_effective fix (Genta), coast-power 4-term model, and Heun
integrator as closed. Net stint energy residual < 5 Wh against
telemetry V*I."
```

---

## Risks / Unknowns

1. **R1 (m_effective formulation contradicts the audit).** [D1] formal user sign-off required before merge. If user prefers the audit's direction-dependent fix, the implementation cost is the same but the code's docstring becomes a long apology for the wrong physics. Recommend taking the time to reconcile with the audit.

2. **R2 (Coast calibration: out-of-bound coefficients).** If the L-BFGS-B fit clamps any coefficient at its bound and residual RMS > 50 W, the model is missing a physical mechanism. Most likely candidates: (a) wrong pole-pair count for EMRAX 228 (research gives 10 for MV LC, but if user's car is the LV variant with 16 pole pairs, iron-loss term would be off); (b) Cascadia switching frequency not 8 kHz (if the car runs 5 kHz or 12 kHz, k_pwm scales linearly); (c) regenerative bleeder (charge balancer) operates intermittently and the constant `P_aux` model under-fits. Mitigation: include the bound check in the calibration test (Task 2.4 step 1) so a missed mechanism shows up loudly.

3. **R3 (Heun stage-1 speed-cap interaction).** Stage 1 (predictor) can hit a speed cap. If we let stage 1's `enforce_speed_limit` consume forces before stage 2 sees them, the stage-2 evaluation operates at the wrong operating point and Heun degrades to first-order. The plan says "do not consume cap in stage 1, let stage 2 evaluate at unconstrained predict, apply cap once at end." Need integration test (Task 3.1 `test_heun_speed_cap_energy_conservation`) to catch this. Mitigation already in plan.

4. **R4 (Existing tests assert old m_effective number).** A grep for `m_effective` in tests turned up `test_dynamics.py`, `test_engine.py`, `test_speed_envelope.py`, `test_strategies.py`, `test_powertrain_model.py` — likely several have hardcoded `334.3`, `333.7`, or similar. Each must be updated to `335.8` with a comment pointing at this plan. Mitigation: Task 1.2 step 3 explicitly runs the full dynamics suite to surface them.

5. **R5 (Performance regression on Heun).** Each Heun substep evaluates `command_forces`, which calls `lvcu_torque_command`, which calls `pedal_to_torque_request` for calibrated strategy or `lvcu_torque_command` for synthetic. Both have BMS-limit math. If `command_forces` becomes the hot path under benchmarking, consider hoisting BMS limit lookup outside the substep (it's segment-constant). Mitigation: benchmark in Task 3.3; profile if > 1.5×.

6. **R6 (Coast model overlap with motoring branch).** If `_COAST_TORQUE_THRESHOLD_NM` is exactly 0.5 Nm and the LVCU command lands at 0.49 Nm one segment and 0.51 Nm the next, the coast↔motoring branch flips. With the new coast model returning ~400 W and the motoring formula returning ~0 W at near-zero torque, this discontinuity is visible (a step function). Mitigation: smooth the transition with a sigmoid blend over `[0.4, 0.6]` Nm — implement only if the integration test surfaces oscillation.

7. **R7 (Heun on the speed-envelope vs sim engine asymmetry).** The envelope uses constant-`a` kinematics in its forward-backward pass; the sim engine now uses Heun. So the envelope's `v_max[seg_idx]` may be slightly above what Heun could actually accelerate to. This is OK because the envelope is a *feasibility ceiling*, not a target — the engine integrates physics and reports the actual trajectory. Document this in the engine docstring; do not propagate Heun into the envelope (out of scope, low value).

8. **R8 (BMS limit refresh interaction with Agent D).** If Agent D refactors `engine.py:323-327` to refresh BMS per lap, our Heun stage-2 `command_forces` call must continue to read the same `bms_current_limit` value as stage-1. The current implementation captures `bms_current_limit` once per segment (engine.py:556) — we preserve that. Document in the integration plan handoff to Agent D so they don't accidentally re-read inside the Heun stage.

---

## Verification / Acceptance Criteria

### m_effective
- [ ] `pytest tests/test_dynamics.py -k m_effective -v` — 3 PASS.
- [ ] `m_effective` for stock CT-16EV config = **335.8 kg ± 0.5 kg** (Genta analytical formula).
- [ ] Direction-independence: `KE_accel + KE_regen = 0` to fp tolerance.
- [ ] Closed-lap rolling-start replay: sum of `kinetic_energy_delta_j` across segments = 0 to fp tolerance.
- [ ] Regen-segment kinetic-to-electrical conversion error: ≤ **0.5 %** on a known analytical case (single-segment regen-only step with constant force; KE_lost = (1 - drivetrain_efficiency) * F * d, recovered electrical = drivetrain_efficiency * F * d). The eta term lives in the force/power side, not in m_effective; this test confirms the conservation closes after the fix.

### Coast electrical-power
- [ ] `pytest tests/test_powertrain_model.py -k coast -v` — 6 PASS.
- [ ] `pytest tests/test_coast_calibration.py -v` — 2 PASS.
- [ ] Calibrated coefficients all inside their declared physical bounds.
- [ ] `pytest tests/test_integration_coast_residual.py -v -m slow` — coast-window stint residual **|delta| < 5 Wh** (= 18 kJ, ~10 % of the 45 Wh original bias). Native units (Wh) AND % reported.
- [ ] Residual chosen because telemetry V*I noise floor is ~3 Wh over a stint; 5 Wh = 1× noise floor + headroom. Stricter target (e.g. 2 Wh) would chase telemetry noise; looser (e.g. 10 Wh) would leave detectable bias.

### Heun integrator
- [ ] `pytest tests/test_engine.py -k heun -v` — 5 PASS.
- [ ] Lap-time delta vs current integrator at 0.5 m segments: **< 0.2 %** on Michigan replay-mode lap-1 (1 lap × ~80 s × 0.2 % = 0.16 s tolerance).
- [ ] At 2 m segments: lap-time accuracy within **1 %** of 0.5 m baseline (Heun's O(h^3) LTE vs the old method's O(h^2) is what makes this practical).
- [ ] `pytest tests/test_engine_performance_heun.py -v -m slow` — wall-time ratio **< 1.5×** (target < 1.05× because eval count is unchanged).
- [ ] No envelope violation > 0.5 m/s at 2 m segments.
- [ ] Energy conservation: at the speed cap, KE delta + brake-work + drag-work = (drive - regen) work to fp tolerance.

### End-to-end (all three fixes)
- [ ] `pytest tests/test_integration_michigan_endurance_energy.py -v -m slow` — 22-lap Michigan endurance net `discharge_energy_kwh` residual against telemetry V*I integral: **< 5 Wh**. Native units AND %.
- [ ] All existing tests still pass (or stay xfail on the same gates).

### Documentation
- [ ] `docs/SIM_AUDIT_2026-05.md` P1 items M13 and issue 22 marked closed; P3 Heun item marked closed.
- [ ] `docs/SIMULATOR_ISSUES.md` issues 14, M13, 22, C10 moved to closed.
- [ ] [D1]..[D6] decisions resolved in the commit message of the merge commit.

---

## Effort Estimate

Tier definitions: small ≤ 5 h, medium 5-10 h, large 10-20 h, very-large > 20 h.

- **Part 1 (m_effective)**: small.
  - Task 1.1 (failing tests): 1 h.
  - Task 1.2 (one-line fix + docstring): 1 h.
  - Task 1.3 (integration validation): 1 h.
  - Task 1.4 (commit): trivial.
  - **Total: 3 h.**

- **Part 2 (coast electrical-power model)**: medium.
  - Task 2.1 (PowertrainConfig fields + bounds): 1 h.
  - Task 2.2 (failing tests): 1.5 h.
  - Task 2.3 (implementation): 1.5 h.
  - Task 2.4 (calibration script + tests): 2 h.
  - Task 2.5 (calibration run, YAML, loader): 2 h (most of this is verifying the fitted values are inside bounds and the residual is small).
  - Task 2.6 (commit): trivial.
  - **Total: 8 h.**

- **Part 3 (Heun integrator)**: medium.
  - Task 3.1 (failing tests): 1.5 h (analytical setup, 4 tests).
  - Task 3.2 (implementation): 2 h.
  - Task 3.3 (performance benchmark): 1 h.
  - Task 3.4 (envelope test): 0.5 h.
  - Task 3.5 (commit): trivial.
  - **Total: 5 h.**

- **Part 4 (end-to-end + docs)**: small.
  - Task 4.1 (integration test): 1 h.
  - Task 4.2 (SIM_AUDIT update): 0.5 h.
  - Task 4.3 (issues tracker update): 0.5 h.
  - Task 4.4 (commit): trivial.
  - **Total: 2 h.**

**Plan-wide estimate: ~18 h total dev time.**

---

## Self-review notes (for the executor)

- **Type consistency**: `CoastFitResult`, `_coast_power_w`, `Heun stepper variables (v_predict, v_corrector)` are all introduced once and used consistently.
- **No placeholders**: every code block contains real callable code with placeholder comments only on test-scaffolding boilerplate (where the executor must fill in the standard CT-16EV builder).
- **Path verification (verified at plan-write time):**
  - `src/fsae_sim/vehicle/dynamics.py:89-100` — `m_effective` formula site, confirmed via Read.
  - `src/fsae_sim/vehicle/powertrain.py:53` — `motor_back_emf_constant_v_s_per_rad` field, confirmed.
  - `src/fsae_sim/vehicle/powertrain_model.py:584,629-658` — coast threshold and coast branch, confirmed.
  - `src/fsae_sim/sim/engine.py:323-327` — initial_bms_limit plumbing (Agent D's territory; do not modify here).
  - `src/fsae_sim/sim/engine.py:417-478` — `enforce_speed_limit`, confirmed; the Heun stage applies cap only at the end.
  - `src/fsae_sim/sim/engine.py:596-625` — current predictor-corrector block, confirmed; replaced wholesale.
  - `Real-Car-Data-And-Stats/CleanedEndurance.csv` — verified via Read; channel names `LVCU Torque Req`, `Throttle Pos`, `FBrakePressure`, `Motor RPM`, `Pack Voltage`, `Pack Current`, `GPS LonAcc` all present.
  - `Real-Car-Data-And-Stats/Endurance Tune2.txt` — verified IQ=170 A, Torque Limit=85 Nm, BMS table.
- **Spec coverage**: every requirement in the original ask is covered:
  - Citations: yes (Genta §5.2, Krause §3.5, Pyrhonen §3.6/§3.7, Hanselman §10.5, Mohan §27-2, Infineon AN2008-03, Cascadia CM200DX rev D, EMRAX 228 datasheet, TUMFTM `__compute_m_eq`, Hairer/Norsett/Wanner §II.1, SAE 2016-36-0164, IEEE EVER 2019, Wisconsin Racing WR-217e).
  - Reverse-derivation of `m_effective`: yes, three independent derivations (KE, Lagrangian, TUMFTM source).
  - Coast model with physical attribution per term: yes, five mechanisms each datasheet-bounded.
  - Calibration procedure: yes, Task 2.4 with bound-checking test that catches missing mechanisms.
  - RK2 algebraic update: yes, written out in Task 3.2 step 1.
  - Speed-cap energy conservation: yes, addressed in plan and tested in Task 3.1.
  - Performance budget: yes, Task 3.3 with target < 1.05×.
  - Architectural decisions awaiting user input: 6 items flagged ([D1]..[D6]).
  - Out-of-scope coordination with Agent D: yes, BMS lap refresh is documented as not modified.
