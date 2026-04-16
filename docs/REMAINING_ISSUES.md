# Remaining Simulator Issues

Post-merge audit of all physics, performance, and correctness issues.
Originally written 2026-04-14. Updated 2026-04-15 with status audit
(5 fixed, 2 partially fixed, 13 still open).

---

## Critical — Fix Before Running Strategy Sweeps

### 1. ~~Cornering solver roll angle is missing the moment arm~~ FIXED

**Fixed in:** `feat/cornering-drag` branch, commit `4ae070a`

Roll angle formula now correctly uses
`total_lateral_force * (h_cg - h_rc) / k_total` instead of
`total_lateral_force / k_total`. Roll was overpredicted ~5x.

---

### 2. ~~BMS current limit does not feed back to speed~~ FIXED

**Fixed:** BMS current limit is now enforced upstream via
`lvcu_torque_command()` / `lvcu_torque_ceiling()` before force computation.
Speed is resolved from the already-limited torque/force, so the feedback
loop is closed.

---

### 3. ~~Regen efficiency applied inconsistently (double-counted)~~ PARTIALLY FIXED

**Fixed in:** `feat/cornering-drag` branch, commit `4ae070a`

Removed the `_REGEN_EFFICIENCY_FACTOR` multiply from `engine.py` line 228.
Regen efficiency is now applied once in `electrical_power()` (0.782 total).

**Remaining:** The `regen_force()` method in `powertrain_model.py` still
applies drivetrain efficiency as `F * eta` on the force side, but in
generator mode gearbox friction adds to retarding force — the correct
formula uses `F / eta`. This understates regen braking force. Lower
priority since regen is rare in the Michigan 2025 strategy.

---

### 4. Battery thermal model has no cooling

**File:** `src/fsae_sim/vehicle/battery_model.py:362`

```python
new_temp = temp_c + dtemp
```

Temperature only increases (I²R heating) with zero dissipation. Over 22
laps, this causes the BMS temperature derating to kick in earlier than
reality. The 2025 car has passive convection/conduction; the 2026 car has
active cooling (h = 50 W/m²K per Voltt simulation metadata).

**Fix:** Add a cooling term:

```python
# Passive convection: Q_cool = h * A_surface * (T_cell - T_ambient)
# For lumped model: dT_cool = h * A * (T - T_amb) * dt / (m * cp)
dtemp_cool = h_eff * (temp_c - ambient_temp) * dt_s / thermal_mass
new_temp = temp_c + dtemp_heat - dtemp_cool
```

Make `h_eff` configurable per car (0 for 2025, ~5-10 W/K for 2026 with
forced air).

---

### 5. Peak force uses scipy optimizer instead of closed-form

**File:** `src/fsae_sim/vehicle/tire_model.py:409, 431`

`peak_lateral_force()` and `peak_longitudinal_force()` call
`minimize_scalar()` (~10-20 function evaluations each). These are also
called by the new cornering drag code's `_find_slip_angle()` method when
tires saturate.

For a 1,000-config sweep: hundreds of millions of optimizer calls.

The peak of the Magic Formula is approximately `|D| = |mu * Fz|`:

```python
def peak_lateral_force(self, normal_load_n, camber_rad=0.0):
    fz = max(normal_load_n, 1.0)
    lfzo = self._sc("LFZO")
    lmuy = self._sc("LMUY")
    fz0 = self.fnomin * lfzo
    dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0
    pdy1 = self._lat("PDY1")
    pdy2 = self._lat("PDY2")
    pdy3 = self._lat("PDY3")
    muy = (pdy1 + pdy2 * dfz) * (1.0 - pdy3 * camber_rad**2) * lmuy
    return abs(muy * fz)
```

This is accurate within 1-3% for typical Pacejka parameters (C ~ 1.3-1.7,
|E| < 1) and replaces an iterative optimizer with a single multiplication.

**Impact:** 100-1000x speedup on cornering and traction calculations.

---

## Significant — Affects Accuracy of Results

### 6. ~~Longitudinal tire model mirrors lateral coefficients~~ FIXED

**Fixed:** Full PAC2002 longitudinal model implemented using real PDX, PKX,
PCX, PEX, PHX, PVX coefficients transplanted from TTC Round 6 R25B data
and scaled to match the LC0's lateral grip envelope. See
`scripts/transplant_fx_coefficients.py` for the transplant methodology.

---

### 7. Powertrain field-weakening uses linear taper

**File:** `src/fsae_sim/vehicle/powertrain_model.py:128-132`

```python
taper_fraction = 1.0 - (excess / span)
return self._torque_limit_nm * taper_fraction
```

Real PMSM field weakening is constant-power: `T = P_max / omega`. The
linear taper underestimates torque by up to 45% in the mid-range of the
field-weakening region (2400-2900 RPM, or 57-69 km/h).

For the 2025 car with its narrow RPM range, most driving is below
brake_speed so the error is modest. For CT-17EV or with higher RPM limits,
this becomes significant.

**Fix:**

```python
if rpm <= self.config.brake_speed_rpm:
    return self._torque_limit_nm
if rpm <= self.config.motor_speed_max_rpm:
    p_max = self._torque_limit_nm * self.config.brake_speed_rpm * self._rad_per_s_per_rpm
    return p_max / (rpm * self._rad_per_s_per_rpm)
return 0.0
```

---

### 8. Internal resistance is temperature-independent

**File:** `src/fsae_sim/vehicle/battery_model.py:257-261`

`R(SOC)` is calibrated at a single temperature from Voltt data. Molicel
P45B internal resistance increases ~30% from 25°C to 60°C and
approximately doubles at -10°C. Since the sim already tracks temperature,
scaling R with temperature would improve voltage and current predictions
as the pack heats up.

**Fix:** Apply a temperature correction factor:

```python
# Typical Li-ion R vs T relationship (from P45B datasheet)
r_factor = 1.0 + 0.005 * (temp_c - 25.0)  # ~0.5% per degree above 25
r_factor = max(r_factor, 0.8)  # don't go below 80% of calibrated R
return float(self._resistance_interp(soc)) * r_factor
```

---

### 9. EFmin hardcoded to zero inflates efficiency scores

**File:** `src/fsae_sim/analysis/scoring.py:214`

```python
efmin = 0.0  # Conservative: makes full range available
```

The FSAE rules (D.13.4.6) define EFmin from the worst eligible team.
Setting it to zero inflates efficiency scores by 15-30%, making the
optimizer over-value energy savings vs lap time. Strategy optimization may
choose an overly conservative approach as a result.

**Fix:** Compute EFmin from the rules:

```python
# EFmin = (Tmin_avg / Tmax_avg) * (CO2min_per_lap / CO2max_per_lap)
# where Tmax_avg = 1.45 * Tmin_avg
# and CO2max_per_lap = 20.02 kg/100km * track_km_per_lap / 100
efmin_time_ratio = 1.0 / self.ENDURANCE_TIME_MAX_FACTOR  # = 1/1.45
co2max_per_lap = self.EV_CO2_MAX_PER_100KM * track_km_per_lap / 100.0
efmin = efmin_time_ratio * (f.efficiency_co2min_kg_per_lap / co2max_per_lap)
```

Requires passing `track_km_per_lap` to `score()`.

---

### 10. `compare_driver_stints` compares same data to itself

**File:** `src/fsae_sim/analysis/telemetry_analysis.py:501-503`

```python
d1 = extract_per_segment_actions(aim_df, track)
d2 = extract_per_segment_actions(aim_df, track)
```

Both calls use identical arguments. Driver 1 (laps 2-10) and Driver 2
(laps 13-21) are never separated.

**Fix:** Pass different `laps=` arguments:

```python
d1 = extract_per_segment_actions(aim_df, track, laps=list(range(1, 10)))
d2 = extract_per_segment_actions(aim_df, track, laps=list(range(12, 21)))
```

---

## Moderate — Worth Adding for Better Results

### 11. ~~No cornering drag~~ FIXED

**Fixed in:** `feat/cornering-drag` branch, commits `7c8814c` through `b5b5ed5`

Cornering drag implemented in `VehicleDynamics.cornering_drag()` with two
paths: Pacejka-based per-tire calculation (using load transfer for
per-tire normal loads, brentq root-finding for slip angles, drag =
Fy*sin(alpha) per tire) and analytical small-angle fallback. Engine passes
`segment.curvature` to `total_resistance()` in both replay and
force-based modes.

With the corrected Pacejka Kya (see fixed issue below), cornering drag
adds ~10 N average across the Michigan track. This is physically correct
for high-grip FSAE tires at moderate speeds — most of the energy gap comes
from other sources (#15, #7, #2).

---

### 12. Python-level simulation loop limits sweep throughput

**File:** `src/fsae_sim/sim/engine.py:141-301`

Every segment is a Python `for` iteration with function calls, attribute
lookups, and dict construction. For 200 segments x 22 laps x 1000 sweep
configs = 4.4M iterations in pure Python.

**Fix options (in order of effort):**
- Replace dict-append with pre-allocated NumPy arrays
- Extract the hot loop into a Numba-jitted function
- Vectorize the entire loop: compute all segment states simultaneously
  using NumPy broadcasting (harder due to sequential SOC/temp dependency)

---

### 13. CalibratedStrategy does not use `max_speed_ms`

**File:** `src/fsae_sim/driver/strategies.py:305-314`

`DriverZone.max_speed_ms` is populated during calibration
(`telemetry_analysis.py:421`) but `CalibratedStrategy.decide()` never
returns it. The engine has no way to enforce zone speed limits. This means
the calibrated driver model can produce speeds higher than the real driver
ever achieved in that zone.

**Fix:** Return `max_speed_ms` in the `ControlCommand` or have the engine
query it separately and clamp speed accordingly.

---

### 14. Effective mass formula includes drivetrain efficiency

**File:** `src/fsae_sim/vehicle/dynamics.py:57-58`

```python
j_eff = (
    vehicle.rotor_inertia_kg_m2 * G * G * eta
    + 4 * vehicle.wheel_inertia_kg_m2
)
```

Multiplying rotor inertia by `eta` is physically incorrect. Rotational
inertia is a property of the spinning mass; it does not change with gearbox
friction. Efficiency affects force transmission, not inertia. The standard
formula is:

```python
j_eff = vehicle.rotor_inertia_kg_m2 * G * G + 4 * vehicle.wheel_inertia_kg_m2
```

The practical impact is small (~1.2 kg or 0.4% of effective mass) but the
physics is wrong.

---

### 15. ~~Motor efficiency is constant across all operating points~~ FIXED

**Fixed:** `MotorEfficiencyMap` class in `motor_efficiency.py` provides 2D
RPM x torque lookup from EMRAX 228 data. Loaded automatically by the
simulation engine when the CSV is available. Falls back to constant
`drivetrain_efficiency` when not.

---

### 16. Track curvature computed from GPS accelerations

**File:** `src/fsae_sim/track/track.py:199`

```python
k_raw = a_lat_ms2 / (v_safe ** 2)
```

Dividing GPS lateral acceleration by speed squared amplifies noise,
especially at low speed. A geometrically cleaner approach is to compute
curvature from the GPS lat/lon trajectory via spline fitting or the Menger
curvature formula. The rolling median (window=5) helps but doesn't fully
eliminate noise.

---

### 17. 5m segment resolution may miss tight FSAE corners

**File:** `src/fsae_sim/track/track.py:23`

```python
_SEGMENT_BIN_M: float = 5.0
```

FSAE hairpins can have 3-5m radius. A 5m segment may average away the
minimum-radius point, under-predicting curvature and over-predicting
corner speed. 2-3m segments would better resolve tight geometry at the cost
of more segments per lap.

---

## Minor / Low-Priority

### 18. ~~Hardcoded tire radius in powertrain model~~ PARTIALLY FIXED

**Partially fixed:** Value updated from 0.228m to 0.2042m (correct
UNLOADED_RADIUS from .tir file). Remaining nit: still a class constant
rather than dynamically computed `loaded_radius()` (~3% difference under
load).

---

### 19. Air density is ISA standard, not Michigan conditions

**File:** `src/fsae_sim/vehicle/dynamics.py:33`

```python
AIR_DENSITY_KG_M3: float = 1.225  # ISA sea level, 15 C
```

Michigan in May is ~25°C, giving air density ~1.184 kg/m³ (3.3% lower).
Affects both drag (lower = good) and downforce (lower = bad). Systematic
but small.

---

### 20. Downforce front distribution default is not from DSS

**File:** `src/fsae_sim/vehicle/load_transfer.py:45`

```python
downforce_dist_front: float = 0.61
```

The 61% front distribution is a default, not a measured DSS value. If the
actual CoP is at a different location, the front/rear aero balance and
cornering behavior will be wrong at speed.

---

## Issues Fixed on `feat/cornering-drag` Branch

| # | Issue | Fix |
|---|-------|-----|
| NEW | Pacejka Kya formula swapped PKY1/PKY2 | `tire_model.py`: `sin(PKY2*atan(FZ/(PKY1*FZ0)))` -> `sin(2*atan(FZ/(PKY2*FZ0)))`. Cornering stiffness was 18x too low. |
| 1 | Roll angle missing moment arm | `cornering_solver.py`: added `* (h_cg - h_rc)` to roll formula. Roll was 5x too high. |
| 3 | Regen double-efficiency (partial) | `engine.py`: removed `* _REGEN_EFFICIENCY_FACTOR` from motor_torque. Efficiency now applied once. |
| 11 | No cornering drag | `dynamics.py` + `engine.py`: full Pacejka-based cornering drag with analytical fallback. |

**Validation after fixes:** Driving time error 10.3% (was 33.5% before fixes, 12% pre-cornering-drag). Energy error 37% (unchanged — driven by #15, #7, #2).

## Status Summary (updated 2026-04-15)

### Fixed (5)
| # | Issue |
|---|-------|
| 1 | Roll angle missing moment arm |
| 2 | BMS current limit does not feed back to speed |
| 6 | Longitudinal tire model mirrors lateral coefficients |
| 11 | No cornering drag |
| 15 | Motor efficiency is constant across all operating points |

### Partially Fixed (2)
| # | Issue | Remaining |
|---|-------|-----------|
| 3 | Regen efficiency double-counted | `regen_force()` uses F*eta instead of F/eta in generator mode |
| 18 | Hardcoded tire radius | Value corrected (0.2042m) but still a class constant, not `loaded_radius()` |

### Still Open (13)
| # | Issue | Category | Sweep Impact | Effort |
|---|-------|----------|-------------|--------|
| 4 | No battery cooling | Physics gap | Temp monotonically rises | Medium |
| 5 | Peak force optimizer bottleneck | Performance | Sweeps 100-1000x slow | Low |
| 7 | Linear field-weakening | Physics approx | Mid-range torque -45% | Low |
| 8 | R(T) independent | Physics gap | Voltage error at high T | Low |
| 9 | EFmin = 0 | Scoring bug | Over-values efficiency | Low |
| 10 | compare_driver_stints broken | Code bug | Can't compare drivers | Low |
| 12 | Python sim loop | Performance | Sweep throughput | High |
| 13 | max_speed_ms unused | Feature gap | Calibrated model too fast | Low |
| 14 | Effective mass has eta | Physics bug | 0.4% error | Low |
| 16 | Curvature from GPS accel | Data quality | Noisy corner speeds | Medium |
| 17 | 5m segment resolution | Data quality | Misses tight corners | Low |
| 19 | ISA air density | Physics approx | ~3% aero error | Low |
| 20 | Downforce distribution | Config gap | Balance at speed wrong | Low |

---

### 21. `PowertrainModel.electrical_power` regen branch references undefined `_REGEN_EFFICIENCY_FACTOR`

**Discovered:** 2026-04-16 (Agent 1, during D-05 implementation)
**File:** `src/fsae_sim/vehicle/powertrain_model.py:563`

The regen branch of `electrical_power()` (active when `motor_torque_nm < 0`)
references `self._REGEN_EFFICIENCY_FACTOR`, which is not defined anywhere
on the class.  Only `_REGEN_EFFICIENCY_OFFSET_PP` exists.  Any call path
that triggers regen through `electrical_power` raises `AttributeError`.

This bug was introduced by the uncommitted diff on `powertrain_model.py`
reserved for Agent 2's electrical-model wave (D-13 / D-17).  Flagged here
because it blocks `scripts/validate_tier3.py` from completing section 1
(replay mode) whenever the replay torque trace crosses into negative
territory.  Agent 2 should fix this as part of the D-17 back-EMF rework
(that rework will likely delete the regen branch entirely in favour of a
unified electrical model — see design spec §3.2).

**Scope:** Out of scope for Agent 1 (D-14 / D-05 / D-23).  Logged, not fixed.
