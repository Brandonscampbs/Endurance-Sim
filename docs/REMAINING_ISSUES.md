# Remaining Simulator Issues

Post-merge audit of all physics, performance, and correctness issues.
Originally written 2026-04-14, updated 2026-04-14 after cornering drag
implementation and physics bug fixes on `feat/cornering-drag` branch.

---

## Critical — Fix Before Running Strategy Sweeps

### 1. ~~Cornering solver roll angle is missing the moment arm~~ FIXED

**Fixed in:** `feat/cornering-drag` branch, commit `4ae070a`

Roll angle formula now correctly uses
`total_lateral_force * (h_cg - h_rc) / k_total` instead of
`total_lateral_force / k_total`. Roll was overpredicted ~5x.

---

### 2. BMS current limit does not feed back to speed

**File:** `src/fsae_sim/sim/engine.py:240-244`

```python
if pack_current > max_current:
    pack_current = max_current
    elec_power = pack_current * pack_voltage
    # speed is NOT recalculated
```

When the BMS clamps current (low SOC or high temperature), the energy
accounting is corrected but the car still travels at the unclamped speed.
This means the sim over-predicts speed in exactly the regime where strategy
decisions matter most — end-of-endurance with SOC taper active.

**Fix:** When pack current is clamped, back-solve for the maximum motor
torque the limited power can sustain, recompute drive force, and re-resolve
exit speed.

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

### 6. Longitudinal tire model mirrors lateral coefficients

**File:** `src/fsae_sim/vehicle/tire_model.py:258-335`

The TTC .tir files have USE_MODE=2 (lateral-only data), so the Fx model
copies the Fy structure with `|PDY1|` for peak mu. This is wrong because:

- Longitudinal peak slip ratio is 0.08-0.15 vs lateral peak slip angle of
  5-10 degrees
- Longitudinal slip stiffness is much higher per unit slip
- The combined force model (friction circle, lines 341-389) depends on both
  pure-slip curves being correct

For traction-limited acceleration out of slow corners — which is what FSAE
endurance is about — the Fx curve shape directly determines exit speed.

**Fix options:**
- Use published longitudinal coefficients for Hoosier LC0 (some teams have
  measured these on brake dynos)
- Use a simplified Fx model: `Fx = mu_x * Fz * sin(C * atan(B * kappa))`
  with B and C estimated from literature for racing slicks (B ~ 10-15, C ~
  1.65)
- Accept the current approximation but document the limitation

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

### 15. Motor efficiency is constant across all operating points

**File:** `src/fsae_sim/vehicle/powertrain_model.py:41` — `drivetrain_efficiency: 0.92`

Real EMRAX 228 efficiency varies from ~80% (low speed, low torque) to ~96%
(peak efficiency point). For energy prediction — which directly determines
efficiency scoring — a 2D efficiency map `eta(torque, rpm)` from the EMRAX
datasheet would improve accuracy without adding simulation time.

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

### 18. Hardcoded tire radius in powertrain model

**File:** `src/fsae_sim/vehicle/powertrain_model.py:33`

```python
TIRE_RADIUS_M: float = 0.228  # 10-inch FSAE wheel
```

This is the unloaded radius. Under load the tire compresses to ~0.222m
(~3% difference). Acceptable but could use the Pacejka `loaded_radius()`
for consistency.

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

## Remaining Summary

| # | Issue | Category | Sweep Impact | Effort |
|---|-------|----------|-------------|--------|
| 2 | BMS limit doesn't limit speed | Physics gap | End-of-endurance wrong | Medium |
| 3 | Regen force direction (F*eta vs F/eta) | Physics bug | Regen force understated | Low |
| 4 | No battery cooling | Physics gap | Temp monotonically rises | Medium |
| 5 | Peak force optimizer bottleneck | Performance | Sweeps 100-1000x slow | Low |
| 6 | Mirrored Fx model | Physics approx | Traction exits wrong | Medium |
| 7 | Linear field-weakening | Physics approx | Mid-range torque -45% | Low |
| 8 | R(T) independent | Physics gap | Voltage error at high T | Low |
| 9 | EFmin = 0 | Scoring bug | Over-values efficiency | Low |
| 10 | compare_driver_stints broken | Code bug | Can't compare drivers | Low |
| 12 | Python sim loop | Performance | Sweep throughput | High |
| 13 | max_speed_ms unused | Feature gap | Calibrated model too fast | Low |
| 14 | Effective mass has eta | Physics bug | 0.4% error | Low |
| 15 | Constant motor efficiency | Physics approx | ~5-15% energy error | Medium |
| 16 | Curvature from GPS accel | Data quality | Noisy corner speeds | Medium |
| 17 | 5m segment resolution | Data quality | Misses tight corners | Low |
| 18 | Hardcoded tire radius | Physics approx | ~3% RPM/force error | Low |
| 19 | ISA air density | Physics approx | ~3% aero error | Low |
| 20 | Downforce distribution | Config gap | Balance at speed wrong | Low |
