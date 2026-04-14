# Cornering Drag Implementation Design

**Date**: 2026-04-14
**Status**: Design approved
**Prerequisite**: [Cornering Drag Findings](2026-04-14-cornering-drag-findings.md)

## Summary

Add tire cornering drag to `VehicleDynamics.total_resistance()`. When a car corners, tires operate at slip angles that dissipate energy as longitudinal drag. The Michigan track is 70%+ corners, and this missing ~205 N of average resistance explains the 37% energy error and 12% time error vs telemetry.

## Files Changed

| File | Change |
|---|---|
| `src/fsae_sim/vehicle/dynamics.py` | Add `cornering_drag()` method, update `total_resistance()` signature |
| `src/fsae_sim/sim/engine.py` | Pass `segment.curvature` to `total_resistance()` in both code paths |
| `tests/test_dynamics.py` | Add cornering drag unit tests |

## Algorithm: `VehicleDynamics.cornering_drag(speed_ms, curvature)`

### With Pacejka tire model + load transfer (physics path)

1. **Guard**: Return 0.0 if `|curvature| < 1e-6` or `speed < 0.5 m/s`
2. **Centripetal force**: `F_lat = mass * v^2 * |curvature|`
3. **Lateral acceleration**: `a_lat_g = v^2 * |curvature| / 9.81`
4. **Per-tire normal loads**: Call `load_transfer.tire_loads(speed, a_lat_g, 0.0)` to get `(FL, FR, RL, RR)` under cornering load transfer
5. **Distribute lateral demand**: Each tire's share of F_lat is proportional to its normal load: `F_lat_i = F_lat * (Fz_i / sum(Fz))`
6. **Find slip angle per tire**: For each tire, use `scipy.optimize.brentq` to solve `|Fy(alpha, Fz_i)| = F_lat_i` for `alpha` in `[0, pi/4]`. The Pacejka `lateral_force` is monotonic up to peak slip (~12-15 deg for FSAE tires). If the demanded lateral force exceeds the tire's peak, clamp to peak slip angle.
7. **Cornering drag per tire**: `drag_i = |Fy(alpha_i, Fz_i)| * sin(alpha_i)` — this is the component of the tire's lateral force vector that opposes the direction of travel
8. **Total**: Sum `drag_i` across all 4 tires

### Slip angle inversion detail

```python
def _find_slip_angle(self, f_lat_needed: float, normal_load: float) -> float:
    """Find slip angle (rad) that produces the needed lateral force magnitude."""
    peak_fy = self.tire_model.peak_lateral_force(normal_load)
    if f_lat_needed >= peak_fy:
        # Tire saturated — return peak slip angle (~0.2 rad for FSAE tires)
        # Use minimize_scalar to find peak location
        result = minimize_scalar(
            lambda a: -abs(self.tire_model.lateral_force(a, normal_load)),
            bounds=(0.0, math.pi / 4),
            method="bounded",
        )
        return result.x

    # Fy is monotonic below peak — brentq finds the unique solution
    return brentq(
        lambda a: abs(self.tire_model.lateral_force(a, normal_load)) - f_lat_needed,
        0.0, math.pi / 4,
        xtol=1e-4,
    )
```

### Without tire model (analytical fallback)

For backward compatibility when `tire_model` or `load_transfer` is None:

```
C_alpha_total = mass * g * 1.5 / 0.15  (estimated from typical FSAE mu/peak_alpha)
drag = F_lat^2 / C_alpha_total
```

This is the small-angle approximation: `Fy ≈ Cα * α`, so `α ≈ Fy/Cα`, and `drag = Fy * sin(α) ≈ Fy * α = Fy^2 / Cα`.

## Signature Change

```python
# Before
def total_resistance(self, speed_ms: float, grade: float = 0.0) -> float:

# After
def total_resistance(self, speed_ms: float, grade: float = 0.0, curvature: float = 0.0) -> float:
```

Default `curvature=0.0` preserves all existing callers — zero curvature produces zero cornering drag.

## Sim Engine Changes

Two call sites in `engine.py` both get the curvature argument:

```python
# Line ~172 (replay mode) and ~203 (force-based mode):
resist_f = self.dynamics.total_resistance(avg_speed, segment.grade, segment.curvature)
```

## Test Plan

1. **Zero curvature returns zero drag** — straight segments produce no cornering drag
2. **Low speed returns zero drag** — below 0.5 m/s threshold
3. **Known analytical case** — for a specific curvature/speed/mass, verify drag against hand calculation
4. **Drag increases with curvature** — tighter corner = more drag
5. **Drag increases with speed** — faster through same corner = more drag
6. **Saturated tire case** — when demanded lateral force exceeds peak grip, drag is clamped (no NaN/explosion)
7. **Legacy mode (no tire model)** — analytical fallback produces reasonable values
8. **total_resistance with curvature > without** — cornering adds to total resistance
9. **Backward compat** — all 383 existing tests pass unchanged

## Validation Targets

After implementation, run full 22-lap sim with CalibratedStrategy. Iterate until errors are minimal — if any metric can't be driven below 5%, document why.

| Metric | Current Error | Initial Target | Stretch Target |
|---|---|---|---|
| Driving time | 12% | < 5% | < 3% |
| Energy consumed | 37% | < 5% | < 3% |
| Mean pack current | 27% | < 8% | < 5% |

**Iteration approach**: After cornering drag lands, run validation. If errors remain above targets, investigate and fix root causes (not bandaids). Possible secondary physics to add if needed:
- Weight transfer effect on rolling resistance (~5-10 N)
- Aerodynamic yaw drag
- Residual brake drag (~5-20 N)

Any remaining error above stretch targets requires a written justification explaining the physics gap.

## Not in scope

- Weight transfer effect on rolling resistance (second-order, ~5-10 N)
- Aerodynamic yaw drag (second-order)
- CalibratedStrategy speed governor removal (already removed from `decide()`)
- Secondary missing physics listed in findings doc (all deferred)
