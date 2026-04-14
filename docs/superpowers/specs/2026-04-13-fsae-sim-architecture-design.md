# FSAE EV Endurance Simulation — Architecture Design

**Date:** 2026-04-13
**Status:** Approved
**Competition:** FSAE Michigan, ~June 2026
**Cars:** CT-16EV (2025, baseline) and CT-17EV (2026, competition car)

## Objective

Build a simulation that predicts FSAE endurance event performance with enough fidelity to maximize combined endurance + efficiency points. The sim must:

- Reproduce the 2025 Michigan endurance run within 5% on key metrics
- Compare CT-16EV vs CT-17EV performance
- Evaluate driver strategies (coasting vs braking vs hybrid approaches)
- Sweep parameters (max RPM, torque limits, braking thresholds, energy management) and find optimal configurations
- Surface results in a browser dashboard for team review

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Simulation fidelity | Enhanced point-mass with empirical corrections | Best accuracy-to-complexity ratio for strategy optimization. Calibrate against real telemetry. |
| Track representation | Michigan only, extracted from GPS telemetry | Real data available. Generalize later if needed. |
| Car comparison scope | Same platform, different pack + weight | CT-16EV: 110S4P P45B. CT-17EV: 100S4P P50B, 20 lbs lighter. Same drivetrain/controls. |
| Dashboard framework | Plotly Dash | Pure Python, purpose-built for interactive sim visualization, fast to develop. |
| Optimization approach | Pareto frontier (time vs energy) first, FSAE scoring overlay later | Pareto is useful regardless of field estimates. Scoring formulas added once sim is validated. |
| Workflow | Claude-driven | Claude builds sim, determines sweeps, runs analysis. Dashboard is for human review and decision-making. |

## Architecture

### Simulation Method

The simulation is **quasi-static**: for each track segment, it computes the maximum achievable speed from the force balance (available traction vs. required centripetal force, drag, rolling resistance) and the driver strategy's speed target. It does not integrate continuous ODEs — each segment is resolved independently given the entry state from the previous segment. This is fast enough for parameter sweeps (thousands of runs) while accurate enough for strategy comparison when calibrated against telemetry.

### Simulation Data Flow

```
Vehicle Config (YAML)
        |
        v
+-----------------------------------------------+
|  Simulation Engine (engine.py)                 |
|                                                |
|  for each track segment:                       |
|    1. Driver decides: throttle / coast / brake |
|    2. Powertrain resolves: torque -> force      |
|    3. Vehicle dynamics: force -> acceleration   |
|    4. Battery updates: current draw -> SOC, temp|
|    5. Check limits (thermal, SOC, voltage)      |
|    6. Record state -> results                  |
+-----------------------------------------------+
        |
        v
Results (per-segment time series)
        |
        v
Metrics (lap times, energy, Pareto data)
        |
        v
Dashboard (Dash app, localhost:3000)
```

### Module Responsibilities

| Module | Inputs | Outputs | Key Logic |
|---|---|---|---|
| `vehicle.py` | Config YAML | Vehicle object | Mass, frontal area, Cd, Crr, wheelbase. Computes aero drag and rolling resistance at a given speed. |
| `powertrain.py` | RPM, throttle | Torque at wheel, power draw | Motor torque curve, inverter IQ/ID limits, gear ratio, torque limits. Regen model for braking. |
| `battery.py` | Current demand, dt | SOC, voltage, temperature, available current | Cell model (P45B or P50B), series/parallel scaling, BMS discharge limits (temp-dependent), SOC taper. Calibrated against Voltt simulation data. |
| `track.py` | GPS telemetry | Ordered list of segments | Each segment: length, curvature (1/radius), grade, grip estimate. Extracted from AiM GPS + lateral acceleration data. |
| `strategy.py` | Current state, upcoming segments | Throttle/brake/coast command | Swappable control policy. This is the primary thing being optimized. |
| `engine.py` | Vehicle, track, strategy | Time-series results | Quasi-static time step: resolve speed at each segment from force balance, step battery state, enforce limits. |
| `scoring.py` | Lap times, total energy | Points | FSAE endurance + efficiency formulas. Optional field estimates for relative scoring. |
| `loader.py` | CSV file paths | DataFrames | Parsers for AiM telemetry CSV and Voltt battery simulation CSVs. |
| `metrics.py` | Sim results | Derived metrics | Lap times, energy per lap, cumulative energy, Pareto computation. |
| `sweep.py` | Config, parameter ranges | Sweep results | Run sim across parameter grid, aggregate results, store to results/. |

### Strategy as a Swappable Object

```python
class DriverStrategy:
    """Base class. Subclass to define different strategies."""
    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        ...

class CoastingStrategy(DriverStrategy):
    """2025 approach: avoid braking, coast into corners."""
    ...

class ThresholdBrakingStrategy(DriverStrategy):
    """Brake above a speed threshold, coast otherwise."""
    ...

class ReplayStrategy(DriverStrategy):
    """Replay real driver behavior extracted from telemetry."""
    ...
```

### Simulation State

```python
@dataclass
class SimState:
    time: float           # seconds
    distance: float       # meters along track
    speed: float          # m/s
    soc: float            # 0-1
    pack_voltage: float   # V
    pack_current: float   # A
    cell_temp: float      # C
    lap: int
    segment_idx: int
```

## Vehicle Configuration

YAML files in `configs/`. Each file fully specifies a car.

### CT-16EV (2025 baseline)

- Mass: 288 kg (with driver, 220 kg car + 68 kg driver)
- Pack: 110S4P Molicel P45B
- Cell voltage: 2.55V - 4.195V
- Max discharge: 100A @ 30C, tapers to 0A @ 65C
- SOC taper: 1A per 1% below 85%
- Discharged SOC: 2%
- Inverter: IQ 170A, ID 30A, torque limit 85 Nm
- LVCU torque limit: 150 Nm
- Motor speed target: 2900 RPM, brake speed: 2400 RPM

### CT-17EV (2026 competition car)

- Mass: ~279 kg (with driver, ~9 kg lighter)
- Pack: 100S4P Molicel P50B
- Same drivetrain, controls, motor, inverter, BMS, LVCU
- P50B cell parameters TBD (from datasheet or team data)

## Repository Structure

```
fsae-sim/
  src/
    fsae_sim/
      __init__.py
      vehicle/
        __init__.py
        vehicle.py
        powertrain.py
        battery.py
      track/
        __init__.py
        track.py
      driver/
        __init__.py
        strategy.py
      sim/
        __init__.py
        engine.py
      optimization/
        __init__.py
        sweep.py
      scoring/
        __init__.py
        scoring.py
      data/
        __init__.py
        loader.py
      analysis/
        __init__.py
        metrics.py
  dashboard/
    __init__.py
    app.py
    layouts/
    callbacks/
  configs/
    ct16ev.yaml
    ct17ev.yaml
  data/                  # raw data (gitignored)
  results/               # sim outputs (gitignored)
  tests/
    test_vehicle.py
    test_battery.py
    test_track.py
    test_sim.py
    test_scoring.py
    conftest.py
  docker/
    Dockerfile
    docker-compose.yaml
  pyproject.toml
  README.md
  .gitignore
```

## Dashboard

Dash app served on port 3000. Reads from `results/`. Never runs simulations.

### Pages

| Page | Purpose | Key Visuals |
|---|---|---|
| Overview | Summary of latest sim runs | Best lap time, total energy, predicted points, SOC at finish |
| Strategy Comparison | Side-by-side strategies on same car | Speed traces overlaid, energy per lap, cumulative SOC |
| Car Comparison | CT-16EV vs CT-17EV on same strategy | Same plots grouped by car config |
| Parameter Sweep | Sweep results visualization | Heatmaps, Pareto frontiers, sensitivity plots |
| Pareto Frontier | Core decision view: time vs energy | Interactive Pareto, click to see strategy/config details |
| Lap Detail | Deep dive into single sim run | Segment-by-segment: speed, torque, current, SOC, thermal over distance |

### Results Storage Format

```
results/sweep_max_rpm_ct17ev_20260413/
  manifest.json      # sweep metadata, parameters
  run_001.parquet    # per-segment time series
  run_002.parquet
  summary.parquet    # one row per run: lap time, energy, config
```

## Docker & Dev Environment

Single container. `docker-compose.yaml` maps port 3000, volume-mounts code for live reload.

```bash
# Start dashboard
docker compose -f docker/docker-compose.yaml up
# Browser -> localhost:3000

# Run sim
docker compose exec app python -m fsae_sim.sim.engine --config configs/ct17ev.yaml

# Run tests
docker compose exec app pytest

# Run sweep
docker compose exec app python -m fsae_sim.optimization.sweep --config ct17ev.yaml --sweep max_rpm
```

Also works without Docker:
```bash
pip install -e ".[dev]"
python -m dashboard.app
pytest
```

## Validation Strategy

### 5% Accuracy Target

The simulation must reproduce the 2025 Michigan endurance run within 5% on key metrics when run with CT-16EV config and a replay strategy matching real driver behavior.

### Channels to Validate

| Channel | AiM Column | Metric | What 5% Means |
|---|---|---|---|
| Speed profile | GPS Speed | RMSE over distance | ~1.5 km/h at most points |
| Longitudinal accel | GPS LonAcc | RMSE per segment | Braking/accel g's match |
| Lateral accel | GPS LatAcc | Peak per corner | Cornering g's consistent |
| Pack SOC | State of Charge | Absolute error over time | SOC curve tracks within ~5% |
| Pack voltage | Pack Voltage | RMSE over time | Voltage sag matches |
| Pack current | Pack Current | RMSE over time | Current draw profile matches |
| Motor RPM | RPM | RMSE over distance | Speed-to-RPM consistent |
| Torque | Torque Feedback | RMSE over distance | Torque commands match |
| Cell temperature | Pack Temp | Absolute error at finish | Thermal model tracks |
| Lap times | Beacon markers | Per-lap error | Each lap within 5% |
| Total energy | Integrated V*I | Total error | kWh consumed matches |

### Driver Behavior Extraction (calibration inputs, not validated outputs)

| Channel | Calibration Use |
|---|---|
| FBrakePressure / RBrakePressure | Where/how hard driver braked |
| Throttle Pos | Throttle application patterns |
| GPS Speed through corners | Corner entry/exit speeds |
| GPS LonAcc during decel | Coasting vs braking rates |

### Validation Output Format

```
=== CT-16EV Michigan Endurance Validation ===
Speed RMSE:          1.2 km/h  (3.8%)  pass
Lon. Accel RMSE:     0.04 g    (4.1%)  pass
Lat. Accel RMSE:     0.06 g    (5.2%)  marginal
SOC final error:     1.8%      (1.9%)  pass
Pack voltage RMSE:   4.2 V     (0.9%)  pass
Lap time error:      avg 2.1s  (3.4%)  pass
Total energy error:  0.08 kWh  (2.3%)  pass
---------------------------------------------
Overall:  6/7 within 5%
```

## Build Order

### Phase 1 — Foundation
1. Repo scaffold, Docker, Dash skeleton on port 3000
2. Data loader — AiM CSV and Voltt CSVs into DataFrames
3. Track extraction — GPS -> segments with curvature and distance
4. Vehicle config system — YAML loading, Vehicle/Powertrain/Battery objects

### Phase 2 — Core Simulation
5. Battery model — calibrate against Voltt data, BMS limits, SOC taper
6. Powertrain model — motor curve, inverter limits, gear ratio, regen
7. Vehicle dynamics — force balance (drag, rolling resistance, grade, traction)
8. Driver model — extract real behavior from telemetry, implement replay strategy
9. Simulation engine — time-step loop
10. Validation — compare against real data, iterate to 5%

### Phase 3 — Optimization & Comparison
11. Strategy abstraction — swappable driver strategies
12. Parameter sweep runner
13. Car comparison (CT-16EV vs CT-17EV)
14. Pareto frontier computation
15. Dashboard buildout — sweep, comparison, Pareto pages

### Phase 4 — Scoring & Decision Support
16. FSAE scoring model
17. Field estimation
18. Points maximization
19. Final decision dashboard view

## Assumptions Requiring Refinement

| Assumption | Current Value | Source | Priority to Refine |
|---|---|---|---|
| Vehicle mass (CT-16EV) | 288 kg with driver | DSS (220 + 68 kg) | Resolved |
| Vehicle mass (CT-17EV) | ~279 kg with driver | ~9 kg lighter | Medium |
| CdA | 1.50 m^2 | DSS (back-derived from force data) | Resolved |
| ClA | 2.18 m^2 | DSS (back-derived from force data) | Resolved |
| Rolling resistance | 0.015 | Typical FSAE | Medium |
| Gear ratio | 3.6363 (40/11) | DSS | Resolved |
| Drivetrain efficiency | 0.92 | Estimate | Medium |
| P50B cell parameters | TBD | Need datasheet or team data | High — required for CT-17EV sim |
