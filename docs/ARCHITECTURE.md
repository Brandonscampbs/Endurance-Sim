# FSAE EV Endurance Simulation — Architecture & Vision

This document is the authoritative reference for what this project is, why it exists, what decisions were made, and where it's going. Anyone working in this repo should read this first.

## The Problem

UConn Formula SAE Electric competes in FSAE endurance events. The endurance event scores teams on two axes:

1. **Endurance points** — based on finishing time relative to the fastest team
2. **Efficiency points** — based on energy consumed relative to the most efficient team

These create an explicit tradeoff: going faster costs more energy. The optimal strategy is **not** "go as fast as possible" and **not** "use as little energy as possible" — it's the configuration and strategy that maximizes the **sum** of endurance + efficiency points.

Last year (2025), the team instructed drivers to avoid braking and instead coast to decelerate before and through corners. Was that actually optimal? What would have happened with light braking? Aggressive braking with regen? Different torque limits? Different RPM targets? We don't know, because we had no way to simulate alternatives.

This repo exists to answer those questions with engineering rigor.

## The Goal

**Maximize the total number of competition points (endurance + efficiency combined) for the CT-17EV at the 2026 FSAE Michigan competition.**

Specifically, this simulation must be able to:

- Predict endurance lap times and energy consumption with enough accuracy to trust strategy decisions
- Compare the 2025 car (CT-16EV) to the 2026 car (CT-17EV) to understand what the hardware changes buy us
- Evaluate driver strategies: coasting vs. braking vs. hybrid approaches, with quantified tradeoffs
- Sweep parameters (max RPM, torque limits, braking thresholds, regen strategy, corner approach behavior) and find the optimal configuration
- Present results in a dashboard so the team can review findings and make informed decisions

The simulation is a decision-support tool, not an academic exercise. Every modeling choice should be justified by "does this help us score more points?"

## Design Decisions and Why

The following decisions were made during the initial architecture design. Each is documented with the question that was asked, the options considered, and the rationale for the choice.

### 1. Simulation Fidelity: Enhanced Point-Mass with Empirical Corrections

**Question asked:** What level of physics fidelity does the simulation need?

**Options considered:**
- **(A) Pure quasi-static / point-mass** — Car is a point on a track centerline. Max speed at each point from grip limits, energy from powertrain model. No suspension, no transient tire dynamics. How most lap-time simulators (OptimumLap) work.
- **(B) Enhanced point-mass with empirical corrections** — Same as (A), but calibrated against real AiM telemetry. The grip model, braking deceleration, and energy consumption are tuned so the sim matches what actually happened in 2025. Still no transient dynamics.
- **(C) Full multi-body dynamics** — Tire models (Pacejka), suspension kinematics, transient weight transfer. Much more complex, much slower, much harder to validate.

**Decision: (B)**

**Rationale:** For endurance strategy optimization (torque maps, braking vs. coasting, energy management), you don't need transient tire dynamics. You need accurate speed profiles, accurate energy consumption, and accurate thermal behavior. (B) achieves this by anchoring the model to real data rather than first-principles tire physics. It's also fast enough to run thousands of parameter sweeps, which (C) is not.

### 2. Track Representation: Michigan Only, From Telemetry

**Question asked:** How should tracks be defined? Generic format, or Michigan-specific?

**Options considered:**
- **(A) Michigan only, extracted from GPS telemetry** — Parse the 2025 AiM GPS trace into segments. Fastest to build.
- **(B) Generic track format from the start** — Define a segment-based format that supports any track.
- **(C) Import from external tools** — Support OptimumLap exports, SVGs, etc.

**Decision: (A)**

**Rationale:** We have real GPS data from Michigan. Michigan is the 2026 competition venue. Building a generic format adds complexity with no near-term payoff. The segment-based representation can always be generalized later — the internal data structure is the same regardless of how segments are created.

### 3. Car Comparison Scope: Same Platform, Different Pack and Weight

**Question asked:** When you say "compare cars," are you comparing software/tune parameters, hardware configurations, or both?

**Context provided:** The CT-16EV (2025) and CT-17EV (2026) share almost everything — same frame, suspension geometry, motor, motor controller, LVCU, and BMS. The key differences are:

| | CT-16EV (2025) | CT-17EV (2026) |
|---|---|---|
| Pack | 110S4P Molicel P45B | 100S4P Molicel P50B |
| Mass | 288 kg with driver (220 + 68) | ~279 kg with driver (~9 kg lighter) |
| Cell voltage range | 2.55V – 4.195V | ~2.50V – 4.20V (TBD from P50B data) |
| Everything else | Identical | Identical |

**Decision: Primarily software/tune comparison (A), with the hardware delta modeled through config differences**

**Rationale:** Since the cars share nearly everything, the comparison reduces to: how does the pack chemistry change (P45B → P50B with 10 fewer series cells) and the mass reduction (-9 kg) affect optimal strategy? The vehicle config YAML system handles this cleanly — swap the config file, re-run the sim.

### 4. Dashboard Purpose: Sim-Focused Decision Support, Not Telemetry Viewer

**Question asked:** What should the browser dashboard show?

**Options considered:**
- **(A) Telemetry viewer** — Explore the raw AiM data interactively.
- **(B) Simulation results viewer** — View sim outputs: lap times, energy, strategy comparisons.
- **(C) Both, telemetry first** — Start with telemetry since we have data now, add sim results later.
- **(D) Parameter sweep dashboard** — Focused on running and visualizing optimization.

**Decision: Sim-focused from day one (combination of B and D)**

**Rationale (from the user):** "We already have Race Studio to analyze data. The purpose of this repo is to have Claude Code build a simulation, determine parameters to sweep, and run tests to let us know the best possible setup. I want this browser interface so we can visualize all the options, sweeps, maps, etc. Compare last year's car vs this year's car. Predict scores. The entire purpose of this simulation is to maximize the number of points at competition."

The dashboard is **not** a telemetry tool. It is a decision-support interface where the team reviews simulation findings:
- Pareto frontiers (time vs. energy tradeoff curves)
- Strategy comparisons (coasting vs. braking vs. hybrid, side-by-side)
- Car comparisons (CT-16EV vs. CT-17EV on the same strategy)
- Parameter sweep heatmaps and sensitivity plots
- Predicted competition points under different configurations

### 5. Optimization Approach: Pareto Frontier First, Scoring Overlay Later

**Question asked:** How should the FSAE scoring model be integrated?

**Context:** FSAE endurance and efficiency scores are relative to the competition field. You need to estimate what other teams will do to compute your own points.

**Options considered:**
- **(A) Assume a competitive field from the start** — Use historical results to estimate the scoring baseline.
- **(B) Absolute optimization first, relative scoring later** — Find the Pareto frontier of time vs. energy. Layer scoring formulas on once the sim is validated.

**Decision: (B)**

**Rationale:** The Pareto frontier is useful regardless of what other teams do — it shows the fundamental tradeoff space for our car. It also doesn't require guessing at field performance, which adds noise before we can trust our own numbers. Once the sim is validated and producing reliable time/energy predictions, adding the FSAE scoring formulas and field estimates is straightforward.

### 6. Workflow: Claude-Driven Analysis

The intended workflow is:
1. **Claude builds** the simulation modules
2. **Claude determines** what parameters to sweep and what strategies to test
3. **Claude runs** the sweeps and analyzes results
4. **Claude surfaces** findings in the dashboard
5. **The team reviews** the dashboard and makes competition decisions

The simulation CLI and Python API are designed for programmatic use. The dashboard is for human consumption of results that Claude produces.

## The Cars

### CT-16EV (2025 Baseline)

The 2025 competition car. We have full endurance telemetry and battery simulation data for this car.

**Known parameters (from Endurance Tune2.txt and BMS configuration):**

| Parameter | Value | Source |
|---|---|---|
| Pack configuration | 110S4P Molicel P45B | Team data |
| Cell voltage range | 2.55V – 4.195V | BMS config |
| Max discharge current | 100A @ 30°C, tapers to 0A @ 65°C | BMS config |
| SOC taper | 1A per 1% below 85% SOC | BMS config |
| Discharged SOC | 2% | BMS config |
| Inverter IQ / ID | 170A / 30A | Inverter config |
| Torque limit (inverter) | 85 Nm | Inverter config |
| Torque limit (LVCU) | 150 Nm | LVCU config |
| Motor speed target | 2900 RPM | Inverter config |
| Brake speed | 2400 RPM | Inverter config |

**Estimated parameters (need refinement):**

| Parameter | Estimate | Notes |
|---|---|---|
| Mass with driver | 288 kg | DSS (220 kg car + 68 kg driver) |
| CdA | 1.50 m² | DSS (431N drag at 80 kph, back-derived) |
| ClA | 2.18 m² | DSS (625N downforce at 80 kph, back-derived) |
| Rolling resistance | 0.015 | Typical for FSAE tires |
| Gear ratio | 3.6363 (40/11) | DSS |
| Drivetrain efficiency | 92% | Estimate |

### CT-17EV (2026 Competition Car)

The car that matters. Competing at Michigan in approximately June 2026.

**Differences from CT-16EV:**
- Pack: 100S4P Molicel P50B (10 fewer series cells, different chemistry)
- Mass: ~261 kg with driver (20 lbs / ~9 kg lighter)
- Everything else is shared

**P50B cell parameters:** The discharge limits in `configs/ct17ev.yaml` are currently copied from the CT-16EV (P45B) as placeholders. These must be updated with actual P50B specifications before trusting CT-17EV simulation results.

## Available Data

### AiM Telemetry — 2025 Michigan Endurance

**File:** `Real-Car-Data-And-Stats/2025 Endurance Data.csv`

Full endurance run telemetry from AiM Race Studio. 20Hz sample rate, ~37,000 samples over approximately 31 minutes.

**Key channels for simulation:**

| Channel | Unit | Use |
|---|---|---|
| GPS Speed | km/h | Ground truth speed profile |
| GPS Latitude / Longitude | deg | Track extraction (segment geometry) |
| GPS LatAcc / LonAcc | g | Cornering and braking accelerations |
| RPM | rpm | Motor operating points |
| Torque Feedback | Nm | Actual torque delivered |
| Pack Voltage | V | Battery voltage under load |
| Pack Current | A | Current draw profile |
| State of Charge | % | SOC depletion curve |
| Pack Temp | °C | Thermal behavior |
| Throttle Pos | % | Driver throttle inputs |
| FBrakePressure / RBrakePressure | bar | Driver braking behavior |
| Distance on GPS Speed | m | Cumulative distance traveled |

**What the telemetry tells us about the 2025 strategy:**
- Brake pressure channels reveal where and how hard the driver braked (or didn't)
- Throttle position shows acceleration behavior
- GPS speed through corners shows actual corner approach strategy
- Longitudinal acceleration during deceleration distinguishes coasting from braking
- The team instructed drivers to coast (no braking) into corners — the telemetry lets us verify whether they actually did, and model that behavior

### Voltt Battery Simulations

Battery cell and pack simulations from About Energy's Voltt platform, driven by the actual endurance power duty cycle.

**2025 Pack (CT-16EV):** `Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/`
- Configuration: 110S4P Molicel P45B
- Cell: P45B (45A continuous, exceeded to 65.1A peak in simulation)
- Ambient: 25°C, no active cooling (h=0 W/m²K)
- Files: `2025_Pack_cell.csv` (single cell values), `2025_Pack_pack.csv` (pack-scaled values)
- ~18,000 samples

**2026 Pack (CT-17EV):** `Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2026/`
- Configuration: 100S4P Molicel P50B
- Cell: P50B (60A continuous, exceeded to 72.7A peak in simulation)
- Ambient: 25°C, active cooling (h=50 W/m²K)
- Files: `2026_Pack_cell.csv` (single cell values), `2026_Pack_pack.csv` (pack-scaled values)

**Columns in both datasets:**
Time, Voltage, SOC, Power, Current, Charge, OCV, Temperature, Heat Generation, Cooling Power, Resistive Heat, Reversible Heat, Hysteresis Heat

**Important difference:** The 2025 simulation uses h=0 (no cooling), while the 2026 simulation uses h=50 W/m²K (active cooling). This reflects a design change in the CT-17EV thermal management.

### BMS Configuration

**File:** `Real-Car-Data-And-Stats/Endurance Tune2.txt`

The actual BMS and inverter parameters used during the 2025 endurance event. This is the source of truth for discharge limits, SOC taper, cell voltage bounds, and inverter/motor limits.

### Additional Files

- **AiM binary files** (`.xrk`, `.xrz`, `.drk`, `.rrk`): Raw race logs. Require AiM Race Studio to view. The CSV export contains the same data in accessible form.
- **DSS spreadsheet** (`301_...DSS...xlsx`): 2025 competition Design Spec Sheet.

## Simulation Architecture

### Method

The simulation is **quasi-static**: for each track segment, it computes the achievable speed from the force balance (available traction vs. required centripetal force, drag, rolling resistance) and the driver strategy's speed target. It does not integrate continuous ODEs — each segment is resolved independently given the entry state from the previous segment.

This is fast enough for parameter sweeps (thousands of runs) while accurate enough for strategy comparison when calibrated against telemetry.

### Data Flow

```
Vehicle Config (YAML)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Simulation Engine                                │
│                                                   │
│  for each track segment:                          │
│    1. Driver strategy decides: throttle/coast/brake│
│    2. Powertrain resolves: torque → force at wheel │
│    3. Vehicle dynamics: net force → speed change   │
│    4. Battery updates: current draw → SOC, temp    │
│    5. Check limits (thermal, SOC, voltage)         │
│    6. Record state                                │
└───────────────────────────────────────────────────┘
        │
        ▼
Results (per-segment time series)
        │
        ▼
Metrics (lap times, energy per lap, Pareto data)
        │
        ▼
Dashboard (localhost:3000)
```

### Module Responsibilities

| Module | What It Does | Key Interfaces |
|---|---|---|
| `fsae_sim.vehicle` | Loads car configuration from YAML. Computes drag force and rolling resistance at a given speed. | `VehicleConfig.from_yaml(path)` |
| `fsae_sim.vehicle.battery` | Models pack behavior: voltage as a function of SOC and current, BMS discharge limits (temperature-dependent), SOC taper. Calibrated against Voltt data. | `BatteryConfig`, pack voltage/current limit methods |
| `fsae_sim.vehicle.powertrain` | Motor torque curve, inverter IQ/ID limits, gear ratio, torque limits. Regen model for braking energy recovery. | `PowertrainConfig`, torque-at-wheel computation |
| `fsae_sim.track` | Ordered sequence of segments extracted from GPS telemetry. Each segment has length, curvature, grade, and grip factor. | `Track.from_telemetry(csv_path)`, `Segment` dataclass |
| `fsae_sim.driver` | Swappable control policy. Given current car state and upcoming track, decides throttle/coast/brake. **This is the primary thing being optimized.** | `DriverStrategy.decide(state, upcoming) → ControlCommand` |
| `fsae_sim.sim` | Time-step loop that ties vehicle, track, and strategy together. Produces a time series of simulation states. | `SimulationEngine.run(num_laps) → SimResult` |
| `fsae_sim.scoring` | FSAE endurance and efficiency point formulas. Optional field estimates for relative scoring. | `calculate_endurance_points()`, `calculate_efficiency_points()` |
| `fsae_sim.optimization` | Parameter sweep runner. Varies one or more parameters across a grid, runs the sim for each, stores results. | `run_sweep(config) → results_path` |
| `fsae_sim.analysis` | Post-processing: lap times, energy per lap, Pareto frontier computation. | `compute_pareto_frontier(results)` |
| `fsae_sim.data` | Loaders for AiM telemetry CSV and Voltt battery simulation CSVs. | `load_aim_csv(path)`, `load_voltt_csv(path)` |

### Driver Strategy — The Core Optimization Target

The driver strategy module is the heart of the optimization work. Different strategies are implemented as subclasses:

```python
class DriverStrategy:
    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        ...

class CoastingStrategy(DriverStrategy):
    """2025 approach: never brake, coast into corners."""

class ThresholdBrakingStrategy(DriverStrategy):
    """Brake above a speed threshold, coast below."""

class ReplayStrategy(DriverStrategy):
    """Replay real driver behavior extracted from telemetry."""
```

The key engineering questions this must answer:

- **Was the 2025 coasting-only strategy optimal?** Run CoastingStrategy vs. ThresholdBrakingStrategy on the Michigan track. Compare total time and energy. Plot the Pareto frontier. The answer is in the data.
- **What's the best braking threshold?** Sweep the threshold parameter in ThresholdBrakingStrategy. Find the point on the Pareto frontier that maximizes total FSAE points.
- **How much does regen recover?** Compare strategies with and without regenerative braking. Quantify the energy recovery vs. the time cost of braking.
- **How should corner approach vary by corner type?** A tight hairpin and a fast sweeper have different optimal approaches. The strategy can be corner-aware.

### Vehicle Configuration System

Cars are defined in YAML files under `configs/`. Each file fully specifies a car — mass, aero, powertrain, battery, BMS limits.

To compare cars, you run the same simulation with different config files:

```bash
python -m fsae_sim.sim.engine --config configs/ct16ev.yaml
python -m fsae_sim.sim.engine --config configs/ct17ev.yaml
```

To test a parameter change, copy a config and modify one value:

```yaml
# configs/ct17ev_high_rpm.yaml
# ... same as ct17ev.yaml but:
powertrain:
  motor_speed_max_rpm: 3200  # was 2900
```

This makes experiments explicit, reproducible, and diffable.

### Dashboard

Plotly Dash app on port 3000. Dark theme (DARKLY). Six pages:

| Page | What It Answers |
|---|---|
| **Overview** | What are the key numbers from the latest sim runs? |
| **Strategy Comparison** | Which strategy is better on the same car? Speed/SOC/energy overlaid. |
| **Car Comparison** | How does CT-17EV compare to CT-16EV on the same strategy? |
| **Parameter Sweep** | What happens when we vary max RPM / torque limit / braking threshold? |
| **Pareto Frontier** | What is the time vs. energy tradeoff? Where does each strategy land? |
| **Lap Detail** | Deep dive into one run: segment-by-segment speed, torque, current, SOC. |

The dashboard reads from `results/`. It never runs simulations. Results are stored as Parquet files with a JSON manifest describing what was simulated.

## Validation Strategy

### 5% Accuracy Target

The simulation must reproduce the 2025 Michigan endurance run within 5% error on key metrics when run with the CT-16EV config and a replay strategy matching real driver behavior.

### Channels to Validate Against Real Data

| What | AiM Channel | How We Measure Error |
|---|---|---|
| Speed profile | GPS Speed | RMSE over distance (~1.5 km/h target) |
| Braking/accel g's | GPS LonAcc | RMSE per segment |
| Cornering g's | GPS LatAcc | Peak per corner |
| SOC depletion | State of Charge | Absolute error over time |
| Pack voltage | Pack Voltage | RMSE over time |
| Pack current | Pack Current | RMSE over time |
| Motor RPM | RPM | RMSE over distance |
| Motor torque | Torque Feedback | RMSE over distance |
| Cell temperature | Pack Temp | Absolute error at end of run |
| Per-lap times | Beacon markers / segment times | Per-lap absolute error |
| Total energy | Integrated V × I | Total kWh error |

### Driver Behavior Extraction (Calibration Inputs)

These channels are used to **build** the driver model, not to validate it:

| Channel | What We Learn |
|---|---|
| FBrakePressure / RBrakePressure | Where and how hard the driver braked |
| Throttle Pos | Throttle application patterns |
| GPS Speed through corners | Actual corner entry/exit speeds |
| GPS LonAcc during deceleration | Whether the driver was coasting or braking at each point |

The replay strategy uses these to faithfully reproduce what the real driver did. Once the sim matches reality with the replay strategy, we can swap in alternative strategies and trust the deltas.

## Assumptions Requiring Refinement

These values are estimates. Each one should be verified or measured before trusting absolute simulation predictions. Relative comparisons (strategy A vs. strategy B on the same car) are less sensitive to these.

| Parameter | Current Value | How to Refine | Priority |
|---|---|---|---|
| Vehicle mass (CT-16EV) | 288 kg | DSS (220 kg car + 68 kg driver) | Resolved |
| Vehicle mass (CT-17EV) | ~279 kg | ~9 kg lighter than CT-16EV | Medium |
| Gear ratio | 3.6363 (40/11) | DSS | Resolved |
| P50B discharge limits | Copied from P45B | Update from P50B datasheet | High |
| CdA | 1.50 m² | DSS (back-derived from force data) | Resolved |
| ClA | 2.18 m² | DSS (back-derived from force data) | Resolved |
| Rolling resistance | 0.015 | Tire data or coastdown test | Medium |
| Drivetrain efficiency | 92% | Dyno test or back-calculate from telemetry | Medium |

## Roadmap

### Phase 1 — Foundation ✅ (Complete)

Repository scaffold, Docker dev environment, Dash dashboard skeleton, vehicle config system, data loaders, module interface stubs. 17 tests passing.

### Phase 2 — Core Simulation (Nearly Done)

The critical path. Simulation engine built and validated against real data. Tier 3 upgrade (4-wheel Pacejka tire model) merged and validated (~2% energy error, 8/8 metrics pass). Driver model built (CalibratedStrategy, zone-based). Remaining: finalize driver model quality/accuracy validation checks.

1. **Battery model** — Calibrate voltage-SOC curve against Voltt data for both P45B and P50B cells. Implement BMS discharge limits (temperature-dependent) and SOC taper. This is the most data-rich module — the Voltt simulations provide detailed cell behavior under the actual endurance duty cycle.

2. **Powertrain model** — Motor torque curve from inverter limits, gear ratio, wheel force computation. Regen model for energy recovery during braking.

3. **Track extraction** — Parse AiM GPS latitude/longitude into an ordered sequence of segments with curvature, length, and grade. Use lateral acceleration to cross-validate curvature estimates.

4. **Vehicle dynamics** — Force balance at each segment: aerodynamic drag + rolling resistance + grade resistance + cornering drag vs. available traction. Resolve achievable speed.

5. **Driver behavior extraction** — Analyze the AiM telemetry to extract what the real driver did: where they braked, how they applied throttle, what speeds they carried through corners. Build a ReplayStrategy that reproduces this behavior.

6. **Simulation engine** — Tie it all together. Step through segments, resolve speed, update battery state, record results.

7. **Validation** — Run the full sim with CT-16EV config and ReplayStrategy on the Michigan track. Compare against real data channel-by-channel. Iterate until within 5% on key metrics.

### Phase 3 — Optimization & Comparison (Next)

Once Phase 2 finalization is complete, use the sim to find answers.

8. **Strategy abstraction** — Implement CoastingStrategy, ThresholdBrakingStrategy, and any other candidate strategies. Make them parameterizable.

9. **Parameter sweep runner** — Vary one or more parameters across a grid. Store results with metadata. Support multi-dimensional sweeps.

10. **Car comparison** — Run CT-16EV vs CT-17EV on the same strategies. Quantify the effect of the pack change and mass reduction.

11. **Pareto frontier** — Compute the time vs. energy Pareto frontier for each car/strategy combination. This is the core analysis output.

12. **Dashboard buildout** — Populate all dashboard pages with real data from sweeps and comparisons.

### Phase 4 — Scoring & Decision Support (Future)

Turn simulation results into competition strategy recommendations.

13. **FSAE scoring model** — Implement the official endurance and efficiency point formulas.

14. **Field estimation** — Estimate what competing teams will score based on historical data. This converts the Pareto frontier into a predicted-points surface.

15. **Points maximization** — Find the strategy and configuration that maximizes total predicted points.

16. **Decision dashboard** — The final deliverable: "Here is what to run at competition, and here is why."

## Repository Structure

```
├── src/fsae_sim/              # Simulation Python package (pip-installable)
│   ├── vehicle/               # Vehicle, powertrain, battery configuration and models
│   │   ├── vehicle.py         # VehicleConfig, VehicleParams, YAML loading
│   │   ├── powertrain.py      # PowertrainConfig
│   │   └── battery.py         # BatteryConfig, DischargeLimitPoint
│   ├── track/                 # Track representation
│   │   └── track.py           # Track, Segment (from_telemetry in Phase 2)
│   ├── driver/                # Driver strategy
│   │   └── strategy.py        # DriverStrategy base, SimState, ControlCommand
│   ├── sim/                   # Simulation engine
│   │   └── engine.py          # SimulationEngine, SimResult
│   ├── scoring/               # FSAE scoring
│   │   └── scoring.py         # EnduranceScore, point calculations
│   ├── optimization/          # Parameter sweeps
│   │   └── sweep.py           # SweepConfig, run_sweep
│   ├── analysis/              # Post-processing
│   │   └── metrics.py         # Lap times, energy, Pareto frontier
│   └── data/                  # Data loading
│       └── loader.py          # load_aim_csv, load_voltt_csv
├── dashboard/                 # Dash web app (port 3000)
│   ├── app.py                 # App setup, sidebar, layout
│   ├── __main__.py            # Entry point: python -m dashboard
│   └── pages/                 # One file per dashboard page
│       ├── overview.py
│       ├── strategy.py
│       ├── cars.py
│       ├── sweep.py
│       ├── pareto.py
│       └── lap_detail.py
├── configs/                   # Vehicle configuration YAML files
│   ├── ct16ev.yaml            # 2025 car: 110S4P P45B, 288 kg (with driver)
│   └── ct17ev.yaml            # 2026 car: 100S4P P50B, ~279 kg (with driver)
├── Real-Car-Data-And-Stats/   # Raw telemetry and simulation data
│   ├── 2025 Endurance Data.csv                        # AiM telemetry export
│   ├── Endurance Tune2.txt                            # BMS/inverter parameters
│   ├── About-Energy-Volt-Simulations-2025-Pack/       # P45B Voltt battery sim
│   │   ├── 2025_Pack_cell.csv
│   │   ├── 2025_Pack_pack.csv
│   │   └── simulation_info.txt
│   └── About-Energy-Volt-Simulations-2026/            # P50B Voltt battery sim
│       ├── 2026_Pack_cell.csv
│       ├── 2026_Pack_pack.csv
│       └── simulation_info.txt
├── results/                   # Simulation outputs (gitignored)
├── tests/                     # pytest test suite
├── docs/                      # Documentation and design specs
├── docker/                    # Dockerfile and docker-compose.yaml
├── pyproject.toml             # Python package definition
└── README.md                  # Quick-start guide
```

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Simulation and dashboard |
| NumPy / SciPy | Numerical computation, interpolation, optimization |
| pandas | Tabular data handling, telemetry processing |
| PyYAML | Vehicle config loading |
| Dash + Plotly | Browser dashboard |
| dash-bootstrap-components | Dashboard styling (DARKLY theme) |
| pyarrow / Parquet | Efficient results storage |
| pytest | Testing |
| Docker | Reproducible dev environment |

## Running the Project

### Dashboard

```bash
# With Docker
docker compose -f docker/docker-compose.yaml up
# → http://localhost:3000

# Without Docker
pip install -e ".[dev]"
python -m dashboard
# → http://localhost:3000
```

### Tests

```bash
pytest -v
```

### Simulations (Phase 2+)

```bash
# Run a single simulation
python -m fsae_sim.sim.engine --config configs/ct17ev.yaml

# Run a parameter sweep
python -m fsae_sim.optimization.sweep --config ct17ev.yaml --sweep max_rpm
```
