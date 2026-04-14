# CLAUDE.md

## What This Repo Is

FSAE EV endurance simulation and optimization for UConn Formula SAE Electric (car CT-16EV).

**Core mission: maximize competition points in Endurance + Efficiency by finding the optimal driver strategy and car tune through simulation.** FSAE endurance is an energy management game — the car has far more grip than the driver needs, so the winning variables are coasting behavior, acceleration profiles, power limits, RPM limits, and SOC management over 22 laps. The simulation exists to explore thousands of strategy/tune combinations, identify the best ones, and produce actionable coaching targets that real drivers can train to execute.

The workflow is: (1) build an accurate physics simulation validated against real telemetry, (2) create a parameterized driver model calibrated to real driver behavior, (3) sweep driver strategy and car tune parameters to find the scoring-optimal combination, (4) translate the optimal parameters into driver coaching and car configuration.

The repo starts with real telemetry and battery simulation data from Michigan 2025.

## Data Assets

### Real-Car-Data-And-Stats/
- **DSS spreadsheet** (`301_Univ_of_Connecticut-DSS-2025-05-05_1957.xlsx`): **Primary source of truth** for vehicle parameters. Contains measured mass, dimensions, suspension geometry, aero coefficients, motor/inverter specs, accumulator details, drivetrain ratios, and brake system data. Always use DSS values over estimates.
- **AiM telemetry** (`2025 Endurance Data.csv`): 20Hz CSV export from AiM Evo 5 data logger. Full Michigan endurance (~22 km, 21 laps, 1859s including driver change). Key channels: GPS Speed, GPS Lat/Lon, GPS LatAcc/LonAcc, RPM, Torque Feedback, Pack Voltage/Current, State of Charge, Pack Temp, Throttle Pos, Brake Pressure, LVCU Torque Req. Binary logs (`.xrk`, `.xrz`, `.drk`, `.rrk`) require AiM Race Studio.
- **Endurance Tune2.txt**: BMS discharge limits, SOC taper, cell voltage bounds, inverter/motor parameter settings.
- **About-Energy-Volt-Simulations/**: Voltt battery simulation export (110S4P, Molicel P45B). Two CSVs -- `_cell.csv` (single-cell level) and `_pack.csv` (pack-scaled). Used for battery model calibration (OCV-SOC curve, internal resistance).

### Key Vehicle Parameters (from DSS + Endurance Tune)
| Parameter | Value | Source |
|---|---|---|
| Mass (car only) | 220 kg | DSS |
| Mass (with 68 kg driver) | 288 kg | DSS |
| Wheelbase | 1549 mm | DSS |
| Final drive ratio | 3.6363:1 (40/11) | DSS |
| CdA (drag coefficient x area) | 1.50 m² | DSS (431N drag at 80 kph, back-derived) |
| ClA (downforce coeff x area) | 2.18 m² | DSS (625N downforce at 80 kph, back-derived) |
| Motor | EMRAX 228 MV LC, 3-phase PMSM | DSS |
| Motor peak / continuous | 230 Nm / 112 Nm | DSS (but inverter limits to 85 Nm) |
| Inverter | Cascadia CM200DX | DSS |
| Inverter torque limit | 85 Nm (IQ=170A setting) | Endurance Tune |
| LVCU torque limit | 150 Nm | Endurance Tune |
| Motor speed / brake speed | 2900 / 2400 RPM | Endurance Tune |
| Pack | 110S4P Molicel P45B (5 segments x 22S x 4P) | DSS |
| Pack energy | 7.128 kWh nominal | DSS |
| Cell voltage range | 2.55 -- 4.20 V | DSS + Endurance Tune |
| Max discharge | 100 A @ 30°C, tapers to 0 A @ 65°C | Endurance Tune |
| SOC taper | 1 A per 1% below 85% SOC | Endurance Tune |
| Tires | Hoosier 16x7.5-10 LC0 (10" wheel) | DSS |
| CG height | 279.4 mm | DSS |

## Project Roadmap

1. **Baseline simulation + validation** (DONE) -- quasi-static lap sim with 4-wheel Pacejka tire model, validated against Michigan 2025 telemetry (~2% energy error, 8/8 metrics pass).
2. **Calibrated driver model** (DONE) -- zone-based driver model (`CalibratedStrategy`) calibrated from AiM telemetry. Collapses ~200 segments into ~30-40 coachable zones (throttle/coast/brake with intensity). FSAE scoring function (`FSAEScoring`) implements full Endurance + Efficiency scoring per D.12.13 / D.13.4, pre-configured with Michigan 2025 field data. Telemetry extraction pipeline in `analysis/telemetry_analysis.py`. Validation target: 3-5% error on time and energy vs telemetry.
3. **Strategy + tune sweeps** (NEXT) -- run thousands of sims varying driver parameters (zone overrides via `CalibratedStrategy.with_zone_override()`) and car tune (max RPM, torque limit, regen intensity). Optimize combined `FSAEScoring.combined_score` (Endurance + Efficiency, max 375 pts). Foundation: zone-based driver model + scoring function from Phase 2.
4. **Driver coaching output** -- translate optimal parameters into actionable targets using `CalibratedStrategy.to_driver_brief()`: "coast X meters before corners," "use Y% throttle out of hairpins," "target Z kWh total energy." Drivers train to match these.

## Architecture Guidance

- **Modular by domain**: separate modules for battery model, drivetrain model, tire/vehicle dynamics, track representation, driver model, and lap simulation orchestration. Each module should be independently testable.
- **Simulation correctness first**: validate every model against real data before adding complexity. Numerical accuracy matters more than abstraction elegance.
- **Performance-aware from the start**: parameter sweeps and optimization will run thousands of simulations. Use NumPy/SciPy vectorized operations. Profile before optimizing, but don't create structures that prevent vectorization later.
- **Data pipelines are first-class**: loading, cleaning, and transforming telemetry and simulation CSVs should be reliable and repeatable. Use pandas for tabular data.
- **Docker for reproducibility**: local dev environment should be containerized. Pin Python version and all dependencies.
- **Testing**: use pytest. Validate models against known analytical solutions and recorded data. Property-based tests (hypothesis) for numerical edge cases.
- **Web/visualization**: FastAPI backend if needed for dashboards or parameter sweep UIs. Matplotlib/Plotly for analysis plots.

## Installed VoltAgent Subagent Packages

Marketplace: `VoltAgent/awesome-claude-code-subagents`

| Package | Version | Contents |
|---|---|---|
| `voltagent-lang` | 1.0.3 | Language specialists (includes `python-pro`) |
| `voltagent-infra` | 1.0.1 | Infrastructure/DevOps (Docker, CI/CD) |
| `voltagent-data-ai` | 1.0.2 | Data engineering, ML, analytics |

## Subagents and When to Use Them

### Core workflow (use frequently)
- **`python-pro`** -- Default for all Python implementation. Use for module design, NumPy/SciPy patterns, packaging, type hints, and Pythonic idioms.
- **`architect-reviewer`** -- Use when adding a new module, changing interfaces between modules, or before any structural refactor. Ask it to review proposed module boundaries and data flow.
- **`code-reviewer`** -- Use after completing any feature branch or before merging. Focus on correctness, not style.

### Simulation and data work
- **`data-scientist`** -- Use for model validation, statistical comparison of simulation vs. telemetry, regression analysis, and designing parameter sweep experiments.
- **`data-analyst`** -- Use for exploratory analysis of telemetry CSVs, generating comparison plots, and summarizing results across simulation runs.
- **`performance-engineer`** -- Use when simulation runtime matters: profiling hot loops, vectorization, memory layout, and parallelization of parameter sweeps.

### Quality and correctness
- **`test-automator`** -- Use when setting up pytest infrastructure, fixtures for simulation data, or parameterized test suites for model validation.
- **`qa-expert`** -- Use for test strategy decisions: what to test, coverage targets, and integration test design for multi-module simulations.
- **`debugger`** -- Use when a simulation produces wrong results and you need to trace through numerical computations or state evolution.

### Infrastructure and API
- **`fastapi-developer`** -- Use if/when building a web API for parameter sweep control, results dashboards, or simulation job management.
- **Docker/infra agents (from voltagent-infra)** -- Use when setting up the dev container, CI pipeline, or reproducible simulation environments.

## Development Methodology

Follow the Superpowers workflow for all implementation:

1. Brainstorm and refine before writing code.
2. Plan in small tasks (2-5 min each) with exact file paths and verification steps.
3. TDD: write a failing test, make it pass, clean up.
4. Use git worktrees for feature branches.
5. Request code review after implementation.
6. Verify all tests pass before marking done.
