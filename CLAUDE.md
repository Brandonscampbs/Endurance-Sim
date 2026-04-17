# CLAUDE.md

## What This Repo Is

FSAE EV endurance simulation for UConn Formula SAE Electric (car CT-16EV).

**Core mission: build the most accurate FSAE EV endurance simulator possible, and expose it through a three-page webapp.** The three pages are:

1. **Verification** — how close is the baseline simulator to reality? (compare sim vs Michigan 2025 telemetry, per-channel residuals, energy budget reconciliation).
2. **Visualization** — a 3D playback of the car so physics bugs become visible.
3. **Simulate** — a what-if tool with three knobs only: **max motor RPM, max motor torque, SOC discharge map**. Run one sim with those overrides, see how endurance changes.

**Out of scope for this repo.** Parameter sweeps, Pareto optimization, multi-run comparison, driver-strategy search, coaching output. Those will live in a separate repo that imports this one as a library. Do not add sweep runners, sweep-results pages, or sweep storage schemas here.

The repo starts with real telemetry and battery simulation data from Michigan 2025.

## Data Assets

### Real-Car-Data-And-Stats/
- **DSS spreadsheet** (`301_Univ_of_Connecticut-DSS-2025-05-05_1957.xlsx`): **Primary source of truth** for vehicle parameters. Contains measured mass, dimensions, suspension geometry, aero coefficients, motor/inverter specs, accumulator details, drivetrain ratios, and brake system data. Always use DSS values over estimates.
- **AiM telemetry** (`2025 Endurance Data.csv`): 20Hz CSV export from AiM Evo 5 data logger. Full Michigan endurance (~22 km, 21 laps, 1859s including driver change). Key channels: GPS Speed, GPS Lat/Lon, GPS LatAcc/LonAcc, RPM, Torque Feedback, Pack Voltage/Current, State of Charge, Pack Temp, Throttle Pos, Brake Pressure, LVCU Torque Req. Binary logs (`.xrk`, `.xrz`, `.drk`, `.rrk`) require AiM Race Studio.
- **Endurance Tune2.txt**: BMS discharge limits, SOC taper, cell voltage bounds, inverter/motor parameter settings.
- **About-Energy-Volt-Simulations/**: Voltt battery simulation export (110S4P, Molicel P45B). Two CSVs -- `_cell.csv` (single-cell level) and `_pack.csv` (pack-scaled). Used for battery model calibration (OCV-SOC curve, internal resistance).
- **LVCU Code.txt**: LVCU firmware source — the torque command chain (`PowertrainModel.lvcu_torque_command()` and related methods) is a faithful translation of this file. Source of truth for `lvcu_power_constant`, `lvcu_rpm_scale`, `lvcu_omega_floor`, and pedal deadzone parameters in `PowertrainConfig`.
- **emrax228_hv_cc_motor_map_long.csv**: EMRAX 228 motor efficiency map (speed_rpm, torque_Nm, efficiency_pct). Loaded by `MotorEfficiencyMap` for 2D operating-point-dependent motor+inverter efficiency. Falls back to constant `drivetrain_efficiency` if missing.
- **Tire Models from TTC/**: PAC02 .tir files for Hoosier LC0 16x7.5-10 at multiple pressures (Round 8 TTC data). Primary: `Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir`. Longitudinal (Fx) coefficients transplanted from R25B donor data via `scripts/transplant_fx_coefficients.py`.
- **CleanedEndurance.csv**: Cleaned AiM telemetry produced by `scripts/clean_endurance_data.py` (removes pre-start, driver change, post-finish). Uses `LFspeed` column (left-front wheel speed) instead of GPS Speed. Latin-1 encoding.

### Known Issues (MUST READ)

**`docs/SIMULATOR_ISSUES.md`** is the authoritative tracker for all known physics bugs, approximations, and code issues (consolidated 2026-04-16 from former REMAINING_ISSUES / SIMULATOR_AUDIT / DRIVER_MODEL_ISSUES docs). **Read this before trusting simulation results or starting new physics work.**

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

See `docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md` for the full plan.

1. **Baseline simulation + validation** (DONE) -- quasi-static lap sim with 4-wheel Pacejka tire model, validated against Michigan 2025 telemetry (~2% energy error, 8/8 metrics pass).
2. **Verification page correctness + physics fixes** (IN PROGRESS) -- fix the per-lap metrics regression, swap GPS Speed → LFspeed, add per-channel residuals + RMS/R²/correlation, add energy budget reconciliation, expand channel coverage to RPM/torque/pack V&I/temp. Also close remaining open physics/code issues tracked in `docs/SIMULATOR_ISSUES.md`.
3. **Visualization page physics + polish** (NEXT) -- fix force-arrow frame math (body vs world), Euler order for roll/pitch/yaw, orbit camera target binding, backend Fy sign-by-curvature. Add trajectory trail, scrubbable time-series strip, friction-circle per wheel, sector/lap markers on timeline.
4. **Simulate page** (AFTER VISUALIZATION) -- backend endpoint accepting `{max_rpm, max_torque_nm, soc_discharge_map}`, runs one sim with those overrides against the baseline, returns summary + per-lap table + time series. Frontend: three-knob form, baseline-comparison delta cards, overlay chart.
5. **Retire Dash dashboard + unify Docker** -- webapp/ + FastAPI as a single Docker image; delete `dashboard/`; update README.

Deliberately **not** in this repo's roadmap: parameter sweeps, Pareto, multi-run comparison, coaching output. Those go to a separate repo.

## Architecture Guidance

- **No bandaid fixes — root cause only**: Never apply superficial patches, fudge factors, or tuning hacks to make results match. Every fix must address the actual root cause. This is especially critical in simulation work: if the sim output is wrong, the physics model or inputs are wrong — find out why. Adding correction factors or clamping outputs to hide errors destroys the simulation's predictive value and makes every downstream result untrustworthy. A simulation that's honestly wrong is more useful than one that's been patched to look right.
- **Modular by domain**: separate modules for battery model, drivetrain model, tire/vehicle dynamics, track representation, driver model, and lap simulation orchestration. Each module should be independently testable.
- **Simulation correctness first**: validate every model against real data before adding complexity. Numerical accuracy matters more than abstraction elegance.
- **Performance-aware**: sims should complete in seconds, not minutes, so the Simulate page feels interactive. Use NumPy/SciPy vectorized operations. Profile before optimizing. Leave room for a future sweep repo to reuse this core, so don't build structures that prevent vectorization later.
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

**Always use `model: "opus"` when deploying agents.** All Agent tool calls must specify the Opus 4.6 model to ensure maximum capability and reasoning quality.

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
