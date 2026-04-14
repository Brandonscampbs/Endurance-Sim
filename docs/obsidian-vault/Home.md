---
title: FSAE EV Endurance Simulation
tags: [home, index]
---

# FSAE EV Endurance Simulation

> [!abstract] Project Overview
> A physics-based endurance simulation for **UConn Formula SAE Electric** — predicting lap times, energy consumption, and competition points to optimize vehicle configuration and driver strategy for the CT-16EV (2025) and CT-17EV (2026) cars.

---

## Quick Navigation

### Architecture & Design
- [[System Overview]] — High-level architecture and simulation approach
- [[Data Flow]] — How data moves through the simulation pipeline
- [[Module Dependencies]] — Dependency graph between all modules

### Simulation Modules
| Module | Description | Status |
|--------|-------------|--------|
| [[Vehicle Module]] | Config loading, vehicle parameters | Complete |
| [[Battery Model]] | Equivalent-circuit runtime model | Complete |
| [[Powertrain Model]] | Motor/inverter/gearbox model | Complete |
| [[Vehicle Dynamics]] | Force-balance with 4-wheel Pacejka tire model | Complete |
| [[Track Module]] | Track geometry from GPS telemetry | Complete |
| [[Driver Strategies]] | Control strategies (replay, coast, brake, CalibratedStrategy) | Complete |
| [[Simulation Engine]] | Main simulation loop orchestration | Complete |
| [[Data Loaders]] | CSV parsers for AiM and Voltt data | Complete |
| [[Analysis Module]] | Validation and post-processing (~2% energy error, 8/8 metrics pass) | Partial |
| [[Scoring Module]] | FSAE endurance/efficiency scoring | Stub |
| [[Dashboard]] | Dash/Plotly web visualization | Stub |

### Vehicle Data
- [[CT-16EV (2025)]] — The 2025 competition car (real telemetry available)
- [[CT-17EV (2026)]] — The 2026 design target car
- [[Vehicle Comparison]] — Side-by-side parameter comparison

### Data Assets
- [[Telemetry Data]] — AiM race logs from 2025 Michigan Endurance
- [[Battery Simulation Data]] — Voltt cell/pack simulations for both packs
- [[BMS Configuration]] — Discharge limits, SOC taper, voltage bounds

### Physics Reference
- [[Quasi-Static Simulation]] — The simulation methodology explained
- [[Battery Physics]] — Equivalent-circuit model, OCV, internal resistance
- [[Motor Torque Curve]] — Constant-torque and field-weakening regions
- [[Aerodynamic Forces]] — Drag, rolling resistance, grade forces

### Project Info
- [[Getting Started]] — How to set up and run the simulation
- [[Roadmap]] — 4-phase development plan and current status
- [[Glossary]] — Key terms and abbreviations

---

## Project Status

```mermaid
gantt
    title Development Phases
    dateFormat YYYY-MM-DD
    axisFormat %b %Y

    section Phase 1
    Foundation & Stubs           :done, p1, 2026-04-01, 2026-04-10

    section Phase 2
    Battery Model                :done, p2a, 2026-04-10, 2026-04-12
    Powertrain Model             :done, p2b, 2026-04-10, 2026-04-12
    Vehicle Dynamics             :done, p2c, 2026-04-10, 2026-04-12
    Track Extraction             :done, p2d, 2026-04-11, 2026-04-13
    Driver Strategies            :done, p2e, 2026-04-11, 2026-04-13
    Simulation Engine            :done, p2f, 2026-04-12, 2026-04-13
    Validation vs Telemetry      :active, p2g, 2026-04-13, 2026-04-20

    section Phase 3
    Parameter Sweeps             :p3a, after p2g, 14d
    Car Comparison               :p3b, after p2g, 14d
    Dashboard Pages              :p3c, after p3a, 14d

    section Phase 4
    FSAE Scoring                 :p4a, after p3c, 7d
    Points Optimization          :p4b, after p4a, 14d
```

---

> [!tip] Getting Started
> Open a terminal and run:
> ```bash
> pip install -e ".[dev]"
> pytest tests/
> ```
> See [[Getting Started]] for full setup instructions including Docker.
