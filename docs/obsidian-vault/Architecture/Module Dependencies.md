---
title: Module Dependencies
tags: [architecture, modules]
---

# Module Dependencies

How the simulation modules depend on each other.

---

## Dependency Graph

```mermaid
flowchart BT
    LOADER["data.loader<br/>load_aim_csv()<br/>load_voltt_csv()"]

    VCONF["vehicle.vehicle<br/>VehicleConfig<br/>VehicleParams"]
    BCONF["vehicle.battery<br/>BatteryConfig"]
    PCONF["vehicle.powertrain<br/>PowertrainConfig"]

    BMOD["vehicle.battery_model<br/>BatteryModel"]
    PMOD["vehicle.powertrain_model<br/>PowertrainModel"]
    DMOD["vehicle.dynamics<br/>VehicleDynamics"]

    TRACK["track.track<br/>Track, Segment"]

    STRAT["driver.strategy<br/>DriverStrategy (ABC)"]
    IMPL["driver.strategies<br/>Replay / Coast / Brake"]

    ENGINE["sim.engine<br/>SimulationEngine"]

    VALID["analysis.validation<br/>ValidationReport"]
    SCORE["scoring.scoring<br/>EnduranceScore"]
    SWEEP["optimization.sweep<br/>SweepConfig"]

    LOADER --> BMOD
    LOADER --> TRACK
    LOADER --> IMPL

    BCONF --> BMOD
    PCONF --> PMOD
    VCONF --> DMOD

    BMOD --> ENGINE
    PMOD --> ENGINE
    DMOD --> ENGINE
    TRACK --> ENGINE
    IMPL --> ENGINE
    STRAT --> IMPL

    ENGINE --> VALID
    ENGINE --> SCORE
    ENGINE --> SWEEP

    style LOADER fill:#2d4a22,stroke:#4a7c34,color:#fff
    style ENGINE fill:#5c1a3a,stroke:#a32d6d,color:#fff
    style VALID fill:#3a1a5c,stroke:#6d2da3,color:#fff
    style SCORE fill:#3a1a5c,stroke:#6d2da3,color:#fff
    style SWEEP fill:#3a1a5c,stroke:#6d2da3,color:#fff
```

---

## Module Responsibility Matrix

| Module | Reads | Produces | Depends On |
|--------|-------|----------|------------|
| `data.loader` | CSV files | DataFrames | — (standalone) |
| `vehicle.vehicle` | YAML configs | VehicleConfig | battery, powertrain configs |
| `vehicle.battery` | — | BatteryConfig | — (pure config) |
| `vehicle.battery_model` | Voltt CSV (via loader) | OCV/R curves, state updates | BatteryConfig, loader |
| `vehicle.powertrain` | — | PowertrainConfig | — (pure config) |
| `vehicle.powertrain_model` | — | Torque/speed/power maps | PowertrainConfig |
| `vehicle.dynamics` | — | Forces, cornering speeds | VehicleParams |
| `track.track` | AiM CSV (via loader) | Segment list | loader |
| `driver.strategy` | — | ABC interface | track.Segment |
| `driver.strategies` | AiM CSV (via loader) | Control commands | strategy, dynamics, loader |
| `sim.engine` | All models + track + strategy | SimResult DataFrame | All above |
| `analysis.validation` | SimResult + AiM telemetry | ValidationReport | loader, engine output |

---

## Import Map

```
src/fsae_sim/
├── data/loader.py          → imported by: battery_model, track, strategies, validation
├── vehicle/
│   ├── vehicle.py          → imported by: engine (via __init__)
│   ├── battery.py          → imported by: battery_model, vehicle.py
│   ├── battery_model.py    → imported by: engine
│   ├── powertrain.py       → imported by: powertrain_model, vehicle.py
│   ├── powertrain_model.py → imported by: engine
│   └── dynamics.py         → imported by: engine, strategies
├── track/track.py          → imported by: engine
├── driver/
│   ├── strategy.py         → imported by: strategies, engine
│   └── strategies.py       → imported by: engine (user code)
└── sim/engine.py           → imported by: user code, validation
```

> [!tip] Key Design Principle
> **Config objects are frozen dataclasses** (immutable). Runtime models are mutable classes that take configs as constructor arguments. This separation means you can create one config and pass it to multiple model instances for parallel parameter sweeps.
