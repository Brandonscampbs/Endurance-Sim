---
title: Getting Started
tags: [setup, quickstart]
---

# Getting Started

> [!tip] Prerequisites
> - Python 3.12+
> - Git
> - (Optional) Docker & Docker Compose

---

## Local Setup

### 1. Clone and Install

```bash
git clone <repo-url>
cd Development-BC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install package with dev dependencies
pip install -e ".[dev]"
```

### 2. Run Tests

```bash
pytest tests/
```

Expected: All tests pass. Some tests may skip if data files are not present in `Real-Car-Data-And-Stats/`.

### 3. Run a Simulation

```python
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.powertrain_model import PowertrainModel
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.track import Track
from fsae_sim.driver.strategies import CoastOnlyStrategy
from fsae_sim.sim import SimulationEngine

# Load config
config = VehicleConfig.from_yaml("configs/ct16ev.yaml")

# Build models
battery = BatteryModel.from_config_and_data(
    config.battery,
    "Real-Car-Data-And-Stats/About-Energy-Volt-Simulations-2025-Pack/2025_Pack_cell.csv"
)
powertrain = PowertrainModel(config.powertrain)
dynamics = VehicleDynamics(config.vehicle)

# Load track from telemetry
track = Track.from_telemetry(
    "Real-Car-Data-And-Stats/2025 Endurance Data.csv"
)

# Choose a strategy
strategy = CoastOnlyStrategy(dynamics, coast_margin_ms=2.0)

# Run simulation
engine = SimulationEngine(config, track, strategy, battery)
result = engine.run(num_laps=1, initial_soc_pct=95.0)

# Inspect results
print(f"Lap time: {result.total_time_s:.1f} s")
print(f"Energy: {result.total_energy_kwh:.3f} kWh")
print(f"Final SOC: {result.final_soc:.1f}%")
print(result.states.head())
```

---

## Docker Setup

```bash
cd docker
docker-compose up --build
```

This starts the dashboard at [http://localhost:3000](http://localhost:3000).

### Docker Volumes

The compose file mounts these directories for live reload:

| Host Path | Container Path |
|-----------|---------------|
| `src/` | `/app/src/` |
| `dashboard/` | `/app/dashboard/` |
| `configs/` | `/app/configs/` |
| `tests/` | `/app/tests/` |
| `results/` | `/app/results/` |
| `Real-Car-Data-And-Stats/` | `/app/Real-Car-Data-And-Stats/` |

---

## Project Structure

```
Development-BC/
├── src/fsae_sim/          # Main simulation package
│   ├── vehicle/           # Vehicle, battery, powertrain, dynamics
│   ├── track/             # Track geometry
│   ├── driver/            # Driver strategies
│   ├── sim/               # Simulation engine
│   ├── scoring/           # FSAE scoring (stub)
│   ├── optimization/      # Parameter sweeps (stub)
│   ├── analysis/          # Validation, metrics
│   └── data/              # CSV loaders
├── configs/               # Vehicle YAML configs
├── tests/                 # pytest test suite
├── dashboard/             # Dash web app (stub)
├── docker/                # Dockerfile + compose
├── Real-Car-Data-And-Stats/  # Telemetry + battery data
├── results/               # Simulation outputs (gitignored)
└── docs/                  # Documentation
```

---

## Running Tests with Coverage

```bash
pytest tests/ --cov=fsae_sim --cov-report=html
# Open htmlcov/index.html in a browser
```

---

## Key Files to Start With

| If you want to... | Start here |
|-------------------|-----------|
| Understand the architecture | [[System Overview]] |
| See how data flows | [[Data Flow]] |
| Understand the physics | [[Quasi-Static Simulation]] |
| Read vehicle specs | [[CT-16EV (2025)]] |
| Modify a model | `src/fsae_sim/vehicle/*.py` |
| Add a driver strategy | [[Driver Strategies]] + `src/fsae_sim/driver/strategies.py` |
| Run a parameter sweep | [[Roadmap]] (Phase 3 — not yet implemented) |

See also: [[Roadmap]], [[Home]]
