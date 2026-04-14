# FSAE EV Endurance Simulation

Endurance simulation and optimization for UConn Formula SAE Electric. Predicts lap times, energy consumption, and competition points to find optimal vehicle configuration and driver strategy.

## Cars

| | CT-16EV (2025) | CT-17EV (2026) |
|---|---|---|
| Pack | 110S4P Molicel P45B | 100S4P Molicel P50B |
| Mass (with driver) | 288 kg | ~279 kg |
| Motor/Inverter | Shared | Shared |
| Controls | Shared | Shared |

## Architecture

```
Vehicle Config (YAML)  →  Simulation Engine  →  Results  →  Dashboard
                              ↑
               Track (from GPS telemetry)
               Driver Strategy (swappable)
```

**Simulation method:** Quasi-static point-mass, calibrated against real AiM telemetry from the 2025 Michigan endurance event. For each track segment, resolves speed from force balance and driver strategy, steps battery state, enforces BMS limits.

**Modules:**

| Module | Purpose |
|---|---|
| `fsae_sim.vehicle` | Vehicle, powertrain, and battery configuration |
| `fsae_sim.track` | Track representation from GPS segments |
| `fsae_sim.driver` | Driver strategy / control policy (swappable) |
| `fsae_sim.sim` | Simulation engine |
| `fsae_sim.scoring` | FSAE endurance + efficiency point formulas |
| `fsae_sim.optimization` | Parameter sweep runner |
| `fsae_sim.analysis` | Post-processing metrics, Pareto computation |
| `fsae_sim.data` | Telemetry and simulation data loaders |
| `dashboard` | Dash web app for viewing results |

## Quick Start

### With Docker

```bash
docker compose -f docker/docker-compose.yaml up
# Browser → http://localhost:3000
```

### Without Docker

```bash
pip install -e ".[dev]"
python -m dashboard
# Browser → http://localhost:3000
```

### Run tests

```bash
pytest -v
```

## Project Structure

```
├── src/fsae_sim/          # Simulation Python package
│   ├── vehicle/           # Vehicle, powertrain, battery models
│   ├── track/             # Track representation
│   ├── driver/            # Driver strategy
│   ├── sim/               # Simulation engine
│   ├── scoring/           # FSAE scoring formulas
│   ├── optimization/      # Parameter sweeps
│   ├── analysis/          # Metrics and post-processing
│   └── data/              # Data loaders
├── dashboard/             # Dash web app (port 3000)
│   └── pages/             # Dashboard pages
├── configs/               # Vehicle config YAML files
├── Real-Car-Data-And-Stats/  # Telemetry and battery data
├── results/               # Simulation outputs (gitignored)
├── tests/                 # pytest test suite
└── docker/                # Dockerfile and compose
```

## Data

- **AiM telemetry:** `2025 Endurance Data.csv` — 20Hz, ~37k samples, ~100 channels from Michigan endurance
- **Voltt battery sim:** Cell and pack level CSVs — voltage, SOC, current, temperature, heat generation for 110S4P P45B
- **BMS tune:** `Endurance Tune2.txt` — discharge limits, SOC taper, inverter/motor parameters

## Roadmap

### Phase 1 — Foundation ✅ (Done)
Repository scaffold, Docker, dashboard skeleton, vehicle configs, data loaders

### Phase 2 — Core Simulation (Nearly Done)
Battery model, powertrain model, vehicle dynamics with 4-wheel Pacejka tire model, driver model (CalibratedStrategy, zone-based), simulation engine validated against real telemetry (~2% energy error, 8/8 metrics pass). Remaining: driver model finalization and quality/accuracy validation checks.

### Phase 3 — Optimization & Comparison (Next)
Swappable strategies, parameter sweeps, car comparison, Pareto frontier, dashboard buildout

### Phase 4 — Scoring & Decision Support (Future)
FSAE scoring model, field estimation, points maximization, final decision dashboard
