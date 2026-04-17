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
Vehicle Config (YAML)  →  Simulation Engine  →  FastAPI backend  →  React webapp
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
| `backend` | FastAPI service exposing sim results to the webapp |
| `webapp` | React + Vite SPA with Verification, Visualization, and Simulate pages |

## Webapp

The webapp has three pages, each tied to one question about the simulator:

1. **Verification** — how close is the baseline sim to real Michigan 2025 telemetry?
2. **Visualization** — 3D playback of the car driving the track (real or sim).
3. **Simulate** — one-shot what-if: change max RPM, max torque, and the SOC discharge map; see how endurance time and energy change vs baseline.

See `docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md` for the full scope and fix list.

## Quick Start

### Local dev (recommended while iterating)

In two terminals:

```bash
# Terminal 1: backend (FastAPI on :8001)
pip install -e ".[dev]"
python -m uvicorn backend.main:app --reload --port 8001
```

```bash
# Terminal 2: webapp (Vite dev server on :5173, proxies /api to :8001)
cd webapp
npm install
npm run dev
```

Open http://localhost:5173. Vite hot-reloads the UI; the backend hot-reloads on `.py` edits.

### Docker (one command, full stack)

```bash
docker compose up --build
```

- Webapp → http://localhost:3000 (nginx serving the built React bundle, reverse-proxying `/api` to the backend)
- Backend → http://localhost:8001 (uvicorn, directly reachable for debugging)

The backend bind-mounts `src/`, `backend/`, `configs/`, and `Real-Car-Data-And-Stats/`, so backend Python edits pick up after restarting the container. Frontend edits require a rebuild (`docker compose up --build`) or — easier — use the local dev flow above.

### Run tests

```bash
pytest -v
```

## Project Structure

```
├── src/fsae_sim/              # Simulation Python package
│   ├── vehicle/               # Vehicle, powertrain, battery models
│   ├── track/                 # Track representation
│   ├── driver/                # Driver strategy
│   ├── sim/                   # Simulation engine
│   ├── scoring/               # FSAE scoring formulas
│   ├── optimization/          # Parameter sweeps
│   ├── analysis/              # Metrics and post-processing
│   └── data/                  # Data loaders
├── backend/                   # FastAPI app (port 8001)
│   ├── routers/               # /api/* route handlers
│   ├── services/              # sim runner, telemetry, export logic
│   └── models/                # Pydantic response models
├── webapp/                    # React + Vite SPA (port 5173 dev, 3000 docker)
│   └── src/pages/             # verification, visualization, simulate
├── configs/                   # Vehicle config YAML files
├── Real-Car-Data-And-Stats/   # Telemetry and battery data
├── results/                   # Simulation outputs (gitignored)
├── tests/                     # pytest test suite
└── docker/                    # Backend Dockerfile (webapp Dockerfile lives in webapp/)
```

## Data

- **AiM telemetry:** `2025 Endurance Data.csv` — 20Hz, ~37k samples, ~100 channels from Michigan endurance
- **Voltt battery sim:** Cell and pack level CSVs — voltage, SOC, current, temperature, heat generation for 110S4P P45B
- **BMS tune:** `Endurance Tune2.txt` — discharge limits, SOC taper, inverter/motor parameters

## Roadmap

### Phase 1 — Foundation ✅ (Done)
Repository scaffold, Docker, webapp skeleton, vehicle configs, data loaders

### Phase 2 — Core Simulation (In Progress)
Battery model, powertrain model, vehicle dynamics with 4-wheel Pacejka tire model, driver model (CalibratedStrategy, zone-based), simulation engine validated against real telemetry (~2% energy error, 8/8 metrics pass). Remaining: physics model alignment and accuracy validation — see `docs/SIMULATOR_ISSUES.md` for open issues.

### Phase 3 — Verification polish + Simulate page (Next)
Close the residual physics gaps visible on the Verification page; ship the Simulate page (max RPM, max torque, SOC discharge-map sliders); keep Visualization honest so it can be used to spot physics bugs.

### Phase 4 — Scoring & Decision Support (Future)
FSAE scoring model, field estimation, points maximization, final decision output.
