# FSAE EV Endurance Simulation

Endurance simulation for UConn Formula SAE Electric (CT-16EV). Predicts lap time and energy for the Michigan endurance event; includes an FSAE scoring library for downstream tooling. Calibrated against real AiM telemetry from the 2025 Michigan endurance event.

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

**Method:** Quasi-static point-mass with 4-wheel Pacejka tires. Resolves speed from force balance per track segment, steps battery state, enforces BMS limits.

**Modules:**

| Module | Purpose |
|---|---|
| `fsae_sim.vehicle` | Vehicle, powertrain, and battery configuration |
| `fsae_sim.track` | Track representation from GPS |
| `fsae_sim.driver` | Driver strategy / control policy |
| `fsae_sim.sim` | Simulation engine |
| `fsae_sim.analysis` | Validation, telemetry analysis, FSAE scoring |
| `fsae_sim.data` | Telemetry + battery-sim CSV loaders |
| `backend` | FastAPI service |
| `webapp` | React + Vite SPA (Verification / Visualization / Simulate) |

## Webapp

Three pages, each answering one question about the simulator:

1. **Verification** — how close is the baseline sim to real Michigan 2025 telemetry?
2. **Visualization** — 3D playback of the car driving the track (real or sim).
3. **Simulate** — one-shot what-if: change max RPM, max torque, SOC discharge map; see how endurance time and energy change vs baseline.

## Quick Start

### Local dev (recommended while iterating)

Two terminals:

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

### Docker (full stack)

```bash
docker compose up --build
```

- Webapp → http://localhost:3000 (nginx serving the built React bundle, reverse-proxying `/api` to the backend)
- Backend → http://localhost:8001 (uvicorn, directly reachable for debugging)

The backend bind-mounts `src/`, `backend/`, `configs/`, and `Real-Car-Data-And-Stats/`, so backend Python edits pick up after restarting the container. Frontend edits require a rebuild or the local dev flow above.

### Tests

```bash
pytest -v
```

## Project Structure

```
├── src/fsae_sim/              # Simulation Python package
│   ├── vehicle/               # Vehicle, powertrain, battery models
│   ├── track/                 # Track representation
│   ├── driver/                # Driver strategies
│   ├── sim/                   # Simulation engine
│   ├── analysis/              # Validation, telemetry analysis, FSAE scoring
│   └── data/                  # Data loaders
├── backend/                   # FastAPI app (port 8001)
│   ├── routers/               # /api/* route handlers
│   ├── services/              # sim runner, telemetry, export logic
│   └── models/                # Pydantic response models
├── webapp/                    # React + Vite SPA (port 5173 dev, 3000 docker)
│   └── src/pages/             # verification, visualization, simulate
├── configs/                   # Vehicle config YAML files (ct16ev, ct17ev)
├── Real-Car-Data-And-Stats/   # Telemetry and battery data
├── results/                   # Simulation outputs (gitignored)
├── tests/                     # pytest test suite
├── scripts/                   # Data pipeline (cleaning, Fx transplant)
└── docker/                    # Backend Dockerfile (webapp Dockerfile in webapp/)
```

## Data

- **AiM telemetry:** `2025 Endurance Data.csv` — 20 Hz Michigan endurance (~22 km, 21 laps).
- **Voltt battery sim:** `..._cell.csv` + `..._pack.csv` for OCV-SOC and resistance calibration.
- **BMS tune:** `Endurance Tune2.txt` — discharge limits, SOC taper.
- **Motor map:** `emrax228_hv_cc_motor_map_long.csv` — 2D efficiency lookup.
- **Tires:** `Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir` (PAC2002, Fx transplanted from R25B donor).

See `docs/SIMULATOR_ISSUES.md` for known physics/code gaps.
