# Webapp Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React + Three.js web application with a Validation page (sim vs telemetry comparison with track maps, overlay charts, tables) and a 3D Visualization page (animated wireframe car with per-wheel forces and playback controls), served by a FastAPI backend.

**Architecture:** FastAPI backend wraps the existing `fsae_sim` package to run simulations, load telemetry, and export structured JSON. React frontend (Vite + TypeScript + Tailwind) consumes the API. Validation page uses Plotly.js for charts and track maps. Visualization page uses React Three Fiber for 3D rendering with Zustand for playback state.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, Plotly.js, React Three Fiber, Three.js, Zustand, SWR, FastAPI, Pydantic v2, uvicorn.

**Spec:** `docs/superpowers/specs/2026-04-15-webapp-rework-design.md`

---

## Phase 1: Project Scaffolding

### Task 1: Scaffold React Frontend

**Files:**
- Create: `webapp/package.json`
- Create: `webapp/tsconfig.json`
- Create: `webapp/vite.config.ts`
- Create: `webapp/tailwind.config.js`
- Create: `webapp/postcss.config.js`
- Create: `webapp/index.html`
- Create: `webapp/src/main.tsx`
- Create: `webapp/src/App.tsx`
- Create: `webapp/src/index.css`

- [ ] **Step 1: Initialize Vite React TypeScript project**

```bash
cd C:/Users/brand/Development-BC
npm create vite@latest webapp -- --template react-ts
cd webapp
npm install
```

- [ ] **Step 2: Install dependencies**

```bash
cd C:/Users/brand/Development-BC/webapp
npm install react-router-dom zustand swr react-plotly.js plotly.js-dist-min @react-three/fiber @react-three/drei three tailwindcss @tailwindcss/vite
npm install -D @types/three
```

- [ ] **Step 3: Configure Tailwind**

Replace `webapp/src/index.css`:
```css
@import "tailwindcss";
```

Update `webapp/vite.config.ts`:
```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 4: Create minimal App shell**

Replace `webapp/src/App.tsx`:
```tsx
export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex">
      <aside className="w-56 bg-gray-900 border-r border-gray-800 p-4">
        <h1 className="text-lg font-bold mb-6">FSAE Sim</h1>
        <nav className="space-y-2">
          <div className="text-sm text-gray-400">Validation</div>
          <div className="text-sm text-gray-400">Visualization</div>
        </nav>
      </aside>
      <main className="flex-1 p-6">
        <p className="text-gray-500">App shell working.</p>
      </main>
    </div>
  )
}
```

- [ ] **Step 5: Verify dev server starts**

```bash
cd C:/Users/brand/Development-BC/webapp
npm run dev
```

Open http://localhost:5173 — should see dark page with sidebar and "App shell working."

- [ ] **Step 6: Commit**

```bash
cd C:/Users/brand/Development-BC
git add webapp/
git commit -m "feat(webapp): scaffold React + Vite + Tailwind frontend"
```

---

### Task 2: Scaffold FastAPI Backend

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/main.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/services/__init__.py`
- Create: `backend/models/__init__.py`

- [ ] **Step 1: Create backend package structure**

Create `backend/__init__.py` (empty file).

Create `backend/routers/__init__.py` (empty file).

Create `backend/services/__init__.py` (empty file).

Create `backend/models/__init__.py` (empty file).

- [ ] **Step 2: Write FastAPI app entry point**

Create `backend/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FSAE Sim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 3: Add uvicorn to project deps and verify**

```bash
cd C:/Users/brand/Development-BC
pip install fastapi uvicorn
python -m uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/health` returns `{"status":"ok"}`.

- [ ] **Step 4: Commit**

```bash
git add backend/
git commit -m "feat(backend): scaffold FastAPI backend with health endpoint"
```

---

## Phase 2: Backend Data Pipeline

### Task 3: Track Service — GPS to XY Coordinates + Sectors

**Files:**
- Create: `backend/services/track_service.py`
- Create: `backend/models/track.py`
- Test: `tests/backend/test_track_service.py`

This service loads the AiM GPS data, converts lat/lon to local XY meters (equirectangular projection), builds a track spline, and detects sectors (corners vs straights).

- [ ] **Step 1: Write Pydantic response models**

Create `backend/models/track.py`:
```python
from pydantic import BaseModel


class TrackPoint(BaseModel):
    x: float
    y: float
    distance_m: float


class Sector(BaseModel):
    name: str
    sector_type: str  # "straight" or "corner"
    start_m: float
    end_m: float


class TrackData(BaseModel):
    centerline: list[TrackPoint]
    sectors: list[Sector]
    curvature: list[float]
    total_distance_m: float
```

- [ ] **Step 2: Write failing test**

Create `tests/backend/__init__.py` (empty).

Create `tests/backend/test_track_service.py`:
```python
import numpy as np
import pytest
from backend.services.track_service import build_track_xy, detect_sectors


def test_build_track_xy_returns_xy_points():
    """XY output should have same number of points as distance bins and be in meters."""
    lats = np.array([42.0, 42.0001, 42.0002, 42.0001, 42.0])
    lons = np.array([-83.0, -83.0, -83.0001, -83.0001, -83.0])
    distances = np.array([0.0, 11.1, 22.2, 33.3, 44.4])
    points = build_track_xy(lats, lons, distances, bin_size_m=10.0)
    assert len(points) > 0
    # First point should be near origin
    assert abs(points[0].x) < 1.0
    assert abs(points[0].y) < 1.0
    # Points should have increasing distance
    for i in range(1, len(points)):
        assert points[i].distance_m > points[i - 1].distance_m


def test_detect_sectors_alternates_types():
    """Sectors should alternate between straight and corner."""
    curvatures = [0.0, 0.0, 0.05, 0.06, 0.0, 0.0, 0.0, 0.04, 0.0]
    distances = [float(i * 5) for i in range(len(curvatures))]
    sectors = detect_sectors(curvatures, distances, threshold=0.01)
    assert len(sectors) >= 3
    types = [s.sector_type for s in sectors]
    assert "corner" in types
    assert "straight" in types
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd C:/Users/brand/Development-BC
python -m pytest tests/backend/test_track_service.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.services.track_service'`

- [ ] **Step 4: Implement track service**

Create `backend/services/track_service.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from backend.models.track import Sector, TrackData, TrackPoint
from fsae_sim.data.loader import load_aim_csv
from fsae_sim.track.track import Track

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AIM_CSV = _PROJECT_ROOT / "Real-Car-Data-And-Stats" / "2025 Endurance Data.csv"

_CURVATURE_CORNER_THRESHOLD = 0.01  # 1/m — above this is a corner
_GPS_POS_ACC_BAD = 200.0
_MIN_SPEED_KMH = 5.0


def build_track_xy(
    lats: np.ndarray,
    lons: np.ndarray,
    distances: np.ndarray,
    bin_size_m: float = 1.0,
) -> list[TrackPoint]:
    """Convert GPS lat/lon to local XY meters via equirectangular projection."""
    lat_ref = lats[0]
    lon_ref = lons[0]
    cos_lat = np.cos(np.radians(lat_ref))

    # Degrees to meters (equirectangular)
    x_raw = (lons - lon_ref) * cos_lat * 111_320.0
    y_raw = (lats - lat_ref) * 110_540.0

    # Remove duplicate distances for spline
    mask = np.diff(distances, prepend=-1) > 0.01
    d_clean = distances[mask]
    x_clean = x_raw[mask]
    y_clean = y_raw[mask]

    if len(d_clean) < 4:
        return [TrackPoint(x=0.0, y=0.0, distance_m=0.0)]

    # Cubic spline interpolation to uniform spacing
    cs_x = CubicSpline(d_clean, x_clean)
    cs_y = CubicSpline(d_clean, y_clean)

    d_uniform = np.arange(0, d_clean[-1], bin_size_m)
    points = [
        TrackPoint(x=float(cs_x(d)), y=float(cs_y(d)), distance_m=float(d))
        for d in d_uniform
    ]
    return points


def detect_sectors(
    curvatures: list[float],
    distances: list[float],
    threshold: float = _CURVATURE_CORNER_THRESHOLD,
) -> list[Sector]:
    """Segment the track into corner and straight sectors."""
    sectors: list[Sector] = []
    corner_count = 0
    straight_count = 0

    i = 0
    while i < len(curvatures):
        is_corner = abs(curvatures[i]) > threshold
        start_idx = i
        while i < len(curvatures) and (abs(curvatures[i]) > threshold) == is_corner:
            i += 1
        end_idx = i - 1

        if is_corner:
            corner_count += 1
            name = f"Turn {corner_count}"
            sector_type = "corner"
        else:
            straight_count += 1
            name = f"Straight {straight_count}"
            sector_type = "straight"

        sectors.append(Sector(
            name=name,
            sector_type=sector_type,
            start_m=distances[start_idx],
            end_m=distances[min(end_idx, len(distances) - 1)],
        ))

    return sectors


def _load_best_lap_gps(aim_df: pd.DataFrame, track: Track) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract GPS data for the lap with the best GPS quality."""
    from fsae_sim.analysis.validation import detect_lap_boundaries

    boundaries = detect_lap_boundaries(aim_df)
    if not boundaries:
        raise ValueError("No laps detected in telemetry")

    # Score each lap by mean GPS accuracy (lower = better, 200 = invalid)
    best_score = float("inf")
    best_lap_idx = 0
    for idx, (start, end, _) in enumerate(boundaries):
        lap_slice = aim_df.iloc[start:end]
        acc = lap_slice["GPS PosAccuracy"]
        valid = acc[acc < _GPS_POS_ACC_BAD]
        if len(valid) == 0:
            continue
        score = valid.mean()
        if score < best_score:
            best_score = score
            best_lap_idx = idx

    start, end, _ = boundaries[best_lap_idx]
    lap_df = aim_df.iloc[start:end].copy()

    # Filter bad GPS
    mask = (
        (lap_df["GPS PosAccuracy"] < _GPS_POS_ACC_BAD)
        & (lap_df["GPS Speed"] > _MIN_SPEED_KMH)
    )
    lap_df = lap_df[mask]

    lats = lap_df["GPS Latitude"].values
    lons = lap_df["GPS Longitude"].values
    dists = lap_df["Distance on GPS Speed"].values
    # Normalize distance to start of lap
    dists = dists - dists[0]

    return lats, lons, dists


def get_track_data() -> TrackData:
    """Build complete track data with XY coordinates and sectors."""
    _, aim_df = load_aim_csv(str(_AIM_CSV))
    track = Track.from_telemetry(df=aim_df)

    lats, lons, dists = _load_best_lap_gps(aim_df, track)

    centerline = build_track_xy(lats, lons, dists, bin_size_m=1.0)

    curvatures = [float(s.curvature) for s in track.segments]
    seg_distances = [float(s.distance_start_m) for s in track.segments]
    sectors = detect_sectors(curvatures, seg_distances)

    return TrackData(
        centerline=centerline,
        sectors=sectors,
        curvature=curvatures,
        total_distance_m=track.total_distance_m,
    )
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/backend/test_track_service.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/models/track.py backend/services/track_service.py tests/backend/
git commit -m "feat(backend): add track service with GPS→XY conversion and sector detection"
```

---

### Task 4: Simulation Runner + Telemetry Service

**Files:**
- Create: `backend/services/sim_runner.py`
- Create: `backend/services/telemetry_service.py`
- Test: `tests/backend/test_sim_runner.py`

These services wrap the existing `fsae_sim` modules following the pattern in `dashboard/data/sim_runner.py`.

- [ ] **Step 1: Write telemetry service**

Create `backend/services/telemetry_service.py`:
```python
from functools import lru_cache
from pathlib import Path

import pandas as pd

from fsae_sim.analysis.validation import detect_lap_boundaries
from fsae_sim.data.loader import load_aim_csv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AIM_CSV = _PROJECT_ROOT / "Real-Car-Data-And-Stats" / "2025 Endurance Data.csv"


@lru_cache(maxsize=1)
def get_telemetry() -> pd.DataFrame:
    """Load and cache the AiM telemetry CSV."""
    _, df = load_aim_csv(str(_AIM_CSV))
    return df


@lru_cache(maxsize=1)
def get_lap_boundaries() -> list[tuple[int, int, float]]:
    """Detect lap boundaries. Returns list of (start_idx, end_idx, lap_distance_m)."""
    df = get_telemetry()
    return detect_lap_boundaries(df)


def get_lap_data(lap_number: int) -> pd.DataFrame:
    """Extract telemetry for a single lap (1-indexed)."""
    boundaries = get_lap_boundaries()
    if lap_number < 1 or lap_number > len(boundaries):
        raise ValueError(f"Lap {lap_number} not found. Available: 1-{len(boundaries)}")
    start, end, _ = boundaries[lap_number - 1]
    df = get_telemetry()
    lap_df = df.iloc[start:end].copy()
    # Normalize distance to start of lap
    d0 = lap_df["Distance on GPS Speed"].iloc[0]
    lap_df["lap_distance_m"] = lap_df["Distance on GPS Speed"] - d0
    return lap_df


def get_lap_gps_quality() -> list[dict]:
    """Score each lap's GPS quality. Returns list of {lap_number, quality_score, time_s}."""
    df = get_telemetry()
    boundaries = get_lap_boundaries()
    results = []
    for idx, (start, end, _) in enumerate(boundaries):
        lap_slice = df.iloc[start:end]
        acc = lap_slice["GPS PosAccuracy"]
        valid = acc[acc < 200.0]
        quality = float(valid.mean()) if len(valid) > 0 else 999.0
        time_s = float(lap_slice["Time"].iloc[-1] - lap_slice["Time"].iloc[0])
        results.append({
            "lap_number": idx + 1,
            "gps_quality_score": round(quality, 1),
            "time_s": round(time_s, 2),
            "valid_gps_pct": round(100 * len(valid) / len(acc), 1),
        })
    return results
```

- [ ] **Step 2: Write simulation runner service**

Create `backend/services/sim_runner.py`:
```python
from functools import lru_cache
from pathlib import Path

from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimResult, SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel

from backend.services.telemetry_service import get_telemetry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "ct16ev.yaml"
_VOLTT_CELL_PATH = (
    _PROJECT_ROOT
    / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)


@lru_cache(maxsize=1)
def get_vehicle_config() -> VehicleConfig:
    return VehicleConfig.from_yaml(str(_CONFIG_PATH))


@lru_cache(maxsize=1)
def get_track() -> Track:
    aim_df = get_telemetry()
    return Track.from_telemetry(df=aim_df)


@lru_cache(maxsize=1)
def get_battery_model() -> BatteryModel:
    vehicle = get_vehicle_config()
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(get_telemetry())
    return battery


@lru_cache(maxsize=1)
def get_baseline_result() -> SimResult:
    """Run baseline 22-lap simulation with CalibratedStrategy."""
    vehicle = get_vehicle_config()
    track = get_track()
    battery = get_battery_model()
    aim_df = get_telemetry()

    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    engine = SimulationEngine(vehicle, track, strategy, battery)
    return engine.run(num_laps=22, initial_soc_pct=95.0, initial_temp_c=29.0)


def run_single_lap_sim(lap_number: int = 1) -> SimResult:
    """Run a single-lap simulation for visualization."""
    vehicle = get_vehicle_config()
    track = get_track()
    aim_df = get_telemetry()

    # Use fresh battery (not calibrated-pack) for single-lap
    battery = BatteryModel.from_config_and_data(vehicle.battery, str(_VOLTT_CELL_PATH))
    battery.calibrate_pack_from_telemetry(aim_df)

    strategy = CalibratedStrategy.from_telemetry(aim_df, track)
    engine = SimulationEngine(vehicle, track, strategy, battery)
    return engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=29.0)
```

- [ ] **Step 3: Write failing test**

Create `tests/backend/test_sim_runner.py`:
```python
import pytest
from backend.services.sim_runner import get_vehicle_config, get_track


def test_vehicle_config_loads():
    config = get_vehicle_config()
    assert config.name is not None
    assert config.vehicle.mass_kg > 0


def test_track_loads():
    track = get_track()
    assert track.num_segments > 100  # Michigan ~200 segments
    assert track.total_distance_m > 900  # ~1005m
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/backend/test_sim_runner.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/sim_runner.py backend/services/telemetry_service.py tests/backend/test_sim_runner.py
git commit -m "feat(backend): add simulation runner and telemetry services"
```

---

### Task 5: Validation Data Export Service

**Files:**
- Create: `backend/services/validation_export.py`
- Create: `backend/models/validation.py`
- Test: `tests/backend/test_validation_export.py`

Aligns sim and real data on a shared distance axis and computes all validation comparison data.

- [ ] **Step 1: Write Pydantic models**

Create `backend/models/validation.py`:
```python
from pydantic import BaseModel


class TraceData(BaseModel):
    """Paired sim and real values at uniform distance points."""
    distance_m: list[float]
    sim: list[float]
    real: list[float]


class ValidationMetricResult(BaseModel):
    name: str
    unit: str
    sim_value: float
    real_value: float
    error_pct: float
    threshold_pct: float
    passed: bool


class SectorComparison(BaseModel):
    name: str
    sector_type: str
    sim_time_s: float
    real_time_s: float
    delta_s: float
    delta_pct: float
    sim_avg_speed_kmh: float
    real_avg_speed_kmh: float
    speed_delta_pct: float


class LapSummary(BaseModel):
    lap_number: int
    sim_time_s: float
    real_time_s: float
    time_error_pct: float
    sim_energy_kwh: float
    real_energy_kwh: float
    energy_error_pct: float
    mean_speed_error_pct: float


class ValidationResponse(BaseModel):
    lap_number: int
    speed: TraceData
    throttle: TraceData
    brake: TraceData
    power: TraceData
    soc: TraceData
    lat_accel: TraceData
    track_sim_speed: list[float]  # per-centerline-point speed for track map coloring
    track_real_speed: list[float]
    sectors: list[SectorComparison]
    metrics: list[ValidationMetricResult]


class AllLapsResponse(BaseModel):
    laps: list[LapSummary]
    metrics: list[ValidationMetricResult]
```

- [ ] **Step 2: Write failing test**

Create `tests/backend/test_validation_export.py`:
```python
import numpy as np
import pandas as pd
import pytest

from backend.services.validation_export import align_traces


def test_align_traces_interpolates_to_uniform_spacing():
    sim_dist = np.array([0, 5, 10, 15, 20])
    sim_vals = np.array([0, 10, 20, 30, 40])
    real_dist = np.array([0, 4, 8, 12, 16, 20])
    real_vals = np.array([0, 8, 16, 24, 32, 40])

    trace = align_traces(sim_dist, sim_vals, real_dist, real_vals, spacing_m=5.0)
    assert len(trace.distance_m) == len(trace.sim) == len(trace.real)
    assert trace.distance_m[0] == 0.0
    # At distance=10, sim should be 20, real should be 20
    idx_10 = trace.distance_m.index(10.0)
    assert abs(trace.sim[idx_10] - 20.0) < 0.1
    assert abs(trace.real[idx_10] - 20.0) < 0.1
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/backend/test_validation_export.py -v
```
Expected: FAIL

- [ ] **Step 4: Implement validation export**

Create `backend/services/validation_export.py`:
```python
import numpy as np
import pandas as pd

from backend.models.track import Sector
from backend.models.validation import (
    AllLapsResponse,
    LapSummary,
    SectorComparison,
    TraceData,
    ValidationMetricResult,
    ValidationResponse,
)
from backend.services.sim_runner import get_baseline_result, get_track
from backend.services.telemetry_service import get_lap_boundaries, get_lap_data, get_telemetry
from backend.services.track_service import detect_sectors, get_track_data
from fsae_sim.analysis.validation import validate_full_endurance

_GRAVITY = 9.81


def align_traces(
    sim_dist: np.ndarray,
    sim_vals: np.ndarray,
    real_dist: np.ndarray,
    real_vals: np.ndarray,
    spacing_m: float = 1.0,
) -> TraceData:
    """Interpolate sim and real onto a shared uniform distance axis."""
    d_min = max(sim_dist[0], real_dist[0])
    d_max = min(sim_dist[-1], real_dist[-1])
    d_uniform = np.arange(d_min, d_max, spacing_m)

    sim_interp = np.interp(d_uniform, sim_dist, sim_vals)
    real_interp = np.interp(d_uniform, real_dist, real_vals)

    return TraceData(
        distance_m=[round(float(d), 1) for d in d_uniform],
        sim=[round(float(v), 3) for v in sim_interp],
        real=[round(float(v), 3) for v in real_interp],
    )


def _compute_sector_comparison(
    sim_states: pd.DataFrame,
    real_df: pd.DataFrame,
    sectors: list[Sector],
    lap_number: int,
) -> list[SectorComparison]:
    """Compare sim vs real timing and speed for each sector."""
    sim_lap = sim_states[sim_states["lap"] == lap_number - 1]  # 0-indexed
    comparisons = []

    for sector in sectors:
        # Sim data in sector
        sim_sec = sim_lap[
            (sim_lap["distance_m"] % sim_lap["distance_m"].max() >= sector.start_m)
            & (sim_lap["distance_m"] % sim_lap["distance_m"].max() < sector.end_m)
        ]
        # Real data in sector
        real_sec = real_df[
            (real_df["lap_distance_m"] >= sector.start_m)
            & (real_df["lap_distance_m"] < sector.end_m)
        ]

        sim_time = float(sim_sec["segment_time_s"].sum()) if len(sim_sec) > 0 else 0.0
        real_time = float(real_sec["Time"].diff().sum()) if len(real_sec) > 1 else 0.0
        sim_speed = float(sim_sec["speed_kmh"].mean()) if len(sim_sec) > 0 else 0.0
        real_speed = float(real_sec["GPS Speed"].mean()) if len(real_sec) > 0 else 0.0

        delta_s = sim_time - real_time
        delta_pct = (delta_s / real_time * 100) if real_time > 0 else 0.0
        speed_delta_pct = ((sim_speed - real_speed) / real_speed * 100) if real_speed > 0 else 0.0

        comparisons.append(SectorComparison(
            name=sector.name,
            sector_type=sector.sector_type,
            sim_time_s=round(sim_time, 3),
            real_time_s=round(real_time, 3),
            delta_s=round(delta_s, 3),
            delta_pct=round(delta_pct, 1),
            sim_avg_speed_kmh=round(sim_speed, 1),
            real_avg_speed_kmh=round(real_speed, 1),
            speed_delta_pct=round(speed_delta_pct, 1),
        ))

    return comparisons


def get_validation_data(lap_number: int) -> ValidationResponse:
    """Produce all validation comparison data for a single lap."""
    result = get_baseline_result()
    sim_states = result.states
    track_data = get_track_data()
    real_df = get_lap_data(lap_number)

    # Sim data for this lap (0-indexed in sim)
    sim_lap = sim_states[sim_states["lap"] == lap_number - 1].copy()
    sim_dist_in_lap = sim_lap["distance_m"].values - sim_lap["distance_m"].values[0]
    real_dist = real_df["lap_distance_m"].values

    # Build overlay traces
    speed = align_traces(
        sim_dist_in_lap, sim_lap["speed_kmh"].values,
        real_dist, real_df["GPS Speed"].values,
    )
    throttle = align_traces(
        sim_dist_in_lap, sim_lap["throttle_pct"].values * 100,
        real_dist, real_df["Throttle Pos"].values,
    )

    # Normalize brake: sim is 0-1, real is bar pressure — normalize both to 0-100
    real_brake = real_df["FBrakePressure"].values
    real_brake_max = real_brake.max() if real_brake.max() > 0 else 1.0
    brake = align_traces(
        sim_dist_in_lap, sim_lap["brake_pct"].values * 100,
        real_dist, real_brake / real_brake_max * 100,
    )

    power = align_traces(
        sim_dist_in_lap, sim_lap["electrical_power_w"].values,
        real_dist, (real_df["Pack Voltage"].values * real_df["Pack Current"].values),
    )
    soc = align_traces(
        sim_dist_in_lap, sim_lap["soc_pct"].values,
        real_dist, real_df["State of Charge"].values,
    )

    # Lateral acceleration: sim = v^2 * curvature / g, real = GPS LatAcc
    sim_lat_g = sim_lap["speed_ms"].values ** 2 * np.abs(sim_lap["curvature"].values) / _GRAVITY
    lat_accel = align_traces(
        sim_dist_in_lap, sim_lat_g,
        real_dist, np.abs(real_df["GPS LatAcc"].values),
    )

    # Track map speed coloring (interpolate speeds onto centerline points)
    cl_dists = np.array([p.distance_m for p in track_data.centerline])
    track_sim_speed = np.interp(cl_dists, sim_dist_in_lap, sim_lap["speed_kmh"].values)
    track_real_speed = np.interp(cl_dists, real_dist, real_df["GPS Speed"].values)

    # Sector comparison
    sectors = _compute_sector_comparison(sim_states, real_df, track_data.sectors, lap_number)

    # Accuracy metrics from existing validation module
    aim_df = get_telemetry()
    report = validate_full_endurance(
        sim_states, aim_df,
        result.total_time_s, result.final_soc,
        result.total_energy_kwh, result.laps_completed,
    )
    metrics = [
        ValidationMetricResult(
            name=m.name, unit=m.unit,
            sim_value=round(m.simulation_value, 3),
            real_value=round(m.telemetry_value, 3),
            error_pct=round(m.relative_error_pct, 2),
            threshold_pct=m.target_pct,
            passed=m.passed,
        )
        for m in report.metrics
    ]

    return ValidationResponse(
        lap_number=lap_number,
        speed=speed,
        throttle=throttle,
        brake=brake,
        power=power,
        soc=soc,
        lat_accel=lat_accel,
        track_sim_speed=[round(float(v), 1) for v in track_sim_speed],
        track_real_speed=[round(float(v), 1) for v in track_real_speed],
        sectors=sectors,
        metrics=metrics,
    )


def get_all_laps_summary() -> AllLapsResponse:
    """Produce per-lap summary table and aggregate metrics."""
    result = get_baseline_result()
    sim_states = result.states
    boundaries = get_lap_boundaries()
    aim_df = get_telemetry()

    laps: list[LapSummary] = []
    for lap_idx in range(min(result.laps_completed, len(boundaries))):
        sim_lap = sim_states[sim_states["lap"] == lap_idx]
        start, end, _ = boundaries[lap_idx]
        real_lap = aim_df.iloc[start:end]

        sim_time = float(sim_lap["segment_time_s"].sum())
        real_time = float(real_lap["Time"].iloc[-1] - real_lap["Time"].iloc[0])
        time_err = abs(sim_time - real_time) / real_time * 100 if real_time > 0 else 0

        # Energy: sim = sum of power * dt, real = integral of V*I*dt
        sim_energy = float(sim_lap["electrical_power_w"].values @ sim_lap["segment_time_s"].values) / 3_600_000
        real_dt = real_lap["Time"].diff().fillna(0).values
        real_power = real_lap["Pack Voltage"].values * real_lap["Pack Current"].values
        real_energy = float(np.sum(real_power * real_dt)) / 3_600_000
        energy_err = abs(sim_energy - real_energy) / abs(real_energy) * 100 if real_energy != 0 else 0

        sim_mean_speed = float(sim_lap["speed_kmh"].mean())
        real_mean_speed = float(real_lap["GPS Speed"].mean())
        speed_err = abs(sim_mean_speed - real_mean_speed) / real_mean_speed * 100 if real_mean_speed > 0 else 0

        laps.append(LapSummary(
            lap_number=lap_idx + 1,
            sim_time_s=round(sim_time, 2),
            real_time_s=round(real_time, 2),
            time_error_pct=round(time_err, 1),
            sim_energy_kwh=round(sim_energy, 4),
            real_energy_kwh=round(real_energy, 4),
            energy_error_pct=round(energy_err, 1),
            mean_speed_error_pct=round(speed_err, 1),
        ))

    # Aggregate metrics
    report = validate_full_endurance(
        sim_states, aim_df,
        result.total_time_s, result.final_soc,
        result.total_energy_kwh, result.laps_completed,
    )
    metrics = [
        ValidationMetricResult(
            name=m.name, unit=m.unit,
            sim_value=round(m.simulation_value, 3),
            real_value=round(m.telemetry_value, 3),
            error_pct=round(m.relative_error_pct, 2),
            threshold_pct=m.target_pct,
            passed=m.passed,
        )
        for m in report.metrics
    ]

    return AllLapsResponse(laps=laps, metrics=metrics)
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/backend/test_validation_export.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/models/validation.py backend/services/validation_export.py tests/backend/test_validation_export.py
git commit -m "feat(backend): add validation data export with trace alignment and sector comparison"
```

---

### Task 6: Visualization Data Export Service

**Files:**
- Create: `backend/services/visualization_export.py`
- Create: `backend/models/visualization.py`
- Test: `tests/backend/test_visualization_export.py`

Computes per-frame 3D animation data: XY position, heading, per-wheel forces, roll, pitch.

- [ ] **Step 1: Write Pydantic models**

Create `backend/models/visualization.py`:
```python
from pydantic import BaseModel


class WheelForce(BaseModel):
    fx: float  # longitudinal force (N)
    fy: float  # lateral force (N)
    fz: float  # normal load (N)
    grip_util: float  # 0-1 ratio of used vs available grip


class Frame(BaseModel):
    time_s: float
    distance_m: float
    x: float
    y: float
    heading_rad: float
    speed_kmh: float
    throttle_pct: float
    brake_pct: float
    motor_rpm: float
    motor_torque_nm: float
    soc_pct: float
    pack_voltage_v: float
    pack_current_a: float
    roll_rad: float
    pitch_rad: float
    action: str
    wheels: list[WheelForce]  # [FL, FR, RL, RR]


class VisualizationResponse(BaseModel):
    lap_number: int
    total_time_s: float
    total_frames: int
    frames: list[Frame]
    track_centerline_x: list[float]
    track_centerline_y: list[float]
    track_speed_colors: list[float]  # speed at each centerline point for coloring
```

- [ ] **Step 2: Write failing test**

Create `tests/backend/test_visualization_export.py`:
```python
import math
import numpy as np
import pytest

from backend.services.visualization_export import compute_heading, distribute_drive_force


def test_compute_heading_from_xy():
    """Heading should point in direction of travel."""
    xs = np.array([0, 1, 2, 3])
    ys = np.array([0, 0, 0, 0])
    headings = compute_heading(xs, ys)
    # Moving in +x direction — heading should be 0 (east)
    assert len(headings) == 4
    for h in headings:
        assert abs(h) < 0.01


def test_distribute_drive_force_rear_wheel_drive():
    """CT-16EV is rear-wheel drive. Drive force should be on rear wheels only."""
    fl, fr, rl, rr = distribute_drive_force(1000.0, 0.0)
    assert fl == 0.0
    assert fr == 0.0
    assert rl > 0
    assert rr > 0
    assert abs(rl + rr - 1000.0) < 0.01
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/backend/test_visualization_export.py -v
```
Expected: FAIL

- [ ] **Step 4: Implement visualization export**

Create `backend/services/visualization_export.py`:
```python
import math

import numpy as np
from scipy.interpolate import CubicSpline

from backend.models.visualization import Frame, VisualizationResponse, WheelForce
from backend.services.sim_runner import (
    get_track,
    get_vehicle_config,
    run_single_lap_sim,
)
from backend.services.telemetry_service import get_lap_data, get_telemetry
from backend.services.track_service import build_track_xy, _load_best_lap_gps, get_track_data

_GRAVITY = 9.81


def compute_heading(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute heading angle (radians, 0=east, CCW positive) from XY path."""
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    return np.arctan2(dy, dx)


def distribute_drive_force(
    total_drive_n: float,
    total_regen_n: float,
) -> tuple[float, float, float, float]:
    """Distribute longitudinal force to wheels. CT-16EV is rear-wheel drive.

    Returns: (fl_fx, fr_fx, rl_fx, rr_fx)
    """
    net = total_drive_n - total_regen_n
    # Rear-wheel drive: all drive force on rear axle, split 50/50
    if net >= 0:
        return (0.0, 0.0, net / 2, net / 2)
    else:
        # Regen also on rear (motor is rear)
        return (0.0, 0.0, net / 2, net / 2)


def _compute_lateral_forces(
    speed_ms: float,
    curvature: float,
    mass_kg: float,
) -> tuple[float, float, float, float]:
    """Estimate per-wheel lateral force from cornering.

    Returns: (fl_fy, fr_fy, rl_fy, rr_fy)
    """
    lat_accel = speed_ms ** 2 * curvature  # m/s^2, signed
    total_lat_force = mass_kg * lat_accel
    # Approximate 53% front weight distribution
    front_share = 0.53
    front_total = total_lat_force * front_share
    rear_total = total_lat_force * (1 - front_share)
    return (front_total / 2, front_total / 2, rear_total / 2, rear_total / 2)


def _compute_tire_loads(
    speed_ms: float,
    curvature: float,
    longitudinal_g: float,
    mass_kg: float,
    cg_height_m: float = 0.2794,
    front_track_m: float = 1.2,
    rear_track_m: float = 1.2,
    wheelbase_m: float = 1.549,
    weight_dist_front: float = 0.53,
) -> tuple[float, float, float, float]:
    """Compute normal loads on each tire (FL, FR, RL, RR)."""
    weight = mass_kg * _GRAVITY
    front_static = weight * weight_dist_front / 2
    rear_static = weight * (1 - weight_dist_front) / 2

    # Longitudinal load transfer
    long_transfer = mass_kg * longitudinal_g * _GRAVITY * cg_height_m / wheelbase_m / 2

    # Lateral load transfer
    lat_g = speed_ms ** 2 * abs(curvature) / _GRAVITY
    lat_transfer_front = mass_kg * lat_g * _GRAVITY * cg_height_m * 0.53 / front_track_m / 2
    lat_transfer_rear = mass_kg * lat_g * _GRAVITY * cg_height_m * 0.47 / rear_track_m / 2

    sign = 1.0 if curvature >= 0 else -1.0  # positive = right turn

    fl = front_static - long_transfer + sign * lat_transfer_front
    fr = front_static - long_transfer - sign * lat_transfer_front
    rl = rear_static + long_transfer + sign * lat_transfer_rear
    rr = rear_static + long_transfer - sign * lat_transfer_rear

    return (max(fl, 0), max(fr, 0), max(rl, 0), max(rr, 0))


def _estimate_grip_utilization(
    fx: float, fy: float, fz: float, mu: float = 1.3,
) -> float:
    """Ratio of combined force to available grip (friction circle)."""
    if fz <= 0:
        return 0.0
    combined = math.sqrt(fx ** 2 + fy ** 2)
    available = fz * mu
    return min(combined / available, 1.0)


def _compute_roll_pitch(
    lat_g: float,
    long_g: float,
    roll_stiffness_nm_per_deg: float = 800.0,
    pitch_stiffness_nm_per_deg: float = 600.0,
    mass_kg: float = 288.0,
    cg_height_m: float = 0.2794,
) -> tuple[float, float]:
    """Estimate roll and pitch angles (radians) from load transfer."""
    roll_moment = mass_kg * abs(lat_g) * _GRAVITY * cg_height_m
    roll_deg = roll_moment / roll_stiffness_nm_per_deg if roll_stiffness_nm_per_deg > 0 else 0
    roll_rad = math.radians(roll_deg) * (1 if lat_g >= 0 else -1)

    pitch_moment = mass_kg * long_g * _GRAVITY * cg_height_m
    pitch_deg = pitch_moment / pitch_stiffness_nm_per_deg if pitch_stiffness_nm_per_deg > 0 else 0
    pitch_rad = math.radians(pitch_deg)

    return (roll_rad, pitch_rad)


def get_visualization_data(source: str = "sim") -> VisualizationResponse:
    """Build per-frame 3D visualization data for the best GPS quality lap.

    Args:
        source: "sim" for simulation data, "real" for telemetry data
    """
    track_data = get_track_data()
    aim_df = get_telemetry()
    track = get_track()

    # Build XY spline from GPS
    lats, lons, dists = _load_best_lap_gps(aim_df, track)
    centerline = build_track_xy(lats, lons, dists, bin_size_m=1.0)
    cl_x = np.array([p.x for p in centerline])
    cl_y = np.array([p.y for p in centerline])
    cl_d = np.array([p.distance_m for p in centerline])

    # XY spline for interpolating car position from distance
    cs_x = CubicSpline(cl_d, cl_x)
    cs_y = CubicSpline(cl_d, cl_y)

    vehicle = get_vehicle_config()
    mass_kg = vehicle.vehicle.mass_kg + 68  # car + driver

    if source == "sim":
        result = run_single_lap_sim()
        sim_df = result.states
        frames = _build_sim_frames(sim_df, cs_x, cs_y, cl_d, mass_kg)
        total_time = result.total_time_s
    else:
        # Real telemetry — find best lap
        from backend.services.telemetry_service import get_lap_gps_quality
        quality = get_lap_gps_quality()
        best = min(quality, key=lambda q: q["gps_quality_score"])
        real_df = get_lap_data(best["lap_number"])
        frames = _build_real_frames(real_df, cs_x, cs_y, cl_d, mass_kg)
        total_time = best["time_s"]

    # Track speed coloring
    if source == "sim":
        result = run_single_lap_sim()
        sim_df = result.states
        sim_speeds = np.interp(cl_d, sim_df["distance_m"].values, sim_df["speed_kmh"].values)
        track_speeds = [round(float(v), 1) for v in sim_speeds]
    else:
        real_df = get_lap_data(min(quality, key=lambda q: q["gps_quality_score"])["lap_number"])
        real_speeds = np.interp(cl_d, real_df["lap_distance_m"].values, real_df["GPS Speed"].values)
        track_speeds = [round(float(v), 1) for v in real_speeds]

    return VisualizationResponse(
        lap_number=1,
        total_time_s=round(total_time, 2),
        total_frames=len(frames),
        frames=frames,
        track_centerline_x=[round(float(x), 3) for x in cl_x],
        track_centerline_y=[round(float(y), 3) for y in cl_y],
        track_speed_colors=track_speeds,
    )


def _build_sim_frames(
    sim_df, cs_x, cs_y, cl_d, mass_kg: float,
) -> list[Frame]:
    """Convert sim result rows into 3D frames."""
    frames = []
    max_d = cl_d[-1]

    for _, row in sim_df.iterrows():
        d = float(row["distance_m"]) % max_d
        d = min(d, max_d - 0.1)  # clamp to spline domain
        x = float(cs_x(d))
        y = float(cs_y(d))

        # Heading from spline derivative
        dx = float(cs_x(d, 1))
        dy = float(cs_y(d, 1))
        heading = math.atan2(dy, dx)

        speed_ms = float(row["speed_ms"])
        curvature = float(row["curvature"])
        drive_force = float(row["drive_force_n"])
        regen_force = float(row["regen_force_n"])

        # Per-wheel longitudinal forces
        fl_fx, fr_fx, rl_fx, rr_fx = distribute_drive_force(drive_force, regen_force)

        # Per-wheel lateral forces
        fl_fy, fr_fy, rl_fy, rr_fy = _compute_lateral_forces(speed_ms, curvature, mass_kg)

        # Normal loads
        long_g = float(row["net_force_n"]) / (mass_kg * _GRAVITY)
        fl_fz, fr_fz, rl_fz, rr_fz = _compute_tire_loads(
            speed_ms, curvature, long_g, mass_kg,
        )

        wheels = [
            WheelForce(fx=round(fl_fx, 1), fy=round(fl_fy, 1), fz=round(fl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fl_fx, fl_fy, fl_fz), 3)),
            WheelForce(fx=round(fr_fx, 1), fy=round(fr_fy, 1), fz=round(fr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fr_fx, fr_fy, fr_fz), 3)),
            WheelForce(fx=round(rl_fx, 1), fy=round(rl_fy, 1), fz=round(rl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rl_fx, rl_fy, rl_fz), 3)),
            WheelForce(fx=round(rr_fx, 1), fy=round(rr_fy, 1), fz=round(rr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rr_fx, rr_fy, rr_fz), 3)),
        ]

        lat_g = speed_ms ** 2 * curvature / _GRAVITY
        roll_rad, pitch_rad = _compute_roll_pitch(lat_g, long_g, mass_kg=mass_kg)

        frames.append(Frame(
            time_s=round(float(row["time_s"]), 3),
            distance_m=round(d, 2),
            x=round(x, 3),
            y=round(y, 3),
            heading_rad=round(heading, 4),
            speed_kmh=round(float(row["speed_kmh"]), 1),
            throttle_pct=round(float(row["throttle_pct"]) * 100, 1),
            brake_pct=round(float(row["brake_pct"]) * 100, 1),
            motor_rpm=round(float(row["motor_rpm"]), 0),
            motor_torque_nm=round(float(row["motor_torque_nm"]), 1),
            soc_pct=round(float(row["soc_pct"]), 2),
            pack_voltage_v=round(float(row["pack_voltage_v"]), 1),
            pack_current_a=round(float(row["pack_current_a"]), 1),
            roll_rad=round(roll_rad, 4),
            pitch_rad=round(pitch_rad, 4),
            action=str(row["action"]),
            wheels=wheels,
        ))

    return frames


def _build_real_frames(
    real_df, cs_x, cs_y, cl_d, mass_kg: float,
) -> list[Frame]:
    """Convert real telemetry into 3D frames."""
    frames = []
    max_d = cl_d[-1]
    t0 = real_df["Time"].iloc[0]

    for _, row in real_df.iterrows():
        d = float(row["lap_distance_m"])
        if d < 0 or d >= max_d:
            continue
        x = float(cs_x(d))
        y = float(cs_y(d))
        dx = float(cs_x(d, 1))
        dy = float(cs_y(d, 1))
        heading = math.atan2(dy, dx)

        speed_kmh = float(row["GPS Speed"])
        speed_ms = speed_kmh / 3.6
        throttle = float(row["Throttle Pos"])
        brake_raw = float(row.get("FBrakePressure", 0))
        brake_max = real_df["FBrakePressure"].max() if "FBrakePressure" in real_df.columns else 1.0
        brake_pct = (brake_raw / brake_max * 100) if brake_max > 0 else 0

        # Determine action from inputs
        if throttle > 5:
            action = "throttle"
        elif brake_raw > 2:
            action = "brake"
        else:
            action = "coast"

        # Approximate forces from telemetry
        lat_g = float(row.get("GPS LatAcc", 0))
        curvature = lat_g * _GRAVITY / (speed_ms ** 2) if speed_ms > 1 else 0

        # No direct force data from telemetry — estimate from torque request
        torque_req = float(row.get("LVCU Torque Req", 0))
        gear_ratio = 3.6363
        tire_radius = 0.2042
        drive_force = torque_req * gear_ratio / tire_radius * 0.97 if torque_req > 0 else 0
        regen_force = abs(torque_req) * gear_ratio / tire_radius * 0.85 if torque_req < 0 else 0

        fl_fx, fr_fx, rl_fx, rr_fx = distribute_drive_force(drive_force, regen_force)
        fl_fy, fr_fy, rl_fy, rr_fy = _compute_lateral_forces(speed_ms, curvature, mass_kg)
        long_g = 0.0  # hard to get from telemetry without speed derivative
        fl_fz, fr_fz, rl_fz, rr_fz = _compute_tire_loads(speed_ms, curvature, long_g, mass_kg)

        wheels = [
            WheelForce(fx=round(fl_fx, 1), fy=round(fl_fy, 1), fz=round(fl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fl_fx, fl_fy, fl_fz), 3)),
            WheelForce(fx=round(fr_fx, 1), fy=round(fr_fy, 1), fz=round(fr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fr_fx, fr_fy, fr_fz), 3)),
            WheelForce(fx=round(rl_fx, 1), fy=round(rl_fy, 1), fz=round(rl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rl_fx, rl_fy, rl_fz), 3)),
            WheelForce(fx=round(rr_fx, 1), fy=round(rr_fy, 1), fz=round(rr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rr_fx, rr_fy, rr_fz), 3)),
        ]

        roll_rad, pitch_rad = _compute_roll_pitch(lat_g, long_g, mass_kg=mass_kg)

        frames.append(Frame(
            time_s=round(float(row["Time"]) - t0, 3),
            distance_m=round(d, 2),
            x=round(x, 3), y=round(y, 3),
            heading_rad=round(heading, 4),
            speed_kmh=round(speed_kmh, 1),
            throttle_pct=round(throttle, 1),
            brake_pct=round(brake_pct, 1),
            motor_rpm=round(float(row.get("RPM", 0)), 0),
            motor_torque_nm=round(float(row.get("LVCU Torque Req", 0)), 1),
            soc_pct=round(float(row.get("State of Charge", 0)), 2),
            pack_voltage_v=round(float(row.get("Pack Voltage", 0)), 1),
            pack_current_a=round(float(row.get("Pack Current", 0)), 1),
            roll_rad=round(roll_rad, 4),
            pitch_rad=round(pitch_rad, 4),
            action=action,
            wheels=wheels,
        ))

    return frames
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/backend/test_visualization_export.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/models/visualization.py backend/services/visualization_export.py tests/backend/test_visualization_export.py
git commit -m "feat(backend): add visualization data export with per-frame 3D data"
```

---

### Task 7: Backend API Routes

**Files:**
- Create: `backend/routers/laps.py`
- Create: `backend/routers/track.py`
- Create: `backend/routers/validation.py`
- Create: `backend/routers/visualization.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Create all route modules**

Create `backend/routers/laps.py`:
```python
from fastapi import APIRouter

from backend.services.telemetry_service import get_lap_gps_quality

router = APIRouter(prefix="/api", tags=["laps"])


@router.get("/laps")
def list_laps():
    return {"laps": get_lap_gps_quality()}
```

Create `backend/routers/track.py`:
```python
from fastapi import APIRouter

from backend.services.track_service import get_track_data

router = APIRouter(prefix="/api", tags=["track"])


@router.get("/track")
def get_track():
    return get_track_data()
```

Create `backend/routers/validation.py`:
```python
from fastapi import APIRouter, Path, Query

from backend.services.validation_export import get_all_laps_summary, get_validation_data

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/validation/{lap}")
def validation_for_lap(lap: int = Path(ge=1, le=30)):
    return get_validation_data(lap)


@router.get("/validation")
def validation_all_laps():
    return get_all_laps_summary()
```

Create `backend/routers/visualization.py`:
```python
from fastapi import APIRouter, Query

from backend.services.visualization_export import get_visualization_data

router = APIRouter(prefix="/api", tags=["visualization"])


@router.get("/visualization")
def visualization(source: str = Query(default="sim", regex="^(sim|real)$")):
    return get_visualization_data(source=source)
```

- [ ] **Step 2: Register routes in main.py**

Replace `backend/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import laps, track, validation, visualization

app = FastAPI(title="FSAE Sim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(laps.router)
app.include_router(track.router)
app.include_router(validation.router)
app.include_router(visualization.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 3: Verify server starts and health endpoint works**

```bash
cd C:/Users/brand/Development-BC
python -m uvicorn backend.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/health` → `{"status":"ok"}`
Verify: `curl http://localhost:8000/api/laps` → should return lap list (may take a moment on first load)

- [ ] **Step 4: Commit**

```bash
git add backend/routers/ backend/main.py
git commit -m "feat(backend): add API routes for laps, track, validation, and visualization"
```

---

## Phase 3: Frontend Shared Infrastructure

### Task 8: App Shell — Router, Sidebar, Theme

**Files:**
- Create: `webapp/src/App.tsx`
- Create: `webapp/src/components/Sidebar.tsx`
- Create: `webapp/src/components/LoadingSpinner.tsx`
- Create: `webapp/src/pages/validation/ValidationPage.tsx` (stub)
- Create: `webapp/src/pages/visualization/VisualizationPage.tsx` (stub)

- [ ] **Step 1: Create Sidebar component**

Create `webapp/src/components/Sidebar.tsx`:
```tsx
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Validation' },
  { to: '/visualization', label: 'Visualization' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <h1 className="text-lg font-bold tracking-tight">FSAE Sim</h1>
        <p className="text-xs text-gray-500 mt-1">CT-16EV · Michigan 2025</p>
      </div>
      <nav className="p-3 space-y-1">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `block px-3 py-2 rounded text-sm transition-colors ${
                isActive
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
```

- [ ] **Step 2: Create LoadingSpinner**

Create `webapp/src/components/LoadingSpinner.tsx`:
```tsx
export default function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-gray-700 border-t-green-500 rounded-full animate-spin mx-auto" />
        <p className="mt-3 text-sm text-gray-500">{message}</p>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create page stubs**

Create `webapp/src/pages/validation/ValidationPage.tsx`:
```tsx
export default function ValidationPage() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Validation</h2>
      <p className="text-gray-500">Validation page — components will be added here.</p>
    </div>
  )
}
```

Create `webapp/src/pages/visualization/VisualizationPage.tsx`:
```tsx
export default function VisualizationPage() {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">3D Visualization</h2>
      <p className="text-gray-500">Visualization page — components will be added here.</p>
    </div>
  )
}
```

- [ ] **Step 4: Wire up App with Router**

Replace `webapp/src/App.tsx`:
```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import ValidationPage from './pages/validation/ValidationPage'
import VisualizationPage from './pages/visualization/VisualizationPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100 flex">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<ValidationPage />} />
            <Route path="/visualization" element={<VisualizationPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
```

- [ ] **Step 5: Verify in browser**

```bash
cd C:/Users/brand/Development-BC/webapp && npm run dev
```

Open http://localhost:5173 — sidebar with two links, both navigate correctly.

- [ ] **Step 6: Commit**

```bash
git add webapp/src/
git commit -m "feat(webapp): add router, sidebar navigation, and page stubs"
```

---

### Task 9: API Client + SWR Hooks + Zustand Stores

**Files:**
- Create: `webapp/src/api/client.ts`
- Create: `webapp/src/stores/playbackStore.ts`
- Create: `webapp/src/stores/validationStore.ts`
- Create: `webapp/src/utils/formatters.ts`

- [ ] **Step 1: Create API client with SWR hooks**

Create `webapp/src/api/client.ts`:
```ts
import useSWR from 'swr'

const API_BASE = '/api'

async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

// Types matching backend Pydantic models
export interface TrackPoint { x: number; y: number; distance_m: number }
export interface Sector { name: string; sector_type: string; start_m: number; end_m: number }
export interface TrackData {
  centerline: TrackPoint[]; sectors: Sector[]; curvature: number[]; total_distance_m: number
}

export interface LapInfo { lap_number: number; gps_quality_score: number; time_s: number; valid_gps_pct: number }
export interface TraceData { distance_m: number[]; sim: number[]; real: number[] }
export interface ValidationMetric {
  name: string; unit: string; sim_value: number; real_value: number;
  error_pct: number; threshold_pct: number; passed: boolean
}
export interface SectorComparison {
  name: string; sector_type: string; sim_time_s: number; real_time_s: number;
  delta_s: number; delta_pct: number; sim_avg_speed_kmh: number;
  real_avg_speed_kmh: number; speed_delta_pct: number
}
export interface LapSummary {
  lap_number: number; sim_time_s: number; real_time_s: number; time_error_pct: number;
  sim_energy_kwh: number; real_energy_kwh: number; energy_error_pct: number;
  mean_speed_error_pct: number
}
export interface ValidationResponse {
  lap_number: number; speed: TraceData; throttle: TraceData; brake: TraceData;
  power: TraceData; soc: TraceData; lat_accel: TraceData;
  track_sim_speed: number[]; track_real_speed: number[];
  sectors: SectorComparison[]; metrics: ValidationMetric[]
}
export interface AllLapsResponse { laps: LapSummary[]; metrics: ValidationMetric[] }

export interface WheelForce { fx: number; fy: number; fz: number; grip_util: number }
export interface VizFrame {
  time_s: number; distance_m: number; x: number; y: number; heading_rad: number;
  speed_kmh: number; throttle_pct: number; brake_pct: number; motor_rpm: number;
  motor_torque_nm: number; soc_pct: number; pack_voltage_v: number; pack_current_a: number;
  roll_rad: number; pitch_rad: number; action: string; wheels: WheelForce[]
}
export interface VisualizationResponse {
  lap_number: number; total_time_s: number; total_frames: number; frames: VizFrame[];
  track_centerline_x: number[]; track_centerline_y: number[]; track_speed_colors: number[]
}

// SWR hooks
export function useLaps() {
  return useSWR<{ laps: LapInfo[] }>(`${API_BASE}/laps`, fetcher)
}

export function useTrack() {
  return useSWR<TrackData>(`${API_BASE}/track`, fetcher)
}

export function useValidation(lap: number | null) {
  return useSWR<ValidationResponse>(
    lap ? `${API_BASE}/validation/${lap}` : null, fetcher
  )
}

export function useAllLaps() {
  return useSWR<AllLapsResponse>(`${API_BASE}/validation`, fetcher)
}

export function useVisualization(source: 'sim' | 'real') {
  return useSWR<VisualizationResponse>(
    `${API_BASE}/visualization?source=${source}`, fetcher
  )
}
```

- [ ] **Step 2: Create Zustand stores**

Create `webapp/src/stores/playbackStore.ts`:
```ts
import { create } from 'zustand'

export type CameraMode = 'chase' | 'birdseye' | 'orbit'

interface PlaybackState {
  isPlaying: boolean
  speed: number  // 0.5, 1, 2, 5
  currentFrame: number
  totalFrames: number
  cameraMode: CameraMode
  showForces: boolean
  showTrackColor: boolean
  dataSource: 'sim' | 'real'

  play: () => void
  pause: () => void
  togglePlay: () => void
  setSpeed: (s: number) => void
  setFrame: (f: number) => void
  nextFrame: () => void
  prevFrame: () => void
  setTotalFrames: (n: number) => void
  setCameraMode: (m: CameraMode) => void
  toggleForces: () => void
  toggleTrackColor: () => void
  setDataSource: (s: 'sim' | 'real') => void
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  isPlaying: false,
  speed: 1,
  currentFrame: 0,
  totalFrames: 0,
  cameraMode: 'chase',
  showForces: true,
  showTrackColor: true,
  dataSource: 'sim',

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  togglePlay: () => set(s => ({ isPlaying: !s.isPlaying })),
  setSpeed: (speed) => set({ speed }),
  setFrame: (f) => set({ currentFrame: Math.max(0, Math.min(f, get().totalFrames - 1)) }),
  nextFrame: () => { const s = get(); if (s.currentFrame < s.totalFrames - 1) set({ currentFrame: s.currentFrame + 1 }) },
  prevFrame: () => { const s = get(); if (s.currentFrame > 0) set({ currentFrame: s.currentFrame - 1 }) },
  setTotalFrames: (n) => set({ totalFrames: n }),
  setCameraMode: (cameraMode) => set({ cameraMode }),
  toggleForces: () => set(s => ({ showForces: !s.showForces })),
  toggleTrackColor: () => set(s => ({ showTrackColor: !s.showTrackColor })),
  setDataSource: (dataSource) => set({ dataSource, currentFrame: 0, isPlaying: false }),
}))
```

Create `webapp/src/stores/validationStore.ts`:
```ts
import { create } from 'zustand'

interface ValidationState {
  selectedLap: number | null  // null = "all laps"
  setSelectedLap: (lap: number | null) => void
}

export const useValidationStore = create<ValidationState>((set) => ({
  selectedLap: null,
  setSelectedLap: (lap) => set({ selectedLap: lap }),
}))
```

- [ ] **Step 3: Create formatters**

Create `webapp/src/utils/formatters.ts`:
```ts
export function formatSpeed(kmh: number): string {
  return `${kmh.toFixed(1)} km/h`
}

export function formatTime(seconds: number): string {
  const min = Math.floor(seconds / 60)
  const sec = seconds % 60
  return `${min}:${sec.toFixed(1).padStart(4, '0')}`
}

export function formatEnergy(kwh: number): string {
  return `${kwh.toFixed(3)} kWh`
}

export function formatPercent(pct: number): string {
  return `${pct.toFixed(1)}%`
}

export function formatForce(n: number): string {
  return `${Math.round(n)} N`
}
```

- [ ] **Step 4: Commit**

```bash
git add webapp/src/api/ webapp/src/stores/ webapp/src/utils/
git commit -m "feat(webapp): add API client, SWR hooks, Zustand stores, and formatters"
```

---

## Phase 4: Validation Page

### Task 10: Track Maps Component

**Files:**
- Create: `webapp/src/pages/validation/TrackMaps.tsx`

- [ ] **Step 1: Implement side-by-side Plotly track maps**

Create `webapp/src/pages/validation/TrackMaps.tsx`:
```tsx
import Plot from 'react-plotly.js'
import { TrackData, ValidationResponse } from '../../api/client'

interface Props {
  track: TrackData
  validation: ValidationResponse
}

const COLOR_SCALE: [number, string][] = [
  [0, '#3b82f6'],    // blue — slow
  [0.25, '#22c55e'], // green
  [0.5, '#eab308'],  // yellow
  [0.75, '#f97316'], // orange
  [1, '#ef4444'],    // red — fast
]

export default function TrackMaps({ track, validation }: Props) {
  const xs = track.centerline.map(p => p.x)
  const ys = track.centerline.map(p => p.y)

  const maxSpeed = Math.max(
    ...validation.track_sim_speed,
    ...validation.track_real_speed,
  )
  const minSpeed = Math.min(
    ...validation.track_sim_speed.filter(v => v > 0),
    ...validation.track_real_speed.filter(v => v > 0),
  )

  const layout = (title: string): Partial<Plotly.Layout> => ({
    title: { text: title, font: { color: '#9ca3af', size: 14 } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { visible: false, scaleanchor: 'y', scaleratio: 1 },
    yaxis: { visible: false },
    margin: { t: 40, b: 10, l: 10, r: 10 },
    showlegend: false,
  })

  const makeTrace = (speeds: number[]): Plotly.Data => ({
    type: 'scatter',
    mode: 'markers',
    x: xs,
    y: ys,
    marker: {
      color: speeds,
      colorscale: COLOR_SCALE as unknown as Plotly.ColorScale,
      cmin: minSpeed,
      cmax: maxSpeed,
      size: 4,
      colorbar: { title: 'km/h', tickfont: { color: '#9ca3af' }, titlefont: { color: '#9ca3af' } },
    },
    hovertemplate: 'Speed: %{marker.color:.1f} km/h<extra></extra>',
  })

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-gray-900 rounded-lg p-2">
        <Plot
          data={[makeTrace(validation.track_sim_speed)]}
          layout={layout('Simulation Speed')}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          style={{ height: 400 }}
        />
      </div>
      <div className="bg-gray-900 rounded-lg p-2">
        <Plot
          data={[makeTrace(validation.track_real_speed)]}
          layout={layout('Real Telemetry Speed')}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          style={{ height: 400 }}
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add webapp/src/pages/validation/TrackMaps.tsx
git commit -m "feat(webapp): add side-by-side speed-colored track map component"
```

---

### Task 11: Overlay Charts Component

**Files:**
- Create: `webapp/src/pages/validation/OverlayCharts.tsx`

- [ ] **Step 1: Implement 6 synced overlay charts**

Create `webapp/src/pages/validation/OverlayCharts.tsx`:
```tsx
import Plot from 'react-plotly.js'
import { TraceData, ValidationResponse } from '../../api/client'

interface Props {
  validation: ValidationResponse
}

function OverlayChart({ trace, title, yLabel }: { trace: TraceData; title: string; yLabel: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-2">
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            x: trace.distance_m,
            y: trace.sim,
            name: 'Sim',
            line: { color: '#22c55e', width: 1.5 },
          },
          {
            type: 'scatter',
            mode: 'lines',
            x: trace.distance_m,
            y: trace.real,
            name: 'Real',
            line: { color: '#3b82f6', width: 1.5 },
          },
        ]}
        layout={{
          title: { text: title, font: { color: '#9ca3af', size: 13 } },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: 'Distance (m)',
            color: '#6b7280',
            gridcolor: '#1f2937',
            zerolinecolor: '#374151',
          },
          yaxis: {
            title: yLabel,
            color: '#6b7280',
            gridcolor: '#1f2937',
            zerolinecolor: '#374151',
          },
          legend: {
            font: { color: '#9ca3af' },
            bgcolor: 'transparent',
            x: 1, xanchor: 'right', y: 1,
          },
          margin: { t: 40, b: 50, l: 60, r: 20 },
          hovermode: 'x unified',
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
        style={{ height: 250 }}
      />
    </div>
  )
}

export default function OverlayCharts({ validation }: Props) {
  const charts: { trace: TraceData; title: string; yLabel: string }[] = [
    { trace: validation.speed, title: 'Speed vs Distance', yLabel: 'Speed (km/h)' },
    { trace: validation.throttle, title: 'Throttle vs Distance', yLabel: 'Throttle (%)' },
    { trace: validation.brake, title: 'Brake vs Distance', yLabel: 'Brake (%)' },
    { trace: validation.power, title: 'Electrical Power vs Distance', yLabel: 'Power (W)' },
    { trace: validation.soc, title: 'SOC vs Distance', yLabel: 'SOC (%)' },
    { trace: validation.lat_accel, title: 'Lateral Acceleration vs Distance', yLabel: 'Lat Accel (g)' },
  ]

  return (
    <div className="space-y-4">
      {charts.map(({ trace, title, yLabel }) => (
        <OverlayChart key={title} trace={trace} title={title} yLabel={yLabel} />
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add webapp/src/pages/validation/OverlayCharts.tsx
git commit -m "feat(webapp): add 6 overlay chart components for validation comparison"
```

---

### Task 12: Sector Table, Lap Table, and Metric Cards

**Files:**
- Create: `webapp/src/pages/validation/SectorTable.tsx`
- Create: `webapp/src/pages/validation/LapTable.tsx`
- Create: `webapp/src/pages/validation/MetricCards.tsx`

- [ ] **Step 1: Implement SectorTable**

Create `webapp/src/pages/validation/SectorTable.tsx`:
```tsx
import { SectorComparison } from '../../api/client'

function deltaColor(pct: number): string {
  const abs = Math.abs(pct)
  if (abs < 5) return 'text-green-400'
  if (abs < 10) return 'text-yellow-400'
  return 'text-red-400'
}

export default function SectorTable({ sectors }: { sectors: SectorComparison[] }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-4 py-3 border-b border-gray-800">
        Sector Breakdown
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase">
              <th className="px-4 py-2 text-left">Sector</th>
              <th className="px-4 py-2 text-left">Type</th>
              <th className="px-4 py-2 text-right">Sim Time</th>
              <th className="px-4 py-2 text-right">Real Time</th>
              <th className="px-4 py-2 text-right">Delta</th>
              <th className="px-4 py-2 text-right">Sim Speed</th>
              <th className="px-4 py-2 text-right">Real Speed</th>
              <th className="px-4 py-2 text-right">Speed Delta</th>
            </tr>
          </thead>
          <tbody>
            {sectors.map((s) => (
              <tr key={s.name} className="border-t border-gray-800 hover:bg-gray-800/50">
                <td className="px-4 py-2 font-medium">{s.name}</td>
                <td className="px-4 py-2 text-gray-400">{s.sector_type}</td>
                <td className="px-4 py-2 text-right">{s.sim_time_s.toFixed(2)}s</td>
                <td className="px-4 py-2 text-right">{s.real_time_s.toFixed(2)}s</td>
                <td className={`px-4 py-2 text-right ${deltaColor(s.delta_pct)}`}>
                  {s.delta_s > 0 ? '+' : ''}{s.delta_s.toFixed(2)}s ({s.delta_pct.toFixed(1)}%)
                </td>
                <td className="px-4 py-2 text-right">{s.sim_avg_speed_kmh.toFixed(1)}</td>
                <td className="px-4 py-2 text-right">{s.real_avg_speed_kmh.toFixed(1)}</td>
                <td className={`px-4 py-2 text-right ${deltaColor(s.speed_delta_pct)}`}>
                  {s.speed_delta_pct > 0 ? '+' : ''}{s.speed_delta_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Implement LapTable**

Create `webapp/src/pages/validation/LapTable.tsx`:
```tsx
import { LapSummary } from '../../api/client'

function errorColor(pct: number): string {
  if (pct < 5) return 'text-green-400'
  if (pct < 10) return 'text-yellow-400'
  return 'text-red-400'
}

export default function LapTable({ laps, selectedLap }: { laps: LapSummary[]; selectedLap: number | null }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-4 py-3 border-b border-gray-800">
        Per-Lap Summary
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase">
              <th className="px-4 py-2 text-left">Lap</th>
              <th className="px-4 py-2 text-right">Sim Time</th>
              <th className="px-4 py-2 text-right">Real Time</th>
              <th className="px-4 py-2 text-right">Time Err</th>
              <th className="px-4 py-2 text-right">Sim Energy</th>
              <th className="px-4 py-2 text-right">Real Energy</th>
              <th className="px-4 py-2 text-right">Energy Err</th>
              <th className="px-4 py-2 text-right">Speed Err</th>
            </tr>
          </thead>
          <tbody>
            {laps.map((l) => (
              <tr
                key={l.lap_number}
                className={`border-t border-gray-800 ${
                  selectedLap === l.lap_number ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                }`}
              >
                <td className="px-4 py-2 font-medium">#{l.lap_number}</td>
                <td className="px-4 py-2 text-right">{l.sim_time_s.toFixed(2)}s</td>
                <td className="px-4 py-2 text-right">{l.real_time_s.toFixed(2)}s</td>
                <td className={`px-4 py-2 text-right ${errorColor(l.time_error_pct)}`}>
                  {l.time_error_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right">{l.sim_energy_kwh.toFixed(3)}</td>
                <td className="px-4 py-2 text-right">{l.real_energy_kwh.toFixed(3)}</td>
                <td className={`px-4 py-2 text-right ${errorColor(l.energy_error_pct)}`}>
                  {l.energy_error_pct.toFixed(1)}%
                </td>
                <td className={`px-4 py-2 text-right ${errorColor(l.mean_speed_error_pct)}`}>
                  {l.mean_speed_error_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Implement MetricCards**

Create `webapp/src/pages/validation/MetricCards.tsx`:
```tsx
import { ValidationMetric } from '../../api/client'

export default function MetricCards({ metrics }: { metrics: ValidationMetric[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {metrics.map((m) => (
        <div
          key={m.name}
          className={`rounded-lg p-4 border ${
            m.passed
              ? 'bg-green-950/30 border-green-800'
              : 'bg-red-950/30 border-red-800'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400 uppercase">{m.name}</span>
            <span
              className={`text-xs font-bold px-2 py-0.5 rounded ${
                m.passed ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'
              }`}
            >
              {m.passed ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div className="text-lg font-bold">{m.error_pct.toFixed(1)}%</div>
          <div className="text-xs text-gray-500 mt-1">
            Threshold: {m.threshold_pct}% · Sim: {m.sim_value.toFixed(2)} · Real: {m.real_value.toFixed(2)} {m.unit}
          </div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add webapp/src/pages/validation/SectorTable.tsx webapp/src/pages/validation/LapTable.tsx webapp/src/pages/validation/MetricCards.tsx
git commit -m "feat(webapp): add sector table, lap table, and metric cards components"
```

---

### Task 13: Validation Page Assembly

**Files:**
- Modify: `webapp/src/pages/validation/ValidationPage.tsx`

- [ ] **Step 1: Assemble all validation components**

Replace `webapp/src/pages/validation/ValidationPage.tsx`:
```tsx
import { useValidation, useAllLaps, useTrack, useLaps } from '../../api/client'
import { useValidationStore } from '../../stores/validationStore'
import LoadingSpinner from '../../components/LoadingSpinner'
import TrackMaps from './TrackMaps'
import OverlayCharts from './OverlayCharts'
import SectorTable from './SectorTable'
import LapTable from './LapTable'
import MetricCards from './MetricCards'

export default function ValidationPage() {
  const { selectedLap, setSelectedLap } = useValidationStore()
  const { data: lapsData } = useLaps()
  const { data: track, isLoading: trackLoading } = useTrack()
  const { data: validation, isLoading: validationLoading } = useValidation(selectedLap)
  const { data: allLaps, isLoading: allLapsLoading } = useAllLaps()

  // Default to best GPS quality lap
  const bestLap = lapsData?.laps
    ? [...lapsData.laps].sort((a, b) => a.gps_quality_score - b.gps_quality_score)[0]?.lap_number ?? 1
    : 1

  const activeLap = selectedLap ?? bestLap

  // Set initial lap on first data load
  if (lapsData && selectedLap === null) {
    setSelectedLap(bestLap)
  }

  return (
    <div className="space-y-6">
      {/* Header + Lap Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Validation</h2>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-400">Lap:</label>
          <select
            value={selectedLap ?? 'all'}
            onChange={(e) => setSelectedLap(e.target.value === 'all' ? null : Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          >
            <option value="all">All Laps</option>
            {lapsData?.laps.map((l) => (
              <option key={l.lap_number} value={l.lap_number}>
                Lap {l.lap_number} — {l.time_s.toFixed(1)}s (GPS: {l.gps_quality_score})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Single-lap view */}
      {selectedLap !== null && (
        <>
          {trackLoading || validationLoading ? (
            <LoadingSpinner message="Running simulation and loading telemetry..." />
          ) : track && validation ? (
            <>
              <TrackMaps track={track} validation={validation} />
              <OverlayCharts validation={validation} />
              <SectorTable sectors={validation.sectors} />
              <MetricCards metrics={validation.metrics} />
            </>
          ) : (
            <p className="text-red-400">Failed to load data.</p>
          )}
        </>
      )}

      {/* All-laps view */}
      {selectedLap === null && (
        <>
          {allLapsLoading ? (
            <LoadingSpinner message="Computing all-laps summary..." />
          ) : allLaps ? (
            <>
              <MetricCards metrics={allLaps.metrics} />
              <LapTable laps={allLaps.laps} selectedLap={null} />
            </>
          ) : (
            <p className="text-red-400">Failed to load data.</p>
          )}
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify in browser**

Start both servers:
```bash
# Terminal 1:
cd C:/Users/brand/Development-BC && python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2:
cd C:/Users/brand/Development-BC/webapp && npm run dev
```

Open http://localhost:5173 — Validation page should load with track maps, charts, tables, and metric cards. The first load may take 30-60 seconds while the simulation runs.

- [ ] **Step 3: Commit**

```bash
git add webapp/src/pages/validation/ValidationPage.tsx
git commit -m "feat(webapp): assemble validation page with all components"
```

---

## Phase 5: 3D Visualization Page

### Task 14: Wireframe Car + Force Arrows

**Files:**
- Create: `webapp/src/pages/visualization/WireframeCar.tsx`
- Create: `webapp/src/pages/visualization/ForceArrows.tsx`

- [ ] **Step 1: Implement wireframe car**

Create `webapp/src/pages/visualization/WireframeCar.tsx`:
```tsx
import { useRef } from 'react'
import { Group } from 'three'
import { VizFrame } from '../../api/client'

// CT-16EV dimensions in meters
const WHEELBASE = 1.549
const TRACK_WIDTH = 1.2
const CHASSIS_HEIGHT = 0.3
const WHEEL_RADIUS = 0.127  // 254mm diameter / 2
const WHEEL_WIDTH = 0.2

const wheelPositions: [number, number, number][] = [
  [WHEELBASE * 0.53, WHEEL_RADIUS, TRACK_WIDTH / 2],   // FL
  [WHEELBASE * 0.53, WHEEL_RADIUS, -TRACK_WIDTH / 2],  // FR
  [-WHEELBASE * 0.47, WHEEL_RADIUS, TRACK_WIDTH / 2],  // RL
  [-WHEELBASE * 0.47, WHEEL_RADIUS, -TRACK_WIDTH / 2], // RR
]

function Wheel({ position }: { position: [number, number, number] }) {
  return (
    <mesh position={position} rotation={[Math.PI / 2, 0, 0]}>
      <cylinderGeometry args={[WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 12]} />
      <meshBasicMaterial color="#6b7280" wireframe />
    </mesh>
  )
}

interface Props {
  frame: VizFrame
}

export default function WireframeCar({ frame }: Props) {
  const groupRef = useRef<Group>(null)

  return (
    <group
      ref={groupRef}
      position={[frame.x, 0, frame.y]}
      rotation={[frame.pitch_rad, -frame.heading_rad + Math.PI / 2, frame.roll_rad]}
    >
      {/* Chassis */}
      <mesh position={[0, CHASSIS_HEIGHT / 2 + WHEEL_RADIUS, 0]}>
        <boxGeometry args={[WHEELBASE, CHASSIS_HEIGHT, TRACK_WIDTH]} />
        <meshBasicMaterial color="#d1d5db" wireframe />
      </mesh>

      {/* Wheels */}
      {wheelPositions.map((pos, i) => (
        <Wheel key={i} position={pos} />
      ))}
    </group>
  )
}
```

- [ ] **Step 2: Implement force arrows**

Create `webapp/src/pages/visualization/ForceArrows.tsx`:
```tsx
import { useMemo } from 'react'
import { Vector3, Color } from 'three'
import { VizFrame, WheelForce } from '../../api/client'

const WHEELBASE = 1.549
const TRACK_WIDTH = 1.2
const WHEEL_RADIUS = 0.127
const MAX_ARROW_LENGTH = 0.4  // meters — max visual arrow length
const FORCE_SCALE = 1 / 2000  // N to meters

const wheelOffsets: [number, number, number][] = [
  [WHEELBASE * 0.53, 0, TRACK_WIDTH / 2],
  [WHEELBASE * 0.53, 0, -TRACK_WIDTH / 2],
  [-WHEELBASE * 0.47, 0, TRACK_WIDTH / 2],
  [-WHEELBASE * 0.47, 0, -TRACK_WIDTH / 2],
]

function gripColor(util: number): string {
  if (util < 0.6) return '#22c55e'
  if (util < 0.8) return '#eab308'
  return '#ef4444'
}

function ForceArrow({ wheel, offset, heading }: { wheel: WheelForce; offset: [number, number, number]; heading: number }) {
  const { dir, length, color } = useMemo(() => {
    const fx = wheel.fx * FORCE_SCALE
    const fy = wheel.fy * FORCE_SCALE
    const raw = Math.sqrt(fx * fx + fy * fy)
    const len = Math.min(raw, MAX_ARROW_LENGTH)

    // Force direction in car-local frame: fx = forward, fy = lateral
    const dir = new Vector3(fx, 0, fy)
    if (dir.length() > 0.001) dir.normalize()

    return { dir, length: len, color: gripColor(wheel.grip_util) }
  }, [wheel])

  if (length < 0.01) return null

  return (
    <group position={offset}>
      <arrowHelper
        args={[
          dir,
          new Vector3(0, 0.02, 0),  // slight lift off ground
          length,
          new Color(color),
          length * 0.3,  // head length
          length * 0.15, // head width
        ]}
      />
    </group>
  )
}

interface Props {
  frame: VizFrame
}

export default function ForceArrows({ frame }: Props) {
  return (
    <group
      position={[frame.x, 0, frame.y]}
      rotation={[0, -frame.heading_rad + Math.PI / 2, 0]}
    >
      {frame.wheels.map((wheel, i) => (
        <ForceArrow
          key={i}
          wheel={wheel}
          offset={wheelOffsets[i]}
          heading={frame.heading_rad}
        />
      ))}
    </group>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add webapp/src/pages/visualization/WireframeCar.tsx webapp/src/pages/visualization/ForceArrows.tsx
git commit -m "feat(webapp): add wireframe car and per-wheel force arrow components"
```

---

### Task 15: Track Line + Camera Controllers

**Files:**
- Create: `webapp/src/pages/visualization/TrackLine.tsx`
- Create: `webapp/src/pages/visualization/CameraController.tsx`

- [ ] **Step 1: Implement track line with speed coloring**

Create `webapp/src/pages/visualization/TrackLine.tsx`:
```tsx
import { useMemo } from 'react'
import { BufferGeometry, Float32BufferAttribute, LineBasicMaterial, Color } from 'three'
import { VisualizationResponse } from '../../api/client'

function speedToColor(speed: number, minSpeed: number, maxSpeed: number): Color {
  const t = maxSpeed > minSpeed ? (speed - minSpeed) / (maxSpeed - minSpeed) : 0
  if (t < 0.25) return new Color('#3b82f6').lerp(new Color('#22c55e'), t / 0.25)
  if (t < 0.5) return new Color('#22c55e').lerp(new Color('#eab308'), (t - 0.25) / 0.25)
  if (t < 0.75) return new Color('#eab308').lerp(new Color('#f97316'), (t - 0.5) / 0.25)
  return new Color('#f97316').lerp(new Color('#ef4444'), (t - 0.75) / 0.25)
}

interface Props {
  data: VisualizationResponse
  showColors: boolean
}

export default function TrackLine({ data, showColors }: Props) {
  const geometry = useMemo(() => {
    const geo = new BufferGeometry()
    const positions: number[] = []
    const colors: number[] = []

    const xs = data.track_centerline_x
    const ys = data.track_centerline_y
    const speeds = data.track_speed_colors
    const minS = Math.min(...speeds.filter(s => s > 0))
    const maxS = Math.max(...speeds)

    for (let i = 0; i < xs.length; i++) {
      positions.push(xs[i], 0.01, ys[i])  // slight elevation above ground
      const c = showColors ? speedToColor(speeds[i], minS, maxS) : new Color('#4b5563')
      colors.push(c.r, c.g, c.b)
    }

    geo.setAttribute('position', new Float32BufferAttribute(positions, 3))
    geo.setAttribute('color', new Float32BufferAttribute(colors, 3))
    return geo
  }, [data, showColors])

  return (
    <line geometry={geometry}>
      <lineBasicMaterial vertexColors linewidth={2} />
    </line>
  )
}
```

- [ ] **Step 2: Implement camera controller**

Create `webapp/src/pages/visualization/CameraController.tsx`:
```tsx
import { useRef, useEffect } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Vector3 } from 'three'
import { CameraMode, usePlaybackStore } from '../../stores/playbackStore'
import { VizFrame } from '../../api/client'

interface Props {
  frame: VizFrame
}

export default function CameraController({ frame }: Props) {
  const cameraMode = usePlaybackStore(s => s.cameraMode)
  const { camera } = useThree()
  const target = useRef(new Vector3())
  const smoothPos = useRef(new Vector3())

  useFrame(() => {
    const carPos = new Vector3(frame.x, 0.3, frame.y)
    target.current.lerp(carPos, 0.1)

    if (cameraMode === 'chase') {
      // Behind and above the car
      const behind = new Vector3(
        frame.x - Math.cos(frame.heading_rad) * 4,
        2.5,
        frame.y - Math.sin(frame.heading_rad) * 4,
      )
      smoothPos.current.lerp(behind, 0.05)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    } else if (cameraMode === 'birdseye') {
      const above = new Vector3(frame.x, 15, frame.y)
      smoothPos.current.lerp(above, 0.1)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    }
    // 'orbit' mode is handled by OrbitControls — we just update the target
  })

  if (cameraMode === 'orbit') {
    return <OrbitControls target={target.current} enableDamping dampingFactor={0.1} />
  }

  return null
}
```

- [ ] **Step 3: Commit**

```bash
git add webapp/src/pages/visualization/TrackLine.tsx webapp/src/pages/visualization/CameraController.tsx
git commit -m "feat(webapp): add track line with speed coloring and 3-mode camera controller"
```

---

### Task 16: Side Panel

**Files:**
- Create: `webapp/src/pages/visualization/SidePanel.tsx`

- [ ] **Step 1: Implement side panel with telemetry, inputs, tire forces, minimap**

Create `webapp/src/pages/visualization/SidePanel.tsx`:
```tsx
import { VizFrame, VisualizationResponse } from '../../api/client'
import { formatSpeed, formatForce, formatPercent, formatTime } from '../../utils/formatters'

function gripBg(util: number): string {
  if (util < 0.6) return 'bg-green-900/50 border-green-700'
  if (util < 0.8) return 'bg-yellow-900/50 border-yellow-700'
  return 'bg-red-900/50 border-red-700'
}

function BarGauge({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="w-8 h-24 bg-gray-800 rounded relative overflow-hidden">
      <div
        className="absolute bottom-0 w-full rounded-b transition-all duration-75"
        style={{ height: `${pct}%`, backgroundColor: color }}
      />
    </div>
  )
}

interface Props {
  frame: VizFrame
  data: VisualizationResponse
}

export default function SidePanel({ frame, data }: Props) {
  const wheelLabels = ['FL', 'FR', 'RL', 'RR']

  return (
    <div className="w-64 bg-gray-900 border-l border-gray-800 flex flex-col overflow-y-auto">
      {/* Telemetry */}
      <div className="p-3 border-b border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Telemetry</h4>
        <div className="text-2xl font-bold text-green-400">{formatSpeed(frame.speed_kmh)}</div>
        <div className="grid grid-cols-2 gap-1 mt-2 text-xs">
          <div className="text-gray-400">RPM</div>
          <div className="text-right">{Math.round(frame.motor_rpm)}</div>
          <div className="text-gray-400">Torque</div>
          <div className="text-right">{frame.motor_torque_nm.toFixed(1)} Nm</div>
          <div className="text-gray-400">Voltage</div>
          <div className="text-right">{frame.pack_voltage_v.toFixed(0)} V</div>
          <div className="text-gray-400">Current</div>
          <div className="text-right">{frame.pack_current_a.toFixed(1)} A</div>
          <div className="text-gray-400">SOC</div>
          <div className="text-right">{frame.soc_pct.toFixed(1)}%</div>
          <div className="text-gray-400">Time</div>
          <div className="text-right">{formatTime(frame.time_s)}</div>
        </div>
        {/* SOC bar */}
        <div className="mt-2 h-2 bg-gray-800 rounded overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-75"
            style={{ width: `${frame.soc_pct}%` }}
          />
        </div>
      </div>

      {/* Driver Inputs */}
      <div className="p-3 border-b border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Driver Inputs</h4>
        <div className="flex items-end justify-center gap-6">
          <div className="text-center">
            <BarGauge value={frame.throttle_pct} max={100} color="#22c55e" />
            <div className="text-xs mt-1 text-gray-400">THR</div>
            <div className="text-xs font-mono">{frame.throttle_pct.toFixed(0)}%</div>
          </div>
          <div className="text-center">
            <BarGauge value={frame.brake_pct} max={100} color="#ef4444" />
            <div className="text-xs mt-1 text-gray-400">BRK</div>
            <div className="text-xs font-mono">{frame.brake_pct.toFixed(0)}%</div>
          </div>
        </div>
      </div>

      {/* Tire Forces */}
      <div className="p-3 border-b border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Tire Forces</h4>
        <div className="grid grid-cols-2 gap-2">
          {frame.wheels.map((w, i) => (
            <div key={i} className={`rounded p-2 border text-center text-xs ${gripBg(w.grip_util)}`}>
              <div className="text-gray-400 font-medium">{wheelLabels[i]}</div>
              <div className="font-mono">{formatForce(Math.sqrt(w.fx ** 2 + w.fy ** 2))}</div>
              <div className="text-gray-500">{formatPercent(w.grip_util * 100)} grip</div>
            </div>
          ))}
        </div>
      </div>

      {/* Minimap */}
      <div className="p-3 mt-auto">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Track Position</h4>
        <svg viewBox="-10 -10 120 120" className="w-full bg-gray-800/50 rounded">
          {/* Track outline */}
          <polyline
            points={data.track_centerline_x
              .map((x, i) => {
                // Normalize to 0-100 range for SVG
                const minX = Math.min(...data.track_centerline_x)
                const maxX = Math.max(...data.track_centerline_x)
                const minY = Math.min(...data.track_centerline_y)
                const maxY = Math.max(...data.track_centerline_y)
                const range = Math.max(maxX - minX, maxY - minY) || 1
                const nx = ((x - minX) / range) * 100
                const ny = ((data.track_centerline_y[i] - minY) / range) * 100
                return `${nx},${ny}`
              })
              .join(' ')}
            fill="none"
            stroke="#4b5563"
            strokeWidth="1.5"
          />
          {/* Car position dot */}
          {(() => {
            const minX = Math.min(...data.track_centerline_x)
            const maxX = Math.max(...data.track_centerline_x)
            const minY = Math.min(...data.track_centerline_y)
            const maxY = Math.max(...data.track_centerline_y)
            const range = Math.max(maxX - minX, maxY - minY) || 1
            const cx = ((frame.x - minX) / range) * 100
            const cy = ((frame.y - minY) / range) * 100
            return <circle cx={cx} cy={cy} r="3" fill="#22c55e" />
          })()}
        </svg>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add webapp/src/pages/visualization/SidePanel.tsx
git commit -m "feat(webapp): add side panel with telemetry, driver inputs, tire forces, and minimap"
```

---

### Task 17: Timeline + Playback Controls

**Files:**
- Create: `webapp/src/pages/visualization/Timeline.tsx`
- Create: `webapp/src/pages/visualization/PlaybackControls.tsx`

- [ ] **Step 1: Implement Timeline with scrubber**

Create `webapp/src/pages/visualization/Timeline.tsx`:
```tsx
import { useRef, useCallback } from 'react'
import { usePlaybackStore } from '../../stores/playbackStore'
import { VisualizationResponse } from '../../api/client'
import { formatTime } from '../../utils/formatters'

interface Props {
  data: VisualizationResponse
}

export default function Timeline({ data }: Props) {
  const { currentFrame, totalFrames, setFrame } = usePlaybackStore()
  const barRef = useRef<HTMLDivElement>(null)

  const progress = totalFrames > 0 ? currentFrame / (totalFrames - 1) : 0
  const currentTime = data.frames[currentFrame]?.time_s ?? 0

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!barRef.current) return
      const rect = barRef.current.getBoundingClientRect()
      const pct = (e.clientX - rect.left) / rect.width
      setFrame(Math.round(pct * (totalFrames - 1)))
    },
    [totalFrames, setFrame],
  )

  const handleDrag = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.buttons !== 1 || !barRef.current) return
      const rect = barRef.current.getBoundingClientRect()
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      setFrame(Math.round(pct * (totalFrames - 1)))
    },
    [totalFrames, setFrame],
  )

  return (
    <div className="flex items-center gap-3 px-4">
      <span className="text-xs text-gray-400 font-mono w-12">{formatTime(currentTime)}</span>
      <div
        ref={barRef}
        className="flex-1 h-2 bg-gray-800 rounded cursor-pointer relative"
        onClick={handleClick}
        onMouseMove={handleDrag}
      >
        <div
          className="h-full bg-green-500 rounded transition-none"
          style={{ width: `${progress * 100}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow"
          style={{ left: `${progress * 100}%`, transform: 'translate(-50%, -50%)' }}
        />
      </div>
      <span className="text-xs text-gray-400 font-mono w-12">{formatTime(data.total_time_s)}</span>
    </div>
  )
}
```

- [ ] **Step 2: Implement PlaybackControls**

Create `webapp/src/pages/visualization/PlaybackControls.tsx`:
```tsx
import { useEffect } from 'react'
import { usePlaybackStore, CameraMode } from '../../stores/playbackStore'

const speeds = [0.5, 1, 2, 5]
const cameras: { mode: CameraMode; label: string }[] = [
  { mode: 'chase', label: 'Chase' },
  { mode: 'birdseye', label: "Bird's Eye" },
  { mode: 'orbit', label: 'Orbit' },
]

export default function PlaybackControls() {
  const store = usePlaybackStore()

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === 'Space') { e.preventDefault(); store.togglePlay() }
      if (e.code === 'ArrowRight') { e.preventDefault(); store.nextFrame() }
      if (e.code === 'ArrowLeft') { e.preventDefault(); store.prevFrame() }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [store])

  return (
    <div className="flex items-center gap-4 px-4">
      {/* Play/Pause */}
      <button
        onClick={store.togglePlay}
        className="text-lg hover:text-green-400 transition-colors"
      >
        {store.isPlaying ? '⏸' : '▶'}
      </button>

      {/* Speed */}
      <div className="flex gap-1">
        {speeds.map((s) => (
          <button
            key={s}
            onClick={() => store.setSpeed(s)}
            className={`px-2 py-0.5 text-xs rounded ${
              store.speed === s ? 'bg-green-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Camera */}
      <div className="flex gap-1">
        {cameras.map(({ mode, label }) => (
          <button
            key={mode}
            onClick={() => store.setCameraMode(mode)}
            className={`px-2 py-0.5 text-xs rounded ${
              store.cameraMode === mode ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Data source toggle */}
      <div className="flex gap-1">
        {(['sim', 'real'] as const).map((src) => (
          <button
            key={src}
            onClick={() => store.setDataSource(src)}
            className={`px-2 py-0.5 text-xs rounded ${
              store.dataSource === src ? 'bg-purple-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {src === 'sim' ? 'Sim' : 'Real'}
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Overlay toggles */}
      <button
        onClick={store.toggleForces}
        className={`px-2 py-0.5 text-xs rounded ${
          store.showForces ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-500'
        }`}
      >
        Forces
      </button>
      <button
        onClick={store.toggleTrackColor}
        className={`px-2 py-0.5 text-xs rounded ${
          store.showTrackColor ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-500'
        }`}
      >
        Track Color
      </button>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add webapp/src/pages/visualization/Timeline.tsx webapp/src/pages/visualization/PlaybackControls.tsx
git commit -m "feat(webapp): add timeline scrubber and playback controls with keyboard shortcuts"
```

---

### Task 18: 3D Viewport + Page Assembly

**Files:**
- Create: `webapp/src/pages/visualization/Viewport.tsx`
- Modify: `webapp/src/pages/visualization/VisualizationPage.tsx`

- [ ] **Step 1: Implement Viewport (R3F Canvas)**

Create `webapp/src/pages/visualization/Viewport.tsx`:
```tsx
import { useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { VisualizationResponse } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import WireframeCar from './WireframeCar'
import ForceArrows from './ForceArrows'
import TrackLine from './TrackLine'
import CameraController from './CameraController'

function PlaybackLoop({ data }: { data: VisualizationResponse }) {
  const { isPlaying, speed, currentFrame, totalFrames, setFrame, showForces, showTrackColor } = usePlaybackStore()
  const accumulator = useRef(0)

  const frame = data.frames[currentFrame] ?? data.frames[0]

  // Calculate time step between frames
  const dt = currentFrame < totalFrames - 1
    ? data.frames[currentFrame + 1].time_s - frame.time_s
    : 0.05  // fallback

  useFrame((_, delta) => {
    if (!isPlaying) return
    accumulator.current += delta * speed
    if (accumulator.current >= dt && dt > 0) {
      const steps = Math.floor(accumulator.current / dt)
      accumulator.current -= steps * dt
      const next = Math.min(currentFrame + steps, totalFrames - 1)
      setFrame(next)
      if (next >= totalFrames - 1) {
        usePlaybackStore.getState().pause()
      }
    }
  })

  return (
    <>
      <WireframeCar frame={frame} />
      {showForces && <ForceArrows frame={frame} />}
      <TrackLine data={data} showColors={showTrackColor} />
      <CameraController frame={frame} />
    </>
  )
}

interface Props {
  data: VisualizationResponse
}

export default function Viewport({ data }: Props) {
  useEffect(() => {
    usePlaybackStore.getState().setTotalFrames(data.frames.length)
  }, [data])

  return (
    <Canvas camera={{ position: [0, 10, 10], fov: 50 }} className="bg-gray-950">
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={0.4} />

      {/* Ground grid */}
      <gridHelper args={[200, 100, '#1f2937', '#111827']} />

      {/* Scene */}
      <PlaybackLoop data={data} />
    </Canvas>
  )
}
```

- [ ] **Step 2: Assemble Visualization page**

Replace `webapp/src/pages/visualization/VisualizationPage.tsx`:
```tsx
import { useVisualization } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import LoadingSpinner from '../../components/LoadingSpinner'
import Viewport from './Viewport'
import SidePanel from './SidePanel'
import Timeline from './Timeline'
import PlaybackControls from './PlaybackControls'

export default function VisualizationPage() {
  const dataSource = usePlaybackStore(s => s.dataSource)
  const currentFrame = usePlaybackStore(s => s.currentFrame)
  const { data, isLoading, error } = useVisualization(dataSource)

  if (isLoading) return <LoadingSpinner message="Computing visualization data..." />
  if (error || !data) return <p className="text-red-400">Failed to load visualization data.</p>

  const frame = data.frames[currentFrame] ?? data.frames[0]

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)]">
      {/* Main area: viewport + side panel */}
      <div className="flex flex-1 min-h-0">
        <div className="flex-1">
          <Viewport data={data} />
        </div>
        <SidePanel frame={frame} data={data} />
      </div>

      {/* Bottom strip */}
      <div className="shrink-0 bg-gray-900 border-t border-gray-800 py-2 space-y-2">
        <Timeline data={data} />
        <PlaybackControls />
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Verify in browser**

Start both servers and navigate to http://localhost:5173/visualization. Should see:
- 3D viewport with wireframe car on a track line
- Side panel with live telemetry readouts
- Bottom strip with timeline and controls
- Play button starts animation, spacebar toggles
- Camera mode buttons switch between chase/birdseye/orbit
- Sim/Real toggle reloads data

- [ ] **Step 4: Commit**

```bash
git add webapp/src/pages/visualization/
git commit -m "feat(webapp): assemble 3D visualization page with viewport, panels, and controls"
```

---

## Phase 6: Integration + Polish

### Task 19: Wrap Main in SWRConfig + Final Wiring

**Files:**
- Modify: `webapp/src/main.tsx`

- [ ] **Step 1: Add SWRConfig to main.tsx**

Replace `webapp/src/main.tsx`:
```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SWRConfig } from 'swr'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SWRConfig
      value={{
        revalidateOnFocus: false,
        dedupingInterval: 60000,
      }}
    >
      <App />
    </SWRConfig>
  </StrictMode>,
)
```

- [ ] **Step 2: Verify full flow end to end**

1. Start backend: `python -m uvicorn backend.main:app --reload --port 8000`
2. Start frontend: `cd webapp && npm run dev`
3. Open http://localhost:5173
4. Validation page: select a lap, verify track maps render, charts load, tables populate
5. Visualization page: verify 3D car appears, press play, watch it drive the track, switch camera modes, toggle Sim/Real

- [ ] **Step 3: Commit**

```bash
git add webapp/src/main.tsx
git commit -m "feat(webapp): add SWRConfig and finalize app wiring"
```

---

### Task 20: Add .gitignore Entries + Cache Directory

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add webapp and cache entries to .gitignore**

Append to `.gitignore`:
```
# Webapp
webapp/node_modules/
webapp/dist/

# Superpowers brainstorm sessions
.superpowers/

# Backend cache
backend/__pycache__/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add webapp and cache entries to .gitignore"
```
