# Webapp Rework: Validation + 3D Visualization

**Date:** 2026-04-15
**Status:** Approved
**Replaces:** Existing Dash dashboard (dashboard/)

## Overview

Replace the existing Python Dash dashboard with a React + Three.js web application focused on two pages: a data-rich **Validation** page comparing simulation vs real telemetry, and a **3D Visualization** page showing an animated wireframe car completing one lap with real-time force/telemetry overlays.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frontend framework | React (Vite) | Unified codebase for charts + 3D, future Phase 3 extensibility |
| 3D engine | React Three Fiber (Three.js) | Mature WebGL, good React integration, handles wireframes + arrows + animation |
| Chart library | Plotly.js (react-plotly.js) | Rich interactivity (zoom/pan/hover), good for data-dense overlays, team familiarity from Dash |
| State management | Zustand | Lightweight, sufficient for playback state + shared UI state |
| Data fetching | SWR | Caching, revalidation, loading states for API data |
| Backend | FastAPI | Python-native, async, direct access to fsae_sim modules |
| Data pipeline | Hybrid (compute on first request, cache as JSON) | Avoids static file generation, supports future parameterized re-runs |
| Theme | Dark | Consistent with racing/sim aesthetic, easier on eyes for data-dense pages |
| Navigation | Sidebar | Two pages for now, expandable later |
| Pages in scope | Validation, Visualization | Overview and Sweeps deferred to later |

## Architecture

### System Components

```
FastAPI Backend (:8000)              React Frontend (Vite :5173)
┌─────────────────────────┐          ┌──────────────────────────┐
│ Simulation Engine        │          │ React Router + Sidebar   │
│  fsae_sim.sim.engine    │          │                          │
│                         │  JSON    │ Validation Page          │
│ Telemetry Loader        │ ──────>  │  react-plotly.js charts  │
│  AiM CSV → DataFrames   │          │  Track maps + overlays   │
│                         │          │                          │
│ Track Builder           │          │ Visualization Page       │
│  GPS → XY coordinates   │          │  React Three Fiber       │
│                         │          │  Wireframe car + forces  │
│ Data Exporter           │          │                          │
│  SimResult → JSON       │          │ Zustand Store            │
│                         │          │ SWR API Client           │
│ Cache Layer             │          └──────────────────────────┘
│  LRU + .json files      │
└─────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Response | Description |
|---|---|---|---|
| `/api/laps` | GET | `{laps: [{lap_number, time_s, energy_kwh, gps_quality_score}]}` | Available laps with metadata and GPS quality ranking |
| `/api/track` | GET | `{centerline: [{x, y, distance_m}], sectors: [{name, type, start_m, end_m}], curvature: [...]}` | Track XY coordinates from GPS, sector definitions |
| `/api/validation/{lap}` | GET | `{sim: {distance, speed, throttle, brake, power, soc, lat_accel}, real: {same}, metrics: {...}, sectors: [...]}` | Paired sim/real data at each distance point for overlay charts, plus computed accuracy metrics |
| `/api/visualization/{lap}` | GET | `{frames: [{time, x, y, heading, speed, throttle, brake, motor_rpm, motor_torque, soc, pack_voltage, pack_current, roll, pitch, wheels: [{fx, fy, fz, grip_util}]}], track: {centerline, speed_colors}}` | Per-frame 3D animation data with per-wheel forces |

### Data Pipeline (Backend)

The backend needs to produce data that the sim engine does not currently output directly. This is handled in the **Data Exporter** layer, not by modifying the core engine:

1. **XY Position + Heading**: Load raw GPS lat/lon from AiM CSV, convert to local XY (meters) via equirectangular projection. Build a track spline. For each sim timestep, interpolate XY position and heading from `distance_m` along the spline.

2. **Per-Wheel Forces**: The sim records aggregate `drive_force_n`, `regen_force_n`, `resistance_force_n`. The exporter calls the existing `VehicleDynamics` tire model and load transfer calculations to decompose into per-wheel Fx, Fy, Fz at each timestep using the recorded speed, curvature, throttle, and brake values.

3. **Roll and Pitch**: Derived from lateral and longitudinal load transfer. Lateral load transfer → roll angle (small angle approximation with estimated roll stiffness). Longitudinal load transfer → pitch angle.

4. **Sector Definitions**: Curvature-based segmentation — consecutive segments with `|curvature| > threshold` form a corner sector, gaps between are straight sectors. Named sequentially (Turn 1, Straight 1, etc.).

5. **GPS Quality Scoring**: For lap selection, score each lap by mean `GPS PosAccuracy` (lower = better, 200 = invalid). Filter out laps with >10% invalid samples.

6. **Telemetry Alignment**: Align sim and real data on a shared distance axis by interpolating both to uniform 1m spacing. This ensures overlay charts line up correctly.

## Validation Page

### Layout (top to bottom, full page scroll)

**Lap Selector** (sticky top bar)
- Dropdown: individual laps (1-21) or "All Laps" for aggregate
- Shows lap time and GPS quality badge for each option
- Default: best GPS quality lap

**Section 1: Track Maps**
- Two Plotly scatter plots side-by-side, equal size
- Left: Sim speed colored on track shape. Right: Real telemetry speed colored on track shape.
- Track shape from GPS XY coordinates (local meters projection)
- Color scale: blue (0 km/h) → green → yellow → red (peak speed), shared between both maps
- Shared color bar legend between the two
- Hover tooltip: distance (m), speed (km/h), segment index
- Equal aspect ratio so the track shape is not distorted

**Section 2: Overlay Charts**
Six Plotly charts stacked vertically, all sharing a synchronized distance x-axis:

1. **Speed vs Distance** — sim (green #22c55e) overlaid on real (blue #3b82f6)
2. **Throttle vs Distance** — sim throttle_pct (0-1 → 0-100%) vs real Throttle Pos (0-100%)
3. **Brake vs Distance** — sim brake_pct vs real FBrakePressure (normalized to 0-100%)
4. **Electrical Power vs Distance** — sim electrical_power_w vs real (Pack Voltage × Pack Current)
5. **SOC vs Distance** — sim soc_pct vs real State of Charge
6. **Lateral Acceleration vs Distance** — sim (speed² × curvature / 9.81, in g) vs real GPS LatAcc

All charts:
- Consistent colors: sim = green, real = blue
- Plotly zoom/pan/hover enabled
- Crosshair sync across all six charts (Plotly subplots with shared x-axis)
- Reset zoom button
- Legend toggle to show/hide individual traces

**Section 3: Sector Breakdown Table**
- Columns: Sector, Type (straight/corner), Sim Time (s), Real Time (s), Delta (s), Delta %, Sim Avg Speed (km/h), Real Avg Speed (km/h), Speed Delta %
- Rows: one per sector (estimated 15-25 sectors per lap)
- Color-coded delta cells: green = sim within 5%, yellow = 5-10%, red = >10%
- Sortable columns

**Section 4: Per-Lap Summary Table**
- Columns: Lap #, Sim Time, Real Time, Time Error %, Sim Energy (kWh), Real Energy (kWh), Energy Error %, Mean Speed Error %
- 21 rows (one per lap) + aggregate row at bottom
- Currently selected lap highlighted
- Only shown when "All Laps" is selected in the lap selector

**Section 5: Accuracy Metrics**
- 8 pass/fail badge cards in a responsive grid (4 across on desktop, 2 on mobile)
- Each card: metric name, value, threshold, pass/fail badge (green/red)
- Metrics: Total Time Error (<5%), Total Energy Error (<5%), Mean Speed Error (<10%), Peak Speed Error (<15%), SOC Tracking Error (<3% absolute), Lap Time Consistency (σ <2s), Energy Per Lap Consistency (σ <0.05 kWh), Corner Speed Accuracy (<10%)

## Visualization Page

### Layout

```
┌────────────────────────────────────┬──────────────┐
│                                    │  Telemetry   │
│                                    │  Speed       │
│         3D VIEWPORT                │  RPM         │
│         (React Three Fiber)        │  Torque      │
│                                    │  Voltage     │
│         Wireframe car              │  SOC         │
│         Force arrows               │              │
│         Track line                 │  Driver      │
│                                    │  ┌───┬───┐   │
│                                    │  │THR│BRK│   │
│                                    │  └───┘───┘   │
│                                    │              │
│                                    │  Tire Forces │
│                                    │  ┌──┬──┐    │
│                                    │  │FL│FR│    │
│                                    │  ├──┼──┤    │
│                                    │  │RL│RR│    │
│                                    │  └──┴──┘    │
│                                    │              │
│                                    │  [Minimap]   │
├────────────────────────────────────┴──────────────┤
│ ▶ ■  0.5x 1x 2x 5x  |===●=============| 0:21/1:02│
│ [Chase] [Bird] [Orbit]  [Sim|Real]  [Forces ✓]   │
└───────────────────────────────────────────────────┘
```

### 3D Viewport

**Wireframe Car**
- Box chassis: 1549mm (L) × 1200mm (W) × 300mm (H), rendered as wireframe edges only (no solid faces), white/light gray color
- 4 cylindrical wheels at corner positions: 254mm diameter (10" Hoosier), ~200mm wide, wireframe, slightly darker color
- Car positioned on the track spline, oriented by heading angle
- Roll applied from lateral load transfer, pitch from longitudinal load transfer (subtle, small angles)

**Force Arrows (per wheel)**
- 3D arrow (cone + cylinder) at each wheel contact patch
- Direction: combined Fx (longitudinal) + Fy (lateral) force vector
- Length: proportional to force magnitude, scaled so max force ≈ 1.5× wheel diameter
- Color by grip utilization (|force| / max_available_grip): green (<60%), yellow (60-80%), red (>80%)
- Toggleable on/off

**Track**
- Centerline rendered as a continuous line on the ground plane (y=0)
- Line colored by speed (same gradient as validation page)
- Full lap visible at all times
- Subtle ground grid for spatial reference

**Camera Modes**
- **Chase cam**: Fixed offset behind and above the car (approx -3m back, +1.5m up), rotates with car heading, smooth follow with slight damping
- **Bird's eye**: Camera 15m above car, looking straight down, follows car position, no rotation
- **Free orbit**: OrbitControls centered on car position, user drags to rotate/zoom, follows car but lets user adjust angle freely

**Lighting**
- Ambient light (soft, global illumination)
- Single directional light (sun-like, subtle shadows)
- Wireframe materials don't need complex lighting — keep it clean

### Side Panel

**Telemetry Readouts**
- Speed: large text (e.g., "72.4 km/h"), updates every frame
- Motor RPM: text with small bar
- Motor Torque: text (Nm)
- Pack Voltage / Current: text
- SOC: percentage + horizontal bar indicator (green→yellow→red as it depletes)
- Elapsed time / total time

**Driver Inputs**
- Throttle: vertical bar, 0-100%, green fill from bottom
- Brake: vertical bar, 0-100%, red fill from bottom
- Side by side, labeled

**Tire Forces (2×2 grid)**
- FL, FR, RL, RR matching car layout
- Each cell: force magnitude (N), grip utilization %, background color matching the 3D arrow color
- Small directional arrow icon

**Track Minimap**
- Top-down outline of full track
- Green dot = current position
- Colored trail for completed portion

### Bottom Strip

**Timeline**
- Full-width scrubber bar: 0m → ~1005m (one lap)
- Draggable playhead
- Time labels at current position and total
- Thin speed sparkline trace along the bar (so you can see upcoming fast/slow sections)

**Playback Controls**
- Play/Pause button (spacebar shortcut)
- Speed buttons: 0.5x, 1x, 2x, 5x (highlighted active)
- Skip backward/forward by sector (|◀ ▶|)
- Frame step: left/right arrow keys

**Mode Toggles**
- Camera: Chase | Bird's Eye | Free Orbit (radio buttons)
- Data source: Sim | Real (toggle, default Sim)
- Overlay toggles: Force Arrows, Track Coloring (checkboxes)

### Lap Selection

- Default: the lap with the best GPS quality score (lowest mean GPS PosAccuracy, excluding samples with accuracy=200)
- Sim runs that single lap using the CalibratedStrategy driver model
- When toggled to "Real", the car follows the actual GPS track with real telemetry values driving the HUD and force computations

## Project Structure

```
webapp/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── index.html
├── public/
├── src/
│   ├── main.tsx                    # App entry point
│   ├── App.tsx                     # Router + layout (sidebar + content)
│   ├── api/
│   │   └── client.ts              # API client functions + SWR hooks
│   ├── stores/
│   │   ├── playbackStore.ts       # Zustand: play/pause, speed, currentFrame, camera mode
│   │   └── validationStore.ts     # Zustand: selected lap, chart sync state
│   ├── components/
│   │   ├── Sidebar.tsx            # Navigation sidebar
│   │   └── LoadingSpinner.tsx     # Shared loading state
│   ├── pages/
│   │   ├── validation/
│   │   │   ├── ValidationPage.tsx # Page container + lap selector
│   │   │   ├── TrackMaps.tsx      # Side-by-side Plotly track maps
│   │   │   ├── OverlayCharts.tsx  # 6 synced Plotly overlay charts
│   │   │   ├── SectorTable.tsx    # Sector breakdown table
│   │   │   ├── LapTable.tsx       # Per-lap summary table
│   │   │   └── MetricCards.tsx    # Pass/fail accuracy badges
│   │   └── visualization/
│   │       ├── VisualizationPage.tsx  # Page container + data loading
│   │       ├── Viewport.tsx           # R3F Canvas + scene setup
│   │       ├── WireframeCar.tsx       # Car chassis + wheels mesh
│   │       ├── ForceArrows.tsx        # Per-wheel force arrow components
│   │       ├── TrackLine.tsx          # Track centerline with speed coloring
│   │       ├── CameraController.tsx   # Chase/bird/orbit camera logic
│   │       ├── SidePanel.tsx          # Telemetry + inputs + tire forces + minimap
│   │       ├── Timeline.tsx           # Bottom scrubber + sparkline
│   │       └── PlaybackControls.tsx   # Play/pause/speed/skip controls
│   ├── theme/
│   │   └── darkTheme.ts          # Color tokens, spacing, typography
│   └── utils/
│       └── formatters.ts          # Unit formatting (km/h, Nm, kWh, etc.)
│
backend/
├── __init__.py
├── main.py                        # FastAPI app, CORS, mount routes
├── routers/
│   ├── laps.py                    # GET /api/laps
│   ├── track.py                   # GET /api/track
│   ├── validation.py              # GET /api/validation/{lap}
│   └── visualization.py           # GET /api/visualization/{lap}
├── services/
│   ├── sim_runner.py              # Run simulation, return SimResult
│   ├── telemetry_service.py       # Load + clean AiM telemetry
│   ├── track_service.py           # GPS → XY spline, sector detection
│   ├── validation_export.py       # Align sim+real on shared distance axis, compute metrics
│   ├── visualization_export.py    # Compute per-frame 3D data (XY, heading, per-wheel forces, roll, pitch)
│   └── cache.py                   # File-based JSON cache
└── models/
    ├── validation.py              # Pydantic response models for validation endpoint
    └── visualization.py           # Pydantic response models for visualization endpoint
```

## Key Dependencies

### Frontend (package.json)
- `react`, `react-dom` (^18)
- `react-router-dom` (^6)
- `@react-three/fiber`, `@react-three/drei` (R3F + helpers)
- `three` (Three.js)
- `react-plotly.js`, `plotly.js-dist-min` (charts)
- `zustand` (state)
- `swr` (data fetching)
- `tailwindcss` (utility-first CSS, dark mode built-in)

### Backend (pyproject.toml additions)
- `fastapi` (already in project deps)
- `uvicorn` (ASGI server)
- Existing `fsae_sim` package (simulation engine, vehicle dynamics, track, telemetry)

## What Is NOT In Scope

- Overview page (deferred)
- Sweeps page (deferred, awaiting Phase 3)
- Multi-lap 3D visualization (single lap only)
- Side-by-side 3D lap comparison (future enhancement)
- Track surface rendering / curbing / elevation (wireframe ground plane only)
- Mobile responsive layout (desktop-first)
- Authentication or multi-user support
- Deployment / Docker configuration (dev server only for now)
