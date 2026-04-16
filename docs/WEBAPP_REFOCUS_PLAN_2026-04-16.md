# Webapp Refocus Plan — 2026-04-16

## 1. Mission (revised)

**Build the most accurate FSAE EV simulator possible for CT-16EV.**

The webapp exists to support that single mission. There are **three pages**, and nothing else belongs in this repo's dashboard:

1. **Verification** — how close is the baseline simulator to reality?
2. **Visualization** — a 3D playback of the car driving the track.
3. **Simulate** — a what-if tool with three knobs (max RPM, max torque, SOC discharge map).

### What is out of scope (moved to a separate repo later)

- Sweep runners (thousands of sims, Pareto fronts, optimization).
- Overview / home page showing aggregate scoring.
- Driver-strategy sweep comparison UI.
- Ghost-car multi-run overlays.
- Sweep-result tables, CSV exports of sweep data.

Anything that treats "compare thousands of runs" as a first-class goal belongs in the next repo. This repo gets simpler: one baseline sim, one tuned sim, one visualization.

---

## 2. The three pages — spec

### Page 1: **Verification** (was "Validation")

**Purpose.** Convince a reader that the baseline sim matches reality well enough to trust. If the sim is wrong, show exactly where.

**Inputs (none — fixed).**
- Baseline config: `configs/ct16ev.yaml`.
- Reference telemetry: `CleanedEndurance.csv` (Michigan 2025).

**What it shows.**
- **Per-lap + all-laps views** (via lap selector, unchanged).
- **Metric cards** — time, energy, min SOC, mean speed (actually computed at the selected granularity, not globally — today this is a bug).
- **Goodness-of-fit per channel** — RMS, R², Pearson correlation for speed, power, SOC, throttle, brake, RPM, torque. These are missing today.
- **Overlay charts with residual band** — sim vs real on top, (sim − real) band underneath with horizontal RMS reference. Today only the top trace exists.
- **Track heatmap** — sim vs real speed along the centerline (exists). Add: speed-delta heatmap in a third panel.
- **Sector table** — per-sector time, average speed, energy, min SOC for sim and real. Today: only time.
- **Lap-to-lap distribution** — box/violin of per-channel error % across the 21 laps. Highlights which laps are outliers (GPS drift, driver change).
- **Channel coverage** — RPM, torque feedback, pack current, pack voltage, pack temp, lat-accel, long-accel overlays. Today only speed/throttle/brake/power/SOC/lat-accel exist.
- **Energy budget reconciliation** — integrated electrical energy split into drive, regen, aero, tire, brake, motor loss, inverter loss. Side-by-side sim vs real. This is the most important view for trusting efficiency predictions.
- **Target-line annotations** — 2% energy, 5% time pass bars always visible.
- **Filters** — "exclude driver-change lap," "exclude low-GPS-quality laps."

**What it does NOT show.** No alternate-tune comparison. That is the Simulate page's job.

### Page 2: **Visualization**

**Purpose.** See the car move around the track. Spot physics bugs (wheel floating, forces pointing wrong way, car sliding when it shouldn't). Make the sim's behavior intuitive.

**Inputs.**
- Data source toggle: `sim` (baseline) vs `real` (telemetry-reconstructed).
- Lap selector (currently best-GPS lap only).

**What it shows (today).**
- 3D wireframe car on a track, driven by per-frame position/heading.
- Four per-wheel force arrows, color-coded by grip utilization.
- Camera modes: chase, overhead, orbit.
- Side panel: speed, RPM, torque, voltage, current, SOC, throttle/brake.
- Timeline scrubber, play/pause, speed 0.5–5×.

**What it should show (ranked by value).**
- **Trajectory trail** — polyline of where the car has been this lap.
- **Scrubbable time-series strip** under the 3D view — speed / pack power / SOC with a vertical cursor synced to playback. This is the single highest-value addition.
- **Friction-circle per wheel** in the side panel — live `Fx/Fz` vs `Fy/Fz` scatter. Shows whether the tire is saturated. Makes combined-slip bugs visible immediately.
- **Sector + lap markers** on the timeline (data is already there in `TrackData.sectors`).
- **Per-wheel slip angle** readout (input to Pacejka — currently hidden).
- **Event markers** — shift, coast-start, regen-on/off, brake-threshold crossed.
- **Driver-input burn-in** — steering wheel icon + throttle/brake bars in a canvas corner so screenshots are self-describing.
- **Keyboard shortcuts** beyond space/arrows — `,`/`.` for frame step, `[`/`]` for lap jump, `M` to bookmark.

**Not doing:** ghost-car overlays, two-run comparisons — those belong in the sweep repo.

### Page 3: **Simulate** (new)

**Purpose.** One-shot what-if. Change three things, run the sim, see whether endurance gets faster or more efficient. Nothing more.

**Inputs (exactly three).**
1. **Max motor RPM** — slider or numeric. Default = current LVCU setting (2900 RPM from Endurance Tune).
2. **Max motor torque (Nm)** — slider or numeric. Default = inverter limit (85 Nm).
3. **SOC discharge map** — a small table or draggable curve mapping `SOC %` → `max pack current (A)`. Default = Endurance Tune table (100 A above 85%, 1 A per 1% SOC below that).

No other parameters. No driver strategy knobs. No tire overrides. No aero tweaks. If a fourth knob feels necessary, it belongs in the sweep repo, not here.

**What it shows after running.**
- **Summary card row** — total endurance time (s), total energy (kWh), min SOC (%), pass/fail on completing 22 laps at cell-voltage-limit.
- **Per-lap table** — lap time, energy, min SOC per lap. Highlight laps that would have been cut short by BMS derating.
- **Delta vs baseline** — for each summary number, show `(sim with new params) − (baseline)` with sign and %.
- **Power / SOC / current time series** overlaid with baseline — shows visually when the tune's limit kicks in.
- **Warning banner** if the tune produces an infeasible result (e.g., SOC < 10% before lap 22, or BMS hit zero current).

**What it does NOT show.** No sweep. No parameter search. No Pareto. Just one alternative tune vs the baseline. The user picks values, clicks run, reads the answer.

---

## 3. Fix list — from 3-agent review

Organized by priority. Each item has a concrete fix + file:line reference.

### CRITICAL — blocks "trust the simulator" claim

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| C1 | `backend/services/validation_export.py:144-160` | `get_validation_data(lap=N)` calls `validate_full_endurance` on the ENTIRE run regardless of lap — per-lap MetricCards are actually full-endurance metrics, identical across laps. | Compute metrics over `sim_lap` / `real_lap` slices. |
| C2 | `backend/services/validation_export.py:104, 138, 202` | Uses `GPS Speed` everywhere; CLAUDE.md says `LFspeed` is the cleaner channel. | Swap all GPS-speed references to `LFspeed`. |
| C3 | `backend/services/validation_export.py:197-198` vs `validation.py:267` | Per-lap energy integral uses signed `V·I·dt`; full-endurance uses positive-only. Numbers in the same UI mean different things. | Unify on positive-only for "energy consumed" and report regen separately. |
| C4 | `backend/services/validation_export.py:56-57` | Sector filter `distance_m % distance_m.max()` drops the last sample of every lap. | Use `distance_m - distance_m.iloc[0]`. |
| C5 | `webapp/src/pages/visualization/ForceArrows.tsx:75-95` | Arrow group is yaw-rotated AND raw body-frame `fx`/`fy` are re-interpreted as world coords → double rotation → arrows point in wrong directions. | Pick one frame and stick to it. If parent is world-oriented, use `(fx, 0, fy)` rotated by heading. If parent is body-oriented, keep arrows local and don't re-rotate. Snapshot-test with synthetic `heading=0, Fx=+1000` on rear wheels. |
| C6 | `webapp/src/pages/visualization/WireframeCar.tsx:44-48` | Euler order default is XYZ — roll applied in world X before yaw. Car tilts wrong way when heading ≠ 0. | Set `rotation.order = 'YXZ'`; assign yaw first, then pitch on body-X, then roll on body-Z. |
| C7 | `webapp/src/pages/visualization/CameraController.tsx:50` | `target={target.current}` captured by value at mount → orbit camera does not follow car. | Use a `ref` on `OrbitControls` and `controls.target.copy(carPos); controls.update()` in `useFrame`. |
| C8 | `backend/services/visualization_export.py:38-49` | `_compute_lateral_forces` never uses `sign(curvature)` → Fy direction doesn't flip between left/right turns. | Multiply Fy by `sign(curvature)`. Regression-test against a constant-radius circle. |

### HIGH — makes the UI work beyond toy-scale

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| H1 | `webapp/src/pages/visualization/TrackLine.tsx:27-29`, `validation/TrackMaps.tsx:22-28`, anywhere using `Math.min(...arr)` / `Math.max(...arr)` | Spread operator throws `RangeError` on arrays >~100k items. 22 km endurance at 10 Hz is 180k frames. | Use `arr.reduce((m,v)=>v<m?v:m, Infinity)`. |
| H2 | `webapp/src/pages/visualization/ForceArrows.tsx:9-10, 86` | `FORCE_SCALE = 1/2000` + `MAX_ARROW_LENGTH = 0.4 m` clips every force > 800 N. Peak cornering is 1500-2500 N. The physics bug the user wants to SEE is being hidden by the scale cap. | Scale adaptively from max observed magnitude across the whole lap, or expose a slider. |
| H3 | `webapp/src/pages/visualization/TrackLine.tsx:19-41` | `THREE.Line` / `BufferGeometry` / `LineBasicMaterial` rebuilt on every dep change, previous never disposed → memory leak over long playback. | Return cleanup via `useEffect` that calls `.dispose()` on the old geometry+material. |
| H4 | `webapp/src/pages/visualization/Timeline.tsx:28` | Scrubbing loses mouse capture when cursor leaves the bar mid-drag. | Attach `mousemove`/`mouseup` to `window` while dragging; release on `mouseup`. |
| H5 | `webapp/src/pages/visualization/Viewport.tsx:48-54` | UI sync happens every 2 ticks regardless of how many frames advanced; at 5× speed on long data, side panel shows stale values. | Increment `ticksSinceSync` inside the while loop (already); also cap `accumulator` to one frame-step per render. |
| H6 | `webapp/src/pages/validation/ValidationPage.tsx:16-29` + `client.ts:70-74` | First-load race: `selectedLap=null`, then `useEffect` sets it. During the gap, both `validationLoading` is false and `validation` is undefined → "Failed to load data." flashes. | Add an `initializing` state or gate on `selectedLap !== null && validation`. |
| H7 | `webapp/src/api/client.ts:5-9` | `fetcher` throws `new Error('API error: <status>')` — backend error bodies are dropped. No `AbortController`. | Typed `ApiError` with `{ status, detail }`; parse response body; support `AbortSignal`. |
| H8 | `webapp/src/api/client.ts:12-16` | `rerunSimulation` calls `mutate(() => true, ...)` → revalidates *every* SWR key at once. One click triggers concurrent multi-second sim runs for both pages. | Invalidate only keys for the current route, or add a global "rerunning" lock. |
| H9 | `webapp/package.json:37` | `typescript` pin is `~6.0.2`. If the build relies on TS 5 semantics, check it still builds. | Run `npm run build` and verify; pin explicitly once confirmed. |
| H10 | `webapp/src/api/client.ts:3` | `API_BASE = '/api'` hardcoded — only works through dev proxy. | `const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'`. Add `.env.example`. |

### MEDIUM — nice-to-have, doesn't block simulator-accuracy work

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| M1 | `webapp/src/pages/visualization/SidePanel.tsx:30-38` vs `TrackLine.tsx:31-32` | Minimap flips Y (`100 - ny`) but 3D view does not → they're mirrored. | Pick one Y convention and use it everywhere. |
| M2 | `webapp/src/App.css` | 185 lines of unused Vite template scaffolding. | Delete. Not imported anywhere. |
| M3 | No global toast / error surface. | Failures silently show "Failed to load data." Button clicks that fail give no feedback. | Add `sonner` or `react-hot-toast`; wire into `fetcher` + `rerunSimulation`. |
| M4 | No `<label htmlFor>` on `Sidebar.tsx` nav items and lap selector. | Accessibility gap. | Add proper labels, `aria-busy`, `aria-current`. |
| M5 | No `Suspense` boundary at router level. | Lazy imports only work because each page wraps its own. | Add top-level `Suspense` around `<Routes>` with `LoadingSpinner`. |
| M6 | `webapp/src/stores/playbackStore.ts:69` | `setDataSource` has side effects (resets `currentFrame`, `isPlaying`). | Move reset logic into the component that calls it. |
| M7 | No URL state — selected lap lives in Zustand, not the URL. | Can't share a link to a specific lap. | Use `useSearchParams` for lap, source, camera. |
| M8 | `backend/main.py` — no logging config, no request-id, no structured errors. | Observability gap. | Add `logging.basicConfig` + `X-Request-ID` middleware. |

### LOW — cosmetic / future cleanup

- `index.html` title is `"webapp"`. Change to `"FSAE Sim - CT-16EV"`.
- `docker/Dockerfile` still runs the old Dash dashboard. Rewrite to build Vite + serve FastAPI. README still documents the Dash dashboard (`README.md:43-53`) — out of date.
- `delta_pct > 0 ? '+' : ''` on `SectorTable.tsx:38,43` reads ambiguously for near-zero values.
- `gps_quality_score` name implies higher=better but the sort treats it as lower=better (correct for its actual meaning). Rename in API to `gps_heading_std` or similar.
- No favicon, no OpenGraph meta.
- `webapp/dist/` exists in the working tree — stale artifact from Apr 15.

---

## 4. Structural decisions made now

These are being implemented as part of this refocus (non-breaking scaffolding), to "get it setup to be fixed":

1. **Rename Validation → Verification** in the webapp UI and route. `/` stays Verification (the landing page). Backend endpoint paths stay `/api/validation/...` for now — rename later in a dedicated cleanup to avoid touching the backend during UI reorganization.
2. **Add `/simulate` route** with a stub page showing the three-knob form and placeholder result cards. No backend endpoint yet; UI wired, "Run" button disabled with a "Not yet implemented" notice.
3. **Sidebar reordered** to: Verification, Visualization, Simulate.
4. **Update `CLAUDE.md`** to drop sweep-optimization language and Phase 3 references. New roadmap: (1) baseline sim DONE, (2) verification polish + physics fixes IN PROGRESS, (3) Simulate page implementation NEXT, (4) coaching output moved out of this repo.
5. **Write this report** at `docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md`.

### Decisions deferred

- **Retire `dashboard/` (Dash)**: not deleting in this pass to avoid breaking Docker. Flagged in report. Dockerfile rewrite is a separate task.
- **Rename backend endpoints** `/api/validation` → `/api/verification`: deferred. Contract change would need coordinated frontend + tests.
- **Fix list**: none of the CRITICAL / HIGH / MEDIUM items above are being fixed in this pass. This plan exists so they can be fixed next.

---

## 5. Implementation order (for follow-up sessions)

1. **Fix C1–C4** (validation backend) — hours. Unblocks the Verification page actually being correct.
2. **Fix C5–C8** (visualization physics/rendering) — 1 day. The 3D view currently misleads; nothing else on Visualization matters until these are right.
3. **Fix H1, H3, H7, H8, H10** (build / array crashes / API client) — hours.
4. **Add residuals + RMS/R²/correlation** to Verification — 1 day. Highest-leverage addition for "is the sim accurate."
5. **Add energy-budget reconciliation view** to Verification — 1 day. Most important view for efficiency trust.
6. **Add trajectory trail + scrubbable time-series strip + friction-circle** to Visualization — 1–2 days.
7. **Implement Simulate page** — backend endpoint accepting `{max_rpm, max_torque_nm, soc_discharge_map}`, running one sim with those overrides, returning summary + per-lap + time series. Frontend wires the form. ~2 days.
8. **Expand Verification channels** (RPM, torque, pack V/I, temp) + sector-table channels — 1 day.
9. **Retire Dash dashboard, rewrite Docker + README** — 0.5 day.
10. **Toast / error surface / accessibility / URL state** — 0.5–1 day.

---

## 6. Files touched in this setup pass

Created:
- `docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md` (this file)
- `webapp/src/pages/verification/*` (renamed from `validation/`)
- `webapp/src/pages/simulate/SimulatePage.tsx` (stub)
- `webapp/src/stores/simulateStore.ts` (stub)

Edited:
- `CLAUDE.md` — new mission + roadmap
- `webapp/src/App.tsx` — router with three routes
- `webapp/src/components/Sidebar.tsx` — three links, reordered
- `webapp/src/pages/verification/VerificationPage.tsx` — header label change
- `webapp/src/stores/validationStore.ts` → `verificationStore.ts`

Nothing in `backend/` is renamed in this pass.
