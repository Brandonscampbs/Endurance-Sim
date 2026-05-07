# UI Build-Out Design — FSAE EV Endurance Simulator

**Date:** 2026-05-06
**Status:** Approved by user via "work independently and finish all tasks"
**Scope:** Bring the three-page webapp (Verification, Visualization, Simulate) from "baseline but pretty broken" to functionally complete and visually polished.

## 1. Problem Statement

Current state (verified via parallel audits):

| Page | Status | Gap |
|---|---|---|
| Verification | ~80% functional | No honest accuracy disclosure; users could mistake circular-validation output for ground truth |
| Visualization | ~80% functional | Works; minor polish gaps |
| Simulate | ~40% stub | Run button disabled; no backend `/api/simulate` endpoint exists; outputs all show "—" |

Plus repo-wide: no shared design tokens, no skeleton loaders, generic empty states, minimal motion.

The user has not asked to refactor the simulator physics — only to make the **UI** good. Known sim accuracy gaps (lap time 7.3% fast, energy 4.3% low, see `docs/PHYSICS_AUDIT.md`) must be **surfaced honestly in the UI**, not hidden.

## 2. Goals

1. **Make Simulate page work end-to-end.** New `POST /api/simulate` endpoint accepts max RPM, max motor torque, and SOC→current map; returns aggregate, per-lap, and time-series output plus baseline comparison.
2. **Make Verification page honest.** Surface the known sim accuracy posture (lap time gap, energy gap, holdout vs calibration) at the top of the page so users don't take results out of context.
3. **Polish the visual layer.** Cohesive design tokens, skeleton loaders, designed empty/error states, consistent card/typography/spacing primitives across all three pages.
4. **Don't break what works.** Visualization and the working parts of Verification must remain fully functional.

## 3. Non-Goals

- Mobile responsive (accept desktop ≥1280px).
- CSV export, multi-run comparison, parameter sweep — explicitly out of scope per `docs/superpowers/plans/2026-04-17-repo-cleanup-simulator-focus.md`.
- Driver-model changes; the Simulate page reuses the existing CalibratedStrategy.
- Fixing the underlying physics gaps (C10/C11/S4-S6 in `docs/SIMULATOR_ISSUES.md`).
- Changing the existing API response shapes for `/api/validation`, `/api/visualization`, `/api/track`, `/api/laps`.

## 4. Architecture

### 4.1 Backend: New `POST /api/simulate` endpoint

Location: `backend/routers/simulate.py` + `backend/services/simulate_runner.py` + `backend/models/simulate.py`.

**Request schema (`SimulateRequest`):**
```python
class SocCurrentPoint(BaseModel):
    soc_pct: float       # 0..100
    max_current_a: float # 0..200

class SimulateRequest(BaseModel):
    max_rpm: float                            # 1000..6500
    max_torque_nm: float                      # 20..230
    soc_discharge_map: list[SocCurrentPoint]  # 2..10 points sorted by soc_pct desc; max_current_a non-decreasing as soc_pct increases
```

**Override application:**
- `max_rpm` → `dataclasses.replace(powertrain, motor_speed_max_rpm=max_rpm)`. Also clamp `lvcu_overspeed_rpm` to ≥ `max_rpm`.
- `max_torque_nm` → `dataclasses.replace(powertrain, torque_limit_inverter_nm=max_torque_nm)`.
- `soc_discharge_map` → convert to a piecewise-linear SOC→current ceiling. Approximate with `BatteryConfig.soc_taper_threshold_pct` (highest SOC where current < max-of-curve) and `soc_taper_rate_a_per_pct` (slope from threshold to the next point). The full 6-point map is documented as "approximated to 2-knee taper" in the response metadata so the UI can show a tooltip. If implementation effort is low, plumb the full piecewise table through `BatteryModel` instead.

**Response schema (`SimulateResponse`):**
```python
class SimulateLapResult(BaseModel):
    lap_number: int
    time_s: float
    charge_used_ah: float
    energy_used_kwh: float
    mean_speed_kmh: float
    peak_speed_kmh: float
    end_soc_pct: float

class SimulateTimeSeries(BaseModel):
    distance_m: list[float]      # downsampled to ~2000 points
    speed_kmh: list[float]
    pack_power_w: list[float]
    pack_current_a: list[float]
    soc_pct: list[float]
    cumulative_charge_ah: list[float]

class SimulateAggregate(BaseModel):
    total_time_s: float
    net_ah: float
    net_kwh: float
    completed_laps: int
    target_laps: int
    final_soc_pct: float

class SimulateResponse(BaseModel):
    request: SimulateRequest
    summary: SimulateAggregate
    laps: list[SimulateLapResult]
    time_series: SimulateTimeSeries
    baseline: SimulateAggregate
    baseline_laps: list[SimulateLapResult]
    baseline_time_series: SimulateTimeSeries
    notes: list[str]                   # e.g. "SOC current map approximated to 2-knee taper"
```

The baseline response is cached (`@lru_cache`) — only the override branch re-runs.

**Caching:** Per-request override results are cached by a frozenset/tuple key derived from the request (rounding floats to one decimal). Not strictly necessary but keeps repeat tweaks fast.

### 4.2 Frontend: Simulate page

Files: `webapp/src/pages/simulate/SimulatePage.tsx` + new components.

**New components under `pages/simulate/`:**
- `TuneForm.tsx` — extracted form (max RPM, max torque, SOC current map), validation, baseline reset. Lives inside SimulatePage.
- `RunSummaryCards.tsx` — 5 metric cards (Total time, Completed laps, Net Ah, Net energy, Final SOC) each showing absolute value + Δ vs baseline (signed, color-coded green better / red worse / gray neutral by metric).
- `LapDeltaTable.tsx` — per-lap table: lap | time (sim, baseline, Δ) | net Ah (sim, baseline, Δ) | net kWh | mean speed.
- `TimeSeriesOverlay.tsx` — Plotly multi-pane figure: speed, pack power, cumulative charge, SOC overlaid (sim vs baseline), shared X axis (distance).
- `SimulateEmpty.tsx` — designed empty state shown before first run.
- `SimulateRunningOverlay.tsx` — overlays form + content with skeleton + progress text while in flight.

**State (extends `simulateStore.ts`):**
```ts
interface SimulateState {
  params: SimulateParams
  setMaxRpm / setMaxTorque / setSocPoint / resetToBaseline
  // new:
  status: 'idle' | 'running' | 'success' | 'error'
  result: SimulateResponse | null
  error: string | null
  runSimulation: () => Promise<void>     // calls POST /api/simulate
  clearResult: () => void
}
```

**API client:** add `runSimulation(params): Promise<SimulateResponse>` to `webapp/src/api/client.ts`. Handles loading toasts ("Running endurance sim…") and error toasts on failure.

### 4.3 Frontend: Verification confidence banner

New component `webapp/src/pages/verification/AccuracyBanner.tsx`. Lives at the top of `VerificationPage.tsx` above the lap selector.

**Content (computed from `useAllLaps()` aggregate metrics, not hard-coded):**
- Driving time error (sim vs real, signed % delta with badge color)
- Net charge error
- Net energy error
- Holdout speed RMSE (if exposed by aggregate; if not, omit and show a TODO note in the spec)

**Static disclaimer line (always rendered):**
> Suitable for replay analysis and rough sensitivity. Not a predictive tune optimizer — the sim and the track were both fit to this telemetry.

The banner has a collapse toggle so power users can dismiss it after reading once (state lives in `localStorage`, key `verification-banner-collapsed`).

### 4.4 Design system pass

New file `webapp/src/styles/tokens.css` (loaded by Tailwind's `@layer base`):
- CSS variables for surface levels (`--surface-0` body, `--surface-1` card, `--surface-2` raised card, `--surface-3` interactive)
- Border (`--border-subtle`, `--border-strong`)
- Text (`--text-primary`, `--text-secondary`, `--text-tertiary`, `--text-muted`)
- Accent (`--accent` racing green, `--accent-strong`, `--data-sim`, `--data-real`)
- Status (`--ok`, `--warn`, `--error`)
- Spacing scale documented as Tailwind utilities (no new abstractions)

New shared components under `webapp/src/components/ui/`:
- `Card.tsx` — `<Card>`, `<CardHeader>`, `<CardTitle>`, `<CardBody>`. Replaces the inline `bg-gray-900 border border-gray-800 rounded-lg p-5` pattern.
- `Skeleton.tsx` — `<Skeleton h={…} w={…} />` shimmer block.
- `EmptyState.tsx` — icon (unicode glyph since no emoji rule) + headline + description + optional CTA.
- `Badge.tsx` — small status pill (ok / warn / error / info / muted).
- `MetricCard.tsx` — extracted from current `MetricCards.tsx`; reused on Verification, Simulate, possibly elsewhere.
- `ChartShell.tsx` — wraps a Plot with consistent header, optional skeleton, optional empty state.

Existing pages migrate to these primitives (Verification, Visualization SidePanel, Sidebar). Visual changes are a refresh, not a redesign — same dark theme, racing green accent, but consistent across pages.

### 4.5 Routing & sidebar

Sidebar (`webapp/src/components/Sidebar.tsx`): keep the three nav links and Rerun button. Add a small status indicator (green dot when `/api/health` is OK, red when down) using a SWR poll every 30s. Keep changes minimal.

## 5. Data Flow

```
User types in TuneForm
  → setMaxRpm/Torque/SocPoint in store
User clicks Run
  → store.runSimulation()
  → fetcher POST /api/simulate {max_rpm, max_torque_nm, soc_discharge_map}
    → backend: dataclasses.replace overrides on PowertrainConfig + BatteryConfig
    → SimulationEngine.run(num_laps=detected, initial_soc_pct=95, initial_temp_c=29)
    → also returns cached baseline (no override)
    → assemble SimulateResponse with both
  ← store.result, status='success'
  → RunSummaryCards / LapDeltaTable / TimeSeriesOverlay re-render off store.result
```

## 6. Error Handling

- Backend: validation errors (out-of-range params, non-monotonic SOC map) raise `HTTPException(422, detail=…)`. Sim runtime errors caught at router level → 500 with detail. Existing `RequestIdMiddleware` and `register_exception_handlers` already provide structured responses.
- Frontend: `fetcher` already converts to `ApiError`, which the store catches → `status='error'`, error message rendered inline + toast.
- Form validation: client-side range clamps (1000-6500 RPM, 20-230 Nm, 0-100% SOC, 0-200 A current) with red border on invalid; Run button disabled while invalid or in flight.

## 7. Testing

**Backend:**
- `tests/test_simulate_endpoint.py`:
  - Returns 200 with baseline params; aggregate matches `get_baseline_result()`.
  - Override max_torque_nm=40 yields longer total time than baseline (slower).
  - Override max_rpm=1500 yields longer total time than baseline (capped speed).
  - Invalid range → 422.
  - Non-monotonic SOC map → 422.
  - Result is deterministic (same input → same output) within float tolerance.

**Frontend:**
- Manual smoke: type-check (`tsc -b`), build (`vite build`), open all three pages, click Run on Simulate, verify cards/table/chart populate, verify Reset to baseline clears, verify banner collapses on Verification.
- No new vitest suite is added — webapp currently has none. Don't introduce a test framework as part of this change.

## 8. Migration / Rollout

Single PR. The visual changes are evolutionary so existing screenshots in docs remain mostly valid. Backend adds an endpoint, doesn't modify existing ones.

## 9. Open Questions / Risks

- **SOC discharge map fidelity.** If the 2-knee approximation diverges materially from the user's 6-point intent, the Simulate page's energy answer will be misleading. Mitigation: response includes `notes` array; UI shows a tooltip when approximation is active. Long-term fix is a piecewise-linear SOC ceiling in `BatteryModel`, deferred unless trivial.
- **Sim runtime.** Each override sim run takes ~? seconds (need to measure). If it exceeds ~10s the UI must show a meaningful progress indicator beyond a spinner. Plan: streaming is overkill; show running state with elapsed timer + skeleton placeholders for cards/table/chart.
- **Caching of the baseline.** `get_baseline_result()` is `lru_cache(maxsize=1)`, so the baseline-half of the response is free after first call. Override branch is the long pole.
- **Style migration scope.** Touching every page's surfaces means risk of accidental regressions. Mitigation: keep the design refresh additive (introduce primitives, migrate page-by-page, never change behavior).

## 10. Implementation Plan

Wave 1 (parallel, no inter-dependencies):
1. Backend: implement `POST /api/simulate` (router + service + models + tests).
2. Frontend: design tokens + UI primitives (Card / Skeleton / EmptyState / Badge / MetricCard / ChartShell). No page changes yet.
3. Frontend: scaffold `AccuracyBanner` component (uses existing `useAllLaps`).

Wave 2 (after Wave 1):
4. Frontend: wire `SimulatePage` to new endpoint using new primitives. Depends on #1 (endpoint) and #2 (primitives).
5. Frontend: refresh `VerificationPage` to use new primitives + mount `AccuracyBanner`. Depends on #2 and #3.
6. Frontend: refresh `VisualizationPage` (`SidePanel`) to use new primitives. Depends on #2.

Wave 3:
7. Type-check (`tsc -b`), backend pytest, build webapp, dev-server smoke test of all three pages.
