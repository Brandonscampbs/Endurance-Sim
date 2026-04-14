# Parallel Workstreams — Repo Gap Audit

**Date:** 2026-04-14
**Context:** Audit of what's missing before Phase 3 (strategy + tune sweeps). Six high-impact workstreams identified, four fully independent.

---

## Parallelism Map

```
Independent (start anytime):        Dependent (need #4 first):
┌─────────────────────────┐         ┌─────────────────────────┐
│ 1. CI Pipeline          │         │ 5. Sweep Runner         │
│ 2. Structured Logging   │         │ 6. Dashboard Wiring     │
│ 3. Config Validation    │         └─────────────────────────┘
│ 4. CLI + Persistence    │──────────────────┘
└─────────────────────────┘
```

---

## 1. CI Pipeline

**Status:** Nothing exists — no `.github/workflows/`, no pre-commit hooks, no linting config.

**What to build:**
- GitHub Actions workflow: run `pytest`, `mypy`, `ruff` on push and PR
- Pre-commit hooks for formatting and lint
- Code coverage reporting (pytest-cov is already in dev deps, just unconfigured)
- Coverage threshold enforcement

**Why it matters:** 18 test files exist but never run automatically. Regressions can slip in silently.

**Depends on:** Nothing.

---

## 2. Structured Logging

**Status:** 1 logging import in the entire `src/`. Zero visibility into simulation runs.

**What to build:**
- Python `logging` module integration across core modules
- Key log points: lap boundaries, energy checkpoints, constraint activations (BMS taper, SOC limits), driver zone transitions
- Configurable log levels (DEBUG for development, INFO for sweep runs)
- Structured format so logs can be parsed programmatically

**Why it matters:** When a sweep produces a weird result across thousands of runs, you need to trace what happened without re-running in a debugger.

**Depends on:** Nothing.

---

## 3. Config Validation (Pydantic)

**Status:** YAML configs in `configs/` are parsed but never validated. Bad values silently propagate through physics.

**What to build:**
- Pydantic models for vehicle configuration with field-level validation
- Range checks: mass > 0, gear ratio > 1, voltage within cell limits, CdA > 0, etc.
- Dimensional consistency checks where possible
- Clear error messages pointing to the offending config field
- Validate on load in the existing config pipeline

**Why it matters:** A typo in `ct16ev.yaml` (e.g., mass in grams instead of kg) produces plausible but wrong simulation results. Catch it at load time, not after a 2-hour sweep.

**Depends on:** Nothing.

---

## 4. CLI + Result Persistence

**Status:** No CLI entry point (dashboard is the only way to interact). Simulations produce no saved output — results vanish after every run.

**What to build:**
- CLI via `typer` or `click`:
  - `fsae-sim run` — single simulation with specified config and strategy
  - `fsae-sim sweep` — parameter sweep (delegates to sweep runner)
  - `fsae-sim validate` — run validation against telemetry
- Result serialization:
  - Parquet or JSON output to `results/`
  - Run metadata: config hash, git SHA, timestamp, parameters used
  - Lap-by-lap breakdown (time, energy, SOC) plus aggregate metrics
- Result indexing: manifest or lightweight SQLite so past runs are discoverable

**Why it matters:** Phase 3 requires running thousands of sims programmatically and comparing results. This is the single biggest blocker. Two other workstreams (sweep runner, dashboard) depend on it.

**Depends on:** Nothing (but #5 and #6 depend on this).

---

## 5. Sweep Runner

**Status:** `optimization/sweep.py` exists but raises `NotImplementedError`.

**What to build:**
- Grid sweep over zone overrides (via `CalibratedStrategy.with_zone_override()`)
- Grid/random sweep over car tune params (max RPM, torque limit, regen intensity)
- Combined sweep optimizing `FSAEScoring.combined_score` (Endurance + Efficiency, max 375 pts)
- Parallel execution (multiprocessing or joblib) — each sim is independent
- Progress reporting and intermediate result saving
- Optional Bayesian optimization (Optuna) for smart search

**Why it matters:** This IS Phase 3. The whole point of the simulation is to find the scoring-optimal strategy+tune combination.

**Depends on:** #4 (CLI + Result Persistence) — sweep results need somewhere to go.

---

## 6. Dashboard Data Integration

**Status:** All 6 dashboard pages are empty stubs with placeholder text. Navigation works, styling works, but there's no data.

**What to build:**
- Wire overview page to load and display real simulation results
- Strategy comparison page: side-by-side zone maps and lap traces
- Sweep results page: heatmaps and Pareto fronts from parameter sweeps
- Lap detail page: speed/energy/SOC traces for individual runs
- Result filtering and selection controls
- Export/download of results and plots

**Why it matters:** The team needs to see and compare results visually, not dig through Parquet files. This is also how you'd present optimal strategies to drivers.

**Depends on:** #4 (CLI + Result Persistence) — dashboard needs saved results to display.

---

## Other Notable Gaps (lower priority)

These aren't full workstreams but are worth addressing as they come up:

| Gap | Current State | Impact |
|-----|--------------|--------|
| Error handling | 27 try/except across entire src | Silent failures in edge cases |
| Telemetry CSV validation | No schema checks on AiM data | Bad data rows propagate |
| Docker health checks | No readiness/liveness probes | Dashboard can appear "up" while broken |
| Multi-stage Docker build | Single-stage, includes dev deps | Larger image than necessary |
| Physics bugs | 10+ documented in `docs/REMAINING_ISSUES.md` | Must fix before sweep results are trustworthy |
| Benchmarking | No profiling infrastructure | Can't measure sweep performance gains |
| API docs | No Sphinx/pdoc generation | Hard for new contributors to navigate |
