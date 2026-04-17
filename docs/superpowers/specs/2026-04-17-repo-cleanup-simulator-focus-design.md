# Repo Cleanup: Simulator Focus (Design)

**Date:** 2026-04-17
**Author:** brainstorming session with user (uconnracing@gmail.com)
**Status:** approved for plan

## Goal

Strip this repo to **exactly one thing**: an accurate FSAE EV endurance simulator
for CT-16EV with a linked 3-page webapp (Verification / Visualization / Simulate).
Remove all optimization, parameter-sweep, Pareto, and historical-planning content.
The user will fork this repo later for sweep/optimization work; this repo stays
focused on simulation accuracy and the three-page tool.

## Non-Goals

- Building or changing the **Simulate** page (stub stays as-is; separate work).
- Any physics changes.
- Any test rewrites beyond deleting tests whose target modules are removed.
- Removing CT-17EV config (explicitly kept — both CT-16EV and CT-17EV configs stay).
- Removing `fsae_sim.analysis.scoring` (explicitly kept — FSAE endurance/efficiency
  points stay in the simulator).

## Scope: Removals (~95 items)

### Code (src + tests)

- `src/fsae_sim/optimization/` (folder: `__init__.py`, `sweep.py`) — sweep runner, stub only
- `src/fsae_sim/analysis/metrics.py` — `NotImplementedError` stubs for
  `compute_lap_times`, `compute_energy_per_lap`, `compute_pareto_frontier`
- `tests/test_analysis_init.py` — only tests the above stubs; whole file goes

### Dev scripts (keep data-pipeline scripts; remove diagnostics)

Remove:
- `scripts/analyze_gps_laps.py`
- `scripts/diagnose_commands.py`
- `scripts/fix_gps_data.py`
- `scripts/track_map.png`
- `scripts/validate_driver_model.py`
- `scripts/validate_tier3.py`

Keep:
- `scripts/clean_endurance_data.py` — produces `CleanedEndurance.csv`
- `scripts/transplant_fx_coefficients.py` — tire Fx coefficient transplant

### Archive folder (full wipe, 27 items)

- `archive/analysis/` (8 exploratory scripts)
- `archive/scripts/` (18 exploratory scripts)
- `archive/README.md`

### Docs — top-level obsolete files

- `docs/ARCHITECTURE.md`
- `docs/DRIVER_MODEL_FIXES_POSTMORTEM_2026-04-16.md`
- `docs/PARALLEL_WORKSTREAMS.md`
- `docs/simulation_alignment_log.md`
- `docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md`

### Docs — folders

- `docs/obsidian-vault/` (34 files, includes `.obsidian/` config)
- `docs/superpowers/specs/` (14 historical design docs)
- `docs/superpowers/plans/` (8 historical implementation plans)

Note: this spec file itself lives in `docs/superpowers/specs/` and **will be
removed** along with the other specs during cleanup commit 4. Its content is
preserved in git history via the commit that adds it.

## Scope: Rewrites (4 files)

### 1. `src/fsae_sim/analysis/__init__.py`

Strip the 8-line NF-55 comment block explaining that metrics stubs are
intentionally excluded from `__all__`. The stubs no longer exist, so the
explanation is dead context. Keep imports and `__all__` verbatim.

### 2. `docs/SIMULATOR_ISSUES.md` — condense ~272 → ~60 lines

**Motivation:** user flagged that this file loads into CLAUDE.md context on
every session; currently over-long.

**Target structure:**

- Status summary table (5 rows, unchanged)
- **OPEN** — the ~50 unresolved issues, one line each, grouped by subsystem
  (powertrain / battery / tires / driver / validation)
- **PARTIAL** — 2 items with one-line "what's still wrong"
- **DEFERRED** — intentionally-out-of-scope list, one line each

**Drop:**
- FIXED section (71 items — git log is authoritative for resolved work)
- REFUTED section (2 items — settled)
- Any sweep/optimization/Pareto entries
- Commit-hash breadcrumbs for fixed items

**Kept purpose:** a concise "what's still wrong with the sim you should know
before touching physics," not "here's the debugging history."

### 3. `README.md` — rewrite to reduced scope

**New structure:**

- One-paragraph "what this is" (FSAE EV endurance sim for CT-16EV, Python
  package + FastAPI + React webapp, validated against Michigan 2025 telemetry)
- Cars table (CT-16EV / CT-17EV, unchanged)
- Architecture diagram (simplified — drop sweep/analysis pipeline, drop
  `optimization` from the module table)
- Webapp pages (Verification / Visualization / Simulate — one sentence each)
- Quick start (local dev + docker, unchanged)
- Project structure tree (post-cleanup layout)
- Tests section (unchanged)

**Drop:** any mention of parameter sweeps, Pareto, optimization runner,
`optimization/` module.

### 4. `CLAUDE.md` — rewrite to reduced scope

**Keep the structure** (What This Repo Is / Data Assets / Vehicle Parameters /
Roadmap / Architecture Guidance / Subagents / Methodology) with these edits:

- **What This Repo Is** — reframe to "simulator + 3-page webapp only." Keep the
  "out of scope" paragraph (fork-later repo) as a guardrail.
- **Data Assets** — unchanged. All entries still used.
- **Known Issues** — pointer to condensed `SIMULATOR_ISSUES.md` stays.
- **Vehicle Parameters** — unchanged.
- **Project Roadmap** — drop reference to `WEBAPP_REFOCUS_PLAN_2026-04-16.md`
  (deleted). Inline a 3-line version of current state: baseline validated;
  webapp 3-page shell in place; Simulate backend endpoint is next.
- **Architecture Guidance** — drop the sweep clause of the "leave room for
  sweep repo" bullet (keep the "don't prevent vectorization" part). Remove any
  other sweep/optimization phrasing.
- **Installed VoltAgent Subagent Packages** — unchanged.
- **Subagents and When to Use Them** — rewrite `performance-engineer` row to
  be about sim-runtime profiling only (drop "parallelization of parameter
  sweeps"). Otherwise unchanged.
- **Development Methodology** — unchanged (Superpowers).

## Preserved (what stays)

### Code

- `src/fsae_sim/` — `vehicle/`, `track/`, `driver/`, `sim/`, `data/`,
  `physics_constants.py`, `__init__.py`
- `src/fsae_sim/analysis/` — `scoring.py`, `telemetry_analysis.py`,
  `validation.py`, `__init__.py` (rewritten per above)
- `backend/` — FastAPI, all routers (`cache`, `laps`, `track`, `validation`,
  `visualization`)
- `webapp/` — React + Vite SPA, three pages (`verification/`, `visualization/`,
  `simulate/`)
- `tests/` — all remaining 22 `test_*.py` files plus `conftest.py` and `__init__.py` (only `test_analysis_init.py` removed)

### Data + configs

- `configs/ct16ev.yaml` — target car
- `configs/ct17ev.yaml` — next-year car (explicitly kept)
- `Real-Car-Data-And-Stats/` — DSS, AiM, Voltt, LVCU, EMRAX map, tires, cleaned endurance

### Infrastructure

- `docker/`, `docker-compose.yaml`, `pyproject.toml`, `.gitignore`, `.claudeignore`, `.claude/`
- `results/` — empty `.gitkeep` marker

## Execution Approach

### Setup

- **Worktree:** `../Endurance-Sim-cleanup`
- **Branch:** `chore/repo-cleanup-simulator-focus`
- **Final PR:** single PR with all commits, title: *chore: strip sweep/optimization scope; simulator + 3-page webapp only*

### Commit sequence (7 commits, ordered safe→risky)

| # | Commit | Verification |
|---|---|---|
| 1 | Remove `src/fsae_sim/optimization/`, `analysis/metrics.py`, `tests/test_analysis_init.py`; strip NF-55 comment from `analysis/__init__.py` | `pytest -v` — all green |
| 2 | Remove `archive/` (27 files) | `pytest -v` — unchanged |
| 3 | Remove 6 diagnostic scripts from `scripts/` | `pytest -v` — unchanged |
| 4 | Remove obsolete top-level docs + `docs/obsidian-vault/` + `docs/superpowers/{specs,plans}/` | none (docs only) |
| 5 | Condense `docs/SIMULATOR_ISSUES.md` | none |
| 6 | Rewrite `README.md` | none |
| 7 | Rewrite `CLAUDE.md` | none |

### Final verification gates (before PR)

1. `pytest -v` — all tests pass
2. `python -c "import backend.main"` — backend imports cleanly
3. `grep -r "from fsae_sim.optimization\|from fsae_sim.analysis.metrics\|from fsae_sim.analysis import compute_"` — no dangling imports
4. `grep -rn "sweep\|pareto\|Pareto" docs/ README.md CLAUDE.md` — no leftover mentions in kept docs
5. `cd webapp && npm run build` — frontend still builds

## Risks & Mitigations

- **Risk:** A still-live code path imports from `optimization/` or `metrics.py`
  and the `pytest` gate misses it.
  **Mitigation:** The grep check in verification gate 3 runs against the full
  repo (incl. scripts, backend, webapp) to catch any surviving references before PR.
- **Risk:** Rewritten CLAUDE.md drops guidance that future sessions need.
  **Mitigation:** The rewrite is additive-preserving — structure stays,
  content only trimmed where it references removed modules. User reviews the
  rewrite commit before merge.
- **Risk:** Condensed `SIMULATOR_ISSUES.md` loses a bug the user cared about.
  **Mitigation:** Full original preserved in git history (pre-commit 5). User
  reviews condensed version in PR.
