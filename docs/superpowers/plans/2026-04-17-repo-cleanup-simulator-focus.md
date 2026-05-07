# Repo Cleanup: Simulator Focus Implementation Plan

> **Historical note:** Older versions of this plan treated displayed SOC as
> an energy comparison. That is superseded. Current validation uses net pack
> amp-hours and net V*I energy; displayed BMS/AiM SOC is not a scored
> sim-vs-telemetry metric.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip this repo to an accurate FSAE EV endurance simulator for CT-16EV plus a linked 3-page webapp (Verification / Visualization / Simulate). Remove optimization/sweep/Pareto code, the `archive/` folder, 6 diagnostic scripts, 5 obsolete top-level docs, the obsidian vault, and all historical superpowers specs/plans. Condense `docs/SIMULATOR_ISSUES.md`. Rewrite `README.md` and `CLAUDE.md` to match the reduced scope.

**Architecture:** Destructive cleanup as 8 commits on a dedicated worktree branch. Ordered safe → risky: code-impacting deletions first (gated by `pytest`), then doc deletions, then doc rewrites. Final verification battery before opening a PR on `main`.

**Tech Stack:** git / git worktrees, pytest, Python 3.11, bash (git-bash on Windows), npm (webapp build check), grep.

**Design spec:** `docs/superpowers/specs/2026-04-17-repo-cleanup-simulator-focus-design.md` (untracked on main; treat as reference).

**Preserved explicitly** (do NOT delete, despite looking adjacent to removed content):
- `src/fsae_sim/analysis/scoring.py` + `tests/test_scoring.py` — FSAE endurance/efficiency points.
- `src/fsae_sim/analysis/telemetry_analysis.py` + `validation.py` — actively used by backend and driver.
- `configs/ct17ev.yaml` — next-year car config.
- `scripts/clean_endurance_data.py`, `scripts/transplant_fx_coefficients.py` — data-pipeline scripts.

---

## Task 0: Setup — baseline pytest + worktree + branch

**Files:**
- Create: worktree at `../Endurance-Sim-cleanup` with branch `chore/repo-cleanup-simulator-focus`
- Reference only: `docs/superpowers/specs/2026-04-17-repo-cleanup-simulator-focus-design.md`, `docs/superpowers/plans/2026-04-17-repo-cleanup-simulator-focus.md`

- [ ] **Step 1: Verify main working tree state**

Run (from the main repo `C:/Users/brand/Endurance-Sim`):
```bash
git status --short
git log -1 --oneline
```

Expected: working tree may show `.claudeignore` as deleted and the untracked spec + plan files. Current HEAD should be `591d79e fix(sim): brake is mechanical, not regen — CT-16EV has no regen` (or later if repo has advanced).

- [ ] **Step 2: Capture baseline pytest result on main**

Run:
```bash
cd C:/Users/brand/Endurance-Sim
pytest -v 2>&1 | tee /tmp/baseline-pytest.log
tail -20 /tmp/baseline-pytest.log
```

Expected: record the pass/fail/xfail counts in the last few lines (something like `N passed, M xfailed in Xs`). Save this number mentally — it's the reference for subsequent gate runs. Any preexisting xfails are NOT regressions.

- [ ] **Step 3: Create worktree**

Run:
```bash
cd C:/Users/brand/Endurance-Sim
git worktree add -b chore/repo-cleanup-simulator-focus ../Endurance-Sim-cleanup main
cd ../Endurance-Sim-cleanup
```

Expected: `HEAD is now at <sha>` and you land in the new worktree directory. `git status` shows a clean working tree.

- [ ] **Step 4: Confirm worktree pytest matches baseline**

Run (from the worktree):
```bash
pytest -v 2>&1 | tail -5
```

Expected: same pass/xfail counts as Step 2. If any test fails that wasn't in the baseline, stop and investigate — don't proceed until the worktree is equivalent to main.

- [ ] **Step 5: Mark task done (no commit — setup only)**

Update progress tracking. Task 0 makes no commits.

---

## Task 1: Remove optimization/, analysis/metrics.py, test_analysis_init.py; strip NF-55 comment

**Files:**
- Delete: `src/fsae_sim/optimization/__init__.py`
- Delete: `src/fsae_sim/optimization/sweep.py`
- Delete: `src/fsae_sim/analysis/metrics.py`
- Delete: `tests/test_analysis_init.py`
- Modify: `src/fsae_sim/analysis/__init__.py` (strip NF-55 comment)

- [ ] **Step 1: Confirm nothing outside the targets imports from the deleted modules**

Run (from the worktree):
```bash
grep -rn "from fsae_sim.optimization\|import fsae_sim.optimization\|from fsae_sim.analysis.metrics\|from fsae_sim.analysis import compute_lap_times\|from fsae_sim.analysis import compute_energy_per_lap\|from fsae_sim.analysis import compute_pareto_frontier" src/ backend/ webapp/ scripts/ tests/
```

Expected: ZERO matches outside `tests/test_analysis_init.py`. (That file is being deleted in this same task, so its matches are fine.) If anything else matches, stop — something is using the module we're about to delete. Re-check the design spec's "Preserved" section.

- [ ] **Step 2: Delete optimization/ folder**

Run:
```bash
rm -rf src/fsae_sim/optimization/
```

- [ ] **Step 3: Delete analysis/metrics.py**

Run:
```bash
rm src/fsae_sim/analysis/metrics.py
```

- [ ] **Step 4: Delete tests/test_analysis_init.py**

Run:
```bash
rm tests/test_analysis_init.py
```

- [ ] **Step 5: Strip NF-55 comment from analysis/__init__.py**

The current file is:

```python
from fsae_sim.analysis.scoring import (
    CompetitionField,
    FSAEScoreResult,
    FSAEScoring,
)
from fsae_sim.analysis.telemetry_analysis import (
    DriverZone,
    extract_per_segment_actions,
    collapse_to_zones,
    detect_laps,
    compare_driver_stints,
)

# NF-55: `compute_lap_times`, `compute_energy_per_lap`, and
# `compute_pareto_frontier` in `fsae_sim.analysis.metrics` still raise
# `NotImplementedError`.  They are intentionally NOT re-exported here so
# IDE autocomplete and `from fsae_sim.analysis import *` do not suggest
# callable APIs that aren't callable.  They remain importable directly
# from `fsae_sim.analysis.metrics` until implemented or removed.

__all__ = [
    "CompetitionField",
    "FSAEScoreResult",
    "FSAEScoring",
    "DriverZone",
    "extract_per_segment_actions",
    "collapse_to_zones",
    "detect_laps",
    "compare_driver_stints",
]
```

Replace with:

```python
from fsae_sim.analysis.scoring import (
    CompetitionField,
    FSAEScoreResult,
    FSAEScoring,
)
from fsae_sim.analysis.telemetry_analysis import (
    DriverZone,
    extract_per_segment_actions,
    collapse_to_zones,
    detect_laps,
    compare_driver_stints,
)

__all__ = [
    "CompetitionField",
    "FSAEScoreResult",
    "FSAEScoring",
    "DriverZone",
    "extract_per_segment_actions",
    "collapse_to_zones",
    "detect_laps",
    "compare_driver_stints",
]
```

(Use the Edit tool to replace the NF-55 comment block — the 7-line block beginning with `# NF-55:` and ending with the blank line before `__all__ = [`.)

- [ ] **Step 6: Run pytest — verify no regressions**

Run:
```bash
pytest -v 2>&1 | tail -5
```

Expected: same pass/xfail count as Task 0 Step 4, minus the 3 tests that were in `test_analysis_init.py` (those are gone, so total-count-minus-3 is correct). Zero NEW failures.

If any new failure appears, stop and debug. A likely cause would be a missed import somewhere.

- [ ] **Step 7: Commit**

Run:
```bash
git add -A
git status
```

Expected `git status` output: 5 deletions (`optimization/__init__.py`, `optimization/sweep.py`, `analysis/metrics.py`, `tests/test_analysis_init.py`) and 1 modification (`analysis/__init__.py`).

Then:
```bash
git commit -m "$(cat <<'EOF'
chore(repo): remove optimization sweep stub and metrics stubs

Drops src/fsae_sim/optimization/ (sweep runner, unimplemented) and
src/fsae_sim/analysis/metrics.py (Pareto/lap-metric stubs that raised
NotImplementedError). These were optimization-scope only; no code in
fsae_sim, backend, webapp, or scripts imported them.

tests/test_analysis_init.py removed — it only guarded the stub export
policy and has no remaining target. analysis/__init__.py loses the
NF-55 comment block explaining the (now non-existent) stubs.

Part of repo-cleanup-simulator-focus: strip sweep/optimization scope.
EOF
)"
```

Verify: `git log -1 --stat` shows 5 deletions + 1 modification.

---

## Task 2: Remove archive/ folder

**Files:**
- Delete: `archive/` (27 items — `archive/analysis/` with 8 scripts, `archive/scripts/` with 18 scripts, `archive/README.md`)

- [ ] **Step 1: Verify archive/ has no inbound references from kept code**

Run:
```bash
grep -rn "archive/\|from archive\|import archive" src/ backend/ webapp/ tests/ scripts/ configs/
```

Expected: ZERO matches. Archive is isolated dev exploration.

- [ ] **Step 2: Delete archive/**

Run:
```bash
rm -rf archive/
ls -la | grep archive || echo "archive/ gone"
```

Expected: `archive/ gone`.

- [ ] **Step 3: Run pytest — verify no regressions**

Run:
```bash
pytest -v 2>&1 | tail -5
```

Expected: same count as Task 1 Step 6. No new failures.

- [ ] **Step 4: Commit**

Run:
```bash
git add -A
git status
```

Expected: ~27 file deletions under `archive/`.

```bash
git commit -m "$(cat <<'EOF'
chore(repo): remove archive/ folder

Drops 27 exploratory/debug scripts under archive/analysis and
archive/scripts accumulated during the 2026-04 physics-fix campaigns.
Nothing in the live codebase imports from archive/.

Part of repo-cleanup-simulator-focus.
EOF
)"
```

---

## Task 3: Remove diagnostic scripts from scripts/

**Files:**
- Delete: `scripts/analyze_gps_laps.py`
- Delete: `scripts/diagnose_commands.py`
- Delete: `scripts/fix_gps_data.py`
- Delete: `scripts/track_map.png`
- Delete: `scripts/validate_driver_model.py`
- Delete: `scripts/validate_tier3.py`
- Preserve: `scripts/clean_endurance_data.py`, `scripts/transplant_fx_coefficients.py`

- [ ] **Step 1: Verify no kept code references the diagnostic scripts**

Run:
```bash
grep -rn "analyze_gps_laps\|diagnose_commands\|fix_gps_data\|validate_driver_model\|validate_tier3\|track_map.png" src/ backend/ webapp/ tests/ configs/ docs/ pyproject.toml README.md CLAUDE.md
```

Expected: references may appear in `docs/CLAUDE.md`, `docs/SIMULATOR_ISSUES.md`, `README.md`. Those docs get rewritten in Tasks 5–7, so any stale pointers will be cleaned up then. No references in code/config is what matters.

- [ ] **Step 2: Delete the 6 files**

Run:
```bash
rm scripts/analyze_gps_laps.py \
   scripts/diagnose_commands.py \
   scripts/fix_gps_data.py \
   scripts/track_map.png \
   scripts/validate_driver_model.py \
   scripts/validate_tier3.py
ls scripts/
```

Expected `ls scripts/`: exactly two entries — `clean_endurance_data.py` and `transplant_fx_coefficients.py`.

- [ ] **Step 3: Run pytest — verify no regressions**

Run:
```bash
pytest -v 2>&1 | tail -5
```

Expected: same count. No new failures.

- [ ] **Step 4: Commit**

Run:
```bash
git add -A
git status
```

Expected: 6 deletions under `scripts/`.

```bash
git commit -m "$(cat <<'EOF'
chore(repo): remove diagnostic dev scripts

Drops 6 pre-webapp diagnostic scripts from scripts/:
analyze_gps_laps.py, diagnose_commands.py, fix_gps_data.py,
validate_driver_model.py, validate_tier3.py, track_map.png.

These were ad-hoc sim-vs-telemetry checks; the Verification webapp
page now covers this interactively. Data-pipeline scripts
(clean_endurance_data.py, transplant_fx_coefficients.py) preserved —
they produce input files the sim needs.

Part of repo-cleanup-simulator-focus.
EOF
)"
```

---

## Task 4: Remove obsolete docs, obsidian vault, historical specs/plans, and .claudeignore

**Files:**
- Delete: `docs/ARCHITECTURE.md`
- Delete: `docs/DRIVER_MODEL_FIXES_POSTMORTEM_2026-04-16.md`
- Delete: `docs/PARALLEL_WORKSTREAMS.md`
- Delete: `docs/simulation_alignment_log.md`
- Delete: `docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md`
- Delete: `docs/obsidian-vault/` (34 files including `.obsidian/`)
- Delete: `docs/superpowers/specs/` (14 files)
- Delete: `docs/superpowers/plans/` (8 files)
- Delete: `.claudeignore` (only excluded `docs/obsidian-vault/`, which this task removes)

- [ ] **Step 1: Sanity-check the 5 top-level docs are the right ones to drop**

Run:
```bash
ls docs/*.md
```

Expected: you should see exactly 6 `.md` files in `docs/` —
`ARCHITECTURE.md`, `DRIVER_MODEL_FIXES_POSTMORTEM_2026-04-16.md`,
`PARALLEL_WORKSTREAMS.md`, `simulation_alignment_log.md`,
`SIMULATOR_ISSUES.md`, `WEBAPP_REFOCUS_PLAN_2026-04-16.md`. We keep `SIMULATOR_ISSUES.md` (condensed in Task 5) and delete the other 5.

- [ ] **Step 2: Delete the 5 obsolete top-level docs**

Run:
```bash
rm docs/ARCHITECTURE.md \
   docs/DRIVER_MODEL_FIXES_POSTMORTEM_2026-04-16.md \
   docs/PARALLEL_WORKSTREAMS.md \
   docs/simulation_alignment_log.md \
   docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md
ls docs/*.md
```

Expected `ls`: only `docs/SIMULATOR_ISSUES.md`.

- [ ] **Step 3: Delete obsidian-vault/**

Run:
```bash
rm -rf docs/obsidian-vault/
ls -la docs/ | grep obsidian || echo "obsidian-vault/ gone"
```

Expected: `obsidian-vault/ gone`.

- [ ] **Step 4: Delete docs/superpowers/specs/ and docs/superpowers/plans/**

Run:
```bash
rm -rf docs/superpowers/specs/ docs/superpowers/plans/
ls docs/superpowers/ 2>/dev/null || echo "superpowers/ now empty — safe to leave or prune"
```

If `docs/superpowers/` is now empty after removing both subfolders, it's fine to leave (git doesn't track empty directories, so it disappears from tracking automatically). If there are OTHER files under `docs/superpowers/` that weren't in the design, stop and investigate.

Expected: `docs/superpowers/` empty (or contains only leftover files not in our design; if so, stop and ask).

- [ ] **Step 5: Delete .claudeignore**

Run:
```bash
cat .claudeignore
```

Expected: file content is `docs/obsidian-vault/` (and possibly blank line). That folder was removed in Step 3, so the ignore is now dead.

```bash
rm .claudeignore
ls -la .claudeignore 2>&1 | head -1
```

Expected: `cannot access '.claudeignore': No such file or directory`.

- [ ] **Step 6: No pytest needed (docs-only change) — quick import sanity check**

Run:
```bash
python -c "import backend.main; print('backend import OK')"
```

Expected: `backend import OK`. Guards against having accidentally deleted a doc that was also a Python module.

- [ ] **Step 7: Commit**

Run:
```bash
git add -A
git status
```

Expected: ~62 deletions (5 top-level docs + 34 obsidian-vault files + 14 specs + 8 plans + .claudeignore ≈ 62). Exact number may vary slightly depending on whether `.obsidian/` config has subdirectories; the count is approximate.

```bash
git commit -m "$(cat <<'EOF'
chore(docs): remove obsolete docs, obsidian vault, and historical plans

Drops:
- 5 top-level docs (ARCHITECTURE, DRIVER_MODEL_FIXES_POSTMORTEM,
  PARALLEL_WORKSTREAMS, simulation_alignment_log, WEBAPP_REFOCUS_PLAN)
- docs/obsidian-vault/ (wiki that duplicates README/CLAUDE.md)
- docs/superpowers/specs/ (14 historical design docs)
- docs/superpowers/plans/ (8 historical implementation plans)
- .claudeignore (only excluded docs/obsidian-vault/, now removed)

SIMULATOR_ISSUES.md retained; rewritten for conciseness in a
following commit.

Part of repo-cleanup-simulator-focus.
EOF
)"
```

---

## Task 5: Condense docs/SIMULATOR_ISSUES.md (272 → ~85 lines)

**Files:**
- Modify: `docs/SIMULATOR_ISSUES.md` (full rewrite — replace whole file)

- [ ] **Step 1: Replace the file with the condensed version**

Use the Write tool to overwrite `docs/SIMULATOR_ISSUES.md` with EXACTLY this content:

```markdown
# FSAE EV Simulator — Known Issues

Current open physics/code gaps. Detail and fix history live in git; this file is a working list of what's still wrong, intended to stay small enough to live in CLAUDE.md context.

## Status

| Status   | Count | Notes |
|----------|-------|-------|
| PARTIAL  | 2     | Regen double-count residual; tire radius not dynamic |
| DEFERRED | ~18   | Engine-arch rewrites, test/config hygiene |
| OPEN     | ~50   | Moderate/Minor buckets untriaged |

Legend: `C*` critical, `S*` significant, `M*` moderate, `m*` minor, `NF-*` new-finding, `D-*` driver-model.

---

## PARTIAL

- **3** `regen_force` generator-mode — S12 addressed sign; confirm with validation run.
- **18** Tire radius 0.2042 m constant, not dynamic `loaded_radius()` (~3% under load).

## OPEN

### Critical
- **C10** Engine integrates speed with entry-speed forces; no Heun corrector.
- **C11** Mechanical vs electrical torque use different operating points.
- **C14** `PedalProfileStrategy` classifier discards torque-based intensity.

### Significant
- **S1** Regen tire-saturation doesn't feed back to electrical power (absorbs into C11).
- **S2/S3** Multiple field-weakening models; replay double-counts.
- **S4** `resolve_exit_speed` clamps without charging energy.
- **S5** Driver decision sees stale `pack_current = 0` per segment.
- **S6** Speed envelope ignores BMS current cap.
- **S7** Combined-slip (Pass 4) dead code.
- **S8** `ReplayStrategy` V×I path — watch for regression.

### Moderate
- **4** `battery_model.py:362` — no cooling term in thermal model.
- **5** Residual scipy optimizer calls may remain in cornering solver.
- **7** Linear field-weakening taper — audit physics path.
- **8** `battery_model.py:257` — internal resistance temperature-independent.
- **9** `analysis/scoring.py:214` — `EFmin = 0.0` inflates efficiency scores.
- **10** `compare_driver_stints` compares same data to itself.
- **12** Python sim loop limits throughput.
- **13** `CalibratedStrategy.decide()` never returns `max_speed_ms`.
- **14** Effective mass includes drivetrain efficiency on rotor inertia.
- **16** Track curvature from GPS acceleration (noisy at low speed).
- **17** 5 m segment resolution — nominal config entry stale (default now 0.5 m).
- **19** ISA air density vs Michigan conditions.
- **20** Downforce front distribution is default, not DSS-measured.
- **22** Back-EMF alone doesn't explain coast power (~45 Wh/stint gap).
- **M1** No anti-squat / anti-dive geometry.
- **M2** Friction ellipse uses peak forces, not combined-slip Pacejka.
- **M3** OCV extrapolated linearly below calibration range.
- **M5** 4P cell current sharing assumed perfect.
- **M6** No OCV hysteresis.
- **M7** Downforce treated inconsistently across resistance functions.
- **M8** Camber sign convention undocumented.
- **M10** Brake distribution load-proportional, not mechanical-bias.
- **M12** PAC2002 Svy missing LKYG on camber term.
- **M13** `m_effective` doesn't distinguish accel vs regen direction.
- **M14** LONGVL speed correction ignored in Fx.
- **M15** `max_traction_force` hardcodes 0.3g load-transfer.
- **M16** Forward-Euler lag between `pack_voltage` and `pack_current`.
- **M17** Regen active at arbitrarily low RPM (no back-EMF cutoff).
- **M18** Driver-change lap not filtered from default calibration.
- **M19** `from_telemetry` ignores user-provided column names.
- **M20** `CoastOnly` / `ThresholdBraking` ignore forward-propagated envelope.

### Minor
- **m2** Plots use `LFspeed`; metrics use `GPS Speed`.
- **m3** Validation plot auto-scales, hiding bias.
- **m4** Pass/fail thresholds loose (15–20 %).
- **m5** Mean-speed / distance / time are algebraic identities.
- **m6** `speed > 5 km/h` filter no-op on cleaned data.
- **m10** Empty-bin carry-forward corrupts curvature across lap wrap.
- **m11** Grade is not smoothed.
- **m12** `ReplayStrategy.decide` 0.05 thresholds inconsistent with calibration units.
- **m13** Cosmetic cluster: linear-scan `zone_for_segment`, standing-start clamp, dead `initial_speed` API, FW constants Michigan-fit, lap-wrap speed clamp, `laps_completed` off-by-one.

### Driver-model (duplicates of above collapsed)
- **D-03** Dead `coast_throttle` parameter in `DriverParams`.
- **D-07** Per-lap distance misalignment at segment sampling.
- **D-08** Brake normalization depends on calibration laps.
- **D-21** `compare_driver_stints` returns segment-level diffs despite zone collapse.
- **D-27** `ControlCommand` dataclass frozen but extension awkward.
- **D-28** `DriverParams` only used by `PedalProfileStrategy`.

### Other / untriaged
- Numerical regularizers (`+1e-6`), `math.fsum` accumulator, `iterrows()` motor map.
- Unit latent: SOC fraction-vs-percent; `lvcu_power_constant` firmware-fit units.
- Conservation: cornering drag ignores load redistribution; distance-accumulator drift.
- Data-loading: CT-17EV YAML stale; `CdA` reference-area convention undocumented.
- Hidden state: module-level track constants not configurable; `.tir` not cached.

## Xfailed tests (deferred)

- `tests/test_engine_envelope.py::test_synthetic_strategy_uses_envelope` — engine exceeds envelope ~1 m/s at tightest corner (related to D-20; needs engine-side fix).
- `tests/test_tire_model.py::test_closed_form_peak_longitudinal_matches_optimizer` — closed-form Fx diverges from optimizer baseline 14–90 % at Fz ≥ 1500 N; needs tire-model audit.

## DEFERRED (intentionally skipped)

- **NF-11** DSS sync script (openpyxl).
- **NF-24** `bms_limit` uses entry SOC/temp vs avg-segment torque — Heun-dependent.
- **NF-58** `isinstance` on concrete strategies — arch refactor.
- **NF-60** `pyproject.toml` / Dockerfile dep + Python version pinning.
- **NF-61** Freeze result/config dataclasses.
- **NF-62** Split `analysis/telemetry_analysis.py`.
- **NF-63** Split `driver/strategies.py`.
- **NF-64** `analysis` ↔ `driver.strategies` import cycle.
- **NF-65** Capability-flag try/except fail-silent at module level.
- **C10** Heun integrator / predictor-corrector.
- **C11** Mechanical-vs-electrical operating-point unification (depends on C10).
- **S2, S3, S5, S6, S7, S8** — engine-arch deep rewrites.

---

Pointer: grep a legacy ID (`C4`, `NF-18`, `D-15`, etc.) against `git log` to find the commit that closed it.
```

- [ ] **Step 2: Verify file size and structure**

Run:
```bash
wc -l docs/SIMULATOR_ISSUES.md
head -15 docs/SIMULATOR_ISSUES.md
```

Expected: ~95 lines total. Header shows the new intro, status table.

- [ ] **Step 3: Confirm no accidental sweep/pareto references**

Run:
```bash
grep -ni "sweep\|pareto\|optimization" docs/SIMULATOR_ISSUES.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Commit**

Run:
```bash
git add docs/SIMULATOR_ISSUES.md
git commit -m "$(cat <<'EOF'
docs(issues): condense SIMULATOR_ISSUES tracker to open items only

Drops the FIXED section (71 items — git log is authoritative),
REFUTED section, commit-hash breadcrumbs, and the reference to
the deleted SIMULATOR_AUDIT_NEW_FINDINGS doc. Keeps PARTIAL + OPEN
+ Xfailed + DEFERRED, restructured as terse one-liners.

Motivation: this file loads into CLAUDE.md context on every session
and was ~272 lines of mostly historical content. Condensed to ~95
lines focused on what's still wrong with the sim.

Part of repo-cleanup-simulator-focus.
EOF
)"
```

---

## Task 6: Rewrite README.md

**Files:**
- Modify: `README.md` (full rewrite — replace whole file)

- [ ] **Step 1: Replace the file with the new version**

Use the Write tool to overwrite `README.md` with EXACTLY this content:

````markdown
# FSAE EV Endurance Simulation

Endurance simulation for UConn Formula SAE Electric (CT-16EV). Predicts lap time, energy, and FSAE endurance points from vehicle config + driver strategy + the Michigan-2025 track. Calibrated against real AiM telemetry from the 2025 Michigan endurance event.

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
3. **Simulate** — one-shot what-if: change max RPM, max torque, and current-limit assumptions; see how endurance time, net Ah, and energy change vs baseline.

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
````

- [ ] **Step 2: Verify content**

Run:
```bash
wc -l README.md
grep -c "^##" README.md
```

Expected: ~100 lines, multiple `##` headers.

- [ ] **Step 3: Confirm no sweep/pareto/optimization references**

Run:
```bash
grep -ni "sweep\|pareto\|optimization\|WEBAPP_REFOCUS" README.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Commit**

Run:
```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): rewrite to simulator + 3-page webapp scope

Drops references to:
- fsae_sim.optimization / fsae_sim.scoring-as-separate-module /
  fsae_sim.analysis "Pareto computation"
- Parameter sweeps in the tagline
- docs/WEBAPP_REFOCUS_PLAN_2026-04-16.md (deleted)
- Phase 4 FSAE scoring roadmap (FSAE scoring lives in analysis now;
  the phased roadmap was sweep-forward)

Updated module table drops `optimization/` and folds scoring into
`analysis/`. Project structure tree matches post-cleanup layout.

Part of repo-cleanup-simulator-focus.
EOF
)"
```

---

## Task 7: Rewrite CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (full rewrite — replace whole file)

- [ ] **Step 1: Replace the file with the new version**

Use the Write tool to overwrite `CLAUDE.md` with EXACTLY this content:

````markdown
# CLAUDE.md

## What This Repo Is

FSAE EV endurance simulation for UConn Formula SAE Electric (car CT-16EV).

**Core mission: build the most accurate FSAE EV endurance simulator possible, and expose it through a three-page webapp.** The three pages are:

1. **Verification** — how close is the baseline simulator to reality? (compare sim vs Michigan 2025 telemetry, per-channel residuals, energy budget reconciliation).
2. **Visualization** — a 3D playback of the car so physics bugs become visible.
3. **Simulate** — a what-if tool for **max motor RPM, max motor torque, and current-limit assumptions**. Run one sim with those overrides, see how endurance time, net Ah, and energy change.

**Out of scope for this repo.** Parameter sweeps, Pareto optimization, multi-run comparison, driver-strategy search, coaching output. Those will live in a separate repo that imports this one as a library. Do not add sweep runners, sweep-results pages, or sweep storage schemas here.

The repo starts with real telemetry and battery simulation data from Michigan 2025.

## Data Assets

### Real-Car-Data-And-Stats/
- **DSS spreadsheet** (`301_Univ_of_Connecticut-DSS-2025-05-05_1957.xlsx`): **Primary source of truth** for vehicle parameters. Contains measured mass, dimensions, suspension geometry, aero coefficients, motor/inverter specs, accumulator details, drivetrain ratios, and brake system data. Always use DSS values over estimates.
- **AiM telemetry** (`2025 Endurance Data.csv`): 20Hz CSV export from AiM Evo 5 data logger. Full Michigan endurance (~22 km, 21 laps, 1859s including driver change). Key channels: GPS Speed, GPS Lat/Lon, GPS LatAcc/LonAcc, RPM, Torque Feedback, Pack Voltage/Current, State of Charge, Pack Temp, Throttle Pos, Brake Pressure, LVCU Torque Req. Binary logs (`.xrk`, `.xrz`, `.drk`, `.rrk`) require AiM Race Studio.
- **Endurance Tune2.txt**: BMS discharge limits, SOC taper, cell voltage bounds, inverter/motor parameter settings.
- **About-Energy-Volt-Simulations/**: Voltt battery simulation export (110S4P, Molicel P45B). Two CSVs -- `_cell.csv` (single-cell level) and `_pack.csv` (pack-scaled). Used for battery model calibration (OCV-SOC curve, internal resistance).
- **LVCU Code.txt**: LVCU firmware source — the torque command chain (`PowertrainModel.lvcu_torque_command()` and related methods) is a faithful translation of this file. Source of truth for `lvcu_power_constant`, `lvcu_rpm_scale`, `lvcu_omega_floor`, and pedal deadzone parameters in `PowertrainConfig`.
- **emrax228_hv_cc_motor_map_long.csv**: EMRAX 228 motor efficiency map (speed_rpm, torque_Nm, efficiency_pct). Loaded by `MotorEfficiencyMap` for 2D operating-point-dependent motor+inverter efficiency. Falls back to constant `drivetrain_efficiency` if missing.
- **Tire Models from TTC/**: PAC02 .tir files for Hoosier LC0 16x7.5-10 at multiple pressures (Round 8 TTC data). Primary: `Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir`. Longitudinal (Fx) coefficients transplanted from R25B donor data via `scripts/transplant_fx_coefficients.py`.
- **CleanedEndurance.csv**: Cleaned AiM telemetry produced by `scripts/clean_endurance_data.py` (removes pre-start, driver change, post-finish). Uses `LFspeed` column (left-front wheel speed) instead of GPS Speed. Latin-1 encoding.

### Known Issues (MUST READ)

**`docs/SIMULATOR_ISSUES.md`** is the concise tracker for open physics gaps and code issues. **Read it before trusting simulation results or starting new physics work.**

### Key Vehicle Parameters (from DSS + Endurance Tune)
| Parameter | Value | Source |
|---|---|---|
| Mass (car only) | 220 kg | DSS |
| Mass (with 68 kg driver) | 288 kg | DSS |
| Wheelbase | 1549 mm | DSS |
| Final drive ratio | 3.6363:1 (40/11) | DSS |
| CdA (drag coefficient x area) | 1.50 m² | DSS (431N drag at 80 kph, back-derived) |
| ClA (downforce coeff x area) | 2.18 m² | DSS (625N downforce at 80 kph, back-derived) |
| Motor | EMRAX 228 MV LC, 3-phase PMSM | DSS |
| Motor peak / continuous | 230 Nm / 112 Nm | DSS (but inverter limits to 85 Nm) |
| Inverter | Cascadia CM200DX | DSS |
| Inverter torque limit | 85 Nm (IQ=170A setting) | Endurance Tune |
| LVCU torque limit | 150 Nm | Endurance Tune |
| Motor speed / brake speed | 2900 / 2400 RPM | Endurance Tune |
| Pack | 110S4P Molicel P45B (5 segments x 22S x 4P) | DSS |
| Pack energy | 7.128 kWh nominal | DSS |
| Cell voltage range | 2.55 -- 4.20 V | DSS + Endurance Tune |
| Max discharge | 100 A @ 30°C, tapers to 0 A @ 65°C | Endurance Tune |
| SOC taper | 1 A per 1% below 85% SOC | Endurance Tune |
| Tires | Hoosier 16x7.5-10 LC0 (10" wheel) | DSS |
| CG height | 279.4 mm | DSS |

## Project State

Baseline sim is validated against Michigan 2025 telemetry (~2% energy error, 8/8 metrics pass). Webapp shell has all three pages; Verification and Visualization are functional; Simulate is a stub pending a backend run endpoint. Current work: close remaining physics gaps (see `docs/SIMULATOR_ISSUES.md`) and implement the Simulate page.

## Architecture Guidance

- **No bandaid fixes — root cause only**: Never apply superficial patches, fudge factors, or tuning hacks to make results match. Every fix must address the actual root cause. This is especially critical in simulation work: if the sim output is wrong, the physics model or inputs are wrong — find out why. Adding correction factors or clamping outputs to hide errors destroys the simulation's predictive value and makes every downstream result untrustworthy. A simulation that's honestly wrong is more useful than one that's been patched to look right.
- **Modular by domain**: separate modules for battery model, drivetrain model, tire/vehicle dynamics, track representation, driver model, and lap simulation orchestration. Each module should be independently testable.
- **Simulation correctness first**: validate every model against real data before adding complexity. Numerical accuracy matters more than abstraction elegance.
- **Performance-aware**: sims should complete in seconds, not minutes, so the Simulate page feels interactive. Use NumPy/SciPy vectorized operations. Profile before optimizing. Prefer data structures that don't prevent future vectorization.
- **Data pipelines are first-class**: loading, cleaning, and transforming telemetry and simulation CSVs should be reliable and repeatable. Use pandas for tabular data.
- **Docker for reproducibility**: local dev environment should be containerized. Pin Python version and all dependencies.
- **Testing**: use pytest. Validate models against known analytical solutions and recorded data. Property-based tests (hypothesis) for numerical edge cases.
- **Web/visualization**: FastAPI backend for webapp endpoints. Matplotlib/Plotly for analysis plots.

## Installed VoltAgent Subagent Packages

Marketplace: `VoltAgent/awesome-claude-code-subagents`

| Package | Version | Contents |
|---|---|---|
| `voltagent-lang` | 1.0.3 | Language specialists (includes `python-pro`) |
| `voltagent-infra` | 1.0.1 | Infrastructure/DevOps (Docker, CI/CD) |
| `voltagent-data-ai` | 1.0.2 | Data engineering, ML, analytics |

## Subagents and When to Use Them

**Always use `model: "opus"` when deploying agents.** All Agent tool calls must specify the Opus 4.6 model to ensure maximum capability and reasoning quality.

### Core workflow (use frequently)
- **`python-pro`** -- Default for all Python implementation. Use for module design, NumPy/SciPy patterns, packaging, type hints, and Pythonic idioms.
- **`architect-reviewer`** -- Use when adding a new module, changing interfaces between modules, or before any structural refactor. Ask it to review proposed module boundaries and data flow.
- **`code-reviewer`** -- Use after completing any feature branch or before merging. Focus on correctness, not style.

### Simulation and data work
- **`data-scientist`** -- Use for model validation, statistical comparison of simulation vs. telemetry, and regression analysis.
- **`data-analyst`** -- Use for exploratory analysis of telemetry CSVs and generating comparison plots.
- **`performance-engineer`** -- Use when simulation runtime matters: profiling hot loops, vectorization, memory layout.

### Quality and correctness
- **`test-automator`** -- Use when setting up pytest infrastructure, fixtures for simulation data, or parameterized test suites for model validation.
- **`qa-expert`** -- Use for test strategy decisions: what to test, coverage targets, and integration test design for multi-module simulations.
- **`debugger`** -- Use when a simulation produces wrong results and you need to trace through numerical computations or state evolution.

### Infrastructure and API
- **`fastapi-developer`** -- Use for backend webapp endpoints (e.g., the Simulate-page run endpoint).
- **Docker/infra agents (from voltagent-infra)** -- Use when setting up the dev container, CI pipeline, or reproducible simulation environments.

## Development Methodology

Follow the Superpowers workflow for all implementation:

1. Brainstorm and refine before writing code.
2. Plan in small tasks (2-5 min each) with exact file paths and verification steps.
3. TDD: write a failing test, make it pass, clean up.
4. Use git worktrees for feature branches.
5. Request code review after implementation.
6. Verify all tests pass before marking done.
````

- [ ] **Step 2: Verify content**

Run:
```bash
wc -l CLAUDE.md
grep -c "^##" CLAUDE.md
```

Expected: ~110 lines, multiple `##` headers (What This Repo Is / Data Assets / Project State / Architecture Guidance / Installed VoltAgent / Subagents / Development Methodology).

- [ ] **Step 3: Confirm no sweep/pareto/optimization references and no dead doc links**

Run:
```bash
grep -ni "sweep\|pareto\|optimization\|WEBAPP_REFOCUS\|PARALLEL_WORKSTREAMS\|ARCHITECTURE.md\|DRIVER_MODEL_FIXES_POSTMORTEM\|simulation_alignment_log\|obsidian-vault" CLAUDE.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Commit**

Run:
```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude): rewrite to simulator + 3-page webapp scope

Drops:
- Project Roadmap section referencing WEBAPP_REFOCUS_PLAN (deleted)
  and the phased parameter-sweep trajectory; replaced with a 3-line
  Project State paragraph.
- Sweep clause in the "Performance-aware" architecture bullet
  ("so a future sweep repo can reuse this core")
- "parallelization of parameter sweeps" from the performance-engineer
  subagent row
- data-scientist row mention of "designing parameter sweep
  experiments"
- data-analyst row mention of "summarizing results across simulation
  runs"

Keeps the guardrail paragraph about sweep/optimization being out of
scope for this repo. Data Assets, Vehicle Parameters, Known Issues
pointer, and Development Methodology sections are unchanged.

Part of repo-cleanup-simulator-focus.
EOF
)"
```

---

## Task 8: Final verification, push, PR

- [ ] **Step 1: Full pytest run**

Run:
```bash
pytest -v 2>&1 | tail -10
```

Expected: same pass/xfail count as Task 0 Step 4 minus the 3 tests in the deleted `test_analysis_init.py`. Zero new failures.

- [ ] **Step 2: Backend import smoke test**

Run:
```bash
python -c "import backend.main; print('backend import OK')"
```

Expected: `backend import OK`.

- [ ] **Step 3: Grep for dangling imports from deleted modules**

Run:
```bash
grep -rn "from fsae_sim.optimization\|from fsae_sim.analysis.metrics\|from fsae_sim.analysis import compute_lap_times\|from fsae_sim.analysis import compute_energy_per_lap\|from fsae_sim.analysis import compute_pareto_frontier" src/ backend/ webapp/ scripts/ tests/
```

Expected: ZERO matches.

- [ ] **Step 4: Grep for sweep/pareto mentions in kept docs**

Run:
```bash
grep -rni "sweep\|pareto" docs/ README.md CLAUDE.md
```

Expected: ZERO matches. (If matches appear only in the plan file at `docs/superpowers/plans/...`, those are fine — that folder was deleted in Task 4, so they won't exist.)

- [ ] **Step 5: Webapp build sanity check**

Run:
```bash
cd webapp
npm install
npm run build
cd ..
```

Expected: build succeeds with no errors. (Warnings about bundle size are fine.) This catches any broken relative doc links that might be referenced in the UI code or broken imports from the three pages.

- [ ] **Step 6: Inspect final working tree**

Run:
```bash
git log --oneline -8
git diff main --stat | tail -5
ls docs/
ls scripts/
ls -la | head -20
```

Expected output highlights:
- 7 cleanup commits plus main's HEAD
- `git diff main --stat` shows large deletion total, small insertion total
- `docs/` contains exactly: `SIMULATOR_ISSUES.md` (and maybe a now-empty `superpowers/` directory — git won't track the empty folder, but the filesystem may retain it)
- `scripts/` contains exactly: `clean_endurance_data.py`, `transplant_fx_coefficients.py`
- Root directory has no `.claudeignore`, no `archive/` folder

- [ ] **Step 7: Push branch**

Run:
```bash
git push -u origin chore/repo-cleanup-simulator-focus
```

Expected: branch published to origin. If the push asks about identity and none is configured, stop and ask the user to set `git config user.email` + `git config user.name` locally for this repo before pushing.

- [ ] **Step 8: Open PR**

Run:
```bash
gh pr create --title "chore: strip sweep/optimization scope; simulator + 3-page webapp only" --body "$(cat <<'EOF'
## Summary

Strips the repo to just the endurance simulator + 3-page webapp (Verification / Visualization / Simulate). Removes optimization/sweep code, the archive folder, 6 diagnostic scripts, 5 obsolete top-level docs, the obsidian vault, and all historical superpowers specs/plans. Condenses `docs/SIMULATOR_ISSUES.md`. Rewrites `README.md` and `CLAUDE.md` to the reduced scope.

Preserved explicitly: `fsae_sim.analysis.scoring` (FSAE points), `configs/ct17ev.yaml`, data-pipeline scripts.

## Scope

- 7 commits, ordered safe→risky (code-impacting first, doc rewrites last)
- ~95 files/folders removed
- ~270 lines of doc condensed to ~95 in `SIMULATOR_ISSUES.md`
- `README.md` and `CLAUDE.md` rewritten to match reduced scope

## Test plan

- [ ] `pytest -v` — same count as pre-cleanup main minus 3 (test_analysis_init.py deleted)
- [ ] `python -c "import backend.main"` — clean import
- [ ] `grep` confirms no dangling `fsae_sim.optimization` / `fsae_sim.analysis.metrics` imports
- [ ] `grep` confirms no `sweep|pareto` in kept docs
- [ ] `cd webapp && npm run build` — succeeds
- [ ] Manually check Verification page still loads (`http://localhost:5173`)
- [ ] Manually check Visualization page still loads
- [ ] Manually check Simulate page still loads (stub is expected)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL returned. Capture it and include in the handoff back to the user.

- [ ] **Step 9: Report back**

Share the PR URL with the user. Note:
- Branch: `chore/repo-cleanup-simulator-focus`
- Worktree: `../Endurance-Sim-cleanup` (can be pruned after merge with `git worktree remove`)
- Untracked spec/plan files remain in main's working tree at `docs/superpowers/specs/` and `docs/superpowers/plans/` as personal records; they're in directories that no longer exist in git.
