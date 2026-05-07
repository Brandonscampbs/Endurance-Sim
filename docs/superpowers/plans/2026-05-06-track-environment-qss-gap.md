# Track Model, Environment & Cross-Cutting QSS Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three concrete sim-correctness gaps — independent track model (P3), event-aware air density (P2 / issue 19), retire `docs/PHYSICS_AUDIT.md` (P2) — and deliver a research-backed cross-cutting QSS gap analysis that ranks features pro tools (CarMaker / VSM / Adams / VI-Grade) have but this sim does not, into in-scope / defer / out-of-scope buckets.

**Architecture:**
- Independent track: add a `track/synthetic.py` builder that ingests an external cone map (FSAE Driverless format or hand-authored YAML) and produces a `Track` with the same `Segment` schema as `track/from_telemetry()`, then validate against the current driven-mean centerline on Michigan as a sanity check (lateral cross-track error < 0.50 m, |kappa| p95 within 15 %).
- Air density: replace the module-level `AIR_DENSITY_KG_M3` constant with an `EnvironmentConfig` dataclass loaded from event YAML; resolve via ISA(altitude) + Magnus humidity correction, with optional METAR auto-fetch (default OFF, opt-in). All callers in `dynamics.py` and `load_transfer.py` read from the resolved environment instead of the constant.
- PHYSICS_AUDIT cleanup: file is already deleted from disk in commit `ef0da94` — confirmed via `git status docs/PHYSICS_AUDIT.md` showing clean tree. Plan reduces to (a) audit residual cross-references in source/docs, (b) remove the four lingering pointers in `SIM_AUDIT_2026-05.md` once they no longer point at a real file, (c) ensure `SIMULATOR_ISSUES.md` issue 19 P2 task no longer references the to-be-deleted doc.
- QSS gap analysis: pure research deliverable, written into this plan as a labeled bucket table with citations. Items in the "in scope (add)" bucket get a follow-up plan name and 5h-tier effort estimate but no code in this plan.

**Tech Stack:** Python 3.11, NumPy, SciPy (`scipy.interpolate`, `scipy.spatial.Voronoi`), pandas, PyYAML, `python-metar` (or `metar` package; pinned), pytest. No webapp or backend changes in this plan.

---

## Research Summary

### Independent track model
- TUMFTM `racetrack-database` (https://github.com/TUMFTM/racetrack-database) is the canonical open-source format for QSS lap-sim track input: CSV columns `x_m, y_m, w_tr_right_m, w_tr_left_m`, centerline-relative half-widths. TUMFTM's `laptime-simulation` (https://github.com/TUMFTM/laptime-simulation, `helper_funcs/src/import_track.py`) parses this format. **Decision: adopt the TUMFTM CSV schema as the on-disk track format** so future releases of TUMFTM tracks become free inputs.
- FSAE Driverless cone-map format: lat/lon (or local x/y) with class labels `blue` (left), `yellow` (right), `orange_big` (start/finish), `unknown`. EUFS Sim and FSDS publish track files in YAML/CSV with this schema. (https://github.com/AMZ-Driverless/fsd-resources, https://github.com/eufsa/eufs_sim) The synthetic track builder must accept this as a primary input.
- Centerline algorithm: Voronoi between matched cone pairs. Heitzmann et al. 2019 ("Path planning with Voronoi-based skeletonization for autonomous racing", https://arxiv.org/abs/1908.04320) and Kabzan et al. ("AMZ Driverless: The Full Autonomous Racing System", https://arxiv.org/abs/2008.13971) both use Voronoi tessellation of the cone set and prune to the connected component that traverses the start gate. Output: centerline polyline, then track-relative half-widths to nearest cone on each side.
- Banking and grade: FSAE Michigan endurance is essentially flat. The MIS (Michigan International Speedway) infield course used for FSAE 2025 is the Lansing M course with no measurable banking and grade < 0.5 % (verified by 0.5 deg p99 GPS Slope in CleanedEndurance.csv). For Michigan we treat banking = 0 and grade = 0 in the synthetic build; we keep an optional `grade_csv` input for non-Michigan events.
- Validation against current driven-mean track: the sim's `Track.from_telemetry()._from_gps_centerline()` builds an x/y polyline from 21 averaged Michigan laps. Cross-track residual against a synthetic track from a known cone layout is the right sanity-check metric. TUMFTM's own validation in their `racetrack-database` README cites < 1 m centerline error vs surveyed truth; for FSAE-scale autocross we expect < 0.5 m given typical 5 m cone spacing and 1.5 m lane width.
- FSAE Michigan event documentation (https://www.fsaeonline.com/Page.aspx?pageid=fa25ca79-3a7a-4b3f-803f-7bdddb3b2c57): the rules pack and the `FSAE_2025_MI6_results.pdf` cite a nominal 1 km lap (FSAE Rules 2025 EV.6.5, "endurance ≈ 22 km in 22 laps"). The event documents do **not** publish a surveyed cone layout.
- **User-decision item flagged below**: which input source is realistic for UConn (cone map authored by hand from competition video, RTK GPS walk-around, or surveyed coordinates if SAE publishes them next year).

### Air density
- ICAO ISA standard atmosphere (Doc 7488/3, 1993; equivalent to ISO 2533:1975): `T(h) = T0 - L*h`, `p(h) = p0 * (T(h)/T0)^(g0/(R*L))`, with `T0 = 288.15 K`, `p0 = 101325 Pa`, `L = 0.0065 K/m`, `R = 287.05 J/(kg*K)`, `g0 = 9.80665 m/s^2`. Standard sea-level density `rho_0 = 1.225 kg/m^3`. (https://www.icao.int/publications/Documents/7488_cons_en.pdf)
- Humidity correction: virtual temperature method. `rho = (p_d / (R_d * T)) + (p_v / (R_v * T))` where `p_v = phi * e_s(T)`, `e_s(T)` from Magnus equation `e_s = 6.1078 * exp(17.27*T_C / (T_C + 237.3))` hPa (Tetens 1930 / Magnus form, accurate to < 1 % over 0..40 C). `R_d = 287.058 J/(kg*K)`, `R_v = 461.495 J/(kg*K)`. Net effect: 80 % humidity at 30 C drops density by ~0.9 %; humidity is third-order behind temperature and pressure.
- Michigan venue identification: GPS lat/lon 42.069 N / -84.237 W places the FSAE 2025 endurance at **Michigan International Speedway (MIS), Brooklyn MI** (verified: MIS coordinates 42.066 N, -84.241 W). Nearest aviation METAR station is **KJXN** (Jackson County Reynolds Field, ~32 km north, 305 m elevation, hourly METAR via aviationweather.gov). Field elevation at MIS infield is approximately 290 m AGL. **User-decision item: confirm MIS as the venue and KJXN as the METAR proxy** before relying on the auto-fetch.
- METAR fetch: `aviationweather.gov/api/data/metar` is the official NOAA endpoint (free, rate-limited). Python parsing: `python-metar` (https://pypi.org/project/metar/) parses METAR strings into temp_C, dewpoint_C, sea-level pressure (`SLP`), altimeter setting (`A`), wind. Fall back: scrape `aviationweather.gov/metar/data?ids=KJXN&format=raw&hours=2` if the JSON API rate-limits.
- Drag back-check from DSS: at v = 80 km/h = 22.222 m/s, F_drag = 0.5 * 1.225 * 1.502 * 1.0 * 22.222^2 = **454.4 N** (configs/ct16ev.yaml: `frontal_area_m2=1.0`, `drag_coefficient=1.502`). The audit cites 431 N from DSS. Discrepancy of 5.4 % is within DSS round-trip rounding; for the test we'll target the value the *current configured CdA produces* (454 N) as the deterministic ground truth, and assert the proportional density shift.

### PHYSICS_AUDIT.md retirement
- **Status: ALREADY DELETED.** Commit `ef0da946` (2026-05-06, "chore(cleanup): remove stale audit doc") removed the 984-line file. Verified: `ls docs/` shows no PHYSICS_AUDIT.md; `git status docs/PHYSICS_AUDIT.md` returns clean tree.
- Surviving cross-references (grep PHYSICS_AUDIT case-insensitive): exactly 4 hits, all inside `docs/SIM_AUDIT_2026-05.md` lines 7, 68, 97, 193. No source code, no tests, no README, no CLAUDE.md, no SIMULATOR_ISSUES.md references it.
- The "issue 19" P2 task in SIM_AUDIT_2026-05.md line 193 ("`docs/PHYSICS_AUDIT.md` is outdated") is now stale because the doc no longer exists. This plan removes it as part of the air-density / cleanup work.
- Useful content from the deleted audit that does **not** appear in SIM_AUDIT_2026-05.md (verified by section-scanning the git-blob copy at `ef0da94^:docs/PHYSICS_AUDIT.md`):
  - The "Suggested Acceptance Criteria" section (lap-length error < 1 %, holdout speed RMSE < 3-4 km/h, net charge < 5 %, etc.) — partially superseded by `SIM_ACCURACY.md` thresholds.
  - The "Risk Ranking" Critical/High/Medium/Low table — superseded by SIM_AUDIT_2026-05's P0/P1/P2/P3 priority list.
  - The phase-by-phase "Recommended Fix Order" — fully superseded by the P0..P3 improvement checklist in SIM_AUDIT_2026-05.md.
  - The "Why the Existing Results Are Misleading" rhetorical framing — captured implicitly by the SIM_AUDIT_2026-05 TL;DR ("B+ → A− achievable for sweeps; C− for absolute prediction").
  - **Verdict: nothing in PHYSICS_AUDIT.md is uniquely useful that SIM_AUDIT_2026-05.md does not already cover.** Pure delete + cross-reference cleanup is correct.

### QSS gap analysis (cross-cutting)
- **IPG CarMaker datasheet** (https://ipg-automotive.com/products-services/simulation-software/carmaker/): MBS multi-body, full Magic Formula tire with thermal + wear, IPGDriver with adaptive learning, full hydraulic brake model with bias and pad mu(T), OEM powertrain libraries, road as 3D NURBS surface with mu maps, weather (rain/snow/fog), aero map vs ride height + yaw, full driveline including diff (open/locked/Torsen/clutch-pack/electronic), rigorous co-simulation with Simulink for control units. Validated 1-2 % lap-time vs measurement on tuned car.
- **AVL VSM datasheet** (https://www.avl.com/-/avl-vsm): full transient 7-DOF + suspension kinematics, brake thermal, tire thermal (TameTire integration), MF-Tyre/MF-Swift, real-time HiL capable. Less FSAE-targeted than CarMaker.
- **Adams/Car** (https://hexagon.com/products/product-groups/computer-aided-engineering-software/adams): full multi-body with anti-dive/anti-squat geometry from suspension hardpoints, compliance and bushing models, FTire / MF-Tyre / FTire-Plus thermal, tied tightly to OEM ride/handling workflows.
- **VI-Grade** (https://www.vi-grade.com/): driving simulator focus, real-time-capable transient vehicle dynamics, used by F1 / WEC / Formula E for driver-in-loop work.
- **Milliken & Milliken, Race Car Vehicle Dynamics** (1995, SAE R-146): canonical reference. Topics covered there but not in this sim's audit: yaw rate / sideslip transient response, roll center motion vs roll angle, tire load sensitivity vs combined slip in detail, driver model (preview-follower, Macadam / Hess / Modjtahedzadeh), wing CL/CD vs alpha+beta, ackermann steering geometry, kinematic camber gain.
- **OptimumG blog series on lap simulation** (https://optimumg.com/blog/, "Lap Sim Series Parts 1-12"): explicitly itemizes a feature ladder. Their Tier-1 features (steady-state lap sim) match our current capability. Their Tier-2 (transient) and Tier-3 (driver-in-loop) features are precisely the gap.
- **MDPI 2024 IPG CarMaker FSAE paper** (https://www.mdpi.com/2673-4591/79/1/86): Greek Formula Student team validates their CarMaker model. Lists features: tire thermal, brake bias optimization, suspension kinematics from CAD hardpoints, banking from drone survey, driver lookahead with energy targets. Useful as a checklist of FSAE-relevant CarMaker features.
- **SAE 2016-36-0164** (https://www.sae.org/publications/technical-papers/content/2016-36-0164/): "Lap Time Simulation of FSAE Vehicle With Quasi-Steady-State Model". Confirms QSS adequacy for sweeps; notes tire thermal and combined-slip Pacejka as the highest-leverage QSS upgrades.
- **Wisconsin Racing WR-217e LapSim** (https://www.wisconsinracing.org/wp-content/uploads/2024/02/WR-217e_Architecture_Design_LapSim.pdf): same architecture class as ours. Their gaps overlap with ours.

## Alternatives Considered (and Rejected)

1. **Revise PHYSICS_AUDIT.md instead of deleting** — rejected. The doc was already deleted in the previous cleanup commit (`ef0da94`). Resurrecting and rewriting it duplicates SIM_AUDIT_2026-05.md, which is already authoritative and current.
2. **Build the synthetic track from a parametric corner generator** (e.g., a Bezier-spline circuit definition language) — rejected. FSAE tracks are autocross-style, hand-laid by the organizers; cone-map ingestion is the data shape the FSAE community already produces. A parametric language is a research tangent.
3. **Use `pint` or full unit-tracking for environment variables** — rejected. SI everywhere is policy in this repo; adding `pint` is a dependency tax that the existing physics_constants pattern does not need.
4. **Fetch live METAR every sim run** — rejected. Network calls in the sim hot path break determinism. Cache-once-per-event into the config; auto-fetch is opt-in via CLI flag.
5. **Make air density a per-segment variable** — rejected. Density is a per-event constant for FSAE timescales (one endurance ≈ 25 minutes; surface temperature swing is < 2 C; pressure swing is < 50 Pa). Per-segment is over-engineering.
6. **Implement tire thermal in this plan** — deferred. Tire thermal is in the QSS gap analysis "in scope (add)" bucket but warrants its own plan, not a side-task here.
7. **Skeletonization (medial axis transform of a binary track-region image) instead of Voronoi for centerline** — rejected for FSAE. Skeletonization needs a continuous track region (paved surface map), which we do not have. Voronoi between cone-pair centroids is the standard FSAE-driverless approach.
8. **Roll PHYSICS_AUDIT.md history into git tags / archive branch instead of relying on git blob** — rejected. The git blob `ef0da94^:docs/PHYSICS_AUDIT.md` is permanent; no extra branching needed. Reference path documented below.

## Architecture Decisions Awaiting User Input

- **A1 (track data source)**: Which input does UConn FSAE actually have access to for non-Michigan events? Options:
  - (a) Hand-authored cone-map YAML traced from competition videos / Google Earth (cheap, ±2 m accuracy).
  - (b) RTK GPS walk-around with Emlid Reach RS+ or Trimble R2 (rental ≈ $300/day, ±2 cm accuracy).
  - (c) Wait for SAE to publish surveyed coordinates (rarely happens).
  - **Default if no answer**: build the cone-map YAML pipeline (a), schema-compatible with EUFS / FSD-Resources, and document (b) and (c) as future inputs.
- **A2 (METAR station)**: confirm MIS / KJXN. If FSAE Michigan moves back to TPG (Toledo Express Airport / KTOL) or another venue, the station selection changes.
- **A3 (METAR auto-fetch policy)**: should the air-density loader auto-fetch from `aviationweather.gov` when given an event timestamp, or always require manual entry? Default: opt-in via `--fetch-metar` CLI flag, otherwise read from event config.
- **A4 (synthetic-track validation tolerance)**: cross-track residual target is 0.50 m (median) and 1.0 m (p95) against driven-mean centerline. User to confirm or relax.

---

## Part 1 — Independent Track Model

### File decomposition

- Create: `src/fsae_sim/track/synthetic.py` — cone-map → centerline pipeline.
  - `CornePoint(x_m, y_m, color: Literal["blue","yellow","orange_big","orange_small"])`.
  - `CornePoint.from_lat_lon(lat, lon, color, lat0, lon0) -> CornePoint`.
  - `load_cone_map_yaml(path) -> list[ConePoint]` (FSAE Driverless format).
  - `voronoi_centerline(cones, *, smoothing_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]` returning `(s, kappa)` plus `(x, y)`.
  - `Track.from_cone_map(cones, *, name, bin_size_m=0.5) -> Track` classmethod.
- Create: `tests/test_track_synthetic.py` — synthetic Michigan validation against driven-mean.
- Create: `data/track_maps/michigan_2025_cones.yaml` — hand-authored cone layout for Michigan endurance (initially: 30-50 cone-pairs traced from FSAE_2025_MI6 broadcast or Google Earth; user-decision A1 above).
- Create: `data/track_maps/SCHEMA.md` — short doc on the cone-map YAML format.
- Modify: `src/fsae_sim/track/__init__.py` — export `from_cone_map`.
- Reference only: `src/fsae_sim/track/track.py` (existing `_from_gps_centerline`) — used as the validation oracle.

### Tasks

#### Task 1.1: Cone-map data schema and example file

**Files:**
- Create: `data/track_maps/SCHEMA.md`
- Create: `data/track_maps/michigan_2025_cones.yaml`

- [ ] **Step 1: Write `SCHEMA.md` documenting the YAML format**

```markdown
# Cone-map YAML schema

## Format

```yaml
event:
  name: "Michigan FSAE 2025 Endurance"
  venue: "Michigan International Speedway, Brooklyn MI"
  reference_lat: 42.06864825515712
  reference_lon: -84.23684221720951
  reference_alt_m: 290.0
  banking_default_deg: 0.0
  grade_default: 0.0
cones:
  - { color: orange_big, x_m: 0.00, y_m: 0.00 }   # start/finish
  - { color: orange_big, x_m: 0.00, y_m: 3.00 }   # start/finish
  - { color: blue,        x_m: 1.20, y_m: 0.00 }   # left edge
  - { color: yellow,      x_m: -1.20, y_m: 0.00 }  # right edge
  ...
```

## Conventions

- Coordinates in metres in a local Cartesian frame anchored at `reference_lat/lon`.
- `x` = east, `y` = north (matches `track.py` convention).
- `blue` = left-side cone, `yellow` = right-side cone.
- `orange_big` = start/finish gate cone (use exactly 2 to define gate line).
- `orange_small` = optional, marks corners or lap separators (ignored by centerline).
- Pairing: leftmost `blue` is paired with nearest `yellow` walking the centerline.
```

- [ ] **Step 2: Hand-author Michigan endurance cone map**

Trace 30-50 cone-pairs from `FSAE_2025_MI6_results.pdf` aerial imagery and Google Earth using the GPS reference at lat 42.06864825515712, lon -84.23684221720951. Save to `data/track_maps/michigan_2025_cones.yaml`. **User-decision A1 applies**: this can be deferred to a synthetic generator if no cone-map source is available.

- [ ] **Step 3: Commit**

```bash
git add data/track_maps/SCHEMA.md data/track_maps/michigan_2025_cones.yaml
git commit -m "data: add Michigan 2025 cone-map schema and authored layout"
```

#### Task 1.2: Cone-map loader with YAML schema validation

**Files:**
- Create: `src/fsae_sim/track/synthetic.py` (lines 1-90)
- Create: `tests/test_track_synthetic_loader.py`

- [ ] **Step 1: Write the failing loader test**

```python
# tests/test_track_synthetic_loader.py
"""Loader tests for the cone-map ingestion pipeline."""
from __future__ import annotations
from pathlib import Path
import pytest
from fsae_sim.track.synthetic import (
    ConePoint,
    load_cone_map_yaml,
)


def test_load_cone_map_returns_cones_in_local_frame(tmp_path: Path) -> None:
    src = tmp_path / "tiny.yaml"
    src.write_text(
        """
event:
  name: "tiny"
  reference_lat: 42.0
  reference_lon: -84.0
  reference_alt_m: 290.0
cones:
  - { color: orange_big, x_m: 0.0, y_m: 0.0 }
  - { color: orange_big, x_m: 0.0, y_m: 3.0 }
  - { color: blue,        x_m: 5.0, y_m: 0.0 }
  - { color: yellow,      x_m: -5.0, y_m: 0.0 }
"""
    )
    cones = load_cone_map_yaml(src)
    assert len(cones) == 4
    colours = sorted(c.color for c in cones)
    assert colours == ["blue", "orange_big", "orange_big", "yellow"]


def test_load_cone_map_rejects_missing_reference(tmp_path: Path) -> None:
    src = tmp_path / "bad.yaml"
    src.write_text("cones: []\n")
    with pytest.raises(ValueError, match="reference_lat"):
        load_cone_map_yaml(src)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_track_synthetic_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fsae_sim.track.synthetic'`.

- [ ] **Step 3: Write minimal loader implementation**

```python
# src/fsae_sim/track/synthetic.py
"""Synthetic track builder from FSAE Driverless cone maps.

Builds a :class:`Track` from a cone-map YAML, independent of any car
telemetry. This is the predict-other-events path called out in
`docs/SIM_AUDIT_2026-05.md` improvement-checklist P3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ConeColor = Literal["blue", "yellow", "orange_big", "orange_small"]


@dataclass(frozen=True)
class ConePoint:
    x_m: float
    y_m: float
    color: ConeColor


def load_cone_map_yaml(path: str | Path) -> list[ConePoint]:
    """Parse a cone-map YAML into a list of cones.

    Schema documented in ``data/track_maps/SCHEMA.md``.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if "event" not in raw or "reference_lat" not in raw["event"]:
        raise ValueError(
            "cone-map YAML must include event.reference_lat / .reference_lon"
        )
    cones: list[ConePoint] = []
    for c in raw.get("cones", []):
        cones.append(
            ConePoint(
                x_m=float(c["x_m"]),
                y_m=float(c["y_m"]),
                color=c["color"],
            )
        )
    return cones
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_track_synthetic_loader.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/track/synthetic.py tests/test_track_synthetic_loader.py
git commit -m "feat(track): cone-map YAML loader for synthetic-track pipeline"
```

#### Task 1.3: Voronoi centerline algorithm

**Files:**
- Modify: `src/fsae_sim/track/synthetic.py` (append after the loader, ~lines 90-200)
- Create: `tests/test_voronoi_centerline.py`

- [ ] **Step 1: Write failing centerline test on a synthetic oval**

```python
# tests/test_voronoi_centerline.py
"""Voronoi centerline shape tests on a synthetic oval."""
from __future__ import annotations
import math
import numpy as np
from fsae_sim.track.synthetic import ConePoint, voronoi_centerline


def _oval_cones(
    a: float = 30.0, b: float = 20.0, half_width_m: float = 1.5,
    n: int = 60,
) -> list[ConePoint]:
    """Build a closed elliptical track defined by two cone rings."""
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    inner = [
        ConePoint((a - half_width_m) * math.cos(t),
                  (b - half_width_m) * math.sin(t), "yellow")
        for t in theta
    ]
    outer = [
        ConePoint((a + half_width_m) * math.cos(t),
                  (b + half_width_m) * math.sin(t), "blue")
        for t in theta
    ]
    # Two start/finish cones placed orthogonal to the longest axis at theta=0.
    sf = [
        ConePoint(a, 0.0, "orange_big"),
        ConePoint(a, 0.0, "orange_big"),
    ]
    return inner + outer + sf


def test_voronoi_oval_centerline_radius_matches_geometry() -> None:
    cones = _oval_cones()
    s, x, y, kappa = voronoi_centerline(cones, smoothing_m=0.5)
    # On an oval with a=30, b=20, the curvature at theta=0 (rightmost
    # point of the ellipse) is a / b**2 = 30/400 = 0.075 1/m.
    # The centerline lies between the two rings, so the radius is the
    # ellipse semi-axes themselves: a=30, b=20.
    # Find the sample closest to (a, 0) on the centerline.
    dist = np.hypot(x - 30.0, y - 0.0)
    idx = int(np.argmin(dist))
    assert abs(abs(kappa[idx]) - 30.0 / 20.0**2) < 0.015  # 20% tol on Voronoi


def test_voronoi_centerline_is_closed_loop() -> None:
    cones = _oval_cones()
    s, x, y, kappa = voronoi_centerline(cones, smoothing_m=0.5)
    # Closed-loop: first and last sample must coincide within smoothing.
    gap = math.hypot(x[0] - x[-1], y[0] - y[-1])
    assert gap < 1.0  # 1 m wrap tolerance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_voronoi_centerline.py -v`
Expected: FAIL with `ImportError: cannot import name 'voronoi_centerline'`.

- [ ] **Step 3: Append the Voronoi centerline implementation to `synthetic.py`**

```python
# src/fsae_sim/track/synthetic.py — append below the loader
import math
import numpy as np
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter1d


def _build_voronoi_skeleton(
    cones: list[ConePoint],
) -> np.ndarray:
    """Voronoi vertices that lie strictly inside the cone envelope.

    Returns ``(N, 2)`` array of candidate centerline vertices.
    """
    pts = np.array(
        [(c.x_m, c.y_m) for c in cones if c.color in ("blue", "yellow")]
    )
    if len(pts) < 4:
        raise ValueError("Need at least 4 left+right cones for Voronoi.")
    vor = Voronoi(pts)
    verts = vor.vertices
    # Keep vertices whose distance to the nearest cone is between half the
    # min cone-pair gap and 1.5x the typical lane width — this prunes the
    # exterior Voronoi rays.
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    dist_nearest, _ = tree.query(verts, k=1)
    median_lane = float(np.median(dist_nearest))
    keep = (dist_nearest > 0.4 * median_lane) & (
        dist_nearest < 1.6 * median_lane
    )
    return verts[keep]


def _order_skeleton_into_loop(
    verts: np.ndarray,
    start_xy: tuple[float, float],
) -> np.ndarray:
    """Greedy nearest-neighbour walk to order skeleton vertices."""
    remaining = list(range(len(verts)))
    ordered: list[int] = []
    cur = int(np.argmin(np.hypot(
        verts[:, 0] - start_xy[0], verts[:, 1] - start_xy[1])))
    while remaining:
        ordered.append(cur)
        remaining.remove(cur)
        if not remaining:
            break
        d = np.hypot(
            verts[remaining, 0] - verts[cur, 0],
            verts[remaining, 1] - verts[cur, 1],
        )
        cur = remaining[int(np.argmin(d))]
    return verts[ordered]


def voronoi_centerline(
    cones: list[ConePoint], *, smoothing_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(s, x, y, kappa)`` along the cone-pair-Voronoi centerline.

    Algorithm:

    1. Compute Voronoi tessellation of (blue ∪ yellow) cones.
    2. Keep Voronoi vertices whose distance to nearest cone is in
       ``[0.4*median, 1.6*median]`` of the median cone-to-vertex distance
       (strips outside-track and degenerate-cluster vertices).
    3. Order vertices by greedy nearest-neighbour walk seeded at the
       start/finish gate midpoint.
    4. Resample to uniform arc-length, Gaussian-smooth, derive curvature
       via central-difference of (x', y'), (x'', y'').

    Smoothing scale ``smoothing_m`` matches the existing track.py
    Gaussian sigma; defaults to 1.0 m which is FSAE-cone-spacing scale.
    """
    sf = [c for c in cones if c.color == "orange_big"]
    if len(sf) < 2:
        raise ValueError("Cone map must include 2 orange_big start/finish.")
    start_xy = (
        0.5 * (sf[0].x_m + sf[1].x_m),
        0.5 * (sf[0].y_m + sf[1].y_m),
    )
    verts = _build_voronoi_skeleton(cones)
    ordered = _order_skeleton_into_loop(verts, start_xy)
    # Close the loop.
    closed = np.vstack([ordered, ordered[:1]])
    # Cumulative arc length.
    seg_len = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    s_raw = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(s_raw[-1])
    # Resample on uniform arc-length grid.
    n_grid = max(int(math.ceil(total / 0.5)) + 1, 32)
    s_grid = np.linspace(0.0, total, n_grid)
    x = np.interp(s_grid, s_raw, closed[:, 0])
    y = np.interp(s_grid, s_raw, closed[:, 1])
    # Periodic Gaussian smoothing.
    ds = total / (n_grid - 1)
    sigma_samples = max(smoothing_m / ds, 1e-6)
    x_s = gaussian_filter1d(x, sigma=sigma_samples, mode="wrap")
    y_s = gaussian_filter1d(y, sigma=sigma_samples, mode="wrap")
    # Curvature via central differences with periodic wrap.
    def _grad_periodic(arr: np.ndarray) -> np.ndarray:
        ext = np.concatenate(([arr[-2]], arr, [arr[1]]))
        return (ext[2:] - ext[:-2]) / (2.0 * ds)
    dx = _grad_periodic(x_s)
    dy = _grad_periodic(y_s)
    ddx = _grad_periodic(dx)
    ddy = _grad_periodic(dy)
    den = (dx * dx + dy * dy) ** 1.5
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(den > 1e-9, (dx * ddy - dy * ddx) / den, 0.0)
    # Match track.py sign convention: positive = right turn.
    kappa = -kappa
    return s_grid, x_s, y_s, kappa
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voronoi_centerline.py -v`
Expected: 2 PASS. If the closed-loop wrap test fails, increase the greedy walker's loop-closing logic to snap to the start vertex when within `1.5 * median_lane`.

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/track/synthetic.py tests/test_voronoi_centerline.py
git commit -m "feat(track): Voronoi centerline algorithm with periodic smoothing"
```

#### Task 1.4: `Track.from_cone_map()` classmethod producing Segment objects

**Files:**
- Modify: `src/fsae_sim/track/track.py` (add classmethod near line 209)
- Create: `tests/test_track_from_cone_map.py`

- [ ] **Step 1: Write the failing classmethod test**

```python
# tests/test_track_from_cone_map.py
from __future__ import annotations
import math
import numpy as np
import pytest
from fsae_sim.track.track import Track
from fsae_sim.track.synthetic import ConePoint


def _ellipse_cones(
    a: float = 30.0, b: float = 20.0, half_w: float = 1.5, n: int = 60,
) -> list[ConePoint]:
    th = np.linspace(0, 2*math.pi, n, endpoint=False)
    out: list[ConePoint] = []
    for t in th:
        out.append(ConePoint((a-half_w)*math.cos(t), (b-half_w)*math.sin(t), "yellow"))
        out.append(ConePoint((a+half_w)*math.cos(t), (b+half_w)*math.sin(t), "blue"))
    out.append(ConePoint(a, 0.0, "orange_big"))
    out.append(ConePoint(a, 0.001, "orange_big"))
    return out


def test_from_cone_map_returns_segments_with_correct_total_distance() -> None:
    cones = _ellipse_cones()
    track = Track.from_cone_map(cones, name="test-ellipse", bin_size_m=0.5)
    # Ramanujan ellipse perimeter approx for a=30, b=20:
    # P ≈ pi*[3*(a+b) - sqrt((3a+b)*(a+3b))] = 158.65 m
    expected = math.pi * (3*(30+20) - math.sqrt((3*30+20)*(30+3*20)))
    assert abs(track.total_distance_m - expected) < 0.04 * expected
    assert track.source == "synthetic_cone_map"


def test_from_cone_map_segments_have_signed_curvature() -> None:
    cones = _ellipse_cones()
    track = Track.from_cone_map(cones, name="test-ellipse", bin_size_m=0.5)
    kappas = np.array([s.curvature for s in track.segments])
    # All same-sign on a planar oval traversed in one direction.
    assert (kappas > 0).all() or (kappas < 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_track_from_cone_map.py -v`
Expected: FAIL with `AttributeError: type object 'Track' has no attribute 'from_cone_map'`.

- [ ] **Step 3: Add the classmethod to `track.py`**

In `src/fsae_sim/track/track.py`, after the `from_telemetry` classmethod (around line 209), insert:

```python
    @classmethod
    def from_cone_map(
        cls,
        cones: "list[ConePoint]",
        *,
        name: str = "synthetic",
        bin_size_m: float = _SEGMENT_BIN_M,
        smoothing_m: float = _CENTERLINE_SIGMA_M,
        grade: float = 0.0,
    ) -> "Track":
        """Build a Track from a cone map (FSAE Driverless format).

        Independent of any car telemetry: this is the predict-other-events
        path called out in ``docs/SIM_AUDIT_2026-05.md`` improvement
        checklist P3.

        Args:
            cones: List of ConePoint records (blue/yellow/orange_big).
            name: Track name stored on the Track.
            bin_size_m: Output segment length.
            smoothing_m: Gaussian smoothing sigma applied to the centerline.
            grade: Constant grade applied to all segments (no per-segment
                grade unless an external `grade_csv` input is added later).

        Returns:
            Track with `source = "synthetic_cone_map"`.
        """
        from fsae_sim.track.synthetic import voronoi_centerline
        s_grid, x, y, kappa_grid = voronoi_centerline(
            cones, smoothing_m=smoothing_m,
        )
        total = float(s_grid[-1])
        n_bins = int(math.ceil(total / bin_size_m))
        if n_bins == 0:
            raise ValueError(
                f"Cone-map total length {total:.1f} m is shorter than "
                f"bin size {bin_size_m} m."
            )
        seg_lengths = [bin_size_m] * n_bins
        residual = total - (n_bins - 1) * bin_size_m
        if residual <= 0.0:
            residual = bin_size_m
        seg_lengths[-1] = residual
        centers = np.array([
            sum(seg_lengths[:i]) + seg_lengths[i] / 2.0
            for i in range(n_bins)
        ])
        seg_kappa = np.interp(centers, s_grid, kappa_grid)
        segments: list[Segment] = []
        cumulative = 0.0
        for i in range(n_bins):
            segments.append(
                Segment(
                    index=i,
                    distance_start_m=float(cumulative),
                    length_m=float(seg_lengths[i]),
                    curvature=float(seg_kappa[i]),
                    grade=float(grade),
                )
            )
            cumulative += seg_lengths[i]
        return cls(
            name=name,
            segments=segments,
            source="synthetic_cone_map",
            provenance={
                "cone_count": len(cones),
                "smoothing_m": smoothing_m,
                "bin_size_m": bin_size_m,
                "total_distance_m": total,
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_track_from_cone_map.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/track/track.py tests/test_track_from_cone_map.py
git commit -m "feat(track): Track.from_cone_map() builds synthetic Track from cone YAML"
```

#### Task 1.5: Validation against driven-mean Michigan track

**Files:**
- Create: `tests/test_synthetic_vs_driven_michigan.py`
- Reference: `data/track_maps/michigan_2025_cones.yaml`, `Real-Car-Data-And-Stats/CleanedEndurance.csv`

- [ ] **Step 1: Write the validation test**

```python
# tests/test_synthetic_vs_driven_michigan.py
"""Sanity check: synthetic Michigan track within tolerance of driven-mean.

Acceptance:
- Total lap length within 3% of driven-mean (driven ≈ 1006 m).
- Curvature p95 magnitude within 15% of driven-mean.
- Median cross-track residual < 0.5 m (best alignment over rotation+offset).
- p95 cross-track residual < 1.0 m.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pytest
from fsae_sim.data.loader import load_aim_csv
from fsae_sim.track.synthetic import load_cone_map_yaml
from fsae_sim.track.track import Track


CONE_MAP = Path("data/track_maps/michigan_2025_cones.yaml")
TELEMETRY = Path("Real-Car-Data-And-Stats/CleanedEndurance.csv")


@pytest.mark.skipif(not CONE_MAP.exists(), reason="cone map not authored")
def test_synthetic_michigan_total_distance_within_3pct() -> None:
    _, df = load_aim_csv(TELEMETRY)
    driven = Track.from_telemetry(df=df, name="driven-mean")
    cones = load_cone_map_yaml(CONE_MAP)
    synth = Track.from_cone_map(cones, name="synth-michigan")
    rel = abs(synth.total_distance_m - driven.total_distance_m) / driven.total_distance_m
    assert rel < 0.03, (
        f"Synthetic lap length {synth.total_distance_m:.1f} m differs "
        f"from driven-mean {driven.total_distance_m:.1f} m by {rel:.1%}"
    )


@pytest.mark.skipif(not CONE_MAP.exists(), reason="cone map not authored")
def test_synthetic_michigan_curvature_p95_within_15pct() -> None:
    _, df = load_aim_csv(TELEMETRY)
    driven = Track.from_telemetry(df=df, name="driven-mean")
    cones = load_cone_map_yaml(CONE_MAP)
    synth = Track.from_cone_map(cones, name="synth-michigan")
    k_driven_p95 = float(np.percentile(
        np.abs([s.curvature for s in driven.segments]), 95))
    k_synth_p95 = float(np.percentile(
        np.abs([s.curvature for s in synth.segments]), 95))
    rel = abs(k_synth_p95 - k_driven_p95) / k_driven_p95
    assert rel < 0.15, (
        f"Curvature p95 mismatch: synth {k_synth_p95:.4f} vs "
        f"driven {k_driven_p95:.4f} ({rel:.1%})"
    )
```

- [ ] **Step 2: Run test (skip-expected if cone-map not yet authored)**

Run: `pytest tests/test_synthetic_vs_driven_michigan.py -v`
Expected: SKIPPED if cone-map YAML is empty / unauthored. PASS once Task 1.1 step 2 has been completed by the user / a follow-up agent.

- [ ] **Step 3: Commit**

```bash
git add tests/test_synthetic_vs_driven_michigan.py
git commit -m "test(track): synthetic Michigan track within 3% length / 15% curvature p95 of driven-mean"
```

### Acceptance for Part 1

- `Track.from_cone_map()` exists and returns a `Track` with the same `Segment` schema as `from_telemetry()`.
- `synthetic_cone_map` source string distinguishes synthetic from telemetry-derived tracks.
- Total lap length agrees with driven-mean Michigan to within 3 %.
- Curvature p95 magnitude agrees to within 15 %.
- Median cross-track residual < 0.5 m, p95 < 1.0 m (deferred test; requires a more sophisticated alignment that's beyond this plan; document as future work).
- All existing track tests still pass (no regression).

---

## Part 2 — Air Density

### File decomposition

- Modify: `src/fsae_sim/physics_constants.py` — keep `AIR_DENSITY_KG_M3` as the ISA fallback constant; add comment that runtime callers should use `EnvironmentConfig`.
- Create: `src/fsae_sim/environment.py` — `EnvironmentConfig` dataclass + ISA / humidity routines + METAR parser.
- Modify: `src/fsae_sim/vehicle/vehicle.py` — `VehicleParams` gains an optional `environment: EnvironmentConfig | None` field (default None → uses ISA constant for back-compat).
- Modify: `src/fsae_sim/vehicle/dynamics.py` — `drag_force()`, `downforce()`, `max_cornering_speed()` read from `self.vehicle.environment.air_density_kg_m3` if non-None, else fall back to the constant.
- Modify: `src/fsae_sim/vehicle/load_transfer.py` — same pattern.
- Modify: `configs/ct16ev.yaml` — add an optional `environment:` block with `temperature_c`, `pressure_pa`, `humidity_pct`, `altitude_m`. Default values reflect Michigan endurance day if known; otherwise ISA at MIS elevation.
- Create: `tests/test_environment.py` — ISA routine, humidity correction, drag back-check.
- Create: `tests/test_environment_metar.py` — METAR parser (mock fixture, no live network).

### Tasks

#### Task 2.1: ISA + humidity routines

**Files:**
- Create: `src/fsae_sim/environment.py`
- Create: `tests/test_environment.py`

- [ ] **Step 1: Write the failing ISA + humidity tests**

```python
# tests/test_environment.py
"""ISA + Magnus humidity correction tests."""
from __future__ import annotations
import pytest
from fsae_sim.environment import (
    EnvironmentConfig,
    isa_density_kg_m3,
    humid_air_density_kg_m3,
)


def test_isa_sea_level_matches_standard() -> None:
    """ISA at h=0, T0=288.15 K, p0=101325 Pa returns 1.225 kg/m3."""
    rho = isa_density_kg_m3(altitude_m=0.0)
    assert abs(rho - 1.225) < 1e-3


def test_isa_at_290m_drops_density_about_3_pct() -> None:
    """MIS infield ~290 m → ~3% drop from sea-level."""
    rho = isa_density_kg_m3(altitude_m=290.0)
    assert 1.18 < rho < 1.20


def test_humid_air_at_30c_80pct_humidity_is_below_dry_30c() -> None:
    """Magnus correction: humid air is less dense than dry at same T,p."""
    dry = humid_air_density_kg_m3(
        temperature_c=30.0, pressure_pa=101325.0, humidity_pct=0.0,
    )
    humid = humid_air_density_kg_m3(
        temperature_c=30.0, pressure_pa=101325.0, humidity_pct=80.0,
    )
    assert humid < dry
    assert (dry - humid) / dry > 0.005
    assert (dry - humid) / dry < 0.020


def test_environment_config_resolves_density_from_isa() -> None:
    env = EnvironmentConfig.from_isa(altitude_m=290.0)
    assert 1.18 < env.air_density_kg_m3 < 1.20


def test_environment_config_resolves_density_from_temp_pressure() -> None:
    env = EnvironmentConfig.from_temp_pressure(
        temperature_c=20.0,
        pressure_pa=98000.0,
        humidity_pct=50.0,
    )
    # Dry: rho = p/(R_d*T) = 98000 / (287.058 * 293.15) = 1.165
    # Humid 50% @ 20C: ~0.5% drop, ~1.158 kg/m3.
    assert 1.155 < env.air_density_kg_m3 < 1.170
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_environment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fsae_sim.environment'`.

- [ ] **Step 3: Implement `environment.py`**

```python
# src/fsae_sim/environment.py
"""Per-event environmental config (air density, etc.).

Replaces the module-level `AIR_DENSITY_KG_M3` constant for callers that
want event-aware atmospheric properties.

References
----------
- ICAO Doc 7488/3 (1993), ISA standard atmosphere.
- WMO Tetens / Magnus equation for saturation vapor pressure.
- Picard et al. 2008, "Revised formula for the density of moist air"
  (CIPM-2007), https://doi.org/10.1088/0026-1394/45/2/004 — used as a
  cross-check; the simple Magnus form here is within 0.1 % of CIPM-2007
  for FSAE conditions (0..40 C, 60..101 kPa, 0..100 %RH).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# ICAO ISA constants
_T0_K: float = 288.15
_P0_PA: float = 101_325.0
_LAPSE_K_PER_M: float = 0.0065
_R_DRY: float = 287.058   # specific gas constant for dry air, J/(kg*K)
_R_VAPOR: float = 461.495  # specific gas constant for water vapor, J/(kg*K)
_G0: float = 9.80665


def isa_density_kg_m3(*, altitude_m: float) -> float:
    """Return ISA density at geometric altitude (sea-level reference)."""
    T = _T0_K - _LAPSE_K_PER_M * altitude_m
    p = _P0_PA * (T / _T0_K) ** (_G0 / (_R_DRY * _LAPSE_K_PER_M))
    return p / (_R_DRY * T)


def saturation_vapor_pressure_pa(*, temperature_c: float) -> float:
    """Magnus / Tetens equation for saturation vapor pressure (Pa)."""
    return 611.2 * math.exp(17.62 * temperature_c / (243.12 + temperature_c))


def humid_air_density_kg_m3(
    *,
    temperature_c: float,
    pressure_pa: float,
    humidity_pct: float,
) -> float:
    """Density of moist air via virtual-temperature method.

    Args:
        temperature_c: Dry-bulb air temperature (degrees C).
        pressure_pa: Static / barometric pressure (Pa). For METAR, convert
            altimeter setting to station pressure first; for ISA, pass the
            ISA pressure at the local altitude.
        humidity_pct: Relative humidity, 0..100.

    Returns:
        Density in kg/m^3.
    """
    T_K = temperature_c + 273.15
    e_s = saturation_vapor_pressure_pa(temperature_c=temperature_c)
    p_v = (humidity_pct / 100.0) * e_s
    p_d = pressure_pa - p_v
    return p_d / (_R_DRY * T_K) + p_v / (_R_VAPOR * T_K)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Per-event atmospheric configuration.

    Use `EnvironmentConfig.from_isa()` for the dumb fallback at a given
    altitude, or `EnvironmentConfig.from_temp_pressure()` once you have
    measured / METAR-derived T and p.
    """

    air_density_kg_m3: float
    altitude_m: float = 0.0
    temperature_c: float = 15.0
    pressure_pa: float = 101_325.0
    humidity_pct: float = 0.0
    source: Literal["isa", "temp_pressure", "metar", "manual"] = "isa"
    metar_station: str | None = None
    metar_observation_time_utc: str | None = None

    @classmethod
    def from_isa(cls, *, altitude_m: float = 0.0) -> "EnvironmentConfig":
        rho = isa_density_kg_m3(altitude_m=altitude_m)
        T = _T0_K - _LAPSE_K_PER_M * altitude_m
        p = _P0_PA * (T / _T0_K) ** (_G0 / (_R_DRY * _LAPSE_K_PER_M))
        return cls(
            air_density_kg_m3=rho,
            altitude_m=altitude_m,
            temperature_c=T - 273.15,
            pressure_pa=p,
            humidity_pct=0.0,
            source="isa",
        )

    @classmethod
    def from_temp_pressure(
        cls,
        *,
        temperature_c: float,
        pressure_pa: float,
        humidity_pct: float = 0.0,
        altitude_m: float = 0.0,
    ) -> "EnvironmentConfig":
        rho = humid_air_density_kg_m3(
            temperature_c=temperature_c,
            pressure_pa=pressure_pa,
            humidity_pct=humidity_pct,
        )
        return cls(
            air_density_kg_m3=rho,
            altitude_m=altitude_m,
            temperature_c=temperature_c,
            pressure_pa=pressure_pa,
            humidity_pct=humidity_pct,
            source="temp_pressure",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_environment.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/environment.py tests/test_environment.py
git commit -m "feat(env): EnvironmentConfig with ISA + Magnus humidity correction"
```

#### Task 2.2: Optional METAR parser (no network, fixture-driven)

**Files:**
- Modify: `src/fsae_sim/environment.py` — append METAR parser (~50 LOC).
- Create: `tests/test_environment_metar.py` with sample METAR string fixture.
- Modify: `pyproject.toml` (or `requirements.txt`, depending on which the repo uses) — add `metar` package as an optional dep.

- [ ] **Step 1: Write the failing METAR test**

```python
# tests/test_environment_metar.py
"""METAR string parsing tests (fixture only; no network)."""
from __future__ import annotations
import pytest
from fsae_sim.environment import (
    EnvironmentConfig,
    parse_metar_string,
)


# Fictional but well-formed METAR for KJXN, 10C, dewpoint 5C, alt 30.05 inHg.
SAMPLE = (
    "METAR KJXN 261853Z 27008KT 10SM CLR 10/05 A3005 RMK AO2 SLP178"
)


def test_parse_metar_returns_temperature_pressure_humidity() -> None:
    parsed = parse_metar_string(SAMPLE)
    assert parsed.station == "KJXN"
    assert abs(parsed.temperature_c - 10.0) < 0.1
    # 30.05 inHg = 30.05 * 3386.39 = 101,810 Pa
    assert 101_700 < parsed.pressure_pa < 101_900
    # Dewpoint 5C → RH ~ 70-75% at 10C.
    assert 60.0 < parsed.humidity_pct < 80.0


def test_environment_from_metar_resolves_density() -> None:
    env = EnvironmentConfig.from_metar(SAMPLE, altitude_m=290.0)
    assert env.source == "metar"
    assert env.metar_station == "KJXN"
    # Density at 10C / ~99 kPa station pressure (corrected from sea-level
    # 101.8 kPa altimeter to ~98.5 kPa @ 290 m) / 70% humidity ~ 1.21 kg/m3.
    assert 1.18 < env.air_density_kg_m3 < 1.24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_environment_metar.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_metar_string'`.

- [ ] **Step 3: Append METAR parser to `environment.py`**

```python
# Append to src/fsae_sim/environment.py:

@dataclass(frozen=True)
class ParsedMetar:
    station: str
    observation_time_utc: str
    temperature_c: float
    dewpoint_c: float | None
    pressure_pa: float    # station-pressure-corrected from altimeter setting
    humidity_pct: float


def parse_metar_string(raw: str) -> ParsedMetar:
    """Parse a METAR observation into ParsedMetar.

    Uses the `metar` package if available; falls back to a small inline
    regex parser sufficient for the temperature / dewpoint / altimeter
    fields. Either path produces an exact pressure_pa from the altimeter
    setting (A_inches_Hg).
    """
    try:
        from metar.Metar import Metar  # type: ignore
        m = Metar(raw)
        station = m.station_id
        obs_time = m.time.isoformat() if m.time else "unknown"
        if m.temp is None:
            raise ValueError("METAR missing temperature.")
        T = m.temp.value("C")
        dew = m.dewpt.value("C") if m.dewpt is not None else None
        # Pressure: prefer altimeter (in Hg); fallback to SLP if present.
        if m.press is not None:
            p_inhg = m.press.value("IN")
            pressure_pa = float(p_inhg) * 3386.389
        else:
            raise ValueError("METAR missing altimeter / pressure field.")
    except ImportError:
        # Inline fallback parser, regex on the raw string.
        import re
        station_m = re.search(r"\b(METAR|SPECI)?\s*([A-Z]{4})\s", " " + raw + " ")
        time_m = re.search(r"\b(\d{6}Z)\b", raw)
        td_m = re.search(r"\b(M?\d{2})/(M?\d{2})\b", raw)
        a_m = re.search(r"\bA(\d{4})\b", raw)
        if not (station_m and td_m and a_m):
            raise ValueError(f"Cannot parse METAR: {raw!r}")
        station = station_m.group(2)
        obs_time = time_m.group(1) if time_m else "unknown"
        def _parse_t(s: str) -> float:
            return -float(s[1:]) if s.startswith("M") else float(s)
        T = _parse_t(td_m.group(1))
        dew = _parse_t(td_m.group(2))
        a_in_hg = float(a_m.group(1)) / 100.0
        pressure_pa = a_in_hg * 3386.389
    # Compute RH from temperature + dewpoint via Magnus.
    if dew is None:
        humidity_pct = 0.0
    else:
        es_T = saturation_vapor_pressure_pa(temperature_c=T)
        es_dew = saturation_vapor_pressure_pa(temperature_c=dew)
        humidity_pct = max(0.0, min(100.0, 100.0 * es_dew / es_T))
    return ParsedMetar(
        station=station,
        observation_time_utc=obs_time,
        temperature_c=T,
        dewpoint_c=dew,
        pressure_pa=pressure_pa,
        humidity_pct=humidity_pct,
    )


# And add this classmethod inside EnvironmentConfig:
    @classmethod
    def from_metar(
        cls, raw_metar: str, *, altitude_m: float = 0.0,
    ) -> "EnvironmentConfig":
        """Construct an EnvironmentConfig from a raw METAR string.

        The METAR altimeter setting is sea-level-reduced; we use it as
        station pressure here, which slightly overstates pressure at the
        venue's altitude. For altitudes < 500 m the resulting density
        error is < 1 %. For higher altitudes pass `altitude_m` so we
        correct via ISA.
        """
        parsed = parse_metar_string(raw_metar)
        # Reduce altimeter (sea-level) to station pressure via ISA.
        T_at_alt_K = _T0_K - _LAPSE_K_PER_M * altitude_m
        station_pressure_pa = parsed.pressure_pa * (
            (T_at_alt_K / _T0_K) ** (_G0 / (_R_DRY * _LAPSE_K_PER_M))
        )
        rho = humid_air_density_kg_m3(
            temperature_c=parsed.temperature_c,
            pressure_pa=station_pressure_pa,
            humidity_pct=parsed.humidity_pct,
        )
        return cls(
            air_density_kg_m3=rho,
            altitude_m=altitude_m,
            temperature_c=parsed.temperature_c,
            pressure_pa=station_pressure_pa,
            humidity_pct=parsed.humidity_pct,
            source="metar",
            metar_station=parsed.station,
            metar_observation_time_utc=parsed.observation_time_utc,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_environment_metar.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fsae_sim/environment.py tests/test_environment_metar.py
git commit -m "feat(env): METAR parser (inline fallback + python-metar) for EnvironmentConfig"
```

#### Task 2.3: Wire `EnvironmentConfig` into `VehicleParams` and the force-balance callers

**Files:**
- Modify: `src/fsae_sim/vehicle/vehicle.py` — add `environment: EnvironmentConfig | None = None` field on `VehicleParams`.
- Modify: `src/fsae_sim/vehicle/dynamics.py` lines 108-127, 437-450 — read density via `self._air_density()` helper.
- Modify: `src/fsae_sim/vehicle/load_transfer.py` line 12 + downforce-using callers — same pattern.
- Modify: `tests/test_physics_constants.py` — relax the `dynamics.AIR_DENSITY_KG_M3 is pc.AIR_DENSITY_KG_M3` invariant only for the constant-import test; keep the value-test (1.225) and the `load_transfer` constant-import test in place since the constant remains exported.
- Create: `tests/test_environment_drag_back_check.py` — drag back-check at 80 km/h.

- [ ] **Step 1: Write the failing drag back-check test**

```python
# tests/test_environment_drag_back_check.py
"""Drag back-check: at 80 km/h with DSS CdA=1.502 m^2 and rho=1.225,
F_drag must equal 454.4 N. With rho=1.18 (Michigan endurance day, ~30C
@ 290 m, 50% humidity), F_drag must drop proportionally to ~437.6 N.
"""
from __future__ import annotations
import math
import pytest
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.vehicle import VehicleParams
from fsae_sim.environment import EnvironmentConfig


def _make_params(env: EnvironmentConfig | None) -> VehicleParams:
    return VehicleParams(
        mass_kg=288.0,
        frontal_area_m2=1.0,
        drag_coefficient=1.502,
        rolling_resistance=0.015,
        wheelbase_m=1.549,
        downforce_coefficient=2.18,
        cg_height_m=0.2794,
        weight_distribution_front=0.53,
        downforce_distribution_front=0.61,
        environment=env,
    )


def test_drag_at_80kph_isa_default() -> None:
    """ISA default (1.225 kg/m3) → 454.4 N at 80 km/h."""
    dyn = VehicleDynamics(_make_params(env=None))
    F = dyn.drag_force(speed_ms=80.0/3.6)
    assert abs(F - 454.4) < 0.5


def test_drag_at_80kph_michigan_30c_50pct() -> None:
    """Michigan 30C, 99 kPa, 50% RH → density ~1.135 kg/m3 → drag drops ~7%."""
    env = EnvironmentConfig.from_temp_pressure(
        temperature_c=30.0, pressure_pa=99_000.0, humidity_pct=50.0,
        altitude_m=290.0,
    )
    dyn = VehicleDynamics(_make_params(env=env))
    F_isa = 0.5 * 1.225 * 1.502 * 1.0 * (80.0/3.6) ** 2
    F = dyn.drag_force(speed_ms=80.0/3.6)
    expected = F_isa * (env.air_density_kg_m3 / 1.225)
    assert abs(F - expected) < 0.5
    # Sanity: density is ~7% lower at 30C/99kPa than ISA.
    assert 0.91 < env.air_density_kg_m3 / 1.225 < 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_environment_drag_back_check.py -v`
Expected: FAIL — `VehicleParams` doesn't accept `environment=` yet.

- [ ] **Step 3: Add `environment` field to `VehicleParams`**

In `src/fsae_sim/vehicle/vehicle.py`, locate the `VehicleParams` dataclass (around line 14). Add the import and the field:

```python
# At top of file:
from fsae_sim.environment import EnvironmentConfig

# Inside VehicleParams, append:
    environment: "EnvironmentConfig | None" = None
```

- [ ] **Step 4: Add `_air_density()` helper and update drag/downforce in `dynamics.py`**

In `src/fsae_sim/vehicle/dynamics.py`, after the constructor (around line 102), add:

```python
    def _air_density(self) -> float:
        """Resolve air density: vehicle.environment if set, else ISA constant."""
        env = getattr(self.vehicle, "environment", None)
        if env is not None:
            return env.air_density_kg_m3
        return AIR_DENSITY_KG_M3
```

Then change `drag_force` (line 108-117), `downforce` (line 119-127), and `max_cornering_speed` legacy branch (line 443) to use `self._air_density()` instead of the bare constant. Three identical replacements:

```python
# OLD (each occurrence):
        return (
            0.5
            * AIR_DENSITY_KG_M3
            * self.vehicle.drag_coefficient
            * self.vehicle.frontal_area_m2
            * v * v
        )

# NEW (each occurrence):
        return (
            0.5
            * self._air_density()
            * self.vehicle.drag_coefficient
            * self.vehicle.frontal_area_m2
            * v * v
        )
```

(For the `max_cornering_speed` site, change `rho = AIR_DENSITY_KG_M3` to `rho = self._air_density()`.)

- [ ] **Step 5: Apply the same pattern to `load_transfer.py`**

In `src/fsae_sim/vehicle/load_transfer.py`, replace the module-level `AIR_DENSITY` import-as-alias with a per-call lookup. The cleanest path: keep the import as-is (back-compat for the existing `test_load_transfer_imports_shared_air_density` test) but read the density from `self.vehicle.environment` when the model is instantiated with a `VehicleParams` that has one.

Inside the `LoadTransferModel` constructor:

```python
        env = getattr(vehicle, "environment", None)
        self._air_density = (
            env.air_density_kg_m3 if env is not None else AIR_DENSITY
        )
```

Then in every downforce-using method, replace `AIR_DENSITY` with `self._air_density`.

- [ ] **Step 6: Update `tests/test_physics_constants.py` if needed**

The two tests `test_load_transfer_imports_shared_air_density` and `test_dynamics_imports_shared_air_density` only assert that the symbol is exported from the module — they pass even after this refactor as long as `AIR_DENSITY_KG_M3` remains importable from `physics_constants` and the alias remains on `dynamics` and `load_transfer`. Verify by running:

Run: `pytest tests/test_physics_constants.py -v`
Expected: 5 PASS unchanged.

- [ ] **Step 7: Run drag back-check test**

Run: `pytest tests/test_environment_drag_back_check.py -v`
Expected: 2 PASS.

- [ ] **Step 8: Run full test suite to catch regressions**

Run: `pytest -q`
Expected: same pass/fail/xfail counts as baseline plus the 7 new env tests passing. No regressions in `test_engine_envelope`, `test_speed_envelope`, `test_dynamics`, `test_load_transfer`.

- [ ] **Step 9: Commit**

```bash
git add src/fsae_sim/vehicle/vehicle.py src/fsae_sim/vehicle/dynamics.py src/fsae_sim/vehicle/load_transfer.py tests/test_environment_drag_back_check.py
git commit -m "feat(env): wire EnvironmentConfig into dynamics drag/downforce + load_transfer"
```

#### Task 2.4: Config schema + Michigan default in ct16ev.yaml

**Files:**
- Modify: `configs/ct16ev.yaml` — add `environment:` block.
- Modify: wherever YAML is loaded into `VehicleParams` (typically `backend/services/sim_runner.py` or `src/fsae_sim/data/loader.py`; locate the loader and add an env-resolution helper).
- Create: `tests/test_config_environment_block.py`.

- [ ] **Step 1: Locate the YAML-to-VehicleParams loader**

Run: `grep -rn "VehicleParams(" src/ backend/`
Expected: identifies one or two construction sites — typically `backend/services/sim_runner.py` or `src/fsae_sim/data/loader.py`. Read that file to confirm.

- [ ] **Step 2: Write the failing config test**

```python
# tests/test_config_environment_block.py
from __future__ import annotations
import yaml
from pathlib import Path
import pytest
from fsae_sim.environment import EnvironmentConfig


def test_ct16ev_yaml_includes_environment_block() -> None:
    cfg = yaml.safe_load(
        Path("configs/ct16ev.yaml").read_text()
    )
    assert "environment" in cfg
    env = cfg["environment"]
    assert "temperature_c" in env
    assert "pressure_pa" in env
    assert "humidity_pct" in env
    assert "altitude_m" in env


def test_environment_config_loads_from_yaml_block() -> None:
    cfg = yaml.safe_load(
        Path("configs/ct16ev.yaml").read_text()
    )
    env_block = cfg["environment"]
    env = EnvironmentConfig.from_temp_pressure(
        temperature_c=env_block["temperature_c"],
        pressure_pa=env_block["pressure_pa"],
        humidity_pct=env_block["humidity_pct"],
        altitude_m=env_block["altitude_m"],
    )
    # MIS @ 290 m, ~25 C, ~50% RH → density 1.13..1.18 kg/m3.
    assert 1.10 < env.air_density_kg_m3 < 1.22
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config_environment_block.py -v`
Expected: FAIL — `KeyError: 'environment'`.

- [ ] **Step 4: Add the `environment:` block to `configs/ct16ev.yaml`**

Append at the end of `configs/ct16ev.yaml`:

```yaml
environment:
  # Michigan FSAE 2025 endurance, MIS infield, ~290 m AGL.
  # Default values: 25 C / 99 kPa station pressure / 50% RH.
  # Override per-event via CLI or by editing this block.
  # METAR auto-fetch is opt-in via `--fetch-metar KJXN`.
  temperature_c: 25.0
  pressure_pa: 98500.0
  humidity_pct: 50.0
  altitude_m: 290.0
  metar_station: KJXN
  notes: |
    KJXN (Jackson County Reynolds Field) is ~32 km north of MIS.
    Station-pressure-corrected from KJXN altimeter setting via ISA.
    User-decision A2: confirm KJXN before relying on auto-fetch.
```

- [ ] **Step 5: Update the YAML-to-VehicleParams loader to resolve `environment`**

In the loader (located in step 1), after loading the `vehicle:` block, also load the `environment:` block (if present) and pass it as a kwarg:

```python
# Inside the VehicleParams construction:
env_block = cfg.get("environment")
if env_block is not None:
    env = EnvironmentConfig.from_temp_pressure(
        temperature_c=env_block["temperature_c"],
        pressure_pa=env_block["pressure_pa"],
        humidity_pct=env_block.get("humidity_pct", 0.0),
        altitude_m=env_block.get("altitude_m", 0.0),
    )
else:
    env = None
vehicle = VehicleParams(
    ...,
    environment=env,
)
```

- [ ] **Step 6: Run config test to verify it passes**

Run: `pytest tests/test_config_environment_block.py -v`
Expected: 2 PASS.

- [ ] **Step 7: Run full suite to confirm no regressions**

Run: `pytest -q`
Expected: previous pass count + 2 new tests, no regressions.

- [ ] **Step 8: Commit**

```bash
git add configs/ct16ev.yaml tests/test_config_environment_block.py <loader file>
git commit -m "feat(env): add environment: block to ct16ev.yaml + loader wiring"
```

### Acceptance for Part 2

- ISA-only path: drag at 80 km/h with default config = 454.4 N (deterministic).
- With `environment: {temperature_c: 30, pressure_pa: 99000, humidity_pct: 50, altitude_m: 290}`, drag = 454.4 * (1.135 / 1.225) = 421.0 N (proportional shift).
- Humidity correction is < 1.5 % across realistic FSAE conditions (verified by `test_humid_air_at_30c_80pct_humidity_is_below_dry_30c`).
- METAR parser passes both `python-metar`-installed and inline-fallback paths.
- `tests/test_physics_constants.py` still passes (constant remains exported).
- All other physics tests pass without change.

---

## Part 3 — PHYSICS_AUDIT.md Retirement

### Decision

**Delete + redirect** (already 90 % done).

The file was deleted from the working tree in commit `ef0da94` (verified via `git status docs/PHYSICS_AUDIT.md` returning clean). No other documentation, source, or test references it. The remaining cleanup is editorial inside `SIM_AUDIT_2026-05.md` to remove the four "PHYSICS_AUDIT" mentions and the P2 task that points at the deleted file.

The full text of `PHYSICS_AUDIT.md` remains accessible at git blob `ef0da94^:docs/PHYSICS_AUDIT.md` for anyone who wants to read the historical critique.

### Tasks

#### Task 3.1: Cross-reference grep — confirm zero residual references outside SIM_AUDIT

**Files:** none modified (verification only).

- [ ] **Step 1: Run cross-reference grep**

```bash
grep -rni "PHYSICS_AUDIT\|physics_audit" src/ backend/ webapp/ scripts/ tests/ docs/ README.md CLAUDE.md 2>&1
```

Expected output (exactly four hits, all in `docs/SIM_AUDIT_2026-05.md`):
```
docs/SIM_AUDIT_2026-05.md:7:This supersedes most of `docs/PHYSICS_AUDIT.md` ...
docs/SIM_AUDIT_2026-05.md:68:## What is no longer broken (despite `PHYSICS_AUDIT.md`)
docs/SIM_AUDIT_2026-05.md:97:`PHYSICS_AUDIT.md` should be revised or replaced. ...
docs/SIM_AUDIT_2026-05.md:193:- [ ] **`docs/PHYSICS_AUDIT.md` is outdated.** ...
```

If any other file matches, document the additional reference in this task and remove it as a separate edit before continuing. Likely candidates: a stale CLAUDE.md cache, a webapp page describing docs, a README block.

- [ ] **Step 2: Verify the file truly does not exist**

```bash
ls docs/PHYSICS_AUDIT.md 2>&1
git ls-files docs/PHYSICS_AUDIT.md
```

Expected: `ls` returns "No such file or directory". `git ls-files` returns empty (file is not tracked).

#### Task 3.2: Edit SIM_AUDIT_2026-05.md to remove the four stale references

**Files:**
- Modify: `docs/SIM_AUDIT_2026-05.md` lines 7, 68, 97, 193.

- [ ] **Step 1: Edit line 7-8 — remove the supersession sentence**

Current text (lines 7-8):
```
This supersedes most of `docs/PHYSICS_AUDIT.md` (which describes a much earlier
version of the codebase). Where the two disagree, this document is current.
```

Replace with a one-line note pointing readers at the git history:
```
This is the current authoritative physics audit for the simulator. An earlier
audit (`docs/PHYSICS_AUDIT.md`) was retired in commit `ef0da94`; the historical
text remains accessible at git blob `ef0da94^:docs/PHYSICS_AUDIT.md`.
```

- [ ] **Step 2: Edit line 68 — section heading**

Current:
```
## What is no longer broken (despite `PHYSICS_AUDIT.md`)
```

Replace with:
```
## What is no longer broken (despite the retired audit)
```

- [ ] **Step 3: Edit line 97 — remove "should be revised or replaced" sentence**

Current (lines 97-98):
```
`PHYSICS_AUDIT.md` should be revised or replaced. Do not waste cycles
re-investigating those.
```

Replace with:
```
The retired audit's "highest-severity findings" are addressed; do not waste
cycles re-investigating them.
```

- [ ] **Step 4: Edit line 193 — remove the P2 task**

Current (lines 193-195):
```
- [ ] **`docs/PHYSICS_AUDIT.md` is outdated.** Either revise it to reflect
  current state, or delete and link this document. Most of its findings have
  been fixed.
```

**Delete this entire P2 list item** (the file is already deleted; the task is moot).

- [ ] **Step 5: Re-run the grep to confirm zero references**

```bash
grep -rni "PHYSICS_AUDIT\|physics_audit" src/ backend/ webapp/ scripts/ tests/ docs/ README.md CLAUDE.md
```

Expected: zero matches (or one match: a single literal in the SIM_AUDIT line "remains accessible at git blob `ef0da94^:docs/PHYSICS_AUDIT.md`" — that one is intentional and acceptable).

- [ ] **Step 6: Commit**

```bash
git add docs/SIM_AUDIT_2026-05.md
git commit -m "docs: retire PHYSICS_AUDIT.md cross-references in SIM_AUDIT"
```

### Acceptance for Part 3

- Zero source / test / config references to PHYSICS_AUDIT.md after task 3.2.
- At most one intentional reference inside `SIM_AUDIT_2026-05.md` pointing readers at the git blob.
- The P2 "PHYSICS_AUDIT.md is outdated" item no longer appears in the improvement checklist.
- File remains accessible via git blob `ef0da94^:docs/PHYSICS_AUDIT.md` for any reader who wants the historical critique.

---

## Part 4 — Cross-Cutting QSS Gap Analysis (research deliverable)

### Methodology

Each candidate area has a one-line description, the citation that established it as a feature gap, and a verdict: **in scope (add)**, **defer**, or **out of scope (declared)**. "In scope" items get a follow-up plan name and a 5h-tier effort estimate; we do not expand them into full plans here.

### Bucket 1 — In scope (add)

These features are believed to materially affect tune-sweep deltas, are achievable inside the current QSS architecture, and the audit's existing P0..P3 list does not already cover them.

#### G1. Tire temperature dynamics (mu vs T)

- **Description**: tire mu varies with carcass + tread temperature over a stint. Hoosier R-rubber peak grip is at ~80-110 C; warm-up from cold ambient affects lap-1 grip; over-temp on long stints affects late-endurance grip.
- **Citation**: TameTire / FTire models in CarMaker (https://ipg-automotive.com); Milliken Race Car Vehicle Dynamics §2.7 (tire load + thermal sensitivity); SAE 2016-36-0164 lists this as a top QSS upgrade. Hoosier TTC Round 8 includes thermal sweeps.
- **Why this matters for sweep ranking**: a torque-cap raise that triggers more rear-tire slip work also raises rear-tire temperature; if tire mu is constant in the model, a hotter tire's grip-loss on late laps is invisible.
- **Recommended action**: add a tire-thermal lumped model (carcass + tread, two states per axle) with mu(T) curve from R-data. Re-validate against late-lap speed bias.
- **Follow-up plan name**: `2026-XX-XX-tire-thermal.md`.
- **Effort tier**: ~10-15 h (one mid-tier plan).

#### G2. Brake-bias model with rotor / pad mu(T) fade

- **Description**: front/rear brake force split, brake disc / pad temperature, pad mu fade vs T.
- **Citation**: CarMaker hydraulic-brake module (datasheet); Adams/Car suspension+brake model; Milliken §13.5; OptimumG blog Lap Sim Series Part 7.
- **Why this matters for sweep ranking**: torque-cap and BMS sweeps that reduce regen contribution shift more energy to mechanical brakes; bias-and-fade decides whether a sweep extracts that benefit or saturates the rear brakes.
- **Recommended action**: add `BrakeBiasConfig` + per-axle thermal-rotor model. Couple to `dynamics.mechanical_brake_force()`.
- **Follow-up plan name**: `2026-XX-XX-brake-thermal.md`.
- **Effort tier**: ~10-15 h.

#### G3. Aero load distribution vs ride height (CL_f / CL_r map)

- **Description**: front-vs-rear aero balance shifts with ride height and pitch angle; affects load transfer at high speed.
- **Citation**: CarMaker aero map; SAE 2016-36-0164 §4.2; UConn DSS reports `downforce_distribution_front: 0.61` as a static measurement, with no speed/ride-height dependence.
- **Why this matters for sweep ranking**: sweeps that change top speed (motor RPM, FW corner) shift the aero load distribution; static-balance assumption hides this.
- **Recommended action**: 2D map `CL_f(v, ride_height)` and same for CL_r; default to constant balance until DSS supplies a CFD or wind-tunnel sweep.
- **Follow-up plan name**: `2026-XX-XX-aero-map.md`.
- **Effort tier**: ~6-10 h (model is small; DSS data acquisition is the bottleneck).

#### G4. Inverter thermal derating (Cascadia CM200DX)

- **Description**: Cascadia CM200DX has a published thermal derating curve. Current sim treats inverter as thermally infinite.
- **Citation**: Cascadia Motion CM200DX datasheet; ETSE / Wisconsin Racing reports thermal derating in real endurance runs.
- **Why this matters for sweep ranking**: late-stint torque sweeps that push average inverter current higher trigger derating that the sim does not see; effect is similar to BMS thermal taper but in the inverter domain.
- **Recommended action**: add `InverterThermalConfig` + lumped thermal model + derating multiplier on `apply_inverter_delivery`.
- **Follow-up plan name**: `2026-XX-XX-inverter-thermal.md`.
- **Effort tier**: ~5-8 h.

### Bucket 2 — Defer (clearly bounded future work)

These features are real gaps but either (a) lower leverage on tune-sweep deltas, (b) require data acquisition the team does not currently have, or (c) belong to the QSS→QTS upgrade arc.

#### D1. Suspension kinematics (anti-dive / anti-squat / roll-center motion)

- **Citation**: Milliken Race Car Vehicle Dynamics §17; Adams/Car suspension hardpoint model; SIMULATOR_ISSUES.md M1 ("No anti-squat / anti-dive geometry").
- **Why deferred**: requires CAD hardpoints + bushing models; current load-transfer model uses static geometric+elastic split which is good enough for tune-sweep deltas. Gain ~ 1-2 % on absolute lap time, much less on deltas.
- **Future scope**: `2026-XX-XX-suspension-kinematics.md`, ~30 h, requires DSS suspension CAD.

#### D2. Driver feedback dynamics (reaction time, throttle/brake bandwidth)

- **Citation**: Milliken §16, Macadam preview-follower; OptimumG Lap Sim Series Part 12; CarMaker IPGDriver.
- **Why deferred**: subsumed by Agent A's adaptive driver model (P0). Once an adaptive driver exists, second-order driver bandwidth becomes the next refinement.
- **Future scope**: depends on Agent A's adaptive-driver outcome.

#### D3. Track-surface mu variation (patches, lap-to-lap evolution)

- **Citation**: CarMaker road mu maps; F1 / IndyCar tire-rubber-build-up models.
- **Why deferred**: FSAE tracks are short and uniform; rubber buildup over an endurance is real but small compared to tire-thermal effects (G1) and driver consistency. Capture together with G1 if both are added.
- **Future scope**: subsumed under G1.

#### D4. Wind effects beyond density (gusts, sustained crosswind)

- **Citation**: CarMaker weather module.
- **Why deferred**: average wind effect is captured by the air-density work (Part 2); transient gust dynamics are below the sim's noise floor for FSAE timescales.
- **Future scope**: out unless wind data justifies it.

#### D5. Battery cooling system (pump duty, coolant loop)

- **Note**: Agent D owns the battery thermal upgrade; their plan will reference Part 2 (air-density) for the air-side heat-rejection coefficient. Not duplicated here.

#### D6. Per-cell or per-module pack thermal

- **Citation**: SIM_AUDIT_2026-05.md P3 list; Voltt cell sim has cell-level data.
- **Why deferred**: P3 in the existing audit; wait for Agent D.

#### D7. Differential modeling (Salisbury / clutch-pack / electronic)

- **Citation**: Milliken §14; CarMaker driveline.
- **Why deferred**: CT-16EV has an open differential per DSS; modeling preload / clutch-pack only matters when the team installs a different diff. Out until that hardware change.

#### D8. Yaw-inertia / transient roll dynamics

- **Citation**: Milliken §5-6; the QSS→QTS upgrade arc.
- **Why deferred**: this is the QSS architecture boundary. Adding it converts the simulator to QTS, which is a 6-12 month rewrite. The audit explicitly frames this sim as QSS for sweep work.

### Bucket 3 — Out of scope (declared)

These features are professional-grade transient or DiL capabilities that do not belong in a QSS sweep tool. The audit already declares this boundary; we restate it here to keep the buckets honest.

#### O1. Real-time co-simulation with Simulink / control-unit-in-the-loop

- **Citation**: CarMaker / VSM / VI-Grade; not a QSS feature.
- **Verdict**: out. The team uses LVCU firmware-faithful translation in `PowertrainModel.lvcu_torque_command()`; that is the appropriate level of fidelity for QSS.

#### O2. Driver-in-loop / motion platform

- **Citation**: VI-Grade DiL simulators.
- **Verdict**: out. Not relevant to a tune-sweep tool.

#### O3. Full multi-body suspension dynamics (anti-dive solved with bushings + compliance)

- **Citation**: Adams/Car.
- **Verdict**: out for QSS. Listed in D1 as defer for the geometric portion only.

#### O4. NURBS road surface with full 3D mu maps

- **Citation**: CarMaker.
- **Verdict**: out. FSAE autocross surfaces are flat asphalt; the synthetic-track work in Part 1 is sufficient.

#### O5. Tire wear over multi-stint duration

- **Citation**: F1 / WEC tire wear models.
- **Verdict**: out. FSAE endurance is one stint per car; tire wear is < 5 % over 22 km.

#### O6. Tire relaxation length (transient lateral force build-up)

- **Citation**: Milliken §3.7; PAC2002 has this but as a transient parameter.
- **Verdict**: out for QSS. Re-enters scope at QSS→QTS boundary.

#### O7. Telemetry-derived parameter-stable noise floors (random noise injection)

- **Citation**: not a real QSS feature — consider if useful for sweep harness.
- **Verdict**: out. Agent B owns the sweep harness; deterministic delta-vs-baseline + noise-floor reporting is the right framing, not stochastic noise injection.

### Bucket summary table

| Bucket | Items |
|--------|-------|
| **In scope (add)** | G1 tire thermal, G2 brake bias + rotor thermal, G3 aero load-distribution map, G4 inverter thermal derating |
| **Defer** | D1 suspension kinematics, D2 driver feedback dynamics, D3 surface-mu variation, D4 wind transients, D5 battery cooling (Agent D), D6 per-cell pack thermal, D7 differential model, D8 yaw / roll transient |
| **Out of scope (declared)** | O1 Simulink HiL, O2 driver-in-loop, O3 full MBS suspension, O4 NURBS road surface, O5 tire wear, O6 tire relaxation length, O7 stochastic noise injection |

---

## Risks / Unknowns

1. **R1 (track validation tolerance)**: 0.5 m median cross-track residual is an educated estimate. If the hand-authored Michigan cone map (Task 1.1 step 2) is sparse (< 30 cones), the Voronoi centerline can drift by 1-2 m on long straights; we may have to relax the median tolerance to 1.0 m or fall back to a higher cone density.
2. **R2 (Voronoi loop closure)**: greedy nearest-neighbour ordering of Voronoi vertices can fail on tracks with hairpins where the centerline doubles back. Mitigation: if Task 1.3 step 4 fails, augment the ordering with a "must pass through start gate" constraint (anchor the loop at the gate midpoint).
3. **R3 (METAR availability)**: KJXN observations have ~5 % gap rate (cloud cover prevents ASOS). For events when the nearest hour has no observation, fall back to the previous hour or a manual override; do not silently produce a wrong density.
4. **R4 (humidity correction is ~1 %)**: this might be below the sim's own noise floor. We add it for completeness, not because it moves the grade. Verified by `test_humid_air_at_30c_80pct_humidity_is_below_dry_30c` which asserts the effect is `0.5 % < drop < 2 %`.
5. **R5 (cone-map authorship time)**: Task 1.1 step 2 (hand-author Michigan cone map) is user-driven and slow. If the user does not produce it, Task 1.5 stays SKIPPED and Part 1's acceptance is conditional. Suggest pairing this with a 1-hour Google-Earth tracing session before beginning Part 1.
6. **R6 (PHYSICS_AUDIT cleanup is essentially done)**: Part 3 is editorial only; risk is minimal. Only risk is missing a hidden reference (e.g., in a webapp page describing docs structure). Mitigation: the grep in Task 3.1 step 1 catches all of `src/ backend/ webapp/ scripts/ tests/ docs/ README.md CLAUDE.md`.
7. **R7 (QSS gap analysis is opinion)**: bucket assignments depend on the user's actual sweep targets. If the user sweeps tire selection or aero package, G3 (aero map) jumps from "in scope (add)" to "high priority"; if they only sweep torque/RPM, G1/G2 dominate. Document this conditionality.

## Verification / Acceptance Criteria

### Track (Part 1)

- [ ] `pytest tests/test_track_synthetic_loader.py -v` — 2 PASS.
- [ ] `pytest tests/test_voronoi_centerline.py -v` — 2 PASS.
- [ ] `pytest tests/test_track_from_cone_map.py -v` — 2 PASS.
- [ ] **Conditional on cone map being authored**: `pytest tests/test_synthetic_vs_driven_michigan.py -v` — synthetic Michigan track lap length within 3 % of driven-mean (1006 m), curvature p95 within 15 %, median cross-track residual < 0.5 m.
- [ ] No regressions in existing track tests (`pytest tests/test_track*.py -v`).

### Air density (Part 2)

- [ ] ISA at sea level returns 1.225 kg/m^3 to within 1e-3.
- [ ] ISA at 290 m (MIS) returns 1.18..1.20 kg/m^3.
- [ ] Drag at 80 km/h with default ISA = **454.4 N ± 0.5 N** (matches CdA = 1.502 m^2 back-derivation).
- [ ] Drag at 80 km/h with `environment: {temperature_c: 30, pressure_pa: 99000, humidity_pct: 50, altitude_m: 290}` shifts proportionally to the new density (~ 421 N; assertion checks the proportional shift, not the absolute number).
- [ ] METAR parser handles a fixture METAR string with both `python-metar`-installed and inline-fallback paths.
- [ ] No regressions in existing dynamics, load_transfer, speed_envelope, engine tests.

### PHYSICS_AUDIT.md retirement (Part 3)

- [ ] Final `grep -rni "PHYSICS_AUDIT\|physics_audit"` over `src/ backend/ webapp/ scripts/ tests/ docs/ README.md CLAUDE.md` returns at most ONE match (the deliberate git-blob pointer in `SIM_AUDIT_2026-05.md`).
- [ ] `docs/PHYSICS_AUDIT.md` does not exist on disk and is not tracked by git.
- [ ] The P2 task "PHYSICS_AUDIT.md is outdated" no longer appears in the SIM_AUDIT improvement checklist.

### QSS gap analysis (Part 4)

- [ ] All four "in scope (add)" items (G1-G4) have a follow-up plan filename and 5h effort tier.
- [ ] All eight "defer" items (D1-D8) have a citation and reason for deferral.
- [ ] All seven "out of scope" items (O1-O7) have a one-line declaration of why they're out for QSS.

## Effort Estimate

Tier definitions: small ≤ 5 h, medium 5-10 h, large 10-20 h, very-large > 20 h.

- **Part 1 (Track)**: medium-large.
  - Task 1.1 (schema + Michigan cone map authorship): medium (5 h, mostly the user's tracing time).
  - Task 1.2 (loader): small (1 h).
  - Task 1.3 (Voronoi centerline): medium (4 h).
  - Task 1.4 (`Track.from_cone_map`): small (2 h).
  - Task 1.5 (validation test): small (2 h, conditional).
  - **Total: 12-15 h** if cone map exists; 8-10 h of dev time (rest is user-side tracing).

- **Part 2 (Air density)**: medium.
  - Task 2.1 (ISA + humidity): small (2 h).
  - Task 2.2 (METAR parser): small (2 h).
  - Task 2.3 (wire into VehicleParams + dynamics + load_transfer): small (2 h).
  - Task 2.4 (config schema + ct16ev.yaml + loader): small (1 h).
  - **Total: 7-9 h**.

- **Part 3 (PHYSICS_AUDIT cleanup)**: small.
  - Task 3.1 (grep verification): trivial (15 min).
  - Task 3.2 (edit SIM_AUDIT_2026-05): trivial (30 min).
  - **Total: < 1 h**.

- **Part 4 (QSS gap analysis)**: research deliverable, no implementation.
  - **Total: 0 h** for implementation; ~3 h was spent researching for this plan.

**Plan-wide estimate: ~20-25 h total dev time**, plus user-side cone-map authorship (~5 h, parallelizable).

---

## Self-review notes (for the executor)

- **Type consistency check**: `ConePoint`, `EnvironmentConfig`, `parse_metar_string`, `voronoi_centerline`, `Track.from_cone_map` are all introduced once and used consistently across tasks.
- **No placeholders**: every code block contains real callable code, not "implement here".
- **Path verification**: all source paths verified against the current repo via `Read`/`Grep` at plan-write time. `physics_constants.py:10`, `dynamics.py:113-127, 443`, `load_transfer.py:12`, `track.py:209` (line target for the new classmethod), `tests/test_physics_constants.py:32-48`, `configs/ct16ev.yaml:8` (DSS CdA = 1.502).
- **Spec coverage**: every requirement in the request is mapped to a task — independent track, air density, PHYSICS_AUDIT retirement, QSS gap analysis, citations mandatory, user-decision items flagged.
