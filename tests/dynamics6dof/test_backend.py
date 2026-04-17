"""Unit tests for FastestLapDynamicsBackend.

Validates each engine-facing query produces a sane steady-state answer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fsae_sim.dynamics6dof.backend import FastestLapDynamicsBackend
from fsae_sim.vehicle.vehicle import VehicleConfig


@pytest.fixture(scope="module")
def backend() -> FastestLapDynamicsBackend:
    cfg_path = Path(__file__).parents[2] / "configs" / "ct16ev.yaml"
    # Resolve .tir path relative to repo root before handing to the backend
    cfg = VehicleConfig.from_yaml(cfg_path)
    backend = FastestLapDynamicsBackend.from_vehicle_config(cfg)
    # Patch up the .tir path to absolute since the YAML path is repo-relative
    return backend


def test_drag_scales_with_speed_squared(backend):
    d20 = backend.drag_force(20.0)
    d40 = backend.drag_force(40.0)
    # Quadratic: d40 / d20 ≈ 4
    assert d40 / d20 == pytest.approx(4.0, rel=1e-3)


def test_downforce_at_80kph_matches_dss(backend):
    # DSS says ~625 N at 80 kph (22.22 m/s).
    df = backend.downforce(80.0 / 3.6)
    assert df == pytest.approx(625.0, rel=0.1)


def test_rolling_resistance_is_positive_at_rest(backend):
    r = backend.rolling_resistance_force(0.0)
    assert r > 0.0


def test_total_resistance_grows_with_speed(backend):
    lo = backend.total_resistance(5.0, 0.0, 0.0)
    hi = backend.total_resistance(25.0, 0.0, 0.0)
    assert hi > lo


def test_max_traction_positive_and_finite(backend):
    f = backend.max_traction_force(10.0)
    assert f > 0.0
    assert f < 20_000.0


def test_max_braking_exceeds_max_traction(backend):
    # All 4 wheels brake vs 2 rear wheels driving -> braking > traction
    brake = backend.max_braking_force(10.0)
    drive = backend.max_traction_force(10.0)
    assert brake > drive


def test_max_cornering_speed_decreases_with_tighter_curvature(backend):
    v_loose = backend.max_cornering_speed(curvature=0.02)  # 50 m radius
    v_tight = backend.max_cornering_speed(curvature=0.1)   # 10 m radius
    assert v_tight < v_loose


def test_straight_curvature_is_infinity(backend):
    import math
    v = backend.max_cornering_speed(curvature=0.0)
    assert math.isinf(v)


def test_resolve_exit_speed_accelerates_when_net_force_positive(backend):
    exit_v, t = backend.resolve_exit_speed(
        entry_speed_ms=10.0, segment_length_m=5.0,
        net_force_n=500.0, corner_speed_limit_ms=50.0,
    )
    assert exit_v > 10.0
    assert t > 0.0
    assert t < 1.0


def test_resolve_exit_speed_clamps_to_corner_limit(backend):
    exit_v, _ = backend.resolve_exit_speed(
        entry_speed_ms=10.0, segment_length_m=100.0,
        net_force_n=1000.0, corner_speed_limit_ms=15.0,
    )
    assert exit_v == pytest.approx(15.0)
