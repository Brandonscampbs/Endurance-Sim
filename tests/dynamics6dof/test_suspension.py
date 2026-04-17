# tests/dynamics6dof/test_suspension.py
"""Tests for the symmetric/asymmetric suspension deformation decomposition.

Ported from axle_car_6dof.hpp:119-157. Verifies that the per-axle tire
deformation behaves correctly for the canonical excitations: zero state,
pure heave, and pure roll.
"""
from __future__ import annotations

import pytest

from fsae_sim.dynamics6dof.params import Dynamics6DofParams
from fsae_sim.dynamics6dof.suspension import axle_tire_deformation


def test_zero_state_yields_zero_deformation():
    p = Dynamics6DofParams.ct16ev_defaults()
    wl, wr, dwl, dwr = axle_tire_deformation(
        z=0.0,
        phi=0.0,
        dz=0.0,
        dphi=0.0,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_antiroll=p.k_antiroll_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0,
        beta_left=0.0,
        beta_right=0.0,
    )
    assert wl == pytest.approx(0.0)
    assert wr == pytest.approx(0.0)
    assert dwl == pytest.approx(0.0)
    assert dwr == pytest.approx(0.0)


def test_pure_heave_deforms_both_wheels_equally():
    p = Dynamics6DofParams.ct16ev_defaults()
    wl, wr, dwl, dwr = axle_tire_deformation(
        z=0.01,
        phi=0.0,
        dz=0.0,
        dphi=0.0,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_antiroll=p.k_antiroll_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0,
        beta_left=0.0,
        beta_right=0.0,
    )
    assert wl == pytest.approx(wr, rel=1e-12)
    assert wl > 0.0


def test_pure_roll_deforms_wheels_oppositely():
    p = Dynamics6DofParams.ct16ev_defaults()
    wl, wr, dwl, dwr = axle_tire_deformation(
        z=0.0,
        phi=0.01,
        dz=0.0,
        dphi=0.0,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_antiroll=p.k_antiroll_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0,
        beta_left=0.0,
        beta_right=0.0,
    )
    assert wl * wr < 0  # opposite sign
    assert wl == pytest.approx(-wr, rel=1e-12)


def test_pure_heave_rate_matches_heave_direction():
    """A positive dz (chassis moving down/compression) should produce equal
    positive deformation rates on both wheels (dwl == dwr > 0)."""
    p = Dynamics6DofParams.ct16ev_defaults()
    wl, wr, dwl, dwr = axle_tire_deformation(
        z=0.0,
        phi=0.0,
        dz=0.05,
        dphi=0.0,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_antiroll=p.k_antiroll_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0,
        beta_left=0.0,
        beta_right=0.0,
    )
    assert wl == pytest.approx(0.0)
    assert wr == pytest.approx(0.0)
    assert dwl == pytest.approx(dwr, rel=1e-12)
    assert dwl > 0.0


def test_pure_roll_rate_matches_roll_direction():
    """A positive dphi should produce opposite-sign rates on the two wheels."""
    p = Dynamics6DofParams.ct16ev_defaults()
    wl, wr, dwl, dwr = axle_tire_deformation(
        z=0.0,
        phi=0.0,
        dz=0.0,
        dphi=0.5,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_antiroll=p.k_antiroll_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0,
        beta_left=0.0,
        beta_right=0.0,
    )
    assert dwl * dwr < 0.0
    assert dwl == pytest.approx(-dwr, rel=1e-12)


def test_antiroll_increases_roll_resistance():
    """Adding an anti-roll bar should amplify the |deformation| response to
    a pure roll input, relative to zero anti-roll."""
    p = Dynamics6DofParams.ct16ev_defaults()
    kwargs = dict(
        z=0.0,
        phi=0.01,
        dz=0.0,
        dphi=0.0,
        track_m=p.track_front_m,
        k_chassis=p.k_chassis_front_npm,
        k_tire=p.k_tire_radial_npm,
        steering_rad=0.0,
        beta_left=0.0,
        beta_right=0.0,
    )
    wl_noar, wr_noar, _, _ = axle_tire_deformation(k_antiroll=0.0, **kwargs)
    wl_ar, wr_ar, _, _ = axle_tire_deformation(
        k_antiroll=p.k_antiroll_front_npm, **kwargs
    )
    assert abs(wl_ar) > abs(wl_noar)
    assert abs(wr_ar) > abs(wr_noar)
