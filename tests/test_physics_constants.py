"""Tests for :mod:`fsae_sim.physics_constants`.

Guards the single-source-of-truth invariant called out in audit finding
NF-22: ``GRAVITY_M_S2`` and ``AIR_DENSITY_KG_M3`` must be defined exactly
once and re-used by every module that needs them.
"""

from __future__ import annotations


def test_physics_constants_values() -> None:
    """Constants expose the expected numerical values."""
    from fsae_sim import physics_constants as pc

    assert pc.GRAVITY_M_S2 == 9.81
    assert pc.AIR_DENSITY_KG_M3 == 1.225


def test_load_transfer_imports_shared_gravity() -> None:
    """``load_transfer`` must re-use the shared gravity constant."""
    from fsae_sim import physics_constants as pc
    from fsae_sim.vehicle import load_transfer

    assert load_transfer.GRAVITY is pc.GRAVITY_M_S2


def test_load_transfer_imports_shared_air_density() -> None:
    """``load_transfer`` must re-use the shared air density constant."""
    from fsae_sim import physics_constants as pc
    from fsae_sim.vehicle import load_transfer

    assert load_transfer.AIR_DENSITY is pc.AIR_DENSITY_KG_M3


def test_dynamics_imports_shared_gravity() -> None:
    """``dynamics`` module exposes gravity from the shared constants module."""
    from fsae_sim import physics_constants as pc
    from fsae_sim.vehicle import dynamics

    assert dynamics.GRAVITY_M_S2 is pc.GRAVITY_M_S2


def test_dynamics_imports_shared_air_density() -> None:
    """``dynamics`` module exposes air density from the shared constants module."""
    from fsae_sim import physics_constants as pc
    from fsae_sim.vehicle import dynamics

    assert dynamics.AIR_DENSITY_KG_M3 is pc.AIR_DENSITY_KG_M3
