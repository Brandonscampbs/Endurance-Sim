"""Tests for ``EnvironmentConfig`` air-density model.

Replaces the bare ``physics_constants.AIR_DENSITY_KG_M3`` constant with
an event-configurable dataclass.  Density is computed from temperature,
absolute pressure, and relative humidity using the ICAO ISA dry-air law
combined with the Magnus/Tetens correction for saturation vapor
pressure.

References:
    - ICAO Doc 7488/3 (1993), ISO 2533:1975 — Standard Atmosphere.
    - CIPM-2007 (Picard) — equation for the density of moist air.
    - Tetens, O. (1930) / Magnus form for saturation vapor pressure.
    - Plan ``docs/superpowers/plans/2026-05-06-track-environment-qss-gap.md``
      Part 2.
"""

from __future__ import annotations

import warnings

import pytest

from fsae_sim.vehicle.environment import EnvironmentConfig


# ---------------------------------------------------------------------------
# ISA / Magnus sanity checks
# ---------------------------------------------------------------------------


class TestISAStandardConditions:
    def test_isa_sea_level_dry_returns_1p225(self) -> None:
        """At ISA sea level (15 C, 101325 Pa, 0 % RH) density is 1.225 kg/m^3."""
        env = EnvironmentConfig(
            ambient_temp_c=15.0,
            ambient_pressure_pa=101325.0,
            relative_humidity=0.0,
            altitude_m=0.0,
        )
        assert env.air_density_kg_m3 == pytest.approx(1.225, abs=1e-3)

    def test_humidity_reduces_density(self) -> None:
        """Magnus correction: moist air is less dense at the same T, P."""
        dry = EnvironmentConfig(
            ambient_temp_c=25.0,
            ambient_pressure_pa=98500.0,
            relative_humidity=0.0,
        )
        humid = EnvironmentConfig(
            ambient_temp_c=25.0,
            ambient_pressure_pa=98500.0,
            relative_humidity=1.0,
        )
        assert humid.air_density_kg_m3 < dry.air_density_kg_m3

    def test_higher_temperature_reduces_density(self) -> None:
        """Ideal-gas behavior: rho ~ 1/T at constant P."""
        cool = EnvironmentConfig(
            ambient_temp_c=10.0, ambient_pressure_pa=101325.0,
            relative_humidity=0.0,
        )
        warm = EnvironmentConfig(
            ambient_temp_c=30.0, ambient_pressure_pa=101325.0,
            relative_humidity=0.0,
        )
        assert warm.air_density_kg_m3 < cool.air_density_kg_m3

    def test_higher_pressure_increases_density(self) -> None:
        """Ideal-gas: rho ~ P at constant T."""
        low_p = EnvironmentConfig(
            ambient_temp_c=25.0, ambient_pressure_pa=95000.0,
            relative_humidity=0.0,
        )
        high_p = EnvironmentConfig(
            ambient_temp_c=25.0, ambient_pressure_pa=101325.0,
            relative_humidity=0.0,
        )
        assert high_p.air_density_kg_m3 > low_p.air_density_kg_m3


# ---------------------------------------------------------------------------
# Drag back-check at Michigan default
# ---------------------------------------------------------------------------


class TestDragBackCheck:
    """Cross-check the drag computation against the audit's DSS-derived
    431 N at 80 km/h citation.

    Discrepancy from the audit number is documented in the test:
    Agent F (track-environment plan) flagged that with the
    currently-configured CdA = 1.50 m^2 the computed drag at 1.225 kg/m^3
    is ~454 N, not 431 N — a ~5 % DSS round-trip difference.  Per
    "no bandaid fixes": we record this gap, do NOT tune the air-density
    or CdA to hide it; rerun this test if either is recalibrated.
    """

    def test_drag_within_audit_range_at_michigan_default(self) -> None:
        """At 80 km/h with Michigan-default environment, drag is within
        5 % of 431 N (the audit's DSS citation).

        Audit cites 431 N; current configured CdA=1.50 m^2 with
        rho=1.225 (ISA sea level) yields 454 N (a known DSS round-trip
        delta).  Michigan environment (25 C, 98500 Pa, 50 % RH, 290 m
        altitude) shifts rho slightly below ISA and brings the result
        closer to the DSS number.  We allow +-5 % to span both.
        """
        env = EnvironmentConfig()  # Michigan defaults
        cda = 1.50  # m^2 (DSS)
        v_ms = 80.0 / 3.6
        drag = 0.5 * env.air_density_kg_m3 * cda * v_ms * v_ms
        assert drag == pytest.approx(431.0, rel=0.05), (
            f"Drag at 80 km/h with Michigan default env = {drag:.1f} N; "
            f"audit citation is 431 N. CdA={cda}, rho={env.air_density_kg_m3:.4f}. "
            "If this fails, do NOT bandaid — investigate CdA recalibration "
            "and document the source in CLAUDE.md / DSS."
        )


# ---------------------------------------------------------------------------
# Deprecation of AIR_DENSITY_KG_M3 constant
# ---------------------------------------------------------------------------


class TestPhysicsConstantsDeprecation:
    def test_air_density_constant_still_importable(self) -> None:
        """``physics_constants.AIR_DENSITY_KG_M3`` remains importable so
        legacy callers and tests keep working until they migrate."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Plain attribute lookup should NOT raise.  Modules that
            # opt out of the deprecation warning per usage site continue
            # to function while the migration is in flight.
            from fsae_sim import physics_constants as pc
            # The module itself loads cleanly; the deprecation is on
            # explicit use, not on import.
            assert hasattr(pc, "AIR_DENSITY_KG_M3")
            assert pc.AIR_DENSITY_KG_M3 == pytest.approx(1.225)


# ---------------------------------------------------------------------------
# VehicleParams.environment wiring
# ---------------------------------------------------------------------------


class TestVehicleParamsEnvironment:
    def test_default_vehicle_params_environment_is_michigan(self) -> None:
        """VehicleParams.environment defaults to the Michigan event block."""
        from fsae_sim.vehicle import VehicleParams

        params = VehicleParams(
            mass_kg=288.0,
            frontal_area_m2=1.0,
            drag_coefficient=1.5,
            rolling_resistance=0.015,
            wheelbase_m=1.549,
        )
        env = params.environment
        # Michigan default: ~25 C, ~98500 Pa, 50 % RH, ~290 m elevation.
        assert env.ambient_temp_c == pytest.approx(25.0)
        assert env.ambient_pressure_pa == pytest.approx(98500.0)
        assert env.relative_humidity == pytest.approx(0.50)
        assert env.altitude_m == pytest.approx(290.0)
        assert 1.10 < env.air_density_kg_m3 < 1.20  # Michigan ~1.14 kg/m^3

    def test_vehicle_params_accepts_explicit_environment(self) -> None:
        from fsae_sim.vehicle import VehicleParams
        from fsae_sim.vehicle.environment import EnvironmentConfig

        env = EnvironmentConfig(
            ambient_temp_c=30.0, ambient_pressure_pa=99000.0,
            relative_humidity=0.6, altitude_m=290.0,
        )
        params = VehicleParams(
            mass_kg=288.0,
            frontal_area_m2=1.0,
            drag_coefficient=1.5,
            rolling_resistance=0.015,
            wheelbase_m=1.549,
            environment=env,
        )
        assert params.environment is env
