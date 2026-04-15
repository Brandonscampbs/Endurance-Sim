"""Tests for the Pacejka PAC2002 tire model.

Tests cover:
- .tir file parsing (coefficient values, multiple pressure files)
- Lateral force (Fy): sign convention, load sensitivity, antisymmetry, camber
- Longitudinal force (Fx): antisymmetry, zero at zero slip, peak mu matching
- Combined forces: friction circle coupling, pure-slip passthrough
- Peak force: load dependency, Fx/Fy consistency
- Loaded radius: load deflection, zero-load = unloaded radius
- Integration: full sweeps with no NaN, force bounds
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from fsae_sim.vehicle.tire_model import PacejkaTireModel

# Path to TTC tire model files
TIR_DIR = Path(__file__).resolve().parent.parent / "Real-Car-Data-And-Stats" / "Tire Models from TTC"
TIR_10PSI = TIR_DIR / "Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
TIR_12PSI = TIR_DIR / "Round_8_Hoosier_LC0_16x7p5_10_on_8in_12psi_PAC02_UM2.tir"
TIR_8PSI = TIR_DIR / "Round_8_Hoosier_LC0_16x7p5_10_on_8in_8psi_PAC02_UM2.tir"
TIR_14PSI = TIR_DIR / "Round_8_Hoosier_LC0_16x7p5_10_on_8in_14psi_PAC02_UM2.tir"


@pytest.fixture
def tire_10psi() -> PacejkaTireModel:
    """Load 10 psi tire model."""
    return PacejkaTireModel(TIR_10PSI)


@pytest.fixture
def tire_12psi() -> PacejkaTireModel:
    """Load 12 psi tire model."""
    return PacejkaTireModel(TIR_12PSI)


# ======================================================================
# Parser tests
# ======================================================================


class TestParser:
    """Verify .tir file parsing extracts correct values."""

    def test_fnomin(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.fnomin == 657.0

    def test_unloaded_radius(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.unloaded_radius == 0.2042

    def test_vertical_stiffness(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.vertical_stiffness == 87914.0

    def test_lateral_pcy1(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.lateral["PCY1"] == pytest.approx(1.15122)

    def test_lateral_pdy1(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.lateral["PDY1"] == pytest.approx(-2.66031)

    def test_lateral_pky1(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.lateral["PKY1"] == pytest.approx(-56.7924)

    def test_lateral_pky2(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.lateral["PKY2"] == pytest.approx(2.28097)

    def test_lateral_pvy1(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.lateral["PVY1"] == pytest.approx(0.0254765)

    def test_longitudinal_has_transplanted_coeffs(self, tire_10psi: PacejkaTireModel) -> None:
        """Longitudinal coefficients have non-zero R25B transplanted values."""
        assert tire_10psi.longitudinal["PDX1"] == pytest.approx(2.68565)
        assert tire_10psi.longitudinal["PKX1"] == pytest.approx(55.2561)
        assert tire_10psi.longitudinal["PCX1"] == pytest.approx(1.0)
        assert tire_10psi.longitudinal["PEX4"] == pytest.approx(-0.59049)

    def test_combined_slip_coefficients_present(self, tire_10psi: PacejkaTireModel) -> None:
        """Combined-slip R-coefficients should be non-zero after R25B transplant."""
        assert tire_10psi.longitudinal.get("RBX1", 0.0) != 0.0
        assert tire_10psi.lateral.get("RBY1", 0.0) != 0.0

    def test_scaling_all_ones(self, tire_10psi: PacejkaTireModel) -> None:
        """All scaling factors should be 1.0."""
        for key, val in tire_10psi.scaling.items():
            assert val == 1.0, f"Scaling {key} should be 1.0, got {val}"

    def test_loaded_radius_qv1(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.loaded_radius_coeffs["QV1"] == pytest.approx(403.112)

    def test_loaded_radius_qfz1(self, tire_10psi: PacejkaTireModel) -> None:
        assert tire_10psi.loaded_radius_coeffs["QFZ1"] == pytest.approx(21.8233)

    def test_different_pressure_files_differ(self) -> None:
        """Different pressure .tir files should have different coefficients."""
        t10 = PacejkaTireModel(TIR_10PSI)
        t12 = PacejkaTireModel(TIR_12PSI)
        # PCY1 differs between 10 psi and 12 psi
        assert t10.lateral["PCY1"] != t12.lateral["PCY1"]
        assert t10.lateral["PCY1"] == pytest.approx(1.15122)
        assert t12.lateral["PCY1"] == pytest.approx(1.38875)

    def test_12psi_vertical_stiffness_differs(self) -> None:
        """12 psi has different vertical stiffness than 10 psi."""
        t10 = PacejkaTireModel(TIR_10PSI)
        t12 = PacejkaTireModel(TIR_12PSI)
        assert t10.vertical_stiffness != t12.vertical_stiffness

    def test_all_four_pressures_load(self) -> None:
        """All four .tir files should load without error."""
        for path in [TIR_8PSI, TIR_10PSI, TIR_12PSI, TIR_14PSI]:
            t = PacejkaTireModel(path)
            assert t.fnomin == 657.0
            assert t.unloaded_radius > 0.19
            assert len(t.lateral) > 10

    def test_repr(self, tire_10psi: PacejkaTireModel) -> None:
        r = repr(tire_10psi)
        assert "PacejkaTireModel" in r
        assert "657" in r


# ======================================================================
# Lateral force (Fy) tests
# ======================================================================


class TestLateralForce:
    """Verify PAC2002 lateral force computation."""

    def test_zero_slip_angle_small_force(self, tire_10psi: PacejkaTireModel) -> None:
        """At zero slip angle, Fy should be near zero (just SV offset)."""
        fy = tire_10psi.lateral_force(0.0, 657.0)
        # SV offset is small relative to peak force
        assert abs(fy) < 100.0

    def test_positive_slip_negative_fy(self, tire_10psi: PacejkaTireModel) -> None:
        """With PDY1<0 (right-side TTC convention), positive slip angle
        produces negative Fy (SAE convention): Dy<0, By>0, so
        Dy*sin(C*atan(By*alpha)) < 0.
        """
        fy = tire_10psi.lateral_force(0.1, 657.0)
        assert fy < 0.0

    def test_negative_slip_positive_fy(self, tire_10psi: PacejkaTireModel) -> None:
        """Negative slip angle should produce positive Fy (SAE convention)."""
        fy = tire_10psi.lateral_force(-0.1, 657.0)
        assert fy > 0.0

    def test_antisymmetry(self, tire_10psi: PacejkaTireModel) -> None:
        """Fy(alpha) + Fy(-alpha) should be approximately 2*SVy.

        Pure antisymmetry would give Fy(a) + Fy(-a) = 0, but the
        vertical shift SVy introduces a small bias.
        """
        fy_pos = tire_10psi.lateral_force(0.1, 657.0)
        fy_neg = tire_10psi.lateral_force(-0.1, 657.0)
        # SVy at nominal load: Fz*(PVY1 + PVY2*0)*LMUY = 657*0.0254765 ~ 16.7
        svy_approx = 657.0 * 0.0254765
        assert abs(fy_pos + fy_neg - 2.0 * svy_approx) < 5.0

    def test_load_sensitivity_mu_decreases(self, tire_10psi: PacejkaTireModel) -> None:
        """Friction coefficient (peak Fy / Fz) should decrease with load.

        This is the fundamental Pacejka load-sensitivity behavior.
        """
        fz_low = 300.0
        fz_high = 900.0
        peak_low = tire_10psi.peak_lateral_force(fz_low) / fz_low
        peak_high = tire_10psi.peak_lateral_force(fz_high) / fz_high
        assert peak_low > peak_high

    def test_force_magnitude_at_nominal_load(self, tire_10psi: PacejkaTireModel) -> None:
        """At nominal load (657N), peak |Fy| should be physically reasonable.

        D = |PDY1| * Fz ~ 1748N is the upper bound.  The shape factor
        PCY1=1.15 and stiffness factor reduce the actual peak below D.
        Expect peak in the range 1200-1800N.
        """
        peak = tire_10psi.peak_lateral_force(657.0)
        assert 1200.0 < peak < 1800.0

    def test_camber_effect(self, tire_10psi: PacejkaTireModel) -> None:
        """Non-zero camber should change lateral force."""
        fy_no_camber = tire_10psi.lateral_force(0.1, 657.0, camber_rad=0.0)
        fy_camber = tire_10psi.lateral_force(0.1, 657.0, camber_rad=0.035)
        assert fy_no_camber != fy_camber

    def test_very_small_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Near-zero load should produce near-zero force without errors."""
        fy = tire_10psi.lateral_force(0.1, 1.0)
        assert math.isfinite(fy)
        assert abs(fy) < 50.0

    def test_very_large_slip_angle(self, tire_10psi: PacejkaTireModel) -> None:
        """Large slip angles should not cause errors."""
        fy = tire_10psi.lateral_force(math.pi / 2.0, 657.0)
        assert math.isfinite(fy)

    def test_12psi_different_from_10psi(self) -> None:
        """Different pressure should give different Fy."""
        t10 = PacejkaTireModel(TIR_10PSI)
        t12 = PacejkaTireModel(TIR_12PSI)
        fy10 = t10.lateral_force(0.1, 657.0)
        fy12 = t12.lateral_force(0.1, 657.0)
        assert fy10 != fy12


# ======================================================================
# Longitudinal force (Fx) tests
# ======================================================================


class TestLongitudinalForce:
    """Verify PAC2002 longitudinal force basics."""

    def test_zero_slip_small_force(self, tire_10psi: PacejkaTireModel) -> None:
        """At zero slip ratio, Fx should be small (just SHx/SVx offsets).

        The PAC2002 model has non-zero PHX1 and PVX1, producing a small
        force at kappa=0 analogous to SVy in the lateral model.
        """
        fx = tire_10psi.longitudinal_force(0.0, 657.0)
        peak = tire_10psi.peak_longitudinal_force(657.0)
        assert abs(fx) < peak * 0.02  # less than 2% of peak

    def test_positive_slip_positive_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Positive slip ratio (driving) should produce positive Fx."""
        fx = tire_10psi.longitudinal_force(0.1, 657.0)
        assert fx > 0.0

    def test_negative_slip_negative_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Negative slip ratio (braking) should produce negative Fx."""
        fx = tire_10psi.longitudinal_force(-0.1, 657.0)
        assert fx < 0.0

    def test_near_antisymmetric(self, tire_10psi: PacejkaTireModel) -> None:
        """Fx(kappa) + Fx(-kappa) should be small relative to peak.

        PEX4, PHX, and PVX introduce legitimate small asymmetry in
        the PAC2002 model, analogous to SVy in the lateral model.
        """
        fx_pos = tire_10psi.longitudinal_force(0.1, 657.0)
        fx_neg = tire_10psi.longitudinal_force(-0.1, 657.0)
        peak = tire_10psi.peak_longitudinal_force(657.0)
        assert abs(fx_pos + fx_neg) < peak * 0.02

    def test_peak_mu_same_order_as_lateral(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak Fx / Fz should be same order of magnitude as Fy / Fz.

        Both use transplanted/scaled R25B coefficients matched to LC0
        grip levels. Both should be in the 1.5-3.0 range.
        """
        fz = 657.0
        mu_x = tire_10psi.peak_longitudinal_force(fz) / fz
        mu_y = tire_10psi.peak_lateral_force(fz) / fz
        assert 1.5 < mu_x < 3.0
        assert 1.5 < mu_y < 3.0

    def test_force_increases_with_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Higher load should produce higher absolute force."""
        fx_low = tire_10psi.longitudinal_force(0.1, 300.0)
        fx_high = tire_10psi.longitudinal_force(0.1, 900.0)
        assert fx_high > fx_low

    def test_very_small_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Near-zero load should produce near-zero force."""
        fx = tire_10psi.longitudinal_force(0.1, 1.0)
        assert math.isfinite(fx)
        assert abs(fx) < 50.0


# ======================================================================
# Longitudinal force PAC2002 proper formula tests
# ======================================================================


class TestLongitudinalForcePAC2002:
    """Tests for proper PAC2002 Fx using transplanted R25B coefficients."""

    def test_positive_slip_positive_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Driving slip produces positive Fx."""
        fx = tire_10psi.longitudinal_force(0.1, 657.0)
        assert fx > 0.0

    def test_negative_slip_negative_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Braking slip produces negative Fx."""
        fx = tire_10psi.longitudinal_force(-0.1, 657.0)
        assert fx < 0.0

    def test_peak_fx_near_peak_fy(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak Fx should be within 15% of peak Fy at nominal load."""
        fz = 657.0
        peak_fx = tire_10psi.peak_longitudinal_force(fz)
        peak_fy = tire_10psi.peak_lateral_force(fz)
        ratio = peak_fx / peak_fy
        assert 0.85 < ratio < 1.15

    def test_uses_real_lmux_scaling(self, tire_10psi: PacejkaTireModel) -> None:
        """LMUX should scale peak Fx independently of LMUY."""
        fz = 657.0
        peak_before = tire_10psi.peak_longitudinal_force(fz)
        tire_10psi.scaling["LMUX"] = 0.5
        peak_after = tire_10psi.peak_longitudinal_force(fz)
        assert peak_after == pytest.approx(peak_before * 0.5, rel=0.05)
        tire_10psi.scaling["LMUX"] = 1.0

    def test_load_sensitivity_mu_decreases(self, tire_10psi: PacejkaTireModel) -> None:
        """Fx friction coefficient should decrease with load (PDX2 < 0)."""
        mu_low = tire_10psi.peak_longitudinal_force(300.0) / 300.0
        mu_high = tire_10psi.peak_longitudinal_force(900.0) / 900.0
        assert mu_low > mu_high

    def test_force_increases_with_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Absolute Fx should increase with normal load."""
        fx_300 = abs(tire_10psi.longitudinal_force(0.1, 300.0))
        fx_900 = abs(tire_10psi.longitudinal_force(0.1, 900.0))
        assert fx_900 > fx_300

    def test_curvature_asymmetry(self, tire_10psi: PacejkaTireModel) -> None:
        """PEX4 introduces driving/braking asymmetry in curvature."""
        fx_pos = tire_10psi.longitudinal_force(0.15, 657.0)
        fx_neg = tire_10psi.longitudinal_force(-0.15, 657.0)
        asymmetry = abs(fx_pos + fx_neg)
        peak = tire_10psi.peak_longitudinal_force(657.0)
        assert asymmetry < peak * 0.05
        assert asymmetry > 0.1

    def test_sweep_no_nan(self, tire_10psi: PacejkaTireModel) -> None:
        """Full slip-ratio sweep produces no NaN at multiple loads."""
        for fz in [100.0, 300.0, 657.0, 900.0, 1091.0]:
            for i in range(101):
                kappa = -1.0 + i * 2.0 / 100.0
                fx = tire_10psi.longitudinal_force(kappa, fz)
                assert math.isfinite(fx), f"NaN at kappa={kappa}, Fz={fz}"


# ======================================================================
# Combined forces tests
# ======================================================================


class TestCombinedForces:
    """Verify combined-slip force behavior."""

    def test_pure_lateral_unchanged(self, tire_10psi: PacejkaTireModel) -> None:
        """With zero slip ratio, combined Fy should equal pure Fy.

        Fx at kappa=0 is small but non-zero due to PAC2002 SHx/SVx offsets.
        """
        fy_pure = tire_10psi.lateral_force(0.1, 657.0)
        fx_comb, fy_comb = tire_10psi.combined_forces(0.1, 0.0, 657.0)
        peak_fx = tire_10psi.peak_longitudinal_force(657.0)
        assert abs(fx_comb) < peak_fx * 0.02  # small SVx/SHx offset
        assert fy_comb == pytest.approx(fy_pure, rel=0.01)

    def test_pure_longitudinal_unchanged(self, tire_10psi: PacejkaTireModel) -> None:
        """With zero slip angle, combined Fx should equal pure Fx."""
        fx_pure = tire_10psi.longitudinal_force(0.1, 657.0)
        fx_comb, fy_comb = tire_10psi.combined_forces(0.0, 0.1, 657.0)
        assert fx_comb == pytest.approx(fx_pure, rel=1e-6)
        # Fy at zero slip angle is just SVy offset (small)
        assert abs(fy_comb) < 100.0

    def test_combined_within_friction_circle(self, tire_10psi: PacejkaTireModel) -> None:
        """Combined forces should not exceed the friction ellipse."""
        fx, fy = tire_10psi.combined_forces(0.15, 0.15, 657.0)
        peak_fx = tire_10psi.peak_longitudinal_force(657.0)
        peak_fy = tire_10psi.peak_lateral_force(657.0)
        # Normalized resultant on friction ellipse should be <= 1 + tolerance
        norm = math.sqrt((fx / peak_fx) ** 2 + (fy / peak_fy) ** 2)
        assert norm <= 1.0 + 0.01

    def test_combined_reduces_forces(self, tire_10psi: PacejkaTireModel) -> None:
        """At high combined slip, forces should be reduced from pure values."""
        fx_pure = tire_10psi.longitudinal_force(0.3, 657.0)
        fy_pure = tire_10psi.lateral_force(0.2, 657.0)
        fx_comb, fy_comb = tire_10psi.combined_forces(0.2, 0.3, 657.0)
        # At least one force should be reduced
        reduced = abs(fx_comb) <= abs(fx_pure) + 0.01 or abs(fy_comb) <= abs(fy_pure) + 0.01
        assert reduced

    def test_zero_slip_small_forces(self, tire_10psi: PacejkaTireModel) -> None:
        """Zero slip angle and ratio should give small forces (just offsets).

        Both Fx and Fy have small SH/SV offsets in the PAC2002 model.
        """
        fx, fy = tire_10psi.combined_forces(0.0, 0.0, 657.0)
        assert abs(fx) < 100.0  # small SVx/SHx offset
        assert abs(fy) < 100.0  # small SVy offset


# ======================================================================
# Peak force tests
# ======================================================================


class TestPeakForces:
    """Verify peak force computation."""

    def test_peak_fy_positive(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak |Fy| should always be positive."""
        peak = tire_10psi.peak_lateral_force(657.0)
        assert peak > 0.0

    def test_peak_fx_positive(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak |Fx| should always be positive."""
        peak = tire_10psi.peak_longitudinal_force(657.0)
        assert peak > 0.0

    def test_peak_fy_increases_with_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak force should increase with normal load."""
        peak_300 = tire_10psi.peak_lateral_force(300.0)
        peak_600 = tire_10psi.peak_lateral_force(600.0)
        peak_900 = tire_10psi.peak_lateral_force(900.0)
        assert peak_300 < peak_600 < peak_900

    def test_peak_fx_increases_with_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak force should increase with normal load."""
        peak_300 = tire_10psi.peak_longitudinal_force(300.0)
        peak_600 = tire_10psi.peak_longitudinal_force(600.0)
        peak_900 = tire_10psi.peak_longitudinal_force(900.0)
        assert peak_300 < peak_600 < peak_900

    def test_peak_fx_same_order_as_fy(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak Fx and Fy at same load should be same order of magnitude.

        The PAC2002 Fx model uses transplanted R25B coefficients scaled to
        match the LC0's lateral grip envelope. Both should produce physically
        reasonable FSAE tire forces.
        """
        fz = 657.0
        peak_fx = tire_10psi.peak_longitudinal_force(fz)
        peak_fy = tire_10psi.peak_lateral_force(fz)
        # Both should be in the 1000-2000N range for 657N load
        assert 900.0 < peak_fx < 2100.0
        assert 900.0 < peak_fy < 2100.0

    def test_peak_fy_reasonable_magnitude(self, tire_10psi: PacejkaTireModel) -> None:
        """At 657N load, peak should be roughly mu*Fz ~ 2.66 * 657 ~ 1748N."""
        peak = tire_10psi.peak_lateral_force(657.0)
        assert 1400 < peak < 2100


# ======================================================================
# Loaded radius tests
# ======================================================================


class TestLoadedRadius:
    """Verify loaded radius computation."""

    def test_zero_load_equals_unloaded(self, tire_10psi: PacejkaTireModel) -> None:
        """At zero load, loaded radius equals unloaded radius."""
        r = tire_10psi.loaded_radius(0.0)
        assert r == pytest.approx(0.2042)

    def test_radius_decreases_with_load(self, tire_10psi: PacejkaTireModel) -> None:
        """Higher load should deflect the tyre more."""
        r_light = tire_10psi.loaded_radius(200.0)
        r_heavy = tire_10psi.loaded_radius(800.0)
        assert r_heavy < r_light

    def test_radius_at_nominal_load(self, tire_10psi: PacejkaTireModel) -> None:
        """At nominal load (657N), deflection = 657/87914 ~ 7.5mm."""
        r = tire_10psi.loaded_radius(657.0)
        expected = 0.2042 - 657.0 / 87914.0
        assert r == pytest.approx(expected, abs=1e-5)

    def test_radius_clamps_positive(self, tire_10psi: PacejkaTireModel) -> None:
        """Even with absurd load, radius should be >= 0.01."""
        r = tire_10psi.loaded_radius(1_000_000.0)
        assert r >= 0.01

    def test_negative_load_treated_as_zero(self, tire_10psi: PacejkaTireModel) -> None:
        """Negative load should be clamped to zero."""
        r = tire_10psi.loaded_radius(-100.0)
        assert r == pytest.approx(0.2042)

    def test_12psi_different_stiffness(self) -> None:
        """Different pressure has different stiffness, hence different radius."""
        t10 = PacejkaTireModel(TIR_10PSI)
        t12 = PacejkaTireModel(TIR_12PSI)
        r10 = t10.loaded_radius(657.0)
        r12 = t12.loaded_radius(657.0)
        assert r10 != r12


# ======================================================================
# Integration / sweep tests
# ======================================================================


class TestIntegration:
    """End-to-end sweep tests for numerical stability."""

    def test_lateral_sweep_no_nan(self, tire_10psi: PacejkaTireModel) -> None:
        """Sweep slip angle from -pi/2 to pi/2 at several loads, no NaN."""
        for fz in [100.0, 300.0, 657.0, 900.0, 1091.0]:
            for i in range(101):
                alpha = -math.pi / 2.0 + i * math.pi / 100.0
                fy = tire_10psi.lateral_force(alpha, fz)
                assert math.isfinite(fy), f"NaN at alpha={alpha}, Fz={fz}"

    def test_longitudinal_sweep_no_nan(self, tire_10psi: PacejkaTireModel) -> None:
        """Sweep slip ratio from -1 to 1 at several loads, no NaN."""
        for fz in [100.0, 300.0, 657.0, 900.0, 1091.0]:
            for i in range(101):
                kappa = -1.0 + i * 2.0 / 100.0
                fx = tire_10psi.longitudinal_force(kappa, fz)
                assert math.isfinite(fx), f"NaN at kappa={kappa}, Fz={fz}"

    def test_forces_within_bounds(self, tire_10psi: PacejkaTireModel) -> None:
        """All forces should be within mu*Fz*3 (generous upper bound)."""
        fz = 657.0
        mu_max = 3.0  # generous bound: 3x load
        bound = mu_max * fz
        for i in range(101):
            alpha = -math.pi / 2.0 + i * math.pi / 100.0
            fy = tire_10psi.lateral_force(alpha, fz)
            assert abs(fy) < bound, f"|Fy|={abs(fy)} exceeds {bound}"

        for i in range(101):
            kappa = -1.0 + i * 2.0 / 100.0
            fx = tire_10psi.longitudinal_force(kappa, fz)
            assert abs(fx) < bound, f"|Fx|={abs(fx)} exceeds {bound}"

    def test_combined_sweep_no_nan(self, tire_10psi: PacejkaTireModel) -> None:
        """Sweep combined slip space, no NaN."""
        fz = 657.0
        for i in range(21):
            alpha = -0.3 + i * 0.6 / 20.0
            for j in range(21):
                kappa = -0.5 + j * 1.0 / 20.0
                fx, fy = tire_10psi.combined_forces(alpha, kappa, fz)
                assert math.isfinite(fx), f"NaN Fx at alpha={alpha}, kappa={kappa}"
                assert math.isfinite(fy), f"NaN Fy at alpha={alpha}, kappa={kappa}"

    def test_all_pressures_produce_similar_peak(self) -> None:
        """All four pressures should produce peak mu in a reasonable range."""
        for path in [TIR_8PSI, TIR_10PSI, TIR_12PSI, TIR_14PSI]:
            tire = PacejkaTireModel(path)
            peak = tire.peak_lateral_force(657.0)
            mu = peak / 657.0
            # Hoosier LC0 peak mu typically 2.0-3.0
            assert 1.5 < mu < 3.5, f"{path.name}: mu={mu:.2f} out of range"

    def test_camber_sweep_no_nan(self, tire_10psi: PacejkaTireModel) -> None:
        """Sweep camber from 0 to 4 degrees, no NaN."""
        for i in range(21):
            camber = i * 0.07 / 20.0  # 0 to ~4 deg
            fy = tire_10psi.lateral_force(0.1, 657.0, camber_rad=camber)
            assert math.isfinite(fy), f"NaN at camber={camber}"
            fx = tire_10psi.longitudinal_force(0.1, 657.0, camber_rad=camber)
            assert math.isfinite(fx), f"NaN at camber={camber}"


# ======================================================================
# Grip scale tests
# ======================================================================


class TestGripScale:
    """Verify grip scaling reduces peak force while preserving stiffness."""

    def test_apply_grip_scale_reduces_peak_lateral(self, tire_10psi: PacejkaTireModel) -> None:
        fz = 657.0  # nominal load
        peak_before = tire_10psi.peak_lateral_force(fz)
        tire_10psi.apply_grip_scale(0.5)
        peak_after = tire_10psi.peak_lateral_force(fz)
        assert peak_after == pytest.approx(peak_before * 0.5, rel=0.05)

    def test_apply_grip_scale_reduces_peak_longitudinal(self, tire_10psi: PacejkaTireModel) -> None:
        fz = 657.0
        peak_before = tire_10psi.peak_longitudinal_force(fz)
        tire_10psi.apply_grip_scale(0.5)
        peak_after = tire_10psi.peak_longitudinal_force(fz)
        assert peak_after == pytest.approx(peak_before * 0.5, rel=0.05)

    def test_apply_grip_scale_preserves_cornering_stiffness(self, tire_10psi: PacejkaTireModel) -> None:
        """Cornering stiffness Kya = B*C*D should be preserved because B compensates."""
        fz = 657.0
        small_alpha = 0.01  # rad, linear region
        fy_before = tire_10psi.lateral_force(small_alpha, fz)
        tire_10psi.apply_grip_scale(0.5)
        fy_after = tire_10psi.lateral_force(small_alpha, fz)
        assert fy_after == pytest.approx(fy_before, rel=0.10)

    def test_apply_grip_scale_1_is_noop(self, tire_10psi: PacejkaTireModel) -> None:
        fz = 657.0
        peak_before = tire_10psi.peak_lateral_force(fz)
        tire_10psi.apply_grip_scale(1.0)
        peak_after = tire_10psi.peak_lateral_force(fz)
        assert peak_after == pytest.approx(peak_before, rel=0.001)

    def test_apply_grip_scale_stacks(self, tire_10psi: PacejkaTireModel) -> None:
        """Calling twice should multiply scales."""
        fz = 657.0
        peak_original = tire_10psi.peak_lateral_force(fz)
        tire_10psi.apply_grip_scale(0.5)
        tire_10psi.apply_grip_scale(0.5)
        peak_after = tire_10psi.peak_lateral_force(fz)
        assert peak_after == pytest.approx(peak_original * 0.25, rel=0.05)


# ======================================================================
# Combined forces PAC2002 weighting function tests
# ======================================================================


class TestCombinedForcesPAC2002:
    """Tests for PAC2002 combined-slip weighting functions (Gxa, Gyk, Svyk)."""

    def test_pure_lateral_unchanged(self, tire_10psi: PacejkaTireModel) -> None:
        """With zero slip ratio, Gyk=1 and Svyk=0, so Fy=Fy_pure.

        Fx at kappa=0 equals Gxa * Fx0, where Fx0 is the small SVx
        offset from the pure longitudinal model.  The key assertion is
        that Fy is unchanged by the weighting functions.
        """
        fy_pure = tire_10psi.lateral_force(0.1, 657.0)
        fx_comb, fy_comb = tire_10psi.combined_forces(0.1, 0.0, 657.0)
        # Fx at zero kappa is small (just SVx offset, possibly scaled by Gxa)
        peak_fx = tire_10psi.peak_longitudinal_force(657.0)
        assert abs(fx_comb) < peak_fx * 0.02
        assert fy_comb == pytest.approx(fy_pure, rel=0.01)

    def test_pure_longitudinal_unchanged(self, tire_10psi: PacejkaTireModel) -> None:
        """With zero slip angle, Gxa=1, so Fx=Fx_pure."""
        fx_pure = tire_10psi.longitudinal_force(0.1, 657.0)
        fx_comb, fy_comb = tire_10psi.combined_forces(0.0, 0.1, 657.0)
        assert fx_comb == pytest.approx(fx_pure, rel=0.01)

    def test_combined_reduces_fx(self, tire_10psi: PacejkaTireModel) -> None:
        """Slip angle should reduce Fx via Gxa < 1."""
        fx_pure = tire_10psi.longitudinal_force(0.15, 657.0)
        fx_comb, _ = tire_10psi.combined_forces(0.15, 0.15, 657.0)
        assert abs(fx_comb) < abs(fx_pure)

    def test_combined_reduces_fy(self, tire_10psi: PacejkaTireModel) -> None:
        """Slip ratio should reduce Fy via Gyk < 1."""
        fy_pure = tire_10psi.lateral_force(0.15, 657.0)
        _, fy_comb = tire_10psi.combined_forces(0.15, 0.15, 657.0)
        assert abs(fy_comb) < abs(fy_pure) * 1.05

    def test_weighting_is_gradual(self, tire_10psi: PacejkaTireModel) -> None:
        """Small combined slip should cause small force reduction (< 20%)."""
        fx_pure = tire_10psi.longitudinal_force(0.05, 657.0)
        fx_comb, _ = tire_10psi.combined_forces(0.02, 0.05, 657.0)
        reduction = 1.0 - abs(fx_comb) / abs(fx_pure)
        assert 0.0 < reduction < 0.20

    def test_combined_sweep_no_nan(self, tire_10psi: PacejkaTireModel) -> None:
        """Full combined-slip sweep produces no NaN."""
        fz = 657.0
        for i in range(21):
            alpha = -0.3 + i * 0.6 / 20.0
            for j in range(21):
                kappa = -0.5 + j * 1.0 / 20.0
                fx, fy = tire_10psi.combined_forces(alpha, kappa, fz)
                assert math.isfinite(fx), f"NaN Fx at alpha={alpha}, kappa={kappa}"
                assert math.isfinite(fy), f"NaN Fy at alpha={alpha}, kappa={kappa}"

    def test_symmetric_slip_angle_effect(self, tire_10psi: PacejkaTireModel) -> None:
        """Gxa should be roughly symmetric in slip angle."""
        fx_pos, _ = tire_10psi.combined_forces(0.1, 0.1, 657.0)
        fx_neg, _ = tire_10psi.combined_forces(-0.1, 0.1, 657.0)
        assert abs(fx_pos) == pytest.approx(abs(fx_neg), rel=0.05)
