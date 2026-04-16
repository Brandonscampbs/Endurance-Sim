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
import re
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

    def test_longitudinal_all_zeros(self, tire_10psi: PacejkaTireModel) -> None:
        """All longitudinal coefficients in the 10 psi file are zero."""
        for key, val in tire_10psi.longitudinal.items():
            assert val == 0.0, f"{key} should be 0.0, got {val}"

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
    """Verify mirrored longitudinal force model."""

    def test_zero_slip_zero_force(self, tire_10psi: PacejkaTireModel) -> None:
        """At zero slip ratio, Fx should be zero (symmetric, no offset)."""
        fx = tire_10psi.longitudinal_force(0.0, 657.0)
        assert abs(fx) < 1e-6

    def test_positive_slip_positive_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Positive slip ratio (driving) should produce positive Fx."""
        fx = tire_10psi.longitudinal_force(0.1, 657.0)
        assert fx > 0.0

    def test_negative_slip_negative_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Negative slip ratio (braking) should produce negative Fx."""
        fx = tire_10psi.longitudinal_force(-0.1, 657.0)
        assert fx < 0.0

    def test_antisymmetric(self, tire_10psi: PacejkaTireModel) -> None:
        """Fx(kappa) should be antisymmetric: Fx(k) = -Fx(-k)."""
        fx_pos = tire_10psi.longitudinal_force(0.1, 657.0)
        fx_neg = tire_10psi.longitudinal_force(-0.1, 657.0)
        assert abs(fx_pos + fx_neg) < 1e-6

    def test_peak_mu_same_order_as_lateral(self, tire_10psi: PacejkaTireModel) -> None:
        """Peak Fx / Fz should be same order of magnitude as Fy / Fz.

        The mirrored model uses the same D (peak mu) but the lateral
        model has SVy offset and a negative-By sign structure that
        shifts the effective peak.  Both should be in the 1.5-3.0 range.
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
# Combined forces tests
# ======================================================================


class TestCombinedForces:
    """Verify friction-circle coupling for combined slip."""

    def test_pure_lateral_unchanged(self, tire_10psi: PacejkaTireModel) -> None:
        """With zero slip ratio, combined Fy should equal pure Fy."""
        fy_pure = tire_10psi.lateral_force(0.1, 657.0)
        fx_comb, fy_comb = tire_10psi.combined_forces(0.1, 0.0, 657.0)
        assert abs(fx_comb) < 1e-6
        assert fy_comb == pytest.approx(fy_pure, rel=1e-6)

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

    def test_zero_slip_zero_force(self, tire_10psi: PacejkaTireModel) -> None:
        """Zero slip angle and ratio should give near-zero forces."""
        fx, fy = tire_10psi.combined_forces(0.0, 0.0, 657.0)
        assert abs(fx) < 1e-6
        # Fy has SV offset
        assert abs(fy) < 100.0


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

        The mirrored Fx model uses absolute mu and no SH/SV offsets, while
        the lateral model has signed By (from negative PKY1 / PDY1) and SVy.
        Both should produce physically reasonable FSAE tire forces.
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
# Closed-form peak regression tests (C4, M11, NF-38)
# ======================================================================


class TestClosedFormPeakRegression:
    """Verify closed-form peak matches prior optimizer-based values.

    Baselines were captured before C4/M11 fix using
    ``scipy.optimize.minimize_scalar`` over the Magic Formula, at 10 psi,
    zero camber.  Closed-form |mu * Fz| should land within 5% across
    Fz in [100, 4000] N.
    """

    # Fz -> (peak_fy_optimizer, peak_fx_optimizer) at 10 psi, zero camber.
    OPTIMIZER_BASELINE = {
        100.0: (288.71013907153724, 284.8316575342449),
        300.0: (844.4447916438131, 834.2429178081028),
        500.0: (1371.2652767883617, 1356.6514383561323),
        657.0: (1764.5617304995765, 1747.8236699999945),
        900.0: (2338.163744793499, 2320.4602602739697),
        1500.0: (3571.6551910954286, 3563.652945205479),
        3000.0: (5516.888464383268, 5608.4017808219005),
        4000.0: (5910.142914459523, 6122.945910775905),
    }

    def test_closed_form_peak_lateral_matches_optimizer(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Closed-form peak_lateral_force should match the prior
        optimizer baseline within 5% for Fz in [100, 4000] N.
        """
        for fz, (baseline_fy, _) in self.OPTIMIZER_BASELINE.items():
            peak = tire_10psi.peak_lateral_force(fz)
            assert peak == pytest.approx(baseline_fy, rel=0.05), (
                f"peak_lateral_force({fz}) = {peak}, baseline {baseline_fy}"
            )

    def test_closed_form_peak_longitudinal_matches_optimizer(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Closed-form peak_longitudinal_force should match the prior
        optimizer baseline within 5% for Fz in [100, 4000] N.
        """
        for fz, (_, baseline_fx) in self.OPTIMIZER_BASELINE.items():
            peak = tire_10psi.peak_longitudinal_force(fz)
            assert peak == pytest.approx(baseline_fx, rel=0.05), (
                f"peak_longitudinal_force({fz}) = {peak}, baseline {baseline_fx}"
            )

    def test_peak_lateral_always_positive(self, tire_10psi: PacejkaTireModel) -> None:
        for fz in [50.0, 100.0, 300.0, 1000.0, 5000.0]:
            assert tire_10psi.peak_lateral_force(fz) > 0.0

    def test_peak_longitudinal_always_positive(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        for fz in [50.0, 100.0, 300.0, 1000.0, 5000.0]:
            assert tire_10psi.peak_longitudinal_force(fz) > 0.0

    def test_peak_lateral_monotonic_in_load(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Peak Fy grows monotonically with Fz within FSAE operating range."""
        peaks = [tire_10psi.peak_lateral_force(fz) for fz in [300, 500, 700, 900, 1200]]
        for i in range(1, len(peaks)):
            assert peaks[i] > peaks[i - 1]

    def test_peak_longitudinal_monotonic_in_load(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Peak Fx grows monotonically with Fz within FSAE operating range."""
        peaks = [tire_10psi.peak_longitudinal_force(fz) for fz in [300, 500, 700, 900, 1200]]
        for i in range(1, len(peaks)):
            assert peaks[i] > peaks[i - 1]


# ======================================================================
# Parser validation tests (NF-4)
# ======================================================================


class TestParserValidation:
    """Verify .tir parser warns on garbage and asserts required fields."""

    def test_warns_on_non_numeric_coefficient(self, tmp_path, tire_10psi):
        """Parser should warn when a coefficient value fails float parse."""
        # Copy a valid .tir and inject a garbage line.
        src_text = TIR_10PSI.read_text(encoding="utf-8", errors="replace")
        # Inject a deliberately malformed coefficient inside lateral section.
        bad_line = "PDY1_BOGUS = 0.5e+  $ malformed number"
        # Place the bad line after the [LATERAL_COEFFICIENTS] header.
        injected = src_text.replace(
            "[LATERAL_COEFFICIENTS]",
            "[LATERAL_COEFFICIENTS]\n" + bad_line,
            1,
        )
        bad_path = tmp_path / "bad.tir"
        bad_path.write_text(injected, encoding="utf-8")

        with pytest.warns(UserWarning, match="Skipped coefficients"):
            PacejkaTireModel(bad_path)

    def test_missing_fnomin_raises(self, tmp_path):
        """Missing FNOMIN should raise ValueError."""
        src_text = TIR_10PSI.read_text(encoding="utf-8", errors="replace")
        # Remove FNOMIN line.
        stripped = re.sub(r"\nFNOMIN\s*=.*", "", src_text)
        bad_path = tmp_path / "no_fnomin.tir"
        bad_path.write_text(stripped, encoding="utf-8")

        with pytest.raises(ValueError, match="FNOMIN"):
            PacejkaTireModel(bad_path)

    def test_missing_unloaded_radius_raises(self, tmp_path):
        """Missing UNLOADED_RADIUS should raise ValueError."""
        src_text = TIR_10PSI.read_text(encoding="utf-8", errors="replace")
        stripped = re.sub(r"\nUNLOADED_RADIUS\s*=.*", "", src_text)
        bad_path = tmp_path / "no_r0.tir"
        bad_path.write_text(stripped, encoding="utf-8")

        with pytest.raises(ValueError, match="UNLOADED_RADIUS"):
            PacejkaTireModel(bad_path)

    def test_missing_vertical_stiffness_raises(self, tmp_path):
        """Missing VERTICAL_STIFFNESS should raise ValueError."""
        src_text = TIR_10PSI.read_text(encoding="utf-8", errors="replace")
        stripped = re.sub(r"\nVERTICAL_STIFFNESS\s*=.*", "", src_text)
        bad_path = tmp_path / "no_kz.tir"
        bad_path.write_text(stripped, encoding="utf-8")

        with pytest.raises(ValueError, match="VERTICAL_STIFFNESS"):
            PacejkaTireModel(bad_path)


# ======================================================================
# Symmetric E-clamp tests (NF-35)
# ======================================================================


class TestCurvatureFactorClamp:
    """Verify ey and ex are clamped symmetrically to [-1, 1]."""

    def test_ey_clamp_lower_bound_produces_finite_force(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Force ey below -1 by inflating PEY2*dfz negatively at high load.

        With PEY1 + PEY2*dfz potentially < -1 at high dfz, the un-clamped
        model would let ey saturate below -1 and produce a discontinuous
        force jump.  The fix clamps ey in [-1, 1] symmetrically.
        """
        fz = 4000.0  # far above fnomin=657, dfz >> 0
        fy = tire_10psi.lateral_force(0.1, fz)
        assert math.isfinite(fy)

    def test_lateral_force_continuous_through_high_dfz(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Sweep Fz up to stress the ey expression.  Force should remain
        bounded and continuous -- no discontinuous jumps that indicate
        an unclamped ey saturating at the Magic Formula's asymptote.
        """
        alpha = 0.1
        prev = None
        for fz in range(100, 5000, 50):
            fy = tire_10psi.lateral_force(alpha, float(fz))
            assert math.isfinite(fy)
            if prev is not None:
                # Consecutive Fz steps of 50N should not jump by more than
                # a few hundred N -- a catastrophic jump would indicate
                # the ey clamp issue.
                assert abs(fy - prev) < 500.0, (
                    f"Discontinuity at Fz={fz}: {prev} -> {fy}"
                )
            prev = fy

    def test_longitudinal_force_continuous_through_high_dfz(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Sweep Fz at fixed slip ratio; ex clamp must be symmetric to
        avoid sign-asymmetric force response."""
        kappa = 0.1
        prev = None
        for fz in range(100, 5000, 50):
            fx = tire_10psi.longitudinal_force(kappa, float(fz))
            assert math.isfinite(fx)
            if prev is not None:
                assert abs(fx - prev) < 500.0, (
                    f"Discontinuity at Fz={fz}: {prev} -> {fx}"
                )
            prev = fx


# ======================================================================
# NF-18 regression: atan(fz / PKY2*Fz0) with magnitude
# ======================================================================


class TestCorneringStiffnessSignGuard:
    """Verify cornering stiffness uses |PKY2*Fz0| in the atan argument."""

    def test_kya_same_sign_with_positive_or_negative_pky2(
        self, tire_10psi: PacejkaTireModel
    ) -> None:
        """Swapping PKY2 sign should not invert the cornering-stiffness sign.

        PAC2002 expects the magnitude in the atan denominator.  A sign flip
        would propagate and invert downstream lateral-force direction.
        """
        fz = 657.0
        alpha = 0.01
        fy_baseline = tire_10psi.lateral_force(alpha, fz)

        # Force negative PKY2 and re-evaluate.
        original_pky2 = tire_10psi.lateral["PKY2"]
        tire_10psi.lateral["PKY2"] = -abs(original_pky2)
        try:
            fy_neg_pky2 = tire_10psi.lateral_force(alpha, fz)
        finally:
            tire_10psi.lateral["PKY2"] = original_pky2

        # Both should have the same sign (stiffness magnitude preserved).
        assert math.copysign(1.0, fy_baseline) == math.copysign(1.0, fy_neg_pky2)
