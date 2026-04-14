"""Tests for CorneringSolver.

Uses mocked tire and load-transfer models to isolate the bisection logic
and verify cornering speed calculations against analytical solutions.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from fsae_sim.vehicle.cornering_solver import CorneringSolver


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_linear_tire(mu: float = 1.3) -> MagicMock:
    """Create a mock tire where peak_lateral_force = mu * Fz.

    This is the simplest possible tire model: perfectly linear grip
    proportional to normal load with no camber sensitivity.
    """
    tire = MagicMock()
    tire.peak_lateral_force.side_effect = lambda fz, camber=0.0: mu * fz
    return tire


def make_static_load_transfer(mass_kg: float) -> MagicMock:
    """Create a mock load transfer with equal quarter-car loads and no transfer.

    Roll stiffness set extremely high so roll angle is negligible.
    """
    quarter = mass_kg * 9.81 / 4.0
    lt = MagicMock()
    lt.tire_loads.return_value = (quarter, quarter, quarter, quarter)
    lt.roll_stiffness_front = 1e6  # very stiff -> negligible roll
    lt.roll_stiffness_rear = 1e6
    return lt


def make_solver(
    mu: float = 1.3,
    mass_kg: float = 278.0,
    tire: MagicMock | None = None,
    lt: MagicMock | None = None,
) -> CorneringSolver:
    """Create a CorneringSolver with linear tire and static loads by default."""
    if tire is None:
        tire = make_linear_tire(mu)
    if lt is None:
        lt = make_static_load_transfer(mass_kg)
    return CorneringSolver(
        tire_model=tire,
        load_transfer=lt,
        mass_kg=mass_kg,
        static_camber_front_rad=0.0,
        static_camber_rear_rad=0.0,
        roll_camber_front=0.0,
        roll_camber_rear=0.0,
    )


def analytical_speed(mu: float, curvature: float) -> float:
    """Analytical cornering speed for linear tire, no load transfer.

    v = sqrt(mu * g / curvature)
    """
    return math.sqrt(mu * 9.81 / curvature)


# ---------------------------------------------------------------------------
# 1. Straight segments: zero / near-zero curvature -> inf
# ---------------------------------------------------------------------------


class TestStraightSegment:
    """Straight or near-straight segments should return infinite speed."""

    def test_zero_curvature_returns_inf(self) -> None:
        solver = make_solver()
        assert solver.max_cornering_speed(0.0) == math.inf

    def test_near_zero_curvature_returns_inf(self) -> None:
        solver = make_solver()
        assert solver.max_cornering_speed(1e-7) == math.inf

    def test_negative_zero_curvature_returns_inf(self) -> None:
        solver = make_solver()
        assert solver.max_cornering_speed(-0.0) == math.inf

    def test_just_below_threshold_returns_inf(self) -> None:
        solver = make_solver()
        # Threshold is 1e-6; value just below should still be inf
        assert solver.max_cornering_speed(9.9e-7) == math.inf


# ---------------------------------------------------------------------------
# 2. Analytical match: linear tire, no load transfer
# ---------------------------------------------------------------------------


class TestAnalyticalMatch:
    """With linear tire (Fy = mu * Fz) and no load transfer,
    the solver must match v = sqrt(mu * g / kappa) within 0.15 m/s.
    """

    @pytest.mark.parametrize(
        "curvature",
        [0.2, 1.0 / 15.0, 0.01, 0.05],
        ids=["kappa=0.2", "kappa=1/15", "kappa=0.01", "kappa=0.05"],
    )
    def test_matches_analytical_solution(self, curvature: float) -> None:
        mu = 1.3
        solver = make_solver(mu=mu)
        v = solver.max_cornering_speed(curvature)
        v_expected = analytical_speed(mu, curvature)
        assert v == pytest.approx(v_expected, abs=0.15), (
            f"curvature={curvature}: got {v:.3f}, expected {v_expected:.3f}"
        )

    def test_different_mass_same_speed(self) -> None:
        """With linear tire and no load transfer, mass cancels out.
        Two different masses should give same cornering speed.
        """
        mu = 1.3
        solver_light = make_solver(mu=mu, mass_kg=200.0)
        solver_heavy = make_solver(mu=mu, mass_kg=350.0)
        curvature = 0.05
        v_light = solver_light.max_cornering_speed(curvature)
        v_heavy = solver_heavy.max_cornering_speed(curvature)
        assert v_light == pytest.approx(v_heavy, abs=0.15)


# ---------------------------------------------------------------------------
# 3. Higher mu -> higher speed
# ---------------------------------------------------------------------------


class TestMuEffect:
    """Higher friction coefficient must produce higher cornering speed."""

    def test_higher_mu_gives_higher_speed(self) -> None:
        curvature = 0.05
        solver_low = make_solver(mu=1.0)
        solver_high = make_solver(mu=1.5)
        v_low = solver_low.max_cornering_speed(curvature)
        v_high = solver_high.max_cornering_speed(curvature)
        assert v_high > v_low

    def test_double_mu_gives_sqrt2_speed(self) -> None:
        """For linear tire: v ~ sqrt(mu), so 4x mu -> 2x speed."""
        curvature = 0.05
        mu_base = 1.0
        mu_quad = 4.0
        solver_base = make_solver(mu=mu_base)
        solver_quad = make_solver(mu=mu_quad)
        v_base = solver_base.max_cornering_speed(curvature)
        v_quad = solver_quad.max_cornering_speed(curvature)
        assert v_quad == pytest.approx(2.0 * v_base, abs=0.15)


# ---------------------------------------------------------------------------
# 4. Negative curvature = same magnitude as positive
# ---------------------------------------------------------------------------


class TestCurvatureSign:
    """Cornering speed depends on |curvature|; sign should not matter."""

    def test_negative_curvature_same_as_positive(self) -> None:
        solver = make_solver()
        curvature = 0.05
        v_pos = solver.max_cornering_speed(curvature)
        v_neg = solver.max_cornering_speed(-curvature)
        assert v_pos == pytest.approx(v_neg, abs=1e-6)

    @pytest.mark.parametrize("curvature", [0.01, 0.1, 0.5])
    def test_sign_invariance_parametrized(self, curvature: float) -> None:
        solver = make_solver()
        v_pos = solver.max_cornering_speed(curvature)
        v_neg = solver.max_cornering_speed(-curvature)
        assert v_pos == pytest.approx(v_neg, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Downforce benefit: speed-dependent loads increase grip
# ---------------------------------------------------------------------------


class TestDownforceBenefit:
    """When load transfer model adds v^2-dependent downforce,
    max cornering speed should exceed the no-downforce case.
    """

    def test_downforce_increases_speed(self) -> None:
        mu = 1.3
        mass_kg = 278.0
        curvature = 0.05
        quarter_static = mass_kg * 9.81 / 4.0

        # No-downforce baseline
        solver_no_df = make_solver(mu=mu, mass_kg=mass_kg)
        v_no_df = solver_no_df.max_cornering_speed(curvature)

        # With downforce: each tire gets extra load proportional to v^2
        cla = 2.18  # m^2 downforce coefficient * area
        rho = 1.225

        def tire_loads_with_df(speed, lat_g, lon_g):
            df_per_tire = 0.5 * rho * speed * speed * cla / 4.0
            load = quarter_static + df_per_tire
            return (load, load, load, load)

        lt_df = MagicMock()
        lt_df.tire_loads.side_effect = tire_loads_with_df
        lt_df.roll_stiffness_front = 1e6
        lt_df.roll_stiffness_rear = 1e6

        solver_df = make_solver(mu=mu, mass_kg=mass_kg, lt=lt_df)
        v_df = solver_df.max_cornering_speed(curvature)

        assert v_df > v_no_df


# ---------------------------------------------------------------------------
# 6. Downforce benefit grows with speed (wider corners)
# ---------------------------------------------------------------------------


class TestDownforceScaling:
    """Downforce benefit should be larger for wider corners (higher speed)
    than for tight corners (lower speed) since df ~ v^2.
    """

    def test_benefit_larger_for_wider_corner(self) -> None:
        mu = 1.3
        mass_kg = 278.0
        quarter_static = mass_kg * 9.81 / 4.0
        cla = 2.18
        rho = 1.225

        def tire_loads_with_df(speed, lat_g, lon_g):
            df_per_tire = 0.5 * rho * speed * speed * cla / 4.0
            load = quarter_static + df_per_tire
            return (load, load, load, load)

        lt_df = MagicMock()
        lt_df.tire_loads.side_effect = tire_loads_with_df
        lt_df.roll_stiffness_front = 1e6
        lt_df.roll_stiffness_rear = 1e6

        solver_no_df = make_solver(mu=mu, mass_kg=mass_kg)
        solver_df = make_solver(mu=mu, mass_kg=mass_kg, lt=lt_df)

        # Tight corner (high curvature, low speed)
        kappa_tight = 0.2
        v_tight_no_df = solver_no_df.max_cornering_speed(kappa_tight)
        v_tight_df = solver_df.max_cornering_speed(kappa_tight)
        benefit_tight = v_tight_df - v_tight_no_df

        # Wide corner (low curvature, high speed)
        kappa_wide = 0.02
        v_wide_no_df = solver_no_df.max_cornering_speed(kappa_wide)
        v_wide_df = solver_df.max_cornering_speed(kappa_wide)
        benefit_wide = v_wide_df - v_wide_no_df

        assert benefit_wide > benefit_tight


# ---------------------------------------------------------------------------
# 7. mu_scale parameter
# ---------------------------------------------------------------------------


class TestMuScale:
    """mu_scale should linearly scale available grip."""

    def test_mu_scale_below_one_reduces_speed(self) -> None:
        solver = make_solver(mu=1.3)
        curvature = 0.05
        v_full = solver.max_cornering_speed(curvature, mu_scale=1.0)
        v_half = solver.max_cornering_speed(curvature, mu_scale=0.5)
        assert v_half < v_full

    def test_mu_scale_above_one_increases_speed(self) -> None:
        solver = make_solver(mu=1.3)
        curvature = 0.05
        v_full = solver.max_cornering_speed(curvature, mu_scale=1.0)
        v_extra = solver.max_cornering_speed(curvature, mu_scale=1.5)
        assert v_extra > v_full

    def test_mu_scale_half_matches_analytical(self) -> None:
        """mu_scale=0.5 with mu=1.3 should match effective_mu=0.65."""
        mu = 1.3
        curvature = 0.05
        solver = make_solver(mu=mu)
        v = solver.max_cornering_speed(curvature, mu_scale=0.5)
        v_expected = analytical_speed(mu * 0.5, curvature)
        assert v == pytest.approx(v_expected, abs=0.15)

    def test_mu_scale_preserves_sign_invariance(self) -> None:
        solver = make_solver(mu=1.3)
        curvature = 0.05
        v_pos = solver.max_cornering_speed(curvature, mu_scale=0.7)
        v_neg = solver.max_cornering_speed(-curvature, mu_scale=0.7)
        assert v_pos == pytest.approx(v_neg, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and edge-case behavior."""

    def test_very_tight_corner_low_speed(self) -> None:
        """Curvature=1.0 (1m radius) should give very low speed."""
        solver = make_solver(mu=1.3)
        v = solver.max_cornering_speed(1.0)
        assert v < 5.0
        assert v > 0.0

    def test_speed_monotonically_decreases_with_curvature(self) -> None:
        """Higher curvature must give lower speed."""
        solver = make_solver(mu=1.3)
        curvatures = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        speeds = [solver.max_cornering_speed(k) for k in curvatures]
        for i in range(len(speeds) - 1):
            assert speeds[i] > speeds[i + 1], (
                f"Non-monotonic at k={curvatures[i]}: "
                f"v={speeds[i]:.3f} <= v={speeds[i+1]:.3f}"
            )

    def test_always_positive_and_finite(self) -> None:
        """Non-straight curvatures always produce positive finite speed."""
        solver = make_solver(mu=1.3)
        for curvature in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            v = solver.max_cornering_speed(curvature)
            assert v > 0.0
            assert math.isfinite(v)

    def test_result_clamped_within_search_bounds(self) -> None:
        """Result should lie within [V_LOW, V_HIGH]."""
        solver = make_solver(mu=1.3)
        v = solver.max_cornering_speed(0.05)
        assert v >= CorneringSolver._V_LOW
        assert v <= CorneringSolver._V_HIGH


# ---------------------------------------------------------------------------
# 9. Degressive tire: load transfer reduces speed
# ---------------------------------------------------------------------------


class TestDegressiveTire:
    """With a degressive tire model (Fy = mu*Fz - k*Fz^2), load transfer
    reduces total grip because the heavily-loaded outside tire gains less
    than the lightly-loaded inside tire loses.
    """

    def test_load_transfer_reduces_speed_with_degressive_tire(self) -> None:
        mu = 1.3
        degression = 0.0003  # N^-1
        mass_kg = 278.0

        def degressive_peak(fz: float, camber: float = 0.0) -> float:
            return max(mu * fz - degression * fz * fz, 0.0)

        tire = MagicMock()
        tire.peak_lateral_force.side_effect = degressive_peak

        quarter = mass_kg * 9.81 / 4.0
        curvature = 0.05

        # No load transfer (equal loads)
        lt_equal = MagicMock()
        lt_equal.tire_loads.return_value = (quarter, quarter, quarter, quarter)
        lt_equal.roll_stiffness_front = 1e6
        lt_equal.roll_stiffness_rear = 1e6

        solver_equal = CorneringSolver(
            tire_model=tire,
            load_transfer=lt_equal,
            mass_kg=mass_kg,
            static_camber_front_rad=0.0,
            static_camber_rear_rad=0.0,
            roll_camber_front=0.0,
            roll_camber_rear=0.0,
        )
        v_equal = solver_equal.max_cornering_speed(curvature)

        # With load transfer: outside gets more, inside gets less
        delta = quarter * 0.3  # 30% transfer

        def tire_loads_transfer(speed, lat_g, lon_g):
            return (
                quarter + delta,  # FL outside
                quarter - delta,  # FR inside
                quarter + delta,  # RL outside
                quarter - delta,  # RR inside
            )

        lt_transfer = MagicMock()
        lt_transfer.tire_loads.side_effect = tire_loads_transfer
        lt_transfer.roll_stiffness_front = 1e6
        lt_transfer.roll_stiffness_rear = 1e6

        solver_transfer = CorneringSolver(
            tire_model=tire,
            load_transfer=lt_transfer,
            mass_kg=mass_kg,
            static_camber_front_rad=0.0,
            static_camber_rear_rad=0.0,
            roll_camber_front=0.0,
            roll_camber_rear=0.0,
        )
        v_transfer = solver_transfer.max_cornering_speed(curvature)

        assert v_transfer < v_equal

    def test_more_degression_more_speed_loss(self) -> None:
        """Stronger degression should cause greater speed reduction."""
        mass_kg = 278.0
        quarter = mass_kg * 9.81 / 4.0
        delta = quarter * 0.3
        curvature = 0.05

        def make_degressive_solver(degression: float) -> CorneringSolver:
            def degressive_peak(fz: float, camber: float = 0.0) -> float:
                return max(1.3 * fz - degression * fz * fz, 0.0)

            tire = MagicMock()
            tire.peak_lateral_force.side_effect = degressive_peak

            def tire_loads_transfer(speed, lat_g, lon_g):
                return (
                    quarter + delta,
                    quarter - delta,
                    quarter + delta,
                    quarter - delta,
                )

            lt = MagicMock()
            lt.tire_loads.side_effect = tire_loads_transfer
            lt.roll_stiffness_front = 1e6
            lt.roll_stiffness_rear = 1e6

            return CorneringSolver(
                tire_model=tire,
                load_transfer=lt,
                mass_kg=mass_kg,
                static_camber_front_rad=0.0,
                static_camber_rear_rad=0.0,
                roll_camber_front=0.0,
                roll_camber_rear=0.0,
            )

        solver_mild = make_degressive_solver(0.0001)
        solver_strong = make_degressive_solver(0.0005)

        v_mild = solver_mild.max_cornering_speed(curvature)
        v_strong = solver_strong.max_cornering_speed(curvature)

        assert v_strong < v_mild


# ---------------------------------------------------------------------------
# 10. Linear tire unaffected by load transfer
# ---------------------------------------------------------------------------


class TestLinearTireLoadTransfer:
    """With a perfectly linear tire (Fy = mu * Fz), load transfer should
    NOT change the total grip: the outside tire's gain exactly offsets
    the inside tire's loss.
    """

    def test_load_transfer_does_not_change_speed(self) -> None:
        mu = 1.3
        mass_kg = 278.0
        quarter = mass_kg * 9.81 / 4.0
        curvature = 0.05

        # Equal loads
        solver_equal = make_solver(mu=mu, mass_kg=mass_kg)
        v_equal = solver_equal.max_cornering_speed(curvature)

        # Asymmetric loads with same total
        delta = quarter * 0.4

        def tire_loads_transfer(speed, lat_g, lon_g):
            return (
                quarter + delta,
                quarter - delta,
                quarter + delta,
                quarter - delta,
            )

        lt_transfer = MagicMock()
        lt_transfer.tire_loads.side_effect = tire_loads_transfer
        lt_transfer.roll_stiffness_front = 1e6
        lt_transfer.roll_stiffness_rear = 1e6

        solver_transfer = make_solver(mu=mu, mass_kg=mass_kg, lt=lt_transfer)
        v_transfer = solver_transfer.max_cornering_speed(curvature)

        assert v_transfer == pytest.approx(v_equal, abs=0.15)

    def test_extreme_transfer_still_matches(self) -> None:
        """Even with 80% load transfer, linear tire gives same speed."""
        mu = 1.3
        mass_kg = 278.0
        quarter = mass_kg * 9.81 / 4.0
        curvature = 0.05

        solver_equal = make_solver(mu=mu, mass_kg=mass_kg)
        v_equal = solver_equal.max_cornering_speed(curvature)

        delta = quarter * 0.8

        def tire_loads_extreme(speed, lat_g, lon_g):
            return (
                quarter + delta,
                quarter - delta,
                quarter + delta,
                quarter - delta,
            )

        lt_extreme = MagicMock()
        lt_extreme.tire_loads.side_effect = tire_loads_extreme
        lt_extreme.roll_stiffness_front = 1e6
        lt_extreme.roll_stiffness_rear = 1e6

        solver_extreme = make_solver(mu=mu, mass_kg=mass_kg, lt=lt_extreme)
        v_extreme = solver_extreme.max_cornering_speed(curvature)

        assert v_extreme == pytest.approx(v_equal, abs=0.15)


# ---------------------------------------------------------------------------
# Additional coverage: class constants and construction
# ---------------------------------------------------------------------------


class TestClassConstants:
    """Verify class-level constants are set correctly."""

    def test_gravity_constant(self) -> None:
        assert CorneringSolver.GRAVITY == 9.81

    def test_velocity_bounds(self) -> None:
        assert CorneringSolver._V_LOW == 0.5
        assert CorneringSolver._V_HIGH == 50.0

    def test_iteration_count(self) -> None:
        assert CorneringSolver._ITERATIONS == 30

    def test_curvature_threshold(self) -> None:
        assert CorneringSolver._CURVATURE_THRESHOLD == 1e-6
