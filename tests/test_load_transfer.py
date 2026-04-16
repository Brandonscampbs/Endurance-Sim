"""Tests for LoadTransferModel.

Validates static loads, aerodynamic downforce, longitudinal and lateral
load transfer, and combined tire loads against hand-calculated values
derived from CT-16EV DSS parameters.
"""

from __future__ import annotations

import math

import pytest

from fsae_sim.vehicle.load_transfer import (
    AIR_DENSITY,
    GRAVITY,
    LoadTransferModel,
)
from fsae_sim.vehicle.vehicle import SuspensionConfig, VehicleParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vehicle() -> VehicleParams:
    """CT-16EV vehicle parameters from DSS."""
    return VehicleParams(
        mass_kg=278.0,
        frontal_area_m2=1.0,
        drag_coefficient=1.502,
        rolling_resistance=0.015,
        wheelbase_m=1.549,
        downforce_coefficient=2.18,
    )


@pytest.fixture
def suspension() -> SuspensionConfig:
    """CT-16EV suspension config from DSS."""
    return SuspensionConfig(
        roll_stiffness_front_nm_per_deg=238.0,
        roll_stiffness_rear_nm_per_deg=258.0,
        roll_center_height_front_mm=88.9,
        roll_center_height_rear_mm=63.5,
        roll_camber_front_deg_per_deg=-0.5,
        roll_camber_rear_deg_per_deg=-0.554,
        front_track_mm=1194.0,
        rear_track_mm=1168.0,
    )


@pytest.fixture
def model(vehicle: VehicleParams, suspension: SuspensionConfig) -> LoadTransferModel:
    """LoadTransferModel with CT-16EV defaults."""
    return LoadTransferModel(
        vehicle=vehicle,
        suspension=suspension,
        cg_height_m=0.2794,
        weight_dist_front=0.53,
        downforce_dist_front=0.61,
    )


# ---------------------------------------------------------------------------
# Static loads
# ---------------------------------------------------------------------------


class TestStaticLoads:
    """Static weight distribution tests."""

    def test_static_load_values(self, model: LoadTransferModel) -> None:
        """FL=FR=722.70, RL=RR=640.89 from 278*9.81, 53/47 split."""
        fl, fr, rl, rr = model.static_loads()
        assert fl == pytest.approx(722.70, abs=0.01)
        assert fr == pytest.approx(722.70, abs=0.01)
        assert rl == pytest.approx(640.89, abs=0.01)
        assert rr == pytest.approx(640.89, abs=0.01)

    def test_static_sum_equals_weight(self, model: LoadTransferModel) -> None:
        """Sum of all four tires must equal total vehicle weight."""
        fl, fr, rl, rr = model.static_loads()
        assert (fl + fr + rl + rr) == pytest.approx(278.0 * GRAVITY, abs=0.01)

    def test_static_left_right_symmetry(self, model: LoadTransferModel) -> None:
        """Left and right loads must be equal on each axle."""
        fl, fr, rl, rr = model.static_loads()
        assert fl == pytest.approx(fr, abs=1e-10)
        assert rl == pytest.approx(rr, abs=1e-10)

    def test_static_front_heavier(self, model: LoadTransferModel) -> None:
        """53% front weight distribution means front tires carry more."""
        fl, fr, rl, rr = model.static_loads()
        assert fl > rl
        assert fr > rr


# ---------------------------------------------------------------------------
# Aero loads
# ---------------------------------------------------------------------------


class TestAeroLoads:
    """Aerodynamic downforce tests."""

    def test_aero_zero_speed(self, model: LoadTransferModel) -> None:
        """No downforce at standstill."""
        df_f, df_r = model.aero_loads(0.0)
        assert df_f == pytest.approx(0.0, abs=1e-10)
        assert df_r == pytest.approx(0.0, abs=1e-10)

    def test_aero_80kph_values(self, model: LoadTransferModel) -> None:
        """At 80 kph: front=402.22, rear=257.16."""
        speed = 80.0 / 3.6
        df_f, df_r = model.aero_loads(speed)
        assert df_f == pytest.approx(402.22, abs=0.01)
        assert df_r == pytest.approx(257.16, abs=0.01)

    def test_aero_sum_equals_total(self, model: LoadTransferModel) -> None:
        """Front + rear must equal 0.5*rho*v^2*ClA."""
        speed = 80.0 / 3.6
        df_f, df_r = model.aero_loads(speed)
        q = 0.5 * AIR_DENSITY * speed * speed
        expected_total = q * 2.18
        assert (df_f + df_r) == pytest.approx(expected_total, abs=0.01)

    def test_aero_v_squared_scaling(self, model: LoadTransferModel) -> None:
        """Downforce scales with v^2: doubling speed -> 4x downforce."""
        df_f1, df_r1 = model.aero_loads(10.0)
        df_f2, df_r2 = model.aero_loads(20.0)
        assert (df_f2 + df_r2) == pytest.approx(4.0 * (df_f1 + df_r1), abs=0.01)

    def test_aero_61_39_split(self, model: LoadTransferModel) -> None:
        """Front/rear distribution is 61/39."""
        speed = 80.0 / 3.6
        df_f, df_r = model.aero_loads(speed)
        total = df_f + df_r
        assert (df_f / total) == pytest.approx(0.61, abs=0.001)
        assert (df_r / total) == pytest.approx(0.39, abs=0.001)


# ---------------------------------------------------------------------------
# Longitudinal transfer
# ---------------------------------------------------------------------------


class TestLongitudinalTransfer:
    """Longitudinal load transfer tests."""

    def test_longitudinal_zero(self, model: LoadTransferModel) -> None:
        """Zero acceleration produces zero transfer."""
        assert model.longitudinal_transfer(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_longitudinal_1g(self, model: LoadTransferModel) -> None:
        """1g forward: delta = 278*1.0*9.81*0.2794/1.549 = 491.91 N."""
        assert model.longitudinal_transfer(1.0) == pytest.approx(491.91, abs=0.01)

    def test_longitudinal_minus_1p5g(self, model: LoadTransferModel) -> None:
        """-1.5g braking: delta = -737.87 N."""
        assert model.longitudinal_transfer(-1.5) == pytest.approx(-737.87, abs=0.01)

    def test_longitudinal_linearity(self, model: LoadTransferModel) -> None:
        """Transfer is linear in acceleration."""
        d1 = model.longitudinal_transfer(1.0)
        d2 = model.longitudinal_transfer(2.0)
        assert d2 == pytest.approx(2.0 * d1, abs=0.01)

    def test_longitudinal_sign(self, model: LoadTransferModel) -> None:
        """Positive accel => positive transfer (rear gains)."""
        assert model.longitudinal_transfer(0.5) > 0
        assert model.longitudinal_transfer(-0.5) < 0


# ---------------------------------------------------------------------------
# Lateral transfer
# ---------------------------------------------------------------------------


class TestLateralTransfer:
    """Lateral load transfer decomposition tests."""

    def test_lateral_zero(self, model: LoadTransferModel) -> None:
        """Zero lateral acceleration produces zero transfer."""
        df, dr = model.lateral_transfer(0.0, 0.0)
        assert df == pytest.approx(0.0, abs=1e-10)
        assert dr == pytest.approx(0.0, abs=1e-10)

    def test_lateral_1g_front_value(self, model: LoadTransferModel) -> None:
        """1g lateral at 0 speed: front transfer = 329.49 N."""
        df, dr = model.lateral_transfer(1.0, 0.0)
        assert df == pytest.approx(329.49, abs=0.01)

    def test_lateral_1g_rear_value(self, model: LoadTransferModel) -> None:
        """1g lateral at 0 speed: rear transfer = 315.55 N."""
        df, dr = model.lateral_transfer(1.0, 0.0)
        assert dr == pytest.approx(315.55, abs=0.01)

    def test_lateral_front_greater_than_rear(self, model: LoadTransferModel) -> None:
        """53% front weight distribution shifts more lateral transfer to front axle."""
        df, dr = model.lateral_transfer(1.0, 0.0)
        assert df > dr

    def test_lateral_moment_balance(self, model: LoadTransferModel) -> None:
        """delta_f*track_f + delta_r*track_r = m*g*lat_g*cg_height."""
        df, dr = model.lateral_transfer(1.0, 0.0)
        check = df * model.front_track + dr * model.rear_track
        expected = 278.0 * GRAVITY * 1.0 * 0.2794
        assert check == pytest.approx(expected, abs=0.2)

    def test_lateral_sign_symmetry(self, model: LoadTransferModel) -> None:
        """Magnitude of transfer is the same for left and right turns."""
        df_pos, dr_pos = model.lateral_transfer(1.0, 0.0)
        df_neg, dr_neg = model.lateral_transfer(-1.0, 0.0)
        assert df_pos == pytest.approx(df_neg, abs=1e-10)
        assert dr_pos == pytest.approx(dr_neg, abs=1e-10)

    def test_lateral_linearity(self, model: LoadTransferModel) -> None:
        """Transfer is linear in lateral_g."""
        df1, dr1 = model.lateral_transfer(0.5, 0.0)
        df2, dr2 = model.lateral_transfer(1.0, 0.0)
        assert df2 == pytest.approx(2.0 * df1, abs=0.01)
        assert dr2 == pytest.approx(2.0 * dr1, abs=0.01)


# ---------------------------------------------------------------------------
# Combined tire loads
# ---------------------------------------------------------------------------


class TestTireLoads:
    """Combined tire load tests."""

    def test_stationary_equals_static(self, model: LoadTransferModel) -> None:
        """At standstill with no acceleration, tire_loads == static_loads."""
        static = model.static_loads()
        combined = model.tire_loads(speed_ms=0.0, lateral_g=0.0, longitudinal_g=0.0)
        for s, c in zip(static, combined):
            assert c == pytest.approx(s, abs=1e-10)

    def test_sum_conservation_with_aero(self, model: LoadTransferModel) -> None:
        """Sum of tire loads = weight + total downforce (no clamping)."""
        speed = 80.0 / 3.6
        loads = model.tire_loads(speed_ms=speed, lateral_g=0.0, longitudinal_g=0.0)
        total_load = sum(loads)
        weight = 278.0 * GRAVITY
        aero_f, aero_r = model.aero_loads(speed)
        expected = weight + aero_f + aero_r
        assert total_load == pytest.approx(expected, abs=0.01)

    def test_no_negative_at_1p5g_braking(self, model: LoadTransferModel) -> None:
        """At 1.5g braking (no lateral), all loads remain positive."""
        loads = model.tire_loads(speed_ms=0.0, lateral_g=0.0, longitudinal_g=-1.5)
        fl, fr, rl, rr = loads
        assert fl >= 0.0
        assert fr >= 0.0
        assert rl >= 0.0
        assert rr >= 0.0

    def test_1p5g_braking_values(self, model: LoadTransferModel) -> None:
        """1.5g braking loads: FL=FR=1091.64, RL=RR=271.95."""
        loads = model.tire_loads(speed_ms=0.0, lateral_g=0.0, longitudinal_g=-1.5)
        fl, fr, rl, rr = loads
        assert fl == pytest.approx(1091.64, abs=0.01)
        assert fr == pytest.approx(1091.64, abs=0.01)
        assert rl == pytest.approx(271.95, abs=0.01)
        assert rr == pytest.approx(271.95, abs=0.01)

    def test_right_turn_left_tires_gain(self, model: LoadTransferModel) -> None:
        """Positive lateral_g (right turn) loads left tires more."""
        loads = model.tire_loads(speed_ms=0.0, lateral_g=1.0, longitudinal_g=0.0)
        fl, fr, rl, rr = loads
        assert fl > fr
        assert rl > rr

    def test_left_right_symmetry(self, model: LoadTransferModel) -> None:
        """Swapping lateral sign swaps left/right loads."""
        r = model.tire_loads(speed_ms=0.0, lateral_g=1.0, longitudinal_g=0.0)
        l = model.tire_loads(speed_ms=0.0, lateral_g=-1.0, longitudinal_g=0.0)
        # Right turn FL should equal left turn FR and vice versa
        assert r[0] == pytest.approx(l[1], abs=1e-10)  # FL_right == FR_left
        assert r[1] == pytest.approx(l[0], abs=1e-10)  # FR_right == FL_left
        assert r[2] == pytest.approx(l[3], abs=1e-10)  # RL_right == RR_left
        assert r[3] == pytest.approx(l[2], abs=1e-10)  # RR_right == RL_left

    def test_combined_loading_conservation(self, model: LoadTransferModel) -> None:
        """Under combined braking + cornering, sum still conserved."""
        speed = 60.0 / 3.6
        loads = model.tire_loads(
            speed_ms=speed, lateral_g=0.8, longitudinal_g=-1.0
        )
        total = sum(loads)
        weight = 278.0 * GRAVITY
        aero_f, aero_r = model.aero_loads(speed)
        expected = weight + aero_f + aero_r
        # If no clamping triggered, sum is conserved exactly
        # Check that total is at most equal to expected (clamping only removes)
        assert total <= expected + 0.01
        # And verify loads are still realistic (all positive here)
        for ld in loads:
            assert ld >= 0.0

    def test_combined_sum_exact_no_clamp(self, model: LoadTransferModel) -> None:
        """Moderate loading: no clamping, sum exactly conserved."""
        speed = 40.0 / 3.6
        loads = model.tire_loads(
            speed_ms=speed, lateral_g=0.5, longitudinal_g=-0.5
        )
        total = sum(loads)
        weight = 278.0 * GRAVITY
        aero_f, aero_r = model.aero_loads(speed)
        expected = weight + aero_f + aero_r
        assert total == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Constructor / attribute exposure
# ---------------------------------------------------------------------------


class TestAttributes:
    """Verify exposed attributes needed by downstream solvers."""

    def test_roll_stiffness_front_exposed(self, model: LoadTransferModel) -> None:
        """roll_stiffness_front in Nm/rad accessible."""
        expected = 238.0 * 180.0 / math.pi
        assert model.roll_stiffness_front == pytest.approx(expected, abs=0.01)

    def test_roll_stiffness_rear_exposed(self, model: LoadTransferModel) -> None:
        """roll_stiffness_rear in Nm/rad accessible."""
        expected = 258.0 * 180.0 / math.pi
        assert model.roll_stiffness_rear == pytest.approx(expected, abs=0.01)

    def test_front_track_in_metres(self, model: LoadTransferModel) -> None:
        """front_track is in metres."""
        assert model.front_track == pytest.approx(1.194, abs=1e-6)

    def test_rear_track_in_metres(self, model: LoadTransferModel) -> None:
        """rear_track is in metres."""
        assert model.rear_track == pytest.approx(1.168, abs=1e-6)

    def test_rc_front_in_metres(self, model: LoadTransferModel) -> None:
        """rc_front is in metres."""
        assert model.rc_front == pytest.approx(0.0889, abs=1e-6)

    def test_rc_rear_in_metres(self, model: LoadTransferModel) -> None:
        """rc_rear is in metres."""
        assert model.rc_rear == pytest.approx(0.0635, abs=1e-6)


# ---------------------------------------------------------------------------
# Property: vertical equilibrium under clamping (C5)
# ---------------------------------------------------------------------------


class TestLoadConservationProperty:
    """C5: sum of tire loads must remain equal to m*g + downforce across
    the full (lateral_g, longitudinal_g) operating envelope, even when
    the zero-clamp fires for inside tires.
    """

    @pytest.mark.parametrize(
        "lat_g,long_g,speed_ms",
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 20.0),
            (1.5, 0.0, 10.0),
            (-1.5, 0.0, 10.0),
            (2.0, 0.5, 15.0),
            (-2.0, -1.0, 15.0),
            (2.0, -1.5, 25.0),   # typical corner entry
            (1.2, 0.8, 12.0),    # corner exit
            (0.0, -1.5, 20.0),   # threshold braking
            (0.0, 0.3, 5.0),     # launch
            # Extreme cases that force clamping:
            (2.0, 0.0, 0.0),     # pure lateral, no downforce
            (-2.0, 0.0, 0.0),
            (1.8, -1.8, 30.0),
        ],
    )
    def test_load_sum_conserved_with_clamping(
        self,
        model: LoadTransferModel,
        lat_g: float,
        long_g: float,
        speed_ms: float,
    ) -> None:
        """fl+fr+rl+rr == m*g + downforce within 0.1 N."""
        loads = model.tire_loads(
            speed_ms=speed_ms, lateral_g=lat_g, longitudinal_g=long_g,
        )
        total = sum(loads)
        weight = 278.0 * GRAVITY
        aero_f, aero_r = model.aero_loads(speed_ms)
        expected = weight + aero_f + aero_r
        assert total == pytest.approx(expected, abs=0.1), (
            f"lat={lat_g} long={long_g} speed={speed_ms}: "
            f"got {total:.3f}, expected {expected:.3f}"
        )

    def test_all_loads_nonnegative_everywhere(
        self, model: LoadTransferModel,
    ) -> None:
        """No tire load may be negative after redistribution."""
        for lat_g in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            for long_g in [-1.5, 0.0, 0.5]:
                for speed in [0.0, 15.0, 30.0]:
                    loads = model.tire_loads(
                        speed_ms=speed, lateral_g=lat_g, longitudinal_g=long_g,
                    )
                    for load in loads:
                        assert load >= 0.0

    def test_inside_lift_transfers_to_outside_same_axle(
        self, model: LoadTransferModel,
    ) -> None:
        """When inside tires lift under hard lateral, the outside tires on
        the same axle must absorb the lifted load."""
        static = model.static_loads()
        fl_s, fr_s, rl_s, rr_s = static
        front_axle_static = fl_s + fr_s
        rear_axle_static = rl_s + rr_s

        # Huge lateral at 0 speed (no downforce): inside tires would
        # otherwise go negative.
        loads = model.tire_loads(
            speed_ms=0.0, lateral_g=3.0, longitudinal_g=0.0,
        )
        fl, fr, rl, rr = loads
        # Per-axle totals must be preserved
        assert (fl + fr) == pytest.approx(front_axle_static, abs=0.1)
        assert (rl + rr) == pytest.approx(rear_axle_static, abs=0.1)


# ---------------------------------------------------------------------------
# Property: constructor validation (NF-40)
# ---------------------------------------------------------------------------


def _replace_tracks(
    suspension: SuspensionConfig, front_mm: float, rear_mm: float,
) -> SuspensionConfig:
    return SuspensionConfig(
        roll_stiffness_front_nm_per_deg=suspension.roll_stiffness_front_nm_per_deg,
        roll_stiffness_rear_nm_per_deg=suspension.roll_stiffness_rear_nm_per_deg,
        roll_center_height_front_mm=suspension.roll_center_height_front_mm,
        roll_center_height_rear_mm=suspension.roll_center_height_rear_mm,
        roll_camber_front_deg_per_deg=suspension.roll_camber_front_deg_per_deg,
        roll_camber_rear_deg_per_deg=suspension.roll_camber_rear_deg_per_deg,
        front_track_mm=front_mm,
        rear_track_mm=rear_mm,
    )


class TestTrackWidthValidation:
    """NF-40: reject malformed SuspensionConfig with zero/negative tracks."""

    def test_zero_front_track_raises(
        self, vehicle: VehicleParams, suspension: SuspensionConfig,
    ) -> None:
        bad = _replace_tracks(suspension, 0.0, suspension.rear_track_mm)
        with pytest.raises(ValueError):
            LoadTransferModel(vehicle=vehicle, suspension=bad)

    def test_zero_rear_track_raises(
        self, vehicle: VehicleParams, suspension: SuspensionConfig,
    ) -> None:
        bad = _replace_tracks(suspension, suspension.front_track_mm, 0.0)
        with pytest.raises(ValueError):
            LoadTransferModel(vehicle=vehicle, suspension=bad)

    def test_negative_front_track_raises(
        self, vehicle: VehicleParams, suspension: SuspensionConfig,
    ) -> None:
        bad = _replace_tracks(suspension, -10.0, suspension.rear_track_mm)
        with pytest.raises(ValueError):
            LoadTransferModel(vehicle=vehicle, suspension=bad)
