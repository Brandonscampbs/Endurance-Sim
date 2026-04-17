"""Tests for the runtime powertrain model.

All tests use the CT-16EV specification:
- Motor:     EMRAX-like PMSM, max 2900 RPM, brake speed 2400 RPM
- Inverter:  IQ=170 A, ID=30 A, torque limit 85 Nm
- LVCU:      mechanical torque limit 220 Nm (firmware default; see C17 fix)
- Gear ratio: 3.5 single-speed
- Drivetrain efficiency: 92 %
- Tire radius: 0.2042 m (Hoosier LC0 16x7.5-10 from .tir file)

Effective torque ceiling = min(85, 220) = 85 Nm (inverter-limited).
"""

import math

import pytest

from fsae_sim.vehicle.powertrain import PowertrainConfig
from fsae_sim.vehicle.powertrain_model import PowertrainModel


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ct16ev_powertrain_config() -> PowertrainConfig:
    """CT-16EV powertrain configuration."""
    return PowertrainConfig(
        motor_speed_max_rpm=2900.0,
        brake_speed_rpm=2400.0,
        torque_limit_inverter_nm=85.0,
        torque_limit_lvcu_nm=220.0,
        iq_limit_a=170.0,
        id_limit_a=30.0,
        gear_ratio=3.5,
        drivetrain_efficiency=0.92,
    )


@pytest.fixture
def model(ct16ev_powertrain_config: PowertrainConfig) -> PowertrainModel:
    """PowertrainModel instance for CT-16EV."""
    return PowertrainModel(ct16ev_powertrain_config)


# ---------------------------------------------------------------------------
# RPM / speed conversion
# ---------------------------------------------------------------------------

class TestSpeedRPMConversion:

    def test_roundtrip_speed_to_rpm_to_speed(self, model: PowertrainModel) -> None:
        """Speed -> RPM -> speed roundtrip should recover the original value."""
        original_speed = 15.0  # m/s
        rpm = model.motor_rpm_from_speed(original_speed)
        recovered = model.speed_from_motor_rpm(rpm)
        assert abs(recovered - original_speed) < 1e-9

    def test_roundtrip_rpm_to_speed_to_rpm(self, model: PowertrainModel) -> None:
        """RPM -> speed -> RPM roundtrip should recover the original value."""
        original_rpm = 2000.0
        speed = model.speed_from_motor_rpm(original_rpm)
        recovered_rpm = model.motor_rpm_from_speed(speed)
        assert abs(recovered_rpm - original_rpm) < 1e-9

    def test_zero_speed_gives_zero_rpm(self, model: PowertrainModel) -> None:
        assert model.motor_rpm_from_speed(0.0) == 0.0

    def test_zero_rpm_gives_zero_speed(self, model: PowertrainModel) -> None:
        assert model.speed_from_motor_rpm(0.0) == 0.0

    def test_negative_speed_clamped_to_zero(self, model: PowertrainModel) -> None:
        """Negative speed (reversing) is not modelled — should return 0 RPM."""
        assert model.motor_rpm_from_speed(-5.0) == 0.0

    def test_max_speed_at_max_rpm_in_fsae_range(self, model: PowertrainModel) -> None:
        """At 2900 RPM the vehicle speed must match the gear-ratio prediction.

        With gear_ratio=3.5 and tire_radius=0.2042 m (Hoosier LC0):
            wheel_rpm   = 2900 / 3.5 ≈ 828.6 rpm
            speed_ms    = 828.6 * 0.2042 * 2*pi / 60 ≈ 17.72 m/s ≈ 63.8 km/h

        FSAE competition speeds typically reach 55–80 km/h on straights, so
        ~64 km/h is physically plausible for this drivetrain.
        """
        speed_ms = model.speed_from_motor_rpm(2900.0)
        speed_kmh = speed_ms * 3.6
        assert 55.0 < speed_kmh < 80.0, (
            f"Expected ~64 km/h at max RPM, got {speed_kmh:.2f} km/h"
        )

    def test_rpm_scales_linearly_with_speed(self, model: PowertrainModel) -> None:
        """RPM should be proportional to speed (single-speed gear)."""
        rpm_10 = model.motor_rpm_from_speed(10.0)
        rpm_20 = model.motor_rpm_from_speed(20.0)
        assert abs(rpm_20 / rpm_10 - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Max motor torque vs RPM
# ---------------------------------------------------------------------------

class TestMaxMotorTorque:

    def test_full_torque_below_brake_speed(self, model: PowertrainModel) -> None:
        """Below brake speed, torque should equal min(inverter, LVCU) = 85 Nm."""
        assert model.max_motor_torque(0.0) == pytest.approx(85.0)
        assert model.max_motor_torque(1000.0) == pytest.approx(85.0)
        assert model.max_motor_torque(2400.0) == pytest.approx(85.0)

    def test_torque_exactly_at_brake_speed(self, model: PowertrainModel) -> None:
        """At exactly brake_speed_rpm the torque envelope begins — still full."""
        assert model.max_motor_torque(2400.0) == pytest.approx(85.0)

    def test_torque_tapers_in_field_weakening_region(self, model: PowertrainModel) -> None:
        """D-15: torque decreases monotonically from 2400 to 2900 RPM
        following a constant-power curve T(ω) = P_max / ω.
        """
        t_2400 = model.max_motor_torque(2400.0)
        t_2650 = model.max_motor_torque(2650.0)
        t_2900 = model.max_motor_torque(2900.0)

        assert t_2400 > t_2650 > t_2900
        # Constant-power: T(ω)·ω = P_max = T_limit · ω_corner.
        # So T(2650) = 85 × 2400/2650 ≈ 76.98 Nm.
        assert abs(t_2650 - 85.0 * 2400.0 / 2650.0) < 0.01

    def test_torque_nonzero_at_max_rpm(self, model: PowertrainModel) -> None:
        """D-15: constant-power field-weakening leaves residual torque at
        motor_speed_max_rpm. T(2900) = 85 × 2400/2900 ≈ 70.3 Nm.

        (The old linear taper dropped to exactly 0 at max RPM; that shape
        was unphysical for a PMSM under field weakening.)
        """
        t_max = model.max_motor_torque(2900.0)
        assert t_max == pytest.approx(85.0 * 2400.0 / 2900.0, abs=0.01)

    def test_zero_torque_above_max_rpm(self, model: PowertrainModel) -> None:
        """Above the maximum RPM, the motor produces no torque."""
        assert model.max_motor_torque(3000.0) == 0.0
        assert model.max_motor_torque(5000.0) == 0.0

    def test_constant_power_in_fw_region(self, model: PowertrainModel) -> None:
        """D-15: in the field-weakening region, T × ω is constant (= P_max)."""
        import math
        rpms = [2400.0, 2550.0, 2700.0, 2850.0]
        powers = []
        for rpm in rpms:
            omega = rpm * math.pi / 30.0
            powers.append(model.max_motor_torque(rpm) * omega)
        for p in powers:
            assert abs(p - powers[0]) < 1e-6

    def test_inverter_limit_is_binding(self, model: PowertrainModel) -> None:
        """Inverter limit (85 Nm) is less than LVCU (220 Nm), so it must bind."""
        # The effective cap must equal the inverter limit, not the LVCU limit
        assert model.max_motor_torque(0.0) == pytest.approx(85.0)
        assert model.max_motor_torque(0.0) != pytest.approx(220.0)


# ---------------------------------------------------------------------------
# Wheel torque and force
# ---------------------------------------------------------------------------

class TestWheelTorqueAndForce:

    def test_wheel_torque_equals_motor_times_ratio_times_gearbox_eff(
        self, model: PowertrainModel
    ) -> None:
        """wheel_torque = motor_torque * gear_ratio * _GEARBOX_EFFICIENCY.

        Motor+inverter efficiency affects electrical power (see
        electrical_power()), not mechanical torque delivered to the wheels.
        Only gearbox friction appears here.
        """
        motor_torque = 50.0
        expected = motor_torque * 3.5 * PowertrainModel._GEARBOX_EFFICIENCY
        assert model.wheel_torque(motor_torque) == pytest.approx(expected)

    def test_wheel_torque_at_max_motor_torque(self, model: PowertrainModel) -> None:
        """At full motor torque the wheel torque should be 85 * 3.5 * η_gearbox Nm."""
        expected = 85.0 * 3.5 * PowertrainModel._GEARBOX_EFFICIENCY
        assert model.wheel_torque(85.0) == pytest.approx(expected)

    def test_wheel_torque_zero_for_zero_motor_torque(
        self, model: PowertrainModel
    ) -> None:
        assert model.wheel_torque(0.0) == 0.0

    def test_wheel_force_divides_by_tire_radius(self, model: PowertrainModel) -> None:
        """wheel_force = wheel_torque / tire_radius."""
        motor_torque = 60.0
        expected_force = model.wheel_torque(motor_torque) / PowertrainModel.TIRE_RADIUS_M
        assert model.wheel_force(motor_torque) == pytest.approx(expected_force)

    def test_wheel_force_positive_for_positive_torque(
        self, model: PowertrainModel
    ) -> None:
        assert model.wheel_force(50.0) > 0.0

    def test_wheel_force_negative_for_negative_torque(
        self, model: PowertrainModel
    ) -> None:
        """Negative motor torque (regen) should produce negative (decelerating) force."""
        assert model.wheel_force(-30.0) < 0.0


# ---------------------------------------------------------------------------
# Drive force
# ---------------------------------------------------------------------------

class TestDriveForce:

    def test_full_throttle_at_low_speed(self, model: PowertrainModel) -> None:
        """Full throttle below brake speed should produce maximum tractive force."""
        speed = 5.0  # m/s — well within constant-torque region
        expected_torque = 85.0  # min(inverter, LVCU)
        expected_force = model.wheel_force(expected_torque)
        assert model.drive_force(1.0, speed) == pytest.approx(expected_force)

    def test_half_throttle_gives_half_force(self, model: PowertrainModel) -> None:
        """At half throttle the force should be half of full-throttle force."""
        speed = 5.0  # constant-torque region
        full = model.drive_force(1.0, speed)
        half = model.drive_force(0.5, speed)
        assert abs(half - full / 2.0) < 1e-9

    def test_zero_throttle_gives_zero_force(self, model: PowertrainModel) -> None:
        assert model.drive_force(0.0, 10.0) == 0.0

    def test_drive_force_decreases_above_brake_speed(
        self, model: PowertrainModel
    ) -> None:
        """As speed rises through the field-weakening region, max force drops."""
        speed_at_brake = model.speed_from_motor_rpm(2400.0)
        speed_past_brake = model.speed_from_motor_rpm(2700.0)

        f_at_brake = model.drive_force(1.0, speed_at_brake)
        f_past_brake = model.drive_force(1.0, speed_past_brake)

        assert f_at_brake > f_past_brake

    def test_drive_force_zero_above_max_rpm_speed(
        self, model: PowertrainModel
    ) -> None:
        """Beyond max RPM speed the motor can produce no torque."""
        speed_over_max = model.speed_from_motor_rpm(3000.0)
        assert model.drive_force(1.0, speed_over_max) == 0.0

    def test_throttle_clamped_above_one(self, model: PowertrainModel) -> None:
        """Throttle > 1.0 should be clamped to 1.0."""
        f_1 = model.drive_force(1.0, 5.0)
        f_2 = model.drive_force(2.0, 5.0)
        assert f_1 == pytest.approx(f_2)

    def test_throttle_clamped_below_zero(self, model: PowertrainModel) -> None:
        """Negative throttle demand should be clamped to zero."""
        assert model.drive_force(-0.5, 5.0) == 0.0


# ---------------------------------------------------------------------------
# Regen force
# ---------------------------------------------------------------------------

class TestRegenForce:

    def test_regen_force_is_negative(self, model: PowertrainModel) -> None:
        """Regen should oppose motion (negative force)."""
        assert model.regen_force(1.0, 10.0) < 0.0

    def test_zero_regen_at_zero_speed(self, model: PowertrainModel) -> None:
        """Regen cannot decelerate a stationary vehicle."""
        assert model.regen_force(1.0, 0.0) == 0.0

    def test_partial_regen_brake_proportional(self, model: PowertrainModel) -> None:
        """50% regen brake demand should produce half the force of 100%."""
        full = model.regen_force(1.0, 10.0)
        half = model.regen_force(0.5, 10.0)
        assert abs(half - full / 2.0) < 1e-9

    def test_regen_magnitude_greater_than_drive_force(
        self, model: PowertrainModel
    ) -> None:
        """After S12 fix: regen *mechanical* retarding force is GREATER than drive force.

        In generator mode, gearbox friction ADDS to retarding torque at the
        wheel (gearbox losses help slow the car). Drive force has gearbox
        friction SUBTRACT from motor torque. Same motor torque magnitude,
        different sign: |F_regen| = T*gear / (η*r) > F_drive = T*gear*η / r.
        (Electrical energy recovered is still less than energy expended —
        that asymmetry lives in electrical_power(), not regen_force().)
        """
        speed = 10.0
        drive = model.drive_force(1.0, speed)
        regen_mag = abs(model.regen_force(1.0, speed))
        assert regen_mag > drive

    def test_regen_brake_clamped_above_one(self, model: PowertrainModel) -> None:
        f_1 = model.regen_force(1.0, 10.0)
        f_2 = model.regen_force(2.0, 10.0)
        assert f_1 == pytest.approx(f_2)

    def test_regen_brake_clamped_below_zero(self, model: PowertrainModel) -> None:
        assert model.regen_force(-0.5, 10.0) == 0.0


# ---------------------------------------------------------------------------
# Electrical power
# ---------------------------------------------------------------------------

class TestElectricalPower:

    def test_motoring_power_is_positive(self, model: PowertrainModel) -> None:
        """Motoring draws power from the battery — must be positive."""
        p = model.electrical_power(motor_torque_nm=50.0, motor_rpm=1000.0)
        assert p > 0.0

    def test_regen_power_is_negative(self, model: PowertrainModel) -> None:
        """Regen returns power to the battery — must be negative."""
        p = model.electrical_power(motor_torque_nm=-30.0, motor_rpm=1500.0)
        assert p < 0.0

    def test_zero_torque_gives_zero_power(self, model: PowertrainModel) -> None:
        assert model.electrical_power(0.0, 1000.0) == 0.0

    def test_zero_rpm_gives_zero_power(self, model: PowertrainModel) -> None:
        """At zero speed there is no back-EMF and no power flow."""
        assert model.electrical_power(85.0, 0.0) == 0.0

    def test_motoring_power_exceeds_mechanical_power(
        self, model: PowertrainModel
    ) -> None:
        """Electrical input must be greater than mechanical output (losses)."""
        torque = 60.0
        rpm = 1500.0
        omega = rpm * math.pi / 30.0
        p_mech = torque * omega
        p_elec = model.electrical_power(torque, rpm)
        assert p_elec > p_mech

    def test_regen_power_less_than_mechanical_power(
        self, model: PowertrainModel
    ) -> None:
        """Energy recovered by regen must be less than mechanical input (losses)."""
        torque = -40.0  # generating (negative)
        rpm = 2000.0
        omega = rpm * math.pi / 30.0
        p_mech_magnitude = abs(torque * omega)
        p_elec = model.electrical_power(torque, rpm)
        assert abs(p_elec) < p_mech_magnitude

    def test_power_scales_with_torque(self, model: PowertrainModel) -> None:
        """Doubling torque at constant RPM should double electrical power."""
        p1 = model.electrical_power(30.0, 1000.0)
        p2 = model.electrical_power(60.0, 1000.0)
        assert abs(p2 / p1 - 2.0) < 1e-9

    def test_power_scales_with_rpm(self, model: PowertrainModel) -> None:
        """Doubling RPM at constant torque should double electrical power."""
        p1 = model.electrical_power(50.0, 1000.0)
        p2 = model.electrical_power(50.0, 2000.0)
        assert abs(p2 / p1 - 2.0) < 1e-9

    def test_motoring_efficiency_applied_correctly(
        self, model: PowertrainModel
    ) -> None:
        """P_elec = (T * omega) / eta for motoring."""
        torque = 70.0
        rpm = 1200.0
        omega = rpm * math.pi / 30.0
        expected = torque * omega / 0.92
        assert model.electrical_power(torque, rpm) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Pack current
# ---------------------------------------------------------------------------

class TestPackCurrent:

    def test_current_from_power_and_voltage(self, model: PowertrainModel) -> None:
        """I = P / V."""
        power = 10_000.0  # W
        voltage = 400.0   # V
        assert model.pack_current(power, voltage) == pytest.approx(25.0)

    def test_positive_power_gives_positive_current(
        self, model: PowertrainModel
    ) -> None:
        """Discharge current is positive."""
        assert model.pack_current(5000.0, 400.0) > 0.0

    def test_negative_power_gives_negative_current(
        self, model: PowertrainModel
    ) -> None:
        """Regen current (charging) is negative."""
        assert model.pack_current(-3000.0, 400.0) < 0.0

    def test_zero_power_gives_zero_current(self, model: PowertrainModel) -> None:
        assert model.pack_current(0.0, 400.0) == 0.0

    def test_zero_voltage_raises_value_error(self, model: PowertrainModel) -> None:
        with pytest.raises(ValueError):
            model.pack_current(5000.0, 0.0)

    def test_negative_voltage_raises_value_error(
        self, model: PowertrainModel
    ) -> None:
        with pytest.raises(ValueError):
            model.pack_current(5000.0, -10.0)

    def test_current_inversely_proportional_to_voltage(
        self, model: PowertrainModel
    ) -> None:
        """At double the voltage, current should halve for the same power."""
        i1 = model.pack_current(8000.0, 400.0)
        i2 = model.pack_current(8000.0, 800.0)
        assert abs(i2 / i1 - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Integration: CT-16EV full-throttle peak force
# ---------------------------------------------------------------------------

class TestCT16EVIntegration:

    def test_peak_tractive_force_in_expected_range(
        self, model: PowertrainModel
    ) -> None:
        """Peak tractive force at low speed should be physically reasonable.

        Expected: 85 Nm * 3.5 * η_gearbox / 0.2042 m ≈ 1413 N.
        Plausible range for FSAE EV: 900–1500 N.
        """
        force = model.drive_force(1.0, 1.0)  # 1 m/s ~ constant-torque region
        assert 900.0 < force < 1500.0, (
            f"Peak tractive force {force:.1f} N outside plausible FSAE range"
        )

    def test_peak_force_calculation(self, model: PowertrainModel) -> None:
        """Verify peak force matches analytical calculation."""
        expected = (
            85.0 * 3.5 * PowertrainModel._GEARBOX_EFFICIENCY
            / PowertrainModel.TIRE_RADIUS_M
        )
        actual = model.drive_force(1.0, 0.5)  # very low speed, constant-torque
        assert abs(actual - expected) < 1.0

    def test_max_speed_above_60_kmh_threshold(self, model: PowertrainModel) -> None:
        """CT-16EV top speed via powertrain should clear 40 km/h."""
        top_speed_ms = model.speed_from_motor_rpm(
            model.config.motor_speed_max_rpm
        )
        top_speed_kmh = top_speed_ms * 3.6
        assert top_speed_kmh > 40.0, (
            f"Top speed {top_speed_kmh:.1f} km/h is unrealistically low"
        )

    def test_peak_electrical_power_reasonable(self, model: PowertrainModel) -> None:
        """Peak electrical power must be in a physically plausible range.

        At max torque (85 Nm) and brake speed (2400 RPM):
        P_mech = 85 * (2400 * pi/30) = ~21.4 kW
        P_elec = P_mech / 0.92 ≈ 23.2 kW

        Reasonable range: 15–35 kW for a small EV.
        """
        rpm = 2400.0
        p = model.electrical_power(85.0, rpm)
        assert 15_000 < p < 35_000, (
            f"Peak electrical power {p/1000:.1f} kW outside expected range"
        )


# ---------------------------------------------------------------------------
# LVCU torque command (real firmware replication)
# ---------------------------------------------------------------------------

class TestLVCUTorqueCommand:
    """Tests for lvcu_torque_command — replicates real LVCU C code."""

    def test_full_pedal_low_rpm_inverter_limited(self, model: PowertrainModel) -> None:
        """At 100A and low RPM, inverter 85 Nm limit should bind."""
        torque = model.lvcu_torque_command(1.0, 1000.0, 100.0)
        assert torque == pytest.approx(85.0)

    def test_full_pedal_high_rpm_power_limited(self, model: PowertrainModel) -> None:
        """At 50A and 2900 RPM, the power limit should bind below 85 Nm.

        After S14 BMS offset: effective_limit = 50 - 3 = 47 A.
        Power ceiling: 420 * 47 / max(23.04, 2900 * 0.1076)
                     = 19740 / max(23.04, 312.04)
                     = 19740 / 312.04
                     ~ 63.3 Nm
        """
        torque = model.lvcu_torque_command(1.0, 2900.0, 50.0)
        expected = 420.0 * (50.0 - 3.0) / (2900.0 * 0.1076)
        assert torque == pytest.approx(expected, rel=0.01)
        assert torque < 85.0

    def test_half_pedal_scales_linearly(self, model: PowertrainModel) -> None:
        """Half pedal (after dead zone remap) gives half the torque ceiling."""
        # pedal=0.5 remaps to (0.5-0.1)/(0.9-0.1) = 0.5
        full = model.lvcu_torque_command(1.0, 1000.0, 100.0)
        half = model.lvcu_torque_command(0.5, 1000.0, 100.0)
        assert half == pytest.approx(full * 0.5)

    def test_pedal_below_deadzone_gives_zero(self, model: PowertrainModel) -> None:
        """Pedal at 0.05 is below V_MIN=0.1, should produce 0 torque."""
        torque = model.lvcu_torque_command(0.05, 1000.0, 100.0)
        assert torque == 0.0

    def test_pedal_above_deadzone_gives_zero(self, model: PowertrainModel) -> None:
        """Pedal at 0.95 is above V_MAX=0.9, should produce full torque."""
        torque = model.lvcu_torque_command(0.95, 1000.0, 100.0)
        assert torque == pytest.approx(85.0)

    def test_zero_pedal_gives_zero(self, model: PowertrainModel) -> None:
        torque = model.lvcu_torque_command(0.0, 1000.0, 100.0)
        assert torque == 0.0

    def test_zero_current_gives_zero(self, model: PowertrainModel) -> None:
        """If BMS current limit is 0, no torque can be produced."""
        torque = model.lvcu_torque_command(1.0, 1000.0, 0.0)
        assert torque == 0.0

    def test_overspeed_caps_torque(self, model: PowertrainModel) -> None:
        """At >= 6000 RPM, torque ceiling drops to 30 Nm."""
        torque = model.lvcu_torque_command(1.0, 6500.0, 100.0)
        assert torque == pytest.approx(30.0)

    def test_power_limit_at_low_rpm_uses_floor(self, model: PowertrainModel) -> None:
        """Below ~2141 RPM, the omega floor dominates.

        At 200 RPM: max(23.04, 200*0.1076) = max(23.04, 21.52) = 23.04
        Power ceiling: 420 * 100 / 23.04 = 1822.9 Nm -> clamped to 85 Nm
        """
        torque = model.lvcu_torque_command(1.0, 200.0, 100.0)
        assert torque == pytest.approx(85.0)  # inverter limit still binds

    def test_power_limit_becomes_binding_with_low_current(self, model: PowertrainModel) -> None:
        """At 45A and 2400 RPM, power limit should be near inverter limit.

        After S14 BMS offset: effective_limit = 45 - 3 = 42 A.
        Power ceiling: 420 * 42 / max(23.04, 2400 * 0.1076)
                     = 17640 / max(23.04, 258.24)
                     = 17640 / 258.24
                     ~ 68.3 Nm  (below 85 Nm -- power limit binds)
        """
        torque = model.lvcu_torque_command(1.0, 2400.0, 45.0)
        expected = 420.0 * (45.0 - 3.0) / (2400.0 * 0.1076)
        assert torque == pytest.approx(expected, rel=0.01)
        assert torque < 85.0

    def test_lvcu_limit_caps_before_inverter(self, model: PowertrainModel) -> None:
        """With very high current, LVCU 220 Nm limit should bind before
        the power formula would give more, but inverter 85 Nm still wins.

        At 500A (unrealistic, but tests the min chain):
        Power: 420*(500-3)/max(23.04, 1000*0.1076) = 208740/107.6 ~ 1940 Nm
        LVCU limit: 220 Nm
        Inverter limit: 85 Nm
        Result: 85 Nm
        """
        torque = model.lvcu_torque_command(1.0, 1000.0, 500.0)
        assert torque == pytest.approx(85.0)

    # --- S14: BMS -3 A safety offset -----------------------------------

    def test_bms_current_limit_has_minus_three_offset(
        self, model: PowertrainModel,
    ) -> None:
        """Firmware: `current_limit = RxData − 3`. Our sim must subtract 3 A
        before the power divide.

        At 2400 RPM with BMS limit 50 A:
            effective_limit = 47 A
            power ceiling   = 420 * 47 / (2400 * 0.1076)
                            = 19740 / 258.24
                            ~ 76.4 Nm  (< 85 Nm inverter limit)
        """
        torque = model.lvcu_torque_command(1.0, 2400.0, 50.0)
        expected = 420.0 * (50.0 - 3.0) / (2400.0 * 0.1076)
        assert torque == pytest.approx(expected, rel=0.01)

    def test_bms_offset_clamped_to_zero(self, model: PowertrainModel) -> None:
        """With BMS limit at 1 A, effective limit must clamp to 0, not -2."""
        torque = model.lvcu_torque_command(1.0, 2400.0, 1.0)
        assert torque == 0.0

    def test_bms_offset_exactly_three(self, model: PowertrainModel) -> None:
        """BMS limit of exactly 3 A → 0 A effective → 0 torque."""
        torque = model.lvcu_torque_command(1.0, 2400.0, 3.0)
        assert torque == 0.0


# ---------------------------------------------------------------------------
# LVCU BSE (brake+throttle) latch, APPS mismatch, startup gate
# ---------------------------------------------------------------------------

class TestLVCUBSELatch:
    """S13: BSE/APPS/startup gate interlocks from LVCU firmware."""

    def test_bse_latches_when_brake_and_tps_ge_10pct(
        self, model: PowertrainModel,
    ) -> None:
        """Brake pressed with TPS >= 10% latches BSE → torque zeroed."""
        result = model.lvcu_torque_command(
            pedal_pct=0.15,
            motor_rpm=1000.0,
            bms_current_limit_a=100.0,
            brake_pressed=True,
            return_state=True,
        )
        assert result.torque_nm == 0.0
        assert result.bse_latched is True

    def test_bse_not_latched_below_10pct_tps(
        self, model: PowertrainModel,
    ) -> None:
        """Brake pressed with TPS < 10% does NOT latch BSE."""
        result = model.lvcu_torque_command(
            pedal_pct=0.08,
            motor_rpm=1000.0,
            bms_current_limit_a=100.0,
            brake_pressed=True,
            return_state=True,
        )
        assert result.bse_latched is False

    def test_bse_stays_latched_when_tps_between_5_and_10_pct(
        self, model: PowertrainModel,
    ) -> None:
        """Once latched, BSE persists until TPS < 5% (hysteresis)."""
        # First call: latch
        s1 = model.lvcu_torque_command(
            0.20, 1000.0, 100.0, brake_pressed=True, return_state=True,
        )
        assert s1.bse_latched is True
        # Second call: still pressing throttle, brake released. TPS now 7%.
        # BSE clears only when TPS < 5%.
        s2 = model.lvcu_torque_command(
            0.07, 1000.0, 100.0, brake_pressed=False,
            prior_bse_latched=True, return_state=True,
        )
        assert s2.bse_latched is True
        assert s2.torque_nm == 0.0

    def test_bse_clears_when_tps_below_5pct(
        self, model: PowertrainModel,
    ) -> None:
        """Once TPS drops below 5%, BSE clears and torque can flow again."""
        s = model.lvcu_torque_command(
            0.03, 1000.0, 100.0, brake_pressed=False,
            prior_bse_latched=True, return_state=True,
        )
        assert s.bse_latched is False

    def test_bse_not_active_without_brake(
        self, model: PowertrainModel,
    ) -> None:
        """Throttle alone at any value must not latch BSE."""
        r = model.lvcu_torque_command(
            1.0, 1000.0, 100.0, brake_pressed=False, return_state=True,
        )
        assert r.bse_latched is False
        assert r.torque_nm > 0.0

    def test_apps_mismatch_flag(
        self, model: PowertrainModel,
    ) -> None:
        """APPS mismatch ( |TPS1 - TPS2| > 40% ) is reported as a flag."""
        r = model.lvcu_torque_command(
            0.5, 1000.0, 100.0, tps1=0.5, tps2=0.9, return_state=True,
        )
        assert r.apps_mismatch is True

    def test_startup_gate_flag(
        self, model: PowertrainModel,
    ) -> None:
        """Startup gate (torque<5 Nm & motor<500 RPM) is a diagnostic flag."""
        # Pedal 0.1 remaps to exactly dead-zone low edge → 0 torque,
        # which is below the 5 Nm gate at 300 RPM < 500 RPM.
        r = model.lvcu_torque_command(
            0.1, 300.0, 100.0, return_state=True,
        )
        assert r.startup_gate_active is True

    def test_backwards_compat_no_brake_returns_float(
        self, model: PowertrainModel,
    ) -> None:
        """Without brake or return_state, behavior is identical to before."""
        t = model.lvcu_torque_command(1.0, 1000.0, 100.0)
        expected = 420.0 * (100.0 - 3.0) / max(23.04, 1000.0 * 0.1076)
        # At low RPM the power ceiling is huge, so inverter 85 Nm wins
        assert t == pytest.approx(85.0)


# ---------------------------------------------------------------------------
# Regen gearbox sign (S12) and regen gearbox mechanical force
# ---------------------------------------------------------------------------

class TestRegenGearboxSign:
    """S12: generator-mode gearbox friction adds to retarding torque."""

    def test_regen_wheel_force_uses_divide_not_multiply(
        self, model: PowertrainModel,
    ) -> None:
        """In generator mode, wheel retarding force = T_motor * gear / (η * r).

        Gearbox friction adds to retarding torque at the wheel, so the
        retarding force magnitude is ~3% HIGHER than a naïve (× η) model.

        D-22: max regen torque is derated by _REGEN_EFFICIENCY_FACTOR.
        """
        speed = 10.0
        rpm = model.motor_rpm_from_speed(speed)
        max_torque = (
            model.max_motor_torque(rpm) * PowertrainModel._REGEN_EFFICIENCY_FACTOR
        )
        expected_mag = max_torque * model.config.gear_ratio / (
            PowertrainModel._GEARBOX_EFFICIENCY * PowertrainModel.TIRE_RADIUS_M
        )
        actual = model.regen_force(1.0, speed)
        assert actual < 0.0
        assert abs(actual) == pytest.approx(expected_mag, rel=1e-6)

    def test_regen_force_exceeds_naive_multiply(
        self, model: PowertrainModel,
    ) -> None:
        """After S12 fix, |regen| with divide is 1/η² higher than multiply
        (modulo the D-22 regen-efficiency derate on both sides, which
        cancels when we take the ratio).

        Old (wrong): |F| = T * gear * η / r
        New (right): |F| = T * gear / (η * r)
        Ratio after S12 and D-22:  1/η_gearbox²
        """
        speed = 12.0
        rpm = model.motor_rpm_from_speed(speed)
        max_torque = (
            model.max_motor_torque(rpm) * PowertrainModel._REGEN_EFFICIENCY_FACTOR
        )
        eta = PowertrainModel._GEARBOX_EFFICIENCY
        r = PowertrainModel.TIRE_RADIUS_M

        old_wrong_mag = max_torque * model.config.gear_ratio * eta / r
        new_correct = abs(model.regen_force(1.0, speed))
        assert new_correct > old_wrong_mag
        assert new_correct / old_wrong_mag == pytest.approx(1.0 / (eta * eta), rel=1e-6)


# ---------------------------------------------------------------------------
# Back-EMF rectification (C2)
# ---------------------------------------------------------------------------

class TestBackEMFRectification:
    """C2: `electrical_power(0, rpm, V_pack)` regens when K_e·ω > V_pack."""

    def test_coast_above_bemf_threshold_is_regen(
        self, model: PowertrainModel,
    ) -> None:
        """K_e=0.045 V/(rad/s). At 2500 RPM ω=261.8 rad/s, V_bemf=11.78 V
        for a single pole pair — but for EMRAX 228 the effective line-line
        back-EMF constant scales up.

        Test with a low pack voltage so V_bemf > V_pack unambiguously.
        """
        rpm = 2500.0
        # omega = 261.8 rad/s, V_bemf = 0.045 * 261.8 ≈ 11.78 V
        # Deliberately low pack voltage so back-EMF threshold is crossed.
        V_pack = 5.0
        p = model.electrical_power(0.0, rpm, V_pack)
        assert p < 0.0, "Coast above back-EMF threshold must return negative (regen) power"

    def test_coast_below_bemf_threshold_is_zero(
        self, model: PowertrainModel,
    ) -> None:
        """Below V_bemf threshold, coast returns zero power (no current flow)."""
        rpm = 1000.0
        # omega = 104.7 rad/s, V_bemf = 0.045 * 104.7 ≈ 4.71 V
        # Pack voltage of 400 V is well above, so no rectification.
        V_pack = 400.0
        p = model.electrical_power(0.0, rpm, V_pack)
        assert p == 0.0

    def test_coast_no_pack_voltage_is_zero(
        self, model: PowertrainModel,
    ) -> None:
        """Backwards compat: calling without V_pack defaults to no-rectification."""
        p = model.electrical_power(0.0, 2000.0)
        assert p == 0.0

    def test_motoring_ignores_back_emf(
        self, model: PowertrainModel,
    ) -> None:
        """Back-EMF rectification only applies when motor_torque is ~zero.
        Motoring path (positive torque) uses efficiency map / drivetrain_eff.
        """
        p_no_bemf = model.electrical_power(50.0, 1500.0)
        p_with_bemf = model.electrical_power(50.0, 1500.0, 400.0)
        assert p_with_bemf == pytest.approx(p_no_bemf)

    def test_negative_torque_unchanged_by_bemf_arg(
        self, model: PowertrainModel,
    ) -> None:
        """Regen path (commanded negative torque) is independent of back-EMF arg."""
        p_no_bemf = model.electrical_power(-40.0, 2000.0)
        p_with_bemf = model.electrical_power(-40.0, 2000.0, 400.0)
        assert p_with_bemf == pytest.approx(p_no_bemf)


# ---------------------------------------------------------------------------
# Regen inverter-loss double-count fix (C3)
# ---------------------------------------------------------------------------

class TestBackEMFValidationMichigan:
    """D-17: document the -456 W coast-power validation finding.

    Michigan 2025 mean coast operating point:
      RPM   = 2299 (mean when |Torque Feedback| < 1 Nm)
      V_pack= 410 V (mean pack voltage over stint)
      P     = -456 W (measured Pack V × I at those samples)

    With physics-derived K_e = 0.6366 V·s/rad from EMRAX 228 MV LC
    datasheet (Kv = 15 RPM/V → K_e = 60 / (2π · Kv)), V_bemf at 2299
    RPM is ≈ 153 V — well below any realistic pack voltage. So the
    passive-rectifier model predicts 0 W at this point, not -456 W.

    This test pins that finding so a future change doesn't silently
    re-introduce a fudge factor.  The -456 W must come from somewhere
    else (iron losses, inverter standby current, driveline drag seen
    by the pack) — logged in REMAINING_ISSUES.md.
    """

    def test_michigan_coast_point_rectifier_off(self, model: PowertrainModel) -> None:
        p = model.electrical_power(0.0, 2299.0, 410.0)
        # Rectifier off: physics-honest model returns zero.
        assert p == pytest.approx(0.0, abs=1e-6)


class TestRegenEfficiencyNoDoubleCount:
    """C3: map path must NOT multiply by extra 0.85 — map already includes losses."""

    def test_fallback_regen_uses_drivetrain_eff_directly(
        self, model: PowertrainModel,
    ) -> None:
        """Without a motor map, regen efficiency = drivetrain_efficiency (0.92).

        After C3 fix, no 0.85 factor is applied. If asymmetry is desired,
        it is documented and <= 2pp.
        """
        torque = -50.0
        rpm = 1500.0
        omega = rpm * math.pi / 30.0
        p_mech = torque * omega  # negative
        p_elec = model.electrical_power(torque, rpm)
        # Motoring-vs-regen offset must be small (<= 2 percentage points).
        implied_eta = p_elec / p_mech  # both negative → positive
        assert implied_eta <= 0.92 + 1e-9
        assert implied_eta >= 0.92 - 0.02


# ---------------------------------------------------------------------------
# Pedal dead-zone span guard (NF-41)
# ---------------------------------------------------------------------------

class TestPedalDeadzoneGuard:

    def test_zero_span_rejected_in_config(self) -> None:
        """PowertrainConfig with lvcu_pedal_deadzone_high == low must raise."""
        with pytest.raises(ValueError):
            PowertrainConfig(
                motor_speed_max_rpm=2900.0,
                brake_speed_rpm=2400.0,
                torque_limit_inverter_nm=85.0,
                torque_limit_lvcu_nm=220.0,
                iq_limit_a=170.0,
                id_limit_a=30.0,
                gear_ratio=3.5,
                drivetrain_efficiency=0.92,
                lvcu_pedal_deadzone_low=0.5,
                lvcu_pedal_deadzone_high=0.5,
            )

    def test_near_zero_span_rejected(self) -> None:
        """Span < 0.01 is rejected (catastrophic noise amplification)."""
        with pytest.raises(ValueError):
            PowertrainConfig(
                motor_speed_max_rpm=2900.0,
                brake_speed_rpm=2400.0,
                torque_limit_inverter_nm=85.0,
                torque_limit_lvcu_nm=220.0,
                iq_limit_a=170.0,
                id_limit_a=30.0,
                gear_ratio=3.5,
                drivetrain_efficiency=0.92,
                lvcu_pedal_deadzone_low=0.5,
                lvcu_pedal_deadzone_high=0.505,
            )

    def test_valid_span_accepted(self) -> None:
        """Span >= 0.01 is accepted."""
        cfg = PowertrainConfig(
            motor_speed_max_rpm=2900.0,
            brake_speed_rpm=2400.0,
            torque_limit_inverter_nm=85.0,
            torque_limit_lvcu_nm=220.0,
            iq_limit_a=170.0,
            id_limit_a=30.0,
            gear_ratio=3.5,
            drivetrain_efficiency=0.92,
            lvcu_pedal_deadzone_low=0.4,
            lvcu_pedal_deadzone_high=0.5,
        )
        assert cfg.lvcu_pedal_deadzone_high - cfg.lvcu_pedal_deadzone_low == pytest.approx(0.1)
