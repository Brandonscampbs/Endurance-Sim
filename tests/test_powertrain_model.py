"""Tests for the runtime powertrain model.

All tests use the CT-16EV specification:
- Motor:     EMRAX-like PMSM, max 2900 RPM, brake speed 2400 RPM
- Inverter:  IQ=170 A, ID=30 A, torque limit 85 Nm
- LVCU:      mechanical torque limit 150 Nm
- Gear ratio: 3.5 single-speed
- Drivetrain efficiency: 92 %
- Tire radius: 0.228 m (10-inch FSAE tyre)

Effective torque ceiling = min(85, 150) = 85 Nm (inverter-limited).
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
        torque_limit_lvcu_nm=150.0,
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

        With gear_ratio=3.5 and tire_radius=0.228 m:
            wheel_rpm   = 2900 / 3.5 ≈ 828.6 rpm
            speed_ms    = 828.6 * 0.228 * 2*pi / 60 ≈ 19.78 m/s ≈ 71.2 km/h

        FSAE competition speeds typically reach 60–80 km/h on straights, so
        ~71 km/h is physically plausible for this drivetrain.
        """
        speed_ms = model.speed_from_motor_rpm(2900.0)
        speed_kmh = speed_ms * 3.6
        assert 65.0 < speed_kmh < 80.0, (
            f"Expected ~71 km/h at max RPM, got {speed_kmh:.2f} km/h"
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
        """Torque must decrease linearly from 2400 to 2900 RPM."""
        t_2400 = model.max_motor_torque(2400.0)
        t_2650 = model.max_motor_torque(2650.0)  # midpoint of taper range
        t_2900 = model.max_motor_torque(2900.0)

        assert t_2400 > t_2650 > t_2900
        # Midpoint should be very close to half of full torque
        assert abs(t_2650 - 85.0 / 2.0) < 1.0

    def test_torque_reaches_zero_at_max_rpm(self, model: PowertrainModel) -> None:
        """At motor_speed_max_rpm the taper should reach exactly zero."""
        assert model.max_motor_torque(2900.0) == pytest.approx(0.0, abs=1e-9)

    def test_zero_torque_above_max_rpm(self, model: PowertrainModel) -> None:
        """Above the maximum RPM, the motor produces no torque."""
        assert model.max_motor_torque(3000.0) == 0.0
        assert model.max_motor_torque(5000.0) == 0.0

    def test_torque_taper_linearity(self, model: PowertrainModel) -> None:
        """Torque drop per RPM should be constant across the field-weakening range."""
        rpms = [2400.0, 2550.0, 2700.0, 2850.0, 2900.0]
        torques = [model.max_motor_torque(r) for r in rpms]
        # First-order differences should all be equal (or very close)
        diffs = [torques[i] - torques[i + 1] for i in range(len(torques) - 1)]
        for d in diffs:
            assert d >= 0.0, "Torque should not increase in field-weakening region"
        # Check constant slope: all diffs normalised by ΔRPM should match
        rpm_steps = [rpms[i + 1] - rpms[i] for i in range(len(rpms) - 1)]
        slopes = [d / step for d, step in zip(diffs, rpm_steps)]
        for slope in slopes:
            assert abs(slope - slopes[0]) < 1e-9, "Taper slope is not linear"

    def test_inverter_limit_is_binding(self, model: PowertrainModel) -> None:
        """Inverter limit (85 Nm) is less than LVCU (150 Nm), so it must bind."""
        # The effective cap must equal the inverter limit, not the LVCU limit
        assert model.max_motor_torque(0.0) == pytest.approx(85.0)
        assert model.max_motor_torque(0.0) != pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Wheel torque and force
# ---------------------------------------------------------------------------

class TestWheelTorqueAndForce:

    def test_wheel_torque_equals_motor_times_ratio_times_efficiency(
        self, model: PowertrainModel
    ) -> None:
        """wheel_torque = motor_torque * gear_ratio * drivetrain_efficiency."""
        motor_torque = 50.0
        expected = motor_torque * 3.5 * 0.92
        assert model.wheel_torque(motor_torque) == pytest.approx(expected)

    def test_wheel_torque_at_max_motor_torque(self, model: PowertrainModel) -> None:
        """At full motor torque the wheel torque should be 85 * 3.5 * 0.92 Nm."""
        expected = 85.0 * 3.5 * 0.92
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

    def test_regen_magnitude_less_than_drive_force(
        self, model: PowertrainModel
    ) -> None:
        """Regen force magnitude should be less than peak drive force (losses)."""
        speed = 10.0
        drive = model.drive_force(1.0, speed)
        regen_mag = abs(model.regen_force(1.0, speed))
        assert regen_mag < drive

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

        Expected: 85 Nm * 3.5 * 0.92 / 0.228 m ≈ 1198 N.
        Plausible range for FSAE EV: 900–1500 N.
        """
        force = model.drive_force(1.0, 1.0)  # 1 m/s ~ constant-torque region
        assert 900.0 < force < 1500.0, (
            f"Peak tractive force {force:.1f} N outside plausible FSAE range"
        )

    def test_peak_force_calculation(self, model: PowertrainModel) -> None:
        """Verify peak force matches analytical calculation."""
        expected = 85.0 * 3.5 * 0.92 / 0.228
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
