"""Tests for vehicle configuration loading."""

from fsae_sim.vehicle import VehicleConfig


class TestVehicleConfigLoading:
    """Test YAML config loading into dataclasses."""

    def test_load_ct16ev(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.name == "CT-16EV"
        assert config.year == 2025

    def test_vehicle_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.vehicle.mass_kg == 270.0
        assert config.vehicle.drag_coefficient == 0.5
        assert config.vehicle.rolling_resistance == 0.015

    def test_powertrain_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.powertrain.motor_speed_max_rpm == 2900
        assert config.powertrain.torque_limit_inverter_nm == 85.0
        assert config.powertrain.iq_limit_a == 170.0
        assert config.powertrain.gear_ratio == 3.5

    def test_battery_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.battery.cell_type == "P45B"
        assert config.battery.series == 110
        assert config.battery.parallel == 4
        assert config.battery.cell_voltage_min_v == 2.55
        assert config.battery.cell_voltage_max_v == 4.195

    def test_discharge_limits(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        limits = config.battery.discharge_limits
        assert len(limits) == 8
        assert limits[0].temp_c == 30.0
        assert limits[0].max_current_a == 100.0
        assert limits[-1].temp_c == 65.0
        assert limits[-1].max_current_a == 0.0

    def test_soc_taper(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.battery.soc_taper_threshold_pct == 85.0
        assert config.battery.soc_taper_rate_a_per_pct == 1.0

    def test_load_ct17ev(self, ct17ev_config_path):
        config = VehicleConfig.from_yaml(ct17ev_config_path)
        assert config.name == "CT-17EV"
        assert config.year == 2026
        assert config.vehicle.mass_kg == 261.0
        assert config.battery.cell_type == "P50B"
        assert config.battery.series == 100
