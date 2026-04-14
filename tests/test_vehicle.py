"""Tests for vehicle configuration loading."""

import pytest
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.vehicle import TireConfig, SuspensionConfig, VehicleParams


class TestVehicleConfigLoading:
    """Test YAML config loading into dataclasses."""

    def test_load_ct16ev(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.name == "CT-16EV"
        assert config.year == 2025

    def test_vehicle_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.vehicle.mass_kg == 278.0  # DSS: 210 car + 68 driver
        assert config.vehicle.drag_coefficient == 1.502  # DSS: CdA from drag data
        assert config.vehicle.rolling_resistance == 0.015

    def test_powertrain_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.powertrain.motor_speed_max_rpm == 2900
        assert config.powertrain.torque_limit_inverter_nm == 85.0
        assert config.powertrain.iq_limit_a == 170.0
        assert config.powertrain.gear_ratio == 3.818  # DSS: final drive ratio

    def test_battery_params(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.battery.cell_type == "P45B"
        assert config.battery.series == 110
        assert config.battery.parallel == 4
        assert config.battery.cell_voltage_min_v == 2.55
        assert config.battery.cell_voltage_max_v == 4.20  # DSS: 4.2V

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


class TestTireConfig:
    def test_tire_config_construction(self):
        tc = TireConfig(tir_file="path/to/file.tir", static_camber_front_deg=-1.25, static_camber_rear_deg=-1.25)
        assert tc.tir_file == "path/to/file.tir"
        assert tc.static_camber_front_deg == -1.25
        assert tc.static_camber_rear_deg == -1.25

    def test_tire_config_is_frozen(self):
        tc = TireConfig(tir_file="path/to/file.tir", static_camber_front_deg=-1.25, static_camber_rear_deg=-1.25)
        with pytest.raises(AttributeError):
            tc.tir_file = "other.tir"

    def test_tire_config_requires_all_fields(self):
        with pytest.raises(TypeError):
            TireConfig(tir_file="path/to/file.tir")


class TestSuspensionConfig:
    def test_suspension_config_construction(self):
        sc = SuspensionConfig(
            roll_stiffness_front_nm_per_deg=238.0, roll_stiffness_rear_nm_per_deg=258.0,
            roll_center_height_front_mm=88.9, roll_center_height_rear_mm=63.5,
            roll_camber_front_deg_per_deg=-0.5, roll_camber_rear_deg_per_deg=-0.554,
            front_track_mm=1194.0, rear_track_mm=1168.0,
        )
        assert sc.roll_stiffness_front_nm_per_deg == 238.0
        assert sc.rear_track_mm == 1168.0

    def test_suspension_config_is_frozen(self):
        sc = SuspensionConfig(
            roll_stiffness_front_nm_per_deg=238.0, roll_stiffness_rear_nm_per_deg=258.0,
            roll_center_height_front_mm=88.9, roll_center_height_rear_mm=63.5,
            roll_camber_front_deg_per_deg=-0.5, roll_camber_rear_deg_per_deg=-0.554,
            front_track_mm=1194.0, rear_track_mm=1168.0,
        )
        with pytest.raises(AttributeError):
            sc.front_track_mm = 1200.0

    def test_suspension_config_requires_all_fields(self):
        with pytest.raises(TypeError):
            SuspensionConfig(roll_stiffness_front_nm_per_deg=238.0, roll_stiffness_rear_nm_per_deg=258.0)


class TestVehicleConfigOptionalFields:
    def test_vehicle_config_with_tire_and_suspension(self):
        from fsae_sim.vehicle.powertrain import PowertrainConfig
        from fsae_sim.vehicle.battery import BatteryConfig

        vp = VehicleParams(
            mass_kg=278.0, frontal_area_m2=1.0, drag_coefficient=1.5,
            rolling_resistance=0.015, wheelbase_m=1.549,
        )
        pt = PowertrainConfig(
            motor_speed_max_rpm=2900, brake_speed_rpm=2400,
            torque_limit_inverter_nm=85.0, torque_limit_lvcu_nm=150.0,
            iq_limit_a=170.0, id_limit_a=30.0, gear_ratio=3.818,
            drivetrain_efficiency=0.92,
        )
        bt = BatteryConfig.from_dict({
            "cell_type": "P45B",
            "topology": {"series": 110, "parallel": 4},
            "cell_voltage_min_v": 2.55,
            "cell_voltage_max_v": 4.20,
            "discharged_soc_pct": 2.0,
            "soc_taper": {"threshold_pct": 85.0, "rate_a_per_pct": 1.0},
            "discharge_limits": [
                {"temp_c": 30.0, "max_current_a": 100.0},
                {"temp_c": 65.0, "max_current_a": 0.0},
            ],
        })
        tc = TireConfig(
            tir_file="path/to/file.tir",
            static_camber_front_deg=-1.25,
            static_camber_rear_deg=-1.25,
        )
        sc = SuspensionConfig(
            roll_stiffness_front_nm_per_deg=238.0,
            roll_stiffness_rear_nm_per_deg=258.0,
            roll_center_height_front_mm=88.9,
            roll_center_height_rear_mm=63.5,
            roll_camber_front_deg_per_deg=-0.5,
            roll_camber_rear_deg_per_deg=-0.554,
            front_track_mm=1194.0,
            rear_track_mm=1168.0,
        )
        config = VehicleConfig(
            name="test", year=2025, description="test",
            vehicle=vp, powertrain=pt, battery=bt, tire=tc, suspension=sc,
        )
        assert config.tire is tc
        assert config.suspension is sc

    def test_from_yaml_parses_tire_and_suspension(self, tmp_path):
        yaml_content = (
            "name: test\nyear: 2025\ndescription: test\n"
            "vehicle:\n  mass_kg: 278.0\n  frontal_area_m2: 1.0\n  drag_coefficient: 1.5\n"
            "  rolling_resistance: 0.015\n  wheelbase_m: 1.549\n"
            "powertrain:\n  motor_speed_max_rpm: 2900\n  brake_speed_rpm: 2400\n"
            "  torque_limit_inverter_nm: 85.0\n  torque_limit_lvcu_nm: 150.0\n"
            "  iq_limit_a: 170.0\n  id_limit_a: 30.0\n  gear_ratio: 3.818\n"
            "  drivetrain_efficiency: 0.92\n"
            "battery:\n  cell_type: P45B\n  topology: {series: 110, parallel: 4}\n"
            "  cell_voltage_min_v: 2.55\n  cell_voltage_max_v: 4.20\n  discharged_soc_pct: 2.0\n"
            "  soc_taper: {threshold_pct: 85.0, rate_a_per_pct: 1.0}\n"
            "  discharge_limits:\n    - {temp_c: 30.0, max_current_a: 100.0}\n"
            "    - {temp_c: 65.0, max_current_a: 0.0}\n"
            "tire:\n  tir_file: path/to/tire.tir\n  static_camber_front_deg: -1.5\n"
            "  static_camber_rear_deg: -2.0\n"
            "suspension:\n  roll_stiffness_front_nm_per_deg: 238.0\n"
            "  roll_stiffness_rear_nm_per_deg: 258.0\n  roll_center_height_front_mm: 88.9\n"
            "  roll_center_height_rear_mm: 63.5\n  roll_camber_front_deg_per_deg: -0.5\n"
            "  roll_camber_rear_deg_per_deg: -0.554\n  front_track_mm: 1194.0\n"
            "  rear_track_mm: 1168.0\n"
        )
        (tmp_path / "cfg.yaml").write_text(yaml_content)
        config = VehicleConfig.from_yaml(tmp_path / "cfg.yaml")
        assert config.tire is not None and config.tire.tir_file == "path/to/tire.tir"
        assert config.suspension is not None and config.suspension.front_track_mm == 1194.0


class TestCT16EVTireSuspensionLoading:
    def test_ct16ev_tire_config_loaded(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.tire is not None
        assert config.tire.tir_file == (
            "Real-Car-Data-And-Stats/Tire Models from TTC/"
            "Round_8_Hoosier_LC0_16x7p5_10_on_8in_10psi_PAC02_UM2.tir"
        )
        assert config.tire.static_camber_front_deg == -1.25

    def test_ct16ev_suspension_config_loaded(self, ct16ev_config_path):
        config = VehicleConfig.from_yaml(ct16ev_config_path)
        assert config.suspension is not None
        assert config.suspension.roll_stiffness_front_nm_per_deg == 238.0
        assert config.suspension.front_track_mm == 1194.0


class TestInitExports:
    def test_tire_config_importable(self):
        from fsae_sim.vehicle import TireConfig
        assert TireConfig is not None

    def test_suspension_config_importable(self):
        from fsae_sim.vehicle import SuspensionConfig
        assert SuspensionConfig is not None
