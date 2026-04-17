"""Tests for the quasi-static simulation engine."""

import dataclasses

import numpy as np
import pytest

from fsae_sim.data.loader import load_aim_csv, load_voltt_csv
from fsae_sim.driver.strategies import CoastOnlyStrategy
from fsae_sim.driver.strategy import ControlAction, ControlCommand, DriverStrategy, SimState
from fsae_sim.sim.engine import SimulationEngine, SimResult
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery import DischargeLimitPoint
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.dynamics import VehicleDynamics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_simple_track(num_segments=20, segment_length=50.0):
    """Flat oval: alternating straights and gentle corners."""
    segments = []
    for i in range(num_segments):
        is_corner = (i % 5 == 3) or (i % 5 == 4)
        segments.append(Segment(
            index=i,
            distance_start_m=i * segment_length,
            length_m=segment_length,
            curvature=0.04 if is_corner else 0.0,  # 25m radius
            grade=0.0,
        ))
    return Track(name="test_oval", segments=segments)


class FullThrottleStrategy(DriverStrategy):
    """Simple test strategy: always full throttle."""
    name = "full_throttle"

    def decide(self, state: SimState, upcoming):
        return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0)


@pytest.fixture
def vehicle_config(ct16ev_config_path):
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def battery_model(vehicle_config, voltt_cell_path):
    df = load_voltt_csv(voltt_cell_path)
    model = BatteryModel(vehicle_config.battery)
    model.calibrate_from_voltt(df)
    return model


@pytest.fixture
def simple_track():
    return _make_simple_track()


@pytest.fixture
def engine_full_throttle(vehicle_config, battery_model, simple_track):
    strategy = FullThrottleStrategy()
    return SimulationEngine(vehicle_config, simple_track, strategy, battery_model)


@pytest.fixture
def engine_coast(vehicle_config, battery_model, simple_track):
    dynamics = VehicleDynamics(vehicle_config.vehicle)
    strategy = CoastOnlyStrategy(dynamics)
    return SimulationEngine(vehicle_config, simple_track, strategy, battery_model)


# ---------------------------------------------------------------------------
# Basic engine behavior
# ---------------------------------------------------------------------------

class TestEngineBasics:

    def test_returns_sim_result(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert isinstance(result, SimResult)

    def test_correct_metadata(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert result.config_name == "CT-16EV"
        assert result.strategy_name == "full_throttle"
        assert result.track_name == "test_oval"

    def test_states_is_dataframe(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert not result.states.empty

    def test_laps_completed(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=2)
        assert result.laps_completed == 2

    def test_one_record_per_segment_per_lap(self, engine_full_throttle, simple_track):
        result = engine_full_throttle.run(num_laps=1)
        assert len(result.states) == simple_track.num_segments

    def test_two_laps_double_records(self, engine_full_throttle, simple_track):
        result = engine_full_throttle.run(num_laps=2)
        assert len(result.states) == 2 * simple_track.num_segments


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------

class TestPhysicalSanity:

    def test_time_positive(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert result.total_time_s > 0

    def test_energy_positive(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert result.total_energy_kwh > 0

    def test_soc_decreases(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1, initial_soc_pct=95.0)
        assert result.final_soc < 95.0

    def test_speed_stays_positive(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert (result.states["speed_ms"] >= 0).all()

    def test_soc_stays_bounded(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        assert (result.states["soc_pct"] >= 0).all()
        assert (result.states["soc_pct"] <= 100).all()

    def test_voltage_reasonable(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        # CT-16EV: 110S * 2.55V = 280.5V min, 110S * 4.195V = 461.5V max
        assert (result.states["pack_voltage_v"] > 250).all()
        assert (result.states["pack_voltage_v"] < 470).all()

    def test_temperature_increases(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1, initial_temp_c=25.0)
        final_temp = result.states["cell_temp_c"].iloc[-1]
        assert final_temp > 25.0

    def test_time_monotonically_increases(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        times = result.states["time_s"].values
        assert np.all(np.diff(times) > 0)


# ---------------------------------------------------------------------------
# Strategy interactions
# ---------------------------------------------------------------------------

class TestStrategyInteractions:

    def test_coast_uses_less_energy(self, engine_full_throttle, engine_coast):
        r_throttle = engine_full_throttle.run(num_laps=1)
        r_coast = engine_coast.run(num_laps=1)
        assert r_coast.total_energy_kwh < r_throttle.total_energy_kwh

    def test_coast_uses_less_energy_than_throttle(self, engine_full_throttle, engine_coast):
        """Coast strategy should consume less energy (may also be slower)."""
        r_throttle = engine_full_throttle.run(num_laps=1)
        r_coast = engine_coast.run(num_laps=1)
        # With downforce, the corner speed limit can be high enough that
        # coast and full-throttle produce similar lap times. But coast
        # always uses less energy.
        assert r_coast.total_energy_kwh <= r_throttle.total_energy_kwh


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------

class TestTermination:

    def test_stops_at_depleted_soc(self, vehicle_config, battery_model, simple_track):
        strategy = FullThrottleStrategy()
        engine = SimulationEngine(vehicle_config, simple_track, strategy, battery_model)
        # Start at very low SOC
        result = engine.run(num_laps=1000, initial_soc_pct=3.0)
        assert result.final_soc <= vehicle_config.battery.discharged_soc_pct

    def test_termination_temp_comes_from_config(
        self, vehicle_config, battery_model, simple_track,
    ):
        """NF-42: hot-termination threshold must track config.battery.discharge_limits[-1].

        The engine previously hardcoded 65 C; this test guards against that
        regression by swapping in a battery config whose last discharge-limit
        row is at 50 C.  Starting above 65 C but below 70 C would pass with
        the old magic number; here we verify the engine terminates at the
        configured 50 C instead of a code-embedded constant.
        """
        # Override the discharge-limit ceiling to 50 C instead of 65 C.
        custom_battery = dataclasses.replace(
            vehicle_config.battery,
            discharge_limits=(
                DischargeLimitPoint(temp_c=30.0, max_current_a=100.0),
                DischargeLimitPoint(temp_c=50.0, max_current_a=0.0),
            ),
        )
        custom_vehicle = dataclasses.replace(vehicle_config, battery=custom_battery)

        strategy = FullThrottleStrategy()
        engine = SimulationEngine(
            custom_vehicle, simple_track, strategy, battery_model,
        )
        # Start above the new 50 C ceiling; the first segment's post-step
        # temperature must already exceed termination_temp_c regardless of
        # battery-model thermal response, so exactly one record should land.
        result = engine.run(num_laps=5, initial_soc_pct=95.0, initial_temp_c=51.0)
        assert len(result.states) == 1
        assert result.laps_completed == 0


# ---------------------------------------------------------------------------
# State columns
# ---------------------------------------------------------------------------

class TestStateColumns:

    def test_required_columns_present(self, engine_full_throttle):
        result = engine_full_throttle.run(num_laps=1)
        expected_cols = [
            "lap", "segment_idx", "time_s", "distance_m", "speed_ms",
            "speed_kmh", "soc_pct", "pack_voltage_v", "pack_current_a",
            "cell_temp_c", "motor_rpm", "motor_torque_nm",
            "electrical_power_w", "action", "segment_time_s",
        ]
        for col in expected_cols:
            assert col in result.states.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Tire model integration tests  (Task 18)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch


class TestTireModelIntegration:
    """Verify engine constructs tire components when config has tire/suspension."""

    @patch("fsae_sim.sim.engine.CorneringSolver")
    @patch("fsae_sim.sim.engine.LoadTransferModel")
    @patch("fsae_sim.sim.engine.PacejkaTireModel")
    def test_constructs_components(
        self, mock_tire, mock_lt, mock_solver,
        vehicle_config, battery_model, simple_track,
    ):
        engine = SimulationEngine(
            vehicle_config, simple_track, FullThrottleStrategy(), battery_model,
        )
        mock_tire.assert_called_once()
        mock_lt.assert_called_once()
        mock_solver.assert_called_once()

    def test_legacy_without_tire_config(
        self, vehicle_config, battery_model, simple_track,
    ):
        mock_cfg = MagicMock(wraps=vehicle_config)
        mock_cfg.tire = None
        mock_cfg.suspension = None
        mock_cfg.vehicle = vehicle_config.vehicle
        mock_cfg.powertrain = vehicle_config.powertrain
        mock_cfg.battery = vehicle_config.battery
        mock_cfg.name = vehicle_config.name
        engine = SimulationEngine(
            mock_cfg, simple_track, FullThrottleStrategy(), battery_model,
        )
        assert engine.dynamics.cornering_solver is None


class TestTractionClamping:
    """Verify drive force is clamped to tire traction limit."""

    def test_drive_force_clamped(
        self, vehicle_config, battery_model, simple_track,
    ):
        engine = SimulationEngine(
            vehicle_config, simple_track, FullThrottleStrategy(), battery_model,
        )
        engine.dynamics.max_traction_force = lambda speed: 200.0
        result = engine.run(num_laps=1)
        throttle_mask = result.states["action"] == "throttle"
        if throttle_mask.any():
            assert result.states.loc[throttle_mask, "drive_force_n"].max() <= 200.01


class TestCorneringDragIntegration:
    """Verify engine passes curvature to total_resistance."""

    def test_corner_segments_have_higher_resistance(
        self, vehicle_config, battery_model,
    ):
        """A track with corners should produce higher resistance than all-straight."""
        straight_track = Track(
            name="straight",
            segments=[
                Segment(i, i * 50.0, 50.0, curvature=0.0, grade=0.0)
                for i in range(10)
            ],
        )
        curvy_track = Track(
            name="curvy",
            segments=[
                Segment(i, i * 50.0, 50.0, curvature=0.04, grade=0.0)
                for i in range(10)
            ],
        )
        strategy = FullThrottleStrategy()
        engine_straight = SimulationEngine(
            vehicle_config, straight_track, strategy, battery_model,
        )
        engine_curvy = SimulationEngine(
            vehicle_config, curvy_track, strategy, battery_model,
        )
        r_straight = engine_straight.run(num_laps=1)
        r_curvy = engine_curvy.run(num_laps=1)

        # Curvy track should consume more energy (higher resistance)
        assert r_curvy.total_energy_kwh > r_straight.total_energy_kwh
        # Curvy track resistance column should show higher values
        assert (
            r_curvy.states["resistance_force_n"].mean()
            > r_straight.states["resistance_force_n"].mean()
        )
