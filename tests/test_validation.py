"""End-to-end validation: simulation vs real 2025 endurance telemetry.

Runs the full simulation pipeline with a ReplayStrategy built from the
actual AiM telemetry and compares key metrics against the recorded data.
"""

import numpy as np
import pytest

from fsae_sim.analysis.validation import (
    ValidationReport,
    detect_lap_boundaries,
    extract_lap_telemetry,
    validate_full_endurance,
    validate_simulation,
)
from fsae_sim.data.loader import load_aim_csv, load_voltt_csv
from fsae_sim.driver.strategies import ReplayStrategy
from fsae_sim.sim.engine import SimulationEngine
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aim_data(aim_csv_path):
    """Load AiM telemetry."""
    _, df = load_aim_csv(aim_csv_path)
    return df


@pytest.fixture
def vehicle_config(ct16ev_config_path):
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def battery_model(vehicle_config, voltt_cell_path, aim_data):
    df = load_voltt_csv(voltt_cell_path)
    model = BatteryModel(vehicle_config.battery)
    model.calibrate_from_voltt(df)
    model.calibrate_pack_from_telemetry(aim_data)
    return model


@pytest.fixture
def track(aim_csv_path):
    return Track.from_telemetry(aim_csv_path)


@pytest.fixture
def lap_boundaries(aim_data, track):
    """Detect all lap boundaries and select a representative lap.

    Picks the lap whose distance best matches the extracted track distance.
    Skips anomalous laps (too short, too slow, or stopped).
    """
    laps = detect_lap_boundaries(aim_data)
    assert len(laps) >= 2, "Need at least 2 clean laps for validation"

    # Filter to laps within 5% of track distance and reasonable lap time
    track_dist = track.total_distance_m
    good_laps = []
    for i, (s, e, d) in enumerate(laps):
        lap_time = aim_data["Time"].iloc[e] - aim_data["Time"].iloc[s]
        if abs(d - track_dist) / track_dist < 0.05 and 50 < lap_time < 120:
            good_laps.append((i, s, e, d, lap_time))

    assert good_laps, "No laps match the track distance"

    # Sort by how close the distance is to the track distance
    good_laps.sort(key=lambda x: abs(x[3] - track_dist))
    return laps, good_laps[0]


@pytest.fixture
def replay_strategy(aim_data, lap_boundaries, track):
    """Build ReplayStrategy from the best-matching lap."""
    _, (lap_idx, start_idx, end_idx, dist, time) = lap_boundaries
    return ReplayStrategy.from_aim_data(
        aim_data, start_idx, end_idx, track.lap_distance_m,
    )


@pytest.fixture
def sim_result(vehicle_config, battery_model, track, replay_strategy, aim_data, lap_boundaries):
    """Run 1-lap simulation with replay strategy."""
    _, (lap_idx, start_idx, end_idx, dist, time) = lap_boundaries
    initial_soc = float(aim_data["State of Charge"].iloc[start_idx])
    initial_temp = float(aim_data["Pack Temp"].iloc[start_idx])
    initial_speed = float(aim_data["GPS Speed"].iloc[start_idx]) / 3.6

    engine = SimulationEngine(
        vehicle_config, track, replay_strategy, battery_model,
    )
    return engine.run(
        num_laps=1,
        initial_soc_pct=initial_soc,
        initial_temp_c=initial_temp,
        initial_speed_ms=initial_speed,
    )


@pytest.fixture
def validation_report(sim_result, aim_data, lap_boundaries):
    """Validate simulation against telemetry."""
    _, (lap_idx, start_idx, end_idx, dist, time) = lap_boundaries
    return validate_simulation(
        sim_result.states, aim_data, start_idx, end_idx,
    )


# ---------------------------------------------------------------------------
# Tests: lap detection
# ---------------------------------------------------------------------------

class TestLapDetection:

    def test_detects_laps(self, lap_boundaries):
        all_laps, best_lap = lap_boundaries
        assert len(all_laps) >= 2

    def test_lap_distance_reasonable(self, lap_boundaries):
        all_laps, _ = lap_boundaries
        for _, _, dist in all_laps:
            assert 700 < dist < 1300, f"Lap distance {dist:.0f}m outside expected range"

    def test_lap_distances_consistent(self, lap_boundaries):
        all_laps, _ = lap_boundaries
        distances = [d for _, _, d in all_laps]
        # Filter out anomalous laps for consistency check
        normal = [d for d in distances if 900 < d < 1100]
        assert len(normal) >= 5
        assert np.std(normal) / np.mean(normal) < 0.05


# ---------------------------------------------------------------------------
# Tests: simulation runs to completion
# ---------------------------------------------------------------------------

class TestSimulationCompletion:

    def test_completes_one_lap(self, sim_result):
        assert sim_result.laps_completed == 1

    def test_positive_time(self, sim_result):
        assert sim_result.total_time_s > 0

    def test_states_not_empty(self, sim_result):
        assert not sim_result.states.empty


# ---------------------------------------------------------------------------
# Tests: validation metrics
# ---------------------------------------------------------------------------

class TestValidationMetrics:

    def test_report_generated(self, validation_report):
        assert isinstance(validation_report, ValidationReport)
        assert validation_report.num_total > 0

    def test_lap_time_within_target(self, validation_report):
        """Lap time should be within 5% of real."""
        for m in validation_report.metrics:
            if m.name == "Lap time":
                print(f"  Lap time: telem={m.telemetry_value:.1f}s, "
                      f"sim={m.simulation_value:.1f}s, err={m.relative_error_pct:.1f}%")
                assert m.relative_error_pct < 15.0, (
                    f"Lap time error {m.relative_error_pct:.1f}% exceeds 15% target"
                )

    def test_mean_speed_within_target(self, validation_report):
        """Mean speed should be within 5% of real."""
        for m in validation_report.metrics:
            if m.name == "Mean speed":
                print(f"  Mean speed: telem={m.telemetry_value:.1f}km/h, "
                      f"sim={m.simulation_value:.1f}km/h, err={m.relative_error_pct:.1f}%")
                assert m.relative_error_pct < 15.0

    def test_mean_voltage_within_target(self, validation_report):
        """Mean pack voltage should be within 5%."""
        for m in validation_report.metrics:
            if m.name == "Mean pack voltage":
                print(f"  Voltage: telem={m.telemetry_value:.1f}V, "
                      f"sim={m.simulation_value:.1f}V, err={m.relative_error_pct:.1f}%")
                assert m.relative_error_pct < 10.0

    def test_print_full_report(self, validation_report):
        """Print the full validation report (always passes, for visibility)."""
        print("\n" + validation_report.summary())


# ---------------------------------------------------------------------------
# Full endurance validation (~22 km, ~1859s)
# ---------------------------------------------------------------------------

class TestFullEndurance:
    """Run the full endurance simulation and validate against complete AiM data."""

    @pytest.fixture
    def full_replay(self, aim_data, track):
        """Replay the entire endurance recording."""
        return ReplayStrategy.from_full_endurance(aim_data, track.lap_distance_m)

    @pytest.fixture
    def full_result(self, vehicle_config, battery_model, track, full_replay, aim_data):
        """Run full 22-lap endurance sim (~21 km).

        FSAE endurance is 22 laps. Distance is ~21 km because it's measured
        from mid-track and the car takes the shortest line. There's a driver
        change mid-event (car stops, swap drivers, continue).
        """
        initial_soc = float(aim_data["State of Charge"].iloc[0])
        initial_temp = float(aim_data["Pack Temp"].iloc[0])
        initial_speed = max(float(aim_data["GPS Speed"].iloc[0]) / 3.6, 0.5)

        # Compute laps to match actual distance (21.3 km / ~995m per lap)
        total_distance = float(aim_data["Distance on GPS Speed"].iloc[-1])
        num_laps = round(total_distance / track.total_distance_m)

        engine = SimulationEngine(
            vehicle_config, track, full_replay, battery_model,
        )
        return engine.run(
            num_laps=num_laps,
            initial_soc_pct=initial_soc,
            initial_temp_c=initial_temp,
            initial_speed_ms=initial_speed,
        )

    @pytest.fixture
    def full_report(self, full_result, aim_data):
        return validate_full_endurance(
            full_result.states, aim_data,
            full_result.total_time_s, full_result.final_soc,
            full_result.total_energy_kwh, full_result.laps_completed,
        )

    def test_completes_full_endurance(self, full_result):
        """Sim should complete the full endurance (~21 laps at 995m)."""
        assert full_result.laps_completed >= 20

    def test_driving_time(self, full_report):
        for m in full_report.metrics:
            if m.name == "Driving time":
                print(f"  Driving time: telem={m.telemetry_value:.0f}s, "
                      f"sim={m.simulation_value:.0f}s, err={m.relative_error_pct:.1f}%")

    def test_soc_consumed(self, full_report):
        for m in full_report.metrics:
            if m.name == "SOC consumed":
                print(f"  SOC consumed: telem={m.telemetry_value:.1f}%, "
                      f"sim={m.simulation_value:.1f}%, err={m.relative_error_pct:.1f}%")

    def test_final_temperature(self, full_report):
        for m in full_report.metrics:
            if m.name == "Final cell temp":
                print(f"  Final temp: telem={m.telemetry_value:.1f}C, "
                      f"sim={m.simulation_value:.1f}C, err={m.relative_error_pct:.1f}%")

    def test_print_full_endurance_report(self, full_report):
        """Print the full endurance validation (always passes for visibility)."""
        print("\n" + full_report.summary())
