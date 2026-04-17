"""Integration test: SimulationEngine uses SpeedEnvelope for synthetic strategies."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle.vehicle import VehicleParams, VehicleConfig, TireConfig, SuspensionConfig
from fsae_sim.vehicle.powertrain import PowertrainConfig
from fsae_sim.vehicle.battery import BatteryConfig, DischargeLimitPoint
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.driver.strategy import ControlAction, ControlCommand, DriverStrategy, SimState
from fsae_sim.sim.engine import SimulationEngine


class StubStrategy(DriverStrategy):
    """Always full throttle."""
    name = "full_throttle"

    def decide(self, state: SimState, upcoming_segments) -> ControlCommand:
        return ControlCommand(
            action=ControlAction.THROTTLE,
            throttle_pct=1.0,
            brake_pct=0.0,
        )


def make_simple_track() -> Track:
    """5 straights + 3 corner (kappa=0.1) + 5 straights = 13 segments."""
    segs = (
        [Segment(i, i * 5.0, 5.0, 0.0, 0.0) for i in range(5)]
        + [Segment(i + 5, (i + 5) * 5.0, 5.0, 0.1, 0.0) for i in range(3)]
        + [Segment(i + 8, (i + 8) * 5.0, 5.0, 0.0, 0.0) for i in range(5)]
    )
    return Track(name="test_circuit", segments=segs)


def make_minimal_config() -> VehicleConfig:
    return VehicleConfig(
        name="test",
        year=2025,
        description="test",
        vehicle=VehicleParams(
            mass_kg=288.0,
            frontal_area_m2=1.0,
            drag_coefficient=1.5,
            rolling_resistance=0.015,
            wheelbase_m=1.549,
            downforce_coefficient=2.18,
        ),
        powertrain=PowertrainConfig(
            motor_speed_max_rpm=2900,
            brake_speed_rpm=2400,
            torque_limit_inverter_nm=85.0,
            torque_limit_lvcu_nm=150.0,
            iq_limit_a=170.0,
            id_limit_a=30.0,
            gear_ratio=3.6363,
            drivetrain_efficiency=0.92,
        ),
        battery=BatteryConfig(
            cell_type="P45B",
            series=110,
            parallel=4,
            cell_voltage_min_v=2.55,
            cell_voltage_max_v=4.2,
            discharged_soc_pct=2.0,
            soc_taper_threshold_pct=85.0,
            soc_taper_rate_a_per_pct=1.0,
            discharge_limits=[
                DischargeLimitPoint(30.0, 100.0),
                DischargeLimitPoint(65.0, 0.0),
            ],
        ),
    )


class TestEnvelopeIntegration:
    """Engine should use speed envelope for synthetic strategies."""

    def test_synthetic_strategy_uses_envelope(self):
        """With envelope, corner entry speed should be lower than corner speed limit."""
        track = make_simple_track()
        config = make_minimal_config()
        strategy = StubStrategy()

        batt = MagicMock(spec=BatteryModel)
        batt.pack_voltage.return_value = 400.0
        batt.max_discharge_current.return_value = 100.0
        batt.step.return_value = (90.0, 26.0, 395.0)

        engine = SimulationEngine(config, track, strategy, batt)
        result = engine.run(num_laps=1, initial_soc_pct=95.0)

        assert result.total_time_s > 0
        assert result.laps_completed == 1

        states = result.states

        # Envelope values are finite for every segment (not inf like raw max_cornering_speed
        # on straights would be), because the envelope is the forward-backward minimum.
        assert (states["corner_speed_limit_ms"] < float("inf")).all(), (
            "Envelope values should all be finite (not inf from raw max_cornering_speed)"
        )

        # Corner segments must be at or below the envelope-derived speed limit.
        corner_rows = states[states["curvature"].abs() > 0.05]
        if len(corner_rows) > 0:
            for _, row in corner_rows.iterrows():
                assert row["speed_ms"] <= row["corner_speed_limit_ms"] + 0.5, (
                    f"Speed {row['speed_ms']:.2f} exceeded envelope limit "
                    f"{row['corner_speed_limit_ms']:.2f} at segment {int(row['segment_idx'])}"
                )


class TestZoneMaxSpeedCap:
    """D-09: engine honors ``ControlCommand.metadata['max_speed_ms']``."""

    def test_metadata_caps_exit_speed(self):
        """Strategy with zone cap of 10 m/s should never let speed exceed it."""
        from fsae_sim.driver.strategy import ControlAction, ControlCommand

        track = make_simple_track()
        config = make_minimal_config()

        class CappedStrategy(DriverStrategy):
            name = "capped"

            def decide(self, state, upcoming):
                return ControlCommand(
                    action=ControlAction.THROTTLE,
                    throttle_pct=1.0,
                    brake_pct=0.0,
                    metadata={"max_speed_ms": 10.0},
                )

        batt = MagicMock(spec=BatteryModel)
        batt.pack_voltage.return_value = 400.0
        batt.max_discharge_current.return_value = 100.0
        batt.step.return_value = (90.0, 26.0, 395.0)

        engine = SimulationEngine(config, track, CappedStrategy(), batt)
        result = engine.run(num_laps=1, initial_soc_pct=95.0)

        # Every recorded state must respect the 10 m/s cap (with a tiny
        # epsilon for the (entry+exit)/2 rounding in avg_speed reporting).
        assert (result.states["speed_ms"] <= 10.0 + 1e-6).all(), (
            f"D-09 regression: speeds exceeded zone cap "
            f"(max={result.states['speed_ms'].max():.3f} > 10.0)"
        )


class TestPackCurrentCarriedForward:
    """D-11: SimState.pack_current carries last segment's pack current."""

    def test_state_pack_current_is_last_iteration_value(self):
        from fsae_sim.driver.strategy import ControlAction, ControlCommand

        track = make_simple_track()
        config = make_minimal_config()

        class SpyStrategy(DriverStrategy):
            name = "spy"

            def __init__(self):
                self.observed_pack_currents = []

            def decide(self, state, upcoming):
                self.observed_pack_currents.append(state.pack_current)
                return ControlCommand(
                    action=ControlAction.THROTTLE,
                    throttle_pct=0.6,
                    brake_pct=0.0,
                )

        batt = MagicMock(spec=BatteryModel)
        batt.pack_voltage.return_value = 400.0
        batt.max_discharge_current.return_value = 100.0
        batt.step.return_value = (90.0, 26.0, 395.0)

        spy = SpyStrategy()
        engine = SimulationEngine(config, track, spy, batt)
        result = engine.run(num_laps=1, initial_soc_pct=95.0)

        # First segment sees 0.0 (no prior segment); later segments must
        # see non-zero pack_current reflecting the previous segment's draw.
        assert spy.observed_pack_currents[0] == 0.0
        nonzero_later = [pc for pc in spy.observed_pack_currents[1:] if pc != 0.0]
        assert len(nonzero_later) > 0, (
            "D-11 regression: SimState.pack_current stuck at 0 across segments."
        )
        # And the recorded DataFrame's final pack_current_a should match
        # what the next iteration would have seen if the loop continued.
        assert result.states["pack_current_a"].iloc[-1] != 0.0


class TestRollingStartFlag:
    """D-26: ``rolling_start=False`` skips the MIN_SPEED clamp."""

    def _make_engine(self):
        track = make_simple_track()
        config = make_minimal_config()
        strategy = StubStrategy()
        batt = MagicMock(spec=BatteryModel)
        batt.pack_voltage.return_value = 400.0
        batt.max_discharge_current.return_value = 100.0
        batt.step.return_value = (90.0, 26.0, 395.0)
        return SimulationEngine(config, track, strategy, batt)

    def test_standing_start_skips_clamp(self):
        """With rolling_start=False the strategy observes a true zero entry."""
        from fsae_sim.driver.strategy import ControlAction, ControlCommand

        track = make_simple_track()
        config = make_minimal_config()

        class SpyStrategy(DriverStrategy):
            name = "spy"

            def __init__(self):
                self.first_observed_speed = None

            def decide(self, state, upcoming):
                if self.first_observed_speed is None:
                    self.first_observed_speed = state.speed
                return ControlCommand(
                    action=ControlAction.COAST,
                    throttle_pct=0.0,
                    brake_pct=0.0,
                )

        batt = MagicMock(spec=BatteryModel)
        batt.pack_voltage.return_value = 400.0
        batt.max_discharge_current.return_value = 100.0
        batt.step.return_value = (90.0, 26.0, 395.0)

        spy = SpyStrategy()
        engine = SimulationEngine(config, track, spy, batt)
        engine.run(
            num_laps=1,
            initial_soc_pct=95.0,
            initial_speed_ms=0.0,
            rolling_start=False,
        )

        # With rolling_start=False the engine must not bump the first
        # entry speed up to _MIN_SPEED_MS (0.5). The strategy therefore
        # sees exactly 0.0 on the first iteration.
        assert spy.first_observed_speed == 0.0, (
            "rolling_start=False should not clamp initial_speed_ms up "
            f"(first observed speed was {spy.first_observed_speed})"
        )

    def test_rolling_start_warns_on_clamp(self):
        import warnings
        engine = self._make_engine()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.run(
                num_laps=1,
                initial_soc_pct=95.0,
                initial_speed_ms=0.0,
                rolling_start=True,
            )
        clamp_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "rolling_start" in str(w.message)
        ]
        assert len(clamp_warnings) >= 1, (
            "rolling_start=True with initial_speed_ms<_MIN_SPEED_MS "
            "should emit a UserWarning"
        )
