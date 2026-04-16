"""Tests for D-05 regen accounting on engine and validation sides.

Both sides of the sim-vs-telemetry comparison previously filtered to
positive power before summing energy, so every regen segment was
discarded.  These tests verify that discharge, regen, and net are now
tracked separately and that the default reporting is net.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fsae_sim.analysis.validation import validate_full_endurance
from fsae_sim.driver.strategies import CoastOnlyStrategy
from fsae_sim.driver.strategy import ControlAction, ControlCommand, DriverStrategy, SimState
from fsae_sim.sim.engine import SimResult, SimulationEngine
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel


def _make_simple_track(num_segments: int = 20, segment_length: float = 50.0) -> Track:
    segments = []
    for i in range(num_segments):
        # Corners on indices 3, 4 of every 5 so throttle/coast alternate.
        is_corner = (i % 5 == 3) or (i % 5 == 4)
        segments.append(Segment(
            index=i,
            distance_start_m=i * segment_length,
            length_m=segment_length,
            curvature=0.04 if is_corner else 0.0,
            grade=0.0,
        ))
    return Track(name="regen_test_oval", segments=segments)


class MixedThrottleCoastStrategy(DriverStrategy):
    """Alternates throttle and coast per segment.

    Uses COAST (rather than BRAKE) to exercise the regen accumulator
    via ``coast_electrical_power()`` — which returns negative watts at
    non-trivial RPM — without depending on the BRAKE→regen-torque
    path.  Keeps the test focused on the D-05 bookkeeping change.
    """

    name = "mixed_throttle_coast"

    def decide(self, state: SimState, upcoming):
        if state.segment_idx % 2 == 0:
            return ControlCommand(ControlAction.THROTTLE, throttle_pct=1.0, brake_pct=0.0)
        return ControlCommand(ControlAction.COAST, throttle_pct=0.0, brake_pct=0.0)


@pytest.fixture
def vehicle_config(ct16ev_config_path):
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def battery(vehicle_config, voltt_cell_path):
    from fsae_sim.data.loader import load_voltt_csv
    df = load_voltt_csv(voltt_cell_path)
    model = BatteryModel(vehicle_config.battery, cell_capacity_ah=4.5)
    model.calibrate(df)
    return model


class TestSimResultRegenFields:
    """SimResult exposes discharge, regen, net as separate fields."""

    def test_simresult_has_discharge_regen_net_fields(self):
        fields = {f.name for f in SimResult.__dataclass_fields__.values()}
        assert "total_discharge_kwh" in fields
        assert "total_regen_kwh" in fields
        assert "total_net_kwh" in fields

    def test_net_equals_discharge_minus_regen(self, vehicle_config, battery):
        track = _make_simple_track()
        engine = SimulationEngine(
            vehicle_config, track, MixedThrottleCoastStrategy(), battery,
        )
        result = engine.run(num_laps=2, initial_soc_pct=95.0, initial_temp_c=25.0)
        # The fundamental identity must hold exactly.
        assert result.total_net_kwh == pytest.approx(
            result.total_discharge_kwh - result.total_regen_kwh, abs=1e-9,
        )

    def test_total_energy_kwh_reports_net(self, vehicle_config, battery):
        track = _make_simple_track()
        engine = SimulationEngine(
            vehicle_config, track, MixedThrottleCoastStrategy(), battery,
        )
        result = engine.run(num_laps=2, initial_soc_pct=95.0, initial_temp_c=25.0)
        # Default total_energy_kwh is net.
        assert result.total_energy_kwh == pytest.approx(result.total_net_kwh, abs=1e-9)

    def test_regen_is_strictly_positive_when_coasting(self, vehicle_config, battery):
        track = _make_simple_track()
        engine = SimulationEngine(
            vehicle_config, track, MixedThrottleCoastStrategy(), battery,
        )
        result = engine.run(num_laps=2, initial_soc_pct=95.0, initial_temp_c=25.0)
        assert result.total_regen_kwh > 0.0, (
            "Coast action across 2 laps must produce some regen via back-EMF"
        )
        # Gross is larger than net (i.e., regen is actually being tracked,
        # not silently zeroed).
        assert (result.total_discharge_kwh + result.total_regen_kwh
                > result.total_discharge_kwh)


class TestValidationRegenAccounting:
    """validate_full_endurance computes telemetry energy as net (D-05)."""

    def _fabricate_aim_df(self, discharge_watts: float, regen_watts: float,
                          seg_seconds: float = 1.0, num_segments: int = 10):
        """Build a minimal telemetry df with known discharge/regen balance.

        Half the samples draw ``discharge_watts``, half return
        ``regen_watts`` (as negative power, since regen has I<0).
        """
        times = np.arange(num_segments, dtype=float) * seg_seconds
        # Split evenly between positive and negative power.
        half = num_segments // 2
        currents = np.concatenate([
            np.full(half, discharge_watts / 400.0),     # I = P/V
            np.full(num_segments - half, -regen_watts / 400.0),
        ])
        voltages = np.full(num_segments, 400.0)
        distance = np.linspace(0.0, 1000.0, num_segments)
        return pd.DataFrame({
            "Time": times,
            "Pack Voltage": voltages,
            "Pack Current": currents,
            "State of Charge": np.full(num_segments, 90.0),
            "Pack Temp": np.full(num_segments, 25.0),
            "GPS Speed": np.full(num_segments, 30.0),
            "Distance on GPS Speed": distance,
        })

    def test_telemetry_energy_reflects_regen(self):
        # Scale the fabricated energies so net > 0.1 kWh (the threshold
        # below which the Energy metric is suppressed).
        aim = self._fabricate_aim_df(discharge_watts=100_000.0, regen_watts=50_000.0,
                                     seg_seconds=10.0, num_segments=10)
        # Fabricate sim_states with matching time/distance so the
        # harness runs to the Energy metric.
        sim = pd.DataFrame({
            "segment_time_s": np.full(10, 10.0),
            "speed_ms": np.full(10, 30.0),
            "distance_m": np.linspace(0, 1000, 10),
            "soc_pct": np.full(10, 90.0),
            "pack_voltage_v": np.full(10, 400.0),
            "pack_current_a": np.full(10, 1.0),
            "cell_temp_c": np.full(10, 25.0),
        })
        # Expected telemetry net.  Note: np.diff(..., prepend=t[0]) gives
        # dt=0 for sample 0 (no contribution); samples 1-4 discharge,
        # samples 5-9 regen, each with dt=10.
        #   discharge: 4 samples * 100 kW * 10 s = 4.0 MJ
        #   regen:     5 samples *  50 kW * 10 s = 2.5 MJ
        #   net = 1.5 MJ = 0.4167 kWh
        expected_net_kwh = (4 * 100_000.0 * 10.0 - 5 * 50_000.0 * 10.0) / 3.6e6

        report = validate_full_endurance(
            sim_states=sim,
            aim_df=aim,
            sim_total_time_s=100.0,
            sim_final_soc=89.0,
            sim_total_energy_kwh=expected_net_kwh,
            sim_laps=1,
        )
        # Find the energy metric; it should now be net and named so.
        energy_metric = next(
            (m for m in report.metrics if "Energy" in m.name), None,
        )
        assert energy_metric is not None, "Energy metric missing from report"
        assert "net" in energy_metric.name.lower()
        assert energy_metric.telemetry_value == pytest.approx(
            expected_net_kwh, rel=1e-6,
        )
