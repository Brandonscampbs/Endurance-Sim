"""End-to-end tests for the engine's ``is_adaptive`` dispatch (Sub-task C).

Sub-task C wires :class:`fsae_sim.driver.strategies.AdaptiveStrategy` into
the SimulationEngine's per-segment torque dispatch:

- AdaptiveStrategy command outputs flow through ``lvcu_torque_command``
  (not ``pedal_to_torque_request``) because the adaptive driver emits
  raw pedal positions that already respect the LVCU deadzone.
- The driver emits a ``regen_request_pct`` channel separately from the
  hydraulic brake. The engine maps it to ``powertrain.regen_force`` /
  the negative-torque path so battery sees negative pack current when
  regen is active.
- ``AdaptiveDriver.reset()`` is called at the start of each run so PI
  integrator state is deterministic across consecutive sims.

The tests cover smoke / sign-convention / determinism only; the
quantitative Michigan replay-equivalent acceptance run lives in Sub-
task D's ``scripts/validate_adaptive.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsae_sim.driver.strategy import ControlAction, ControlCommand
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain_model import PowertrainModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_track(num_segments: int = 80, seg_len: float = 1.0) -> Track:
    """Short synthetic track: a flat straight then a single mild corner.

    Anchor corners at the lap-wrap boundary (first/last few segments)
    so the speed envelope's backward pass has a finite v_corner to brake
    against — otherwise the lap-wrap fixed point starts at infinity and
    the brake-force helper blows up on inf inputs.
    """
    segs = []
    for i in range(num_segments):
        # Anchor curvature at lap wrap and an interior corner cluster.
        if i < 3 or i >= num_segments - 3 or (30 <= i < 50):
            curvature = 0.02  # ~50 m radius
        else:
            curvature = 0.0
        segs.append(Segment(
            index=i,
            distance_start_m=i * seg_len,
            length_m=seg_len,
            curvature=curvature,
            grade=0.0,
        ))
    return Track(name="synth_corner", segments=segs, source="synthetic")


@pytest.fixture
def vehicle_config(ct16ev_config_path) -> VehicleConfig:
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def battery(vehicle_config, voltt_cell_path) -> BatteryModel:
    from fsae_sim.data.loader import load_voltt_csv
    df = load_voltt_csv(voltt_cell_path)
    model = BatteryModel(vehicle_config.battery)
    model.calibrate_from_voltt(df)
    return model


@pytest.fixture
def synth_track() -> Track:
    return _synthetic_track()


# ---------------------------------------------------------------------------
# 1. ControlCommand regen_request_pct slot
# ---------------------------------------------------------------------------


def test_control_command_regen_request_pct_defaults_zero():
    """Sub-task C adds a ``regen_request_pct`` slot to ControlCommand
    with default 0.0 so existing strategies remain bytewise-compatible.
    Adaptive strategy will route this to the regen channel.
    """
    cmd = ControlCommand(action=ControlAction.COAST)
    assert hasattr(cmd, "regen_request_pct")
    assert cmd.regen_request_pct == 0.0


def test_control_command_regen_request_pct_can_be_set():
    cmd = ControlCommand(
        action=ControlAction.BRAKE,
        throttle_pct=0.0,
        brake_pct=0.3,
        regen_request_pct=0.7,
    )
    assert cmd.regen_request_pct == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 2. End-to-end smoke test with AdaptiveStrategy
# ---------------------------------------------------------------------------


def test_adaptive_engine_run_completes_on_synthetic_track(
    vehicle_config, synth_track, battery,
):
    """Sub-task C smoke: an AdaptiveStrategy sim runs end-to-end without
    raising. The track is short enough that the adaptive driver should
    behave well across one lap."""
    from fsae_sim.driver.strategies import AdaptiveStrategy
    from fsae_sim.sim.engine import SimulationEngine, SimulationMode

    strategy = AdaptiveStrategy.from_config(vehicle_config, synth_track)
    engine = SimulationEngine(
        vehicle_config, synth_track, strategy, battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    )
    result = engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=30.0)
    assert result.total_time_s > 0.0
    assert result.laps_completed >= 1
    assert len(result.states) == synth_track.num_segments


# ---------------------------------------------------------------------------
# 3. Regen sign convention
# ---------------------------------------------------------------------------


def test_adaptive_regen_request_yields_negative_pack_current(
    vehicle_config, battery,
):
    """When the adaptive driver wants decel and emits regen_request_pct,
    the engine must route it as negative motor torque so pack_current is
    negative on regen segments.

    Track shape: long straight followed by a tight corner. The speed
    envelope must drop into the corner, so the adaptive driver issues
    regen + brake into corner entry. The engine must observe negative
    pack current on those segments.
    """
    from fsae_sim.driver.strategies import AdaptiveStrategy
    from fsae_sim.sim.engine import SimulationEngine, SimulationMode

    n = 120
    segs = []
    for i in range(n):
        # Long straight, then a tight corner in the middle so the
        # forward-backward envelope produces a meaningful decel zone.
        if 60 <= i < 80:
            curvature = 0.08  # ~12.5 m radius — very tight corner
        elif i < 3 or i >= n - 3:
            curvature = 0.02  # gentle lap-wrap anchor
        else:
            curvature = 0.0
        segs.append(Segment(
            index=i, distance_start_m=i * 0.5, length_m=0.5,
            curvature=curvature, grade=0.0,
        ))
    track = Track(name="straight_then_corner", segments=segs, source="synthetic")
    strategy = AdaptiveStrategy.from_config(vehicle_config, track)

    engine = SimulationEngine(
        vehicle_config, track, strategy, battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    )
    # Roll into the lap at a reasonable speed; the corner forces the
    # envelope to slow the car around the apex.
    result = engine.run(
        num_laps=1,
        initial_soc_pct=95.0,
        initial_temp_c=30.0,
        initial_speed_ms=20.0,
    )
    states = result.states
    # Some segments must have negative pack current (regen).
    assert (states["pack_current_a"] < 0.0).any(), (
        "Expected at least one segment with negative pack current on a "
        "corner-entry track with AdaptiveStrategy; got pack_current range "
        f"[{states['pack_current_a'].min():.2f}, {states['pack_current_a'].max():.2f}] A. "
        f"regen_force_n range "
        f"[{states['regen_force_n'].min():.2f}, {states['regen_force_n'].max():.2f}] N, "
        f"motor_torque_nm range "
        f"[{states['motor_torque_nm'].min():.2f}, {states['motor_torque_nm'].max():.2f}] Nm."
    )


# ---------------------------------------------------------------------------
# 4. PI state resets at the start of each run
# ---------------------------------------------------------------------------


def test_adaptive_pi_state_resets_between_runs(
    vehicle_config, synth_track, battery,
):
    """Back-to-back runs from the same AdaptiveStrategy instance must
    produce identical state DataFrames. Without ``reset()`` at run
    start, the PI integrator would carry over and break determinism.
    """
    from fsae_sim.driver.strategies import AdaptiveStrategy
    from fsae_sim.sim.engine import SimulationEngine, SimulationMode

    strategy = AdaptiveStrategy.from_config(vehicle_config, synth_track)
    engine_a = SimulationEngine(
        vehicle_config, synth_track, strategy, battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    )
    result_a = engine_a.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=30.0)
    # Fresh engine but same strategy instance — driver-side PI state
    # should be cleared so the second run reproduces the first.
    engine_b = SimulationEngine(
        vehicle_config, synth_track, strategy, battery,
        mode=SimulationMode.PREDICTION,
        allow_telemetry_track=True,
        allow_empirical_grip=True,
    )
    result_b = engine_b.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=30.0)
    assert result_a.total_time_s == pytest.approx(result_b.total_time_s, abs=1e-6)
    assert result_a.net_charge_ah == pytest.approx(result_b.net_charge_ah, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Existing strategies still carry default regen_request_pct = 0
# ---------------------------------------------------------------------------


def test_calibrated_strategy_emits_regen_request_pct_zero():
    """CalibratedStrategy never asks for regen; the default 0.0 slot
    must round-trip through decide()."""
    from fsae_sim.driver.strategies import CalibratedStrategy
    from fsae_sim.driver.strategy import SimState
    from fsae_sim.analysis.telemetry_analysis import DriverZone

    track = _synthetic_track()
    zone = DriverZone(
        zone_id=0,
        segment_start=0,
        segment_end=track.num_segments - 1,
        action=ControlAction.THROTTLE,
        intensity=0.7,
        distance_start_m=0.0,
        distance_end_m=track.total_distance_m,
        label="straight",
        max_speed_ms=0.0,
        max_speed_source="none",
    )
    strategy = CalibratedStrategy([zone], track.num_segments)
    state = SimState(
        time=0.0, distance=10.0, speed=15.0, soc=0.95,
        pack_voltage=400.0, pack_current=0.0, cell_temp=30.0,
        lap=0, segment_idx=5,
    )
    cmd = strategy.decide(state, [track.segments[i] for i in range(5, 10)])
    assert cmd.regen_request_pct == 0.0
