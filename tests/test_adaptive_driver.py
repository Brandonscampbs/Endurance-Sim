"""Analytical sanity tests for the adaptive envelope-following driver (Wave 4a).

These tests do not touch telemetry. They construct synthetic envelopes and
single-segment situations and assert the driver issues pedal commands that
make sense under the firmware-faithful LVCU + inverter-delivery + tire chain.

Telemetry-grounded regression (Michigan endurance) is Wave 4b's job.

Sub-task A scope: AdaptiveDriver core only. Strategy-wrapper plumbing is
covered in tests/test_adaptive_strategy.py (Sub-task B).
"""

from __future__ import annotations

import numpy as np
import pytest

from fsae_sim.driver.adaptive import AdaptiveDriver, AdaptiveDriverParams
from fsae_sim.driver.strategy import ControlAction, ControlCommand, SimState
from fsae_sim.track.track import Segment, Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain_model import PowertrainModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _flat_track(num_segments: int = 100, seg_len: float = 0.5) -> Track:
    """Flat synthetic track for envelope-tracking sanity tests."""
    segs = [
        Segment(
            index=i,
            distance_start_m=i * seg_len,
            length_m=seg_len,
            curvature=0.0,
            grade=0.0,
        )
        for i in range(num_segments)
    ]
    return Track(name="flat100", segments=segs, source="synthetic")


@pytest.fixture
def vehicle_config(ct16ev_config_path) -> VehicleConfig:
    return VehicleConfig.from_yaml(ct16ev_config_path)


@pytest.fixture
def models(vehicle_config):
    """Bare powertrain + dynamics models (no tire/inverter map) so the
    inverse solve is exercised under the same path the production engine
    uses in legacy mode. Wave 4b can extend to the full attached-map
    pipeline once telemetry regression is in scope."""
    pt = PowertrainModel(vehicle_config.powertrain)
    dyn = VehicleDynamics(
        vehicle_config.vehicle,
        powertrain_config=vehicle_config.powertrain,
    )
    return vehicle_config, pt, dyn


@pytest.fixture
def driver(models):
    _cfg, pt, dyn = models
    return AdaptiveDriver(dynamics=dyn, powertrain=pt)


@pytest.fixture
def flat_track() -> Track:
    return _flat_track()


def _state_at(speed_ms: float, seg_idx: int, distance_m: float | None = None) -> SimState:
    if distance_m is None:
        distance_m = seg_idx * 0.5
    return SimState(
        time=0.0,
        distance=float(distance_m),
        speed=float(speed_ms),
        soc=0.95,
        pack_voltage=400.0,
        pack_current=0.0,
        cell_temp=30.0,
        lap=0,
        segment_idx=seg_idx,
    )


# ---------------------------------------------------------------------------
# 1. Flat envelope, no resistance -> roughly steady-state (no force demand)
# ---------------------------------------------------------------------------


class _ZeroResistanceDynamics:
    """Hand-rolled dynamics double: no drag, no rolling, no cornering, no
    parasitic. Mirrors the real VehicleDynamics interface only for the
    surface the AdaptiveDriver touches."""

    def __init__(self, real_dynamics: VehicleDynamics) -> None:
        self._real = real_dynamics
        self.m_effective = real_dynamics.m_effective
        # Mirror vehicle attribute so the driver can read mass for brake
        # peak-force ceiling.
        self.vehicle = real_dynamics.vehicle
        # Expose the brake-decel cap so the driver matches what the
        # real engine would do when mechanical brakes are saturated.
        self._MAX_BRAKE_DECEL_G = real_dynamics._MAX_BRAKE_DECEL_G

    def total_resistance(self, speed_ms: float, *, grade: float = 0.0, curvature: float = 0.0) -> float:
        return 0.0

    def mechanical_brake_force(self, brake_pct: float, speed_ms: float) -> float:
        return self._real.mechanical_brake_force(brake_pct, speed_ms)

    def max_traction_force(self, speed_ms: float) -> float:
        # Defer to the real model (legacy = inf, real models with tire =
        # tire-grip ceiling).
        return self._real.max_traction_force(speed_ms)


def test_flat_envelope_no_resistance_commands_near_zero_throttle(models, flat_track):
    """With v_max(s) = v for all s and zero resistance, the driver should
    request ~0 throttle (any positive force would push v above v_max)."""
    _cfg, pt, _real_dyn = models
    dyn_zero = _ZeroResistanceDynamics(_real_dyn)
    driver = AdaptiveDriver(dynamics=dyn_zero, powertrain=pt)
    v_const = 20.0
    v_max = np.full(flat_track.num_segments, v_const, dtype=np.float64)
    driver.set_envelope(v_max)
    state = _state_at(v_const, seg_idx=20)
    upcoming = [flat_track.segments[i] for i in range(20, 25)]
    cmd = driver.decide(state, upcoming)
    # Tolerance per dispatch: |throttle| < 0.01.
    assert abs(cmd.throttle_pct) < 0.01, (
        f"Expected ~0 throttle on zero-resistance flat envelope, got "
        f"throttle_pct={cmd.throttle_pct:.4f}"
    )
    assert cmd.brake_pct == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Flat envelope WITH drag -> throttle balances drag (F_drive ~ F_resist)
# ---------------------------------------------------------------------------


def test_flat_envelope_with_drag_throttle_balances_drag(driver, flat_track, models):
    """With v_max = 30 m/s and the real CT-16EV resistance model, the
    driver must command non-zero throttle to maintain steady state. The
    resulting drive force should approximately equal the resistance."""
    _cfg, pt, dyn = models
    v_const = 30.0
    v_max = np.full(flat_track.num_segments, v_const, dtype=np.float64)
    driver.set_envelope(v_max)
    state = _state_at(v_const, seg_idx=50)
    upcoming = [flat_track.segments[i] for i in range(50, 55)]
    cmd = driver.decide(state, upcoming)
    assert cmd.brake_pct == pytest.approx(0.0, abs=1e-6)
    assert cmd.throttle_pct > 0.05, (
        f"Expected positive throttle to overcome drag at 30 m/s, got "
        f"throttle_pct={cmd.throttle_pct:.4f}"
    )

    # Force-balance check: the realized drive force should be close to
    # the total resistance at this speed (within 5 % of resistance).
    rpm = pt.motor_rpm_from_speed(v_const)
    lvcu = float(pt.lvcu_torque_command(cmd.throttle_pct, rpm, 200.0))
    delivered = pt.apply_inverter_delivery(rpm, lvcu)
    f_drive = pt.wheel_force(delivered)
    f_resist = dyn.total_resistance(v_const, grade=0.0, curvature=0.0)
    assert f_drive == pytest.approx(f_resist, rel=0.05), (
        f"Drive force {f_drive:.1f} N should balance resistance "
        f"{f_resist:.1f} N within 5%."
    )


# ---------------------------------------------------------------------------
# 3. Ramp-up envelope -> throttle > 0 (and roughly scales with demand)
# ---------------------------------------------------------------------------


def test_ramp_up_envelope_commands_positive_throttle(driver, flat_track):
    """v_max(s+ds) > v_max(s) -> throttle should be positive."""
    n = flat_track.num_segments
    v_max = np.linspace(5.0, 25.0, n, dtype=np.float64)
    driver.set_envelope(v_max)
    state = _state_at(float(v_max[10]), seg_idx=10)
    upcoming = [flat_track.segments[i] for i in range(10, 15)]
    cmd = driver.decide(state, upcoming)
    assert cmd.brake_pct == pytest.approx(0.0, abs=1e-6)
    assert cmd.throttle_pct > 0.0
    # Action label should describe drive.
    assert cmd.action in (ControlAction.THROTTLE, ControlAction.COAST)


def test_ramp_up_throttle_increases_with_demand(models):
    """A steeper acceleration demand should produce a meaningfully larger
    throttle command than a gentle one, both within the achievable region.

    Use longer segments (5 m) so a 0.2 m/s and a 1.0 m/s step both stay
    well within the powertrain's wheel-force envelope at this operating
    speed — otherwise both demands saturate at full pedal and the test
    cannot distinguish them.
    """
    cfg, pt, dyn = models
    drv = AdaptiveDriver(dynamics=dyn, powertrain=pt)
    # Build a longer-segment synthetic track for this test.
    long_track = _flat_track(num_segments=30, seg_len=5.0)
    n = long_track.num_segments
    # Both ramps are well within powertrain capability at ~15 m/s.
    v_max_gentle = np.full(n, 15.0)
    v_max_gentle[12] = 15.2  # gentle: a ~0.6 m/s^2
    v_max_steep = np.full(n, 15.0)
    v_max_steep[12] = 16.0  # steeper: a ~3.0 m/s^2

    drv.set_envelope(v_max_gentle)
    state = _state_at(15.0, seg_idx=11, distance_m=55.0)
    upcoming = [long_track.segments[i] for i in range(11, 16)]
    cmd_gentle = drv.decide(state, upcoming)

    drv.set_envelope(v_max_steep)
    cmd_steep = drv.decide(state, upcoming)

    assert cmd_gentle.throttle_pct > 0.0
    assert cmd_steep.throttle_pct > cmd_gentle.throttle_pct, (
        f"Steeper ramp ({cmd_steep.throttle_pct:.3f}) should demand more "
        f"throttle than gentle ramp ({cmd_gentle.throttle_pct:.3f})."
    )


# ---------------------------------------------------------------------------
# 4. Ramp-down envelope -> brake > 0, throttle = 0
# ---------------------------------------------------------------------------


def test_ramp_down_envelope_commands_brake_no_throttle(driver, flat_track):
    """v_max(s+ds) < v_max(s) -> brake (and/or regen) > 0; throttle = 0."""
    n = flat_track.num_segments
    v_max = np.linspace(30.0, 10.0, n, dtype=np.float64)
    driver.set_envelope(v_max)
    state = _state_at(float(v_max[40]), seg_idx=40)
    upcoming = [flat_track.segments[i] for i in range(40, 45)]
    cmd = driver.decide(state, upcoming)
    assert cmd.throttle_pct == pytest.approx(0.0, abs=1e-6)
    assert cmd.brake_pct > 0.0, (
        f"Ramp-down envelope must invoke brake/regen, got "
        f"brake_pct={cmd.brake_pct:.4f}"
    )
    assert cmd.action == ControlAction.BRAKE


# ---------------------------------------------------------------------------
# 5. Saturation: unphysical demand -> throttle saturates at 1.0
# ---------------------------------------------------------------------------


def test_saturation_unphysical_demand_does_not_crash(driver, flat_track):
    """When target_exit is unphysically far above current speed (cannot be
    reached in one segment), throttle should saturate at 1.0 and the
    driver must not raise."""
    n = flat_track.num_segments
    v_max = np.full(n, 5.0)
    # Demand a 50 m/s exit out of a 5 m/s entry in 0.5 m -> requires
    # several thousand m/s^2 of accel, impossible.
    v_max[51] = 50.0
    driver.set_envelope(v_max)
    state = _state_at(5.0, seg_idx=50)
    upcoming = [flat_track.segments[i] for i in range(50, 55)]
    cmd = driver.decide(state, upcoming)
    assert cmd.brake_pct == pytest.approx(0.0, abs=1e-6)
    assert cmd.throttle_pct == pytest.approx(1.0, abs=1e-6), (
        f"Expected saturated throttle (==1.0), got {cmd.throttle_pct:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Determinism: same state in, same command out
# ---------------------------------------------------------------------------


def test_decide_is_deterministic(driver, flat_track):
    """Calling decide() twice with identical state must yield identical
    commands. No RNG anywhere."""
    n = flat_track.num_segments
    v_max = np.linspace(10.0, 25.0, n, dtype=np.float64)
    driver.set_envelope(v_max)
    state = _state_at(15.0, seg_idx=30)
    upcoming = [flat_track.segments[i] for i in range(30, 35)]
    cmd_a = driver.decide(state, upcoming)
    cmd_b = driver.decide(state, upcoming)
    assert cmd_a.action == cmd_b.action
    assert cmd_a.throttle_pct == cmd_b.throttle_pct
    assert cmd_a.brake_pct == cmd_b.brake_pct


# ---------------------------------------------------------------------------
# 7. PREDICTION compatibility
# ---------------------------------------------------------------------------


def test_driver_does_not_use_observed_speed_caps(driver):
    """Adaptive driver lives entirely off the (analytical) envelope. It
    must report uses_observed_speed_caps == False so the engine accepts
    it in SimulationMode.PREDICTION."""
    assert driver.uses_observed_speed_caps is False


# ---------------------------------------------------------------------------
# 8. No telemetry required at construction or run time
# ---------------------------------------------------------------------------


def test_driver_does_not_require_telemetry(models, flat_track):
    """Construction is purely from physics models; decide() does not
    open any CSV / open any file. Smoke test: build, drive, no FS calls."""
    cfg, pt, dyn = models
    drv = AdaptiveDriver(dynamics=dyn, powertrain=pt)
    drv.set_envelope(np.full(flat_track.num_segments, 20.0))
    cmd = drv.decide(
        _state_at(20.0, seg_idx=5),
        [flat_track.segments[i] for i in range(5, 10)],
    )
    assert isinstance(cmd, ControlCommand)


# ---------------------------------------------------------------------------
# 9. AdaptiveDriverParams sanity (structure, defaults)
# ---------------------------------------------------------------------------


def test_adaptive_driver_params_defaults_are_pure_feedforward():
    """Wave 4a ships with PI gains disabled and energy shaper off. Wave
    4b will enable them via this same dataclass."""
    params = AdaptiveDriverParams()
    assert params.lookahead_segments == 5
    # Wave 4b fields must default to a disabled state.
    assert params.kp_velocity == 0.0
    assert params.ki_velocity == 0.0
    assert params.i_clamp_mps == 0.0
    assert params.energy_budget_kwh is None
    assert params.energy_shaper_strategy == "FCFB"
