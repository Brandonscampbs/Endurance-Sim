"""Adaptive envelope-following driver (Wave 4a — feedforward core).

This module implements the pure feedforward "envelope tracker" half of the
adaptive driver described in
``docs/superpowers/plans/2026-05-06-driver-model-adaptive.md``. Given a
pre-computed forward-backward speed envelope ``v_max(s)`` and the current
``(v, s, state)``, ``AdaptiveDriver.decide()`` returns the throttle / brake
pedal commands whose realized force balance under the firmware-faithful
LVCU + inverter-delivery + tire chain makes the next-segment exit speed
match the envelope.

**Wave 4a scope (this file):**

- :class:`AdaptiveDriverParams` — frozen-dataclass parameter bundle. Wave
  4a uses only ``lookahead_segments``; PI gains and energy-shaper fields
  are structural placeholders for Wave 4b and default to disabled values.
- :class:`AdaptiveDriver` — the feedforward core. Deterministic (no RNG)
  and ``uses_observed_speed_caps == False`` so it is compatible with
  :class:`fsae_sim.sim.engine.SimulationMode.PREDICTION`. Does not load
  any telemetry.

**Out of scope (Wave 4b):** PI velocity-error corrector, LBP/LS energy
shaper, end-to-end telemetry regression, and the engine swap-in. See the
plan for context.

Architecture (single principle): the driver does NOT plan a trajectory.
The pre-computed ``SpeedEnvelope.compute()`` IS the optimal velocity
profile under the current vehicle physics — the driver tracks it.

Inverse mapping (the core math):
1. Compute target acceleration from kinematics:
   ``a_target = (v_target² − v_entry²) / (2·ds)``.
2. Compute required tractive force from longitudinal force balance:
   ``F_required = m_eff·a_target + F_resist(v, grade, curvature)``.
3. If ``F_required > 0``: solve for the throttle pedal that produces
   ``F_required`` through the powertrain chain. Use bisection
   (``scipy.optimize.brentq``) because ``apply_inverter_delivery`` is
   piecewise-monotonic and not always closed-form invertible.
4. If ``F_required < 0``: regen-first per plan A1, then mechanical
   brakes for the remainder.
5. Saturated demands return the boundary value (throttle = 1 or brake =
   1); the engine integrator will resolve the shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from fsae_sim.driver.energy_shaper import EnergyShaper
from fsae_sim.driver.strategy import (
    ControlAction,
    ControlCommand,
    DriverStrategy,
    SimState,
)
from fsae_sim.track.track import Segment

if TYPE_CHECKING:
    from fsae_sim.vehicle.dynamics import VehicleDynamics
    from fsae_sim.vehicle.powertrain_model import PowertrainModel


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveDriverParams:
    """Configuration bundle for :class:`AdaptiveDriver`.

    Wave 4a uses only ``lookahead_segments`` and the BMS reference
    ``default_bms_current_limit_a``. The remaining fields are reserved
    for Wave 4b and default to a disabled (pure feedforward, no shaping)
    configuration.

    Attributes:
        lookahead_segments: How many segments ahead the driver consults
            when picking the next-segment target exit speed. Wave 4a
            uses the first upcoming segment only (so a value of 1 is
            sufficient); the default of 5 matches the engine's existing
            pre-computed lookahead window.
        default_bms_current_limit_a: Pack discharge current limit (A)
            used when the engine has not pushed a live limit through
            :meth:`AdaptiveDriver.set_bms_limit`. 200 A is the
            ``EnduranceTune2.txt`` 30 °C ceiling on CT-16EV and is
            therefore the right "no information yet" default.
        kp_velocity: Wave 4b — proportional gain on (v − v_max). Wave
            4a leaves this at 0.0 so the driver is pure feedforward.
        ki_velocity: Wave 4b — integral gain on accumulated velocity
            error. Disabled (0.0) in Wave 4a.
        i_clamp_mps: Wave 4b — anti-windup clamp on the velocity-error
            integral (m/s). Disabled (0.0) in Wave 4a.
        energy_budget_kwh: Wave 4b — stint-level energy budget (kWh).
            When ``None``, the energy shaper is off (FCFB = "full
            capability, full boost"). Disabled in Wave 4a.
        energy_shaper_strategy: Wave 4b — allocator name. One of
            ``"FCFB"`` (no shaping; default), ``"LBP"`` (longest time
            to next brake point — TUMFTM), or ``"LS"`` (lowest speed).
    """

    # Sub-task D: lookahead bumped from 5 -> 60 (~30 m at the default
    # 0.5 m segment grid; ~1 s of preview at 30 m/s). The plan's
    # Lookahead section motivates 60 for envelope-following: the
    # driver can pre-lift and trail-brake across an entire corner-
    # entry sequence rather than reacting one segment at a time.
    lookahead_segments: int = 60
    default_bms_current_limit_a: float = 200.0

    # Wave 4b Sub-task A: PI velocity-error corrector. Plan A3 defaults.
    # The corrector closes the loop on small QSS modeling errors that
    # would otherwise drift across a stint. Gains derived analytically
    # in the plan; kept conservative so they cannot destabilize the
    # forward integration when the segment grid changes.
    kp_velocity: float = 0.05  # proportional gain (1/s)
    ki_velocity: float = 0.005  # integral gain (1/s^2)
    i_clamp_mps: float = 0.5  # anti-windup clamp on the integral (m/s)

    # Wave 4b Sub-task B: energy-budget shaper. Defaults to None which
    # means "no shaping" (FCFB pass-through). Provide an EnergyShaper
    # instance to engage attenuation behavior. The historical placeholder
    # fields ``energy_budget_kwh`` and ``energy_shaper_strategy`` are
    # retained as construction helpers for callers that want defaults
    # without instantiating an EnergyShaper directly; if both are set
    # we still consume ``energy_shaper`` (explicit beats implicit).
    energy_shaper: EnergyShaper | None = None  # Wave 4b
    energy_budget_kwh: float | None = None  # legacy / helper for ad-hoc setups
    energy_shaper_strategy: str = "FCFB"  # legacy / helper


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class AdaptiveDriver:
    """Pure feedforward envelope follower (Wave 4a).

    The driver consumes a pre-computed speed envelope ``v_max(s)`` and
    returns, for each segment, the throttle / brake pedal commands whose
    realized force balance puts the next-segment exit speed on the
    envelope. The mapping inverts the firmware-faithful LVCU + inverter-
    delivery + tire chain (``lvcu_torque_command`` -> ``apply_inverter_
    delivery`` -> ``wheel_force``).

    Determinism: the only state carried between calls is the envelope
    array and the BMS reference; no RNG, no time-varying internal state
    in Wave 4a.

    PREDICTION-mode compatibility: :attr:`uses_observed_speed_caps`
    returns ``False`` because the envelope is the only speed reference
    and is built analytically from the vehicle physics.
    """

    name: str = "adaptive-feedforward"

    def __init__(
        self,
        dynamics: "VehicleDynamics",
        powertrain: "PowertrainModel",
        params: AdaptiveDriverParams | None = None,
    ) -> None:
        self._dyn = dynamics
        self._pt = powertrain
        self._params = params if params is not None else AdaptiveDriverParams()
        self._envelope: np.ndarray | None = None
        self._bms_current_limit_a: float = self._params.default_bms_current_limit_a
        # PI velocity-error integrator (Sub-task A).
        self._velocity_error_integral_mps_s: float = 0.0
        # Energy-shaper state (Sub-task B). The engine pushes the running
        # net-kWh / lap-index / lap-progress through ``set_energy_state``
        # before each ``decide()``; absent that, the shaper sees a
        # zero-spend state and returns ``v_max`` unchanged, so the
        # default behavior is bit-identical to the Wave 4a feedforward.
        self._energy_used_kwh: float = 0.0
        self._lap_index: int = 0
        self._segment_progress: float = 0.0
        # Engine wires set_envelope() before reset() in normal flow, so
        # the integrator naturally starts at zero. Tests can call reset()
        # explicitly to guard against stale state when re-driving the
        # same driver instance.

    # ------------------------------------------------------------------
    # Configuration hooks (driven by the engine; see Wave 4b)
    # ------------------------------------------------------------------

    @property
    def params(self) -> AdaptiveDriverParams:
        return self._params

    @property
    def uses_observed_speed_caps(self) -> bool:
        """Always ``False`` — the envelope is the only speed reference."""
        return False

    def set_envelope(self, v_max: np.ndarray) -> None:
        """Install the pre-computed speed envelope (m/s per segment).

        Does NOT reset PI integrator state — that is the caller's
        responsibility, because the engine may push a refreshed envelope
        mid-stint (BMS lap-refresh) and we do not want to discard the
        accumulated integral every time the envelope is recomputed.
        Use :meth:`reset` at the *start* of a sim run instead.
        """
        self._envelope = np.asarray(v_max, dtype=np.float64).copy()

    def set_bms_limit(self, bms_current_limit_a: float) -> None:
        """Update the BMS reference used by the inverse pedal solve."""
        self._bms_current_limit_a = float(bms_current_limit_a)

    def set_energy_state(
        self,
        *,
        energy_used_kwh: float,
        lap_index: int,
        segment_progress: float,
    ) -> None:
        """Push current stint energy / lap state for the energy shaper.

        Called by the engine before each ``decide()`` when an energy
        shaper is attached so the shaper has the live spend reference.
        When no shaper is attached this is a cheap noop — the driver
        does not introspect the state.
        """
        self._energy_used_kwh = float(energy_used_kwh)
        self._lap_index = int(lap_index)
        self._segment_progress = float(segment_progress)

    # ------------------------------------------------------------------
    # PI velocity-error corrector (Sub-task A)
    # ------------------------------------------------------------------

    @property
    def velocity_error_integral_mps_s(self) -> float:
        """Current integral state in m/s.

        Exposed for tests and for instrumentation; callers should not
        mutate this directly — use :meth:`reset` to zero it.
        """
        return self._velocity_error_integral_mps_s

    def reset(self) -> None:
        """Reset internal PI integrator and energy-shaper state.

        Called by the engine at the start of each sim run so back-to-back
        runs with the same driver instance produce deterministic output
        (acceptance criterion AC4 in the plan).
        """
        self._velocity_error_integral_mps_s = 0.0
        self._energy_used_kwh = 0.0
        self._lap_index = 0
        self._segment_progress = 0.0

    def compute_pi_correction(
        self,
        *,
        v_measured_mps: float,
        v_target_mps: float,
        dt_s: float,
    ) -> float:
        """Return the corrective acceleration ``a_corr`` (m/s^2).

        Closed-loop correction: ``a_corr = kp * e + ki * integral`` where
        ``e = v_target - v_measured`` (positive = need to speed up).

        The proportional term is computed BEFORE the integral update so
        a single-call evaluation at integral=0 returns ``kp * e`` exactly
        (user spec: "no integral on first call since integral=0"). The
        integral is then accumulated and clamped to ``+/- i_clamp_mps``.

        Args:
            v_measured_mps: Current vehicle speed (m/s).
            v_target_mps: Envelope target speed at the current segment.
            dt_s: Time elapsed since the previous call (s). Used only
                to scale the integrator: ``integral += e * dt``.

        Returns:
            Corrective acceleration in m/s^2 to add to the feedforward
            target acceleration. Sign convention: positive = push the
            car faster, negative = pull it back below the envelope.
        """
        error_mps = float(v_target_mps) - float(v_measured_mps)
        # Proportional contribution at the entry-time integral state.
        a_corr = (
            self._params.kp_velocity * error_mps
            + self._params.ki_velocity * self._velocity_error_integral_mps_s
        )
        # Accumulate the integral with anti-windup clamp.
        self._velocity_error_integral_mps_s += error_mps * float(dt_s)
        clamp = self._params.i_clamp_mps
        if clamp > 0.0:
            if self._velocity_error_integral_mps_s > clamp:
                self._velocity_error_integral_mps_s = clamp
            elif self._velocity_error_integral_mps_s < -clamp:
                self._velocity_error_integral_mps_s = -clamp
        return float(a_corr)

    # ------------------------------------------------------------------
    # Decision (the inverse force-balance solve)
    # ------------------------------------------------------------------

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        """Return the throttle / brake command for the current segment.

        Algorithm:
            1. Read ``v_max`` at the next segment as the target exit speed.
            2. Compute the required net force from kinematics and the
               total resistance at the average operating speed.
            3. If positive (drive direction): solve for the pedal that
               produces this force via brentq inversion of the
               firmware-faithful pedal -> wheel-force chain.
            4. If negative (decel): regen first, friction for the
               remainder.
        """
        if self._envelope is None:
            raise RuntimeError(
                "AdaptiveDriver.decide() called before set_envelope(); "
                "the engine wires this automatically — call set_envelope() "
                "with the SpeedEnvelope output before driving."
            )
        if not upcoming:
            raise ValueError(
                "AdaptiveDriver.decide() requires at least one upcoming "
                "segment to read the segment length."
            )

        n = len(self._envelope)
        idx = state.segment_idx % n
        next_idx = (idx + 1) % n
        target_exit_ms = float(self._envelope[next_idx])
        seg = upcoming[0]
        seg_len = max(float(seg.length_m), 1e-6)
        v_entry = max(float(state.speed), 0.0)

        # Energy shaper attenuation (Sub-task B). Pure feedforward when
        # ``params.energy_shaper is None``; when present, attenuate the
        # next-segment target speed against the stint budget. The PI
        # corrector below sees the shaped target so it does not fight
        # the shaper.
        shaper = self._params.energy_shaper
        if shaper is not None:
            target_exit_ms = float(shaper.shape_target_speed(
                target_exit_ms,
                energy_used_kwh=self._energy_used_kwh,
                lap_index=self._lap_index,
                segment_progress=self._segment_progress,
            ))

        # Mid-segment operating speed mirrors the engine's predictor-
        # corrector midpoint — keeps the inverse and forward solves in
        # the same regime.
        v_op = max(0.5, (v_entry + target_exit_ms) / 2.0)

        # Kinematic acceleration demand and corresponding net force.
        a_target = (target_exit_ms * target_exit_ms - v_entry * v_entry) / (
            2.0 * seg_len
        )

        # PI velocity-error correction (Sub-task A). When the measured
        # speed drifts from the envelope target at the current segment,
        # add a corrective acceleration that the feedforward layer would
        # not otherwise produce. dt is approximated by the segment
        # transit time at v_op so the integrator scales with the actual
        # time spent at this state.
        v_target_now_ms = float(self._envelope[idx])
        dt_seg_s = seg_len / max(v_op, 0.5)
        a_corr = self.compute_pi_correction(
            v_measured_mps=v_entry,
            v_target_mps=v_target_now_ms,
            dt_s=dt_seg_s,
        )
        a_target += a_corr

        m_eff = float(self._dyn.m_effective)
        f_resist = float(
            self._dyn.total_resistance(
                v_op, grade=float(seg.grade), curvature=float(seg.curvature),
            )
        )
        f_net_required = m_eff * a_target
        # Drive minus brake (positive = need drive force, negative =
        # need decel beyond resistance).
        f_drive_minus_brake = f_net_required + f_resist

        if f_drive_minus_brake >= 0.0:
            return self._command_drive(f_drive_minus_brake, v_op, v_entry)
        return self._command_decel(-f_drive_minus_brake, v_op, v_entry)

    # ------------------------------------------------------------------
    # Drive direction: invert pedal -> wheel force
    # ------------------------------------------------------------------

    def _command_drive(
        self,
        f_required_n: float,
        v_op_ms: float,
        v_entry_ms: float,
    ) -> ControlCommand:
        """Return the throttle command (brake = 0) that realizes
        ``f_required_n`` Newtons of drive force at operating speed
        ``v_op_ms`` through the firmware-faithful LVCU chain.

        ``f_required_n`` is the net wheel-force demand (positive). The
        method clamps to whatever the chain can deliver at full pedal
        and at the tire's traction ceiling; saturated demands return
        ``throttle_pct == 1.0``.
        """
        # Step 1: how much drive force can the chain produce at full
        # pedal at this operating speed?
        rpm = self._pt.motor_rpm_from_speed(v_op_ms)
        f_full_pedal = self._wheel_force_at_pedal(1.0, rpm)
        # Step 2: tire traction ceiling.
        f_traction_cap = self._max_traction_force(v_op_ms)
        f_achievable = min(f_full_pedal, f_traction_cap)

        if f_full_pedal <= 1e-6:
            # Powertrain produces no force here (e.g. overspeed). Best
            # the driver can do is full pedal in case the chain returns
            # a tiny positive value next step.
            return ControlCommand(
                action=ControlAction.THROTTLE,
                throttle_pct=1.0,
                brake_pct=0.0,
            )

        if f_required_n >= f_achievable:
            # Saturated: full pedal.
            return ControlCommand(
                action=ControlAction.THROTTLE,
                throttle_pct=1.0,
                brake_pct=0.0,
            )

        if f_required_n <= 0.0:
            # Hand-off case: rounded down to zero demand.
            return ControlCommand(
                action=ControlAction.COAST,
                throttle_pct=0.0,
                brake_pct=0.0,
            )

        # Step 3: invert. wheel_force(pedal) is monotonic non-decreasing
        # in pedal at fixed rpm/BMS, so brentq lands the unique root.
        def residual(pedal: float) -> float:
            return self._wheel_force_at_pedal(pedal, rpm) - f_required_n

        # Guard the bracket: at pedal=0 the chain may still deliver a
        # tiny positive force if the LVCU power ceiling makes the
        # remapped-pedal-times-ceiling product non-zero at the deadzone
        # boundary. residual(0) should already be <= 0 for any
        # f_required_n > 0 because we just handled saturation above; if
        # the chain is non-monotonic at exactly pedal=0 (degenerate
        # config), fall back to the linear-ceiling estimate.
        r_lo = residual(0.0)
        r_hi = residual(1.0)
        if r_lo > 0.0:
            # Powertrain already exceeds demand at zero pedal. This
            # happens if f_required_n is below the chain's pedal=0
            # output (the LVCU's `pedal_remapped * ceiling` term is 0 at
            # pedal_raw <= deadzone_low, so this branch is rare; the
            # safe response is COAST).
            return ControlCommand(
                action=ControlAction.COAST,
                throttle_pct=0.0,
                brake_pct=0.0,
            )
        if r_hi < 0.0:
            # Sanity guard: we should have caught this in the saturation
            # check above, but if floating-point edge cases land us
            # here, return saturated pedal rather than crashing brentq.
            return ControlCommand(
                action=ControlAction.THROTTLE,
                throttle_pct=1.0,
                brake_pct=0.0,
            )

        pedal = float(brentq(residual, 0.0, 1.0, xtol=1e-5))
        pedal = max(0.0, min(1.0, pedal))
        action = (
            ControlAction.THROTTLE if pedal > 0.0 else ControlAction.COAST
        )
        return ControlCommand(
            action=action,
            throttle_pct=pedal,
            brake_pct=0.0,
        )

    # ------------------------------------------------------------------
    # Decel direction: regen-first, friction for the remainder
    # ------------------------------------------------------------------

    def _command_decel(
        self,
        f_decel_required_n: float,
        v_op_ms: float,
        v_entry_ms: float,
    ) -> ControlCommand:
        """Return the brake command (throttle = 0) that realizes
        ``f_decel_required_n`` Newtons of decel beyond resistance.

        Default brake-bias (plan A1): regen first, up to the motor
        envelope at this RPM; friction brakes for any remainder. Wave
        4b Sub-task C splits the channels at the strategy boundary —
        ``brake_pct`` carries the hydraulic-friction component (no
        electrical effect) and ``regen_request_pct`` carries the motor
        regen request (negative motor torque -> negative pack current).
        The engine routes each to its own physics in the adaptive
        dispatch branch.
        """
        # Regen ceiling at this rpm — driver can recover up to this
        # without touching mechanical brakes.
        rpm = self._pt.motor_rpm_from_speed(v_op_ms)
        regen_torque_max = self._pt.max_motor_torque(rpm)
        f_regen_max = self._pt.wheel_force(regen_torque_max)  # positive

        f_regen_used = min(f_decel_required_n, max(0.0, f_regen_max))
        f_friction_needed = max(0.0, f_decel_required_n - f_regen_used)

        # Mechanical brake capacity: the engine uses
        # ``mechanical_brake_force(brake_pct, v)``, which is linear in
        # brake_pct up to ``max_brake_decel_g * m * g`` (further clamped
        # by tire grip). For the inverse we use the same scaling. The
        # leading-underscore alias is preserved on VehicleDynamics for
        # back-compat, but new code reads the public attribute.
        peak_brake_force_n = (
            self._dyn.max_brake_decel_g
            * 9.81
            * float(self._dyn.vehicle.mass_kg)
        )
        # The friction system can also be tire-grip-limited; query that
        # ceiling so the saturation check matches engine behavior.
        friction_capacity_n = self._dyn.mechanical_brake_force(1.0, v_op_ms)
        if friction_capacity_n <= 0.0:
            friction_capacity_n = peak_brake_force_n

        if friction_capacity_n <= 1e-6:
            brake_pct_friction = 0.0
        else:
            brake_pct_friction = min(
                1.0, f_friction_needed / friction_capacity_n,
            )

        # Regen pedal fraction is reported separately on the
        # ``regen_request_pct`` channel; the engine routes it through
        # ``regen_force`` / the negative-torque path so battery
        # bookkeeping is correct.
        regen_pedal_fraction = (
            f_regen_used / max(f_regen_max, 1e-6) if f_regen_max > 0 else 0.0
        )
        brake_pct = max(0.0, min(1.0, brake_pct_friction))
        regen_request_pct = max(0.0, min(1.0, regen_pedal_fraction))

        return ControlCommand(
            action=ControlAction.BRAKE,
            throttle_pct=0.0,
            brake_pct=brake_pct,
            regen_request_pct=regen_request_pct,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wheel_force_at_pedal(self, pedal_pct: float, rpm: float) -> float:
        """Realized wheel force (N) for a given raw pedal position.

        Flow: ``pedal -> tmap_lut -> torque_lut -> inverter delivery ->
        wheel_force``. Mirrors the production engine's drive-force path
        for non-replay strategies.
        """
        lvcu = self._pt.lvcu_torque_command(
            float(pedal_pct), float(rpm), self._bms_current_limit_a,
        )
        # The bare return is a float (no return_state=True), but the
        # type checker sees a union — cast through float() to make the
        # intent explicit.
        lvcu_nm = float(lvcu)
        delivered = self._pt.apply_inverter_delivery(rpm, lvcu_nm)
        return float(self._pt.wheel_force(delivered))

    def _max_traction_force(self, v_op_ms: float) -> float:
        """Tire traction ceiling at this operating speed.

        Falls back to ``inf`` when ``max_traction_force`` itself returns
        ``inf`` (legacy dynamics without tire+load-transfer attached).
        """
        f = float(self._dyn.max_traction_force(v_op_ms))
        return f if np.isfinite(f) else float("inf")
