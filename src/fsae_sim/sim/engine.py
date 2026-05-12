"""Quasi-static endurance simulation engine.

Orchestrates the per-segment simulation loop, integrating driver strategy,
powertrain model, vehicle dynamics, and battery model into a time-stepped
endurance simulation.
"""

from __future__ import annotations

import math
import time as _time
import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# BMS lap-refresh threshold (A). At lap boundaries, if the
# (temp, soc)-derived BMS discharge limit has drifted more than this
# from the limit the envelope was last built with, the speed envelope
# is recomputed. Default 5 A is below the BMS LUT bin width (10 A in
# Endurance Tune2) so we catch every cross-bin transition while
# preserving stability against per-segment noise. See plan
# docs/superpowers/plans/2026-05-06-battery-upgrades.md (BMS lap-refresh).
BMS_REFRESH_DELTA_A: float = 5.0

from fsae_sim.driver.strategy import ControlAction, DriverStrategy, SimState
from fsae_sim.driver.strategies import CalibratedStrategy, ReplayStrategy
from fsae_sim.track.track import Track
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain_model import PowertrainModel

try:
    from fsae_sim.vehicle.motor_efficiency import MotorEfficiencyMap
    _HAS_MOTOR_MAP = True
except ImportError:
    _HAS_MOTOR_MAP = False

try:
    from fsae_sim.vehicle.inverter_delivery import InverterDeliveryMap
    _HAS_INVERTER_DELIVERY_MAP = True
except ImportError:
    _HAS_INVERTER_DELIVERY_MAP = False
from pathlib import Path

from fsae_sim.sim.speed_envelope import SpeedEnvelope
from fsae_sim.physics_constants import GRAVITY_M_S2

try:
    from fsae_sim.vehicle.tire_model import PacejkaTireModel
    from fsae_sim.vehicle.load_transfer import LoadTransferModel
    from fsae_sim.vehicle.cornering_solver import CorneringSolver
    _HAS_TIRE_MODELS = True
except ImportError:
    _HAS_TIRE_MODELS = False


@dataclass
class SimResult:
    """Output of a single simulation run."""

    config_name: str
    strategy_name: str
    track_name: str
    states: pd.DataFrame  # time series of per-segment state snapshots
    total_time_s: float
    total_energy_kwh: float
    final_soc: float
    laps_completed: int
    # C8: track discharge and regen separately so we can reconcile the
    # energy budget against telemetry.  ``total_energy_kwh`` remains the
    # discharge-only number for backwards compatibility.
    discharge_energy_kwh: float = 0.0
    regen_energy_kwh: float = 0.0
    net_energy_kwh: float = 0.0
    discharge_charge_ah: float = 0.0
    regen_charge_ah: float = 0.0
    net_charge_ah: float = 0.0
    # BMS lap-by-lap envelope refresh diagnostics. The envelope is
    # rebuilt whenever the live BMS discharge limit drifts more than
    # ``BMS_REFRESH_DELTA_A`` from the limit it was built with.
    # ``bms_limit_a_per_lap`` is the live limit at lap entry.
    bms_limit_a_per_lap: list[float] = field(default_factory=list)
    envelope_recomputes: int = 0
    envelope_recompute_ms_total: float = 0.0


class SimulationMode(str, Enum):
    """Controls whether telemetry-derived shortcuts are allowed."""

    REPLAY = "replay"
    CALIBRATION = "calibration"
    PREDICTION = "prediction"
    VALIDATION = "validation"

    @classmethod
    def coerce(cls, value: str | "SimulationMode") -> "SimulationMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            allowed = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown simulation mode {value!r}; expected {allowed}") from exc


class SimulationEngine:
    """Quasi-static endurance simulation.

    For each track segment, resolves speed from force balance and driver
    strategy, steps battery state, and records results.
    """

    # Minimum speed to avoid divide-by-zero in power calculations
    _MIN_SPEED_MS = 0.5

    def __init__(
        self,
        vehicle: VehicleConfig,
        track: Track,
        strategy: DriverStrategy,
        battery_model: BatteryModel,
        *,
        mode: str | SimulationMode = SimulationMode.CALIBRATION,
        allow_telemetry_track: bool = False,
        allow_empirical_grip: bool = False,
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.strategy = strategy
        self.battery_model = battery_model
        self.mode = SimulationMode.coerce(mode)

        if (
            self.mode in {SimulationMode.PREDICTION, SimulationMode.VALIDATION}
            and getattr(strategy, "uses_observed_speed_caps", False)
        ):
            raise ValueError(
                f"{self.mode.value} mode forbids telemetry-derived speed caps. "
                "Use strategy.without_observed_speed_caps() or construct "
                "the strategy with use_observed_speed_caps=False."
            )

        if self.mode == SimulationMode.PREDICTION:
            vehicle.require_predictive_ready(
                allow_empirical_grip=allow_empirical_grip,
            )
            if (
                getattr(track, "source", "unknown").startswith("telemetry")
                and not allow_telemetry_track
            ):
                raise ValueError(
                    "prediction mode requires an independent track model; "
                    f"track {track.name!r} was built from {track.source!r}."
                )
        # Termination temperature comes from the config's discharge_limits:
        # the hottest point where max_current_a == 0 (battery can no longer
        # deliver power).  Avoids the earlier hardcoded 65 C.
        zero_current_temps = [
            dl.temp_c for dl in vehicle.battery.discharge_limits
            if dl.max_current_a == 0.0
        ]
        self._termination_temp_c = (
            max(zero_current_temps) if zero_current_temps else 65.0
        )

        # Load motor efficiency map if available.
        # NF-3: resolve path relative to the package, not the caller's CWD,
        # so sims run from any directory (tests, webapp, notebooks).  If
        # the map is missing, fall back to the constant drivetrain_efficiency
        # — but warn once so silent accuracy loss is visible to callers.
        motor_map = None
        if _HAS_MOTOR_MAP:
            repo_root = Path(__file__).resolve().parents[3]
            motor_map_path = (
                repo_root
                / "Real-Car-Data-And-Stats"
                / "emrax228_hv_cc_motor_map_long.csv"
            )
            if motor_map_path.exists():
                motor_map = MotorEfficiencyMap(motor_map_path)
            else:
                warnings.warn(
                    f"MotorEfficiencyMap CSV not found at {motor_map_path}; "
                    "falling back to constant drivetrain_efficiency. Motor/"
                    "inverter efficiency will not track operating point.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "fsae_sim.vehicle.motor_efficiency not importable; falling "
                "back to constant drivetrain_efficiency.",
                UserWarning,
                stacklevel=2,
            )

        # Inverter torque-delivery map: translates LVCU command into the
        # actual shaft torque the Cascadia inverter delivers, so replay
        # and calibrated paths produce honest wheel forces. Falls back
        # to identity (command == delivered) when the map is missing.
        inverter_delivery_map = None
        if _HAS_INVERTER_DELIVERY_MAP:
            repo_root = Path(__file__).resolve().parents[3]
            delivery_map_path = (
                repo_root
                / "Real-Car-Data-And-Stats"
                / "inverter_delivery_map.csv"
            )
            if delivery_map_path.exists():
                inverter_delivery_map = InverterDeliveryMap.from_csv(
                    delivery_map_path,
                    inverter_torque_cap_nm=vehicle.powertrain.torque_limit_inverter_nm,
                )
            else:
                warnings.warn(
                    f"InverterDeliveryMap CSV not found at {delivery_map_path}; "
                    "the sim will treat LVCU commands as if the inverter "
                    "delivered them perfectly. Run "
                    "scripts/build_inverter_delivery_map.py to regenerate.",
                    UserWarning,
                    stacklevel=2,
                )

        tire_cfg = getattr(vehicle, "tire", None)
        susp_cfg = getattr(vehicle, "suspension", None)

        # Build the tire model before the PowertrainModel so the load-
        # dependent rolling radius can be routed through motor RPM and
        # wheel force (issue 18 partial closure).  See
        # `PowertrainModel.rolling_radius_for`.
        tire_model: "PacejkaTireModel | None" = None
        if _HAS_TIRE_MODELS and tire_cfg is not None and susp_cfg is not None:
            tire_model = PacejkaTireModel(tire_cfg.tir_file)
            if tire_cfg.grip_scale != 1.0:
                tire_model.apply_grip_scale(tire_cfg.grip_scale)

        self.powertrain = PowertrainModel(
            vehicle.powertrain,
            efficiency_map=motor_map,
            inverter_delivery_map=inverter_delivery_map,
            tire_model=tire_model,
        )

        if tire_model is not None and susp_cfg is not None:
            load_transfer = LoadTransferModel(
                vehicle.vehicle,
                susp_cfg,
                cg_height_m=vehicle.vehicle.cg_height_m,
                weight_dist_front=vehicle.vehicle.weight_distribution_front,
                downforce_dist_front=vehicle.vehicle.downforce_distribution_front,
            )
            cornering_solver = CorneringSolver(
                tire_model,
                load_transfer,
                vehicle.vehicle.mass_kg,
                math.radians(tire_cfg.static_camber_front_deg),
                math.radians(tire_cfg.static_camber_rear_deg),
                susp_cfg.roll_camber_front_deg_per_deg,
                susp_cfg.roll_camber_rear_deg_per_deg,
            )
            self.dynamics = VehicleDynamics(
                vehicle.vehicle, tire_model, load_transfer, cornering_solver,
                powertrain_config=vehicle.powertrain,
            )
        else:
            self.dynamics = VehicleDynamics(
                vehicle.vehicle, powertrain_config=vehicle.powertrain,
            )

        self._envelope = SpeedEnvelope(self.dynamics, self.powertrain, self.track)

    def run(
        self,
        num_laps: int = 1,
        initial_soc_pct: float = 95.0,
        initial_temp_c: float = 25.0,
        initial_speed_ms: float = 0.0,
        rolling_start: bool = True,
    ) -> SimResult:
        """Run the endurance simulation.

        Args:
            num_laps: Number of laps to simulate.
            initial_soc_pct: Starting state-of-charge (percent).
            initial_temp_c: Starting cell temperature (Celsius).
            initial_speed_ms: Starting vehicle speed (m/s).
            rolling_start: D-26. When True (default), ``initial_speed_ms``
                is clamped up to ``_MIN_SPEED_MS`` to avoid divide-by-zero
                in the power calculation — matches the Michigan endurance
                rolling-start convention. If the clamp fires, emit a
                ``UserWarning`` so the silent bump is visible. When False,
                the initial speed is passed through as-is; the deep
                divide-by-zero guards inside resolve_exit_speed and the
                MIN_SPEED floor on exit_speed still keep the loop stable.

        Returns:
            SimResult with per-segment state history and summary metrics.
        """
        segments = self.track.segments
        num_segments = len(segments)
        lap_distance = self.track.total_distance_m

        # Mutable state
        time = 0.0
        distance = 0.0
        # D-26: apply rolling-start clamp only when requested.  When
        # applied and the caller's initial_speed was below the floor,
        # warn so the implicit bump is visible in logs.
        if rolling_start:
            if initial_speed_ms < self._MIN_SPEED_MS:
                warnings.warn(
                    f"rolling_start=True: clamping initial_speed_ms="
                    f"{initial_speed_ms:.3f} up to _MIN_SPEED_MS="
                    f"{self._MIN_SPEED_MS} to avoid divide-by-zero. Pass "
                    "rolling_start=False for a true standing start.",
                    UserWarning,
                    stacklevel=2,
                )
            speed = max(initial_speed_ms, self._MIN_SPEED_MS)
        else:
            speed = float(initial_speed_ms)
        soc = initial_soc_pct
        temp = initial_temp_c
        pack_voltage = self.battery_model.pack_voltage(soc, 0.0)

        # Accumulator for energy (C8: discharge and regen tracked separately)
        total_energy_j = 0.0
        discharge_energy_j = 0.0
        regen_energy_j = 0.0

        # D-11: carry previous segment's pack current into the next
        # segment's SimState so strategies and controllers that consult
        # state.pack_current see the actual last-segment value rather
        # than a perpetual 0.
        last_pack_current = 0.0

        # State log
        records: list[dict] = []
        laps_completed = 0

        # Pre-compute speed envelope (cornering + accel/brake feasibility).
        # Use the initial BMS limit so the forward pass is not more
        # optimistic than the runtime torque ceiling.
        initial_bms_limit = self.battery_model.max_discharge_current(temp, soc)
        v_max = self._envelope.compute(
            initial_speed=speed,
            bms_current_limit_a=initial_bms_limit,
        )
        # BMS lap-refresh diagnostics. ``bms_limit_a_per_lap`` records
        # the BMS-limit at each lap entry; ``envelope_recomputes`` and
        # ``envelope_recompute_ms_total`` track refresh frequency and
        # cost. The envelope is rebuilt at the top of a lap when the
        # live BMS limit has drifted more than BMS_REFRESH_DELTA_A
        # from the limit the envelope was last built with.
        bms_limit_a_per_lap: list[float] = []
        envelope_recomputes: int = 0
        envelope_recompute_ms_total: float = 0.0

        is_replay = isinstance(self.strategy, ReplayStrategy)
        is_calibrated = isinstance(self.strategy, CalibratedStrategy)

        # D-20: push the envelope into synthetic corner-braking strategies
        # so their lookahead uses the forward-backward-solved ceiling
        # rather than the isolated per-corner v_max.
        if hasattr(self.strategy, "set_envelope"):
            try:
                self.strategy.set_envelope(v_max)
            except Exception:
                pass

        lookahead = 5
        upcoming_by_segment = [
            [
                segments[(seg_idx + offset) % num_segments]
                for offset in range(lookahead)
            ]
            for seg_idx in range(num_segments)
        ]

        def solve_exit_speed(v0: float, length_m: float, net_force_n: float) -> float:
            if length_m <= 0.0:
                return max(v0, 0.0)
            a = net_force_n / self.dynamics.m_effective
            v_sq = v0 * v0 + 2.0 * a * length_m
            return math.sqrt(max(0.0, v_sq))

        def segment_time(v0: float, v1: float, length_m: float) -> float:
            avg = (v0 + v1) / 2.0
            return length_m / max(avg, 0.1)

        def commanded_motor_torque(
            op_speed_ms: float,
            command: ControlCommand,
            current_limit_a: float,
            seg_mid_distance_m: float,
        ) -> float:
            rpm = self.powertrain.motor_rpm_from_speed(op_speed_ms)
            if is_replay:
                torque = float(self.strategy.target_torque(seg_mid_distance_m))
                if getattr(self.strategy, "torque_is_delivered", False):
                    return torque
                return self.powertrain.apply_inverter_delivery(rpm, torque)
            # Non-replay: any positive throttle command produces motor
            # torque, regardless of the action label. The label is a
            # coaching descriptor; the pct fields are the source of
            # truth (matching how replay always reads both pedals).
            # This lets the calibrated lap-mean driver model emit
            # simultaneous throttle+brake — physically real (trail-
            # braking, mixed-action segments averaged across laps) and
            # not modellable when action gates the torque path.
            if command.throttle_pct > 0.0:
                if is_calibrated:
                    # The driver strategy calibrates to the firmware-
                    # remapped pedal position (AiM "Throttle Pos" =
                    # ``tmap_lut(tps_combined)``). Route through
                    # ``torque_lut(tps)`` directly — no second deadzone
                    # remap, no LVCU-side inverter cap (the inverter
                    # delivery map handles the 85 Nm clip downstream).
                    lvcu = self.powertrain.pedal_to_torque_request(
                        command.throttle_pct, rpm, current_limit_a,
                    )
                else:
                    lvcu = float(self.powertrain.lvcu_torque_command(
                        command.throttle_pct, rpm, current_limit_a,
                    ))
                return self.powertrain.apply_inverter_delivery(rpm, lvcu)
            return 0.0

        def command_forces(
            op_speed_ms: float,
            command: ControlCommand,
            current_limit_a: float,
            seg_mid_distance_m: float,
        ) -> tuple[float, float, float, float]:
            """Return motor torque, drive force, brake force, motor regen force."""
            torque = commanded_motor_torque(
                op_speed_ms, command, current_limit_a, seg_mid_distance_m,
            )
            drive_force = 0.0
            brake_force = 0.0
            regen_force = 0.0

            # Drive force: any positive motor torque produces drive force,
            # regardless of action label. Replay can also produce regen
            # via a negative torque command.
            if torque > 0.0:
                drive_force = self.powertrain.wheel_force(torque)
                drive_force = min(
                    drive_force,
                    self.dynamics.max_traction_force(op_speed_ms),
                )
            elif torque < 0.0 and is_replay:
                regen_force = self.powertrain.wheel_force(torque)

            # Brake force: any positive brake_pct produces brake force,
            # regardless of action label. Lets calibrated lap-mean driver
            # apply brake force on segments where some laps braked even
            # if the segment's majority action is throttle / coast.
            if command.brake_pct > 0.0:
                brake_force = self.dynamics.mechanical_brake_force(
                    command.brake_pct, op_speed_ms,
                )

            return torque, drive_force, brake_force, regen_force

        def enforce_speed_limit(
            entry_speed_ms: float,
            length_m: float,
            drive_force_n: float,
            brake_force_n: float,
            regen_force_n: float,
            resistance_force_n: float,
            speed_limit_ms: float,
        ) -> tuple[float, float, float, float, bool, bool]:
            """Apply a speed ceiling through force work, not speed deletion."""
            net = drive_force_n + regen_force_n - brake_force_n - resistance_force_n
            unconstrained_exit = solve_exit_speed(entry_speed_ms, length_m, net)
            if (
                not math.isfinite(speed_limit_ms)
                or unconstrained_exit <= speed_limit_ms
                or length_m <= 0.0
            ):
                return (
                    unconstrained_exit,
                    drive_force_n,
                    brake_force_n,
                    net,
                    False,
                    False,
                )

            target_exit = max(0.0, speed_limit_ms)
            required_net = (
                self.dynamics.m_effective
                * (target_exit * target_exit - entry_speed_ms * entry_speed_ms)
                / (2.0 * length_m)
            )
            required_drive_minus_brake = (
                required_net + resistance_force_n - regen_force_n
            )

            limited = True
            violated = False
            if required_drive_minus_brake >= 0.0:
                drive_force_n = min(drive_force_n, required_drive_minus_brake)
                brake_force_n = 0.0
            else:
                drive_force_n = 0.0
                needed_brake = -required_drive_minus_brake
                max_brake = self.dynamics.mechanical_brake_force(
                    1.0, max(entry_speed_ms, target_exit),
                )
                brake_force_n = min(needed_brake, max_brake)
                violated = needed_brake > max_brake + 1e-6

            net = drive_force_n + regen_force_n - brake_force_n - resistance_force_n
            exit_speed_ms = solve_exit_speed(entry_speed_ms, length_m, net)
            if not violated:
                exit_speed_ms = target_exit
            return (
                exit_speed_ms,
                drive_force_n,
                brake_force_n,
                net,
                limited,
                violated,
            )

        # Stitched per-lap tracks tag each Segment with its source lap
        # index. When that's the case, the outer ``for lap in
        # range(num_laps)`` loop lies — the segments themselves know
        # which lap they belong to, and the loop counter would just
        # repeat the same stitched track ``num_laps`` times. Detect this
        # by checking the first segment, then read ``seg.lap_index`` for
        # the recorded ``lap`` field below.
        segments_carry_lap_index = (
            len(segments) > 0
            and any(getattr(s, "lap_index", -1) >= 0 for s in segments)
        )

        # Helper for lap-boundary BMS refresh. Called at each true
        # lap transition (outer loop for non-stitched tracks; segment
        # lap_index transitions for stitched tracks). Records the
        # live BMS limit and recomputes the envelope iff the limit
        # has drifted >= BMS_REFRESH_DELTA_A.
        def _bms_lap_refresh() -> None:
            nonlocal v_max, envelope_recomputes, envelope_recompute_ms_total
            bms_limit_now = self.battery_model.max_discharge_current(temp, soc)
            bms_limit_a_per_lap.append(float(bms_limit_now))
            last_built = self._envelope.last_built_bms_limit_a
            if (
                last_built is not None
                and abs(bms_limit_now - last_built) >= BMS_REFRESH_DELTA_A
            ):
                _t0 = _time.perf_counter()
                v_max = self._envelope.compute(
                    initial_speed=speed,
                    bms_current_limit_a=bms_limit_now,
                )
                envelope_recompute_ms_total += (
                    (_time.perf_counter() - _t0) * 1000.0
                )
                envelope_recomputes += 1
                # Re-push to synthetic strategies if they consume the
                # envelope directly.
                if hasattr(self.strategy, "set_envelope"):
                    try:
                        self.strategy.set_envelope(v_max)
                    except Exception:
                        pass

        # Track the last seen segment lap_index so stitched tracks
        # can fire the BMS refresh at true lap boundaries (segment
        # lap_index transitions) rather than the outer-loop counter.
        last_seen_lap_index: int = -1

        for lap in range(num_laps):
            # Non-stitched tracks: outer-loop iteration is the lap.
            if not segments_carry_lap_index:
                _bms_lap_refresh()

            for seg_idx, segment in enumerate(segments):
                effective_lap = (
                    segment.lap_index if segments_carry_lap_index else lap
                )
                # Stitched tracks: refresh on lap_index transitions.
                if (
                    segments_carry_lap_index
                    and effective_lap >= 0
                    and effective_lap != last_seen_lap_index
                ):
                    _bms_lap_refresh()
                    last_seen_lap_index = effective_lap
                # Build SimState for driver strategy
                sim_state = SimState(
                    time=time,
                    distance=distance,
                    speed=speed,
                    soc=soc / 100.0,  # strategy expects 0-1
                    pack_voltage=pack_voltage,
                    pack_current=last_pack_current,  # D-11
                    cell_temp=temp,
                    lap=effective_lap,
                    segment_idx=seg_idx,
                )

                # 1. Driver decision
                cmd = self.strategy.decide(
                    sim_state, upcoming_by_segment[seg_idx],
                )

                # --- Force-based resolution (all strategies) ---

                # 2. Speed limit from pre-computed envelope
                speed_limit = float(v_max[seg_idx])

                # 2a. Honor zone-level ``max_speed_ms`` from the driver's
                # ControlCommand.metadata. CalibratedStrategy can carry a
                # per-zone cap that the envelope doesn't see (e.g. driver-
                # discipline cap below the tire-limit envelope). Take the
                # min so whichever constraint is tighter wins.
                if cmd.metadata is not None:
                    meta_cap = cmd.metadata.get("max_speed_ms")
                    if meta_cap is not None:
                        speed_limit = min(speed_limit, float(meta_cap))

                # 2b. Entry-speed clamp.  If the previous segment exited
                # above this segment's envelope (driver over-accelerated
                # relative to the envelope's forward-pass assumptions),
                # physically the tires cannot sustain a corner at that
                # speed — clamp down.  Without this, resolve_exit_speed
                # clamps the *exit* but average = (entry+exit)/2 can
                # exceed the corner limit.
                # BMS current limit for LVCU torque command
                bms_current_limit = self.battery_model.max_discharge_current(temp, soc)

                seg_mid_dist = distance + segment.length_m / 2.0

                def evaluate_forces_at(
                    op_speed_ms: float,
                ) -> tuple[float, float, float, float, float]:
                    """Evaluate forces at one operating speed (no cap).

                    Used by both Heun stages. Returns
                    ``(motor_torque, drive_f, brake_f, regen_f,
                    resist_f)``. The speed cap is *not* enforced here;
                    that saturation is applied to the trapezoidal-
                    average output so energy bookkeeping is unbroken.
                    """
                    motor_torque_n, drive_n, brake_n, regen_n = command_forces(
                        op_speed_ms, cmd, bms_current_limit, seg_mid_dist,
                    )
                    resist_n = self.dynamics.total_resistance(
                        op_speed_ms, segment.grade, segment.curvature,
                    )
                    return (
                        motor_torque_n, drive_n, brake_n, regen_n, resist_n,
                    )

                # Heun's method (RK2 trapezoidal) on the segment.
                #
                # Stage 1 (predictor): evaluate forces at entry speed,
                # solve for the unconstrained predicted exit.
                # Stage 2 (corrector): evaluate forces at the predicted
                # exit speed, average with stage-1 forces, solve again.
                # Final: apply ``enforce_speed_limit`` once on the
                # corrected output to saturate at the envelope cap
                # without distorting the segment's energy bookkeeping.
                #
                # Reference: Hairer/Norsett/Wanner, Solving ODEs I
                # (1993) §II.1; Plan
                # docs/superpowers/plans/2026-05-06-engine-numerics-and-powertrain-losses.md.

                # --- Stage 1: predictor at entry speed ---
                (
                    motor_torque_p,
                    drive_f_p,
                    brake_f_p,
                    regen_f_p,
                    resist_f_p,
                ) = evaluate_forces_at(speed)
                net_p = drive_f_p + regen_f_p - brake_f_p - resist_f_p
                v_predict = solve_exit_speed(speed, segment.length_m, net_p)

                # --- Stage 2: corrector at predicted exit speed ---
                v_predict_clipped = max(v_predict, self._MIN_SPEED_MS)
                (
                    motor_torque_c,
                    drive_f_c,
                    brake_f_c,
                    regen_f_c,
                    resist_f_c,
                ) = evaluate_forces_at(v_predict_clipped)

                # --- Trapezoidal average of net forces ---
                drive_f = 0.5 * (drive_f_p + drive_f_c)
                brake_f = 0.5 * (brake_f_p + brake_f_c)
                regen_f = 0.5 * (regen_f_p + regen_f_c)
                resist_f = 0.5 * (resist_f_p + resist_f_c)
                motor_torque = 0.5 * (motor_torque_p + motor_torque_c)

                # --- Speed-cap saturation on the corrected output ---
                # If the unconstrained Heun exit exceeds the envelope,
                # ``enforce_speed_limit`` rebalances drive/brake to
                # land exactly at v_cap. Apply only once, at the end,
                # so the cap-induced work is bookkept in the segment
                # without a second force evaluation.
                (
                    exit_speed,
                    drive_f,
                    brake_f,
                    net_force,
                    speed_limited,
                    speed_limit_violation,
                ) = enforce_speed_limit(
                    speed,
                    segment.length_m,
                    drive_f,
                    brake_f,
                    regen_f,
                    resist_f,
                    speed_limit,
                )
                exit_speed = max(exit_speed, self._MIN_SPEED_MS)
                # Recompute net_force after MIN_SPEED clamp so downstream
                # state reporting is consistent.
                net_force = drive_f + regen_f - brake_f - resist_f

                avg_speed = (speed + exit_speed) / 2.0
                seg_time = segment_time(speed, exit_speed, segment.length_m)
                motor_rpm = self.powertrain.motor_rpm_from_speed(avg_speed)

                # Recompute torque at resolved avg speed for accurate
                # power/state reporting. Replay always follows the
                # recorded torque channel; non-replay strategies use
                # whichever throttle command they sent regardless of
                # action label, so simultaneous throttle+brake (the
                # lap-mean driver model) integrates correctly.
                if is_replay:
                    seg_mid_dist = distance + segment.length_m / 2.0
                    lvcu_command = self.strategy.target_torque(seg_mid_dist)
                    if getattr(self.strategy, "torque_is_delivered", False):
                        motor_torque = float(lvcu_command)
                    else:
                        motor_torque = self.powertrain.apply_inverter_delivery(
                            motor_rpm, lvcu_command,
                        )
                elif cmd.throttle_pct > 0.0:
                    if is_calibrated:
                        lvcu_command = self.powertrain.pedal_to_torque_request(
                            cmd.throttle_pct, motor_rpm, bms_current_limit,
                        )
                    else:
                        lvcu_command = self.powertrain.lvcu_torque_command(
                            cmd.throttle_pct, motor_rpm, bms_current_limit,
                        )
                    motor_torque = self.powertrain.apply_inverter_delivery(
                        motor_rpm, lvcu_command,
                    )
                else:
                    motor_torque = 0.0

                # Keep electrical power consistent with the force actually
                # delivered after traction and speed-limit resolution.
                if motor_torque > 0.0:
                    delivered_torque = (
                        self.powertrain.motor_torque_from_wheel_force(drive_f)
                    )
                    if delivered_torque < motor_torque:
                        motor_torque = max(0.0, delivered_torque)
                        drive_f = self.powertrain.wheel_force(motor_torque)

                net_force = drive_f + regen_f - brake_f - resist_f
                kinetic_energy_delta_j = (
                    0.5
                    * self.dynamics.m_effective
                    * (exit_speed * exit_speed - speed * speed)
                )
                mechanical_brake_energy_j = brake_f * segment.length_m

                # 7. Electrical power and pack current.
                #
                # D-13: when replay carries measured V × I power, use it
                # directly (ground truth at the battery terminals — CAN
                # Torque Feedback has ~10 % positive bias vs V×I energy).
                # D-17: otherwise dispatch to the back-EMF-aware electrical
                # model with pack_voltage so coast segments route through
                # the rectifier branch.
                if is_replay and getattr(self.strategy, "has_electrical_power", False):
                    seg_mid_dist = distance + segment.length_m / 2.0
                    elec_power = self.strategy.measured_electrical_power(seg_mid_dist)
                else:
                    elec_power = self.powertrain.electrical_power(
                        motor_torque, motor_rpm, pack_voltage,
                    )
                new_soc, new_temp, new_voltage, pack_current = (
                    self.battery_model.step_power(
                        elec_power, seg_time, soc, temp, time_s=time,
                        vehicle_speed_ms=avg_speed,
                    )
                )
                if abs(pack_current) > 1e-9:
                    terminal_voltage = elec_power / pack_current
                else:
                    terminal_voltage = self.battery_model.pack_voltage(
                        soc, 0.0, time_s=time,
                    )

                # BMS current limit is enforced upstream in
                # lvcu_torque_command for non-replay paths.  Replay
                # bypasses that (it uses measured V*I) — preserve the
                # telemetry ground truth rather than double-clamping,
                # but warn once if replay ever exceeds the sim's BMS
                # ceiling so a calibration mismatch is visible.
                if (
                    is_replay
                    and getattr(self.strategy, "has_electrical_power", False)
                    and pack_current > bms_current_limit + 1.0
                    and not getattr(self, "_replay_bms_warned", False)
                ):
                    warnings.warn(
                        f"Replay pack_current={pack_current:.1f} A exceeds "
                        f"sim BMS ceiling={bms_current_limit:.1f} A. Replay "
                        "preserves telemetry ground truth; investigate if "
                        "A/B calibrated-vs-replay comparisons diverge.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._replay_bms_warned = True

                # 10. Energy accounting.
                # C8: keep discharge-only (positive pack power) for legacy
                # metric ``total_energy_kwh`` but also accumulate regen and
                # net so the verification page can reconcile against
                # telemetry's pack V*I integral.
                segment_energy_j = elec_power * seg_time
                if segment_energy_j > 0:
                    total_energy_j += segment_energy_j
                    discharge_energy_j += segment_energy_j
                else:
                    regen_energy_j += -segment_energy_j

                longitudinal_g = (
                    net_force / (self.dynamics.m_effective * GRAVITY_M_S2)
                    if self.dynamics.m_effective > 0.0 else 0.0
                )
                lateral_g = avg_speed * avg_speed * segment.curvature / GRAVITY_M_S2
                if self.dynamics.load_transfer is not None:
                    fl_fz, fr_fz, rl_fz, rr_fz = self.dynamics.load_transfer.tire_loads(
                        avg_speed, lateral_g, longitudinal_g,
                    )
                else:
                    front = (
                        self.vehicle.vehicle.mass_kg
                        * GRAVITY_M_S2
                        * self.vehicle.vehicle.weight_distribution_front
                    )
                    rear = self.vehicle.vehicle.mass_kg * GRAVITY_M_S2 - front
                    fl_fz = fr_fz = front / 2.0
                    rl_fz = rr_fz = rear / 2.0

                total_fz = max(fl_fz + fr_fz + rl_fz + rr_fz, 1e-9)
                brake_fx = [
                    -brake_f * fl_fz / total_fz,
                    -brake_f * fr_fz / total_fz,
                    -brake_f * rl_fz / total_fz,
                    -brake_f * rr_fz / total_fz,
                ]
                fl_fx = brake_fx[0]
                fr_fx = brake_fx[1]
                rl_fx = brake_fx[2] + drive_f / 2.0 + regen_f / 2.0
                rr_fx = brake_fx[3] + drive_f / 2.0 + regen_f / 2.0
                total_lat_force = (
                    self.vehicle.vehicle.mass_kg
                    * avg_speed
                    * avg_speed
                    * segment.curvature
                )
                front_lat = (
                    total_lat_force
                    * self.vehicle.vehicle.weight_distribution_front
                )
                rear_lat = total_lat_force - front_lat

                # 11. Record state
                records.append({
                    "lap": effective_lap,
                    "segment_idx": seg_idx,
                    "time_s": time,
                    "distance_m": distance,
                    "speed_ms": avg_speed,
                    "speed_kmh": avg_speed * 3.6,
                    "soc_pct": soc,
                    "pack_voltage_v": terminal_voltage,
                    "pack_current_a": pack_current,
                    "cell_temp_c": temp,
                    "motor_rpm": motor_rpm,
                    "motor_torque_nm": motor_torque,
                    "electrical_power_w": elec_power,
                    "drive_force_n": drive_f,
                    "regen_force_n": regen_f,
                    "brake_force_n": brake_f,
                    "resistance_force_n": resist_f,
                    "net_force_n": net_force,
                    "kinetic_energy_delta_j": kinetic_energy_delta_j,
                    "mechanical_brake_energy_j": mechanical_brake_energy_j,
                    "entry_speed_ms": speed,
                    "exit_speed_ms": exit_speed,
                    "segment_time_s": seg_time,
                    "action": cmd.action.value,
                    "throttle_pct": cmd.throttle_pct,
                    "brake_pct": cmd.brake_pct,
                    "curvature": segment.curvature,
                    "corner_speed_limit_ms": speed_limit,
                    "speed_limit_active": speed_limited,
                    "speed_limit_violation": speed_limit_violation,
                    "grade": segment.grade,
                    "lateral_g": lateral_g,
                    "longitudinal_g": longitudinal_g,
                    "wheel_fl_fx_n": fl_fx,
                    "wheel_fr_fx_n": fr_fx,
                    "wheel_rl_fx_n": rl_fx,
                    "wheel_rr_fx_n": rr_fx,
                    "wheel_fl_fy_n": front_lat / 2.0,
                    "wheel_fr_fy_n": front_lat / 2.0,
                    "wheel_rl_fy_n": rear_lat / 2.0,
                    "wheel_rr_fy_n": rear_lat / 2.0,
                    "wheel_fl_fz_n": fl_fz,
                    "wheel_fr_fz_n": fr_fz,
                    "wheel_rl_fz_n": rl_fz,
                    "wheel_rr_fz_n": rr_fz,
                })

                # 12. Advance state
                time += seg_time
                distance += segment.length_m
                speed = exit_speed
                soc = new_soc
                temp = new_temp
                pack_voltage = new_voltage
                last_pack_current = pack_current  # D-11

                if segments_carry_lap_index:
                    if effective_lap >= 0:
                        laps_completed = max(laps_completed, effective_lap + 1)

                # Check termination conditions
                if soc <= self.vehicle.battery.discharged_soc_pct:
                    return self._build_result(
                        records, time, total_energy_j, soc, laps_completed,
                        discharge_energy_j, regen_energy_j,
                        bms_limit_a_per_lap=bms_limit_a_per_lap,
                        envelope_recomputes=envelope_recomputes,
                        envelope_recompute_ms_total=envelope_recompute_ms_total,
                    )
                if temp >= self._termination_temp_c:
                    return self._build_result(
                        records, time, total_energy_j, soc, laps_completed,
                        discharge_energy_j, regen_energy_j,
                        bms_limit_a_per_lap=bms_limit_a_per_lap,
                        envelope_recomputes=envelope_recomputes,
                        envelope_recompute_ms_total=envelope_recompute_ms_total,
                    )

            if segments_carry_lap_index:
                # Stitched-per-lap track: the outer loop walked through
                # all real laps in one pass, count them by unique tag.
                laps_completed = max(
                    laps_completed,
                    len({
                        s.lap_index for s in segments
                        if getattr(s, "lap_index", -1) >= 0
                    }),
                )
            else:
                laps_completed += 1

        return self._build_result(
            records, time, total_energy_j, soc, laps_completed,
            discharge_energy_j, regen_energy_j,
            bms_limit_a_per_lap=bms_limit_a_per_lap,
            envelope_recomputes=envelope_recomputes,
            envelope_recompute_ms_total=envelope_recompute_ms_total,
        )

    def _build_result(
        self,
        records: list[dict],
        total_time: float,
        total_energy_j: float,
        final_soc: float,
        laps_completed: int,
        discharge_energy_j: float = 0.0,
        regen_energy_j: float = 0.0,
        bms_limit_a_per_lap: list[float] | None = None,
        envelope_recomputes: int = 0,
        envelope_recompute_ms_total: float = 0.0,
    ) -> SimResult:
        states = pd.DataFrame(records)
        discharge_charge_ah = 0.0
        regen_charge_ah = 0.0
        if {"pack_current_a", "segment_time_s"}.issubset(states.columns):
            current = states["pack_current_a"].values
            dt = states["segment_time_s"].values
            discharge_charge_ah = float(
                np.sum(np.maximum(current, 0.0) * dt) / 3600.0
            )
            regen_charge_ah = float(
                np.sum(np.maximum(-current, 0.0) * dt) / 3600.0
            )
        return SimResult(
            config_name=self.vehicle.name,
            strategy_name=self.strategy.name,
            track_name=self.track.name,
            states=states,
            total_time_s=total_time,
            total_energy_kwh=(discharge_energy_j - regen_energy_j) / 3.6e6,
            final_soc=final_soc,
            laps_completed=laps_completed,
            discharge_energy_kwh=discharge_energy_j / 3.6e6,
            regen_energy_kwh=regen_energy_j / 3.6e6,
            net_energy_kwh=(discharge_energy_j - regen_energy_j) / 3.6e6,
            discharge_charge_ah=discharge_charge_ah,
            regen_charge_ah=regen_charge_ah,
            net_charge_ah=discharge_charge_ah - regen_charge_ah,
            bms_limit_a_per_lap=list(bms_limit_a_per_lap or []),
            envelope_recomputes=envelope_recomputes,
            envelope_recompute_ms_total=envelope_recompute_ms_total,
        )
