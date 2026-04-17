"""Quasi-static endurance simulation engine.

Orchestrates the per-segment simulation loop, integrating driver strategy,
powertrain model, vehicle dynamics, and battery model into a time-stepped
endurance simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
from pathlib import Path

from fsae_sim.sim.speed_envelope import SpeedEnvelope

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
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.strategy = strategy
        self.battery_model = battery_model

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
                import warnings
                warnings.warn(
                    f"MotorEfficiencyMap CSV not found at {motor_map_path}; "
                    "falling back to constant drivetrain_efficiency. Motor/"
                    "inverter efficiency will not track operating point.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            import warnings
            warnings.warn(
                "fsae_sim.vehicle.motor_efficiency not importable; falling "
                "back to constant drivetrain_efficiency.",
                UserWarning,
                stacklevel=2,
            )

        self.powertrain = PowertrainModel(vehicle.powertrain, efficiency_map=motor_map)

        tire_cfg = getattr(vehicle, "tire", None)
        susp_cfg = getattr(vehicle, "suspension", None)

        if _HAS_TIRE_MODELS and tire_cfg is not None and susp_cfg is not None:
            tire_model = PacejkaTireModel(tire_cfg.tir_file)
            if tire_cfg.grip_scale != 1.0:
                tire_model.apply_grip_scale(tire_cfg.grip_scale)
            load_transfer = LoadTransferModel(vehicle.vehicle, susp_cfg)
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
                import warnings
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

        # Pre-compute speed envelope (cornering limits from tire grip)
        v_max = self._envelope.compute(initial_speed=speed)
        is_replay = isinstance(self.strategy, ReplayStrategy)
        is_calibrated = isinstance(self.strategy, CalibratedStrategy)

        for lap in range(num_laps):
            for seg_idx, segment in enumerate(segments):
                # Build SimState for driver strategy
                sim_state = SimState(
                    time=time,
                    distance=distance,
                    speed=speed,
                    soc=soc / 100.0,  # strategy expects 0-1
                    pack_voltage=pack_voltage,
                    pack_current=last_pack_current,  # D-11
                    cell_temp=temp,
                    lap=lap,
                    segment_idx=seg_idx,
                )

                # Look ahead: upcoming segments (current + next few)
                lookahead = 5
                upcoming = []
                for i in range(lookahead):
                    idx = (seg_idx + i) % num_segments
                    upcoming.append(segments[idx])

                # 1. Driver decision
                cmd = self.strategy.decide(sim_state, upcoming)

                # --- Force-based resolution (all strategies) ---

                # 2. Speed limit from pre-computed envelope
                corner_limit = float(v_max[seg_idx])

                # 2a. D-09: honor zone-level ``max_speed_ms`` from the
                # driver's ControlCommand.metadata.  CalibratedStrategy /
                # PedalProfileStrategy can carry a per-zone cap that the
                # envelope doesn't see (e.g. driver-discipline cap below
                # the tire-limit envelope).  Take the min so whichever
                # constraint is tighter wins.
                if cmd.metadata is not None:
                    meta_cap = cmd.metadata.get("max_speed_ms")
                    if meta_cap is not None:
                        corner_limit = min(corner_limit, float(meta_cap))

                # 2b. Entry-speed clamp.  If the previous segment exited
                # above this segment's envelope (driver over-accelerated
                # relative to the envelope's forward-pass assumptions),
                # physically the tires cannot sustain a corner at that
                # speed — clamp down.  Without this, resolve_exit_speed
                # clamps the *exit* but average = (entry+exit)/2 can
                # exceed the corner limit.
                if speed > corner_limit:
                    speed = corner_limit

                # 2c. BMS current limit for LVCU torque command
                bms_current_limit = self.battery_model.max_discharge_current(temp, soc)
                motor_rpm = self.powertrain.motor_rpm_from_speed(speed)

                # 3. Compute forces based on driver action
                #    ReplayStrategy: use recorded LVCU Torque Req directly
                #    (already the final inverter command, no re-processing).
                #    CalibratedStrategy: intensity is a torque fraction
                #    (LVCU Torque Req / 85 Nm), already through the dead
                #    zone remap. Use lvcu_torque_ceiling to apply power
                #    limiting without double-processing the dead zone.
                #    Other strategies: raw throttle through full LVCU model.
                if cmd.action == ControlAction.THROTTLE:
                    if is_replay:
                        # D-15: replay torque is the measured delivered
                        # torque — field-weakening is already baked in.
                        seg_mid_dist = distance + segment.length_m / 2.0
                        motor_torque = self.strategy.target_torque(seg_mid_dist)
                    elif is_calibrated:
                        ceiling = self.powertrain.lvcu_torque_ceiling(
                            motor_rpm, bms_current_limit,
                        )
                        motor_torque = cmd.throttle_pct * ceiling
                    else:
                        motor_torque = self.powertrain.lvcu_torque_command(
                            cmd.throttle_pct, motor_rpm, bms_current_limit,
                        )
                    drive_f = self.powertrain.wheel_force(motor_torque)
                    drive_f = min(drive_f, self.dynamics.max_traction_force(speed))
                    regen_f = 0.0
                elif cmd.action == ControlAction.BRAKE:
                    drive_f = 0.0
                    regen_f = self.powertrain.regen_force(cmd.brake_pct, speed)
                    max_brake = self.dynamics.max_braking_force(speed)
                    if abs(regen_f) > max_brake:
                        regen_f = -max_brake
                else:  # COAST
                    drive_f = 0.0
                    regen_f = 0.0

                # 4. Resistive forces
                resist_f = self.dynamics.total_resistance(speed, segment.grade, segment.curvature)

                # 5. Net force and speed resolution
                net_force = drive_f + regen_f - resist_f

                exit_speed, seg_time = self.dynamics.resolve_exit_speed(
                    speed, segment.length_m, net_force, corner_limit,
                )
                exit_speed = max(exit_speed, self._MIN_SPEED_MS)

                # D-09: zone-level speed cap.  If the driver strategy
                # attached ``max_speed_ms`` via metadata (observed peak
                # speed for the current zone), clamp exit speed to it
                # so the sim never carries more speed through a zone
                # than the real driver demonstrated.  Strategies without
                # metadata are unaffected (metadata defaults to None).
                if cmd.metadata is not None and "max_speed_ms" in cmd.metadata:
                    zone_cap = float(cmd.metadata["max_speed_ms"])
                    if zone_cap > 0.0:
                        exit_speed = min(exit_speed, zone_cap)

                avg_speed = (speed + exit_speed) / 2.0
                motor_rpm = self.powertrain.motor_rpm_from_speed(avg_speed)

                # Recompute torque at resolved avg speed for accurate power calc
                if cmd.action == ControlAction.THROTTLE:
                    if is_replay:
                        # D-15: replay torque is the measured delivered
                        # torque — field-weakening is already baked in.
                        seg_mid_dist = distance + segment.length_m / 2.0
                        motor_torque = self.strategy.target_torque(seg_mid_dist)
                    elif is_calibrated:
                        ceiling = self.powertrain.lvcu_torque_ceiling(
                            motor_rpm, bms_current_limit,
                        )
                        motor_torque = cmd.throttle_pct * ceiling
                    else:
                        motor_torque = self.powertrain.lvcu_torque_command(
                            cmd.throttle_pct, motor_rpm, bms_current_limit,
                        )
                elif cmd.action == ControlAction.BRAKE:
                    # NF-5: derive motor_torque from the tire-clipped regen_f
                    # rather than the raw brake command.  When the wheel can't
                    # support the requested regen force we must reflect the
                    # clipped force in the electrical power calculation too,
                    # otherwise the sim reports more regen energy than the
                    # tires actually transferred.
                    gear = self.powertrain.config.gear_ratio
                    eff = self.powertrain._GEARBOX_EFFICIENCY
                    tire_r = self.powertrain.TIRE_RADIUS_M
                    # regen_f is negative (braking); motor_torque is negative.
                    motor_torque = regen_f * tire_r / (gear * eff)
                else:
                    motor_torque = 0.0

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
                if pack_voltage > 0:
                    pack_current = elec_power / pack_voltage
                else:
                    pack_current = 0.0

                # BMS current limit is now enforced upstream via
                # lvcu_torque_command — no after-the-fact clamp needed.

                # 8. Step battery state
                new_soc, new_temp, new_voltage = self.battery_model.step(
                    pack_current, seg_time, soc, temp,
                )

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

                # 11. Record state
                records.append({
                    "lap": lap,
                    "segment_idx": seg_idx,
                    "time_s": time,
                    "distance_m": distance,
                    "speed_ms": avg_speed,
                    "speed_kmh": avg_speed * 3.6,
                    "soc_pct": soc,
                    "pack_voltage_v": pack_voltage,
                    "pack_current_a": pack_current,
                    "cell_temp_c": temp,
                    "motor_rpm": motor_rpm,
                    "motor_torque_nm": motor_torque,
                    "electrical_power_w": elec_power,
                    "drive_force_n": drive_f,
                    "regen_force_n": regen_f,
                    "resistance_force_n": resist_f,
                    "net_force_n": net_force,
                    "segment_time_s": seg_time,
                    "action": cmd.action.value,
                    "throttle_pct": cmd.throttle_pct,
                    "brake_pct": cmd.brake_pct,
                    "curvature": segment.curvature,
                    "corner_speed_limit_ms": corner_limit,
                    "grade": segment.grade,
                })

                # 12. Advance state
                time += seg_time
                distance += segment.length_m
                speed = exit_speed
                soc = new_soc
                temp = new_temp
                pack_voltage = new_voltage
                last_pack_current = pack_current  # D-11

                # Check termination conditions
                if soc <= self.vehicle.battery.discharged_soc_pct:
                    return self._build_result(
                        records, time, total_energy_j, soc, laps_completed,
                        discharge_energy_j, regen_energy_j,
                    )
                if temp >= self._termination_temp_c:
                    return self._build_result(
                        records, time, total_energy_j, soc, laps_completed,
                        discharge_energy_j, regen_energy_j,
                    )

            laps_completed += 1

        return self._build_result(
            records, time, total_energy_j, soc, laps_completed,
            discharge_energy_j, regen_energy_j,
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
    ) -> SimResult:
        states = pd.DataFrame(records)
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
        )
