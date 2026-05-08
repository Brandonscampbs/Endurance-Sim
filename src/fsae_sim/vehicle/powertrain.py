"""Powertrain configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CoastLossConfig:
    """Physical decomposition of inverter+motor coast electrical-power loss.

    Coast = neither motoring nor commanded regen (``|motor_torque| <=
    COAST_THRESHOLD``). At any non-zero motor speed, the inverter is
    actively maintaining ``Iq=Id=0`` against motor cogging while the
    motor itself bleeds iron, windage, and bearing losses; the inverter
    bleeds gate-drive overhead and constant control-supply current. All
    of this shows up as positive (discharging) pack power.

    Each parameter maps to a named physical mechanism with documented
    bounds; calibration only fits 4 scalars, none of which is allowed
    outside its physics-derived range. Defaults are middle-of-bounds
    informed by datasheet typicals (Cascadia CM200DX, EMRAX 228 MV LC,
    Infineon FF600R12IS4F-style modules) so a sim with default coast
    config still returns a physically plausible coast power; calibrate
    against telemetry via :mod:`scripts.calibrate_coast_loss` for
    production accuracy.

    References:
        - Pyrhönen, Jokinen, Hrabovcová, *Design of Rotating Electrical
          Machines* 2nd ed. (Wiley 2014) §3.6 (iron loss),
          §3.7 (windage).
        - Krause, Wasynczuk, Sudhoff, *Analysis of Electric Machinery
          and Drive Systems* 3rd ed. (IEEE/Wiley 2013) §3.5, §6 (PMSM
          no-load decomposition).
        - Hanselman, *Brushless Permanent Magnet Motor Design* 2nd ed.
          (Magna Physics 2006) §10.5 (mechanical losses).
        - Infineon AN2008-03 "Switching losses of high-power IGBT
          modules" (gate-charge overhead).
        - Mohan, Undeland, Robbins, *Power Electronics: Converters,
          Applications, and Design* 3rd ed. (Wiley 2002) §27-2.
        - Cascadia CM200DX user manual rev D §5.4 (PWM/switching freq);
          spec rev 11 (control supply, DC link bleeder).
        - EMRAX 228 datasheet rev 2022 (no-load loss curve, peak windage).

    Attributes:
        p_aux_w: HV-side aux power: control supply + DC link bleeder +
            contactor coils. RPM- and voltage-independent first-order.
            Bounds [20, 80] W per Cascadia CM200DX spec.
        k_h_w_per_hz: Hysteresis-loss coefficient (W per electrical Hz).
            ``P_iron_h = k_h * f_e``. Bounds [0.05, 0.5] W/Hz per
            Pyrhönen §3.6 for steel laminations of EMRAX 228 size.
        k_e_w_per_hz2: Eddy-current loss coefficient (W per Hz²).
            ``P_iron_e = k_e * f_e^2``. Bounds [1e-5, 1e-3] W/Hz²
            per Pyrhönen §3.6.
        a_w_w_per_rad_s: Linear windage / bearing-drag coefficient
            (W per rad/s). ``P_w_lin = a_w * omega``. Bounds [0, 0.5]
            W/(rad/s); 0 is allowed (windage may be quadratic-only).
        b_w_w_per_rad_s2: Quadratic windage coefficient (W/(rad/s)²).
            ``P_w_quad = b_w * omega^2``. Bounds [0, 5e-4] W/(rad/s)².
        pwm_overhead_w: PWM gate-drive overhead at the calibration pack
            voltage. Multiplied by ``V_pack / pack_voltage_nominal_v``
            so model is voltage-aware. ``P_pwm = pwm_overhead_w *
            V_pack / V_nom``. Bounds [50, 600] W per Infineon AN2008-03
            for the CM200DX at 8 kHz.
        pole_pairs: PMSM pole-pair count, used to convert mechanical
            angular velocity to electrical frequency
            ``f_e = pole_pairs * omega_m / (2*pi)``. EMRAX 228 = 10
            pole pairs (datasheet rev 2022).
        pack_voltage_nominal_v: Reference pack voltage for PWM scaling.
            CT-16EV mean pack voltage over Michigan endurance ~390 V
            at 110S Molicel P45B near 75 % SOC.
    """

    p_aux_w: float = 40.0
    k_h_w_per_hz: float = 0.18
    k_e_w_per_hz2: float = 1.5e-4
    a_w_w_per_rad_s: float = 0.05
    b_w_w_per_rad_s2: float = 5e-5
    pwm_overhead_w: float = 350.0
    pole_pairs: int = 10
    pack_voltage_nominal_v: float = 390.0

    def __post_init__(self) -> None:
        if self.p_aux_w < 0.0:
            raise ValueError(f"p_aux_w must be >= 0, got {self.p_aux_w!r}")
        if self.k_h_w_per_hz < 0.0:
            raise ValueError(f"k_h_w_per_hz must be >= 0, got {self.k_h_w_per_hz!r}")
        if self.k_e_w_per_hz2 < 0.0:
            raise ValueError(f"k_e_w_per_hz2 must be >= 0, got {self.k_e_w_per_hz2!r}")
        if self.a_w_w_per_rad_s < 0.0:
            raise ValueError(
                f"a_w_w_per_rad_s must be >= 0, got {self.a_w_w_per_rad_s!r}"
            )
        if self.b_w_w_per_rad_s2 < 0.0:
            raise ValueError(
                f"b_w_w_per_rad_s2 must be >= 0, got {self.b_w_w_per_rad_s2!r}"
            )
        if self.pwm_overhead_w < 0.0:
            raise ValueError(
                f"pwm_overhead_w must be >= 0, got {self.pwm_overhead_w!r}"
            )
        if self.pole_pairs <= 0:
            raise ValueError(f"pole_pairs must be > 0, got {self.pole_pairs!r}")
        if self.pack_voltage_nominal_v <= 0.0:
            raise ValueError(
                f"pack_voltage_nominal_v must be > 0, got "
                f"{self.pack_voltage_nominal_v!r}"
            )


@dataclass(frozen=True)
class PowertrainConfig:
    """Motor, inverter, and drivetrain parameters."""
    motor_speed_max_rpm: float
    brake_speed_rpm: float
    torque_limit_inverter_nm: float
    # LVCU software torque limit. Firmware default is 2200//x10 = 220 Nm
    # (LVCU Code.txt:109). Older configs used 150 Nm; that was an
    # operational cap for Michigan 2025, not the firmware value. See
    # C17 in docs/SIMULATOR_ISSUES.md.
    torque_limit_lvcu_nm: float
    iq_limit_a: float
    id_limit_a: float
    gear_ratio: float
    drivetrain_efficiency: float
    # Rolling radius used for motor RPM, wheel force, and reflected inertia.
    # Keep this single value authoritative across the longitudinal model.
    rolling_radius_m: float = 0.2042
    # LVCU torque command parameters (from real LVCU Code.txt)
    lvcu_power_constant: float = 420.0        # 4200 in 0.1Nm CAN units / 10
    # Firmware fixed-point constant; intentionally offset from pi/30 ≈ 0.10472
    # (~2.7%).  Matches the LVCU's own rpm→omega conversion so the
    # power-ceiling math reproduces firmware behaviour.  Do NOT replace
    # with math.pi/30 — doing so would shift the constant-power torque
    # curve by ~2.7% relative to the real car.
    lvcu_rpm_scale: float = 0.1076            # RPM to angular velocity scale (firmware)
    lvcu_omega_floor: float = 23.04           # 230.4 in CAN units / 10
    lvcu_pedal_deadzone_low: float = 0.1      # tmap_lut V_MIN
    lvcu_pedal_deadzone_high: float = 0.9     # tmap_lut V_MAX
    lvcu_overspeed_rpm: float = 6000.0        # hard torque override threshold
    lvcu_overspeed_torque_nm: float = 30.0    # torque at overspeed (300/10)

    # S14: BMS current-limit safety offset used inside the LVCU
    # (`current_limit = (RxData[1] << 8 | RxData[0]) - 3` — LVCU Code.txt:151).
    lvcu_bms_current_offset_a: float = 3.0

    # Optional operational safety cap layered on top of the firmware LVCU
    # limit. Used to record the 150 Nm competition cap ("run conservatively")
    # without pretending the firmware imposed it. Leave as None to disable.
    safety_torque_cap_nm: Optional[float] = None

    # Motor back-EMF constant (V per rad/s of electrical frequency at the
    # motor terminals, treated here as V per rad/s of mechanical angular
    # velocity because pole pairs are folded into the calibration).
    # EMRAX 228 MV LC: ~0.045 V/(rad/s). Used by the C2 back-EMF
    # rectification term in `PowertrainModel.electrical_power`.
    motor_back_emf_constant_v_s_per_rad: float = 0.045

    # Coast electrical-power loss model. See ``CoastLossConfig`` for the
    # 4-term physical decomposition and bounds. Default
    # ``CoastLossConfig()`` returns ~physical mid-bound values; calibrate
    # via ``scripts/calibrate_coast_loss.py`` for production use.
    coast_loss: CoastLossConfig = field(default_factory=CoastLossConfig)

    def __post_init__(self) -> None:
        # NF-41: reject configs with a pedal-deadzone span that would cause
        # divide-by-zero or catastrophic noise amplification.
        span = self.lvcu_pedal_deadzone_high - self.lvcu_pedal_deadzone_low
        if span < 0.01:
            raise ValueError(
                f"lvcu_pedal_deadzone_high ({self.lvcu_pedal_deadzone_high}) "
                f"must exceed lvcu_pedal_deadzone_low "
                f"({self.lvcu_pedal_deadzone_low}) by at least 0.01; got "
                f"span={span:.6f}."
            )
        if self.rolling_radius_m <= 0.0:
            raise ValueError(
                f"rolling_radius_m must be positive, got {self.rolling_radius_m!r}"
            )
