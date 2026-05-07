"""m_effective is a kinematic quantity; should not depend on drivetrain efficiency.

Regression tests for issue M13: η was incorrectly folded into the rotor-inertia
term. KE is a state function of speed alone (Genta §5.2; Krause/Wasynczuk/
Sudhoff §3.5); a dissipation parameter cannot appear in the equivalent inertia.

The directional asymmetry between accel and regen lives correctly in
``drive_force`` (× η motoring) and ``regen_force`` (× 1/η generating, S12 fix).
It does not need to be re-applied at the inertia level.
"""

from __future__ import annotations

import pytest

from fsae_sim.vehicle.dynamics import VehicleDynamics
from fsae_sim.vehicle.powertrain import PowertrainConfig
from fsae_sim.vehicle.vehicle import VehicleParams


def _make_dynamics(eta: float) -> VehicleDynamics:
    """Construct a VehicleDynamics with a given drivetrain efficiency.

    All other parameters are held fixed at CT-16EV-like values.
    """
    vehicle = VehicleParams(
        mass_kg=288.0,
        frontal_area_m2=1.0,
        drag_coefficient=1.502,
        rolling_resistance=0.015,
        wheelbase_m=1.549,
        downforce_coefficient=2.18,
        rotor_inertia_kg_m2=0.06,
        wheel_inertia_kg_m2=0.3,
    )
    pt = PowertrainConfig(
        motor_speed_max_rpm=2900.0,
        brake_speed_rpm=2400.0,
        torque_limit_inverter_nm=85.0,
        torque_limit_lvcu_nm=150.0,
        iq_limit_a=170.0,
        id_limit_a=30.0,
        gear_ratio=3.6363,
        drivetrain_efficiency=eta,
        rolling_radius_m=0.2042,
    )
    return VehicleDynamics(vehicle=vehicle, powertrain_config=pt)


def test_m_effective_independent_of_drivetrain_efficiency() -> None:
    """KE-derived m_eff cannot depend on a dissipation parameter.

    Drivetrain efficiency is a power-flow / force concept; folding it into
    rotational inertia would make m_eff direction-dependent and violate
    KE = (1/2) m_eff v² at constant speed.
    """
    m_eff_high = _make_dynamics(eta=0.95).m_effective
    m_eff_low = _make_dynamics(eta=0.85).m_effective
    assert m_eff_high == pytest.approx(m_eff_low, rel=1e-12)


def test_m_effective_matches_textbook_formula() -> None:
    """m_eff = m + (J_m·G² + 4·J_w) / r² — the Lagrangian/KE result.

    Reference: Genta, Motor Vehicle Dynamics §5.2; TUMFTM laptime-simulation
    `__compute_m_eq`.
    """
    eta = 0.92
    dyn = _make_dynamics(eta=eta)
    m, J_m, J_w, G, r = 288.0, 0.06, 0.3, 3.6363, 0.2042
    expected = m + (J_m * G * G + 4.0 * J_w) / (r * r)
    assert dyn.m_effective == pytest.approx(expected, rel=1e-12)
