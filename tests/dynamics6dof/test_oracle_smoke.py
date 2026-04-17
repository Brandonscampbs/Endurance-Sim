import pytest


@pytest.mark.oracle
def test_oracle_returns_finite_ax(oracle):
    # State (13): rear_omega, vx, vy, wz, z, phi, mu, dz, dphi, dmu, t, n, chi
    q = [144.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u = [0.0, 0.3]
    ax = oracle.get_output(b"kart", q, u, 0.0, "chassis.acceleration.x")
    drag = oracle.get_output(b"kart", q, u, 0.0, "chassis.aerodynamics.drag")
    assert abs(ax) < 10.0  # finite, reasonable
    assert drag > 100.0 and drag < 300.0  # ~168 N observed in reconnaissance
