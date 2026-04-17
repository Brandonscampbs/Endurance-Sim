# tests/dynamics6dof/test_aero.py
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fsae_sim.dynamics6dof.aero import aero_force
from fsae_sim.dynamics6dof.params import Dynamics6DofParams


def _read_kart_aero_params(oracle) -> tuple[float, float, float, float]:
    """Read ρ, Cd, Cl, area from the kart XML.

    The v0.5 fastest-lap Python module does not expose `vehicle_get_parameter`
    (only `vehicle_set_parameter`), so we read directly from the XML used by the
    oracle. This is still rigorous: the oracle was constructed from this exact
    file via `create_vehicle_from_xml` in the session fixture.
    """
    tree = ET.parse(str(oracle.kart_xml))
    root = tree.getroot()
    aero = root.find(".//chassis/aerodynamics")
    assert aero is not None, "kart XML missing chassis/aerodynamics block"
    rho = float(aero.findtext("rho").strip())
    cd = float(aero.findtext("cd").strip())
    cl = float(aero.findtext("cl").strip())
    area = float(aero.findtext("area").strip())
    return rho, cd, cl, area


# The kart XML defines ρ, Cd, A, Cl. We read them once to keep comparison
# rigorous, then match locally.
@pytest.mark.oracle
@pytest.mark.parametrize("vx,vy", [(5.0, 0.0), (20.0, 0.0), (30.0, 2.0)])
def test_aero_matches_oracle(oracle, vx, vy):
    # Zero-everything-but-velocity state
    q = [0.0, vx, vy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u = [0.0, 0.0]
    # The oracle exposes aerodynamic drag as 'chassis.aerodynamics.drag' (N, scalar).
    drag_oracle = oracle.get_output(b"kart", q, u, 0.0, "chassis.aerodynamics.drag")

    # Use the *kart* aero params so numbers match the oracle.
    rho, cd, cl, area = _read_kart_aero_params(oracle)

    params = Dynamics6DofParams.ct16ev_defaults()
    # Swap aero to match kart for apples-to-apples
    params = _replace_aero(params, rho=rho, cda=cd * area, cla=cl * area)

    wind = np.zeros(3)
    f_drag, f_lift = aero_force(np.array([vx, vy, 0.0]), wind, params)

    assert float(np.linalg.norm(f_drag)) == pytest.approx(abs(drag_oracle), rel=1e-6, abs=1e-6)


def _replace_aero(p: Dynamics6DofParams, rho: float, cda: float, cla: float) -> Dynamics6DofParams:
    from dataclasses import replace
    return replace(p, rho_air_kgpm3=rho, cd_a_m2=cda, cl_a_m2=cla)
