# tests/dynamics6dof/test_state_params.py
import numpy as np
import pytest

from fsae_sim.dynamics6dof.state import State6Dof
from fsae_sim.dynamics6dof.params import Dynamics6DofParams


def test_state_from_array_round_trip():
    q = np.array([144.0, 20.0, 0.1, 0.2, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    s = State6Dof.from_array(q)
    np.testing.assert_array_equal(s.to_array(), q)


def test_state_named_fields():
    s = State6Dof.initial(vx=20.0, rear_omega=100.0)
    assert s.vx == 20.0
    assert s.rear_omega == 100.0
    assert s.vy == s.wz == 0.0


def test_params_from_dicts():
    # Minimal construction to be fleshed out when Task 1 code exists.
    p = Dynamics6DofParams.ct16ev_defaults()
    assert p.mass_kg == pytest.approx(288.0)
    assert p.wheelbase_m == pytest.approx(1.549)
    assert p.cg_height_m == pytest.approx(0.2794)
    assert p.track_front_m == pytest.approx(1.194)
