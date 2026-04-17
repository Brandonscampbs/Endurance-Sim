import numpy as np
import pytest

from fsae_sim.dynamics6dof.gravity import gravity_body_force
from fsae_sim.dynamics6dof.params import Dynamics6DofParams


def test_level_road_gravity_is_minus_mg_z():
    p = Dynamics6DofParams.ct16ev_defaults()
    f = gravity_body_force(roll_rad=0.0, pitch_rad=0.0, params=p)
    np.testing.assert_allclose(f, [0.0, 0.0, -p.mass_kg * 9.81], atol=1e-9)


def test_pitch_up_projects_some_gravity_onto_minus_x():
    p = Dynamics6DofParams.ct16ev_defaults()
    f = gravity_body_force(roll_rad=0.0, pitch_rad=0.1, params=p)
    # With nose up, gravity gains a -x component (pulls car back)
    assert f[0] < 0.0
    assert f[2] < 0.0
