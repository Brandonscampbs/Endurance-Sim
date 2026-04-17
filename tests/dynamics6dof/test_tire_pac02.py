from pathlib import Path

import pytest

from fsae_sim.dynamics6dof.tire_pac02 import PAC02Corner


@pytest.fixture
def pac02_model():
    from fsae_sim.vehicle.tire_model import PacejkaTireModel
    from fsae_sim.vehicle.vehicle import VehicleConfig

    cfg_path = Path(__file__).parents[2] / "configs" / "ct16ev.yaml"
    cfg = VehicleConfig.from_yaml(cfg_path)
    tir_path = Path(__file__).parents[2] / cfg.tire.tir_file
    model = PacejkaTireModel(tir_path)
    model.apply_grip_scale(cfg.tire.grip_scale)
    return model


def test_adapter_fy_sign_is_flipped_for_iso_convention(pac02_model):
    """PAC02 is fit in SAE; dynamics6dof uses ISO. Fy must flip."""
    corner = PAC02Corner(pac02_model)
    fx_sae, fy_sae = pac02_model.combined_forces(0.05, 0.02, 700.0, 0.0)
    fx, fy = corner.forces(slip_angle_rad=0.05, slip_ratio=0.02, fz_n=700.0)
    assert fx == pytest.approx(float(fx_sae), rel=1e-12)
    assert fy == pytest.approx(float(-fy_sae), rel=1e-12)


def test_positive_slip_angle_gives_positive_fy_in_iso(pac02_model):
    """In ISO convention, positive α means velocity is left of wheel plane;
    the restoring tire force should be to the right (negative y_ISO)... but
    PAC02 fit returns Fy_SAE<0 for α>0, so after flip Fy_ISO>0. Verify the
    adapter consistently yields a restoring torque for forward-driving cars
    under positive steering (see end-to-end tests)."""
    corner = PAC02Corner(pac02_model)
    _, fy = corner.forces(slip_angle_rad=0.1, slip_ratio=0.0, fz_n=700.0)
    assert fy > 0  # flipped from SAE's negative


def test_adapter_produces_nonzero_forces_at_nonzero_slip(pac02_model):
    corner = PAC02Corner(pac02_model)
    fx, fy = corner.forces(slip_angle_rad=0.05, slip_ratio=0.02, fz_n=700.0)
    # Both should be nonzero and within the physically plausible range for an
    # FSAE tire at 700 N vertical load.
    assert abs(fx) > 0.0
    assert abs(fy) > 0.0
    assert abs(fx) < 3000.0
    assert abs(fy) < 3000.0
