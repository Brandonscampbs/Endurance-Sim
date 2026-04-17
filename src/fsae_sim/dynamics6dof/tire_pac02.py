"""Adapter that routes (slip_angle, slip_ratio, Fz) to the existing PAC02 model.

This is the seam where fastest-lap's built-in tire is replaced and also where
the SAE tire axis convention used by the PAC02 coefficients is translated to
the dynamics6dof ISO 8855 convention (x forward, y LEFT, z up).

Convention mapping between the TTC Round-8 PAC02 fit (SAE: y-right, z-down)
and dynamics6dof (ISO: y-left, z-up):

- Slip angle sign is invariant: ``α_SAE = -vy_ISO/vx = α_ISO``. No flip.
- Slip ratio sign is invariant: ``κ = (ωR − vx)/vx``. No flip.
- Fx direction is the same in both (forward is +x). No flip.
- Fy direction flips: ``Fy_ISO = -Fy_SAE`` because y_ISO points LEFT while
  y_SAE points RIGHT. A force in +y_SAE is in -y_ISO.

We flip Fy here so callers of this adapter can work in ISO throughout.
"""
from __future__ import annotations

from dataclasses import dataclass

from fsae_sim.vehicle.tire_model import PacejkaTireModel


@dataclass
class PAC02Corner:
    """Bridges dynamics6dof kinematics (ISO) to the SAE-convention PAC02 model."""

    model: PacejkaTireModel

    def forces(
        self,
        *,
        slip_angle_rad: float,
        slip_ratio: float,
        fz_n: float,
        camber_rad: float = 0.0,
    ) -> tuple[float, float]:
        fx_sae, fy_sae = self.model.combined_forces(
            slip_angle_rad, slip_ratio, fz_n, camber_rad,
        )
        # ISO body frame: Fy flipped, Fx unchanged.
        return float(fx_sae), float(-fy_sae)
