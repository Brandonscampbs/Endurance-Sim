"""Adapter that routes (slip_angle, slip_ratio, Fz) to the existing PAC02 model.

This is the seam where fastest-lap's built-in tire is replaced. The PAC02
model owns TTC Round-8 coefficients + grip scaling; the adapter just routes
inputs through `combined_forces`.
"""
from __future__ import annotations

from dataclasses import dataclass

from fsae_sim.vehicle.tire_model import PacejkaTireModel


@dataclass
class PAC02Corner:
    """Bridges dynamics6dof kinematics to the project's existing PAC02 model."""

    model: PacejkaTireModel

    def forces(
        self,
        *,
        slip_angle_rad: float,
        slip_ratio: float,
        fz_n: float,
        camber_rad: float = 0.0,
    ) -> tuple[float, float]:
        fx, fy = self.model.combined_forces(slip_angle_rad, slip_ratio, fz_n, camber_rad)
        return float(fx), float(fy)
