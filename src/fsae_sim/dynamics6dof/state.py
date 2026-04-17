# src/fsae_sim/dynamics6dof/state.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STATE_SIZE = 10


@dataclass(frozen=True)
class State6Dof:
    """10-state kart-6dof Cartesian state.

    Indices match the C++ order from fastest-lap's curvilinear kart, minus the
    three road states (t, n, chi) that our outer lap sim owns separately.
    """

    rear_omega: float  # rad/s, rear axle shaft speed
    vx: float          # m/s, body frame forward
    vy: float          # m/s, body frame lateral (positive = left)
    wz: float          # rad/s, yaw rate (positive z-up = CCW)
    z: float           # m, chassis vertical displacement from nominal
    phi: float         # rad, roll (x-axis, positive = left side up)
    mu: float          # rad, pitch (y-axis, positive = nose up)
    dz: float          # m/s, rate of z
    dphi: float        # rad/s
    dmu: float         # rad/s

    def to_array(self) -> np.ndarray:
        return np.asarray(
            [self.rear_omega, self.vx, self.vy, self.wz, self.z,
             self.phi, self.mu, self.dz, self.dphi, self.dmu],
            dtype=float,
        )

    @classmethod
    def from_array(cls, q: np.ndarray) -> "State6Dof":
        q = np.asarray(q, dtype=float)
        if q.shape != (STATE_SIZE,):
            raise ValueError(f"expected shape ({STATE_SIZE},), got {q.shape}")
        return cls(*q.tolist())

    @classmethod
    def initial(cls, vx: float = 0.0, rear_omega: float = 0.0) -> "State6Dof":
        return cls(rear_omega=rear_omega, vx=vx, vy=0.0, wz=0.0, z=0.0,
                   phi=0.0, mu=0.0, dz=0.0, dphi=0.0, dmu=0.0)
