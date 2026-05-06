"""Inverter torque-delivery map: LVCU command -> motor shaft torque.

The Cascadia CM200DX inverter does not perfectly track the LVCU's
torque request. Two regimes dominate the deviation:

* **Low RPM / launch transients.** The inverter ramps IQ over a
  finite time; the steady-state-quasi-static sim replays a step input,
  so the bin-averaged delivered torque sits below the request.
* **Field weakening (above ~2800 RPM on this car).** Back-EMF
  approaches the DC bus voltage and the inverter physically cannot
  push enough IQ to meet the request, regardless of LVCU intent.

Across the heart of the operating envelope (1000-2700 RPM, 5-70 Nm),
delivery is 97-100% of request: the LVCU and inverter agree.

Loading a CSV produced by ``scripts/build_inverter_delivery_map.py``
lets ``PowertrainModel`` translate an LVCU command into a realistic
motor shaft torque, so replay no longer over-torques the wheels and
the Simulate page's torque-limit knob respects the inverter's actual
delivery characteristics.
"""

from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path

import numpy as np


class InverterDeliveryMap:
    """2D lookup ``(motor_rpm, lvcu_command_nm) -> delivered_torque_nm``.

    Loads a CSV with columns ``rpm``, ``command_nm``, ``delivered_nm``.
    The grid must be rectangular: every (rpm, command_nm) pair appears
    exactly once. Bilinear interpolation is used inside the grid;
    outside, lookups clamp to the nearest grid edge.

    The class enforces two physical bounds at lookup time:

    1. ``delivered <= command`` — the inverter cannot amplify torque.
    2. ``delivered <= inverter_torque_cap_nm`` — hard mechanical ceiling
       set by the inverter's IQ limit (85 Nm for CM200DX at IQ=170 A).
    """

    DEFAULT_INVERTER_CAP_NM: float = 85.0

    def __init__(
        self,
        rpm_grid: np.ndarray,
        command_grid: np.ndarray,
        delivered_grid: np.ndarray,
        inverter_torque_cap_nm: float = DEFAULT_INVERTER_CAP_NM,
    ) -> None:
        rpm_grid = np.asarray(rpm_grid, dtype=float)
        command_grid = np.asarray(command_grid, dtype=float)
        delivered_grid = np.asarray(delivered_grid, dtype=float)

        if delivered_grid.shape != (len(rpm_grid), len(command_grid)):
            raise ValueError(
                "delivered_grid shape "
                f"{delivered_grid.shape} does not match "
                f"({len(rpm_grid)}, {len(command_grid)})"
            )
        if not np.all(np.diff(rpm_grid) > 0):
            raise ValueError("rpm_grid must be strictly increasing")
        if not np.all(np.diff(command_grid) > 0):
            raise ValueError("command_grid must be strictly increasing")

        self._rpm_grid = rpm_grid
        self._command_grid = command_grid
        self._delivered_grid = delivered_grid
        self._inverter_cap = float(inverter_torque_cap_nm)

        self._rpm_min = float(rpm_grid[0])
        self._rpm_max = float(rpm_grid[-1])
        self._command_min = float(command_grid[0])
        self._command_max = float(command_grid[-1])
        self._rpm_inv_step = self._uniform_inv_step(rpm_grid)
        self._command_inv_step = self._uniform_inv_step(command_grid)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        inverter_torque_cap_nm: float = DEFAULT_INVERTER_CAP_NM,
    ) -> "InverterDeliveryMap":
        import pandas as pd

        df = pd.read_csv(csv_path)
        required = {"rpm", "command_nm", "delivered_nm"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {sorted(missing)}"
            )

        rpm_grid = np.sort(df["rpm"].unique())
        command_grid = np.sort(df["command_nm"].unique())

        if len(df) != len(rpm_grid) * len(command_grid):
            raise ValueError(
                f"CSV {csv_path} is not a complete rectangular grid: "
                f"got {len(df)} rows, expected "
                f"{len(rpm_grid) * len(command_grid)}"
            )

        delivered = np.full((len(rpm_grid), len(command_grid)), np.nan)
        rpm_to_idx = {r: i for i, r in enumerate(rpm_grid)}
        cmd_to_idx = {c: j for j, c in enumerate(command_grid)}
        for _, row in df.iterrows():
            i = rpm_to_idx[row["rpm"]]
            j = cmd_to_idx[row["command_nm"]]
            delivered[i, j] = float(row["delivered_nm"])

        if np.isnan(delivered).any():
            raise ValueError(
                f"CSV {csv_path} has NaN entries after grid assembly"
            )

        return cls(
            rpm_grid=rpm_grid,
            command_grid=command_grid,
            delivered_grid=delivered,
            inverter_torque_cap_nm=inverter_torque_cap_nm,
        )

    def delivered_torque(
        self, motor_rpm: float, command_nm: float,
    ) -> float:
        """Return the inverter-delivered motor torque in Nm.

        The result is clamped so that ``0 <= delivered <= command`` and
        ``delivered <= inverter_torque_cap_nm``. Negative commands are
        not modelled (this map covers motoring only); regen torque
        passes through unchanged via the calling site.
        """
        cmd = float(command_nm)
        if cmd <= 0.0:
            return 0.0

        rpm = max(self._rpm_min, min(self._rpm_max, float(motor_rpm)))
        cmd_clamped = max(self._command_min, min(self._command_max, cmd))

        delivered = self._bilinear_lookup(rpm, cmd_clamped)
        delivered = min(delivered, cmd, self._inverter_cap)
        return max(0.0, delivered)

    @staticmethod
    def _uniform_inv_step(grid: np.ndarray) -> float | None:
        if len(grid) < 2:
            return None
        step = float(grid[1] - grid[0])
        if step <= 0.0:
            return None
        if np.allclose(np.diff(grid), step):
            return 1.0 / step
        return None

    @staticmethod
    def _bracket(grid: np.ndarray, value: float) -> tuple[int, int, float]:
        if len(grid) == 1:
            return 0, 0, 0.0
        hi = bisect_right(grid, value)
        if hi <= 0:
            return 0, 1, 0.0
        if hi >= len(grid):
            return len(grid) - 2, len(grid) - 1, 1.0
        lo = hi - 1
        width = float(grid[hi] - grid[lo])
        weight = 0.0 if width <= 0.0 else (value - float(grid[lo])) / width
        return lo, hi, float(weight)

    @staticmethod
    def _uniform_bracket(
        value: float,
        grid_min: float,
        inv_step: float,
        size: int,
    ) -> tuple[int, int, float]:
        if size == 1:
            return 0, 0, 0.0
        position = (value - grid_min) * inv_step
        lo = int(math.floor(position))
        if lo <= 0:
            if position <= 0.0:
                return 0, 1, 0.0
            lo = 0
        if lo >= size - 1:
            return size - 2, size - 1, 1.0
        return lo, lo + 1, float(position - lo)

    def _bilinear_lookup(self, rpm: float, command_nm: float) -> float:
        if self._rpm_inv_step is not None:
            ri0, ri1, rw = self._uniform_bracket(
                rpm, self._rpm_min, self._rpm_inv_step, len(self._rpm_grid),
            )
        else:
            ri0, ri1, rw = self._bracket(self._rpm_grid, rpm)
        if self._command_inv_step is not None:
            ci0, ci1, cw = self._uniform_bracket(
                command_nm,
                self._command_min,
                self._command_inv_step,
                len(self._command_grid),
            )
        else:
            ci0, ci1, cw = self._bracket(self._command_grid, command_nm)
        z00 = self._delivered_grid[ri0, ci0]
        z01 = self._delivered_grid[ri0, ci1]
        z10 = self._delivered_grid[ri1, ci0]
        z11 = self._delivered_grid[ri1, ci1]
        z0 = z00 + (z01 - z00) * cw
        z1 = z10 + (z11 - z10) * cw
        return float(z0 + (z1 - z0) * rw)

    @property
    def rpm_range(self) -> tuple[float, float]:
        return self._rpm_min, self._rpm_max

    @property
    def command_range(self) -> tuple[float, float]:
        return self._command_min, self._command_max
