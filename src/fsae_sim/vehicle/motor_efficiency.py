"""Motor + inverter efficiency map from EMRAX 228 characterization data.

Provides a 2D lookup (RPM, torque) -> combined motor+inverter efficiency,
replacing the fixed drivetrain_efficiency scalar for more accurate power
and energy calculations.

The gearbox friction (~2-3%) is a separate, approximately constant loss
that is applied on top of the motor+inverter efficiency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class MotorEfficiencyMap:
    """2D efficiency lookup for the EMRAX 228 + Cascadia CM200DX.

    Loads a CSV with columns: speed_rpm, torque_Nm, efficiency_pct.
    Builds a bilinear interpolator over the (RPM, torque) grid.

    At zero torque or zero RPM, efficiency is undefined (no mechanical
    power).  The map returns a floor value in those cases.
    """

    # Minimum efficiency to return (avoids division by zero and
    # unrealistic values at near-zero operating points).
    _FLOOR_EFFICIENCY: float = 0.80

    # Gearbox mechanical efficiency (chain/gear drive, single speed).
    # Applied on top of motor+inverter efficiency.
    GEARBOX_EFFICIENCY: float = 0.97

    def __init__(self, csv_path: str | Path) -> None:
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Extract unique RPM and torque values
        rpm_vals = np.sort(df["speed_rpm"].unique())
        torque_vals = np.sort(df["torque_Nm"].unique())

        # Build 2D grid: efficiency[rpm_idx, torque_idx]
        eff_grid = np.full((len(rpm_vals), len(torque_vals)), np.nan)

        rpm_to_idx = {r: i for i, r in enumerate(rpm_vals)}
        torque_to_idx = {t: i for i, t in enumerate(torque_vals)}

        for _, row in df.iterrows():
            ri = rpm_to_idx[row["speed_rpm"]]
            ti = torque_to_idx[row["torque_Nm"]]
            if not np.isnan(row["efficiency_pct"]):
                eff_grid[ri, ti] = row["efficiency_pct"] / 100.0

        # Fill NaN with nearest valid value (for out-of-envelope points)
        # Use forward/backward fill along torque axis, then RPM axis
        for i in range(len(rpm_vals)):
            row = eff_grid[i, :]
            valid = ~np.isnan(row)
            if np.any(valid):
                # Interpolate/extrapolate from valid points
                valid_idx = np.where(valid)[0]
                eff_grid[i, :] = np.interp(
                    np.arange(len(torque_vals)),
                    valid_idx,
                    row[valid],
                )
            else:
                eff_grid[i, :] = self._FLOOR_EFFICIENCY

        self._interpolator = RegularGridInterpolator(
            (rpm_vals.astype(float), torque_vals.astype(float)),
            eff_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate
        )

        self._rpm_range = (float(rpm_vals[0]), float(rpm_vals[-1]))
        self._torque_range = (float(torque_vals[0]), float(torque_vals[-1]))

    def efficiency(self, motor_rpm: float, motor_torque_nm: float) -> float:
        """Combined motor + inverter efficiency at the given operating point.

        Args:
            motor_rpm: Motor shaft speed in RPM (>= 0).
            motor_torque_nm: Motor torque magnitude in Nm (>= 0).

        Returns:
            Combined motor+inverter efficiency as a fraction (0-1).
        """
        rpm = max(0.0, motor_rpm)
        torque = max(0.0, abs(motor_torque_nm))

        # Clamp to grid range for interpolation
        rpm = min(rpm, self._rpm_range[1])
        torque = min(torque, self._torque_range[1])

        if rpm < 1.0 or torque < 1.0:
            return self._FLOOR_EFFICIENCY

        eff = float(self._interpolator((rpm, torque)))
        return max(self._FLOOR_EFFICIENCY, min(1.0, eff))

    def total_efficiency(self, motor_rpm: float, motor_torque_nm: float) -> float:
        """Total drivetrain efficiency: motor+inverter × gearbox.

        This replaces the fixed ``drivetrain_efficiency`` scalar in the
        powertrain model.
        """
        return self.efficiency(motor_rpm, motor_torque_nm) * self.GEARBOX_EFFICIENCY
