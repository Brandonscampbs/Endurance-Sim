"""Runtime battery model calibrated against Voltt cell simulation data.

Implements an equivalent-circuit model: V = OCV(SOC) - I * R_int(SOC)
with temperature-dependent discharge limits, SOC taper, and a lumped
thermal model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from fsae_sim.vehicle.battery import BatteryConfig


@dataclass
class BatteryState:
    """Mutable battery state during simulation."""
    soc_pct: float
    cell_temp_c: float
    pack_voltage_v: float = 0.0
    pack_current_a: float = 0.0


class BatteryModel:
    """Equivalent-circuit battery model calibrated from Voltt cell data.

    The model uses:
    - OCV vs SOC lookup from Voltt simulation OCV column
    - Internal resistance R(SOC) extracted from voltage drop under load
    - BMS discharge limits interpolated over temperature
    - SOC taper below configurable threshold
    - Lumped thermal model for cell temperature evolution
    """

    # P45B cell physical properties (Molicel datasheet)
    CELL_MASS_KG = 0.070
    CELL_SPECIFIC_HEAT_J_PER_KG_K = 1000.0

    def __init__(self, config: BatteryConfig, cell_capacity_ah: float = 4.5) -> None:
        self.config = config
        self.cell_capacity_ah = cell_capacity_ah
        self.pack_capacity_ah = cell_capacity_ah * config.parallel

        # Discharge limit interpolator from BMS config
        temps = np.array([dl.temp_c for dl in config.discharge_limits])
        currents = np.array([dl.max_current_a for dl in config.discharge_limits])
        self._discharge_limit_interp = interp1d(
            temps, currents, kind="linear",
            bounds_error=False,
            fill_value=(float(currents[0]), 0.0),
        )

        # Calibration state (set by calibrate())
        self._ocv_interp: interp1d | None = None
        self._resistance_interp: interp1d | None = None
        self._calibrated = False

        # Pack-level calibration (set by calibrate_pack_from_telemetry())
        self._pack_ocv_interp: interp1d | None = None
        self._pack_resistance: float | None = None
        self._pack_calibrated = False

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, voltt_cell_df: "pd.DataFrame") -> None:
        """Calibrate OCV curve and internal resistance from Voltt cell data.

        The Voltt CSV provides OCV (equilibrium open-circuit voltage) and
        terminal Voltage at each SOC and current point.  Internal resistance
        is computed as R = (OCV - V) / |I| on discharge-only samples.
        """
        import pandas as pd

        soc = voltt_cell_df["SOC [%]"].values
        ocv = voltt_cell_df["OCV [V]"].values
        voltage = voltt_cell_df["Voltage [V]"].values
        current = voltt_cell_df["Current [A]"].values  # negative = discharge

        # --- OCV-SOC curve ---
        # Voltt OCV is already the equilibrium voltage at each SOC. Build a
        # monotonic lookup by sorting on SOC ascending and deduplicating.
        sort_idx = np.argsort(soc)
        soc_sorted = soc[sort_idx]
        ocv_sorted = ocv[sort_idx]

        # Resample to uniform 0.5% SOC grid for a clean interpolator
        soc_grid = np.linspace(soc_sorted[0], soc_sorted[-1], 200)
        ocv_grid = np.interp(soc_grid, soc_sorted, ocv_sorted)

        self._ocv_interp = interp1d(
            soc_grid, ocv_grid, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )

        # --- Internal resistance vs SOC ---
        # Only use samples with meaningful current to avoid 0/0
        discharge_mask = current < -0.1  # Voltt convention: negative = discharge
        if np.sum(discharge_mask) < 10:
            # Fallback: use a constant ~20 mOhm (typical P45B)
            self._resistance_interp = interp1d(
                [0, 100], [0.020, 0.020], kind="linear",
                bounds_error=False, fill_value=0.020,
            )
        else:
            i_discharge = np.abs(current[discharge_mask])
            soc_discharge = soc[discharge_mask]
            ocv_discharge = ocv[discharge_mask]
            v_discharge = voltage[discharge_mask]
            r_values = (ocv_discharge - v_discharge) / i_discharge

            # Clip unreasonable resistance values (negative or very large)
            valid = (r_values > 0.001) & (r_values < 0.5)
            if np.sum(valid) < 10:
                self._resistance_interp = interp1d(
                    [0, 100], [0.020, 0.020], kind="linear",
                    bounds_error=False, fill_value=0.020,
                )
            else:
                r_valid = r_values[valid]
                soc_valid = soc_discharge[valid]

                # Bin by SOC (2% bins) and take median for robustness
                bin_edges = np.arange(
                    max(0, np.floor(soc_valid.min())),
                    min(100, np.ceil(soc_valid.max())) + 2,
                    2.0,
                )
                bin_centers = []
                bin_r = []
                for i in range(len(bin_edges) - 1):
                    mask = (soc_valid >= bin_edges[i]) & (soc_valid < bin_edges[i + 1])
                    if np.sum(mask) >= 3:
                        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        bin_r.append(float(np.median(r_valid[mask])))

                if len(bin_centers) >= 2:
                    self._resistance_interp = interp1d(
                        bin_centers, bin_r, kind="linear",
                        bounds_error=False,
                        fill_value=(bin_r[0], bin_r[-1]),
                    )
                else:
                    median_r = float(np.median(r_valid))
                    self._resistance_interp = interp1d(
                        [0, 100], [median_r, median_r], kind="linear",
                        bounds_error=False, fill_value=median_r,
                    )

        self._calibrated = True

    def calibrate_pack_from_telemetry(self, aim_df: "pd.DataFrame") -> None:
        """Refine pack-level OCV curve and cell capacity from AiM telemetry.

        Uses low-current samples to build a pack OCV(SOC) curve, and
        computes effective cell capacity from SOC change vs integrated
        current.  Must be called AFTER ``calibrate``.
        """
        soc = aim_df["State of Charge"].values
        voltage = aim_df["Pack Voltage"].values
        current = aim_df["Pack Current"].values
        speed = aim_df["GPS Speed"].values

        # Select low-current, moving samples for clean OCV extraction
        mask = (speed > 5) & (np.abs(current) < 3.0)
        if np.sum(mask) < 20:
            return  # not enough data to calibrate

        soc_lc = soc[mask]
        v_lc = voltage[mask]

        # Bin by SOC (2% bins) and take median voltage
        bin_edges = np.arange(
            max(0, np.floor(soc_lc.min())),
            min(100, np.ceil(soc_lc.max())) + 2,
            2.0,
        )
        pack_soc_pts = []
        pack_ocv_pts = []
        for i in range(len(bin_edges) - 1):
            bmask = (soc_lc >= bin_edges[i]) & (soc_lc < bin_edges[i + 1])
            if np.sum(bmask) >= 5:
                pack_soc_pts.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                pack_ocv_pts.append(float(np.median(v_lc[bmask])))

        if len(pack_soc_pts) >= 3:
            # Build pack-level OCV interpolator (replaces cell-level * series)
            self._pack_ocv_interp = interp1d(
                pack_soc_pts, pack_ocv_pts, kind="linear",
                bounds_error=False,
                fill_value=(pack_ocv_pts[0], pack_ocv_pts[-1]),
            )
            self._pack_calibrated = True

            # Recalibrate pack internal resistance from high-current points
            high_i_mask = (speed > 5) & (current > 10)
            if np.sum(high_i_mask) > 20:
                soc_hi = soc[high_i_mask]
                v_hi = voltage[high_i_mask]
                i_hi = current[high_i_mask]
                v_ocv_at_soc = np.array([
                    float(self._pack_ocv_interp(s)) for s in soc_hi
                ])
                r_pack = (v_ocv_at_soc - v_hi) / i_hi
                valid_r = (r_pack > 0.1) & (r_pack < 5.0)
                if np.sum(valid_r) > 10:
                    self._pack_resistance = float(np.median(r_pack[valid_r]))

        # --- Calibrate effective cell capacity from BMS SOC tracking ---
        # The BMS SOC definition may differ from the Voltt model's coulomb
        # counting.  Compute the effective capacity that makes sim SOC
        # match BMS SOC over the full endurance.
        time_arr = aim_df["Time"].values
        dt_arr = np.diff(time_arr, prepend=time_arr[0])
        total_charge_ah = float(np.sum(current * dt_arr)) / 3600  # pack Ah
        soc_change_pct = float(soc[0] - soc[-1])
        if soc_change_pct > 5 and total_charge_ah > 1:
            cell_charge_ah = total_charge_ah / self.config.parallel
            effective_capacity = cell_charge_ah / (soc_change_pct / 100)
            if 3.0 < effective_capacity < 10.0:
                self.cell_capacity_ah = effective_capacity
                self.pack_capacity_ah = effective_capacity * self.config.parallel

    @classmethod
    def from_config_and_data(
        cls,
        config: BatteryConfig,
        voltt_cell_path: str | "Path",
        cell_capacity_ah: float = 4.5,
    ) -> BatteryModel:
        """Construct and calibrate from config + Voltt cell CSV path."""
        from fsae_sim.data.loader import load_voltt_csv

        model = cls(config, cell_capacity_ah)
        df = load_voltt_csv(voltt_cell_path)
        model.calibrate(df)
        return model

    # ------------------------------------------------------------------
    # Voltage model
    # ------------------------------------------------------------------

    def ocv(self, soc_pct: float) -> float:
        """Cell open-circuit voltage at given SOC (percent)."""
        if not self._calibrated:
            raise RuntimeError("Model must be calibrated before use")
        return float(self._ocv_interp(np.clip(soc_pct, 0, 100)))

    def internal_resistance(self, soc_pct: float) -> float:
        """Cell internal resistance (Ohms) at given SOC."""
        if not self._calibrated:
            raise RuntimeError("Model must be calibrated before use")
        return float(np.clip(self._resistance_interp(np.clip(soc_pct, 0, 100)), 0.001, 0.5))

    def cell_voltage(self, soc_pct: float, cell_current_a: float) -> float:
        """Cell terminal voltage.  Positive current = discharge."""
        v = self.ocv(soc_pct) - cell_current_a * self.internal_resistance(soc_pct)
        return max(v, self.config.cell_voltage_min_v)

    def pack_voltage(self, soc_pct: float, pack_current_a: float) -> float:
        """Pack terminal voltage.  Positive current = discharge.

        If pack-level calibration is available (from AiM telemetry), uses
        the pack OCV curve and pack resistance instead of cell-level scaling.
        """
        if self._pack_calibrated:
            pack_ocv = float(self._pack_ocv_interp(np.clip(soc_pct, 0, 100)))
            r_pack = self._pack_resistance if self._pack_resistance else 0.0
            v = pack_ocv - pack_current_a * r_pack
            return max(v, self.config.cell_voltage_min_v * self.config.series)

        cell_i = pack_current_a / self.config.parallel
        cell_v = self.cell_voltage(soc_pct, cell_i)
        return cell_v * self.config.series

    # ------------------------------------------------------------------
    # Current limits
    # ------------------------------------------------------------------

    def max_discharge_current(self, temp_c: float, soc_pct: float) -> float:
        """Maximum pack discharge current (A, positive) given temp and SOC.

        Accounts for:
        1. BMS temperature-dependent limit (linear interpolation)
        2. SOC taper (reduces limit below threshold SOC)
        3. Cell voltage floor (prevents cell voltage from going below minimum)
        """
        # Temperature limit
        temp_limit = float(self._discharge_limit_interp(np.clip(temp_c, 0, 80)))
        temp_limit = max(0.0, temp_limit)

        # SOC taper: reduce by rate_a_per_pct for each % below threshold
        if soc_pct < self.config.soc_taper_threshold_pct:
            deficit = self.config.soc_taper_threshold_pct - soc_pct
            soc_reduction = deficit * self.config.soc_taper_rate_a_per_pct
            soc_limit = max(0.0, temp_limit - soc_reduction)
        else:
            soc_limit = temp_limit

        # Voltage floor: I_max_cell = (OCV - V_min) / R
        if self._calibrated:
            cell_ocv = self.ocv(soc_pct)
            r = self.internal_resistance(soc_pct)
            if r > 0:
                v_headroom = cell_ocv - self.config.cell_voltage_min_v
                voltage_limit_pack = max(0.0, v_headroom / r) * self.config.parallel
            else:
                voltage_limit_pack = float("inf")
        else:
            voltage_limit_pack = float("inf")

        return max(0.0, min(soc_limit, voltage_limit_pack))

    # ------------------------------------------------------------------
    # State stepping
    # ------------------------------------------------------------------

    def step(
        self,
        pack_current_a: float,
        dt_s: float,
        soc_pct: float,
        temp_c: float,
    ) -> tuple[float, float, float]:
        """Advance battery state by one timestep.

        Args:
            pack_current_a: Pack current (positive = discharge).
            dt_s: Timestep in seconds.
            soc_pct: Current state-of-charge (percent, 0-100).
            temp_c: Current cell temperature (Celsius).

        Returns:
            (new_soc_pct, new_temp_c, pack_voltage_v)
        """
        # Coulomb counting: SOC change
        cell_current = pack_current_a / self.config.parallel
        dsoc = -(cell_current * dt_s) / (self.cell_capacity_ah * 3600) * 100
        new_soc = float(np.clip(soc_pct + dsoc, 0.0, 100.0))

        # Pack voltage at updated SOC
        v_pack = self.pack_voltage(new_soc, pack_current_a)

        # Thermal model: lumped I^2*R heating, no active cooling (2025 car)
        r_cell = self.internal_resistance(new_soc)
        heat_per_cell_w = cell_current ** 2 * r_cell
        num_cells = self.config.series * self.config.parallel
        total_heat_w = heat_per_cell_w * num_cells

        thermal_mass = (
            self.CELL_MASS_KG * self.CELL_SPECIFIC_HEAT_J_PER_KG_K * num_cells
        )
        dtemp = (total_heat_w * dt_s) / thermal_mass if thermal_mass > 0 else 0.0
        new_temp = temp_c + dtemp

        return new_soc, new_temp, v_pack
