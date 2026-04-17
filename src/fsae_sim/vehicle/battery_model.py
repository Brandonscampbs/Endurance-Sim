"""Runtime battery model calibrated against Voltt cell simulation data.

Implements an equivalent-circuit model: V = OCV(SOC) - I * R_int(SOC)
with temperature-dependent discharge limits, SOC taper, and a lumped
thermal model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import interp1d

from fsae_sim.vehicle.battery import BatteryConfig


@dataclass
class BatteryViolation:
    """Record of a BMS/battery physical-limit violation during simulation.

    Emitted (but not raised) by the battery model so the engine can decide
    whether to terminate, flag, or just log.
    """

    kind: str  # "voltage_floor", "ocv_extrapolation", ...
    time_s: float
    soc_pct: float
    cell_voltage_v: float
    pack_current_a: float
    message: str = ""


@dataclass
class BatteryState:
    """Mutable battery state during simulation.

    Note: ``mean_cell_temp_c`` is the lumped-mass mean cell temperature.
    AiM telemetry ``Pack Temp`` is the *max* cell temperature (hottest
    BMS sensor); comparisons between the two require either a separate
    telemetry-mean computation or a max-cell model. See audit NF-10.
    """
    soc_pct: float
    mean_cell_temp_c: float
    pack_voltage_v: float = 0.0
    pack_current_a: float = 0.0
    violations: list[BatteryViolation] = field(default_factory=list)


class BatteryModel:
    """Equivalent-circuit battery model calibrated from Voltt cell data.

    The model uses:
    - OCV vs SOC lookup from Voltt simulation OCV column
    - Internal resistance R(SOC) extracted from voltage drop under load
    - BMS discharge limits interpolated over temperature
    - SOC taper below configurable threshold
    - Lumped thermal model for cell temperature evolution
    """

    # P45B cell physical properties (Molicel datasheet).
    # Structural thermal mass (busbars, plates, enclosure) is config-driven
    # via ``pack_structural_thermal_mass_kj_per_k`` on ``BatteryConfig``.
    CELL_MASS_KG = 0.070
    CELL_SPECIFIC_HEAT_J_PER_KG_K = 1000.0

    def __init__(self, config: BatteryConfig, cell_capacity_ah: float | None = None) -> None:
        """Initialize battery model.

        Args:
            config: Battery configuration (carries ``cell_capacity_ah``).
            cell_capacity_ah: Deprecated override for backward compatibility.
                If given, overrides ``config.cell_capacity_ah`` and emits a
                ``DeprecationWarning``. Prefer setting capacity in config.
        """
        self.config = config
        if cell_capacity_ah is not None:
            warnings.warn(
                "Passing cell_capacity_ah to BatteryModel is deprecated; "
                "set cell_capacity_ah on BatteryConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.cell_capacity_ah = float(cell_capacity_ah)
        else:
            self.cell_capacity_ah = float(config.cell_capacity_ah)
        self.pack_capacity_ah = self.cell_capacity_ah * config.parallel

        # Discharge limit interpolator from BMS config
        temps = np.array([dl.temp_c for dl in config.discharge_limits])
        currents = np.array([dl.max_current_a for dl in config.discharge_limits])
        self._discharge_limit_interp = interp1d(
            temps, currents, kind="linear",
            bounds_error=False,
            fill_value=(float(currents[0]), 0.0),
        )

        # Calibration state (set by calibrate_from_voltt())
        self._ocv_interp: interp1d | None = None
        self._resistance_interp: interp1d | None = None
        self._ocv_soc_min: float | None = None  # for extrapolation floor
        self._ocv_soc_max: float | None = None
        self._ocv_extrap_warned: bool = False
        self._calibrated = False

        # Pack-level calibration (set by calibrate_pack_from_telemetry())
        self._pack_ocv_interp: interp1d | None = None
        self._pack_resistance_interp: interp1d | None = None
        self._pack_calibrated = False

        # Idempotency guard for pack-from-telemetry calibration (NF-26)
        self._pack_telemetry_calibrated: bool = False

        # Event sink for voltage-floor / extrapolation violations (S16)
        self.violations: list[BatteryViolation] = []

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def pack_calibrated(self) -> bool:
        return self._pack_calibrated

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_from_voltt(self, voltt_cell_df: "pd.DataFrame") -> None:
        """Calibrate cell-level OCV and resistance from Voltt cell-level data.

        The Voltt CSV provides OCV (equilibrium open-circuit voltage) and
        terminal Voltage at each SOC and current point.  Internal resistance
        is computed as R = (OCV - V) / |I| on discharge-only samples.

        This is the *only* battery calibration source; pack-level
        calibration from AiM telemetry is a separate, optional
        downstream step and does NOT overwrite these parameters.
        """
        import pandas as pd  # noqa: F401 — imported lazily for type hints

        soc = voltt_cell_df["SOC [%]"].values
        ocv = voltt_cell_df["OCV [V]"].values
        voltage = voltt_cell_df["Voltage [V]"].values
        current = voltt_cell_df["Current [A]"].values  # negative = discharge

        # --- OCV-SOC curve ---
        sort_idx = np.argsort(soc)
        soc_sorted = soc[sort_idx]
        ocv_sorted = ocv[sort_idx]

        # Resample to uniform 0.5% SOC grid for a clean interpolator
        soc_grid = np.linspace(soc_sorted[0], soc_sorted[-1], 200)
        ocv_grid = np.interp(soc_grid, soc_sorted, ocv_sorted)

        # bounds_error=False + fill_value="extrapolate" lets us flag
        # extrapolation explicitly at call time (NF-16); we also
        # track the calibrated SOC range.
        self._ocv_interp = interp1d(
            soc_grid, ocv_grid, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )
        self._ocv_soc_min = float(soc_grid[0])
        self._ocv_soc_max = float(soc_grid[-1])

        # --- Internal resistance vs SOC ---
        discharge_mask = current < -0.1  # Voltt convention: negative = discharge
        if np.sum(discharge_mask) < 10:
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

            valid = (r_values > 0.001) & (r_values < 0.5)
            if np.sum(valid) < 10:
                self._resistance_interp = interp1d(
                    [0, 100], [0.020, 0.020], kind="linear",
                    bounds_error=False, fill_value=0.020,
                )
            else:
                r_valid = r_values[valid]
                soc_valid = soc_discharge[valid]

                bin_edges = np.arange(
                    max(0, np.floor(soc_valid.min())),
                    min(100, np.ceil(soc_valid.max())) + 2,
                    2.0,
                )
                bin_centers: list[float] = []
                bin_r: list[float] = []
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

        # S17: build a pack-level R(SOC) interpolator by scaling cell R
        # geometrically to pack wiring (series / parallel).  This gives
        # the pack model a SOC-dependent resistance even without AiM
        # telemetry calibration.
        soc_grid_pack = np.linspace(0.0, 100.0, 101)
        r_pack_grid = np.array([
            float(self._resistance_interp(s)) * self.config.series / self.config.parallel
            for s in soc_grid_pack
        ])
        self._pack_resistance_interp = interp1d(
            soc_grid_pack, r_pack_grid, kind="linear",
            bounds_error=False,
            fill_value=(float(r_pack_grid[0]), float(r_pack_grid[-1])),
        )

        self._calibrated = True

    # Backward-compatible alias for existing call sites.  Deprecated.
    def calibrate(self, voltt_cell_df: "pd.DataFrame") -> None:
        """Alias for :meth:`calibrate_from_voltt`. Prefer the new name."""
        self.calibrate_from_voltt(voltt_cell_df)

    def calibrate_pack_from_telemetry(
        self,
        aim_df: "pd.DataFrame",
        holdout_laps: Sequence[int] | None = None,
    ) -> None:
        """Refine pack-level OCV curve from AiM telemetry.

        **Idempotent**: calling twice raises ``RuntimeError``. Capacity is
        NOT re-derived from telemetry (C15 fix); it comes from config only.

        Args:
            aim_df: Cleaned AiM telemetry DataFrame with ``State of Charge``,
                ``Pack Voltage``, ``Pack Current``, ``GPS Speed``, ``Time``,
                and optionally ``lap`` columns.
            holdout_laps: Optional iterable of lap indices to exclude from
                the fit. When the validation harness holds out laps
                13-21, pass them here to prevent the calibrate-then-
                validate-on-the-same-file circularity (C15).
        """
        if self._pack_telemetry_calibrated:
            raise RuntimeError(
                "calibrate_pack_from_telemetry has already been called on "
                "this model. Create a new BatteryModel instance instead."
            )
        # D-04: make the train/test leak explicit. Fitting pack OCV and
        # resistance on the same AiM recording used for validation is a
        # circular comparison; `scripts/validate_tier3.py` deliberately
        # does NOT call this. When callers opt in (e.g. sweep scripts that
        # train on stint 1 and validate on stint 2), they must pass
        # `holdout_laps`.
        import warnings
        warnings.warn(
            "calibrate_pack_from_telemetry fits pack OCV/R on AiM telemetry "
            "— do not use for validation against the same data. Pass "
            "`holdout_laps=` or rely on `calibrate_from_voltt` alone.",
            stacklevel=2,
        )
        self._pack_telemetry_calibrated = True

        # Apply holdout filter.  We look for a 'lap' column; if absent,
        # we silently accept the full frame (no laps to hold out).
        if holdout_laps is not None and "lap" in aim_df.columns:
            mask_keep = ~aim_df["lap"].isin(list(holdout_laps))
            aim_df = aim_df.loc[mask_keep]

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
        pack_soc_pts: list[float] = []
        pack_ocv_pts: list[float] = []
        for i in range(len(bin_edges) - 1):
            bmask = (soc_lc >= bin_edges[i]) & (soc_lc < bin_edges[i + 1])
            if np.sum(bmask) >= 5:
                pack_soc_pts.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                pack_ocv_pts.append(float(np.median(v_lc[bmask])))

        if len(pack_soc_pts) >= 3:
            self._pack_ocv_interp = interp1d(
                pack_soc_pts, pack_ocv_pts, kind="linear",
                bounds_error=False,
                fill_value=(pack_ocv_pts[0], pack_ocv_pts[-1]),
            )
            self._pack_calibrated = True

            # S17: fit pack R(SOC) interpolator from high-current bins.
            # Falls back to the Voltt-derived geometric pack R if we
            # can't bin enough points.
            high_i_mask = (speed > 5) & (current > 10)
            if np.sum(high_i_mask) > 40:
                soc_hi = soc[high_i_mask]
                v_hi = voltage[high_i_mask]
                i_hi = current[high_i_mask]
                v_ocv_at_soc = np.array([
                    float(self._pack_ocv_interp(s)) for s in soc_hi
                ])
                r_pack = (v_ocv_at_soc - v_hi) / i_hi
                valid_r = (r_pack > 0.05) & (r_pack < 5.0)
                if np.sum(valid_r) > 20:
                    r_valid = r_pack[valid_r]
                    soc_valid = soc_hi[valid_r]

                    r_bin_edges = np.arange(
                        max(0, np.floor(soc_valid.min())),
                        min(100, np.ceil(soc_valid.max())) + 4,
                        4.0,
                    )
                    r_bin_centers: list[float] = []
                    r_bin_vals: list[float] = []
                    for i in range(len(r_bin_edges) - 1):
                        bmask = (soc_valid >= r_bin_edges[i]) & (soc_valid < r_bin_edges[i + 1])
                        if np.sum(bmask) >= 5:
                            r_bin_centers.append(
                                (r_bin_edges[i] + r_bin_edges[i + 1]) / 2
                            )
                            r_bin_vals.append(float(np.median(r_valid[bmask])))

                    if len(r_bin_centers) >= 2:
                        self._pack_resistance_interp = interp1d(
                            r_bin_centers, r_bin_vals, kind="linear",
                            bounds_error=False,
                            fill_value=(r_bin_vals[0], r_bin_vals[-1]),
                        )

        # C15: capacity is config-driven. We DO NOT rewrite
        # ``cell_capacity_ah`` here; do that only in the config.

    @classmethod
    def from_config_and_data(
        cls,
        config: BatteryConfig,
        voltt_cell_path: str | "Path",
        cell_capacity_ah: float | None = None,
    ) -> BatteryModel:
        """Construct and calibrate from config + Voltt cell CSV path."""
        from fsae_sim.data.loader import load_voltt_csv

        model = cls(config, cell_capacity_ah)
        df = load_voltt_csv(voltt_cell_path)
        model.calibrate_from_voltt(df)
        return model

    # ------------------------------------------------------------------
    # Voltage model
    # ------------------------------------------------------------------

    def ocv(self, soc_pct: float) -> float:
        """Cell open-circuit voltage at given SOC (percent).

        NF-16: when the interpolator would extrapolate below the calibrated
        SOC range, the returned OCV is floored at
        ``cell_voltage_min_v + 0.01 V``. A warning is emitted once per
        model instance to surface the event.
        """
        if not self._calibrated:
            raise RuntimeError("Model must be calibrated before use")
        soc_clipped = float(np.clip(soc_pct, 0, 100))
        raw = float(self._ocv_interp(soc_clipped))

        below = (
            self._ocv_soc_min is not None and soc_clipped < self._ocv_soc_min
        )
        floor_v = self.config.cell_voltage_min_v + 0.01
        if below and raw < floor_v:
            if not self._ocv_extrap_warned:
                warnings.warn(
                    f"BatteryModel.ocv: SOC {soc_clipped:.2f}% below calibrated "
                    f"range ({self._ocv_soc_min:.2f}%); OCV extrapolation "
                    f"clipped to {floor_v:.3f} V. Only warned once per model.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._ocv_extrap_warned = True
            return floor_v
        return raw

    def internal_resistance(self, soc_pct: float) -> float:
        """Cell internal resistance (Ohms) at given SOC."""
        if not self._calibrated:
            raise RuntimeError("Model must be calibrated before use")
        return float(np.clip(self._resistance_interp(np.clip(soc_pct, 0, 100)), 0.001, 0.5))

    def pack_resistance(self, soc_pct: float) -> float:
        """Pack internal resistance (Ohms) at given SOC.

        S17: uses the pack-level R(SOC) interpolator built from Voltt
        (geometric scaling) and optionally refined from AiM telemetry.
        Replaces the previous single-scalar pack resistance that
        erased SOC/temperature dependence.
        """
        if not self._calibrated:
            raise RuntimeError("Model must be calibrated before use")
        assert self._pack_resistance_interp is not None
        return float(np.clip(
            self._pack_resistance_interp(np.clip(soc_pct, 0, 100)),
            0.001,
            50.0,
        ))

    def cell_voltage(
        self,
        soc_pct: float,
        cell_current_a: float,
        *,
        time_s: float | None = None,
        pack_current_a: float | None = None,
    ) -> float:
        """Cell terminal voltage.  Positive current = discharge.

        S16: if the unclamped voltage falls below ``cell_voltage_min_v``,
        a :class:`BatteryViolation` event is appended to ``self.violations``
        (so the engine can terminate or flag), and the clamped floor is
        returned for continuity.
        """
        v_unclamped = self.ocv(soc_pct) - cell_current_a * self.internal_resistance(soc_pct)
        floor = self.config.cell_voltage_min_v
        if v_unclamped < floor:
            self.violations.append(BatteryViolation(
                kind="voltage_floor",
                time_s=float(time_s) if time_s is not None else float("nan"),
                soc_pct=float(soc_pct),
                cell_voltage_v=float(v_unclamped),
                pack_current_a=float(pack_current_a) if pack_current_a is not None else float(cell_current_a) * self.config.parallel,
                message=(
                    f"Predicted cell voltage {v_unclamped:.3f} V below floor "
                    f"{floor:.3f} V at SOC={soc_pct:.2f}%, I_cell={cell_current_a:.2f} A"
                ),
            ))
            return floor
        return v_unclamped

    def pack_voltage(
        self,
        soc_pct: float,
        pack_current_a: float,
        *,
        time_s: float | None = None,
    ) -> float:
        """Pack terminal voltage.  Positive current = discharge.

        Uses pack-level OCV (if AiM-calibrated) and pack-level R(SOC)
        (S17).  Voltage-floor violations are logged via
        :class:`BatteryViolation` (S16).
        """
        if self._pack_calibrated:
            pack_ocv = float(self._pack_ocv_interp(np.clip(soc_pct, 0, 100)))
            r_pack = self.pack_resistance(soc_pct)
            v_unclamped = pack_ocv - pack_current_a * r_pack
            floor = self.config.cell_voltage_min_v * self.config.series
            if v_unclamped < floor:
                cell_v_implied = v_unclamped / self.config.series
                self.violations.append(BatteryViolation(
                    kind="voltage_floor",
                    time_s=float(time_s) if time_s is not None else float("nan"),
                    soc_pct=float(soc_pct),
                    cell_voltage_v=float(cell_v_implied),
                    pack_current_a=float(pack_current_a),
                    message=(
                        f"Predicted pack voltage {v_unclamped:.1f} V below floor "
                        f"{floor:.1f} V at SOC={soc_pct:.2f}%, I_pack={pack_current_a:.2f} A"
                    ),
                ))
                return floor
            return v_unclamped

        cell_i = pack_current_a / self.config.parallel
        cell_v = self.cell_voltage(
            soc_pct, cell_i,
            time_s=time_s, pack_current_a=pack_current_a,
        )
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

        # Voltage floor: I_max_cell = (OCV - V_min) / R.
        # NF-16: ``self.ocv`` already floors extrapolated OCV at
        # ``cell_voltage_min_v + 0.01 V`` so ``v_headroom`` stays
        # non-negative (and ≥ 0.01 V).
        if self._calibrated:
            cell_ocv = self.ocv(soc_pct)
            r = self.internal_resistance(soc_pct)
            if r > 0:
                v_headroom = max(0.0, cell_ocv - self.config.cell_voltage_min_v)
                voltage_limit_pack = (v_headroom / r) * self.config.parallel
            else:
                voltage_limit_pack = float("inf")
        else:
            voltage_limit_pack = float("inf")

        return max(0.0, min(soc_limit, voltage_limit_pack))

    # ------------------------------------------------------------------
    # Pack energy / thermal mass
    # ------------------------------------------------------------------

    @property
    def pack_energy_kwh_nominal(self) -> float:
        """Nominal pack energy at ``cell_voltage_max_v`` fully charged."""
        return (
            self.pack_capacity_ah
            * self.config.cell_voltage_max_v
            * self.config.series
            / 1000.0
        )

    @property
    def thermal_mass_j_per_k(self) -> float:
        """Lumped thermal mass of the pack (J/K).

        S15: includes both cell thermal mass and structural (busbars,
        compression plates, enclosure) via
        ``config.pack_structural_thermal_mass_kj_per_k``.
        """
        num_cells = self.config.series * self.config.parallel
        cell_j_per_k = (
            self.CELL_MASS_KG * self.CELL_SPECIFIC_HEAT_J_PER_KG_K * num_cells
        )
        struct_j_per_k = (
            self.config.pack_structural_thermal_mass_kj_per_k * 1000.0
        )
        return cell_j_per_k + struct_j_per_k

    # ------------------------------------------------------------------
    # State stepping
    # ------------------------------------------------------------------

    def step(
        self,
        pack_current_a: float,
        dt_s: float,
        soc_pct: float,
        temp_c: float,
        *,
        time_s: float | None = None,
    ) -> tuple[float, float, float]:
        """Advance battery state by one timestep.

        Args:
            pack_current_a: Pack current (positive = discharge).
            dt_s: Timestep in seconds.
            soc_pct: Current state-of-charge (percent, 0-100).
            temp_c: Current mean cell temperature (Celsius).
            time_s: Optional simulation time (for violation events).

        Returns:
            (new_soc_pct, new_mean_cell_temp_c, pack_voltage_v)
        """
        # Coulomb counting: SOC change
        cell_current = pack_current_a / self.config.parallel
        dsoc = -(cell_current * dt_s) / (self.cell_capacity_ah * 3600) * 100
        new_soc = float(np.clip(soc_pct + dsoc, 0.0, 100.0))

        # Pack voltage at updated SOC
        v_pack = self.pack_voltage(new_soc, pack_current_a, time_s=time_s)

        # Thermal model: lumped I^2*R heating plus passive cooling.
        # S15: thermal mass includes structural components.
        # Newton cooling h·A·(T − T_ambient) prevents unbounded drift
        # during sustained discharge — without it the model has no
        # equilibrium and every long sim crosses the BMS kill temp.
        r_cell = self.internal_resistance(new_soc)
        num_cells = self.config.series * self.config.parallel
        heat_in_w = cell_current ** 2 * r_cell * num_cells
        heat_out_w = self.config.thermal_conductance_w_per_k * (
            temp_c - self.config.ambient_temperature_c
        )
        net_heat_w = heat_in_w - heat_out_w
        thermal_mass = self.thermal_mass_j_per_k
        dtemp = (net_heat_w * dt_s) / thermal_mass if thermal_mass > 0 else 0.0
        new_temp = temp_c + dtemp

        return new_soc, new_temp, v_pack
