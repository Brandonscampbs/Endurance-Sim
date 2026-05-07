"""Pydantic v2 schemas for the ``POST /api/simulate`` endpoint.

The Simulate page in the webapp lets the user override three things on top
of the calibrated baseline endurance simulation:

* ``max_rpm`` -- the motor speed limit (``PowertrainConfig.motor_speed_max_rpm``)
* ``max_torque_nm`` -- the inverter torque ceiling
  (``PowertrainConfig.torque_limit_inverter_nm``)
* ``soc_discharge_map`` -- a piecewise SOC-vs-current ceiling that gets
  approximated to the 2-knee form expected by ``BatteryConfig``
  (``soc_taper_threshold_pct`` and ``soc_taper_rate_a_per_pct``).

The response carries both the override-result aggregates and the cached
baseline aggregates so the UI can render side-by-side comparisons without a
second round-trip.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SocCurrentPoint(BaseModel):
    """One knee in the user-specified SOC -> max-current curve.

    ``soc_pct`` is in the closed interval [0, 100]. ``max_current_a`` is the
    pack-level discharge ceiling at that SOC, in [0, 200] A.
    """

    model_config = ConfigDict(extra="forbid")

    soc_pct: float = Field(..., ge=0.0, le=100.0)
    max_current_a: float = Field(..., ge=0.0, le=200.0)


class SimulateRequest(BaseModel):
    """User-supplied overrides for a single endurance simulation run.

    Validation rules (enforced via Pydantic):

    * ``max_rpm`` in [1000, 6500]
    * ``max_torque_nm`` in [20, 230]
    * ``soc_discharge_map`` length in [2, 10]
    * SOC values strictly descending across the map
    * ``max_current_a`` is non-decreasing as ``soc_pct`` increases (i.e.
      low-SOC entries draw less or equal current than higher-SOC entries)
    * SOC and current values stay inside their per-point bounds
    """

    model_config = ConfigDict(extra="forbid")

    max_rpm: float = Field(..., ge=1000.0, le=6500.0)
    max_torque_nm: float = Field(..., ge=20.0, le=230.0)
    soc_discharge_map: List[SocCurrentPoint] = Field(..., min_length=2, max_length=10)

    @model_validator(mode="after")
    def _validate_soc_curve(self) -> "SimulateRequest":
        """Enforce monotonicity rules on the SOC discharge map."""
        socs = [pt.soc_pct for pt in self.soc_discharge_map]
        currents = [pt.max_current_a for pt in self.soc_discharge_map]

        # Strictly descending SOC values -- the UI presents the map with
        # high-SOC at the top and discharges down to low-SOC.
        for prev, curr in zip(socs, socs[1:]):
            if curr >= prev:
                raise ValueError(
                    "soc_discharge_map.soc_pct must be strictly descending; "
                    f"got {prev} followed by {curr}."
                )

        # Current must be non-decreasing as SOC increases. Walking the list
        # in storage order (high SOC -> low SOC) means current must be
        # non-INCREASING in that direction (low-SOC entries have less current).
        for prev, curr in zip(currents, currents[1:]):
            if curr > prev:
                raise ValueError(
                    "soc_discharge_map.max_current_a must be non-decreasing as "
                    "soc_pct increases (low-SOC entries should draw the same "
                    f"or less current than higher-SOC entries); got {prev} A "
                    f"followed by {curr} A while SOC was decreasing."
                )

        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SimulateLapResult(BaseModel):
    """Per-lap aggregate metrics extracted from the override sim states."""

    model_config = ConfigDict(extra="forbid")

    lap_number: int
    time_s: float
    charge_used_ah: float
    energy_used_kwh: float
    mean_speed_kmh: float
    peak_speed_kmh: float
    end_soc_pct: float


class SimulateTimeSeries(BaseModel):
    """Downsampled per-segment trace (target ~2000 points)."""

    model_config = ConfigDict(extra="forbid")

    distance_m: list[float]
    speed_kmh: list[float]
    pack_power_w: list[float]
    pack_current_a: list[float]
    soc_pct: list[float]
    cumulative_charge_ah: list[float]


class SimulateAggregate(BaseModel):
    """Run-level summary metrics."""

    model_config = ConfigDict(extra="forbid")

    total_time_s: float
    net_ah: float
    net_kwh: float
    completed_laps: int
    target_laps: int
    final_soc_pct: float


class SimulateResponse(BaseModel):
    """Full payload returned by ``POST /api/simulate``.

    ``baseline_*`` mirrors the same shape as ``summary``/``laps``/
    ``time_series`` for the cached calibrated baseline so the UI can show
    deltas without making a second request.

    ``notes`` carries any caveats the runner had to record -- most notably
    when the user-supplied SOC curve was approximated to the 2-knee taper
    that ``BatteryConfig`` actually models.
    """

    model_config = ConfigDict(extra="forbid")

    request: SimulateRequest
    summary: SimulateAggregate
    laps: list[SimulateLapResult]
    time_series: SimulateTimeSeries
    baseline: SimulateAggregate
    baseline_laps: list[SimulateLapResult]
    baseline_time_series: SimulateTimeSeries
    notes: list[str] = Field(default_factory=list)
