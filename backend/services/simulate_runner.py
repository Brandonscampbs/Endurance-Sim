"""Orchestrate one ``POST /api/simulate`` run and summarise the result.

The Simulate page hands us three knobs (max RPM, max torque, SOC->current
map). We translate them onto the existing ``VehicleConfig`` via
``dataclasses.replace`` (configs are frozen) and reuse the same
``SimulationEngine``/``CalibratedStrategy`` wiring as
``backend.services.sim_runner.get_baseline_result``.

Caching:
    * ``get_baseline_summary`` is cached so the cached
      ``get_baseline_result`` and the derived summary tuple are computed
      exactly once across the process lifetime.
    * The override branch is intentionally **not** memoised yet -- the
      heavy work is the SimulationEngine itself, not the summary, and
      profiling on a developer laptop puts a single endurance run inside
      the < 10 s budget the UI's "Running endurance sim..." overlay
      assumes. ``run_overridden_simulation`` is documented as the place
      to add ``lru_cache`` keyed on a tuple representation of the request
      if that ever changes.

Schema mapping notes (kept here, not in the response, because they affect
the *override* branch only and would otherwise force a second response field
just to surface them):

* ``max_rpm`` overrides ``PowertrainConfig.motor_speed_max_rpm``.  The
  firmware-style overspeed cutoff (``lvcu_overspeed_rpm``) is clamped up to
  the new ceiling so a legitimate higher-RPM setpoint isn't immediately
  bounced by the LVCU's hard torque override.
* ``max_torque_nm`` overrides ``PowertrainConfig.torque_limit_inverter_nm``
  -- the inverter cap is what binds in practice (LVCU ceiling is higher).
* ``soc_discharge_map`` is approximated to ``BatteryConfig``'s 2-knee
  taper. The threshold is the highest SOC where current first dips below
  the curve's maximum; the rate is the average slope across the points
  where current is still decreasing (in A per percent of SOC).  When the
  approximation is in effect we add an explanatory line to the response's
  ``notes`` array so the UI can show the tooltip described in the spec.
"""
from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from fsae_sim.analysis.validation import charge_counters_ah
from fsae_sim.driver.strategies import CalibratedStrategy
from fsae_sim.sim.engine import SimResult, SimulationEngine, SimulationMode

from backend.models.simulate import (
    SimulateAggregate,
    SimulateLapResult,
    SimulateRequest,
    SimulateTimeSeries,
    SocCurrentPoint,
)
from backend.services.sim_runner import (
    _detected_lap_count,
    get_baseline_result,
    get_battery_model,
    get_track,
    get_validation_lap_split,
    get_vehicle_config,
)
from backend.services.telemetry_service import get_telemetry

_logger = logging.getLogger(__name__)

# Aim for ~2000 points in the time-series trace; the simulator typically
# emits ~30 k segment rows for a Michigan endurance, so this keeps the
# payload light enough for Plotly to render smoothly.
_TIME_SERIES_TARGET_POINTS = 2000


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------


def _apply_powertrain_overrides(
    powertrain,
    max_rpm: float,
    max_torque_nm: float,
):
    """Apply the user's RPM / torque caps onto a frozen PowertrainConfig.

    ``motor_speed_max_rpm`` is the headline override -- the available-torque
    curve drops to zero above this RPM.

    ``lvcu_overspeed_rpm`` (the LVCU's torque-override threshold consulted
    by ``lvcu_torque_ceiling`` and therefore by both the calibrated strategy
    and the speed envelope) is set to ``max_rpm`` as well.  Two reasons:

    1. The calibrated strategy never consults ``available_torque`` directly;
       it walks through ``lvcu_torque_ceiling``.  Without aligning the LVCU
       overspeed value, an override like ``max_rpm=1500`` would have no
       observable effect on a calibrated run because the LVCU's overspeed
       threshold would still sit at the firmware default 6000.
    2. The user's intent in the Simulate page is "cap the motor at this
       RPM"; aligning both knobs keeps that intent honest end-to-end.
    """
    return dataclasses.replace(
        powertrain,
        motor_speed_max_rpm=max_rpm,
        torque_limit_inverter_nm=max_torque_nm,
        lvcu_overspeed_rpm=max_rpm,
    )


def _approximate_soc_taper(
    soc_map: list[SocCurrentPoint],
) -> tuple[float, float, bool]:
    """Reduce the user's piecewise SOC->current map to the 2-knee taper.

    Returns ``(threshold_pct, rate_a_per_pct, used_approximation)`` where
    ``used_approximation`` is True when the curve actually has a tapering
    region. A flat curve (all points the same current) returns the full SOC
    span as the threshold and a 0 rate -- the BMS effectively never tapers,
    matching the user's intent.

    Algorithm:
        * threshold = highest SOC value where the corresponding current is
          strictly less than the curve's maximum (i.e. the first point
          where the BMS has already started backing off).
        * rate = average of |slope| across consecutive points where current
          is decreasing (in the storage order high SOC -> low SOC, current
          decreases as SOC drops).
    """
    socs = np.array([p.soc_pct for p in soc_map], dtype=float)
    currents = np.array([p.max_current_a for p in soc_map], dtype=float)
    max_current = float(currents.max())

    # Locate the highest-SOC point where current first drops below the peak.
    below_peak = currents < max_current - 1e-9
    if not below_peak.any():
        # Flat curve: no taper at all.  Threshold at the lowest SOC in the
        # map so the existing ``soc <= threshold`` check inside the battery
        # model is effectively never true within the run, and rate=0.
        return float(socs.min()), 0.0, False

    # Walk from the high-SOC end (index 0) until we find the first below-peak
    # point.  The threshold is the SOC just above that point -- i.e. the last
    # SOC at which current was still at peak.
    first_taper_idx = int(np.argmax(below_peak))
    if first_taper_idx == 0:
        # Curve starts already tapered; use the highest SOC as the threshold.
        threshold = float(socs[0])
    else:
        threshold = float(socs[first_taper_idx - 1])

    # Average |slope| across the descending region (current decreasing as we
    # walk to lower SOC).  Slopes are in A per % of SOC.  We restrict to
    # consecutive pairs where the current actually drops to avoid pulling
    # the average toward 0 with a flat tail.
    slopes: list[float] = []
    for i in range(len(soc_map) - 1):
        d_soc = socs[i] - socs[i + 1]  # positive -- we're descending in SOC
        d_current = currents[i] - currents[i + 1]  # positive when tapering
        if d_soc > 1e-9 and d_current > 1e-9:
            slopes.append(d_current / d_soc)
    rate = float(np.mean(slopes)) if slopes else 0.0
    return threshold, rate, True


def _apply_battery_overrides(
    battery,
    soc_map: list[SocCurrentPoint],
) -> tuple[object, str | None]:
    """Override the battery's SOC taper with the 2-knee approximation.

    Returns the new ``BatteryConfig`` plus an optional note describing the
    approximation when it was actually applied.
    """
    threshold, rate, used_approximation = _approximate_soc_taper(soc_map)
    new_battery = dataclasses.replace(
        battery,
        soc_taper_threshold_pct=threshold,
        soc_taper_rate_a_per_pct=rate,
    )
    note: str | None = None
    if used_approximation:
        note = (
            "SOC discharge map approximated to a 2-knee taper "
            f"(threshold={threshold:.1f}%, rate={rate:.2f} A per %). The "
            "BatteryModel currently exposes a single threshold/rate pair; "
            "the full piecewise curve is summarised by these two values."
        )
    return new_battery, note


# ---------------------------------------------------------------------------
# Simulation entry points
# ---------------------------------------------------------------------------


def run_overridden_simulation(req: SimulateRequest) -> tuple[SimResult, list[str]]:
    """Run the endurance sim with the user's overrides applied.

    Mirrors the wiring in ``sim_runner.get_baseline_result`` so the override
    result is comparable to the baseline (same strategy, same track, same
    initial conditions). Returns ``(result, notes)`` where ``notes`` carries
    any caveats added during override application (e.g. the SOC map
    approximation).
    """
    base_vehicle = get_vehicle_config()
    track = get_track()
    battery_model = get_battery_model()
    aim_df = get_telemetry()
    num_laps = _detected_lap_count(aim_df)
    _, validation_laps = get_validation_lap_split(num_laps)

    new_powertrain = _apply_powertrain_overrides(
        base_vehicle.powertrain,
        req.max_rpm,
        req.max_torque_nm,
    )
    new_battery_cfg, soc_note = _apply_battery_overrides(
        base_vehicle.battery,
        req.soc_discharge_map,
    )
    overridden_vehicle = dataclasses.replace(
        base_vehicle,
        powertrain=new_powertrain,
        battery=new_battery_cfg,
    )

    # The cached BatteryModel was built around the baseline BatteryConfig.
    # Swap in the overridden config so the discharge taper is honoured at
    # runtime, but reuse the calibrated OCV / IR map (it depends on cell
    # chemistry, not on the discharge ceiling).  BatteryModel is a regular
    # class, not a frozen dataclass, so we shallow-copy and re-bind the
    # config to keep the cached singleton untouched.
    overridden_battery_model = _swap_battery_config(battery_model, new_battery_cfg)

    strategy = CalibratedStrategy.from_telemetry(
        aim_df,
        track,
        holdout_laps=list(validation_laps),
        use_observed_speed_caps=False,
    )
    engine = SimulationEngine(
        overridden_vehicle,
        track,
        strategy,
        overridden_battery_model,
        mode=SimulationMode.VALIDATION,
    )
    result = engine.run(num_laps=num_laps, initial_soc_pct=95.0, initial_temp_c=29.0)

    notes: list[str] = []
    if soc_note is not None:
        notes.append(soc_note)
    return result, notes


def _swap_battery_config(battery_model, new_battery_cfg):
    """Return a battery model whose ``.config`` reflects the overrides.

    BatteryModel is a regular class (not a frozen dataclass); we make a
    shallow copy with ``copy.copy`` to avoid mutating the cached singleton
    that other callers (the validation page, the visualization page) rely
    on.  All calibrated state -- the OCV table, the internal-resistance
    map, the temperature model -- remains intact.
    """
    import copy

    new_model = copy.copy(battery_model)
    new_model.config = new_battery_cfg
    return new_model


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _build_aggregate(
    result: SimResult,
    target_laps: int,
) -> SimulateAggregate:
    """Roll a SimResult up into the run-level summary card payload."""
    return SimulateAggregate(
        total_time_s=round(float(result.total_time_s), 3),
        net_ah=round(float(result.net_charge_ah), 3),
        net_kwh=round(float(result.net_energy_kwh), 3),
        completed_laps=int(result.laps_completed),
        target_laps=int(target_laps),
        final_soc_pct=round(float(result.final_soc), 1),
    )


def _build_lap_summaries(result: SimResult) -> list[SimulateLapResult]:
    """Per-lap timing / charge / energy table.

    Mirrors ``backend.services.validation_export.get_all_laps_summary`` but
    drops the real-telemetry comparison columns -- the Simulate page only
    needs sim-side numbers.
    """
    states = result.states
    if states.empty:
        return []

    lap_results: list[SimulateLapResult] = []
    unique_laps = sorted(int(lap) for lap in states["lap"].unique())
    for lap_idx in unique_laps:
        lap_df = states[states["lap"] == lap_idx]
        if lap_df.empty:
            continue

        time_s = float(lap_df["segment_time_s"].sum())

        current = lap_df["pack_current_a"].values
        dt = lap_df["segment_time_s"].values
        _, _, net_ah = charge_counters_ah(current, dt)

        power = lap_df["electrical_power_w"].values
        net_kwh = float(np.sum(power * dt)) / 3_600_000.0

        speed = lap_df["speed_kmh"].values
        mean_speed = float(np.mean(speed)) if len(speed) else 0.0
        peak_speed = float(np.max(speed)) if len(speed) else 0.0

        end_soc = float(lap_df["soc_pct"].iloc[-1])

        lap_results.append(
            SimulateLapResult(
                lap_number=int(lap_idx) + 1,  # back to 1-indexed for the UI
                time_s=round(time_s, 3),
                charge_used_ah=round(net_ah, 3),
                energy_used_kwh=round(net_kwh, 3),
                mean_speed_kmh=round(mean_speed, 1),
                peak_speed_kmh=round(peak_speed, 1),
                end_soc_pct=round(end_soc, 1),
            )
        )
    return lap_results


def _build_time_series(result: SimResult) -> SimulateTimeSeries:
    """Downsample the per-segment state log to ~2000 points.

    We use ``np.linspace`` indexing rather than rolling-window decimation so
    the trace preserves the absolute distance grid (handy for the time-axis
    sync on the frontend overlays).
    """
    states = result.states
    if states.empty:
        return SimulateTimeSeries(
            distance_m=[],
            speed_kmh=[],
            pack_power_w=[],
            pack_current_a=[],
            soc_pct=[],
            cumulative_charge_ah=[],
        )

    n = len(states)
    if n <= _TIME_SERIES_TARGET_POINTS:
        idx = np.arange(n, dtype=int)
    else:
        idx = np.linspace(0, n - 1, _TIME_SERIES_TARGET_POINTS, dtype=int)

    sampled = states.iloc[idx]

    # Cumulative charge in Ah (signed: positive = pack discharging). We
    # accumulate over the full trace and then index, so the sampled trace
    # still has monotone-increasing energy at each point.
    full_dt = states["segment_time_s"].values
    full_current = states["pack_current_a"].values
    cum_charge_full = np.cumsum(full_current * full_dt) / 3600.0
    cum_charge_sampled = cum_charge_full[idx]

    return SimulateTimeSeries(
        distance_m=[round(float(v), 3) for v in sampled["distance_m"].values],
        speed_kmh=[round(float(v), 1) for v in sampled["speed_kmh"].values],
        pack_power_w=[round(float(v), 3) for v in sampled["electrical_power_w"].values],
        pack_current_a=[round(float(v), 3) for v in sampled["pack_current_a"].values],
        soc_pct=[round(float(v), 1) for v in sampled["soc_pct"].values],
        cumulative_charge_ah=[round(float(v), 3) for v in cum_charge_sampled],
    )


def summarize_run(
    result: SimResult,
    num_laps: int,
) -> tuple[SimulateAggregate, list[SimulateLapResult], SimulateTimeSeries]:
    """Bundle the three response fields produced from a single SimResult."""
    aggregate = _build_aggregate(result, target_laps=num_laps)
    laps = _build_lap_summaries(result)
    time_series = _build_time_series(result)
    return aggregate, laps, time_series


@lru_cache(maxsize=1)
def get_baseline_summary() -> tuple[SimulateAggregate, list[SimulateLapResult], SimulateTimeSeries, int]:
    """Run the cached calibrated baseline and bundle its summary.

    Cached so the summary serialisation (per-lap loop + downsample) runs
    exactly once. ``get_baseline_result`` is itself cached; this layer just
    avoids re-doing the per-lap/time-series work for every Simulate call.
    """
    aim_df = get_telemetry()
    num_laps = _detected_lap_count(aim_df)
    result = get_baseline_result()
    aggregate, laps, time_series = summarize_run(result, num_laps)
    return aggregate, laps, time_series, num_laps
