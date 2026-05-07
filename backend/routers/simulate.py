"""Router for the ``POST /api/simulate`` endpoint.

Accepts a :class:`backend.models.simulate.SimulateRequest` body, runs the
overridden endurance simulation, and returns a
:class:`backend.models.simulate.SimulateResponse` carrying both the override
and cached-baseline aggregates / per-lap tables / time-series traces.

Validation errors (out-of-range params, malformed SOC map) propagate to the
422 handler registered in :mod:`backend.errors`. Any unexpected runtime
error inside the simulator is caught and re-raised as ``HTTPException(500)``
so the structured ``http_exception_handler`` can format a request-id-tagged
JSON body.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.models.simulate import SimulateRequest, SimulateResponse
from backend.services.simulate_runner import (
    get_baseline_summary,
    run_overridden_simulation,
    summarize_run,
)

router = APIRouter(prefix="/api", tags=["simulate"])

_logger = logging.getLogger(__name__)


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """Run the overridden endurance simulation and return a comparison payload."""
    _logger.info(
        "simulate request: max_rpm=%.1f max_torque_nm=%.1f soc_points=%d",
        req.max_rpm,
        req.max_torque_nm,
        len(req.soc_discharge_map),
    )

    baseline_aggregate, baseline_laps, baseline_time_series, target_laps = (
        get_baseline_summary()
    )

    try:
        result, notes = run_overridden_simulation(req)
    except HTTPException:
        # If a caller (or downstream code) raised an HTTPException directly,
        # preserve its status code instead of wrapping it in a 500.
        raise
    except Exception as exc:  # noqa: BLE001 -- surface the raw simulator error
        _logger.exception("simulate run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    aggregate, laps, time_series = summarize_run(result, target_laps)
    return SimulateResponse(
        request=req,
        summary=aggregate,
        laps=laps,
        time_series=time_series,
        baseline=baseline_aggregate,
        baseline_laps=baseline_laps,
        baseline_time_series=baseline_time_series,
        notes=notes,
    )
