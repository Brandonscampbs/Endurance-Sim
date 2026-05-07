"""End-to-end tests for the ``POST /api/simulate`` endpoint.

The full Michigan endurance run takes single-digit seconds on a developer
laptop; we still keep the heavy assertions consolidated into a small number
of tests and reuse the cached baseline + module-scoped TestClient so the
suite finishes quickly.

Network is never touched -- ``TestClient`` runs the FastAPI app in-process
against the cached telemetry CSV that ships with the repo.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.simulate_runner import get_baseline_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A single TestClient for the module so the FastAPI lifespan only fires once."""
    return TestClient(app)


@pytest.fixture(scope="module")
def baseline_payload(client: TestClient) -> dict:
    """Run one baseline-equivalent request and reuse its payload across tests.

    Also priming ``get_baseline_summary``'s lru_cache here means downstream
    tests don't pay for the baseline twice.
    """
    # Make sure the cached baseline summary is warm before we time anything.
    get_baseline_summary()

    start = time.perf_counter()
    response = client.post("/api/simulate", json=_baseline_request_dict())
    elapsed = time.perf_counter() - start
    assert response.status_code == 200, response.text

    payload = response.json()
    payload["__elapsed_s"] = elapsed  # not part of the schema; for diagnostics
    return payload


def _baseline_request_dict() -> dict:
    """Request body that should reproduce (or come very close to) the baseline."""
    return {
        "max_rpm": 3000.0,
        "max_torque_nm": 85.0,
        "soc_discharge_map": [
            {"soc_pct": 100.0, "max_current_a": 100.0},
            {"soc_pct": 85.0, "max_current_a": 100.0},
            {"soc_pct": 50.0, "max_current_a": 65.0},
            {"soc_pct": 20.0, "max_current_a": 35.0},
            {"soc_pct": 5.0, "max_current_a": 10.0},
            {"soc_pct": 0.0, "max_current_a": 0.0},
        ],
    }


# ---------------------------------------------------------------------------
# Schema and happy-path tests (single combined test to keep wall-clock down)
# ---------------------------------------------------------------------------


class TestBaselineEquivalent:
    """Cover the schema, baseline payload, and determinism in one fixture run."""

    def test_status_code_200(self, baseline_payload: dict) -> None:
        # Already asserted in the fixture but kept explicit for readability.
        assert "summary" in baseline_payload

    def test_summary_fields_populate(self, baseline_payload: dict) -> None:
        summary = baseline_payload["summary"]
        assert summary["total_time_s"] > 0
        # Either positive (net discharge) or close to zero -- never NaN.
        assert summary["net_ah"] == summary["net_ah"]  # NaN check
        assert summary["target_laps"] >= 1
        assert 0.0 <= summary["final_soc_pct"] <= 100.0

    def test_baseline_block_is_populated(self, baseline_payload: dict) -> None:
        baseline = baseline_payload["baseline"]
        assert baseline["total_time_s"] > 0
        assert baseline["target_laps"] == baseline_payload["summary"]["target_laps"]

    def test_per_lap_table_shape(self, baseline_payload: dict) -> None:
        laps = baseline_payload["laps"]
        assert len(laps) == baseline_payload["summary"]["completed_laps"]
        first = laps[0]
        for key in (
            "lap_number",
            "time_s",
            "charge_used_ah",
            "energy_used_kwh",
            "mean_speed_kmh",
            "peak_speed_kmh",
            "end_soc_pct",
        ):
            assert key in first
        # 1-indexed lap numbers so the UI doesn't have to translate.
        assert first["lap_number"] == 1

    def test_time_series_downsampled(self, baseline_payload: dict) -> None:
        ts = baseline_payload["time_series"]
        # Target downsample is ~2000 points; allow a small slack either side
        # because the engine emits fewer points for short runs and we cap at n.
        assert 1 <= len(ts["distance_m"]) <= 2000
        # All channels share the same length.
        n = len(ts["distance_m"])
        for channel in (
            "speed_kmh",
            "pack_power_w",
            "pack_current_a",
            "soc_pct",
            "cumulative_charge_ah",
        ):
            assert len(ts[channel]) == n

    def test_baseline_matches_cached_aggregate(
        self, baseline_payload: dict
    ) -> None:
        """The 'baseline' block must equal whatever ``get_baseline_summary`` exposes.

        Both call ``get_baseline_result`` under the hood; differences here
        would mean the summariser is non-deterministic or rounding leaked in
        somewhere unexpected.
        """
        cached_aggregate, _, _, _ = get_baseline_summary()
        cached = cached_aggregate.model_dump()
        api = baseline_payload["baseline"]
        for key, val in cached.items():
            assert api[key] == pytest.approx(val, abs=1e-6), key

    def test_baseline_request_echoed(self, baseline_payload: dict) -> None:
        assert baseline_payload["request"]["max_rpm"] == 3000.0
        assert baseline_payload["request"]["max_torque_nm"] == 85.0


class TestDeterminism:
    """Same input must produce the same output across two calls."""

    def test_two_calls_match(
        self, client: TestClient, baseline_payload: dict
    ) -> None:
        second = client.post("/api/simulate", json=_baseline_request_dict())
        assert second.status_code == 200, second.text
        second_payload = second.json()

        # Compare summary numbers (most condensed determinism signal).
        first_summary = baseline_payload["summary"]
        second_summary = second_payload["summary"]
        for key, val in first_summary.items():
            assert second_summary[key] == pytest.approx(val, abs=1e-6), key

        # Per-lap numbers should also be bit-stable.
        for first_lap, second_lap in zip(
            baseline_payload["laps"], second_payload["laps"]
        ):
            for key in (
                "time_s",
                "charge_used_ah",
                "energy_used_kwh",
                "mean_speed_kmh",
                "peak_speed_kmh",
            ):
                assert second_lap[key] == pytest.approx(
                    first_lap[key], abs=1e-6
                ), f"lap {first_lap['lap_number']} {key}"


# ---------------------------------------------------------------------------
# Override behaviour
# ---------------------------------------------------------------------------


class TestTorqueOverride:
    """Lower torque cap -> longer endurance time."""

    def test_lower_torque_yields_longer_time(
        self, client: TestClient, baseline_payload: dict
    ) -> None:
        body = _baseline_request_dict()
        body["max_torque_nm"] = 40.0
        response = client.post("/api/simulate", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()

        baseline_time = baseline_payload["baseline"]["total_time_s"]
        override_time = payload["summary"]["total_time_s"]
        assert override_time > baseline_time, (
            f"slower car (40 Nm) should take longer than baseline (85 Nm); "
            f"got override={override_time:.2f} s, baseline={baseline_time:.2f} s"
        )


class TestRpmOverride:
    """Lower RPM cap -> reduced peak speed."""

    def test_lower_rpm_reduces_peak_speed(
        self, client: TestClient, baseline_payload: dict
    ) -> None:
        body = _baseline_request_dict()
        body["max_rpm"] = 1500.0
        response = client.post("/api/simulate", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()

        baseline_peak = max(
            lap["peak_speed_kmh"] for lap in baseline_payload["baseline_laps"]
        )
        override_peak = max(
            lap["peak_speed_kmh"] for lap in payload["laps"]
        )
        assert override_peak < baseline_peak, (
            f"capping motor RPM at 1500 should reduce peak speed below baseline; "
            f"got override={override_peak:.1f} km/h, baseline={baseline_peak:.1f} km/h"
        )


# ---------------------------------------------------------------------------
# Validation (422) cases
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """All bad-input shapes should produce 422 from the Pydantic layer."""

    def test_max_rpm_too_low(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["max_rpm"] = 500.0  # below the 1000 floor
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_max_rpm_too_high(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["max_rpm"] = 9000.0  # above the 6500 ceiling
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_max_torque_too_low(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["max_torque_nm"] = 5.0
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_max_torque_too_high(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["max_torque_nm"] = 500.0
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_too_few_soc_points(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["soc_discharge_map"] = [
            {"soc_pct": 100.0, "max_current_a": 100.0},
        ]
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_too_many_soc_points(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["soc_discharge_map"] = [
            {"soc_pct": 100.0 - i, "max_current_a": 100.0 - i}
            for i in range(11)
        ]
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_soc_not_descending(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["soc_discharge_map"] = [
            {"soc_pct": 50.0, "max_current_a": 65.0},
            {"soc_pct": 100.0, "max_current_a": 100.0},  # SOC INCREASED -- invalid
        ]
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_soc_current_non_monotonic(self, client: TestClient) -> None:
        # Walking high SOC -> low SOC, current INCREASES at one step, which
        # is invalid (low-SOC entries must draw the same or less current).
        body = _baseline_request_dict()
        body["soc_discharge_map"] = [
            {"soc_pct": 100.0, "max_current_a": 50.0},
            {"soc_pct": 80.0, "max_current_a": 100.0},  # current went UP
            {"soc_pct": 20.0, "max_current_a": 30.0},
        ]
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_soc_out_of_range(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["soc_discharge_map"] = [
            {"soc_pct": 120.0, "max_current_a": 100.0},  # > 100
            {"soc_pct": 50.0, "max_current_a": 50.0},
        ]
        assert client.post("/api/simulate", json=body).status_code == 422

    def test_current_out_of_range(self, client: TestClient) -> None:
        body = _baseline_request_dict()
        body["soc_discharge_map"] = [
            {"soc_pct": 100.0, "max_current_a": 250.0},  # > 200
            {"soc_pct": 50.0, "max_current_a": 50.0},
        ]
        assert client.post("/api/simulate", json=body).status_code == 422


# ---------------------------------------------------------------------------
# Sanity: SOC map approximation note populates only when the curve actually tapers
# ---------------------------------------------------------------------------


class TestSocApproximationNote:
    """The 2-knee taper helper is exercised here without a full sim run.

    Flat-curve / no-taper inputs should not trigger the explanatory note;
    the actual sim only need run once to verify a tapered input does. We
    piggyback on ``baseline_payload`` for the positive case to avoid paying
    for an additional 11 s endurance simulation.
    """

    def test_flat_curve_no_note(self) -> None:
        # Direct unit-test on the helper rather than a 200 response, to keep
        # wall-clock under control. SimulateRequest validation is identical
        # to the round-trip path -- the note is computed inside
        # _apply_battery_overrides regardless of how the request arrived.
        from backend.models.simulate import SimulateRequest
        from backend.services.simulate_runner import _approximate_soc_taper

        req = SimulateRequest(
            max_rpm=3000.0,
            max_torque_nm=85.0,
            soc_discharge_map=[
                {"soc_pct": 100.0, "max_current_a": 100.0},
                {"soc_pct": 0.0, "max_current_a": 100.0},
            ],
        )
        threshold, rate, used_approximation = _approximate_soc_taper(
            req.soc_discharge_map
        )
        assert used_approximation is False
        assert rate == 0.0

    def test_tapered_curve_emits_note(self, baseline_payload: dict) -> None:
        # The baseline body has a tapered SOC curve; expect a note mentioning
        # the 2-knee approximation.
        notes = baseline_payload["notes"]
        assert any("approximated" in note.lower() for note in notes), notes
