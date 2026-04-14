"""Tests for FSAE Endurance + Efficiency scoring function.

Validates scoring formulas against 2025 Michigan competition results
and FSAE Rules D.12.13 / D.13.4.
"""

import math

import pytest

from fsae_sim.analysis.scoring import (
    CompetitionField,
    FSAEScoreResult,
    FSAEScoring,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def michigan_2025():
    """Pre-configured scorer with 2025 Michigan field data."""
    return FSAEScoring.michigan_2025_field()


@pytest.fixture
def field_2025():
    """Raw 2025 Michigan field parameters."""
    return CompetitionField(
        endurance_tmin_s=1369.936,
        efficiency_co2min_kg_per_lap=0.0967,
        efficiency_efmax=0.848,
    )


# ---------------------------------------------------------------------------
# Endurance time score (D.12.13.2)
# ---------------------------------------------------------------------------

class TestEnduranceTimeScore:
    def test_fastest_team_gets_max(self, michigan_2025):
        """Team matching Tmin gets 250 points."""
        result = michigan_2025.score(
            total_time_s=1369.936, total_energy_kwh=2.0,
            laps_completed=22,
        )
        assert result.endurance_time_score == pytest.approx(250.0, abs=0.1)

    def test_slower_team_gets_partial(self, michigan_2025):
        """A team slower than Tmin but under Tmax gets partial score."""
        # Tmax = 1.45 * 1369.936 = 1986.407
        mid_time = (1369.936 + 1986.407) / 2.0
        result = michigan_2025.score(
            total_time_s=mid_time, total_energy_kwh=2.0,
            laps_completed=22,
        )
        assert 0 < result.endurance_time_score < 250

    def test_at_tmax_gets_zero(self, michigan_2025):
        """Team at exactly Tmax gets 0 time score."""
        tmax = 1.45 * 1369.936
        result = michigan_2025.score(
            total_time_s=tmax, total_energy_kwh=2.0,
            laps_completed=22,
        )
        assert result.endurance_time_score == pytest.approx(0.0, abs=0.1)

    def test_beyond_tmax_gets_zero(self, michigan_2025):
        """Team slower than Tmax still gets 0."""
        result = michigan_2025.score(
            total_time_s=2500.0, total_energy_kwh=2.0,
            laps_completed=22,
        )
        assert result.endurance_time_score == 0.0


# ---------------------------------------------------------------------------
# Endurance laps score (D.12.13.3)
# ---------------------------------------------------------------------------

class TestEnduranceLapsScore:
    def test_full_22_laps(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.0,
            laps_completed=22,
        )
        # 11 laps * 1pt + 3pt driver change + 10 laps * 1pt = 25
        assert result.endurance_laps_score == 25.0

    def test_partial_first_stint(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=700.0, total_energy_kwh=1.0,
            laps_completed=8,
        )
        assert result.endurance_laps_score == 8.0

    def test_exactly_11_laps_no_change(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=750.0, total_energy_kwh=1.0,
            laps_completed=11,
        )
        assert result.endurance_laps_score == 11.0

    def test_12_laps_includes_driver_change(self, michigan_2025):
        """Completing lap 12 = 1pt (lap) + 3pt (driver change bonus) = 4pt."""
        result = michigan_2025.score(
            total_time_s=850.0, total_energy_kwh=1.5,
            laps_completed=12,
        )
        # 11 * 1 + 1 (lap 12) + 3 (bonus) = 15
        assert result.endurance_laps_score == 15.0

    def test_zero_laps(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=0.0, total_energy_kwh=0.0,
            laps_completed=0,
        )
        assert result.endurance_laps_score == 0.0


# ---------------------------------------------------------------------------
# Efficiency scoring (D.13.4)
# ---------------------------------------------------------------------------

class TestEfficiencyScore:
    def test_uconn_2025_perfect_efficiency(self, michigan_2025):
        """UConn won efficiency at Michigan 2025 with 100 pts.

        UConn's actual results: ~1859s driving time, ~2.13 kWh energy, 22 laps.
        CO2 = 2.13 * 0.65 = 1.3845 kg. CO2/lap = 0.0629.
        But they set CO2min = 0.0967 in the field (this is what the official
        results say). EFmax = 0.848.
        """
        # If UConn IS the CO2min and EFmax team, scoring themselves should
        # give 100 points (or close, depending on formula)
        result = michigan_2025.score(
            total_time_s=1859.0, total_energy_kwh=2.13,
            laps_completed=22,
        )
        # UConn had EF = 0.848 which IS EFmax, so score = 100
        assert result.efficiency_score == pytest.approx(100.0, abs=5.0)

    def test_ineligible_under_12_laps(self, michigan_2025):
        """Teams with <=11 laps are not eligible for efficiency."""
        result = michigan_2025.score(
            total_time_s=700.0, total_energy_kwh=0.5,
            laps_completed=11,
        )
        assert result.efficiency_score == 0.0

    def test_high_energy_low_efficiency(self, michigan_2025):
        """Team using lots of energy gets lower efficiency."""
        result_efficient = michigan_2025.score(
            total_time_s=1600.0, total_energy_kwh=2.0,
            laps_completed=22,
        )
        result_wasteful = michigan_2025.score(
            total_time_s=1600.0, total_energy_kwh=5.0,
            laps_completed=22,
        )
        assert result_efficient.efficiency_score > result_wasteful.efficiency_score


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------

class TestCombinedScore:
    def test_combined_is_sum(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.5,
            laps_completed=22,
        )
        expected = result.endurance_total + result.efficiency_score
        assert result.combined_score == pytest.approx(expected)

    def test_endurance_total_is_time_plus_laps(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.5,
            laps_completed=22,
        )
        expected = result.endurance_time_score + result.endurance_laps_score
        assert result.endurance_total == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Penalty handling
# ---------------------------------------------------------------------------

class TestPenalties:
    def test_cone_penalties_add_time(self, michigan_2025):
        base = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.5,
            laps_completed=22, cone_penalties=0,
        )
        penalized = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.5,
            laps_completed=22, cone_penalties=5,
        )
        # Each cone = 2 seconds
        assert penalized.your_time_s == pytest.approx(base.your_time_s + 10.0)
        assert penalized.endurance_time_score < base.endurance_time_score

    def test_off_course_penalties(self, michigan_2025):
        base = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.5,
            laps_completed=22, off_course_penalties=0,
        )
        penalized = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=2.5,
            laps_completed=22, off_course_penalties=2,
        )
        # Each off-course = 20 seconds
        assert penalized.your_time_s == pytest.approx(base.your_time_s + 40.0)


# ---------------------------------------------------------------------------
# CO2 and energy fields in result
# ---------------------------------------------------------------------------

class TestResultFields:
    def test_co2_calculation(self, michigan_2025):
        result = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=3.0,
            laps_completed=22,
        )
        assert result.your_co2_kg == pytest.approx(3.0 * 0.65)
        assert result.your_energy_kwh == 3.0
        assert result.your_time_s == 1500.0
        assert result.your_avg_lap_s == pytest.approx(1500.0 / 22.0)
        assert result.your_co2_per_lap == pytest.approx(3.0 * 0.65 / 22.0)


# ---------------------------------------------------------------------------
# Convenience: score_sim_result
# ---------------------------------------------------------------------------

class TestScoreSimResult:
    def test_scores_sim_result(self, michigan_2025):
        """score_sim_result extracts fields from SimResult correctly."""
        # Use a mock-like object with the needed attributes
        class FakeSimResult:
            total_time_s = 1600.0
            total_energy_kwh = 2.5
            laps_completed = 22

        result = michigan_2025.score_sim_result(
            FakeSimResult(),
            track_distance_km=1.0,
        )
        assert isinstance(result, FSAEScoreResult)
        assert result.your_time_s == 1600.0
        assert result.your_energy_kwh == 2.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_energy_raises_or_handles(self, michigan_2025):
        """Zero energy shouldn't crash — efficiency = infinite."""
        result = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=0.0,
            laps_completed=22,
        )
        # Should handle gracefully (max efficiency or capped)
        assert result.efficiency_score >= 0

    def test_ef_exceeding_field_max(self, michigan_2025):
        """If your EF exceeds the field EFmax, you become the new max -> 100 pts."""
        # Very fast, very efficient
        result = michigan_2025.score(
            total_time_s=1370.0, total_energy_kwh=1.0,
            laps_completed=22,
        )
        assert result.efficiency_score == pytest.approx(100.0, abs=1.0)

    def test_co2_eligibility_cap(self, michigan_2025):
        """Teams exceeding 20.02 kg CO2/100km are ineligible for efficiency."""
        # 22 laps of ~1km = 22 km. 20.02 kg/100km => 4.404 kg max CO2.
        # CO2 = energy * 0.65. So max energy ~ 6.77 kWh.
        # Use way more than that:
        result = michigan_2025.score(
            total_time_s=1500.0, total_energy_kwh=50.0,
            laps_completed=22,
            total_distance_km=22.0,
        )
        assert result.efficiency_score == 0.0
