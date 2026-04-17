"""FSAE Endurance + Efficiency scoring function.

Implements the combined scoring formulas from FSAE Rules:
- Endurance: D.12.13 (time score + laps score, max 275 points)
- Efficiency: D.13.4 (energy efficiency factor, max 100 points)

Competition field parameters (Tmin, CO2min, EFmax) are configurable
to evaluate strategies against any competition scenario.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompetitionField:
    """Assumptions about the competition field for scoring."""

    endurance_tmin_s: float
    """Fastest team's corrected endurance time (seconds)."""

    efficiency_co2min_kg_per_lap: float
    """Most efficient team's CO2 per lap (kg/lap)."""

    efficiency_efmax: float
    """Highest efficiency factor in the field."""

    efficiency_tmin_laps: int = 22
    """Laps completed by the fastest team."""

    efficiency_co2min_laps: int = 22
    """Laps completed by the most efficient team."""


@dataclass
class FSAEScoreResult:
    """Complete scoring breakdown for one team's endurance run."""

    endurance_time_score: float
    """0-250 points from endurance time formula."""

    endurance_laps_score: float
    """0-25 points from laps completed."""

    endurance_total: float
    """Sum of time + laps score (0-275)."""

    efficiency_factor: float
    """Your efficiency factor (EF)."""

    efficiency_score: float
    """0-100 points from efficiency formula."""

    combined_score: float
    """endurance_total + efficiency_score (0-375)."""

    your_time_s: float
    """Your corrected endurance time (including penalties)."""

    your_energy_kwh: float
    """Total energy consumed."""

    your_co2_kg: float
    """CO2 equivalent (energy_kwh * CO2_PER_KWH_EV)."""

    your_avg_lap_s: float
    """Average lap time (corrected — includes penalties + driver change)."""

    your_co2_per_lap: float
    """CO2 per lap."""

    raw_avg_lap_s: float = 0.0
    """Raw average lap time (driving only, no penalties). Used for efficiency
    eligibility check per FSAE rules (D.13.4): raw pace is what the car is
    actually capable of; endurance time score uses corrected time."""


class FSAEScoring:
    """FSAE Endurance + Efficiency scorer.

    References:
        - D.12.13: Endurance scoring (time score + laps score)
        - D.13.4: Efficiency scoring (CO2 efficiency factor)
    """

    CO2_PER_KWH_EV = 0.65
    """kg CO2 per kWh for EVs (D.13.4.1c)."""

    ENDURANCE_TIME_MAX_FACTOR = 1.45
    """Tmax = 1.45 * Tmin (D.12.13.1)."""

    CONE_PENALTY_S = 2.0
    """Seconds added per cone hit."""

    OFF_COURSE_PENALTY_S = 20.0
    """Seconds added per off-course."""

    EV_CO2_MAX_PER_100KM = 20.02
    """kg CO2/100km eligibility cap for EVs (D.13.4.5)."""

    def __init__(self, field: CompetitionField) -> None:
        self.field = field

    @classmethod
    def michigan_2025_field(cls) -> FSAEScoring:
        """Pre-configured with 2025 Michigan competition field data."""
        return cls(CompetitionField(
            endurance_tmin_s=1369.936,
            efficiency_co2min_kg_per_lap=0.0967,
            efficiency_efmax=0.848,
            efficiency_tmin_laps=22,
            efficiency_co2min_laps=22,
        ))

    def score(
        self,
        total_time_s: float,
        total_energy_kwh: float,
        laps_completed: int,
        cone_penalties: int = 0,
        off_course_penalties: int = 0,
        total_distance_km: float | None = None,
        driver_change_time_s: float = 0.0,
        driver_change_completed: bool | None = None,
        track_km_per_lap: float | None = None,
    ) -> FSAEScoreResult:
        """Compute combined Endurance + Efficiency score.

        Args:
            total_time_s: Raw driving time (seconds).
            total_energy_kwh: Total electrical energy consumed.
            laps_completed: Number of laps finished.
            cone_penalties: Number of cones hit.
            off_course_penalties: Number of off-course incidents.
            total_distance_km: Total distance driven (km). If provided,
                used for CO2/100km eligibility check.
            driver_change_time_s: Seconds of driver-change stopped time
                added to the corrected endurance time (NF-43). Per FSAE
                D.12.13, the corrected endurance time includes the DC.
            driver_change_completed: Whether the driver change was actually
                completed (NF-44). The 3-point DC bonus in the laps score
                only applies when True. Defaults to ``laps_completed > 11``
                to preserve backward compatibility with the "completing
                lap 12 means DC happened" assumption.
            track_km_per_lap: Track length per lap in km (NF-59). Used to
                compute EFmin from the 20.02 kg CO2/100km eligibility cap.
                If None, EFmin defaults to 0 (conservative, gives full range).
        """
        f = self.field

        if driver_change_completed is None:
            # Backward-compat: completing lap 12 implied DC was done.
            driver_change_completed = laps_completed > 11

        # S11: raw vs corrected time. Efficiency uses raw pace (what the car
        # is capable of); endurance time score uses corrected (with penalties
        # and driver change).
        raw_time = float(total_time_s)
        corrected_time = (
            raw_time
            + float(driver_change_time_s)
            + cone_penalties * self.CONE_PENALTY_S
            + off_course_penalties * self.OFF_COURSE_PENALTY_S
        )

        # --- Endurance laps score (D.12.13.3) ---
        laps_score = self._laps_score(laps_completed, driver_change_completed)

        # --- Endurance time score (D.12.13.2) ---
        tmax = self.ENDURANCE_TIME_MAX_FACTOR * f.endurance_tmin_s
        if laps_completed > 0 and corrected_time < tmax:
            time_score = 250.0 * ((tmax / corrected_time) - 1.0) / (
                (tmax / f.endurance_tmin_s) - 1.0
            )
            time_score = max(0.0, min(250.0, time_score))
        else:
            time_score = 0.0

        endurance_total = time_score + laps_score

        # --- Efficiency (D.13.4) ---
        co2_yours = total_energy_kwh * self.CO2_PER_KWH_EV
        # S11: your_avg_lap_s is the corrected average (shown to users,
        # matches FSAE results sheets); raw_avg_lap_s is used internally for
        # efficiency pace checks where raw driving pace is what matters.
        avg_lap_s = corrected_time / laps_completed if laps_completed > 0 else 0.0
        raw_avg_lap_s = raw_time / laps_completed if laps_completed > 0 else 0.0
        co2_per_lap = co2_yours / laps_completed if laps_completed > 0 else 0.0

        efficiency_factor = 0.0
        efficiency_score = 0.0

        # Eligibility: must have completed driver change (NF-44 semantics)
        eligible = driver_change_completed and laps_completed > 0

        # Eligibility: average lap pace (raw) < 1.45 * fastest avg lap pace.
        # S11: compare raw pace; the 1.45x envelope is about the car's pace,
        # not penalty-inflated time.
        if eligible and laps_completed > 0:
            tmin_avg = f.endurance_tmin_s / f.efficiency_tmin_laps
            if raw_avg_lap_s >= self.ENDURANCE_TIME_MAX_FACTOR * tmin_avg:
                eligible = False

        # Eligibility: CO2/100km cap
        if eligible and total_distance_km is not None and total_distance_km > 0:
            co2_per_100km = co2_yours / total_distance_km * 100.0
            if co2_per_100km > self.EV_CO2_MAX_PER_100KM:
                eligible = False

        if eligible and laps_completed > 0:
            # Efficiency factor (D.13.4.4): uses raw pace
            tmin_avg = f.endurance_tmin_s / f.efficiency_tmin_laps
            your_avg = raw_avg_lap_s
            co2min_per_lap = f.efficiency_co2min_kg_per_lap

            if co2_per_lap > 0:
                efficiency_factor = (tmin_avg / your_avg) * (co2min_per_lap / co2_per_lap)
            else:
                # Zero energy => infinite efficiency; cap at a high value
                efficiency_factor = 100.0

            # If your EF exceeds field max, you become the new max
            efmax = max(f.efficiency_efmax, efficiency_factor)

            # EFmin: worst eligible team (D.13.4.6)
            # Uses CO2_max (20.02 kg/100km for EV) and Tyour = Tmax
            # Approximate: EFmin uses the slowest eligible time and worst energy
            tmax_avg = self.ENDURANCE_TIME_MAX_FACTOR * tmin_avg
            # For Michigan track (~1.0 km/lap), 20.02 kg/100km = 0.2002 kg/km
            # = 0.2002 kg/km * ~1.0 km/lap ≈ 0.2 kg/lap
            # But the exact formula: EFmin = (Tmin_avg / Tmax_avg) * (CO2min_per_lap / CO2max_per_lap)
            # where CO2max_per_lap depends on track length.
            # Per the rules, EFmin is computed from the maximum CO2 allowed.
            # We approximate using a generous CO2max per lap.
            # CO2max = 20.02 kg/100km. At ~1km/lap, that's 0.2002 kg/lap.
            # But we don't always know track distance. Use the field's CO2min
            # scaled approach: EFmin = (1/1.45) * (CO2min/CO2max_equiv)
            # For a conservative EFmin, use the formula directly:
            efmin_time_ratio = tmin_avg / tmax_avg  # = 1/1.45
            # CO2max per lap: worst case eligible. Without track distance,
            # assume the worst-case EFmin is small enough that it doesn't matter.
            # The spec says EFmin uses 20.02 kg/100km and Tyour=1.45*Tmin.
            # We need track distance to convert. If not provided, estimate
            # from laps and a reasonable per-lap distance.
            # NF-59: compute EFmin from CO2max = 20.02 kg/100km when track
            # length is known. Per D.13.4.6:
            #   EFmin = (Tmin_avg / Tmax_avg) * (CO2min_per_lap / CO2max_per_lap)
            if track_km_per_lap is not None and track_km_per_lap > 0:
                co2max_per_lap = (
                    self.EV_CO2_MAX_PER_100KM / 100.0 * track_km_per_lap
                )
                efmin = (
                    efmin_time_ratio * (co2min_per_lap / co2max_per_lap)
                    if co2max_per_lap > 0 else 0.0
                )
            else:
                efmin = 0.0  # Conservative: makes full range available

            if efmax > efmin:
                efficiency_score = 100.0 * (efficiency_factor - efmin) / (efmax - efmin)
                efficiency_score = max(0.0, min(100.0, efficiency_score))
            else:
                efficiency_score = 0.0

        combined = endurance_total + efficiency_score

        return FSAEScoreResult(
            endurance_time_score=time_score,
            endurance_laps_score=laps_score,
            endurance_total=endurance_total,
            efficiency_factor=efficiency_factor,
            efficiency_score=efficiency_score,
            combined_score=combined,
            your_time_s=corrected_time,
            your_energy_kwh=total_energy_kwh,
            your_co2_kg=co2_yours,
            your_avg_lap_s=avg_lap_s,
            your_co2_per_lap=co2_per_lap,
            raw_avg_lap_s=raw_avg_lap_s,
        )

    def score_sim_result(
        self,
        result,
        track_distance_km: float,
        cone_penalties: int = 0,
        off_course_penalties: int = 0,
        driver_change_time_s: float = 0.0,
        driver_change_completed: bool | None = None,
    ) -> FSAEScoreResult:
        """Score a SimResult directly.

        Args:
            result: A SimResult (or any object with total_time_s,
                total_energy_kwh, laps_completed attributes).
            track_distance_km: Total track distance per lap (km).
            cone_penalties: Number of cones hit.
            off_course_penalties: Number of off-course incidents.
        """
        total_dist_km = track_distance_km * result.laps_completed
        return self.score(
            total_time_s=result.total_time_s,
            total_energy_kwh=result.total_energy_kwh,
            laps_completed=result.laps_completed,
            cone_penalties=cone_penalties,
            off_course_penalties=off_course_penalties,
            total_distance_km=total_dist_km,
            driver_change_time_s=driver_change_time_s,
            driver_change_completed=driver_change_completed,
            track_km_per_lap=track_distance_km,
        )

    @staticmethod
    def _laps_score(laps_completed: int, driver_change_completed: bool = True) -> float:
        """Compute laps score per D.12.13.3.

        - 1 point per lap for all completed laps
        - +3 point bonus when the driver change is completed (NF-44)
        - Max 25 points

        The 3-point DC bonus is gated on ``driver_change_completed`` rather
        than on a lap count, so sims that don't yet model the DC can be
        scored honestly.
        """
        if laps_completed <= 0:
            return 0.0
        score = float(laps_completed)
        if driver_change_completed:
            score += 3.0
        return min(score, 25.0)
