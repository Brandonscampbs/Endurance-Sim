"""Energy-budget shaper for the adaptive driver (Wave 4b Sub-task B).

Two strategies (per the TUMFTM / Heilmeier IEEE EVER 2019 taxonomy):

- **FCFB** (First-Come-First-Brake, also "Full-Capability-Full-Boost"
  default): no shaping until the *total* stint budget is exceeded; on
  exhaustion, attenuate the per-segment target speed proportionally
  to the overage so the driver smoothly winds down rather than
  going flat-out into a brick wall when the pack runs empty.

- **LBP** (Lap-Budget Proportional): allocate the budget evenly across
  the laps and shape against the lap-pro-rata share. At lap k of N,
  the expected spend is (k + 1) / N * budget. If actual spend at
  segment progress p is over the projected pro-rata target, attenuate.

Both strategies are pure: same inputs always yield the same output, no
state, no randomness, no side effects on the upstream envelope. The
adaptive driver consumes the shaper output as a per-segment v_target
override before issuing the inverse-pedal solve.

Reference: A. Heilmeier, A. Geisslinger, J. Betz, "A Quasi-Steady-State
Lap Time Simulation for Electrified Race Cars" (IEEE EVER 2019), and
TUMFTM's open-source `laptime-simulation` companion code which exports
the LBP allocator at ``laptimesim/src/driver.py:__strategy_lbp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Strategy literal aliases. Lowercase tokens align with command-line
# conventions in scripts/sim_compare.py and elsewhere.
EnergyShaperStrategy = Literal["fcfb", "lbp"]


def _normalize_strategy(strategy: str) -> str:
    s = str(strategy).strip().lower()
    if s not in ("fcfb", "lbp"):
        raise ValueError(
            f"Unknown energy shaper strategy {strategy!r}; expected "
            "'fcfb' or 'lbp'."
        )
    return s


@dataclass(frozen=True)
class EnergyShaperConfig:
    """Frozen configuration bundle for :class:`EnergyShaper`.

    Attributes:
        strategy: "fcfb" (default) or "lbp". See module docstring.
        total_budget_kwh: Net stint energy budget in kWh. CT-16EV
            Michigan endurance default is 6.4 kWh — a 10 % margin
            below the 7.128 kWh nominal pack to leave headroom for
            BMS-derate / cell imbalance / safety.
        laps_total: Total laps the budget covers. CT-16EV Michigan
            endurance is 22 laps (see CLAUDE.md "FSAE Michigan
            endurance is 22 laps"). Only consumed by LBP; FCFB
            ignores it.
    """

    strategy: EnergyShaperStrategy = "fcfb"
    total_budget_kwh: float = 6.4
    laps_total: int = 22

    def __post_init__(self) -> None:
        # Validate strategy literal — dataclass equality and hashing are
        # unaffected because the field type stays the same. We can't
        # mutate the field on a frozen dataclass, so this is a guard
        # that raises on construction with an invalid literal.
        _ = _normalize_strategy(self.strategy)


class EnergyShaper:
    """Per-segment target-speed shaper.

    Pure function dispatcher on :attr:`EnergyShaperConfig.strategy`. No
    internal state — consume :meth:`shape_target_speed` and feed the
    returned ``v_target`` back into the adaptive driver's inverse
    pedal solve.
    """

    def __init__(self, config: EnergyShaperConfig) -> None:
        self._config = config
        # Cache the normalized strategy token so we can match without
        # re-normalizing on every call.
        self._strategy = _normalize_strategy(config.strategy)

    @property
    def config(self) -> EnergyShaperConfig:
        return self._config

    def shape_target_speed(
        self,
        v_max: float,
        *,
        energy_used_kwh: float,
        lap_index: int,
        segment_progress: float,
    ) -> float:
        """Return the attenuated target speed (m/s).

        Args:
            v_max: Original envelope target speed at the current segment.
            energy_used_kwh: Net energy spent since the start of the
                stint (kWh). Positive numbers mean discharge.
            lap_index: 0-based index of the current lap.
            segment_progress: Fraction of the way through the current
                lap, in [0, 1]. Used by LBP for fine-grained pro-rata;
                FCFB ignores it.

        Returns:
            The (possibly attenuated) target speed (m/s). Always
            >= 0; never larger than ``v_max``.
        """
        v_max_f = float(v_max)
        if v_max_f <= 0.0:
            return 0.0
        if self._strategy == "fcfb":
            return self._shape_fcfb(v_max_f, energy_used_kwh)
        # LBP
        return self._shape_lbp(
            v_max_f, energy_used_kwh, lap_index, segment_progress,
        )

    # ------------------------------------------------------------------
    # FCFB: attenuate when total budget is exceeded
    # ------------------------------------------------------------------

    def _shape_fcfb(self, v_max: float, energy_used_kwh: float) -> float:
        budget = float(self._config.total_budget_kwh)
        if budget <= 0.0:
            return v_max
        overage_kwh = float(energy_used_kwh) - budget
        if overage_kwh <= 0.0:
            return v_max
        # Attenuate proportional to overage. Cap at 90% so the car can
        # still limp; deeper cuts would freeze the sim and hide actual
        # battery exhaustion behavior (the BMS / OCV floor handles
        # the hard stop separately).
        attenuation = min(0.9, overage_kwh / budget)
        return max(0.0, v_max * (1.0 - attenuation))

    # ------------------------------------------------------------------
    # LBP: attenuate when over lap-pro-rata share
    # ------------------------------------------------------------------

    def _shape_lbp(
        self,
        v_max: float,
        energy_used_kwh: float,
        lap_index: int,
        segment_progress: float,
    ) -> float:
        laps_total = max(1, int(self._config.laps_total))
        # Lap-pro-rata budget at the *end* of the current lap. Using
        # (lap_index + 1) means at lap 0 the driver should be on track
        # to spend 1/N of the budget, at lap N-1 the full N/N budget.
        per_lap_budget = float(self._config.total_budget_kwh) / laps_total
        # Add segment_progress so the within-lap budget interpolates
        # linearly between the previous-lap end and this-lap end.
        # E.g. at lap 10, progress 0.5 -> expected spend = 10.5 * per_lap.
        progress = max(0.0, min(1.0, float(segment_progress)))
        expected_spend_kwh = per_lap_budget * (lap_index + progress + 1.0 - 0.5)
        # At lap N-1, progress 0.5 -> N * per_lap = full budget.
        # The fine-grained interpolation is what gives LBP its name;
        # if you're under your pro-rata mark, lift not needed.
        overage_kwh = float(energy_used_kwh) - expected_spend_kwh
        if overage_kwh <= 0.0:
            return v_max
        # Attenuate proportional to overage relative to per-lap budget
        # rather than total budget — gives a stronger per-lap response
        # than FCFB but never zeros out the target.
        attenuation = min(0.9, overage_kwh / max(per_lap_budget, 1e-9))
        return max(0.0, v_max * (1.0 - attenuation))
