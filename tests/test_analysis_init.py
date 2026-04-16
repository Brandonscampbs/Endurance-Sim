"""Tests for :mod:`fsae_sim.analysis` package-level re-exports.

Guards the NF-55 policy that unimplemented stub functions
(``compute_lap_times``, ``compute_energy_per_lap``, and
``compute_pareto_frontier``) must not appear in ``__all__``.  They may
still be imported directly from ``fsae_sim.analysis.metrics``, but IDE
autocomplete on the package should not suggest them.
"""

from __future__ import annotations

import pytest


def test_stub_functions_not_in_all() -> None:
    """The three unimplemented stubs must be excluded from ``__all__``."""
    import fsae_sim.analysis as analysis

    for name in ("compute_lap_times", "compute_energy_per_lap", "compute_pareto_frontier"):
        assert name not in analysis.__all__, (
            f"{name!r} is still exported from fsae_sim.analysis.__all__ "
            "but the implementation raises NotImplementedError."
        )


def test_implemented_symbols_still_exported() -> None:
    """Implemented public APIs remain in ``__all__``."""
    import fsae_sim.analysis as analysis

    for name in (
        "FSAEScoring",
        "FSAEScoreResult",
        "CompetitionField",
        "DriverZone",
        "extract_per_segment_actions",
        "collapse_to_zones",
        "detect_laps",
        "compare_driver_stints",
    ):
        assert name in analysis.__all__
        assert hasattr(analysis, name)


def test_metrics_stubs_still_raise_when_imported_directly() -> None:
    """The stubs remain available via the private module path; they raise."""
    from fsae_sim.analysis import metrics

    with pytest.raises(NotImplementedError):
        metrics.compute_lap_times(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        metrics.compute_energy_per_lap(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        metrics.compute_pareto_frontier(None)  # type: ignore[arg-type]
