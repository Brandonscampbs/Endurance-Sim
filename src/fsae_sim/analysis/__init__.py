from fsae_sim.analysis.scoring import (
    CompetitionField,
    FSAEScoreResult,
    FSAEScoring,
)
from fsae_sim.analysis.telemetry_analysis import (
    DriverZone,
    extract_per_segment_actions,
    collapse_to_zones,
    detect_laps,
    compare_driver_stints,
)

# NF-55: `compute_lap_times`, `compute_energy_per_lap`, and
# `compute_pareto_frontier` in `fsae_sim.analysis.metrics` still raise
# `NotImplementedError`.  They are intentionally NOT re-exported here so
# IDE autocomplete and `from fsae_sim.analysis import *` do not suggest
# callable APIs that aren't callable.  They remain importable directly
# from `fsae_sim.analysis.metrics` until implemented or removed.

__all__ = [
    "CompetitionField",
    "FSAEScoreResult",
    "FSAEScoring",
    "DriverZone",
    "extract_per_segment_actions",
    "collapse_to_zones",
    "detect_laps",
    "compare_driver_stints",
]
