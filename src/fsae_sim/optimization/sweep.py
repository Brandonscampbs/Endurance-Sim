"""Parameter sweep runner for exploring configuration spaces."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepConfig:
    """Definition of a parameter sweep."""
    parameter_name: str
    values: list[float]
    base_config_path: str
    output_dir: str
    description: str = ""


def run_sweep(config: SweepConfig) -> Path:
    """Run a parameter sweep and store results.

    Implemented in Phase 3.
    """
    raise NotImplementedError
