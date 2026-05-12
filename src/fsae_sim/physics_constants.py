"""Physical constants shared across simulator modules.

``AIR_DENSITY_KG_M3`` is preserved for back-compat (modules that have not
yet migrated to :class:`fsae_sim.vehicle.environment.EnvironmentConfig`)
but is *deprecated*: any new caller should resolve density from a
``VehicleParams.environment`` instance instead, which captures the per-
event temperature / pressure / humidity correctly.

The deprecation is emitted on explicit ``from ... import
AIR_DENSITY_KG_M3`` use via the module-level ``__getattr__`` indirection
(PEP 562).  Bare ``import fsae_sim.physics_constants`` does *not* warn,
so legacy modules that still hold a name reference do not flood the
test output.  To opt in to the warning during migration audits, set
the env var ``FSAE_SIM_WARN_AIR_DENSITY_IMPORT=1``.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

GRAVITY_M_S2: float = 9.81
"""Standard gravitational acceleration (m/s^2).

ISA sea level value.  Used for weight force, grade force, load transfer,
and g-unit conversions throughout the simulator.
"""

# Internal store for the legacy constant.  Exposed through __getattr__ to
# allow a (gated) deprecation warning on explicit attribute access.
_AIR_DENSITY_KG_M3: float = 1.225


def __getattr__(name: str) -> Any:
    if name == "AIR_DENSITY_KG_M3":
        if os.environ.get("FSAE_SIM_WARN_AIR_DENSITY_IMPORT") == "1":
            warnings.warn(
                "physics_constants.AIR_DENSITY_KG_M3 is deprecated; resolve "
                "air density from VehicleParams.environment "
                "(EnvironmentConfig.air_density_kg_m3) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return _AIR_DENSITY_KG_M3
    raise AttributeError(f"module 'fsae_sim.physics_constants' has no attribute {name!r}")
