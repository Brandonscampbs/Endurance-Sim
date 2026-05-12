"""Environment configuration: temperature, pressure, humidity, air density.

Replaces the bare ``physics_constants.AIR_DENSITY_KG_M3`` constant with
a dataclass that resolves air density from event conditions:

    rho = P_d / (R_d * T) + P_v / (R_v * T)

with ``P_v = phi * e_s(T)`` from the Magnus / Tetens equation and
``P_d = P_total - P_v``.  Default values reflect the Michigan
International Speedway endurance day (KJXN proxy, ~290 m elevation).

The ``physics_constants.AIR_DENSITY_KG_M3`` constant remains exported
for legacy callers (load_transfer, dynamics) so the migration is
incremental — callers that have explicit ``EnvironmentConfig`` access
should prefer ``EnvironmentConfig.air_density_kg_m3``.

References:
    - ICAO Doc 7488/3 (1993) / ISO 2533:1975 — International Standard
      Atmosphere.
    - CIPM-2007 (Picard) — equation for the density of moist air
      (preferred metrological form).
    - Tetens, O. (1930) / Magnus form — saturation vapor pressure over
      water, accurate within 1 % over 0–40 °C.
    - Plan ``docs/superpowers/plans/2026-05-06-track-environment-qss-gap.md``
      Part 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Specific gas constants (J/(kg·K)).
_R_DRY = 287.058
_R_VAPOR = 461.495


@dataclass(frozen=True)
class EnvironmentConfig:
    """Per-event ambient-air state for aerodynamic computations.

    Default values bootstrap the Michigan International Speedway
    endurance event (≈290 m elevation, 25 °C, 98500 Pa, 50 % RH —
    KJXN METAR proxy at ~30 km).  Override per-event when local
    METAR data is available.

    Attributes:
        ambient_temp_c: Dry-bulb temperature (°C).
        ambient_pressure_pa: Absolute atmospheric pressure (Pa).
        relative_humidity: Relative humidity as a fraction in [0, 1].
        altitude_m: Field elevation above mean sea level (m).  Carried
            for future ISA-based fallbacks and grade calculations;
            does not enter ``air_density_kg_m3`` because
            ``ambient_pressure_pa`` already encodes elevation.
    """

    ambient_temp_c: float = 25.0
    ambient_pressure_pa: float = 98500.0  # Michigan default ~290 m elev
    relative_humidity: float = 0.50
    altitude_m: float = 290.0  # MIS / KJXN approx field elevation

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError(
                "relative_humidity must be in [0, 1], got "
                f"{self.relative_humidity!r}"
            )
        if self.ambient_pressure_pa <= 0.0:
            raise ValueError(
                "ambient_pressure_pa must be positive, got "
                f"{self.ambient_pressure_pa!r}"
            )
        if self.ambient_temp_c <= -273.15:
            raise ValueError(
                "ambient_temp_c must be above absolute zero, got "
                f"{self.ambient_temp_c!r}"
            )

    @property
    def air_density_kg_m3(self) -> float:
        """Moist-air density (kg/m^3) for this environment.

        Computed via partial pressures:

            rho = P_d / (R_d · T) + P_v / (R_v · T)

        with ``P_v`` from Magnus / Tetens saturation vapor pressure
        ``e_s(T) = 611.2 · exp(17.62 · T_C / (243.12 + T_C))`` Pa
        (Bolton 1980 / WMO form, accurate within 0.1 % over 0–40 °C).
        """
        t_k = self.ambient_temp_c + 273.15
        pv_sat = 611.2 * math.exp(
            17.62 * self.ambient_temp_c / (243.12 + self.ambient_temp_c)
        )
        pv = self.relative_humidity * pv_sat
        pd = self.ambient_pressure_pa - pv
        return pd / (_R_DRY * t_k) + pv / (_R_VAPOR * t_k)
