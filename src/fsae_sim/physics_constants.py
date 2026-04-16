"""Physical constants shared across simulator modules."""

GRAVITY_M_S2: float = 9.81
"""Standard gravitational acceleration (m/s^2).

ISA sea level value.  Used for weight force, grade force, load transfer,
and g-unit conversions throughout the simulator.
"""

AIR_DENSITY_KG_M3: float = 1.225
"""Air density (kg/m^3) at ISA sea level, 15 C, dry air.

Used for aerodynamic drag and downforce computations.
"""
