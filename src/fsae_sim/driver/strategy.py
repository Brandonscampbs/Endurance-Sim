"""Driver strategy and simulation state definitions."""

from dataclasses import dataclass
from enum import Enum

from fsae_sim.track.track import Segment


class ControlAction(Enum):
    """Discrete driver action."""
    THROTTLE = "throttle"
    COAST = "coast"
    BRAKE = "brake"


@dataclass(frozen=True)
class ControlCommand:
    """Output of a driver strategy decision.

    Attributes:
        action: discrete control action (THROTTLE/COAST/BRAKE).
        throttle_pct: fraction of LVCU/inverter torque-limit envelope
            to request, in [0, 1]. Sim consumes it as
            ``motor_torque = throttle_pct * max_motor_torque(rpm)``.
        brake_pct: **brake-pressure** fraction in [0, 1], normalized to
            the 99th-percentile recorded brake-line pressure. NF-30:
            this is NOT the same as "fraction of max regen torque."
            Consumers that need a regen-torque command must map
            pressure→regen explicitly; see `regen_force`.
    """

    action: ControlAction
    throttle_pct: float = 0.0  # 0 to 1
    brake_pct: float = 0.0  # 0 to 1 — brake-PRESSURE fraction, not regen-torque


@dataclass
class SimState:
    """Instantaneous simulation state passed through the time-step loop."""
    time: float  # seconds
    distance: float  # meters along track
    speed: float  # m/s
    soc: float  # 0 to 1
    pack_voltage: float  # V
    pack_current: float  # A
    cell_temp: float  # degrees C
    lap: int
    segment_idx: int


class DriverStrategy:
    """Base class for driver control strategies. Subclass to implement."""
    name: str = "base"

    def decide(self, state: SimState, upcoming: list[Segment]) -> ControlCommand:
        """Given current state and upcoming track, choose an action.

        Implemented in Phase 2/3 by subclasses.
        """
        raise NotImplementedError
