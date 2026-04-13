"""Track representation as an ordered sequence of segments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A discrete track segment with geometric properties."""
    index: int
    distance_start_m: float
    length_m: float
    curvature: float  # 1/radius in 1/m, 0 for straight, signed for direction
    grade: float  # rise/run, positive = uphill
    grip_factor: float = 1.0  # multiplier on baseline grip, 1.0 = nominal


@dataclass
class Track:
    """Ordered sequence of segments representing a circuit."""
    name: str
    segments: list[Segment]

    @property
    def total_distance_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @classmethod
    def from_telemetry(cls, aim_csv_path: str) -> "Track":
        """Extract track segments from AiM GPS telemetry.

        Implemented in Phase 2.
        """
        raise NotImplementedError("Track extraction from telemetry not yet implemented")
