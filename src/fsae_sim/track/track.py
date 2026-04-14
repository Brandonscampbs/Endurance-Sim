"""Track representation as an ordered sequence of segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Minimum speed (km/h) for a GPS sample to be considered valid.
_GPS_SPEED_MIN_KMH: float = 5.0

# GPS position-accuracy threshold.  AiM reports 200 mm while the initial cold
# fix is being acquired; anything at exactly 200 is frozen/invalid.
_GPS_POS_ACC_BAD: float = 200.0

# GPS Radius sentinel: AiM reports 10000 m when the car is on a straight or
# the fix is uncertain.
_GPS_RADIUS_STRAIGHT: float = 10_000.0

# Bin size for segmenting the lap.
_SEGMENT_BIN_M: float = 5.0

# Rolling-median window for curvature smoothing.
_CURVATURE_SMOOTH_WINDOW: int = 5

# Minimum speed (m/s) for curvature computation to be valid.
_V_MIN_FOR_CURVATURE_MS: float = 2.0

# Longitude tolerance (degrees) for start/finish line crossing detection.
# ~90 m at Michigan latitude.
_SF_LON_TOLERANCE_DEG: float = 0.001


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

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def total_distance_m(self) -> float:
        """Sum of all segment lengths (full lap distance)."""
        return sum(s.length_m for s in self.segments)

    @property
    def lap_distance_m(self) -> float:
        """Distance of one complete lap (alias for total_distance_m)."""
        return self.total_distance_m

    @property
    def num_segments(self) -> int:
        """Number of segments in the track."""
        return len(self.segments)

    # ------------------------------------------------------------------ #
    # Construction from AiM telemetry                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_telemetry(
        cls,
        aim_csv_path: str | Path,
        *,
        bin_size_m: float = _SEGMENT_BIN_M,
        name: str = "Michigan Endurance",
    ) -> "Track":
        """Extract track geometry from AiM GPS telemetry.

        The method isolates the first full, clean lap from a Michigan FSAE
        endurance run, bins it into ``bin_size_m``-metre segments, and
        computes signed curvature and grade for each segment.

        Algorithm
        ---------
        1. Load the AiM CSV via :func:`fsae_sim.data.loader.load_aim_csv`.
        2. Filter to rows where the GPS fix is reliable:
           ``GPS Speed > 5 km/h``, ``GPS PosAccuracy != 200`` (cold-fix
           sentinel), and ``GPS Radius != 10000`` (straight/uncertain
           sentinel).
        3. Detect start/finish crossings as upward crossings of the median
           latitude, restricted to the longitude band shared by all
           consistent crossings.
        4. Isolate the **second** crossing-to-crossing interval (the first
           crossing-to-crossing interval is short because the GPS fix was
           just acquired).
        5. Bin that lap into ``bin_size_m``-metre windows.
        6. Per bin:

           - **curvature** = median(``GPS LatAcc`` × 9.81 / ``GPS Speed²``),
             where speed is in m/s.  Sign encodes direction: positive =
             right-hand turn, negative = left-hand turn.
           - **grade** = mean(tan(``GPS Slope`` × π/180)).

        7. Apply a rolling-median smoother (window = 5) to curvature.

        Args:
            aim_csv_path: Path to the AiM Race Studio CSV export.
            bin_size_m: Length of each output segment in metres.
                Defaults to 5 m.
            name: Name stored on the returned :class:`Track` object.

        Returns:
            A :class:`Track` whose segments represent one full lap.

        Raises:
            RuntimeError: If fewer than two start/finish crossings are
                detected in the telemetry (cannot isolate a complete lap).
            ValueError: If no segments are produced (empty lap after
                filtering).
        """
        from fsae_sim.data.loader import load_aim_csv  # local import avoids circular

        _metadata, df = load_aim_csv(aim_csv_path)

        # ---- 1. Filter to reliable GPS rows --------------------------------
        good_mask: pd.Series = (
            (df["GPS Speed"] > _GPS_SPEED_MIN_KMH)
            & (df["GPS PosAccuracy"] != _GPS_POS_ACC_BAD)
            & (df["GPS Radius"] != _GPS_RADIUS_STRAIGHT)
        )
        good: pd.DataFrame = df[good_mask].reset_index(drop=True)

        lat: np.ndarray = good["GPS Latitude"].values
        cum_dist: np.ndarray = good["Distance on GPS Speed"].values

        # ---- 2. Detect start/finish crossings ------------------------------
        center_lat: float = float(np.median(lat))
        lon_arr: np.ndarray = good["GPS Longitude"].values

        crossings: list[tuple[int, float, float]] = []
        for i in range(1, len(lat)):
            if lat[i - 1] < center_lat <= lat[i]:
                crossings.append((i, float(cum_dist[i]), float(lon_arr[i])))

        if not crossings:
            raise RuntimeError(
                "No start/finish crossings detected in telemetry. "
                "Check GPS data quality."
            )

        lons_at_crossings = np.array([c[2] for c in crossings])
        median_lon: float = float(np.median(lons_at_crossings))

        sf_crossings: list[tuple[int, float, float]] = [
            c for c in crossings if abs(c[2] - median_lon) < _SF_LON_TOLERANCE_DEG
        ]

        if len(sf_crossings) < 2:
            raise RuntimeError(
                f"Only {len(sf_crossings)} start/finish crossing(s) detected; "
                "need at least 2 to isolate a complete lap."
            )

        # ---- 3. Isolate the first *complete* lap ---------------------------
        # sf_crossings[0] is often short (GPS just acquired); use [1] → [2].
        lap_index = 1 if len(sf_crossings) >= 3 else 0
        lap_start_dist: float = sf_crossings[lap_index][1]
        lap_end_dist: float = sf_crossings[lap_index + 1][1]

        lap_mask: np.ndarray = (cum_dist >= lap_start_dist) & (cum_dist <= lap_end_dist)
        lap_df: pd.DataFrame = good[lap_mask].reset_index(drop=True)

        if lap_df.empty:
            raise ValueError(
                "Lap extraction produced an empty DataFrame. "
                f"Check distance range {lap_start_dist:.1f}–{lap_end_dist:.1f} m."
            )

        # ---- 4. Normalise distance within lap ------------------------------
        dist_in_lap: np.ndarray = lap_df["Distance on GPS Speed"].values - lap_start_dist

        # ---- 5. Pre-compute per-sample curvature and grade -----------------
        v_ms: np.ndarray = lap_df["GPS Speed"].values * (1_000.0 / 3_600.0)
        a_lat_ms2: np.ndarray = lap_df["GPS LatAcc"].values * 9.81
        slope_deg: np.ndarray = lap_df["GPS Slope"].values

        # κ = a_lat / v²  (signed: positive = right turn, negative = left turn)
        # Zero for very low speeds to avoid division noise.
        v_safe = np.where(v_ms > _V_MIN_FOR_CURVATURE_MS, v_ms, np.nan)
        k_raw: np.ndarray = a_lat_ms2 / (v_safe ** 2)
        k_raw = np.nan_to_num(k_raw, nan=0.0)

        grade_raw: np.ndarray = np.tan(slope_deg * (math.pi / 180.0))

        # ---- 6. Bin into segments ------------------------------------------
        lap_length: float = float(dist_in_lap[-1])
        n_bins: int = int(math.floor(lap_length / bin_size_m))

        if n_bins == 0:
            raise ValueError(
                f"Lap length {lap_length:.1f} m is shorter than bin size "
                f"{bin_size_m} m; cannot create any segments."
            )

        raw_curvatures: list[float] = []
        raw_grades: list[float] = []

        for i in range(n_bins):
            bin_lo = i * bin_size_m
            bin_hi = (i + 1) * bin_size_m
            idx_mask: np.ndarray = (dist_in_lap >= bin_lo) & (dist_in_lap < bin_hi)
            if idx_mask.any():
                raw_curvatures.append(float(np.median(k_raw[idx_mask])))
                raw_grades.append(float(np.mean(grade_raw[idx_mask])))
            else:
                # Empty bin (can happen at the boundary): carry previous value.
                prev_k = raw_curvatures[-1] if raw_curvatures else 0.0
                prev_g = raw_grades[-1] if raw_grades else 0.0
                raw_curvatures.append(prev_k)
                raw_grades.append(prev_g)

        # ---- 7. Smooth curvature with rolling median -----------------------
        smoothed_k: np.ndarray = (
            pd.Series(raw_curvatures)
            .rolling(
                window=_CURVATURE_SMOOTH_WINDOW,
                center=True,
                min_periods=1,
            )
            .median()
            .to_numpy()
        )

        # ---- 8. Build Segment list -----------------------------------------
        segments: list[Segment] = [
            Segment(
                index=i,
                distance_start_m=float(i * bin_size_m),
                length_m=bin_size_m,
                curvature=float(smoothed_k[i]),
                grade=float(raw_grades[i]),
            )
            for i in range(n_bins)
        ]

        return cls(name=name, segments=segments)
