"""Track representation as an ordered sequence of segments."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum speed (km/h) for a GPS sample to be considered valid.
_GPS_SPEED_MIN_KMH: float = 5.0

# GPS position-accuracy threshold.  AiM reports 200 mm while the initial cold
# fix is being acquired; anything at exactly 200 is frozen/invalid.
_GPS_POS_ACC_BAD: float = 200.0

# GPS Radius sentinel: AiM reports 10000 m when the car is on a straight or
# the fix is uncertain.
_GPS_RADIUS_STRAIGHT: float = 10_000.0

# Bin size for segmenting the lap.
# D-14: 0.5 m default. Smoother window (_SMOOTH_DISTANCE_M) is a fixed
# physical distance, so finer segmentation doesn't change smoothing scale.
_SEGMENT_BIN_M: float = 0.5

# Rolling-median smoother physical distance (metres). Fixed 5 m window
# retains hairpin peaks (~10-15 m arcs) while still suppressing per-sample
# GPS-acceleration noise.  Was 25 m (C12) — flattened hairpin peaks.
_SMOOTH_DISTANCE_M: float = 5.0

# Minimum speed (m/s) for curvature computation to be valid.
_V_MIN_FOR_CURVATURE_MS: float = 2.0

# Start/finish detection gate tolerances (2D gate per S19).
# Proximity radius to the reference start point (degrees; ~11 m at MI lat).
# Chosen so all 21 Michigan laps cleanly trigger while still being far
# tighter than the nearest return-pass of the track (~50 m+).
_SF_GATE_RADIUS_DEG: float = 1.0e-4
# Minimum physical distance between consecutive valid crossings (metres).
# Prevents same-lap re-triggers when the gate passes through a slow section.
_SF_MIN_LAP_DISTANCE_M: float = 400.0


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
        aim_csv_path: str | Path | None = None,
        *,
        df: pd.DataFrame | None = None,
        bin_size_m: float = _SEGMENT_BIN_M,
        smooth_distance_m: float = _SMOOTH_DISTANCE_M,
        name: str = "Michigan Endurance",
    ) -> "Track":
        """Extract track geometry from AiM GPS telemetry.

        The method isolates the first full, clean lap from a Michigan FSAE
        endurance run, bins it into ``bin_size_m``-metre segments, and
        computes signed curvature and grade for each segment.

        Can accept either a file path (loaded via ``load_aim_csv``) or a
        pre-loaded DataFrame.

        Algorithm
        ---------
        1. Load the AiM CSV via :func:`fsae_sim.data.loader.load_aim_csv`.
        2. Filter to rows where the GPS fix is reliable:
           ``GPS Speed > 5 km/h``, and if available,
           ``GPS PosAccuracy != 200`` and ``GPS Radius != 10000``.
        3. Detect start/finish crossings with a 2D proximity gate: a
           crossing is registered when the car approaches the reference
           start point within :data:`_SF_GATE_RADIUS_DEG` and has travelled
           at least :data:`_SF_MIN_LAP_DISTANCE_M` since the previous
           crossing.  Laps that fail the minimum-distance gate are logged
           and dropped (S19).
        4. Isolate a crossing-to-crossing interval where GPS LatAcc data
           is available.
        5. Bin that lap into ``bin_size_m``-metre windows (``math.ceil`` so
           the fractional tail segment is preserved, NF-20).
        6. Per bin:

           - **curvature** = median(``GPS LatAcc`` × 9.81 / ``GPS Speed²``),
             where speed is in m/s.  Sign encodes direction: positive =
             right-hand turn, negative = left-hand turn.  At samples where
             ``v_ms <= V_MIN``, ``k_raw`` is recovered by interpolating from
             neighbouring high-speed samples (or GPS Radius if present)
             rather than forced to zero (NF-7).
           - **grade** = mean(tan(``GPS Slope`` × π/180)).

        7. Apply a rolling-median smoother whose physical window is
           ``smooth_distance_m`` (default 5 m) to curvature.

        Args:
            aim_csv_path: Path to the AiM Race Studio CSV export.
                Mutually exclusive with ``df``.
            df: Pre-loaded AiM DataFrame (e.g. from ``load_cleaned_csv``).
                Must contain GPS Speed, Distance on GPS Speed, GPS Latitude,
                GPS Longitude, GPS LatAcc columns.
            bin_size_m: Length of each output segment in metres.
                Defaults to 5 m.
            smooth_distance_m: Physical window (m) of the rolling-median
                curvature smoother.  Defaults to 5 m (retains hairpin
                peaks while suppressing GPS-accel noise).
            name: Name stored on the returned :class:`Track` object.

        Returns:
            A :class:`Track` whose segments represent one full lap.

        Raises:
            RuntimeError: If fewer than two start/finish crossings are
                detected in the telemetry (cannot isolate a complete lap).
            ValueError: If no segments are produced (empty lap after
                filtering).
        """
        if df is None:
            from fsae_sim.data.loader import load_aim_csv  # local import avoids circular
            _metadata, df = load_aim_csv(aim_csv_path)

        # ---- 1. Filter to reliable GPS rows --------------------------------
        good_mask: pd.Series = df["GPS Speed"] > _GPS_SPEED_MIN_KMH
        if "GPS PosAccuracy" in df.columns:
            good_mask = good_mask & (df["GPS PosAccuracy"] != _GPS_POS_ACC_BAD)
        if "GPS Radius" in df.columns:
            good_mask = good_mask & (df["GPS Radius"] != _GPS_RADIUS_STRAIGHT)
        good: pd.DataFrame = df[good_mask].reset_index(drop=True)

        lat: np.ndarray = good["GPS Latitude"].values
        cum_dist: np.ndarray = good["Distance on GPS Speed"].values
        lon_arr: np.ndarray = good["GPS Longitude"].values

        # ---- 2. Detect start/finish crossings via 2D gate (S19) -----------
        # Pick a reference point: the first reliable GPS sample is the gate.
        # A crossing is triggered when the car is within SF_GATE_RADIUS_DEG
        # of the reference AND has travelled SF_MIN_LAP_DISTANCE_M since the
        # previous crossing.  This replaces the east-west-only latitude
        # crossing heuristic which silently dropped laps whose S/F line was
        # not east-west aligned.
        if len(lat) < 2:
            raise RuntimeError(
                "Not enough reliable GPS samples to detect a lap."
            )

        ref_lat: float = float(lat[0])
        ref_lon: float = float(lon_arr[0])
        # Approximate metre-to-degree scaling near the reference latitude.
        # At 42.7 deg N (Michigan), 1 deg lat ~ 111 320 m, 1 deg lon ~ 81 800 m
        # — a 1.36× anisotropy.  Scale d_lon by cos(ref_lat) so the gate is
        # a real circle rather than an ellipse stretched east-west.
        d_lat = lat - ref_lat
        d_lon = (lon_arr - ref_lon) * math.cos(math.radians(ref_lat))
        dist_to_ref = np.sqrt(d_lat ** 2 + d_lon ** 2)

        inside_gate = dist_to_ref < _SF_GATE_RADIUS_DEG
        # Rising-edge of gate entry = sample just entered proximity band.
        entry_mask = np.zeros(len(inside_gate), dtype=bool)
        entry_mask[1:] = inside_gate[1:] & ~inside_gate[:-1]

        raw_crossings: list[tuple[int, float, float, float]] = [
            (i, float(cum_dist[i]), float(lat[i]), float(lon_arr[i]))
            for i in np.where(entry_mask)[0]
        ]

        # Enforce minimum-lap-distance gate: drop re-triggers within the
        # same lap.  Log dropped candidates for audit.  Preserve physical
        # lap number in the tuples by keeping the original enumeration.
        sf_crossings: list[tuple[int, int, float, float, float]] = []
        dropped: list[tuple[int, float]] = []
        last_dist = -math.inf
        for phys_lap, (i, d, la, lo) in enumerate(raw_crossings):
            if d - last_dist >= _SF_MIN_LAP_DISTANCE_M:
                sf_crossings.append((phys_lap, i, d, la, lo))
                last_dist = d
            else:
                dropped.append((phys_lap, d - last_dist))

        if dropped:
            logger.info(
                "S/F detection dropped %d sub-minimum-distance crossings: %s",
                len(dropped), dropped[:10],
            )

        if len(sf_crossings) < 2:
            raise RuntimeError(
                f"Only {len(sf_crossings)} start/finish crossing(s) detected; "
                "need at least 2 to isolate a complete lap."
            )

        # ---- 3. Isolate a complete lap with GPS LatAcc data -----------------
        # Pick the first crossing-to-crossing interval that has GPS LatAcc
        # data available (some cleaned datasets have NaN LatAcc early on).
        lat_acc_col: np.ndarray = good["GPS LatAcc"].values
        lap_df: pd.DataFrame = pd.DataFrame()
        lap_start_dist: float = 0.0

        for lap_index in range(len(sf_crossings) - 1):
            _start_dist = sf_crossings[lap_index][2]
            _end_dist = sf_crossings[lap_index + 1][2]
            _mask = (cum_dist >= _start_dist) & (cum_dist <= _end_dist)
            _lat_acc_in_lap = lat_acc_col[_mask]
            # Require at least 80% of samples to have valid LatAcc
            valid_frac = np.sum(~np.isnan(_lat_acc_in_lap)) / max(len(_lat_acc_in_lap), 1)
            if valid_frac >= 0.8:
                lap_start_dist = _start_dist
                lap_df = good[_mask].reset_index(drop=True)
                break

        if lap_df.empty:
            raise ValueError(
                "No lap with sufficient GPS LatAcc data found. "
                "Need at least 80% valid samples in one crossing-to-crossing interval."
            )

        # ---- 4. Normalise distance within lap ------------------------------
        dist_in_lap: np.ndarray = lap_df["Distance on GPS Speed"].values - lap_start_dist

        # ---- 5. Pre-compute per-sample curvature and grade -----------------
        v_ms: np.ndarray = lap_df["GPS Speed"].values * (1_000.0 / 3_600.0)
        a_lat_raw: np.ndarray = lap_df["GPS LatAcc"].values.copy()
        # Fill any remaining NaN in LatAcc with 0 (straight assumption)
        a_lat_raw = np.nan_to_num(a_lat_raw, nan=0.0)
        a_lat_ms2: np.ndarray = a_lat_raw * 9.81
        if "GPS Slope" in lap_df.columns:
            slope_deg: np.ndarray = np.nan_to_num(
                lap_df["GPS Slope"].values, nan=0.0,
            )
        else:
            slope_deg = np.zeros(len(lap_df))

        # κ = a_lat / v²  (signed: positive = right turn, negative = left turn)
        # NF-7: at samples where v_ms <= V_MIN, fall back to GPS Radius (signed
        # by LatAcc direction) when present, or interpolate k_raw from
        # neighbouring high-speed samples.  Do NOT force to zero.
        valid_v = v_ms > _V_MIN_FOR_CURVATURE_MS
        v_safe = np.where(valid_v, v_ms, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            k_raw: np.ndarray = a_lat_ms2 / (v_safe ** 2)

        low_speed = ~valid_v
        if low_speed.any():
            # First try GPS Radius if available.
            filled_from_radius = np.zeros_like(low_speed)
            if "GPS Radius" in lap_df.columns:
                radius = lap_df["GPS Radius"].values.astype(float)
                radius_ok = (
                    low_speed
                    & np.isfinite(radius)
                    & (radius > 0.0)
                    & (radius < _GPS_RADIUS_STRAIGHT)
                )
                if radius_ok.any():
                    # Sign κ by the sign of LatAcc at the low-speed sample.
                    sign = np.sign(a_lat_ms2[radius_ok])
                    # If LatAcc is exactly 0, default positive sign.
                    sign = np.where(sign == 0.0, 1.0, sign)
                    k_raw[radius_ok] = sign / radius[radius_ok]
                    filled_from_radius = radius_ok

            # Remaining low-speed samples: interpolate from valid neighbours.
            still_missing = low_speed & ~filled_from_radius & ~np.isfinite(k_raw)
            if still_missing.any():
                idx = np.arange(len(k_raw))
                known = np.isfinite(k_raw) & ~still_missing
                if known.any():
                    k_raw[still_missing] = np.interp(
                        idx[still_missing], idx[known], k_raw[known]
                    )
                else:
                    # No high-speed anchor at all — fall back to zero, the
                    # lap is effectively static and curvature is undefined.
                    k_raw[still_missing] = 0.0

        # Final safety: any residual NaN (e.g. leading edge with no anchor)
        # gets zeroed so downstream code sees finite values.
        k_raw = np.nan_to_num(k_raw, nan=0.0, posinf=0.0, neginf=0.0)

        grade_raw: np.ndarray = np.tan(slope_deg * (math.pi / 180.0))

        # ---- 6. Bin into segments (NF-20: ceil + fractional tail) ---------
        lap_length: float = float(dist_in_lap[-1])
        n_bins: int = int(math.ceil(lap_length / bin_size_m))

        if n_bins == 0:
            raise ValueError(
                f"Lap length {lap_length:.1f} m is shorter than bin size "
                f"{bin_size_m} m; cannot create any segments."
            )

        # Per-segment lengths: all bins are bin_size_m except possibly the
        # last, which is the residual so that the total is exactly lap_length.
        segment_lengths: list[float] = [bin_size_m] * n_bins
        residual = lap_length - (n_bins - 1) * bin_size_m
        # Numerical safety: residual is in (0, bin_size_m]; if the lap_length
        # is an exact multiple of bin_size_m, residual == bin_size_m.
        if residual <= 0.0:
            # Can happen with floating-point near-exact multiples; clamp.
            residual = bin_size_m
        segment_lengths[-1] = residual
        assert abs(sum(segment_lengths) - lap_length) < 1e-6, (
            f"Segment-length sum {sum(segment_lengths)} != lap_length {lap_length}"
        )

        raw_curvatures: list[float] = []
        raw_grades: list[float] = []

        for i in range(n_bins):
            bin_lo = i * bin_size_m
            bin_hi = bin_lo + segment_lengths[i]
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
        # Window covers smooth_distance_m of physical track, rounded to an
        # odd bin count so the centred median is symmetric.
        smooth_window = max(1, int(round(smooth_distance_m / bin_size_m)))
        if smooth_window % 2 == 0:
            smooth_window += 1

        smoothed_k: np.ndarray = (
            pd.Series(raw_curvatures)
            .rolling(
                window=smooth_window,
                center=True,
                min_periods=1,
            )
            .median()
            .to_numpy()
        )

        # ---- 8. Build Segment list -----------------------------------------
        segments: list[Segment] = []
        cumulative = 0.0
        for i in range(n_bins):
            segments.append(
                Segment(
                    index=i,
                    distance_start_m=float(cumulative),
                    length_m=float(segment_lengths[i]),
                    curvature=float(smoothed_k[i]),
                    grade=float(raw_grades[i]),
                )
            )
            cumulative += segment_lengths[i]

        return cls(name=name, segments=segments)
