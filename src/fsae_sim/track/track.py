"""Track representation as an ordered sequence of segments."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
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

# Centerline smoothing sigma in metres. Applied to the (x, y) centerline
# Gaussian-filtered with periodic boundary so closed-track wrap is exact.
# 1.0 m matches a single-sample GPS pixel — enough to reject driver-line
# variance across laps without rounding off real hairpins.
_CENTERLINE_SIGMA_M: float = 1.25

# Earth's WGS-84 radius (m) used for the lat/lon -> local cartesian step.
# A flat-earth approximation is fine for a 1 km circuit at 42 deg N: the
# residual second-order error is sub-millimetre.
_M_PER_DEG_LAT: float = 111_320.0

# Minimum speed (m/s) for curvature computation to be valid.
_V_MIN_FOR_CURVATURE_MS: float = 2.0

# Minimum number of laps to fall back to the legacy single-lap LatAcc/v^2
# extraction. With 21 Michigan laps we always exercise the GPS-coord path,
# but unit tests using synthetic short telemetry need the safety net.
_MIN_LAPS_FOR_GPS_AVERAGE: int = 3

# Start/finish detection gate tolerances (2D gate per S19).
# Proximity radius to the reference start point (degrees; ~11 m at MI lat).
# Chosen so all 21 Michigan laps cleanly trigger while still being far
# tighter than the nearest return-pass of the track (~50 m+).
_SF_GATE_RADIUS_DEG: float = 1.0e-4
# Minimum physical distance between consecutive valid crossings (metres).
# Prevents same-lap re-triggers when the gate passes through a slow section.
_SF_MIN_LAP_DISTANCE_M: float = 400.0


def _telemetry_speed_col(df: pd.DataFrame) -> str:
    """Prefer wheel-speed telemetry; keep GPS Speed as fixture fallback."""
    if "LFspeed" in df.columns:
        return "LFspeed"
    return "GPS Speed"


@dataclass(frozen=True)
class Segment:
    """A discrete track segment with geometric properties."""

    index: int
    distance_start_m: float
    length_m: float
    curvature: float  # 1/radius in 1/m, 0 for straight, signed for direction
    grade: float  # rise/run, positive = uphill
    grip_factor: float = 1.0  # multiplier on baseline grip, 1.0 = nominal
    # Source lap for per-lap reconstructed tracks. -1 (default) means the
    # segment belongs to a single-lap track and the engine's outer loop
    # counter sets the lap index instead.
    lap_index: int = -1


@dataclass
class Track:
    """Ordered sequence of segments representing a circuit."""

    name: str
    segments: list[Segment]
    source: str = "unknown"
    provenance: dict[str, object] = field(default_factory=dict)

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
        smooth_distance_m: float = _CENTERLINE_SIGMA_M,
        centerline_sigma_m: float | None = None,
        name: str = "Michigan Endurance",
    ) -> "Track":
        """Extract track geometry from AiM GPS telemetry.

        With enough repeated laps and GPS position channels, builds a
        physical centerline from latitude/longitude samples projected into
        local cartesian coordinates and averaged across laps. Curvature is
        then computed from the centerline geometry.

        Falls back to the legacy single-lap ``a_lat / v^2`` extraction when
        too few laps are present or GPS latitude/longitude are unavailable,
        so synthetic unit-test fixtures still exercise that path.

        Args:
            aim_csv_path: Path to the AiM Race Studio CSV export.
                Mutually exclusive with ``df``.
            df: Pre-loaded AiM DataFrame.  Must contain LFspeed or GPS Speed,
                Distance on GPS Speed, GPS Latitude, GPS Longitude.
            bin_size_m: Length of each output segment in metres.
                Defaults to 0.5 m.
            smooth_distance_m: Smoothing scale applied to the lap-1
                curvature signal.  Default 1 m matches the GPS pixel.
            centerline_sigma_m: Alias for ``smooth_distance_m`` retained
                for backwards compatibility with callers that used the
                old multi-lap centerline path.
            name: Name stored on the returned :class:`Track` object.

        Returns:
            A :class:`Track` whose segments represent the GPS centerline or
            the legacy single-lap fallback geometry.

        Raises:
            RuntimeError: If no laps are detected.
            ValueError: If no segments are produced.
        """
        if df is None:
            from fsae_sim.data.loader import load_aim_csv  # local import avoids circular
            _metadata, df = load_aim_csv(aim_csv_path)

        from fsae_sim.analysis.validation import detect_lap_boundaries

        lap_boundaries = detect_lap_boundaries(df)
        if len(lap_boundaries) < 1:
            raise RuntimeError(
                "detect_lap_boundaries() returned no laps; cannot build track."
            )

        sigma_m = (
            centerline_sigma_m if centerline_sigma_m is not None
            else smooth_distance_m
        )

        if (
            len(lap_boundaries) >= _MIN_LAPS_FOR_GPS_AVERAGE
            and {"GPS Latitude", "GPS Longitude"}.issubset(df.columns)
        ):
            return cls._from_gps_centerline(
                df=df,
                lap_boundaries=lap_boundaries,
                bin_size_m=bin_size_m,
                centerline_sigma_m=sigma_m,
                name=name,
            )

        return cls._from_single_lap_latacc(
            df=df,
            lap_boundaries=lap_boundaries,
            bin_size_m=bin_size_m,
            smooth_distance_m=sigma_m,
            name=name,
        )

    # ------------------------------------------------------------------ #
    # New construction path: GPS-coord centerline averaged across laps    #
    # ------------------------------------------------------------------ #

    @classmethod
    def _from_gps_centerline(
        cls,
        *,
        df: pd.DataFrame,
        lap_boundaries: list[tuple[int, int, float]],
        bin_size_m: float,
        centerline_sigma_m: float,
        name: str,
    ) -> "Track":
        """Build a Track from the GPS-coord-averaged centerline."""
        # ---- 1. Set up the local cartesian frame ------------------------
        # Use the median latitude across all GPS samples for the cosine
        # correction.  A constant scale factor for the whole 1 km circuit
        # introduces sub-millimetre error vs an exact ECEF projection.
        lat_med = float(np.median(df["GPS Latitude"].values))
        m_per_deg_lon = _M_PER_DEG_LAT * float(np.cos(np.radians(lat_med)))

        # Anchor (x, y) = (0, 0) at the start of lap 1 so distances align
        # with downstream lap-relative coordinates.
        ref_idx = lap_boundaries[0][0]
        lat0 = float(df["GPS Latitude"].iloc[ref_idx])
        lon0 = float(df["GPS Longitude"].iloc[ref_idx])

        # ---- 2. Common arc-length grid ----------------------------------
        # All laps share the same physical track but record slightly
        # different ``Distance on GPS Speed`` totals (~1006 m mean, +-3 m).
        # Use the mean lap distance as the canonical lap length and resample
        # each lap onto the same normalised arc-length grid.
        lap_lengths = np.array(
            [lap_d for _, _, lap_d in lap_boundaries], dtype=float
        )
        mean_lap_length = float(lap_lengths.mean())

        n_grid = int(math.ceil(mean_lap_length / bin_size_m)) + 1
        s_grid = np.linspace(0.0, mean_lap_length, n_grid)

        # ---- 3. Project + resample each lap -----------------------------
        x_stack: list[np.ndarray] = []
        y_stack: list[np.ndarray] = []
        slope_stack: list[np.ndarray] = []
        weights: list[float] = []

        has_slope = "GPS Slope" in df.columns
        has_pos_acc = "GPS PosAccuracy" in df.columns

        for s_idx, e_idx, lap_d in lap_boundaries:
            lap = df.iloc[s_idx:e_idx]

            lat = lap["GPS Latitude"].values
            lon = lap["GPS Longitude"].values
            dist = lap["Distance on GPS Speed"].values

            # Drop laps that have any NaN in the GPS lat/lon channel.
            if (
                not np.all(np.isfinite(lat))
                or not np.all(np.isfinite(lon))
                or not np.all(np.isfinite(dist))
            ):
                continue
            if dist[-1] - dist[0] <= 0.0:
                continue

            # Cosine-corrected local cartesian projection.
            x = (lon - lon0) * m_per_deg_lon
            y = (lat - lat0) * _M_PER_DEG_LAT

            # Re-zero distance to lap-relative and rescale to the canonical
            # mean lap length so every lap shares the same s-axis.
            dist_lap = dist - dist[0]
            scale = mean_lap_length / dist_lap[-1]
            s_lap = dist_lap * scale

            # Linear interpolation onto the common grid.  np.interp pins
            # endpoints to the input data, which is exactly what we want
            # since s_lap[0] = 0 and s_lap[-1] = mean_lap_length.
            x_g = np.interp(s_grid, s_lap, x)
            y_g = np.interp(s_grid, s_lap, y)

            # GPS-quality weight: laps with PosAccuracy at the cold-fix
            # sentinel get downweighted.  Default is 1.0 if the channel
            # isn't present.
            if has_pos_acc:
                pos_acc = lap["GPS PosAccuracy"].values
                bad_frac = float(
                    np.mean(pos_acc == _GPS_POS_ACC_BAD)
                )
                w = max(0.0, 1.0 - bad_frac)
                if w <= 0.0:
                    continue
            else:
                w = 1.0

            x_stack.append(x_g)
            y_stack.append(y_g)
            weights.append(w)

            if has_slope:
                slope = np.nan_to_num(lap["GPS Slope"].values, nan=0.0)
                slope_stack.append(np.interp(s_grid, s_lap, slope))

        if not x_stack:
            raise ValueError(
                "No laps with valid GPS Lat/Lon found; cannot build "
                "centerline."
            )

        X = np.stack(x_stack, axis=0)  # (n_laps, n_grid)
        Y = np.stack(y_stack, axis=0)
        w = np.asarray(weights, dtype=float)
        w_sum = float(w.sum())

        # Weighted average across laps.  This is the clean centerline.
        x_mean = (X * w[:, None]).sum(axis=0) / w_sum
        y_mean = (Y * w[:, None]).sum(axis=0) / w_sum

        # Per-lap residual statistics (logged for diagnostics).
        x_std = float(np.median(X.std(axis=0)))
        y_std = float(np.median(Y.std(axis=0)))
        logger.info(
            "GPS centerline built from %d laps: "
            "median across-lap stddev x=%.3f m, y=%.3f m, mean lap=%.2f m",
            len(x_stack), x_std, y_std, mean_lap_length,
        )

        # ---- 4. Smooth the centerline with a periodic Gaussian ----------
        # The track is a closed loop, so we wrap the convolution to avoid
        # endpoint bias.  Sigma is in metres; convert to grid samples.
        ds = mean_lap_length / (n_grid - 1)
        sigma_samples = max(centerline_sigma_m / ds, 1e-6)

        x_smooth = _periodic_gaussian_filter(x_mean, sigma_samples)
        y_smooth = _periodic_gaussian_filter(y_mean, sigma_samples)

        # ---- 5. Curvature from the centerline geometry ------------------
        # Use central-difference gradients with periodic wrap so the start
        # and end stitch together cleanly (closed track).  kappa for a 2D
        # parametric curve is (dx*ddy - dy*ddx) / (dx^2 + dy^2)^(3/2).
        # The cross-product sign of np.gradient (x=east, y=north) is
        # positive for counter-clockwise (left) turns; the existing
        # codebase encodes positive = right turn, so we negate.
        dx = _periodic_gradient(x_smooth, ds)
        dy = _periodic_gradient(y_smooth, ds)
        ddx = _periodic_gradient(dx, ds)
        ddy = _periodic_gradient(dy, ds)

        num = dx * ddy - dy * ddx
        den = (dx * dx + dy * dy) ** 1.5
        with np.errstate(invalid="ignore", divide="ignore"):
            kappa_grid = np.where(den > 1e-9, num / den, 0.0)
        kappa_grid = -kappa_grid

        # ---- 6. Per-grid grade (averaged GPS Slope) ---------------------
        if slope_stack:
            slope_arr = np.stack(slope_stack, axis=0)
            slope_mean = (slope_arr * w[:, None]).sum(axis=0) / w_sum
            grade_grid = np.tan(slope_mean * (math.pi / 180.0))
        else:
            grade_grid = np.zeros(n_grid)
        if mean_lap_length > 0.0:
            grade_grid = grade_grid - float(
                np.trapz(grade_grid, s_grid) / mean_lap_length
            )

        # ---- 7. Bin onto the requested segment grid ---------------------
        n_bins = int(math.ceil(mean_lap_length / bin_size_m))
        if n_bins == 0:
            raise ValueError(
                f"Lap length {mean_lap_length:.1f} m is shorter than "
                f"bin size {bin_size_m} m; cannot create any segments."
            )

        segment_lengths = [bin_size_m] * n_bins
        residual = mean_lap_length - (n_bins - 1) * bin_size_m
        if residual <= 0.0:
            residual = bin_size_m
        segment_lengths[-1] = residual

        # Each segment is the bin centred on its midpoint sampled from the
        # high-resolution centerline kappa.  This preserves curvature peaks
        # because the centerline already lives at 0.5 m resolution.
        segment_centers = np.array([
            sum(segment_lengths[:i]) + segment_lengths[i] / 2.0
            for i in range(n_bins)
        ])
        seg_kappa = np.interp(segment_centers, s_grid, kappa_grid)
        seg_grade = np.interp(segment_centers, s_grid, grade_grid)

        segments: list[Segment] = []
        cumulative = 0.0
        for i in range(n_bins):
            segments.append(
                Segment(
                    index=i,
                    distance_start_m=float(cumulative),
                    length_m=float(segment_lengths[i]),
                    curvature=float(seg_kappa[i]),
                    grade=float(seg_grade[i]),
                )
            )
            cumulative += segment_lengths[i]

        kappa_max = float(np.abs(kappa_grid).max())
        r_min = (1.0 / kappa_max) if kappa_max > 1e-9 else float("inf")
        logger.info(
            "Track curvature: max|kappa|=%.4f /m (R_min=%.2f m), "
            "p95|kappa|=%.4f, p99|kappa|=%.4f",
            kappa_max,
            r_min,
            float(np.percentile(np.abs(kappa_grid), 95)),
            float(np.percentile(np.abs(kappa_grid), 99)),
        )

        return cls(
            name=name,
            segments=segments,
            source="telemetry_gps_centerline",
            provenance={
                "lap_count": len(x_stack),
                "mean_lap_length_m": mean_lap_length,
                "centerline_sigma_m": centerline_sigma_m,
                "bin_size_m": bin_size_m,
            },
        )

    # ------------------------------------------------------------------ #
    # Legacy fallback: single-lap LatAcc / v^2 with YawRate fill-in       #
    # (used only when there are too few laps for the GPS averaging path)  #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_telemetry_per_lap(
        cls,
        aim_csv_path: str | Path | None = None,
        *,
        df: pd.DataFrame | None = None,
        bin_size_m: float = _SEGMENT_BIN_M,
        smooth_distance_m: float = _CENTERLINE_SIGMA_M,
        name: str = "Michigan Endurance per-lap",
    ) -> "Track":
        """Build a stitched track where each lap uses its own GPS data.

        For each detected lap, attempts a single-lap reconstruction from
        that lap's own GPS latitude/longitude samples. If a lap's GPS
        coordinate coverage is insufficient, it falls back to curvature
        from GPS LatAcc, then ``YawRate / v``. If a lap's coverage is
        still insufficient,
        substitutes segments from a fallback single-lap reconstruction
        built from the first usable lap.

        The returned :class:`Track` concatenates per-lap segments so the
        engine can run with ``num_laps=1`` and still see lap-to-lap
        variation in curvature, length, and grade. Each segment is tagged
        with its source ``lap_index`` so per-lap analysis can group sim
        records correctly.

        Args:
            aim_csv_path: Path to the AiM Race Studio CSV export.
                Mutually exclusive with ``df``.
            df: Pre-loaded AiM DataFrame.
            bin_size_m: Length of each output segment in metres.
            smooth_distance_m: Curvature smoothing scale per lap.
            name: Name stored on the returned :class:`Track`.

        Raises:
            RuntimeError: If no laps are detected.
            ValueError: If no usable lap exists to seed the fallback.
        """
        if df is None:
            from fsae_sim.data.loader import load_aim_csv  # local import
            _meta, df = load_aim_csv(aim_csv_path)

        from fsae_sim.analysis.validation import detect_lap_boundaries
        lap_boundaries = detect_lap_boundaries(df)
        if not lap_boundaries:
            raise RuntimeError(
                "detect_lap_boundaries() returned no laps; cannot build "
                "per-lap track."
            )

        # Try a per-lap build for every detected lap, then patch any laps
        # that failed quality checks with the first lap that succeeded
        # (which itself acts as the fallback "average" lap for this car).
        per_lap: list[tuple[int, list[Segment], dict] | None] = []
        for lap_idx, (s_idx, e_idx, _len) in enumerate(lap_boundaries):
            try:
                segs, prov = cls._build_lap_segments_gps_coords(
                    df=df,
                    lap_idx=lap_idx,
                    lap_boundary=(s_idx, e_idx, _len),
                    bin_size_m=bin_size_m,
                    smooth_distance_m=smooth_distance_m,
                )
            except ValueError:
                try:
                    segs, prov = cls._build_lap_segments(
                        df=df,
                        lap_idx=lap_idx,
                        lap_boundary=(s_idx, e_idx, _len),
                        bin_size_m=bin_size_m,
                        smooth_distance_m=smooth_distance_m,
                    )
                except ValueError:
                    per_lap.append(None)
                    continue
            per_lap.append((lap_idx, segs, prov))

        successful = [p for p in per_lap if p is not None]
        if not successful:
            raise ValueError(
                "No detected lap produced valid GPS-derived segments; "
                "cannot stitch per-lap track."
            )
        fallback_lap_idx, fallback_segs, fallback_prov = successful[0]

        # Walk laps in order, substituting the fallback for failed ones.
        stitched_segments: list[Segment] = []
        provenance_per_lap: list[dict] = []
        cum_distance = 0.0
        global_seg_idx = 0
        bad_laps: list[int] = []
        for lap_idx, entry in enumerate(per_lap):
            if entry is not None:
                _, lap_segs, lap_prov = entry
                lap_prov_for_track = dict(lap_prov)
                lap_prov_for_track["substituted_from_fallback"] = False
            else:
                lap_segs = fallback_segs
                lap_prov_for_track = dict(fallback_prov)
                lap_prov_for_track["substituted_from_fallback"] = True
                lap_prov_for_track["fallback_lap_zero_based"] = (
                    fallback_lap_idx
                )
                bad_laps.append(lap_idx)

            for s in lap_segs:
                stitched_segments.append(
                    Segment(
                        index=global_seg_idx,
                        distance_start_m=float(cum_distance),
                        length_m=s.length_m,
                        curvature=s.curvature,
                        grade=s.grade,
                        grip_factor=s.grip_factor,
                        lap_index=lap_idx,
                    )
                )
                cum_distance += s.length_m
                global_seg_idx += 1
            provenance_per_lap.append(lap_prov_for_track)

        return cls(
            name=name,
            segments=stitched_segments,
            source="telemetry_per_lap_gps_stitched",
            provenance={
                "num_laps": len(lap_boundaries),
                "bad_laps_zero_based": tuple(bad_laps),
                "fallback_lap_zero_based": fallback_lap_idx,
                "bin_size_m": bin_size_m,
                "smooth_distance_m": smooth_distance_m,
                "total_distance_m": cum_distance,
                "per_lap": provenance_per_lap,
            },
        )

    @classmethod
    def _build_lap_segments_gps_coords(
        cls,
        *,
        df: pd.DataFrame,
        lap_idx: int,
        lap_boundary: tuple[int, int, float],
        bin_size_m: float,
        smooth_distance_m: float,
    ) -> tuple[list[Segment], dict]:
        """Construct one lap's segments from GPS latitude/longitude."""
        if not {"GPS Latitude", "GPS Longitude"}.issubset(df.columns):
            raise ValueError("GPS latitude/longitude columns are required.")

        s_idx, e_idx, lap_length = lap_boundary
        lap = df.iloc[s_idx:e_idx].copy()
        if len(lap) < 4 or lap_length <= 0.0:
            raise ValueError(f"Lap {lap_idx}: not enough GPS coord samples.")

        speed_col = _telemetry_speed_col(lap)
        good_mask = (
            (lap[speed_col] > _GPS_SPEED_MIN_KMH)
            & np.isfinite(lap["GPS Latitude"])
            & np.isfinite(lap["GPS Longitude"])
            & np.isfinite(lap["Distance on GPS Speed"])
        )
        if "GPS PosAccuracy" in lap.columns:
            good_mask = (
                good_mask
                & (lap["GPS PosAccuracy"] != _GPS_POS_ACC_BAD)
            )
        good = lap[good_mask].reset_index(drop=True)
        if len(good) < 4:
            raise ValueError(
                f"Lap {lap_idx}: too few valid GPS coordinate samples."
            )

        lat = good["GPS Latitude"].values.astype(float)
        lon = good["GPS Longitude"].values.astype(float)
        dist = good["Distance on GPS Speed"].values.astype(float)
        lap_start_dist = float(df["Distance on GPS Speed"].iloc[s_idx])
        s_raw = dist - lap_start_dist
        order = np.argsort(s_raw)
        s_raw = s_raw[order]
        lat = lat[order]
        lon = lon[order]

        # Collapse duplicate or non-increasing distance samples before
        # interpolation; wheel-speed distance can have tiny plateaus in
        # very slow sections.
        keep = np.concatenate(([True], np.diff(s_raw) > 1e-6))
        s_raw = s_raw[keep]
        lat = lat[keep]
        lon = lon[keep]
        if len(s_raw) < 4:
            raise ValueError(
                f"Lap {lap_idx}: GPS coordinate distance is not monotonic."
            )

        lat_med = float(np.median(lat))
        m_per_deg_lon = _M_PER_DEG_LAT * float(np.cos(np.radians(lat_med)))
        lat0 = float(lat[0])
        lon0 = float(lon[0])
        x = (lon - lon0) * m_per_deg_lon
        y = (lat - lat0) * _M_PER_DEG_LAT

        n_grid = int(math.ceil(lap_length / bin_size_m)) + 1
        s_grid = np.linspace(0.0, lap_length, n_grid)
        x_g = np.interp(s_grid, s_raw, x)
        y_g = np.interp(s_grid, s_raw, y)

        # Enforce closed-course continuity before periodic derivatives.
        x_g[-1] = x_g[0]
        y_g[-1] = y_g[0]

        ds = lap_length / (n_grid - 1)
        sigma_samples = max(smooth_distance_m / ds, 1e-6)
        x_smooth = _periodic_gaussian_filter(x_g, sigma_samples)
        y_smooth = _periodic_gaussian_filter(y_g, sigma_samples)

        dx = _periodic_gradient(x_smooth, ds)
        dy = _periodic_gradient(y_smooth, ds)
        ddx = _periodic_gradient(dx, ds)
        ddy = _periodic_gradient(dy, ds)
        den = (dx * dx + dy * dy) ** 1.5
        with np.errstate(invalid="ignore", divide="ignore"):
            kappa_grid = np.where(
                den > 1e-9, -(dx * ddy - dy * ddx) / den, 0.0,
            )
        kappa_grid = np.nan_to_num(
            kappa_grid, nan=0.0, posinf=0.0, neginf=0.0,
        )

        if "GPS Slope" in good.columns:
            slope_raw = np.nan_to_num(good["GPS Slope"].values, nan=0.0)
            slope_raw = slope_raw[order][keep]
            slope_grid = np.interp(s_grid, s_raw, slope_raw)
            grade_grid = np.tan(slope_grid * (math.pi / 180.0))
        else:
            grade_grid = np.zeros(n_grid)
        if lap_length > 0.0:
            grade_grid = grade_grid - float(
                np.trapz(grade_grid, s_grid) / lap_length
            )

        n_bins = int(math.ceil(lap_length / bin_size_m))
        if n_bins == 0:
            raise ValueError(
                f"Lap length {lap_length:.1f} m is shorter than bin size "
                f"{bin_size_m} m; cannot create any segments."
            )

        segment_lengths = [bin_size_m] * n_bins
        residual = lap_length - (n_bins - 1) * bin_size_m
        if residual <= 0.0:
            residual = bin_size_m
        segment_lengths[-1] = residual

        segment_centers = np.array([
            sum(segment_lengths[:i]) + segment_lengths[i] / 2.0
            for i in range(n_bins)
        ])
        seg_kappa = np.interp(segment_centers, s_grid, kappa_grid)
        seg_grade = np.interp(segment_centers, s_grid, grade_grid)
        if segment_lengths:
            seg_grade = seg_grade - float(
                np.average(seg_grade, weights=np.asarray(segment_lengths))
            )

        segments: list[Segment] = []
        cumulative = 0.0
        for i, length in enumerate(segment_lengths):
            segments.append(
                Segment(
                    index=i,
                    distance_start_m=float(cumulative),
                    length_m=float(length),
                    curvature=float(seg_kappa[i]),
                    grade=float(seg_grade[i]),
                )
            )
            cumulative += length

        return segments, {
            "lap_index_zero_based": lap_idx,
            "curvature_source": "GPS Lat/Lon",
            "bin_size_m": bin_size_m,
            "smooth_distance_m": smooth_distance_m,
            "lap_length_m": float(lap_length),
            "valid_gps_coord_samples": int(len(good)),
        }

    @classmethod
    def _build_lap_segments(
        cls,
        *,
        df: pd.DataFrame,
        lap_idx: int,
        lap_boundary: tuple[int, int, float],
        bin_size_m: float,
        smooth_distance_m: float,
    ) -> tuple[list[Segment], dict]:
        """Construct segments from a single detected lap.

        Pure helper used by both ``from_telemetry_per_lap`` (called once
        per lap) and ``_from_single_lap_latacc`` (called once with the
        first usable lap). Raises ``ValueError`` when GPS quality is
        insufficient — the caller decides how to recover.
        """
        s_idx, e_idx, _length = lap_boundary
        _slice = df.iloc[s_idx:e_idx]
        _lat_acc = _slice["GPS LatAcc"].values
        lat_valid_frac = (
            np.sum(np.isfinite(_lat_acc)) / max(len(_lat_acc), 1)
        )
        yaw_valid_frac = 0.0
        if "YawRate" in _slice.columns:
            _yaw = _slice["YawRate"].values
            yaw_valid_frac = (
                np.sum(np.isfinite(_yaw)) / max(len(_yaw), 1)
            )

        if lat_valid_frac < 0.8 and yaw_valid_frac < 0.8:
            raise ValueError(
                f"Lap {lap_idx}: neither GPS LatAcc ({lat_valid_frac:.0%}) "
                f"nor YawRate ({yaw_valid_frac:.0%}) reaches 80% validity."
            )

        speed_col = _telemetry_speed_col(_slice)
        good_mask = _slice[speed_col] > _GPS_SPEED_MIN_KMH
        if "GPS PosAccuracy" in _slice.columns:
            good_mask = (
                good_mask
                & (_slice["GPS PosAccuracy"] != _GPS_POS_ACC_BAD)
            )
        lap_df = _slice[good_mask].reset_index(drop=True)
        if lap_df.empty:
            raise ValueError(
                f"Lap {lap_idx}: no samples pass GPS quality filters."
            )

        lap_start_dist = float(df["Distance on GPS Speed"].iloc[s_idx])
        curv_source = (
            "GPS LatAcc" if lat_valid_frac >= 0.8 else "YawRate"
        )

        segs, lap_length = cls._segments_from_lap_df(
            lap_df=lap_df,
            lap_start_dist=lap_start_dist,
            bin_size_m=bin_size_m,
            smooth_distance_m=smooth_distance_m,
        )
        provenance = {
            "lap_index_zero_based": lap_idx,
            "curvature_source": curv_source,
            "bin_size_m": bin_size_m,
            "smooth_distance_m": smooth_distance_m,
            "lap_length_m": lap_length,
        }
        return segs, provenance

    @classmethod
    def _from_single_lap_latacc(
        cls,
        *,
        df: pd.DataFrame,
        lap_boundaries: list[tuple[int, int, float]],
        bin_size_m: float,
        smooth_distance_m: float,
        name: str,
    ) -> "Track":
        """Build a track from the first lap whose GPS data is usable."""
        for lap_idx, (s_idx, e_idx, _length) in enumerate(lap_boundaries):
            try:
                segs, prov = cls._build_lap_segments(
                    df=df,
                    lap_idx=lap_idx,
                    lap_boundary=(s_idx, e_idx, _length),
                    bin_size_m=bin_size_m,
                    smooth_distance_m=smooth_distance_m,
                )
            except ValueError:
                continue

            logger.info(
                "Track built from detect_lap_boundaries lap %d "
                "(curvature source=%s, length=%.1f m)",
                lap_idx + 1,
                prov["curvature_source"],
                prov["lap_length_m"],
            )
            return cls(
                name=name,
                segments=segs,
                source="telemetry_single_lap",
                provenance=prov,
            )

        raise ValueError(
            "No lap with sufficient GPS LatAcc or YawRate data found. "
            "Need at least 80% valid samples of one channel in one "
            "detect_lap_boundaries lap."
        )

    @staticmethod
    def _segments_from_lap_df(
        *,
        lap_df: pd.DataFrame,
        lap_start_dist: float,
        bin_size_m: float,
        smooth_distance_m: float,
    ) -> tuple[list[Segment], float]:
        """Bin a quality-filtered lap slice into Segment records.

        Returns ``(segments, lap_length_m)``. Segment indices are
        lap-local (0..n-1); callers that stitch multiple laps must
        re-index globally.
        """
        dist_in_lap = lap_df["Distance on GPS Speed"].values - lap_start_dist

        v_ms = lap_df[_telemetry_speed_col(lap_df)].values * (1_000.0 / 3_600.0)
        a_lat_raw = lap_df["GPS LatAcc"].values.copy()
        a_lat_valid_mask = np.isfinite(a_lat_raw)
        a_lat_raw = np.nan_to_num(a_lat_raw, nan=0.0)
        a_lat_ms2 = a_lat_raw * 9.81
        if "GPS Slope" in lap_df.columns:
            slope_deg = np.nan_to_num(
                lap_df["GPS Slope"].values, nan=0.0,
            )
        else:
            slope_deg = np.zeros(len(lap_df))

        valid_v = v_ms > _V_MIN_FOR_CURVATURE_MS
        v_safe = np.where(valid_v, v_ms, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            k_raw = a_lat_ms2 / (v_safe ** 2)

        if "YawRate" in lap_df.columns:
            yaw_rate_deg_s = np.nan_to_num(
                lap_df["YawRate"].values, nan=0.0,
            )
            yaw_rate_rad_s = yaw_rate_deg_s * (math.pi / 180.0)
            need_fallback = (~a_lat_valid_mask) & valid_v
            if need_fallback.any():
                k_fallback = np.zeros_like(k_raw)
                k_fallback[need_fallback] = (
                    yaw_rate_rad_s[need_fallback] / v_ms[need_fallback]
                )
                k_raw = np.where(need_fallback, k_fallback, k_raw)

        low_speed = ~valid_v
        if low_speed.any():
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
                    sign = np.sign(a_lat_ms2[radius_ok])
                    sign = np.where(sign == 0.0, 1.0, sign)
                    k_raw[radius_ok] = sign / radius[radius_ok]
                    filled_from_radius = radius_ok

            still_missing = low_speed & ~filled_from_radius & ~np.isfinite(k_raw)
            if still_missing.any():
                idx = np.arange(len(k_raw))
                known = np.isfinite(k_raw) & ~still_missing
                if known.any():
                    k_raw[still_missing] = np.interp(
                        idx[still_missing], idx[known], k_raw[known]
                    )
                else:
                    k_raw[still_missing] = 0.0

        k_raw = np.nan_to_num(k_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if "GPS Radius" in lap_df.columns:
            radius = lap_df["GPS Radius"].values.astype(float)
            straight_radius = (
                np.isfinite(radius)
                & (radius >= _GPS_RADIUS_STRAIGHT)
            )
            k_raw[straight_radius] = 0.0
        lap_length = float(dist_in_lap[-1])
        n_bins = int(math.ceil(lap_length / bin_size_m))
        if n_bins == 0:
            raise ValueError(
                f"Lap length {lap_length:.1f} m is shorter than bin size "
                f"{bin_size_m} m; cannot create any segments."
            )

        segment_lengths = [bin_size_m] * n_bins
        residual = lap_length - (n_bins - 1) * bin_size_m
        if residual <= 0.0:
            residual = bin_size_m
        segment_lengths[-1] = residual

        grade_raw = np.tan(slope_deg * (math.pi / 180.0))
        if lap_length > 0.0 and len(dist_in_lap) > 1:
            # Closed-course constraint: one lap must have zero net
            # elevation change. AiM GPS Slope carries a small positive
            # bias in the Michigan endurance data; integrating it as-is
            # creates an impossible uphill-only circuit and injects
            # artificial work into the simulator.
            grade_bias = float(np.trapz(grade_raw, dist_in_lap) / lap_length)
            grade_raw = grade_raw - grade_bias

        raw_curvatures: list[float] = []
        raw_grades: list[float] = []
        for i in range(n_bins):
            bin_lo = i * bin_size_m
            bin_hi = bin_lo + segment_lengths[i]
            idx_mask = (dist_in_lap >= bin_lo) & (dist_in_lap < bin_hi)
            if idx_mask.any():
                raw_curvatures.append(float(np.median(k_raw[idx_mask])))
                raw_grades.append(float(np.mean(grade_raw[idx_mask])))
            else:
                prev_k = raw_curvatures[-1] if raw_curvatures else 0.0
                prev_g = raw_grades[-1] if raw_grades else 0.0
                raw_curvatures.append(prev_k)
                raw_grades.append(prev_g)

        smooth_window = max(1, int(round(smooth_distance_m / bin_size_m)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        smoothed_k = (
            pd.Series(raw_curvatures)
            .rolling(window=smooth_window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        smoothed_grade = (
            pd.Series(raw_grades)
            .rolling(window=smooth_window, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        if segment_lengths:
            seg_grade_bias = float(
                np.average(smoothed_grade, weights=np.asarray(segment_lengths))
            )
            smoothed_grade = smoothed_grade - seg_grade_bias

        segments: list[Segment] = []
        cumulative = 0.0
        for i in range(n_bins):
            segments.append(
                Segment(
                    index=i,
                    distance_start_m=float(cumulative),
                    length_m=float(segment_lengths[i]),
                    curvature=float(smoothed_k[i]),
                    grade=float(smoothed_grade[i]),
                )
            )
            cumulative += segment_lengths[i]

        return segments, lap_length


# ----------------------------------------------------------------------
# Periodic helpers used by the GPS-coord centerline construction
# ----------------------------------------------------------------------

def _periodic_gaussian_filter(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoother with periodic (wrap-around) boundary.

    The track is a closed loop, so we want the convolution kernel to wrap
    rather than reflect or zero-pad.  ``scipy.ndimage.gaussian_filter1d``
    supports ``mode='wrap'`` which does exactly this.
    """
    if sigma <= 0.0:
        return arr.copy()
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma, mode="wrap")


def _periodic_gradient(arr: np.ndarray, ds: float) -> np.ndarray:
    """Central-difference gradient with periodic wrap.

    ``np.gradient`` uses one-sided differences at the endpoints, which on a
    closed track introduces a discontinuity at the start/end seam.  Wrapping
    one sample on each side makes the difference identical at every point
    and keeps the curvature continuous across s = 0 / s = lap_length.
    """
    n = len(arr)
    if n < 3:
        return np.zeros_like(arr)
    # arr is sampled at s = 0, ds, 2*ds, ..., (n-1)*ds.  Because s_grid uses
    # np.linspace(0, mean_lap_length, n_grid), the endpoints are coincident
    # in physical-track space, so we drop the duplicate sample by treating
    # the period as (n-1)*ds.
    extended = np.concatenate(([arr[-2]], arr, [arr[1]]))
    grad = (extended[2:] - extended[:-2]) / (2.0 * ds)
    return grad
