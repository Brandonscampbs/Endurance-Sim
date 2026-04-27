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

# Centerline smoothing sigma in metres. Applied to the (x, y) centerline
# Gaussian-filtered with periodic boundary so closed-track wrap is exact.
# 1.0 m matches a single-sample GPS pixel — enough to reject driver-line
# variance across laps without rounding off real hairpins.
_CENTERLINE_SIGMA_M: float = 1.0

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
        smooth_distance_m: float = _CENTERLINE_SIGMA_M,
        centerline_sigma_m: float | None = None,
        name: str = "Michigan Endurance",
    ) -> "Track":
        """Extract track geometry from AiM GPS telemetry.

        Builds the centerline from **lap 1 alone**: the lap's GPS (lat, lon)
        samples are projected to a local cartesian frame and the curvature
        is computed pointwise from per-sample lateral acceleration (or the
        ``YawRate / v`` fallback when GPS LatAcc is missing).

        Why lap 1 only? Multi-lap GPS averaging using each lap's
        ``Distance on GPS Speed`` axis suffers from per-lap distance drift
        (~+/-3 m): rescaling each lap to a canonical length still leaves
        the same physical apex point landing at slightly different s-values
        across laps. Averaging (x, y) at a common s smears the centerline
        and shifts apex locations by 10-20 m. Lap 1 is the reference frame
        the comparison harness (``sim_compare.py``) uses to interpolate
        sim-vs-telem residuals, so building the track from lap 1 alone
        guarantees that sim distance and telem distance see the same
        physical features at the same s. Multi-lap averaging is a future
        optimisation; until the per-lap alignment uses physical (x, y)
        instead of rescaled distance, lap 1 is the correct choice.

        Falls back to the legacy single-lap ``a_lat / v^2`` extraction when
        fewer than :data:`_MIN_LAPS_FOR_GPS_AVERAGE` laps are present (so
        unit tests with synthetic short telemetry still exercise that
        path).  When the single-lap path is taken with full Michigan data
        it picks lap 1 directly; the legacy path already supports
        YawRate-derived curvature, which lap 1 needs because its GPS
        LatAcc channel is empty.

        Args:
            aim_csv_path: Path to the AiM Race Studio CSV export.
                Mutually exclusive with ``df``.
            df: Pre-loaded AiM DataFrame.  Must contain GPS Speed,
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
            A :class:`Track` whose segments represent lap 1's geometry.

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

        # Lap 1 alignment: build the track from the first detected lap so
        # the canonical s-axis coincides with the comparison harness's
        # lap-1 frame (no rescaled-distance smearing of the apex location).
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

        return cls(name=name, segments=segments)

    # ------------------------------------------------------------------ #
    # Legacy fallback: single-lap LatAcc / v^2 with YawRate fill-in       #
    # (used only when there are too few laps for the GPS averaging path)  #
    # ------------------------------------------------------------------ #

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
        """Single-lap extraction.

        Builds the track from the first detected lap that has either
        sufficient GPS LatAcc validity OR sufficient YawRate validity
        (curvature is then computed from ``yaw_rate / v``).  In the
        Michigan 2025 dataset lap 1 has 0% LatAcc but 100% YawRate, so
        the YawRate-only path is what the production pipeline takes.
        """
        # ---- Pick the first lap that has either GPS LatAcc or YawRate
        # validity above the 80% threshold.  We prefer the *first* such
        # lap (rather than the first LatAcc-rich lap) so the resulting
        # track shares its s-axis with lap 1 of the comparison harness.
        lap_df: pd.DataFrame = pd.DataFrame()
        lap_start_dist: float = 0.0
        chosen_lap_idx: int = -1
        chosen_curv_source: str = ""

        for lap_idx, (s_idx, e_idx, _length) in enumerate(lap_boundaries):
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

            # Accept the lap if either curvature source is usable.
            if lat_valid_frac < 0.8 and yaw_valid_frac < 0.8:
                continue

            good_mask = _slice["GPS Speed"] > _GPS_SPEED_MIN_KMH
            if "GPS PosAccuracy" in _slice.columns:
                good_mask = (
                    good_mask
                    & (_slice["GPS PosAccuracy"] != _GPS_POS_ACC_BAD)
                )
            if "GPS Radius" in _slice.columns:
                good_mask = (
                    good_mask
                    & (_slice["GPS Radius"] != _GPS_RADIUS_STRAIGHT)
                )
            lap_df = _slice[good_mask].reset_index(drop=True)
            if lap_df.empty:
                continue
            lap_start_dist = float(
                df["Distance on GPS Speed"].iloc[s_idx]
            )
            chosen_lap_idx = lap_idx
            chosen_curv_source = (
                "GPS LatAcc" if lat_valid_frac >= 0.8 else "YawRate"
            )
            break

        if lap_df.empty:
            raise ValueError(
                "No lap with sufficient GPS LatAcc or YawRate data found. "
                "Need at least 80% valid samples of one channel in one "
                "detect_lap_boundaries lap."
            )

        logger.info(
            "Track built from detect_lap_boundaries lap %d "
            "(cum_dist start=%.1f m, %d samples, curvature source=%s)",
            chosen_lap_idx + 1,
            lap_start_dist,
            len(lap_df),
            chosen_curv_source,
        )

        dist_in_lap: np.ndarray = lap_df["Distance on GPS Speed"].values - lap_start_dist

        v_ms: np.ndarray = lap_df["GPS Speed"].values * (1_000.0 / 3_600.0)
        a_lat_raw: np.ndarray = lap_df["GPS LatAcc"].values.copy()
        a_lat_valid_mask: np.ndarray = np.isfinite(a_lat_raw)
        a_lat_raw = np.nan_to_num(a_lat_raw, nan=0.0)
        a_lat_ms2: np.ndarray = a_lat_raw * 9.81
        if "GPS Slope" in lap_df.columns:
            slope_deg: np.ndarray = np.nan_to_num(
                lap_df["GPS Slope"].values, nan=0.0,
            )
        else:
            slope_deg = np.zeros(len(lap_df))

        valid_v = v_ms > _V_MIN_FOR_CURVATURE_MS
        v_safe = np.where(valid_v, v_ms, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            k_raw: np.ndarray = a_lat_ms2 / (v_safe ** 2)

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

        grade_raw: np.ndarray = np.tan(slope_deg * (math.pi / 180.0))

        lap_length: float = float(dist_in_lap[-1])
        n_bins: int = int(math.ceil(lap_length / bin_size_m))

        if n_bins == 0:
            raise ValueError(
                f"Lap length {lap_length:.1f} m is shorter than bin size "
                f"{bin_size_m} m; cannot create any segments."
            )

        segment_lengths: list[float] = [bin_size_m] * n_bins
        residual = lap_length - (n_bins - 1) * bin_size_m
        if residual <= 0.0:
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
                prev_k = raw_curvatures[-1] if raw_curvatures else 0.0
                prev_g = raw_grades[-1] if raw_grades else 0.0
                raw_curvatures.append(prev_k)
                raw_grades.append(prev_g)

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
