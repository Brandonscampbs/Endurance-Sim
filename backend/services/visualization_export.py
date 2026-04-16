import math

import numpy as np
from scipy.interpolate import CubicSpline

from backend.models.visualization import Frame, VisualizationResponse, WheelForce
from backend.services.sim_runner import (
    get_track,
    get_vehicle_config,
    run_single_lap_sim,
)
from backend.services.telemetry_service import get_lap_data, get_telemetry
from backend.services.track_service import build_track_xy, _load_best_lap_gps, get_track_data

_GRAVITY = 9.81


def compute_heading(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute heading angle (radians, 0=east, CCW positive) from XY path."""
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    return np.arctan2(dy, dx)


def distribute_drive_force(
    total_drive_n: float,
    total_regen_n: float,
) -> tuple[float, float, float, float]:
    """Distribute longitudinal force to wheels. CT-16EV is rear-wheel drive.

    Returns: (fl_fx, fr_fx, rl_fx, rr_fx)
    """
    net = total_drive_n - total_regen_n
    # Rear-wheel drive: all drive force on rear axle, split 50/50
    return (0.0, 0.0, net / 2, net / 2)


def _compute_lateral_forces(
    speed_ms: float,
    curvature: float,
    mass_kg: float,
) -> tuple[float, float, float, float]:
    """Estimate per-wheel lateral force from cornering.

    Sign convention (C8): Fy is signed by ``sign(curvature)`` so arrows in a
    left turn point opposite to a right turn. The magnitude uses
    ``|curvature|`` (matching the pattern in ``_compute_tire_loads``) and the
    sign is applied once via ``np.sign(curvature)``. Convention matches the
    frontend 3D renderer: positive Y = left of the car, so a left turn
    (positive curvature) produces positive Fy (centripetal force on the car
    points toward the turn center, i.e. to the left).

    Straight line (curvature == 0) cleanly yields zero lateral force because
    ``np.sign(0) == 0``.
    """
    # Magnitude first, then apply the curvature sign (mirrors
    # _compute_tire_loads which also takes abs(curvature) then multiplies by
    # a sign factor).
    lat_accel_mag = speed_ms ** 2 * abs(curvature)
    total_lat_force = mass_kg * lat_accel_mag * float(np.sign(curvature))
    front_share = 0.53
    front_total = total_lat_force * front_share
    rear_total = total_lat_force * (1 - front_share)
    return (front_total / 2, front_total / 2, rear_total / 2, rear_total / 2)


def _compute_tire_loads(
    speed_ms: float,
    curvature: float,
    longitudinal_g: float,
    mass_kg: float,
    cg_height_m: float = 0.2794,
    front_track_m: float = 1.194,
    rear_track_m: float = 1.168,
    wheelbase_m: float = 1.549,
    weight_dist_front: float = 0.53,
) -> tuple[float, float, float, float]:
    """Compute normal loads on each tire (FL, FR, RL, RR)."""
    weight = mass_kg * _GRAVITY
    front_static = weight * weight_dist_front / 2
    rear_static = weight * (1 - weight_dist_front) / 2

    long_transfer = mass_kg * longitudinal_g * _GRAVITY * cg_height_m / wheelbase_m / 2

    lat_g = speed_ms ** 2 * abs(curvature) / _GRAVITY
    lat_transfer_front = mass_kg * lat_g * _GRAVITY * cg_height_m * 0.53 / front_track_m
    lat_transfer_rear = mass_kg * lat_g * _GRAVITY * cg_height_m * 0.47 / rear_track_m

    sign = 1.0 if curvature >= 0 else -1.0

    fl = front_static - long_transfer + sign * lat_transfer_front
    fr = front_static - long_transfer - sign * lat_transfer_front
    rl = rear_static + long_transfer + sign * lat_transfer_rear
    rr = rear_static + long_transfer - sign * lat_transfer_rear

    return (max(fl, 0), max(fr, 0), max(rl, 0), max(rr, 0))


def _estimate_grip_utilization(
    fx: float, fy: float, fz: float, mu: float = 1.3,
) -> float:
    """Ratio of combined force to available grip (friction circle)."""
    if fz <= 0:
        return 0.0
    combined = math.sqrt(fx ** 2 + fy ** 2)
    available = fz * mu
    return min(combined / available, 1.0)


def _compute_roll_pitch(
    lat_g: float,
    long_g: float,
    roll_stiffness_nm_per_deg: float = 496.0,  # 238 front + 258 rear from DSS
    pitch_stiffness_nm_per_deg: float = 600.0,
    mass_kg: float = 288.0,
    cg_height_m: float = 0.2794,
) -> tuple[float, float]:
    """Estimate roll and pitch angles (radians) from load transfer."""
    roll_moment = mass_kg * abs(lat_g) * _GRAVITY * cg_height_m
    roll_deg = roll_moment / roll_stiffness_nm_per_deg if roll_stiffness_nm_per_deg > 0 else 0
    roll_rad = math.radians(roll_deg) * (1 if lat_g >= 0 else -1)

    pitch_moment = mass_kg * long_g * _GRAVITY * cg_height_m
    pitch_deg = pitch_moment / pitch_stiffness_nm_per_deg if pitch_stiffness_nm_per_deg > 0 else 0
    pitch_rad = math.radians(pitch_deg)

    return (roll_rad, pitch_rad)


def get_visualization_data(source: str = "sim") -> VisualizationResponse:
    """Build per-frame 3D visualization data for the best GPS quality lap."""
    aim_df = get_telemetry()
    track = get_track()
    vehicle = get_vehicle_config()
    mass_kg = vehicle.vehicle.mass_kg + 68
    gear_ratio = vehicle.powertrain.gear_ratio
    tire_radius = 0.2042  # Hoosier 16x7.5-10 LC0 unloaded radius

    # (#3) Pick the best lap ONCE and use the same lap for both centerline and frames
    from backend.services.telemetry_service import get_lap_gps_quality
    quality = get_lap_gps_quality()
    best = min(quality, key=lambda q: q["gps_quality_score"])
    best_lap_idx = best["lap_number"] - 1  # 0-based for _load_best_lap_gps

    # (#4) Build centerline from the same lap, no speed filter, so distance axes match
    lats, lons, dists = _load_best_lap_gps(aim_df, track, lap_idx=best_lap_idx)
    centerline = build_track_xy(lats, lons, dists, bin_size_m=1.0)
    cl_x = np.array([p.x for p in centerline])
    cl_y = np.array([p.y for p in centerline])
    cl_d = np.array([p.distance_m for p in centerline])

    cs_x = CubicSpline(cl_d, cl_x)
    cs_y = CubicSpline(cl_d, cl_y)

    if source == "sim":
        result = run_single_lap_sim()
        sim_df = result.states
        frames = _build_sim_frames(sim_df, cs_x, cs_y, cl_d, mass_kg)
        total_time = result.total_time_s
        sim_speeds = np.interp(cl_d, sim_df["distance_m"].values, sim_df["speed_kmh"].values)
        track_speeds = [round(float(v), 1) for v in sim_speeds]
    else:
        real_df = get_lap_data(best["lap_number"])
        frames = _build_real_frames(
            real_df, cs_x, cs_y, cl_d, mass_kg,
            gear_ratio=gear_ratio, tire_radius=tire_radius,
        )
        total_time = best["time_s"]
        real_speeds = np.interp(cl_d, real_df["lap_distance_m"].values, real_df["GPS Speed"].values)
        track_speeds = [round(float(v), 1) for v in real_speeds]

    return VisualizationResponse(
        lap_number=1,
        total_time_s=round(total_time, 2),
        total_frames=len(frames),
        frames=frames,
        track_centerline_x=[round(float(x), 3) for x in cl_x],
        track_centerline_y=[round(float(y), 3) for y in cl_y],
        track_speed_colors=track_speeds,
    )


def _build_sim_frames(
    sim_df, cs_x, cs_y, cl_d, mass_kg: float,
) -> list[Frame]:
    """Convert sim result rows into 3D frames."""
    frames = []
    max_d = cl_d[-1]

    for _, row in sim_df.iterrows():
        d = float(row["distance_m"]) % max_d
        d = min(d, max_d - 0.1)
        x = float(cs_x(d))
        y = float(cs_y(d))

        dx = float(cs_x(d, 1))
        dy = float(cs_y(d, 1))
        heading = math.atan2(dy, dx)

        speed_ms = float(row["speed_ms"])
        curvature = float(row["curvature"])
        drive_force = float(row["drive_force_n"])
        regen_force = float(row["regen_force_n"])

        fl_fx, fr_fx, rl_fx, rr_fx = distribute_drive_force(drive_force, regen_force)
        fl_fy, fr_fy, rl_fy, rr_fy = _compute_lateral_forces(speed_ms, curvature, mass_kg)

        long_g = float(row["net_force_n"]) / (mass_kg * _GRAVITY)
        fl_fz, fr_fz, rl_fz, rr_fz = _compute_tire_loads(
            speed_ms, curvature, long_g, mass_kg,
        )

        wheels = [
            WheelForce(fx=round(fl_fx, 1), fy=round(fl_fy, 1), fz=round(fl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fl_fx, fl_fy, fl_fz), 3)),
            WheelForce(fx=round(fr_fx, 1), fy=round(fr_fy, 1), fz=round(fr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fr_fx, fr_fy, fr_fz), 3)),
            WheelForce(fx=round(rl_fx, 1), fy=round(rl_fy, 1), fz=round(rl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rl_fx, rl_fy, rl_fz), 3)),
            WheelForce(fx=round(rr_fx, 1), fy=round(rr_fy, 1), fz=round(rr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rr_fx, rr_fy, rr_fz), 3)),
        ]

        lat_g = speed_ms ** 2 * curvature / _GRAVITY
        roll_rad, pitch_rad = _compute_roll_pitch(lat_g, long_g, mass_kg=mass_kg)

        frames.append(Frame(
            time_s=round(float(row["time_s"]), 3),
            distance_m=round(d, 2),
            x=round(x, 3),
            y=round(y, 3),
            heading_rad=round(heading, 4),
            speed_kmh=round(float(row["speed_kmh"]), 1),
            throttle_pct=round(float(row["throttle_pct"]) * 100, 1),
            brake_pct=round(float(row["brake_pct"]) * 100, 1),
            motor_rpm=round(float(row["motor_rpm"]), 0),
            motor_torque_nm=round(float(row["motor_torque_nm"]), 1),
            soc_pct=round(float(row["soc_pct"]), 2),
            pack_voltage_v=round(float(row["pack_voltage_v"]), 1),
            pack_current_a=round(float(row["pack_current_a"]), 1),
            roll_rad=round(roll_rad, 4),
            pitch_rad=round(pitch_rad, 4),
            action=str(row["action"]),
            wheels=wheels,
        ))

    return frames


def _safe_float(val, default: float = 0.0) -> float:
    """Extract a float from a pandas value, returning default if NaN or missing."""
    if val is None:
        return default
    f = float(val)
    if math.isnan(f):
        return default
    return f


def _build_real_frames(
    real_df, cs_x, cs_y, cl_d, mass_kg: float,
    gear_ratio: float = 3.6363,
    tire_radius: float = 0.2042,
    gearbox_efficiency: float = 0.97,
) -> list[Frame]:
    """Convert real telemetry into 3D frames."""
    frames = []
    max_d = cl_d[-1]
    t0 = real_df["Time"].iloc[0]

    # Pre-compute brake normalization outside the loop (#2, #14)
    has_rear_brake = "RBrakePressure" in real_df.columns
    if has_rear_brake:
        brake_col = real_df["RBrakePressure"]
        brake_baseline = float(brake_col.median())
        brake_max = float((brake_col - brake_baseline).clip(lower=0).max())
        if brake_max <= 0:
            brake_max = 1.0
    else:
        brake_baseline = 0.0
        brake_max = 1.0

    for _, row in real_df.iterrows():
        d = float(row["lap_distance_m"])
        if d < 0 or d > max_d:  # (#17) use > instead of >= to keep end-of-lap frames
            continue
        d_clamped = min(d, max_d - 0.01)
        x = float(cs_x(d_clamped))
        y = float(cs_y(d_clamped))
        dx = float(cs_x(d_clamped, 1))
        dy = float(cs_y(d_clamped, 1))
        heading = math.atan2(dy, dx)

        speed_kmh = float(row["GPS Speed"])
        speed_ms = speed_kmh / 3.6
        throttle = float(row["Throttle Pos"])

        # (#2) Use RBrakePressure (working sensor), zero-offset by baseline
        if has_rear_brake:
            brake_raw = max(0.0, float(row["RBrakePressure"]) - brake_baseline)
            brake_pct = (brake_raw / brake_max * 100)
        else:
            brake_raw = 0.0
            brake_pct = 0.0

        if throttle > 5:
            action = "throttle"
        elif brake_raw > 1.0:
            action = "brake"
        else:
            action = "coast"

        # (#11) NaN-safe reads for accelerometer channels
        lat_g = _safe_float(row.get("GPS LatAcc"))
        curvature = lat_g * _GRAVITY / (speed_ms ** 2) if speed_ms > 1 else 0

        # (#5) Use real longitudinal acceleration instead of hardcoded 0
        long_g = _safe_float(row.get("GPS LonAcc"))

        # (#15) Use passed parameters instead of hardcoded values
        torque_req = _safe_float(row.get("LVCU Torque Req"))
        # (#7) Both drive and regen use gearbox_efficiency for mechanical force
        drive_force = torque_req * gear_ratio / tire_radius * gearbox_efficiency if torque_req > 0 else 0
        regen_force = abs(torque_req) * gear_ratio / tire_radius * gearbox_efficiency if torque_req < 0 else 0

        fl_fx, fr_fx, rl_fx, rr_fx = distribute_drive_force(drive_force, regen_force)
        fl_fy, fr_fy, rl_fy, rr_fy = _compute_lateral_forces(speed_ms, curvature, mass_kg)
        fl_fz, fr_fz, rl_fz, rr_fz = _compute_tire_loads(speed_ms, curvature, long_g, mass_kg)

        wheels = [
            WheelForce(fx=round(fl_fx, 1), fy=round(fl_fy, 1), fz=round(fl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fl_fx, fl_fy, fl_fz), 3)),
            WheelForce(fx=round(fr_fx, 1), fy=round(fr_fy, 1), fz=round(fr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(fr_fx, fr_fy, fr_fz), 3)),
            WheelForce(fx=round(rl_fx, 1), fy=round(rl_fy, 1), fz=round(rl_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rl_fx, rl_fy, rl_fz), 3)),
            WheelForce(fx=round(rr_fx, 1), fy=round(rr_fy, 1), fz=round(rr_fz, 1),
                       grip_util=round(_estimate_grip_utilization(rr_fx, rr_fy, rr_fz), 3)),
        ]

        roll_rad, pitch_rad = _compute_roll_pitch(lat_g, long_g, mass_kg=mass_kg)

        frames.append(Frame(
            time_s=round(float(row["Time"]) - t0, 3),
            distance_m=round(d, 2),
            x=round(x, 3), y=round(y, 3),
            heading_rad=round(heading, 4),
            speed_kmh=round(speed_kmh, 1),
            throttle_pct=round(throttle, 1),
            brake_pct=round(brake_pct, 1),
            motor_rpm=round(_safe_float(row.get("Motor RPM")), 0),  # (#1) was "RPM"
            motor_torque_nm=round(_safe_float(row.get("LVCU Torque Req")), 1),
            soc_pct=round(_safe_float(row.get("State of Charge")), 2),
            pack_voltage_v=round(_safe_float(row.get("Pack Voltage")), 1),
            pack_current_a=round(_safe_float(row.get("Pack Current")), 1),
            roll_rad=round(roll_rad, 4),
            pitch_rad=round(pitch_rad, 4),
            action=action,
            wheels=wheels,
        ))

    return frames
