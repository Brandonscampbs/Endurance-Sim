from pydantic import BaseModel


class WheelForce(BaseModel):
    fx: float  # longitudinal force (N)
    fy: float  # lateral force (N)
    fz: float  # normal load (N)
    grip_util: float  # 0-1 ratio of used vs available grip


class Frame(BaseModel):
    time_s: float
    distance_m: float
    x: float
    y: float
    heading_rad: float
    speed_kmh: float
    throttle_pct: float
    brake_pct: float
    motor_rpm: float
    motor_torque_nm: float
    soc_pct: float
    pack_voltage_v: float
    pack_current_a: float
    roll_rad: float
    pitch_rad: float
    action: str
    wheels: list[WheelForce]  # [FL, FR, RL, RR]


class VisualizationResponse(BaseModel):
    lap_number: int
    total_time_s: float
    total_frames: int
    frames: list[Frame]
    track_centerline_x: list[float]
    track_centerline_y: list[float]
    track_speed_colors: list[float]  # speed at each centerline point for coloring
