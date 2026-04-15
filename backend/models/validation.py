from pydantic import BaseModel


class TraceData(BaseModel):
    """Paired sim and real values at uniform distance points."""
    distance_m: list[float]
    sim: list[float]
    real: list[float]


class ValidationMetricResult(BaseModel):
    name: str
    unit: str
    sim_value: float
    real_value: float
    error_pct: float
    threshold_pct: float
    passed: bool


class SectorComparison(BaseModel):
    name: str
    sector_type: str
    sim_time_s: float
    real_time_s: float
    delta_s: float
    delta_pct: float
    sim_avg_speed_kmh: float
    real_avg_speed_kmh: float
    speed_delta_pct: float


class LapSummary(BaseModel):
    lap_number: int
    sim_time_s: float
    real_time_s: float
    time_error_pct: float
    sim_energy_kwh: float
    real_energy_kwh: float
    energy_error_pct: float
    mean_speed_error_pct: float


class ValidationResponse(BaseModel):
    lap_number: int
    speed: TraceData
    throttle: TraceData
    brake: TraceData
    power: TraceData
    soc: TraceData
    lat_accel: TraceData
    track_sim_speed: list[float]  # per-centerline-point speed for track map coloring
    track_real_speed: list[float]
    sectors: list[SectorComparison]
    metrics: list[ValidationMetricResult]


class AllLapsResponse(BaseModel):
    laps: list[LapSummary]
    metrics: list[ValidationMetricResult]
