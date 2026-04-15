import useSWR, { mutate } from 'swr'

const API_BASE = '/api'

async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

/** Clear all backend caches and refetch all SWR data. */
export async function rerunSimulation(): Promise<void> {
  await fetch(`${API_BASE}/cache/clear`, { method: 'POST' })
  // Revalidate every SWR key so the UI refreshes
  await mutate(() => true, undefined, { revalidate: true })
}

// Types matching backend Pydantic models
export interface TrackPoint { x: number; y: number; distance_m: number }
export interface Sector { name: string; sector_type: string; start_m: number; end_m: number }
export interface TrackData {
  centerline: TrackPoint[]; sectors: Sector[]; curvature: number[]; total_distance_m: number
}

export interface LapInfo { lap_number: number; gps_quality_score: number; time_s: number; valid_gps_pct: number }
export interface TraceData { distance_m: number[]; sim: number[]; real: number[] }
export interface ValidationMetric {
  name: string; unit: string; sim_value: number; real_value: number;
  error_pct: number; threshold_pct: number; passed: boolean
}
export interface SectorComparison {
  name: string; sector_type: string; sim_time_s: number; real_time_s: number;
  delta_s: number; delta_pct: number; sim_avg_speed_kmh: number;
  real_avg_speed_kmh: number; speed_delta_pct: number
}
export interface LapSummary {
  lap_number: number; sim_time_s: number; real_time_s: number; time_error_pct: number;
  sim_energy_kwh: number; real_energy_kwh: number; energy_error_pct: number;
  mean_speed_error_pct: number
}
export interface ValidationResponse {
  lap_number: number; speed: TraceData; throttle: TraceData; brake: TraceData;
  power: TraceData; soc: TraceData; lat_accel: TraceData;
  track_sim_speed: number[]; track_real_speed: number[];
  sectors: SectorComparison[]; metrics: ValidationMetric[]
}
export interface AllLapsResponse { laps: LapSummary[]; metrics: ValidationMetric[] }

export interface WheelForce { fx: number; fy: number; fz: number; grip_util: number }
export interface VizFrame {
  time_s: number; distance_m: number; x: number; y: number; heading_rad: number;
  speed_kmh: number; throttle_pct: number; brake_pct: number; motor_rpm: number;
  motor_torque_nm: number; soc_pct: number; pack_voltage_v: number; pack_current_a: number;
  roll_rad: number; pitch_rad: number; action: string; wheels: WheelForce[]
}
export interface VisualizationResponse {
  lap_number: number; total_time_s: number; total_frames: number; frames: VizFrame[];
  track_centerline_x: number[]; track_centerline_y: number[]; track_speed_colors: number[]
}

// SWR hooks
export function useLaps() {
  return useSWR<{ laps: LapInfo[] }>(`${API_BASE}/laps`, fetcher)
}

export function useTrack() {
  return useSWR<TrackData>(`${API_BASE}/track`, fetcher)
}

export function useValidation(lap: number | null) {
  return useSWR<ValidationResponse>(
    lap ? `${API_BASE}/validation/${lap}` : null, fetcher
  )
}

export function useAllLaps() {
  return useSWR<AllLapsResponse>(`${API_BASE}/validation`, fetcher)
}

export function useVisualization(source: 'sim' | 'real') {
  return useSWR<VisualizationResponse>(
    `${API_BASE}/visualization?source=${source}`, fetcher
  )
}
