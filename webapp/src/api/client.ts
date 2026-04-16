import useSWR, { mutate } from 'swr'
import { toast } from 'sonner'

export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

/**
 * Error thrown by {@link fetcher} for non-2xx responses. Carries the HTTP
 * status and, when the backend returned a JSON body with a `detail` field,
 * the parsed detail string for display.
 */
export class ApiError extends Error {
  readonly status: number
  readonly detail?: string

  constructor(status: number, detail?: string) {
    super(detail ? `API ${status}: ${detail}` : `API error: ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/**
 * Fetch JSON from `url`. Accepts an optional `init` so callers (or SWR) can
 * pass an `AbortSignal` for cancellation. Throws {@link ApiError} on non-OK
 * responses, populating `detail` from the JSON body's `detail` field when
 * present.
 *
 * On non-OK responses we also surface a `toast.error` for visibility; the
 * `ApiError` is still thrown so SWR's error state and any page-level error
 * handling continue to work. Toasts are keyed by URL to dedupe retries.
 *
 * SWR 2.x calls the fetcher with the key as the first argument; the optional
 * second argument is tolerated by SWR's type contract since it's optional.
 */
export async function fetcher<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as
      | { detail?: string }
      | null
    const detail = body?.detail ?? res.statusText
    const err = new ApiError(res.status, detail)
    toast.error(`API error (${res.status})`, {
      id: url,
      description: detail || undefined,
    })
    throw err
  }
  return res.json() as Promise<T>
}

let rerunInFlight = false

/**
 * Clear backend caches and revalidate only SWR keys owned by this app
 * (those starting with `API_BASE`), so unrelated caches stay put. A simple
 * in-flight guard prevents double-clicks from stacking multi-second runs.
 *
 * Shows a loading toast while the rerun is in progress, promoted to a
 * success toast on completion or an error toast (with detail) on failure.
 * Errors are re-thrown so callers still see them.
 */
export async function rerunSimulation(): Promise<void> {
  if (rerunInFlight) return
  rerunInFlight = true
  const toastId = toast.loading('Rerunning simulation...')
  try {
    const res = await fetch(`${API_BASE}/cache/clear`, { method: 'POST' })
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as
        | { detail?: string }
        | null
      throw new ApiError(res.status, body?.detail ?? res.statusText)
    }
    await mutate(
      (key) => typeof key === 'string' && key.startsWith(API_BASE),
      undefined,
      { revalidate: true },
    )
    toast.success('Simulation refreshed', { id: toastId })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    toast.error('Failed to rerun simulation', {
      id: toastId,
      description: message,
    })
    throw err
  } finally {
    rerunInFlight = false
  }
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
