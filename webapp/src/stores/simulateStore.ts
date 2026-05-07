import { create } from 'zustand'
import {
  ApiError,
  runSimulation as apiRunSimulation,
  type SimulateResponse,
} from '../api/client'

export interface SocDischargePoint {
  soc_pct: number
  max_current_a: number
}

export interface SimulateParams {
  max_rpm: number
  max_torque_nm: number
  soc_discharge_map: SocDischargePoint[]
}

export const BASELINE_PARAMS: SimulateParams = {
  max_rpm: 2900,
  max_torque_nm: 85,
  soc_discharge_map: [
    { soc_pct: 100, max_current_a: 100 },
    { soc_pct: 85, max_current_a: 100 },
    { soc_pct: 50, max_current_a: 65 },
    { soc_pct: 20, max_current_a: 35 },
    { soc_pct: 5, max_current_a: 20 },
    { soc_pct: 0, max_current_a: 0 },
  ],
}

export type SimulateStatus = 'idle' | 'running' | 'success' | 'error'

interface SimulateState {
  params: SimulateParams
  setMaxRpm: (v: number) => void
  setMaxTorque: (v: number) => void
  setSocPoint: (idx: number, point: SocDischargePoint) => void
  resetToBaseline: () => void

  // Run lifecycle. `result` survives a `resetToBaseline` so users can compare
  // their previous run against a baseline they've just restored on the form.
  status: SimulateStatus
  result: SimulateResponse | null
  error: string | null
  runStartedAt: number | null

  runSimulation: () => Promise<void>
  cancelRun: () => void
  clearResult: () => void
}

// AbortController is held outside the store so it doesn't trigger renders
// when swapped between runs. Cancellation is best-effort: backend already
// completed work won't be undone, but the response is dropped.
let activeController: AbortController | null = null

export const useSimulateStore = create<SimulateState>((set, get) => ({
  params: BASELINE_PARAMS,
  setMaxRpm: (v) => set((s) => ({ params: { ...s.params, max_rpm: v } })),
  setMaxTorque: (v) => set((s) => ({ params: { ...s.params, max_torque_nm: v } })),
  setSocPoint: (idx, point) =>
    set((s) => {
      const next = [...s.params.soc_discharge_map]
      next[idx] = point
      return { params: { ...s.params, soc_discharge_map: next } }
    }),
  resetToBaseline: () => set({ params: BASELINE_PARAMS }),

  status: 'idle',
  result: null,
  error: null,
  runStartedAt: null,

  runSimulation: async () => {
    // Replace any in-flight controller so a stale request doesn't write back.
    activeController?.abort()
    const controller = new AbortController()
    activeController = controller

    const params = get().params
    set({
      status: 'running',
      error: null,
      runStartedAt: Date.now(),
    })

    try {
      const response = await apiRunSimulation(params, controller.signal)
      // Guard against a stale completion writing over a newer run.
      if (activeController !== controller) return
      set({
        status: 'success',
        result: response,
        error: null,
        runStartedAt: null,
      })
    } catch (err) {
      if (activeController !== controller) return
      // AbortError: user (or a newer call) cancelled — leave state as caller set it.
      if (err instanceof DOMException && err.name === 'AbortError') return
      const message =
        err instanceof ApiError
          ? err.detail ?? err.message
          : err instanceof Error
            ? err.message
            : 'Simulation failed.'
      set({ status: 'error', error: message, runStartedAt: null })
    } finally {
      if (activeController === controller) {
        activeController = null
      }
    }
  },

  cancelRun: () => {
    activeController?.abort()
    activeController = null
    set({ status: 'idle', runStartedAt: null })
  },

  clearResult: () =>
    set({ status: 'idle', result: null, error: null, runStartedAt: null }),
}))
