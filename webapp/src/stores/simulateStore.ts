import { create } from 'zustand'

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

interface SimulateState {
  params: SimulateParams
  setMaxRpm: (v: number) => void
  setMaxTorque: (v: number) => void
  setSocPoint: (idx: number, point: SocDischargePoint) => void
  resetToBaseline: () => void
}

export const useSimulateStore = create<SimulateState>((set) => ({
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
}))
