import { create } from 'zustand'

interface ValidationState {
  selectedLap: number | null  // null = "all laps"
  setSelectedLap: (lap: number | null) => void
}

export const useValidationStore = create<ValidationState>((set) => ({
  selectedLap: null,
  setSelectedLap: (lap) => set({ selectedLap: lap }),
}))
