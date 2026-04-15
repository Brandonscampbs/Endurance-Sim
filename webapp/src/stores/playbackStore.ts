import { create } from 'zustand'
import { syncAnimIndex } from '../pages/visualization/animationState'

export type CameraMode = 'chase' | 'birdseye' | 'orbit'

interface PlaybackState {
  isPlaying: boolean
  speed: number  // 0.5, 1, 2, 5
  currentFrame: number
  totalFrames: number
  cameraMode: CameraMode
  showForces: boolean
  showTrackColor: boolean
  dataSource: 'sim' | 'real'

  play: () => void
  pause: () => void
  togglePlay: () => void
  setSpeed: (s: number) => void
  setFrame: (f: number) => void
  nextFrame: () => void
  prevFrame: () => void
  setTotalFrames: (n: number) => void
  setCameraMode: (m: CameraMode) => void
  toggleForces: () => void
  toggleTrackColor: () => void
  setDataSource: (s: 'sim' | 'real') => void
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  isPlaying: false,
  speed: 1,
  currentFrame: 0,
  totalFrames: 0,
  cameraMode: 'chase',
  showForces: true,
  showTrackColor: true,
  dataSource: 'sim',

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  togglePlay: () => set(s => ({ isPlaying: !s.isPlaying })),
  setSpeed: (speed) => set({ speed }),
  setFrame: (f) => {
    const clamped = Math.max(0, Math.min(f, get().totalFrames - 1))
    set({ currentFrame: clamped })
    syncAnimIndex(clamped)
  },
  nextFrame: () => {
    const s = get()
    if (s.currentFrame < s.totalFrames - 1) {
      const next = s.currentFrame + 1
      set({ currentFrame: next })
      syncAnimIndex(next)
    }
  },
  prevFrame: () => {
    const s = get()
    if (s.currentFrame > 0) {
      const prev = s.currentFrame - 1
      set({ currentFrame: prev })
      syncAnimIndex(prev)
    }
  },
  setTotalFrames: (n) => set({ totalFrames: n }),
  setCameraMode: (cameraMode) => set({ cameraMode }),
  toggleForces: () => set(s => ({ showForces: !s.showForces })),
  toggleTrackColor: () => set(s => ({ showTrackColor: !s.showTrackColor })),
  setDataSource: (dataSource) => set({ dataSource, currentFrame: 0, isPlaying: false }),
}))
