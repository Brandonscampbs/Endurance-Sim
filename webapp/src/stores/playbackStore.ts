import { create } from 'zustand'
import { syncAnimIndex } from '../pages/visualization/animationState'

interface PlaybackState {
  isPlaying: boolean
  speed: number  // 0.5, 1, 2, 5
  currentFrame: number
  totalFrames: number
  showForces: boolean
  showTrackColor: boolean

  play: () => void
  pause: () => void
  togglePlay: () => void
  setSpeed: (s: number) => void
  setFrame: (f: number) => void
  nextFrame: () => void
  prevFrame: () => void
  setTotalFrames: (n: number) => void
  toggleForces: () => void
  toggleTrackColor: () => void
  setIsPlaying: (p: boolean) => void
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  isPlaying: false,
  speed: 1,
  currentFrame: 0,
  totalFrames: 0,
  showForces: true,
  showTrackColor: true,

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
  toggleForces: () => set(s => ({ showForces: !s.showForces })),
  toggleTrackColor: () => set(s => ({ showTrackColor: !s.showTrackColor })),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
}))
