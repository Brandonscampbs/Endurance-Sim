import { useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { usePlaybackStore } from '../../stores/playbackStore'
import { parseCameraMode, type CameraMode } from './CameraController'

const speeds = [0.5, 1, 2, 5]
const cameras: { mode: CameraMode; label: string }[] = [
  { mode: 'chase', label: 'Chase' },
  { mode: 'birdseye', label: "Bird's Eye" },
  { mode: 'orbit', label: 'Orbit' },
]

const baseBtn =
  'px-2 py-0.5 text-xs rounded transition-colors bg-[var(--surface-3)] ' +
  'text-[var(--text-secondary)] hover:bg-[var(--accent)]/20 hover:text-[var(--text-primary)]'
const activeBtn = 'bg-[var(--accent)]/30 text-[var(--accent)]'

export default function PlaybackControls() {
  const store = usePlaybackStore()
  const [searchParams, setSearchParams] = useSearchParams()

  const cameraMode = parseCameraMode(searchParams.get('camera'))

  const setCameraMode = (m: CameraMode) => {
    const next = new URLSearchParams(searchParams)
    next.set('camera', m)
    setSearchParams(next, { replace: true })
  }

  const handleKeydown = useCallback((e: KeyboardEvent) => {
    if (e.code === 'Space') { e.preventDefault(); usePlaybackStore.getState().togglePlay() }
    if (e.code === 'ArrowRight') { e.preventDefault(); usePlaybackStore.getState().nextFrame() }
    if (e.code === 'ArrowLeft') { e.preventDefault(); usePlaybackStore.getState().prevFrame() }
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    window.addEventListener('keydown', handleKeydown)
    return () => window.removeEventListener('keydown', handleKeydown)
  }, [handleKeydown])

  return (
    <div className="flex items-center gap-4 px-4 flex-wrap">
      {/* Play/Pause */}
      <button
        onClick={store.togglePlay}
        className={`px-2 py-0.5 text-base rounded transition-colors ${
          store.isPlaying
            ? 'bg-[var(--accent)]/30 text-[var(--accent)]'
            : 'bg-[var(--surface-3)] text-[var(--text-primary)] hover:bg-[var(--accent)]/20'
        }`}
        aria-label={store.isPlaying ? 'Pause' : 'Play'}
      >
        {store.isPlaying ? '⏸' : '▶'}
      </button>

      {/* Speed */}
      <div className="flex gap-1">
        {speeds.map((s) => (
          <button
            key={s}
            onClick={() => store.setSpeed(s)}
            className={`${baseBtn} ${store.speed === s ? activeBtn : ''}`}
          >
            {s}x
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-[var(--border-subtle)]" />

      {/* Camera */}
      <div className="flex gap-1">
        {cameras.map(({ mode, label }) => (
          <button
            key={mode}
            onClick={() => setCameraMode(mode)}
            className={`${baseBtn} ${cameraMode === mode ? activeBtn : ''}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-[var(--border-subtle)]" />

      {/* Overlay toggles */}
      <button
        onClick={store.toggleForces}
        className={`${baseBtn} ${store.showForces ? activeBtn : ''}`}
      >
        Forces
      </button>
      <button
        onClick={store.toggleTrackColor}
        className={`${baseBtn} ${store.showTrackColor ? activeBtn : ''}`}
      >
        Track Color
      </button>
    </div>
  )
}
