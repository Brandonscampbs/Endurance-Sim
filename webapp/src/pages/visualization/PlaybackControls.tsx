import { useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { usePlaybackStore } from '../../stores/playbackStore'
import { parseCameraMode, type CameraMode } from './CameraController'
import { parseDataSource, type DataSource } from './VisualizationPage'

const speeds = [0.5, 1, 2, 5]
const cameras: { mode: CameraMode; label: string }[] = [
  { mode: 'chase', label: 'Chase' },
  { mode: 'birdseye', label: "Bird's Eye" },
  { mode: 'orbit', label: 'Orbit' },
]

export default function PlaybackControls() {
  const store = usePlaybackStore()
  const [searchParams, setSearchParams] = useSearchParams()

  const cameraMode = parseCameraMode(searchParams.get('camera'))
  const dataSource = parseDataSource(searchParams.get('source'))

  const setCameraMode = (m: CameraMode) => {
    const next = new URLSearchParams(searchParams)
    next.set('camera', m)
    setSearchParams(next, { replace: true })
  }

  const setDataSource = (s: DataSource) => {
    const next = new URLSearchParams(searchParams)
    next.set('source', s)
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
    <div className="flex items-center gap-4 px-4">
      {/* Play/Pause */}
      <button
        onClick={store.togglePlay}
        className="text-lg hover:text-green-400 transition-colors"
        aria-label={store.isPlaying ? 'Pause' : 'Play'}
      >
        {store.isPlaying ? '\u23F8' : '\u25B6'}
      </button>

      {/* Speed */}
      <div className="flex gap-1">
        {speeds.map((s) => (
          <button
            key={s}
            onClick={() => store.setSpeed(s)}
            className={`px-2 py-0.5 text-xs rounded ${
              store.speed === s ? 'bg-green-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Camera */}
      <div className="flex gap-1">
        {cameras.map(({ mode, label }) => (
          <button
            key={mode}
            onClick={() => setCameraMode(mode)}
            className={`px-2 py-0.5 text-xs rounded ${
              cameraMode === mode ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Data source toggle */}
      <div className="flex gap-1">
        {(['sim', 'real'] as const).map((src) => (
          <button
            key={src}
            onClick={() => setDataSource(src)}
            className={`px-2 py-0.5 text-xs rounded ${
              dataSource === src ? 'bg-purple-700 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {src === 'sim' ? 'Sim' : 'Real'}
          </button>
        ))}
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Overlay toggles */}
      <button
        onClick={store.toggleForces}
        className={`px-2 py-0.5 text-xs rounded ${
          store.showForces ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-500'
        }`}
      >
        Forces
      </button>
      <button
        onClick={store.toggleTrackColor}
        className={`px-2 py-0.5 text-xs rounded ${
          store.showTrackColor ? 'bg-gray-700 text-white' : 'bg-gray-800 text-gray-500'
        }`}
      >
        Track Color
      </button>
    </div>
  )
}
