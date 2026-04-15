import { useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import type { VisualizationResponse } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import { animState } from './animationState'
import WireframeCar from './WireframeCar'
import ForceArrows from './ForceArrows'
import TrackLine from './TrackLine'
import CameraController from './CameraController'

/** Push frame index to Zustand every N steps (for timeline + side panel) */
const UI_SYNC_INTERVAL = 2

function PlaybackLoop({ data }: { data: VisualizationResponse }) {
  const accumulator = useRef(0)
  const ticksSinceSync = useRef(0)

  // Load frames into shared animation state
  useEffect(() => {
    animState.frames = data.frames
    animState.index = 0
    animState.frac = 0
    usePlaybackStore.getState().setTotalFrames(data.frames.length)
    usePlaybackStore.getState().setFrame(0)
  }, [data])

  useFrame((_, delta) => {
    const store = usePlaybackStore.getState()
    if (store.totalFrames === 0) return

    if (!store.isPlaying) {
      accumulator.current = 0
      ticksSinceSync.current = 0
      return
    }

    const frames = animState.frames
    let idx = animState.index
    if (idx >= store.totalFrames - 1) {
      store.pause()
      store.setFrame(idx)
      return
    }

    accumulator.current += delta * store.speed

    // Advance through frames as the accumulator allows
    let dt = frames[idx + 1].time_s - frames[idx].time_s
    while (dt > 0 && accumulator.current >= dt && idx < store.totalFrames - 2) {
      accumulator.current -= dt
      idx++
      dt = frames[idx + 1].time_s - frames[idx].time_s
      ticksSinceSync.current++
    }

    animState.index = idx
    // Interpolation fraction: how far between frames[idx] and frames[idx+1]
    animState.frac = dt > 0 ? Math.min(accumulator.current / dt, 1) : 0

    // Throttled sync to Zustand for UI panels
    if (ticksSinceSync.current >= UI_SYNC_INTERVAL) {
      store.setFrame(idx)
      ticksSinceSync.current = 0
    }
  })

  const showTrackColor = usePlaybackStore(s => s.showTrackColor)

  return (
    <>
      <WireframeCar />
      <ForceArrows />
      <TrackLine data={data} showColors={showTrackColor} />
      <CameraController />
    </>
  )
}

interface Props {
  data: VisualizationResponse
}

export default function Viewport({ data }: Props) {
  return (
    <Canvas camera={{ position: [0, 10, 10], fov: 50 }} className="bg-gray-950">
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={0.4} />
      <gridHelper args={[200, 100, '#1f2937', '#111827']} />
      <PlaybackLoop data={data} />
    </Canvas>
  )
}
