import { useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import type { VisualizationResponse } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import { animState } from './animationState'
import WireframeCar from './WireframeCar'
import ForceArrows from './ForceArrows'
import TrackLine from './TrackLine'
import CameraController from './CameraController'

/** Push frame index to Zustand every N frame-advances (for timeline + side panel) */
const UI_SYNC_INTERVAL = 2
/**
 * Maximum simulated seconds consumed per render tick. At 5x speed a 60fps tick
 * asks for ~83ms of sim time (~1.6 frames at 20Hz), which is fine. But if the
 * browser tab is backgrounded and then resumed, delta can spike to seconds,
 * causing unbounded catch-up and frame stutter. Clamp to avoid this.
 */
const MAX_ACCUMULATOR_SECONDS = 0.25

function PlaybackLoop({ data }: { data: VisualizationResponse }) {
  const accumulator = useRef(0)
  const framesSinceSync = useRef(0)

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
      framesSinceSync.current = 0
      return
    }

    const frames = animState.frames
    let idx = animState.index
    if (idx >= store.totalFrames - 1) {
      store.pause()
      store.setFrame(idx)
      return
    }

    // Advance sim time, but cap how much we can accumulate in one tick so a
    // paused/backgrounded tab doesn't trigger unbounded catch-up on resume.
    accumulator.current = Math.min(
      accumulator.current + delta * store.speed,
      MAX_ACCUMULATOR_SECONDS,
    )

    // Advance through frames as the accumulator allows. At high playback speed
    // this loop runs multiple times per tick; we count each frame-advance
    // toward the UI sync threshold so the side panel stays fresh.
    let dt = frames[idx + 1].time_s - frames[idx].time_s
    while (dt > 0 && accumulator.current >= dt && idx < store.totalFrames - 2) {
      accumulator.current -= dt
      idx++
      framesSinceSync.current++
      dt = frames[idx + 1].time_s - frames[idx].time_s
    }

    animState.index = idx
    // Interpolation fraction: how far between frames[idx] and frames[idx+1]
    animState.frac = dt > 0 ? Math.min(accumulator.current / dt, 1) : 0

    // Throttled sync to Zustand for UI panels. Push the LAST advanced frame
    // (idx), not any intermediate index, so side-panel readouts match where
    // the car actually is on screen.
    if (framesSinceSync.current >= UI_SYNC_INTERVAL) {
      store.setFrame(idx)
      framesSinceSync.current = 0
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
