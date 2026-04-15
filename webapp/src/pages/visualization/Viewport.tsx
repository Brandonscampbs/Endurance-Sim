import { useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import type { VisualizationResponse } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import WireframeCar from './WireframeCar'
import ForceArrows from './ForceArrows'
import TrackLine from './TrackLine'
import CameraController from './CameraController'

function PlaybackLoop({ data }: { data: VisualizationResponse }) {
  const isPlaying = usePlaybackStore(s => s.isPlaying)
  const speed = usePlaybackStore(s => s.speed)
  const currentFrame = usePlaybackStore(s => s.currentFrame)
  const totalFrames = usePlaybackStore(s => s.totalFrames)
  const setFrame = usePlaybackStore(s => s.setFrame)
  const showForces = usePlaybackStore(s => s.showForces)
  const showTrackColor = usePlaybackStore(s => s.showTrackColor)
  const accumulator = useRef(0)

  const frame = data.frames[currentFrame] ?? data.frames[0]

  // Calculate time step between frames
  const dt = currentFrame < totalFrames - 1
    ? data.frames[currentFrame + 1].time_s - frame.time_s
    : 0.05  // fallback

  useFrame((_, delta) => {
    if (!isPlaying) return
    accumulator.current += delta * speed
    if (accumulator.current >= dt && dt > 0) {
      const steps = Math.floor(accumulator.current / dt)
      accumulator.current -= steps * dt
      const next = Math.min(currentFrame + steps, totalFrames - 1)
      setFrame(next)
      if (next >= totalFrames - 1) {
        usePlaybackStore.getState().pause()
      }
    }
  })

  return (
    <>
      <WireframeCar frame={frame} />
      {showForces && <ForceArrows frame={frame} />}
      <TrackLine data={data} showColors={showTrackColor} />
      <CameraController frame={frame} />
    </>
  )
}

interface Props {
  data: VisualizationResponse
}

export default function Viewport({ data }: Props) {
  useEffect(() => {
    usePlaybackStore.getState().setTotalFrames(data.frames.length)
  }, [data])

  return (
    <Canvas camera={{ position: [0, 10, 10], fov: 50 }} className="bg-gray-950">
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={0.4} />

      {/* Ground grid */}
      <gridHelper args={[200, 100, '#1f2937', '#111827']} />

      {/* Scene */}
      <PlaybackLoop data={data} />
    </Canvas>
  )
}
