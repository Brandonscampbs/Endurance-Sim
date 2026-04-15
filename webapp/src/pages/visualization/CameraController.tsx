import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { usePlaybackStore } from '../../stores/playbackStore'
import { animState, lerpAngle } from './animationState'

export default function CameraController() {
  const cameraMode = usePlaybackStore(s => s.cameraMode)
  const { camera } = useThree()
  const target = useRef(new THREE.Vector3())
  const smoothPos = useRef(new THREE.Vector3())

  useFrame((_, delta) => {
    const curr = animState.current
    if (!curr) return

    // Interpolated position for smooth camera tracking
    const next = animState.frames[animState.index + 1]
    const t = animState.frac
    const ix = next ? curr.x + (next.x - curr.x) * t : curr.x
    const iy = next ? curr.y + (next.y - curr.y) * t : curr.y
    const ih = next ? lerpAngle(curr.heading_rad, next.heading_rad, t) : curr.heading_rad

    const carPos = new THREE.Vector3(ix, 0.3, iy)
    // Frame-rate-independent smoothing: target rate = 60fps
    const targetSmooth = 1 - Math.pow(1 - 0.1, delta * 60)
    const camSmooth = 1 - Math.pow(1 - 0.05, delta * 60)
    target.current.lerp(carPos, targetSmooth)

    const mode = usePlaybackStore.getState().cameraMode
    if (mode === 'chase') {
      const behind = new THREE.Vector3(
        ix - Math.cos(ih) * 4,
        2.5,
        iy - Math.sin(ih) * 4,
      )
      smoothPos.current.lerp(behind, camSmooth)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    } else if (mode === 'birdseye') {
      const above = new THREE.Vector3(ix, 15, iy)
      smoothPos.current.lerp(above, targetSmooth)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    }
  })

  if (cameraMode === 'orbit') {
    return <OrbitControls target={target.current} enableDamping dampingFactor={0.1} />
  }

  return null
}
