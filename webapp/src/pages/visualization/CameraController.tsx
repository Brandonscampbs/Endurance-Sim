import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useSearchParams } from 'react-router-dom'
import * as THREE from 'three'
import { animState, lerpAngle } from './animationState'

export type CameraMode = 'chase' | 'birdseye' | 'orbit'

const VALID_CAMERAS: readonly CameraMode[] = ['chase', 'birdseye', 'orbit'] as const

/**
 * Defensively parse the ?camera URL param. Unknown / missing values
 * fall back to "chase".
 */
export function parseCameraMode(raw: string | null): CameraMode {
  return VALID_CAMERAS.includes(raw as CameraMode) ? (raw as CameraMode) : 'chase'
}

export default function CameraController() {
  const [searchParams] = useSearchParams()
  const cameraMode = parseCameraMode(searchParams.get('camera'))
  const { camera } = useThree()
  // Shared smoothed target (world-space car position) used by all camera modes.
  const target = useRef(new THREE.Vector3())
  const smoothPos = useRef(new THREE.Vector3())
  // drei's <OrbitControls> copies the `target` prop into its own Vector3 at
  // mount and never re-reads it. To keep the orbit center following the car
  // we grab the controls imperatively and mutate controls.target each frame.
  const controlsRef = useRef<OrbitControlsImpl | null>(null)

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

    if (cameraMode === 'chase') {
      const behind = new THREE.Vector3(
        ix - Math.cos(ih) * 4,
        2.5,
        iy - Math.sin(ih) * 4,
      )
      smoothPos.current.lerp(behind, camSmooth)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    } else if (cameraMode === 'birdseye') {
      const above = new THREE.Vector3(ix, 15, iy)
      smoothPos.current.lerp(above, targetSmooth)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    } else if (cameraMode === 'orbit' && controlsRef.current) {
      // Update OrbitControls' internal target (not the prop) so the orbit
      // center tracks the car. update() applies damping + recomputes the
      // camera transform.
      controlsRef.current.target.copy(target.current)
      controlsRef.current.update()
    }
  })

  if (cameraMode === 'orbit') {
    return <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.1} />
  }

  return null
}
