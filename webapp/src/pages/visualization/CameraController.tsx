import { useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { usePlaybackStore } from '../../stores/playbackStore'
import type { VizFrame } from '../../api/client'

interface Props {
  frame: VizFrame
}

export default function CameraController({ frame }: Props) {
  const cameraMode = usePlaybackStore(s => s.cameraMode)
  const { camera } = useThree()
  const target = useRef(new THREE.Vector3())
  const smoothPos = useRef(new THREE.Vector3())

  useFrame(() => {
    const carPos = new THREE.Vector3(frame.x, 0.3, frame.y)
    target.current.lerp(carPos, 0.1)

    if (cameraMode === 'chase') {
      // Behind and above the car
      const behind = new THREE.Vector3(
        frame.x - Math.cos(frame.heading_rad) * 4,
        2.5,
        frame.y - Math.sin(frame.heading_rad) * 4,
      )
      smoothPos.current.lerp(behind, 0.05)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    } else if (cameraMode === 'birdseye') {
      const above = new THREE.Vector3(frame.x, 15, frame.y)
      smoothPos.current.lerp(above, 0.1)
      camera.position.copy(smoothPos.current)
      camera.lookAt(target.current)
    }
    // 'orbit' mode is handled by OrbitControls -- we just update the target
  })

  if (cameraMode === 'orbit') {
    return <OrbitControls target={target.current} enableDamping dampingFactor={0.1} />
  }

  return null
}
