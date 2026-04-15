import { useRef, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { usePlaybackStore } from '../../stores/playbackStore'
import { animState, lerpAngle } from './animationState'

const WHEELBASE = 1.549
const TRACK_WIDTH = 1.2
const FORCE_SCALE = 1 / 2000  // N to meters
const MAX_ARROW_LENGTH = 0.4

const wheelOffsets: [number, number, number][] = [
  [WHEELBASE * 0.53, 0.02, TRACK_WIDTH / 2],
  [WHEELBASE * 0.53, 0.02, -TRACK_WIDTH / 2],
  [-WHEELBASE * 0.47, 0.02, TRACK_WIDTH / 2],
  [-WHEELBASE * 0.47, 0.02, -TRACK_WIDTH / 2],
]

const GRIP_COLORS = {
  low: new THREE.Color('#22c55e'),
  mid: new THREE.Color('#eab308'),
  high: new THREE.Color('#ef4444'),
}

function gripColor(util: number): THREE.Color {
  if (util < 0.6) return GRIP_COLORS.low
  if (util < 0.8) return GRIP_COLORS.mid
  return GRIP_COLORS.high
}

export default function ForceArrows() {
  const groupRef = useRef<THREE.Group>(null)
  const arrowRefs = useRef<THREE.ArrowHelper[]>([])

  // Create 4 arrow helpers once
  useEffect(() => {
    const group = groupRef.current
    if (!group) return
    const defaultDir = new THREE.Vector3(1, 0, 0)
    const origin = new THREE.Vector3(0, 0, 0)
    const arrows: THREE.ArrowHelper[] = []
    for (let i = 0; i < 4; i++) {
      const arrow = new THREE.ArrowHelper(defaultDir, origin, 0.01, 0x22c55e, 0.003, 0.002)
      arrow.position.set(wheelOffsets[i][0], wheelOffsets[i][1], wheelOffsets[i][2])
      group.add(arrow)
      arrows.push(arrow)
    }
    arrowRefs.current = arrows
    return () => {
      arrows.forEach(a => {
        group.remove(a)
        a.dispose()
      })
      arrowRefs.current = []
    }
  }, [])

  const dir = new THREE.Vector3() // reusable

  useFrame(() => {
    const frame = animState.current
    const group = groupRef.current
    if (!frame || !group) return

    group.visible = usePlaybackStore.getState().showForces

    // Interpolate group position/rotation to match the car
    const next = animState.frames[animState.index + 1]
    const t = animState.frac
    const ix = next ? frame.x + (next.x - frame.x) * t : frame.x
    const iy = next ? frame.y + (next.y - frame.y) * t : frame.y
    const ih = next ? lerpAngle(frame.heading_rad, next.heading_rad, t) : frame.heading_rad
    const ir = next ? frame.roll_rad + (next.roll_rad - frame.roll_rad) * t : frame.roll_rad
    const ip = next ? frame.pitch_rad + (next.pitch_rad - frame.pitch_rad) * t : frame.pitch_rad
    group.position.set(ix, 0, iy)
    group.rotation.set(ir, -ih, ip)

    for (let i = 0; i < 4; i++) {
      const arrow = arrowRefs.current[i]
      const wheel = frame.wheels[i]
      if (!arrow || !wheel) continue

      const fx = wheel.fx * FORCE_SCALE
      const fy = wheel.fy * FORCE_SCALE
      const raw = Math.sqrt(fx * fx + fy * fy)
      const len = Math.min(raw, MAX_ARROW_LENGTH)

      if (len < 0.01) {
        arrow.visible = false
        continue
      }

      arrow.visible = true
      dir.set(fx, 0, fy).normalize()
      arrow.setDirection(dir)
      arrow.setLength(len, len * 0.3, len * 0.15)
      arrow.setColor(gripColor(wheel.grip_util))
    }
  })

  return <group ref={groupRef} />
}
