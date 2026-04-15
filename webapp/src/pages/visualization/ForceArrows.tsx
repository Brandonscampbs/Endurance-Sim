import { useMemo } from 'react'
import * as THREE from 'three'
import type { VizFrame, WheelForce } from '../../api/client'

const WHEELBASE = 1.549
const TRACK_WIDTH = 1.2
const FORCE_SCALE = 1 / 2000  // N to meters
const MAX_ARROW_LENGTH = 0.4  // meters -- max visual arrow length

const wheelOffsets: [number, number, number][] = [
  [WHEELBASE * 0.53, 0, TRACK_WIDTH / 2],
  [WHEELBASE * 0.53, 0, -TRACK_WIDTH / 2],
  [-WHEELBASE * 0.47, 0, TRACK_WIDTH / 2],
  [-WHEELBASE * 0.47, 0, -TRACK_WIDTH / 2],
]

function gripColor(util: number): string {
  if (util < 0.6) return '#22c55e'
  if (util < 0.8) return '#eab308'
  return '#ef4444'
}

function ForceArrow({ wheel, offset }: { wheel: WheelForce; offset: [number, number, number] }) {
  const arrow = useMemo(() => {
    const fx = wheel.fx * FORCE_SCALE
    const fy = wheel.fy * FORCE_SCALE
    const raw = Math.sqrt(fx * fx + fy * fy)
    const len = Math.min(raw, MAX_ARROW_LENGTH)

    if (len < 0.01) return null

    // Force direction in car-local frame: fx = forward, fy = lateral
    const dir = new THREE.Vector3(fx, 0, fy)
    if (dir.length() > 0.001) dir.normalize()

    const color = new THREE.Color(gripColor(wheel.grip_util))
    const origin = new THREE.Vector3(0, 0.02, 0) // slight lift off ground

    return new THREE.ArrowHelper(
      dir,
      origin,
      len,
      color,
      len * 0.3,  // head length
      len * 0.15, // head width
    )
  }, [wheel.fx, wheel.fy, wheel.grip_util])

  if (!arrow) return null

  return (
    <group position={offset}>
      <primitive object={arrow} />
    </group>
  )
}

interface Props {
  frame: VizFrame
}

export default function ForceArrows({ frame }: Props) {
  return (
    <group
      position={[frame.x, 0, frame.y]}
      rotation={[0, -frame.heading_rad + Math.PI / 2, 0]}
    >
      {frame.wheels.map((wheel, i) => (
        <ForceArrow
          key={i}
          wheel={wheel}
          offset={wheelOffsets[i]}
        />
      ))}
    </group>
  )
}
