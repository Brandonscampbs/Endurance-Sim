import { useRef } from 'react'
import { Group } from 'three'
import type { VizFrame } from '../../api/client'

// CT-16EV dimensions in meters
const WHEELBASE = 1.549
const TRACK_WIDTH = 1.2
const CHASSIS_HEIGHT = 0.3
const WHEEL_RADIUS = 0.127  // 254mm diameter / 2
const WHEEL_WIDTH = 0.2

const wheelPositions: [number, number, number][] = [
  [WHEELBASE * 0.53, WHEEL_RADIUS, TRACK_WIDTH / 2],   // FL
  [WHEELBASE * 0.53, WHEEL_RADIUS, -TRACK_WIDTH / 2],  // FR
  [-WHEELBASE * 0.47, WHEEL_RADIUS, TRACK_WIDTH / 2],  // RL
  [-WHEELBASE * 0.47, WHEEL_RADIUS, -TRACK_WIDTH / 2], // RR
]

function Wheel({ position }: { position: [number, number, number] }) {
  return (
    <mesh position={position} rotation={[Math.PI / 2, 0, 0]}>
      <cylinderGeometry args={[WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 12]} />
      <meshBasicMaterial color="#6b7280" wireframe />
    </mesh>
  )
}

interface Props {
  frame: VizFrame
}

export default function WireframeCar({ frame }: Props) {
  const groupRef = useRef<Group>(null)

  return (
    <group
      ref={groupRef}
      position={[frame.x, 0, frame.y]}
      rotation={[frame.pitch_rad, -frame.heading_rad + Math.PI / 2, frame.roll_rad]}
    >
      {/* Chassis */}
      <mesh position={[0, CHASSIS_HEIGHT / 2 + WHEEL_RADIUS, 0]}>
        <boxGeometry args={[WHEELBASE, CHASSIS_HEIGHT, TRACK_WIDTH]} />
        <meshBasicMaterial color="#d1d5db" wireframe />
      </mesh>

      {/* Wheels */}
      {wheelPositions.map((pos, i) => (
        <Wheel key={i} position={pos} />
      ))}
    </group>
  )
}
