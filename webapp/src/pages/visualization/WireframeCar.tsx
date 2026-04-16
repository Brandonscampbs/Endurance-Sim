import { useRef, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { Group } from 'three'
import { animState, lerpAngle } from './animationState'

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

export default function WireframeCar() {
  const groupRef = useRef<Group>(null)

  // Three.js default Euler order is 'XYZ' (intrinsic), which applies roll
  // about world-X first — that tilts the car the wrong way when heading != 0.
  // 'YXZ' applies yaw first (Y), then pitch about the body-lateral axis (X),
  // then roll about the body-longitudinal axis (Z) — correct automotive order.
  useEffect(() => {
    if (groupRef.current) groupRef.current.rotation.order = 'YXZ'
  }, [])

  useFrame(() => {
    const curr = animState.current
    if (!curr || !groupRef.current) return

    const next = animState.frames[animState.index + 1]
    if (next) {
      const t = animState.frac
      groupRef.current.position.set(
        curr.x + (next.x - curr.x) * t,
        0,
        curr.y + (next.y - curr.y) * t,
      )
      // YXZ order => rotation.set(X=pitch, Y=yaw, Z=roll).
      // Backend heading_rad increases CCW (math convention). World-Y in our
      // scene is up; our XZ ground plane uses +X right, +Z forward only after
      // yaw = -heading puts body +X to world +X at heading=0.
      groupRef.current.rotation.set(
        curr.pitch_rad + (next.pitch_rad - curr.pitch_rad) * t,
        -(lerpAngle(curr.heading_rad, next.heading_rad, t)),
        curr.roll_rad + (next.roll_rad - curr.roll_rad) * t,
      )
    } else {
      groupRef.current.position.set(curr.x, 0, curr.y)
      groupRef.current.rotation.set(
        curr.pitch_rad,
        -curr.heading_rad,
        curr.roll_rad,
      )
    }
  })

  return (
    <group ref={groupRef}>
      <mesh position={[0, CHASSIS_HEIGHT / 2 + WHEEL_RADIUS, 0]}>
        <boxGeometry args={[WHEELBASE, CHASSIS_HEIGHT, TRACK_WIDTH]} />
        <meshBasicMaterial color="#d1d5db" wireframe />
      </mesh>
      {wheelPositions.map((pos, i) => (
        <Wheel key={i} position={pos} />
      ))}
    </group>
  )
}
