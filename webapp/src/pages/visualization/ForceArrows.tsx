import { useRef, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { usePlaybackStore } from '../../stores/playbackStore'
import { animState, lerpAngle } from './animationState'

const WHEELBASE = 1.549
const TRACK_WIDTH = 1.2

// Fallback scale if no per-lap max has been computed yet. Peak cornering forces
// are ~1500-2500 N; 1/3000 gives a ~0.67 m arrow at 2 kN before clipping.
const DEFAULT_FORCE_SCALE = 1 / 3000
const MAX_ARROW_LENGTH = 1.0  // meters in world space

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

/**
 * Compute an adaptive force -> meters scale from the entire lap's peak wheel
 * force magnitude, so the max force in the lap renders at MAX_ARROW_LENGTH and
 * everything else is proportional. Recomputed whenever frames identity changes.
 */
function computeAdaptiveScale(frames: ReturnType<() => typeof animState.frames>): number {
  let peak = 0
  for (let i = 0; i < frames.length; i++) {
    const wheels = frames[i]?.wheels
    if (!wheels) continue
    for (let w = 0; w < wheels.length; w++) {
      const { fx, fy } = wheels[w]
      if (!Number.isFinite(fx) || !Number.isFinite(fy)) continue
      const mag = Math.sqrt(fx * fx + fy * fy)
      if (mag > peak) peak = mag
    }
  }
  if (peak <= 0) return DEFAULT_FORCE_SCALE
  return MAX_ARROW_LENGTH / peak
}

export default function ForceArrows() {
  const groupRef = useRef<THREE.Group>(null)
  const arrowRefs = useRef<THREE.ArrowHelper[]>([])
  const scaleRef = useRef<number>(DEFAULT_FORCE_SCALE)
  const framesIdentityRef = useRef<unknown>(null)

  // Create 4 arrow helpers once. They're children of a group oriented to the
  // car (see useFrame below), so the directions we pass to setDirection are
  // interpreted in the car's body frame:
  //   local +X = car forward, local +Z = car right (matches world axes after
  //   the parent group's yaw rotation).
  useEffect(() => {
    const group = groupRef.current
    if (!group) return
    const defaultDir = new THREE.Vector3(1, 0, 0)
    const origin = new THREE.Vector3(0, 0, 0)
    const arrows: THREE.ArrowHelper[] = []
    for (let i = 0; i < 4; i++) {
      const arrow = new THREE.ArrowHelper(defaultDir, origin, 0.01, 0x22c55e, 0.05, 0.03)
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

  // Set YXZ rotation order on the parent group so heading (Y) is applied
  // before roll (Z) / pitch (X) — matches the car's WireframeCar.
  useEffect(() => {
    if (groupRef.current) groupRef.current.rotation.order = 'YXZ'
  }, [])

  const dir = new THREE.Vector3() // reusable scratch

  useFrame(() => {
    const frame = animState.current
    const group = groupRef.current
    if (!frame || !group) return

    group.visible = usePlaybackStore.getState().showForces

    // Recompute adaptive scale when the frames array changes (e.g. new lap).
    if (framesIdentityRef.current !== animState.frames) {
      framesIdentityRef.current = animState.frames
      scaleRef.current = computeAdaptiveScale(animState.frames)
    }
    const forceScale = scaleRef.current

    // Interpolate group position/rotation to match the car.
    const next = animState.frames[animState.index + 1]
    const t = animState.frac
    const ix = next ? frame.x + (next.x - frame.x) * t : frame.x
    const iy = next ? frame.y + (next.y - frame.y) * t : frame.y
    const ih = next ? lerpAngle(frame.heading_rad, next.heading_rad, t) : frame.heading_rad
    const ir = next ? frame.roll_rad + (next.roll_rad - frame.roll_rad) * t : frame.roll_rad
    const ip = next ? frame.pitch_rad + (next.pitch_rad - frame.pitch_rad) * t : frame.pitch_rad
    group.position.set(ix, 0, iy)
    // YXZ order: (X=pitch, Y=yaw, Z=roll). See WireframeCar for rationale.
    group.rotation.set(ip, -ih, ir)

    for (let i = 0; i < 4; i++) {
      const arrow = arrowRefs.current[i]
      const wheel = frame.wheels[i]
      if (!arrow || !wheel) continue

      // Body-frame longitudinal (fx, +forward) and lateral (fy, +right) forces.
      // The parent group is already rotated into the world, so we just pass
      // the body-frame direction and ArrowHelper will orient the arrow in
      // parent-local coords; the parent's rotation then produces world-frame
      // arrows pointing in the actual force direction.
      const bodyFx = wheel.fx
      const bodyFy = wheel.fy
      const rawMag = Math.sqrt(bodyFx * bodyFx + bodyFy * bodyFy)
      const len = Math.min(rawMag * forceScale, MAX_ARROW_LENGTH)

      if (len < 0.01 || rawMag <= 0) {
        arrow.visible = false
        continue
      }

      arrow.visible = true
      // Body frame: +X forward, +Z right (so fy maps to local Z).
      dir.set(bodyFx, 0, bodyFy).normalize()
      arrow.setDirection(dir)
      arrow.setLength(len, len * 0.3, len * 0.15)
      arrow.setColor(gripColor(wheel.grip_util))
    }
  })

  return <group ref={groupRef} />
}
