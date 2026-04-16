import { useRef, useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import type { VisualizationResponse } from '../../api/client'

function speedToColor(speed: number, minSpeed: number, maxSpeed: number): THREE.Color {
  const t = maxSpeed > minSpeed ? (speed - minSpeed) / (maxSpeed - minSpeed) : 0
  if (t < 0.25) return new THREE.Color('#3b82f6').lerp(new THREE.Color('#22c55e'), t / 0.25)
  if (t < 0.5) return new THREE.Color('#22c55e').lerp(new THREE.Color('#eab308'), (t - 0.25) / 0.25)
  if (t < 0.75) return new THREE.Color('#eab308').lerp(new THREE.Color('#f97316'), (t - 0.5) / 0.25)
  return new THREE.Color('#f97316').lerp(new THREE.Color('#ef4444'), (t - 0.75) / 0.25)
}

/**
 * Spread-free min/max: large arrays (~40k centerline points) blow the
 * JS engine's argument-count limit when passed via `Math.min(...arr)`.
 * Also skips NaN/Infinity.
 */
function reduceMin(arr: number[], predicate?: (v: number) => boolean): number {
  let m = Infinity
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    if (!Number.isFinite(v)) continue
    if (predicate && !predicate(v)) continue
    if (v < m) m = v
  }
  return m
}
function reduceMax(arr: number[], predicate?: (v: number) => boolean): number {
  let m = -Infinity
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    if (!Number.isFinite(v)) continue
    if (predicate && !predicate(v)) continue
    if (v > m) m = v
  }
  return m
}

interface Props {
  data: VisualizationResponse
  showColors: boolean
}

export default function TrackLine({ data, showColors }: Props) {
  const { scene } = useThree()
  const lineRef = useRef<THREE.Line | null>(null)

  // Build the line as a side effect so we can dispose the previous instance
  // (geometry + material + line object) cleanly when deps change.
  useEffect(() => {
    const geometry = new THREE.BufferGeometry()
    const positions: number[] = []
    const colors: number[] = []

    const xs = data.track_centerline_x
    const ys = data.track_centerline_y
    const speeds = data.track_speed_colors
    const minS = reduceMin(speeds, v => v > 0)
    const maxS = reduceMax(speeds)
    const safeMinS = Number.isFinite(minS) ? minS : 0
    const safeMaxS = Number.isFinite(maxS) ? maxS : 1

    for (let i = 0; i < xs.length; i++) {
      positions.push(xs[i], 0.01, ys[i])  // slight elevation above ground
      const c = showColors
        ? speedToColor(speeds[i], safeMinS, safeMaxS)
        : new THREE.Color('#4b5563')
      colors.push(c.r, c.g, c.b)
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))

    const material = new THREE.LineBasicMaterial({ vertexColors: true })
    const line = new THREE.Line(geometry, material)
    scene.add(line)
    lineRef.current = line

    return () => {
      scene.remove(line)
      geometry.dispose()
      material.dispose()
      lineRef.current = null
    }
  }, [data, showColors, scene])

  return null
}
