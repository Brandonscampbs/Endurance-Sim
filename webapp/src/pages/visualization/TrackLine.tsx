import { useMemo } from 'react'
import * as THREE from 'three'
import type { VisualizationResponse } from '../../api/client'

function speedToColor(speed: number, minSpeed: number, maxSpeed: number): THREE.Color {
  const t = maxSpeed > minSpeed ? (speed - minSpeed) / (maxSpeed - minSpeed) : 0
  if (t < 0.25) return new THREE.Color('#3b82f6').lerp(new THREE.Color('#22c55e'), t / 0.25)
  if (t < 0.5) return new THREE.Color('#22c55e').lerp(new THREE.Color('#eab308'), (t - 0.25) / 0.25)
  if (t < 0.75) return new THREE.Color('#eab308').lerp(new THREE.Color('#f97316'), (t - 0.5) / 0.25)
  return new THREE.Color('#f97316').lerp(new THREE.Color('#ef4444'), (t - 0.75) / 0.25)
}

interface Props {
  data: VisualizationResponse
  showColors: boolean
}

export default function TrackLine({ data, showColors }: Props) {
  const lineObj = useMemo(() => {
    const geometry = new THREE.BufferGeometry()
    const positions: number[] = []
    const colors: number[] = []

    const xs = data.track_centerline_x
    const ys = data.track_centerline_y
    const speeds = data.track_speed_colors
    const positiveS = speeds.filter(s => s > 0)
    const minS = positiveS.length > 0 ? Math.min(...positiveS) : 0
    const maxS = Math.max(...speeds)

    for (let i = 0; i < xs.length; i++) {
      positions.push(xs[i], 0.01, ys[i])  // slight elevation above ground
      const c = showColors ? speedToColor(speeds[i], minS, maxS) : new THREE.Color('#4b5563')
      colors.push(c.r, c.g, c.b)
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))

    const material = new THREE.LineBasicMaterial({ vertexColors: true })
    return new THREE.Line(geometry, material)
  }, [data, showColors])

  return <primitive object={lineObj} />
}
