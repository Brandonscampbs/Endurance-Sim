import Plot from '../../components/Plot'
import type { TrackData, ValidationResponse } from '../../api/client'

interface Props {
  track: TrackData
  validation: ValidationResponse
}

const COLOR_SCALE: [number, string][] = [
  [0, '#3b82f6'],    // blue - slow
  [0.25, '#22c55e'], // green
  [0.5, '#eab308'],  // yellow
  [0.75, '#f97316'], // orange
  [1, '#ef4444'],    // red - fast
]

// Spread-free min/max: large speed arrays (~40k points) blow the JS engine's
// argument-count limit when passed via Math.min(...arr) / Math.max(...arr).
// Also skips non-finite values.
function reduceMin(arrs: number[][], predicate?: (v: number) => boolean): number {
  let m = Infinity
  for (const arr of arrs) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i]
      if (!Number.isFinite(v)) continue
      if (predicate && !predicate(v)) continue
      if (v < m) m = v
    }
  }
  return m
}
function reduceMax(arrs: number[][], predicate?: (v: number) => boolean): number {
  let m = -Infinity
  for (const arr of arrs) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i]
      if (!Number.isFinite(v)) continue
      if (predicate && !predicate(v)) continue
      if (v > m) m = v
    }
  }
  return m
}

export default function TrackMaps({ track, validation }: Props) {
  const xs = track.centerline.map(p => p.x)
  const ys = track.centerline.map(p => p.y)

  const rawMax = reduceMax([validation.track_sim_speed, validation.track_real_speed])
  const rawMin = reduceMin(
    [validation.track_sim_speed, validation.track_real_speed],
    v => v > 0,
  )
  const maxSpeed = Number.isFinite(rawMax) ? rawMax : 1
  const minSpeed = Number.isFinite(rawMin) ? rawMin : 0

  const layout = (title: string) => ({
    title: { text: title, font: { color: '#9ca3af', size: 14 } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { visible: false, scaleanchor: 'y', scaleratio: 1 },
    yaxis: { visible: false },
    margin: { t: 40, b: 10, l: 10, r: 10 },
    showlegend: false,
  })

  const makeTrace = (speeds: number[]) => ({
    type: 'scatter' as const,
    mode: 'markers' as const,
    x: xs,
    y: ys,
    marker: {
      color: speeds,
      colorscale: COLOR_SCALE,
      cmin: minSpeed,
      cmax: maxSpeed,
      size: 4,
      colorbar: { title: 'km/h', tickfont: { color: '#9ca3af' }, titlefont: { color: '#9ca3af' } },
    },
    hovertemplate: 'Speed: %{marker.color:.1f} km/h<extra></extra>',
  })

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-gray-900 rounded-lg p-2">
        <Plot
          data={[makeTrace(validation.track_sim_speed)]}
          layout={layout('Simulation Speed')}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          style={{ height: 400 }}
        />
      </div>
      <div className="bg-gray-900 rounded-lg p-2">
        <Plot
          data={[makeTrace(validation.track_real_speed)]}
          layout={layout('Real Telemetry Speed')}
          config={{ responsive: true, displayModeBar: false }}
          className="w-full"
          style={{ height: 400 }}
        />
      </div>
    </div>
  )
}
