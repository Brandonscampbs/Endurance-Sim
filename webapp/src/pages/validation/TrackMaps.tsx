import Plot from 'react-plotly.js'
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

export default function TrackMaps({ track, validation }: Props) {
  const xs = track.centerline.map(p => p.x)
  const ys = track.centerline.map(p => p.y)

  const maxSpeed = Math.max(
    ...validation.track_sim_speed,
    ...validation.track_real_speed,
  )
  const minSpeed = Math.min(
    ...validation.track_sim_speed.filter(v => v > 0),
    ...validation.track_real_speed.filter(v => v > 0),
  )

  const layout = (title: string): Partial<Plotly.Layout> => ({
    title: { text: title, font: { color: '#9ca3af', size: 14 } },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: { visible: false, scaleanchor: 'y', scaleratio: 1 },
    yaxis: { visible: false },
    margin: { t: 40, b: 10, l: 10, r: 10 },
    showlegend: false,
  })

  const makeTrace = (speeds: number[]): Plotly.Data => ({
    type: 'scatter',
    mode: 'markers',
    x: xs,
    y: ys,
    marker: {
      color: speeds,
      colorscale: COLOR_SCALE as unknown as Plotly.ColorScale,
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
