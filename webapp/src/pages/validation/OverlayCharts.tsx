import Plot from 'react-plotly.js'
import type { TraceData, ValidationResponse } from '../../api/client'

interface Props {
  validation: ValidationResponse
}

function OverlayChart({ trace, title, yLabel }: { trace: TraceData; title: string; yLabel: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-2">
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            x: trace.distance_m,
            y: trace.sim,
            name: 'Sim',
            line: { color: '#22c55e', width: 1.5 },
          },
          {
            type: 'scatter',
            mode: 'lines',
            x: trace.distance_m,
            y: trace.real,
            name: 'Real',
            line: { color: '#3b82f6', width: 1.5 },
          },
        ]}
        layout={{
          title: { text: title, font: { color: '#9ca3af', size: 13 } },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: 'Distance (m)',
            color: '#6b7280',
            gridcolor: '#1f2937',
            zerolinecolor: '#374151',
          },
          yaxis: {
            title: yLabel,
            color: '#6b7280',
            gridcolor: '#1f2937',
            zerolinecolor: '#374151',
          },
          legend: {
            font: { color: '#9ca3af' },
            bgcolor: 'transparent',
            x: 1, xanchor: 'right', y: 1,
          },
          margin: { t: 40, b: 50, l: 60, r: 20 },
          hovermode: 'x unified',
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
        style={{ height: 250 }}
      />
    </div>
  )
}

export default function OverlayCharts({ validation }: Props) {
  const charts: { trace: TraceData; title: string; yLabel: string }[] = [
    { trace: validation.speed, title: 'Speed vs Distance', yLabel: 'Speed (km/h)' },
    { trace: validation.throttle, title: 'Throttle vs Distance', yLabel: 'Throttle (%)' },
    { trace: validation.brake, title: 'Brake vs Distance', yLabel: 'Brake (%)' },
    { trace: validation.power, title: 'Electrical Power vs Distance', yLabel: 'Power (W)' },
    { trace: validation.soc, title: 'SOC vs Distance', yLabel: 'SOC (%)' },
    { trace: validation.lat_accel, title: 'Lateral Acceleration vs Distance', yLabel: 'Lat Accel (g)' },
  ]

  return (
    <div className="space-y-4">
      {charts.map(({ trace, title, yLabel }) => (
        <OverlayChart key={title} trace={trace} title={title} yLabel={yLabel} />
      ))}
    </div>
  )
}
