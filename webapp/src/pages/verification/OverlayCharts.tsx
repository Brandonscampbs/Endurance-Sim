import Plot from '../../components/Plot'
import type { TraceData, ValidationResponse } from '../../api/client'
import { ChartShell } from '../../components/ui'

type ChannelKey = 'speed' | 'throttle' | 'brake' | 'power' | 'charge_ah' | 'lat_accel'

interface ChartSpec {
  key: ChannelKey
  trace: TraceData
  title: string
  subtitle: string
  yLabel: string
}

interface Props {
  validation: ValidationResponse
}

// Hex equivalents of the --data-sim and --data-real CSS tokens. Plotly's SVG
// color attributes resolve at chart render time and don't pick up CSS vars,
// so we duplicate the values here. Keep in sync with `webapp/src/styles/tokens.css`.
const SIM_COLOR = '#2ec27e' // --data-sim
const REAL_COLOR = '#5b9dd9' // --data-real
const TEXT_TERTIARY = '#6f7787' // --text-tertiary
const TEXT_SECONDARY = '#a4acba' // --text-secondary
const BORDER_SUBTLE = '#1f2530' // --border-subtle
const BORDER_STRONG = '#2c3340' // --border-strong

function OverlayChart({
  trace,
  title,
  subtitle,
  yLabel,
}: {
  trace: TraceData
  title: string
  subtitle: string
  yLabel: string
}) {
  return (
    <ChartShell title={title} subtitle={subtitle} loading={false} height={250}>
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            x: trace.distance_m,
            y: trace.sim,
            name: 'Sim',
            line: { color: SIM_COLOR, width: 1.5 },
          },
          {
            type: 'scatter',
            mode: 'lines',
            x: trace.distance_m,
            y: trace.real,
            name: 'Real',
            line: { color: REAL_COLOR, width: 1.5 },
          },
        ]}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: {
            title: 'Distance (m)',
            color: TEXT_TERTIARY,
            gridcolor: BORDER_SUBTLE,
            zerolinecolor: BORDER_STRONG,
          },
          yaxis: {
            title: yLabel,
            color: TEXT_TERTIARY,
            gridcolor: BORDER_SUBTLE,
            zerolinecolor: BORDER_STRONG,
          },
          legend: {
            font: { color: TEXT_SECONDARY },
            bgcolor: 'transparent',
            x: 1,
            xanchor: 'right',
            y: 1,
          },
          margin: { t: 8, b: 50, l: 60, r: 20 },
          hovermode: 'x unified',
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full h-full"
        style={{ width: '100%', height: '100%' }}
      />
    </ChartShell>
  )
}

export default function OverlayCharts({ validation }: Props) {
  const charts: ChartSpec[] = [
    {
      key: 'speed',
      trace: validation.speed,
      title: 'Speed',
      subtitle: 'Sim vs real, km/h along distance',
      yLabel: 'Speed (km/h)',
    },
    {
      key: 'throttle',
      trace: validation.throttle,
      title: 'Throttle',
      subtitle: 'Driver pedal position, %',
      yLabel: 'Throttle (%)',
    },
    {
      key: 'brake',
      trace: validation.brake,
      title: 'Brake',
      subtitle: 'Brake pedal application, %',
      yLabel: 'Brake (%)',
    },
    {
      key: 'power',
      trace: validation.power,
      title: 'Electrical power',
      subtitle: 'Pack terminal V × I, watts',
      yLabel: 'Power (W)',
    },
    {
      key: 'charge_ah',
      trace: validation.charge_ah,
      title: 'Charge used',
      subtitle: 'Cumulative net amp-hours from pack',
      yLabel: 'Charge (Ah)',
    },
    {
      key: 'lat_accel',
      trace: validation.lat_accel,
      title: 'Lateral acceleration',
      subtitle: 'Cornering load, g',
      yLabel: 'Lat accel (g)',
    },
  ]

  return (
    <div className="space-y-4">
      {charts.map(({ key, trace, title, subtitle, yLabel }) => (
        <OverlayChart key={key} trace={trace} title={title} subtitle={subtitle} yLabel={yLabel} />
      ))}
    </div>
  )
}
