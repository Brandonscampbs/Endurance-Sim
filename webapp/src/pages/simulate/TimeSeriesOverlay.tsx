import { useMemo } from 'react'
import Plot from '../../components/Plot'
import { ChartShell } from '../../components/ui'
import type { SimulateTimeSeries } from '../../api/client'
import { useSimulateStore } from '../../stores/simulateStore'

// Read these once. Plotly cannot resolve CSS variables; we mirror tokens.css.
const COLOR_SIM = '#2ec27e' // var(--data-sim)
const COLOR_BASELINE = '#d6a85a' // var(--data-baseline)
const COLOR_GRID = '#1f2530' // var(--border-subtle)
const COLOR_AXIS = '#6f7787' // var(--text-tertiary)
const COLOR_TICK = '#a4acba' // var(--text-secondary)

interface PaneProps {
  title: string
  yLabel: string
  sim: number[]
  baseline: number[]
  distanceSim: number[]
  distanceBaseline: number[]
  loading: boolean
}

function Pane({
  title,
  yLabel,
  sim,
  baseline,
  distanceSim,
  distanceBaseline,
  loading,
}: PaneProps) {
  return (
    <ChartShell title={title} loading={loading} height={260}>
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            x: distanceBaseline,
            y: baseline,
            name: 'Baseline',
            line: { color: COLOR_BASELINE, width: 1.4 },
            hovertemplate: '%{y:.2f}<extra>Baseline</extra>',
          },
          {
            type: 'scatter',
            mode: 'lines',
            x: distanceSim,
            y: sim,
            name: 'Sim',
            line: { color: COLOR_SIM, width: 1.6 },
            hovertemplate: '%{y:.2f}<extra>Sim</extra>',
          },
        ]}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          margin: { t: 8, b: 36, l: 52, r: 12 },
          xaxis: {
            title: { text: 'Distance (m)', font: { color: COLOR_AXIS, size: 11 } },
            color: COLOR_TICK,
            gridcolor: COLOR_GRID,
            zerolinecolor: COLOR_GRID,
            tickfont: { color: COLOR_TICK, size: 10 },
          },
          yaxis: {
            title: { text: yLabel, font: { color: COLOR_AXIS, size: 11 } },
            color: COLOR_TICK,
            gridcolor: COLOR_GRID,
            zerolinecolor: COLOR_GRID,
            tickfont: { color: COLOR_TICK, size: 10 },
          },
          legend: {
            orientation: 'h',
            font: { color: COLOR_TICK, size: 11 },
            bgcolor: 'transparent',
            x: 1,
            xanchor: 'right',
            y: 1.05,
            yanchor: 'bottom',
          },
          hovermode: 'x unified',
        }}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full h-full"
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </ChartShell>
  )
}

export default function TimeSeriesOverlay() {
  const status = useSimulateStore((s) => s.status)
  const result = useSimulateStore((s) => s.result)
  const isRunning = status === 'running'

  // Convert pack power from watts to kW for readability.
  const packPowerKwSim = useMemo(
    () => result?.time_series.pack_power_w.map((w) => w / 1000) ?? [],
    [result],
  )
  const packPowerKwBaseline = useMemo(
    () => result?.baseline_time_series.pack_power_w.map((w) => w / 1000) ?? [],
    [result],
  )

  if (!result && !isRunning) return null

  const sim: SimulateTimeSeries | null = result?.time_series ?? null
  const baseline: SimulateTimeSeries | null = result?.baseline_time_series ?? null

  const distSim = sim?.distance_m ?? []
  const distBaseline = baseline?.distance_m ?? []

  const panes = [
    {
      title: 'Speed vs distance',
      yLabel: 'Speed (km/h)',
      sim: sim?.speed_kmh ?? [],
      baseline: baseline?.speed_kmh ?? [],
    },
    {
      title: 'Pack power vs distance',
      yLabel: 'Pack power (kW)',
      sim: packPowerKwSim,
      baseline: packPowerKwBaseline,
    },
    {
      title: 'SOC vs distance',
      yLabel: 'SOC (%)',
      sim: sim?.soc_pct ?? [],
      baseline: baseline?.soc_pct ?? [],
    },
    {
      title: 'Cumulative charge vs distance',
      yLabel: 'Charge (Ah)',
      sim: sim?.cumulative_charge_ah ?? [],
      baseline: baseline?.cumulative_charge_ah ?? [],
    },
  ]

  return (
    <section className="grid grid-cols-2 gap-3">
      {panes.map((p) => (
        <Pane
          key={p.title}
          title={p.title}
          yLabel={p.yLabel}
          sim={p.sim}
          baseline={p.baseline}
          distanceSim={distSim}
          distanceBaseline={distBaseline}
          loading={isRunning && (sim === null || baseline === null)}
        />
      ))}
    </section>
  )
}
