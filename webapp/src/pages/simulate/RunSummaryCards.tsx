import { MetricCard, type MetricTone } from '../../components/ui'
import type { SimulateAggregate } from '../../api/client'
import {
  formatDurationHMS,
  formatSignedDelta,
  formatSignedPct,
} from '../../utils/formatters'
import { useSimulateStore } from '../../stores/simulateStore'

/** Tone for "lower is better" metrics (time, charge, energy). */
function lowerBetterTone(deltaPct: number): MetricTone {
  if (Math.abs(deltaPct) < 0.5) return 'neutral'
  if (deltaPct < 0) return 'ok'
  if (deltaPct < 5) return 'warn'
  return 'error'
}

/** Tone for "higher is better" metrics (final SOC). */
function higherBetterTone(deltaPct: number): MetricTone {
  if (Math.abs(deltaPct) < 0.5) return 'neutral'
  if (deltaPct > 0) return 'ok'
  if (deltaPct > -5) return 'warn'
  return 'error'
}

/** Tone for "completed laps" — match-target wins, fewer than baseline is bad. */
function lapsTone(
  completed: number,
  baselineCompleted: number,
  target: number,
): MetricTone {
  if (completed >= target) return 'ok'
  if (completed < baselineCompleted) return 'error'
  if (completed > baselineCompleted) return 'ok'
  return 'neutral'
}

function deltaPct(sim: number, baseline: number): number {
  if (Math.abs(baseline) < 1e-9) return 0
  return ((sim - baseline) / baseline) * 100
}

interface SummaryProps {
  sim: SimulateAggregate
  baseline: SimulateAggregate
}

function buildSummary({ sim, baseline }: SummaryProps) {
  const timeDeltaSec = sim.total_time_s - baseline.total_time_s
  const timeDeltaPct = deltaPct(sim.total_time_s, baseline.total_time_s)

  const ahDelta = sim.net_ah - baseline.net_ah
  const ahDeltaPct = deltaPct(sim.net_ah, baseline.net_ah)

  const kwhDelta = sim.net_kwh - baseline.net_kwh
  const kwhDeltaPct = deltaPct(sim.net_kwh, baseline.net_kwh)

  const socDelta = sim.final_soc_pct - baseline.final_soc_pct
  const socDeltaPct = deltaPct(sim.final_soc_pct, baseline.final_soc_pct)

  const lapsDelta = sim.completed_laps - baseline.completed_laps

  return [
    {
      label: 'Total time',
      value: formatDurationHMS(sim.total_time_s),
      unit: '',
      hint: `${sim.total_time_s.toFixed(1)} s`,
      delta: {
        value: `${formatSignedDelta(timeDeltaSec, 1, 's')} (${formatSignedPct(timeDeltaPct)})`,
        tone: lowerBetterTone(timeDeltaPct),
        label: 'vs baseline',
      },
    },
    {
      label: 'Completed laps',
      value: `${sim.completed_laps} / ${sim.target_laps}`,
      unit: '',
      hint: `baseline ${baseline.completed_laps}`,
      delta: {
        value: lapsDelta === 0 ? '±0 laps' : formatSignedDelta(lapsDelta, 1, ' laps'),
        tone: lapsTone(sim.completed_laps, baseline.completed_laps, sim.target_laps),
        label: 'vs baseline',
      },
    },
    {
      label: 'Net charge',
      value: sim.net_ah.toFixed(2),
      unit: 'Ah',
      delta: {
        value: `${formatSignedDelta(ahDelta, 2, ' Ah')} (${formatSignedPct(ahDeltaPct)})`,
        tone: lowerBetterTone(ahDeltaPct),
        label: 'vs baseline',
      },
    },
    {
      label: 'Net energy',
      value: sim.net_kwh.toFixed(3),
      unit: 'kWh',
      delta: {
        value: `${formatSignedDelta(kwhDelta, 3, ' kWh')} (${formatSignedPct(kwhDeltaPct)})`,
        tone: lowerBetterTone(kwhDeltaPct),
        label: 'vs baseline',
      },
    },
    {
      label: 'Final SOC',
      value: sim.final_soc_pct.toFixed(1),
      unit: '%',
      delta: {
        value: `${formatSignedDelta(socDelta, 1, '%')} (${formatSignedPct(socDeltaPct)})`,
        tone: higherBetterTone(socDeltaPct),
        label: 'vs baseline',
      },
    },
  ]
}

export default function RunSummaryCards() {
  const status = useSimulateStore((s) => s.status)
  const result = useSimulateStore((s) => s.result)

  const isRunning = status === 'running'

  // Show 5 loading-state cards while a run is in flight even if there's no
  // prior result; once we have a result the previous shape stays put.
  const items = result
    ? buildSummary({ sim: result.summary, baseline: result.baseline })
    : null

  // Static placeholder labels for the loading state.
  const loadingLabels = [
    'Total time',
    'Completed laps',
    'Net charge',
    'Net energy',
    'Final SOC',
  ]

  if (!items && !isRunning) return null

  return (
    <section className="grid grid-cols-5 gap-3">
      {(items
        ? items.map((m) => (
            <MetricCard
              key={m.label}
              label={m.label}
              value={m.value}
              unit={m.unit}
              hint={m.hint}
              delta={m.delta}
              loading={isRunning}
            />
          ))
        : loadingLabels.map((l) => (
            <MetricCard key={l} label={l} value="" loading={true} />
          )))}
    </section>
  )
}
