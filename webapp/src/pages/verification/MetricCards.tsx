import type { ValidationMetric } from '../../api/client'
import { MetricCard, type MetricTone } from '../../components/ui'

const signedPctFormatter = new Intl.NumberFormat('en-US', {
  signDisplay: 'always',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

const valueFormatter = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

/** Signed delta as percent: (sim − real) / real * 100. Returns 0 when real ~ 0. */
function signedDeltaPct(metric: ValidationMetric): number {
  if (Math.abs(metric.real_value) < 1e-9) return 0
  return ((metric.sim_value - metric.real_value) / metric.real_value) * 100
}

/**
 * Tone matrix:
 *  - passed              → ok
 *  - within 2× threshold → warn
 *  - else                → error
 */
function deltaTone(metric: ValidationMetric): MetricTone {
  if (metric.passed) return 'ok'
  const abs = Math.abs(metric.error_pct)
  if (abs < metric.threshold_pct * 2) return 'warn'
  return 'error'
}

function formatValue(metric: ValidationMetric): string {
  return valueFormatter.format(metric.sim_value)
}

function formatHint(metric: ValidationMetric): string {
  const real = valueFormatter.format(metric.real_value)
  const unit = metric.unit ? metric.unit : ''
  return `Real: ${real}${unit ? ` ${unit}` : ''}`
}

export default function MetricCards({ metrics }: { metrics: ValidationMetric[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {metrics.map((m) => {
        const tone = deltaTone(m)
        const deltaPct = signedDeltaPct(m)
        return (
          <MetricCard
            key={m.name}
            label={m.name}
            value={formatValue(m)}
            unit={m.unit || undefined}
            delta={{
              value: `${signedPctFormatter.format(deltaPct)}%`,
              tone,
            }}
            hint={formatHint(m)}
            badge={{
              text: m.passed ? 'PASS' : 'FAIL',
              tone: m.passed ? 'ok' : 'error',
            }}
          />
        )
      })}
    </div>
  )
}
