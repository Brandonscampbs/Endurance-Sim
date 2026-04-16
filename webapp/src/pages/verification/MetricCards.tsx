import type { ValidationMetric } from '../../api/client'

const errorFormatter = new Intl.NumberFormat('en-US', {
  signDisplay: 'always',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

export default function MetricCards({ metrics }: { metrics: ValidationMetric[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {metrics.map((m) => {
        const statusGlyph = m.passed ? '✓' : '✗'
        const statusLabel = m.passed ? 'Passed' : 'Failed'
        return (
          <div
            key={m.name}
            className={`rounded-lg p-4 border ${
              m.passed
                ? 'bg-green-950/30 border-green-800'
                : 'bg-red-950/30 border-red-800'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-400 uppercase">{m.name}</span>
              <span
                aria-label={statusLabel}
                className={`inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded ${
                  m.passed ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'
                }`}
              >
                <span aria-hidden="true">{statusGlyph}</span>
                <span>{m.passed ? 'PASS' : 'FAIL'}</span>
              </span>
            </div>
            <div
              className="text-lg font-bold"
              title={`Threshold: ${m.threshold_pct}% (sim vs real)`}
            >
              {errorFormatter.format(m.error_pct)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Threshold: {m.threshold_pct}% · Sim: {m.sim_value.toFixed(2)} · Real: {m.real_value.toFixed(2)} {m.unit}
            </div>
          </div>
        )
      })}
    </div>
  )
}
