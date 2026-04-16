import type { ValidationMetric } from '../../api/client'

export default function MetricCards({ metrics }: { metrics: ValidationMetric[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {metrics.map((m) => (
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
              className={`text-xs font-bold px-2 py-0.5 rounded ${
                m.passed ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'
              }`}
            >
              {m.passed ? 'PASS' : 'FAIL'}
            </span>
          </div>
          <div className="text-lg font-bold">{m.error_pct.toFixed(1)}%</div>
          <div className="text-xs text-gray-500 mt-1">
            Threshold: {m.threshold_pct}% · Sim: {m.sim_value.toFixed(2)} · Real: {m.real_value.toFixed(2)} {m.unit}
          </div>
        </div>
      ))}
    </div>
  )
}
