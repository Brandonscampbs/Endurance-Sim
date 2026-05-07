import { useMemo } from 'react'
import { Card, CardBody, CardHeader, CardTitle, SkeletonText } from '../../components/ui'
import type { SimulateLapResult } from '../../api/client'
import { formatSignedDelta } from '../../utils/formatters'
import { useSimulateStore } from '../../stores/simulateStore'

interface MergedRow {
  lap: number
  sim: SimulateLapResult | null
  baseline: SimulateLapResult | null
}

function mergeLaps(
  sim: SimulateLapResult[],
  baseline: SimulateLapResult[],
): MergedRow[] {
  const map = new Map<number, MergedRow>()
  for (const lap of sim) {
    map.set(lap.lap_number, { lap: lap.lap_number, sim: lap, baseline: null })
  }
  for (const lap of baseline) {
    const existing = map.get(lap.lap_number)
    if (existing) {
      existing.baseline = lap
    } else {
      map.set(lap.lap_number, { lap: lap.lap_number, sim: null, baseline: lap })
    }
  }
  return Array.from(map.values()).sort((a, b) => a.lap - b.lap)
}

/** Color a delta by tone given a "lower-is-better" interpretation. */
function deltaClass(delta: number | null, eps: number): string {
  if (delta === null) return 'text-[var(--text-muted)]'
  if (Math.abs(delta) < eps) return 'text-[var(--text-tertiary)]'
  return delta < 0 ? 'text-[var(--ok)]' : 'text-[var(--error)]'
}

function fmt(v: number | undefined | null, digits: number): string {
  if (v === undefined || v === null || !Number.isFinite(v)) return '—'
  return v.toFixed(digits)
}

function delta(
  sim: number | undefined,
  baseline: number | undefined,
): number | null {
  if (
    sim === undefined ||
    baseline === undefined ||
    !Number.isFinite(sim) ||
    !Number.isFinite(baseline)
  ) {
    return null
  }
  return sim - baseline
}

export default function LapDeltaTable() {
  const status = useSimulateStore((s) => s.status)
  const result = useSimulateStore((s) => s.result)
  const isRunning = status === 'running'

  const rows = useMemo<MergedRow[]>(
    () => (result ? mergeLaps(result.laps, result.baseline_laps) : []),
    [result],
  )

  if (!result && !isRunning) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle>Per-lap results</CardTitle>
      </CardHeader>
      <CardBody className="px-0 py-0">
        {isRunning && rows.length === 0 ? (
          <div className="px-5 py-4">
            <SkeletonText lines={6} />
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[480px]">
            <table className="w-full text-sm tabular-nums">
              <thead className="sticky top-0 bg-[var(--surface-1)]">
                <tr className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)] border-b border-[var(--border-subtle)]">
                  <th scope="col" className="text-left px-4 py-2 font-medium">
                    Lap
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Sim time
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Baseline time
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Δ time
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Sim Ah
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Baseline Ah
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Δ Ah
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Sim kWh
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Baseline kWh
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Δ kWh
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Mean (km/h)
                  </th>
                  <th scope="col" className="text-right px-3 py-2 font-medium">
                    Peak (km/h)
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => {
                  const dt = delta(row.sim?.time_s, row.baseline?.time_s)
                  const dah = delta(row.sim?.charge_used_ah, row.baseline?.charge_used_ah)
                  const dkwh = delta(row.sim?.energy_used_kwh, row.baseline?.energy_used_kwh)
                  const rowBg = i % 2 === 1 ? 'bg-[var(--surface-2)]/40' : ''
                  return (
                    <tr
                      key={row.lap}
                      className={`${rowBg} border-t border-[var(--border-subtle)]/40 hover:bg-[var(--surface-2)]`}
                    >
                      <td className="px-4 py-2 font-medium">#{row.lap}</td>
                      <td className="px-3 py-2 text-right">
                        {fmt(row.sim?.time_s, 2)}s
                      </td>
                      <td className="px-3 py-2 text-right text-[var(--text-secondary)]">
                        {fmt(row.baseline?.time_s, 2)}s
                      </td>
                      <td className={`px-3 py-2 text-right ${deltaClass(dt, 0.05)}`}>
                        {dt === null ? '—' : `${formatSignedDelta(dt, 2, 's')}`}
                      </td>
                      <td className="px-3 py-2 text-right">{fmt(row.sim?.charge_used_ah, 2)}</td>
                      <td className="px-3 py-2 text-right text-[var(--text-secondary)]">
                        {fmt(row.baseline?.charge_used_ah, 2)}
                      </td>
                      <td className={`px-3 py-2 text-right ${deltaClass(dah, 0.005)}`}>
                        {dah === null ? '—' : formatSignedDelta(dah, 2)}
                      </td>
                      <td className="px-3 py-2 text-right">{fmt(row.sim?.energy_used_kwh, 3)}</td>
                      <td className="px-3 py-2 text-right text-[var(--text-secondary)]">
                        {fmt(row.baseline?.energy_used_kwh, 3)}
                      </td>
                      <td className={`px-3 py-2 text-right ${deltaClass(dkwh, 0.0005)}`}>
                        {dkwh === null ? '—' : formatSignedDelta(dkwh, 3)}
                      </td>
                      <td className="px-3 py-2 text-right">{fmt(row.sim?.mean_speed_kmh, 1)}</td>
                      <td className="px-3 py-2 text-right">{fmt(row.sim?.peak_speed_kmh, 1)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardBody>
    </Card>
  )
}
