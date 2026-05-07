import type { SectorComparison } from '../../api/client'
import { Card, CardHeader, CardTitle, CardBody } from '../../components/ui'

const DELTA_S_NEUTRAL_THRESHOLD = 0.05 // seconds

const timeFormatter = new Intl.NumberFormat('en-US', {
  signDisplay: 'always',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const pctFormatter = new Intl.NumberFormat('en-US', {
  signDisplay: 'always',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

interface DeltaStyle {
  /** Inline color (CSS variable or computed value). */
  color: string
  glyph: string
  label: string
}

function deltaStyle(deltaS: number, deltaPct: number): DeltaStyle {
  // Sim slower than real (positive delta) = error tone; sim faster = ok tone; near-zero = muted.
  if (Math.abs(deltaS) < DELTA_S_NEUTRAL_THRESHOLD) {
    return { color: 'var(--text-tertiary)', glyph: '—', label: 'near-zero delta' }
  }
  const abs = Math.abs(deltaPct)
  if (deltaS > 0) {
    // Slower: warn for small misses, error for large ones.
    const tone = abs >= 5 ? 'var(--error)' : 'var(--warn)'
    return { color: tone, glyph: '▲', label: 'sim slower than real' }
  }
  const tone = abs >= 5 ? 'var(--ok)' : 'var(--ok)'
  return { color: tone, glyph: '▼', label: 'sim faster than real' }
}

function speedDeltaStyle(pct: number): DeltaStyle {
  // For speed, positive = sim faster (good-ish, but still an error); treat magnitude-only.
  if (Math.abs(pct) < 0.5) {
    return { color: 'var(--text-tertiary)', glyph: '—', label: 'near-zero speed delta' }
  }
  const abs = Math.abs(pct)
  if (pct > 0) {
    const tone = abs >= 10 ? 'var(--warn)' : 'var(--warn)'
    return { color: tone, glyph: '▲', label: 'sim faster than real' }
  }
  const tone = abs >= 10 ? 'var(--error)' : 'var(--error)'
  return { color: tone, glyph: '▼', label: 'sim slower than real' }
}

export default function SectorTable({ sectors }: { sectors: SectorComparison[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Sector breakdown</CardTitle>
      </CardHeader>
      <CardBody className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[var(--text-tertiary)] text-xs uppercase tracking-wider">
                <th scope="col" className="px-4 py-2 text-left">Sector</th>
                <th scope="col" className="px-4 py-2 text-left">Type</th>
                <th scope="col" className="px-4 py-2 text-right">Sim Time</th>
                <th scope="col" className="px-4 py-2 text-right">Real Time</th>
                <th scope="col" className="px-4 py-2 text-right">Delta</th>
                <th scope="col" className="px-4 py-2 text-right">Sim Speed</th>
                <th scope="col" className="px-4 py-2 text-right">Real Speed</th>
                <th scope="col" className="px-4 py-2 text-right">Speed Delta</th>
              </tr>
            </thead>
            <tbody>
              {sectors.map((s) => {
                const timeStyle = deltaStyle(s.delta_s, s.delta_pct)
                const spdStyle = speedDeltaStyle(s.speed_delta_pct)
                return (
                  <tr
                    key={s.name}
                    className="border-t border-[var(--border-subtle)] hover:bg-[var(--surface-2)]"
                  >
                    <td className="px-4 py-2 font-medium text-[var(--text-primary)]">{s.name}</td>
                    <td className="px-4 py-2 text-[var(--text-tertiary)]">{s.sector_type}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{s.sim_time_s.toFixed(2)}s</td>
                    <td className="px-4 py-2 text-right tabular-nums">{s.real_time_s.toFixed(2)}s</td>
                    <td
                      className="px-4 py-2 text-right tabular-nums"
                      style={{ color: timeStyle.color }}
                      aria-label={`${timeStyle.label}: ${timeFormatter.format(s.delta_s)} seconds`}
                    >
                      <span aria-hidden="true" className="mr-1 inline-block w-3 text-center">
                        {timeStyle.glyph}
                      </span>
                      {timeFormatter.format(s.delta_s)}s ({pctFormatter.format(s.delta_pct)}%)
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums">{s.sim_avg_speed_kmh.toFixed(1)}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{s.real_avg_speed_kmh.toFixed(1)}</td>
                    <td
                      className="px-4 py-2 text-right tabular-nums"
                      style={{ color: spdStyle.color }}
                      aria-label={`${spdStyle.label}: ${pctFormatter.format(s.speed_delta_pct)} percent`}
                    >
                      <span aria-hidden="true" className="mr-1 inline-block w-3 text-center">
                        {spdStyle.glyph}
                      </span>
                      {pctFormatter.format(s.speed_delta_pct)}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </CardBody>
    </Card>
  )
}
