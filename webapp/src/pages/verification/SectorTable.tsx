import type { SectorComparison } from '../../api/client'

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
  color: string
  glyph: string
  label: string
}

function deltaStyle(deltaS: number, deltaPct: number): DeltaStyle {
  // Sim slower than real (positive delta) = red; sim faster = green; near-zero = gray.
  if (Math.abs(deltaS) < DELTA_S_NEUTRAL_THRESHOLD) {
    return { color: 'text-gray-400', glyph: '—', label: 'near-zero delta' }
  }
  const abs = Math.abs(deltaPct)
  if (deltaS > 0) {
    const tone = abs >= 10 ? 'text-red-400' : abs >= 5 ? 'text-red-300' : 'text-red-400'
    return { color: tone, glyph: '▲', label: 'sim slower than real' }
  }
  const tone = abs >= 10 ? 'text-green-400' : abs >= 5 ? 'text-green-300' : 'text-green-400'
  return { color: tone, glyph: '▼', label: 'sim faster than real' }
}

function speedDeltaStyle(pct: number): DeltaStyle {
  // For speed, positive = sim faster (good-ish, but still an error); treat magnitude-only.
  if (Math.abs(pct) < 0.5) {
    return { color: 'text-gray-400', glyph: '—', label: 'near-zero speed delta' }
  }
  const abs = Math.abs(pct)
  if (pct > 0) {
    const tone = abs >= 10 ? 'text-yellow-300' : 'text-yellow-400'
    return { color: tone, glyph: '▲', label: 'sim faster than real' }
  }
  const tone = abs >= 10 ? 'text-red-400' : 'text-red-300'
  return { color: tone, glyph: '▼', label: 'sim slower than real' }
}

export default function SectorTable({ sectors }: { sectors: SectorComparison[] }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-4 py-3 border-b border-gray-800">
        Sector Breakdown
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase">
              <th className="px-4 py-2 text-left">Sector</th>
              <th className="px-4 py-2 text-left">Type</th>
              <th className="px-4 py-2 text-right">Sim Time</th>
              <th className="px-4 py-2 text-right">Real Time</th>
              <th className="px-4 py-2 text-right">Delta</th>
              <th className="px-4 py-2 text-right">Sim Speed</th>
              <th className="px-4 py-2 text-right">Real Speed</th>
              <th className="px-4 py-2 text-right">Speed Delta</th>
            </tr>
          </thead>
          <tbody>
            {sectors.map((s) => {
              const timeStyle = deltaStyle(s.delta_s, s.delta_pct)
              const spdStyle = speedDeltaStyle(s.speed_delta_pct)
              return (
                <tr key={s.name} className="border-t border-gray-800 hover:bg-gray-800/50">
                  <td className="px-4 py-2 font-medium">{s.name}</td>
                  <td className="px-4 py-2 text-gray-400">{s.sector_type}</td>
                  <td className="px-4 py-2 text-right">{s.sim_time_s.toFixed(2)}s</td>
                  <td className="px-4 py-2 text-right">{s.real_time_s.toFixed(2)}s</td>
                  <td
                    className={`px-4 py-2 text-right ${timeStyle.color}`}
                    aria-label={`${timeStyle.label}: ${timeFormatter.format(s.delta_s)} seconds`}
                  >
                    <span aria-hidden="true" className="mr-1 inline-block w-3 text-center">
                      {timeStyle.glyph}
                    </span>
                    {timeFormatter.format(s.delta_s)}s ({pctFormatter.format(s.delta_pct)}%)
                  </td>
                  <td className="px-4 py-2 text-right">{s.sim_avg_speed_kmh.toFixed(1)}</td>
                  <td className="px-4 py-2 text-right">{s.real_avg_speed_kmh.toFixed(1)}</td>
                  <td
                    className={`px-4 py-2 text-right ${spdStyle.color}`}
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
    </div>
  )
}
