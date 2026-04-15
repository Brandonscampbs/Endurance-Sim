import type { SectorComparison } from '../../api/client'

function deltaColor(pct: number): string {
  const abs = Math.abs(pct)
  if (abs < 5) return 'text-green-400'
  if (abs < 10) return 'text-yellow-400'
  return 'text-red-400'
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
            {sectors.map((s) => (
              <tr key={s.name} className="border-t border-gray-800 hover:bg-gray-800/50">
                <td className="px-4 py-2 font-medium">{s.name}</td>
                <td className="px-4 py-2 text-gray-400">{s.sector_type}</td>
                <td className="px-4 py-2 text-right">{s.sim_time_s.toFixed(2)}s</td>
                <td className="px-4 py-2 text-right">{s.real_time_s.toFixed(2)}s</td>
                <td className={`px-4 py-2 text-right ${deltaColor(s.delta_pct)}`}>
                  {s.delta_s > 0 ? '+' : ''}{s.delta_s.toFixed(2)}s ({s.delta_pct.toFixed(1)}%)
                </td>
                <td className="px-4 py-2 text-right">{s.sim_avg_speed_kmh.toFixed(1)}</td>
                <td className="px-4 py-2 text-right">{s.real_avg_speed_kmh.toFixed(1)}</td>
                <td className={`px-4 py-2 text-right ${deltaColor(s.speed_delta_pct)}`}>
                  {s.speed_delta_pct > 0 ? '+' : ''}{s.speed_delta_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
