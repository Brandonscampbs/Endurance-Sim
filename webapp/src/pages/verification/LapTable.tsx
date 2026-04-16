import type { LapSummary } from '../../api/client'

function errorColor(pct: number): string {
  if (pct < 5) return 'text-green-400'
  if (pct < 10) return 'text-yellow-400'
  return 'text-red-400'
}

export default function LapTable({ laps, selectedLap }: { laps: LapSummary[]; selectedLap: number | null }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-4 py-3 border-b border-gray-800">
        Per-Lap Summary
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase">
              <th className="px-4 py-2 text-left">Lap</th>
              <th className="px-4 py-2 text-right">Sim Time</th>
              <th className="px-4 py-2 text-right">Real Time</th>
              <th className="px-4 py-2 text-right">Time Err</th>
              <th className="px-4 py-2 text-right">Sim Energy</th>
              <th className="px-4 py-2 text-right">Real Energy</th>
              <th className="px-4 py-2 text-right">Energy Err</th>
              <th className="px-4 py-2 text-right">Speed Err</th>
            </tr>
          </thead>
          <tbody>
            {laps.map((l) => (
              <tr
                key={l.lap_number}
                className={`border-t border-gray-800 ${
                  selectedLap === l.lap_number ? 'bg-gray-800' : 'hover:bg-gray-800/50'
                }`}
              >
                <td className="px-4 py-2 font-medium">#{l.lap_number}</td>
                <td className="px-4 py-2 text-right">{l.sim_time_s.toFixed(2)}s</td>
                <td className="px-4 py-2 text-right">{l.real_time_s.toFixed(2)}s</td>
                <td className={`px-4 py-2 text-right ${errorColor(l.time_error_pct)}`}>
                  {l.time_error_pct.toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right">{l.sim_energy_kwh.toFixed(3)}</td>
                <td className="px-4 py-2 text-right">{l.real_energy_kwh.toFixed(3)}</td>
                <td className={`px-4 py-2 text-right ${errorColor(l.energy_error_pct)}`}>
                  {l.energy_error_pct.toFixed(1)}%
                </td>
                <td className={`px-4 py-2 text-right ${errorColor(l.mean_speed_error_pct)}`}>
                  {l.mean_speed_error_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
