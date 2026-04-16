import { useMemo, useState } from 'react'
import type { LapSummary } from '../../api/client'

type SortKey = keyof Pick<
  LapSummary,
  | 'lap_number'
  | 'sim_time_s'
  | 'real_time_s'
  | 'time_error_pct'
  | 'sim_energy_kwh'
  | 'real_energy_kwh'
  | 'energy_error_pct'
  | 'mean_speed_error_pct'
>

type SortDir = 'asc' | 'desc'

// Michigan 2025 driver change happened mid-endurance (~lap 10-11). No precise
// per-lap marker ships in the LapSummary payload today, so we highlight the
// boundary heuristically when the full 21-lap endurance is visible.
// TODO: plumb an explicit driver_change flag through LapSummary once backend exposes it.
const DRIVER_CHANGE_LAP = 11
const DRIVER_CHANGE_MIN_LAPS = 15 // only highlight when enough laps are shown to disambiguate

function errorColor(pct: number): string {
  const abs = Math.abs(pct)
  if (abs < 5) return 'text-green-400'
  if (abs < 10) return 'text-yellow-400'
  return 'text-red-400'
}

interface HeaderProps {
  label: string
  sortKey: SortKey
  activeKey: SortKey
  activeDir: SortDir
  align?: 'left' | 'right'
  onSort: (key: SortKey) => void
}

function SortableHeader({ label, sortKey, activeKey, activeDir, align = 'right', onSort }: HeaderProps) {
  const isActive = activeKey === sortKey
  const ariaSort = isActive ? (activeDir === 'asc' ? 'ascending' : 'descending') : 'none'
  const indicator = isActive ? (activeDir === 'asc' ? '↑' : '↓') : ''
  const alignClass = align === 'right' ? 'text-right' : 'text-left'
  return (
    <th scope="col" aria-sort={ariaSort} className={`px-4 py-2 ${alignClass}`}>
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={`inline-flex items-center gap-1 uppercase tracking-wider hover:text-gray-300 focus:outline-none focus:text-white ${
          isActive ? 'text-gray-300' : 'text-gray-500'
        }`}
      >
        <span>{label}</span>
        <span aria-hidden="true" className="w-2 text-xs">
          {indicator}
        </span>
      </button>
    </th>
  )
}

export default function LapTable({ laps, selectedLap }: { laps: LapSummary[]; selectedLap: number | null }) {
  const [sortKey, setSortKey] = useState<SortKey>('lap_number')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      // Sensible defaults: lap_number asc, everything else desc (worst first).
      setSortDir(key === 'lap_number' ? 'asc' : 'desc')
    }
  }

  const sortedLaps = useMemo(() => {
    const copy = [...laps]
    copy.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av === bv) return a.lap_number - b.lap_number
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return copy
  }, [laps, sortKey, sortDir])

  const showDriverChangeMarker = laps.length >= DRIVER_CHANGE_MIN_LAPS

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider px-4 py-3 border-b border-gray-800">
        Per-Lap Summary
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs">
              <SortableHeader label="Lap" sortKey="lap_number" activeKey={sortKey} activeDir={sortDir} align="left" onSort={handleSort} />
              <SortableHeader label="Sim Time" sortKey="sim_time_s" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
              <SortableHeader label="Real Time" sortKey="real_time_s" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
              <SortableHeader label="Time Err" sortKey="time_error_pct" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
              <SortableHeader label="Sim Energy" sortKey="sim_energy_kwh" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
              <SortableHeader label="Real Energy" sortKey="real_energy_kwh" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
              <SortableHeader label="Energy Err" sortKey="energy_error_pct" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
              <SortableHeader label="Speed Err" sortKey="mean_speed_error_pct" activeKey={sortKey} activeDir={sortDir} onSort={handleSort} />
            </tr>
          </thead>
          <tbody>
            {sortedLaps.map((l) => {
              const isSelected = selectedLap === l.lap_number
              const isDriverChange = showDriverChangeMarker && l.lap_number === DRIVER_CHANGE_LAP
              const rowClass = [
                'border-t border-gray-800',
                isSelected ? 'bg-gray-800' : 'hover:bg-gray-800/50',
                isDriverChange && !isSelected ? 'bg-blue-950/30' : '',
              ]
                .filter(Boolean)
                .join(' ')
              return (
                <tr key={l.lap_number} className={rowClass}>
                  <td className="px-4 py-2 font-medium">
                    <span className="inline-flex items-center gap-1.5">
                      <span>#{l.lap_number}</span>
                      {isDriverChange && (
                        <span
                          className="text-[10px] uppercase tracking-wider text-blue-300 bg-blue-900/40 border border-blue-800 rounded px-1.5 py-0.5"
                          title="Driver change occurred around this lap in Michigan 2025 endurance"
                          aria-label="Driver change"
                        >
                          <span aria-hidden="true">ℹ </span>Driver change
                        </span>
                      )}
                    </span>
                  </td>
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
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
