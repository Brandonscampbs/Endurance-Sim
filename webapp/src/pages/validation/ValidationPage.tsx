import { useEffect, lazy, Suspense } from 'react'
import { useValidation, useAllLaps, useTrack, useLaps } from '../../api/client'
import { useValidationStore } from '../../stores/validationStore'
import LoadingSpinner from '../../components/LoadingSpinner'
import SectorTable from './SectorTable'
import LapTable from './LapTable'
import MetricCards from './MetricCards'

const TrackMaps = lazy(() => import('./TrackMaps'))
const OverlayCharts = lazy(() => import('./OverlayCharts'))

export default function ValidationPage() {
  const { selectedLap, setSelectedLap } = useValidationStore()
  const { data: lapsData } = useLaps()
  const { data: track, isLoading: trackLoading } = useTrack()
  const { data: validation, isLoading: validationLoading } = useValidation(selectedLap)
  const { data: allLaps, isLoading: allLapsLoading } = useAllLaps()

  // Default to best GPS quality lap
  const bestLap = lapsData?.laps
    ? [...lapsData.laps].sort((a, b) => a.gps_quality_score - b.gps_quality_score)[0]?.lap_number ?? 1
    : 1

  // Set initial lap on first data load
  useEffect(() => {
    if (lapsData && selectedLap === null) {
      setSelectedLap(bestLap)
    }
  }, [lapsData, selectedLap, bestLap, setSelectedLap])

  return (
    <div className="space-y-6">
      {/* Header + Lap Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Validation</h2>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-400">Lap:</label>
          <select
            value={selectedLap ?? 'all'}
            onChange={(e) => setSelectedLap(e.target.value === 'all' ? null : Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          >
            <option value="all">All Laps</option>
            {lapsData?.laps.map((l) => (
              <option key={l.lap_number} value={l.lap_number}>
                Lap {l.lap_number} — {l.time_s.toFixed(1)}s (GPS: {l.gps_quality_score})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Single-lap view */}
      {selectedLap !== null && (
        <>
          {trackLoading || validationLoading ? (
            <LoadingSpinner message="Running simulation and loading telemetry..." />
          ) : track && validation ? (
            <>
              <Suspense fallback={<LoadingSpinner message="Loading charts..." />}>
                <TrackMaps track={track} validation={validation} />
                <OverlayCharts validation={validation} />
              </Suspense>
              <SectorTable sectors={validation.sectors} />
              <MetricCards metrics={validation.metrics} />
            </>
          ) : (
            <p className="text-red-400">Failed to load data.</p>
          )}
        </>
      )}

      {/* All-laps view */}
      {selectedLap === null && (
        <>
          {allLapsLoading ? (
            <LoadingSpinner message="Computing all-laps summary..." />
          ) : allLaps ? (
            <>
              <MetricCards metrics={allLaps.metrics} />
              <LapTable laps={allLaps.laps} selectedLap={null} />
            </>
          ) : (
            <p className="text-red-400">Failed to load data.</p>
          )}
        </>
      )}
    </div>
  )
}
