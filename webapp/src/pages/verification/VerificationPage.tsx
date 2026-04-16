import { lazy, Suspense, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  useValidation,
  useAllLaps,
  useTrack,
  useLaps,
  ApiError,
} from '../../api/client'
import LoadingSpinner from '../../components/LoadingSpinner'
import SectorTable from './SectorTable'
import LapTable from './LapTable'
import MetricCards from './MetricCards'

const TrackMaps = lazy(() => import('./TrackMaps'))
const OverlayCharts = lazy(() => import('./OverlayCharts'))

function errorMessage(err: unknown): string {
  if (err instanceof ApiError && err.detail) return err.detail
  if (err instanceof Error) return err.message
  return 'Failed to load data.'
}

/**
 * Parse the ?lap=<N> URL param. Returns:
 *   - null for "all" (the all-laps view) or when param is "all"
 *   - a positive integer when param is a valid integer string
 *   - undefined when the param is missing or malformed (so callers can
 *     fall back to the best-GPS-lap default)
 */
function parseLapParam(raw: string | null): number | null | undefined {
  if (raw === null) return undefined
  if (raw === 'all') return null
  const n = Number(raw)
  if (!Number.isInteger(n) || n < 1) return undefined
  return n
}

export default function VerificationPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const { data: lapsData, error: lapsError } = useLaps()
  const { data: track, isLoading: trackLoading, error: trackError } = useTrack()
  const { data: allLaps, isLoading: allLapsLoading, error: allLapsError } = useAllLaps()

  // Best GPS quality lap — the default when no ?lap param is provided.
  const bestLap = lapsData?.laps
    ? [...lapsData.laps].sort((a, b) => a.gps_quality_score - b.gps_quality_score)[0]?.lap_number ?? 1
    : 1

  // Parse URL; fall back to best-GPS lap when missing/invalid.
  const parsed = parseLapParam(searchParams.get('lap'))
  const selectedLap: number | null = useMemo(() => {
    if (parsed === undefined) {
      // Missing/invalid → best lap (once loaded), else 1.
      return lapsData ? bestLap : 1
    }
    return parsed
  }, [parsed, lapsData, bestLap])

  const setSelectedLap = (lap: number | null) => {
    const next = new URLSearchParams(searchParams)
    next.set('lap', lap === null ? 'all' : String(lap))
    setSearchParams(next, { replace: true })
  }

  const {
    data: validation,
    isLoading: validationLoading,
    error: validationError,
  } = useValidation(selectedLap)

  // First-render race: lapsData hasn't resolved yet, so selectedLap is still
  // undefined and useValidation's key is undefined — SWR reports isLoading=false.
  // Treat that as "initializing" instead of falling through to the error branch.
  const initializingLaps = lapsData === undefined && !lapsError
  const singleLapInitializing =
    selectedLap !== null &&
    validation === undefined &&
    !validationError &&
    !validationLoading

  return (
    <div className="space-y-6">
      {/* Header + Lap Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Verification</h2>
        <div className="flex items-center gap-3">
          <label htmlFor="lap-select" className="text-sm text-gray-400">Lap:</label>
          <select
            id="lap-select"
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
          {initializingLaps ||
          singleLapInitializing ||
          trackLoading ||
          validationLoading ? (
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
            <p className="text-red-400">
              {errorMessage(trackError ?? validationError ?? lapsError)}
            </p>
          )}
        </>
      )}

      {/* All-laps view */}
      {selectedLap === null && (
        <>
          {initializingLaps || allLapsLoading ? (
            <LoadingSpinner message="Computing all-laps summary..." />
          ) : allLaps ? (
            <>
              <MetricCards metrics={allLaps.metrics} />
              <LapTable laps={allLaps.laps} selectedLap={null} />
            </>
          ) : (
            <p className="text-red-400">
              {errorMessage(allLapsError ?? lapsError)}
            </p>
          )}
        </>
      )}
    </div>
  )
}
