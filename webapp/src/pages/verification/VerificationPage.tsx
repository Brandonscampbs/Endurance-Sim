import { lazy, Suspense, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  useValidation,
  useAllLaps,
  useTrack,
  useLaps,
  ApiError,
} from '../../api/client'
import { EmptyState, Skeleton } from '../../components/ui'
import AccuracyBanner from './AccuracyBanner'
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

/** Skeleton stand-ins reserved while data loads — preserves page height. */
function MetricCardsSkeleton() {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} h="h-[112px]" w="w-full" rounded="rounded-lg" />
      ))}
    </div>
  )
}

function ChartPanelSkeleton() {
  return <Skeleton h="h-[290px]" w="w-full" rounded="rounded-lg" />
}

function TrackMapsSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-4">
      <Skeleton h="h-[440px]" w="w-full" rounded="rounded-lg" />
      <Skeleton h="h-[440px]" w="w-full" rounded="rounded-lg" />
    </div>
  )
}

function OverlayChartsSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <ChartPanelSkeleton key={i} />
      ))}
    </div>
  )
}

function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      <Skeleton h="h-10" w="w-full" rounded="rounded-lg" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} h="h-9" w="w-full" rounded="rounded-md" />
      ))}
    </div>
  )
}

/** Skeleton arrangement for the single-lap branch — mirrors the real layout. */
function SingleLapSkeleton() {
  return (
    <>
      <TrackMapsSkeleton />
      <OverlayChartsSkeleton />
      <TableSkeleton rows={6} />
      <MetricCardsSkeleton />
    </>
  )
}

/** Skeleton arrangement for the all-laps branch. */
function AllLapsSkeleton() {
  return (
    <>
      <MetricCardsSkeleton />
      <TableSkeleton rows={10} />
    </>
  )
}

/** Designed error fallback with Retry. Uses window.location.reload() — every
 *  affected SWR key is involved, so a full reload is the simplest correct retry. */
function VerificationError({ message }: { message: string }) {
  return (
    <EmptyState
      title="Couldn't load verification data"
      description={message}
      action={
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="rounded-md border border-[var(--border-strong)] bg-[var(--surface-3)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] hover:bg-[var(--surface-2)]"
        >
          Retry
        </button>
      }
    />
  )
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
    <div className="space-y-5">
      <AccuracyBanner />

      {/* Header + Lap Selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-[var(--text-primary)]">Verification</h2>
        <div className="flex items-center gap-3">
          <label htmlFor="lap-select" className="text-sm text-[var(--text-tertiary)]">Lap:</label>
          {lapsData ? (
            <select
              id="lap-select"
              value={selectedLap ?? 'all'}
              onChange={(e) => setSelectedLap(e.target.value === 'all' ? null : Number(e.target.value))}
              className="bg-[var(--surface-3)] border border-[var(--border-subtle)] hover:border-[var(--border-strong)] focus:border-[var(--border-strong)] focus:outline-none rounded px-3 py-1.5 text-sm text-[var(--text-primary)]"
            >
              <option value="all">All Laps</option>
              {lapsData.laps.map((l) => (
                <option key={l.lap_number} value={l.lap_number}>
                  Lap {l.lap_number} — {l.time_s.toFixed(1)}s (GPS: {l.gps_quality_score})
                </option>
              ))}
            </select>
          ) : (
            <Skeleton h="h-9" w="w-64" rounded="rounded" />
          )}
        </div>
      </div>

      {/* Single-lap view */}
      {selectedLap !== null && (
        <>
          {initializingLaps ||
          singleLapInitializing ||
          trackLoading ||
          validationLoading ? (
            <SingleLapSkeleton />
          ) : track && validation ? (
            <>
              <Suspense fallback={<TrackMapsSkeleton />}>
                <TrackMaps track={track} validation={validation} />
              </Suspense>
              <Suspense fallback={<OverlayChartsSkeleton />}>
                <OverlayCharts validation={validation} />
              </Suspense>
              <SectorTable sectors={validation.sectors} />
              <MetricCards metrics={validation.metrics} />
            </>
          ) : (
            <VerificationError
              message={errorMessage(trackError ?? validationError ?? lapsError)}
            />
          )}
        </>
      )}

      {/* All-laps view */}
      {selectedLap === null && (
        <>
          {initializingLaps || allLapsLoading ? (
            <AllLapsSkeleton />
          ) : allLaps ? (
            <>
              <MetricCards metrics={allLaps.metrics} />
              <LapTable laps={allLaps.laps} selectedLap={null} />
            </>
          ) : (
            <VerificationError message={errorMessage(allLapsError ?? lapsError)} />
          )}
        </>
      )}
    </div>
  )
}
