import { useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useVisualization } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import { Card, CardBody } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import Viewport from './Viewport'
import SidePanel from './SidePanel'
import Timeline from './Timeline'
import PlaybackControls from './PlaybackControls'

export type DataSource = 'sim' | 'real'

/**
 * Defensively parse the ?source URL param. Unknown / missing values
 * fall back to "sim".
 */
export function parseDataSource(raw: string | null): DataSource {
  return raw === 'real' ? 'real' : 'sim'
}

interface SourceSelectorProps {
  value: DataSource
  onChange: (next: DataSource) => void
}

/**
 * Two-button segment control for the data source. Active half is tinted
 * with the appropriate data-series color (sim=green, real=blue) so the
 * picker doubles as a legend hint.
 */
function SourceSelector({ value, onChange }: SourceSelectorProps) {
  const items: { id: DataSource; label: string; activeClass: string }[] = [
    {
      id: 'sim',
      label: 'Sim',
      activeClass: 'bg-[var(--ok-bg)] text-[var(--ok)]',
    },
    {
      id: 'real',
      label: 'Real',
      activeClass: 'bg-[var(--info-bg)] text-[var(--info)]',
    },
  ]
  return (
    <div className="inline-flex rounded-md border border-[var(--border-subtle)] bg-[var(--surface-2)] p-0.5">
      {items.map(({ id, label, activeClass }) => {
        const isActive = value === id
        return (
          <button
            key={id}
            type="button"
            onClick={() => onChange(id)}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              isActive
                ? activeClass
                : 'text-[var(--text-tertiary)] hover:text-[var(--text-primary)]'
            }`}
            aria-pressed={isActive}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

export default function VisualizationPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const dataSource = parseDataSource(searchParams.get('source'))
  const currentFrame = usePlaybackStore(s => s.currentFrame)
  const setFrame = usePlaybackStore(s => s.setFrame)
  const pause = usePlaybackStore(s => s.pause)
  const { data, isLoading, error, mutate } = useVisualization(dataSource)

  // When the user switches data source (via URL), reset playback state.
  // Previously this was a side effect of setDataSource inside playbackStore;
  // keeping it here makes the store pure (setters only set).
  useEffect(() => {
    pause()
    setFrame(0)
  }, [dataSource, pause, setFrame])

  const setDataSource = (s: DataSource) => {
    const next = new URLSearchParams(searchParams)
    next.set('source', s)
    setSearchParams(next, { replace: true })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3rem)]">
        <EmptyState
          title="Loading visualization frames..."
          description="Sim and replay traces are being assembled."
        />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3rem)]">
        <EmptyState
          icon="!"
          title="Failed to load visualization data"
          description="The backend rejected the request or returned an unexpected payload."
          action={
            <button
              type="button"
              onClick={() => mutate()}
              className="px-3 py-1.5 text-xs font-medium rounded bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 text-[var(--accent)] transition-colors"
            >
              Retry
            </button>
          }
        />
      </div>
    )
  }

  const frame = data.frames[currentFrame] ?? data.frames[0]

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)]">
      {/* Top bar: source selector */}
      <div className="shrink-0 flex items-center justify-between px-4 py-2 border-b border-[var(--border-subtle)] bg-[var(--surface-1)]">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">
          Replay
        </h2>
        <SourceSelector value={dataSource} onChange={setDataSource} />
      </div>

      {/* Main area: viewport + side panel */}
      <div className="flex flex-1 min-h-0">
        <div className="flex-1">
          <Viewport data={data} />
        </div>
        <SidePanel frame={frame} data={data} />
      </div>

      {/* Bottom strip — wrapped in raised Card to separate from viewport */}
      <div className="shrink-0 p-2 bg-[var(--surface-0)]">
        <Card tone="raised">
          <CardBody className="px-2 py-2 space-y-2">
            <Timeline data={data} />
            <PlaybackControls />
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
