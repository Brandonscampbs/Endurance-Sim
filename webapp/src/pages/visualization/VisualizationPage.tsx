import { useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useVisualization } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import LoadingSpinner from '../../components/LoadingSpinner'
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

export default function VisualizationPage() {
  const [searchParams] = useSearchParams()
  const dataSource = parseDataSource(searchParams.get('source'))
  const currentFrame = usePlaybackStore(s => s.currentFrame)
  const setFrame = usePlaybackStore(s => s.setFrame)
  const pause = usePlaybackStore(s => s.pause)
  const { data, isLoading, error } = useVisualization(dataSource)

  // When the user switches data source (via URL), reset playback state.
  // Previously this was a side effect of setDataSource inside playbackStore;
  // keeping it here makes the store pure (setters only set).
  useEffect(() => {
    pause()
    setFrame(0)
  }, [dataSource, pause, setFrame])

  if (isLoading) return <LoadingSpinner message="Computing visualization data..." />
  if (error || !data) return <p className="text-red-400">Failed to load visualization data.</p>

  const frame = data.frames[currentFrame] ?? data.frames[0]

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)]">
      {/* Main area: viewport + side panel */}
      <div className="flex flex-1 min-h-0">
        <div className="flex-1">
          <Viewport data={data} />
        </div>
        <SidePanel frame={frame} data={data} />
      </div>

      {/* Bottom strip */}
      <div className="shrink-0 bg-gray-900 border-t border-gray-800 py-2 space-y-2">
        <Timeline data={data} />
        <PlaybackControls />
      </div>
    </div>
  )
}
