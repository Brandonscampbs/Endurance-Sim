import { useVisualization } from '../../api/client'
import { usePlaybackStore } from '../../stores/playbackStore'
import LoadingSpinner from '../../components/LoadingSpinner'
import Viewport from './Viewport'
import SidePanel from './SidePanel'
import Timeline from './Timeline'
import PlaybackControls from './PlaybackControls'

export default function VisualizationPage() {
  const dataSource = usePlaybackStore(s => s.dataSource)
  const currentFrame = usePlaybackStore(s => s.currentFrame)
  const { data, isLoading, error } = useVisualization(dataSource)

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
