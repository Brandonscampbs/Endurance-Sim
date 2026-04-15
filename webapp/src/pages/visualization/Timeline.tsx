import { useRef, useCallback } from 'react'
import { usePlaybackStore } from '../../stores/playbackStore'
import type { VisualizationResponse } from '../../api/client'
import { formatTime } from '../../utils/formatters'

interface Props {
  data: VisualizationResponse
}

export default function Timeline({ data }: Props) {
  const { currentFrame, totalFrames, setFrame } = usePlaybackStore()
  const barRef = useRef<HTMLDivElement>(null)

  const progress = totalFrames > 0 ? currentFrame / (totalFrames - 1) : 0
  const currentTime = data.frames[currentFrame]?.time_s ?? 0

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!barRef.current) return
      const rect = barRef.current.getBoundingClientRect()
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      setFrame(Math.round(pct * (totalFrames - 1)))
    },
    [totalFrames, setFrame],
  )

  const handleDrag = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.buttons !== 1 || !barRef.current) return
      const rect = barRef.current.getBoundingClientRect()
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      setFrame(Math.round(pct * (totalFrames - 1)))
    },
    [totalFrames, setFrame],
  )

  return (
    <div className="flex items-center gap-3 px-4">
      <span className="text-xs text-gray-400 font-mono w-12">{formatTime(currentTime)}</span>
      <div
        ref={barRef}
        className="flex-1 h-2 bg-gray-800 rounded cursor-pointer relative"
        onClick={handleClick}
        onMouseMove={handleDrag}
      >
        <div
          className="h-full bg-green-500 rounded transition-none"
          style={{ width: `${progress * 100}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow"
          style={{ left: `${progress * 100}%`, transform: 'translate(-50%, -50%)' }}
        />
      </div>
      <span className="text-xs text-gray-400 font-mono w-12">{formatTime(data.total_time_s)}</span>
    </div>
  )
}
