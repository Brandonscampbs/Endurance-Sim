import { useRef, useCallback, useEffect } from 'react'
import { usePlaybackStore } from '../../stores/playbackStore'
import type { VisualizationResponse } from '../../api/client'
import { formatTime } from '../../utils/formatters'

interface Props {
  data: VisualizationResponse
}

export default function Timeline({ data }: Props) {
  const { currentFrame, totalFrames, setFrame } = usePlaybackStore()
  const barRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)

  const progress = totalFrames > 0 ? currentFrame / (totalFrames - 1) : 0
  const currentTime = data.frames[currentFrame]?.time_s ?? 0

  // Compute frame index from a raw clientX against the bar's current bounds.
  const frameFromClientX = useCallback(
    (clientX: number): number | null => {
      if (!barRef.current) return null
      const rect = barRef.current.getBoundingClientRect()
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
      return Math.round(pct * Math.max(0, totalFrames - 1))
    },
    [totalFrames],
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.button !== 0) return
      const frame = frameFromClientX(e.clientX)
      if (frame !== null) setFrame(frame)
      draggingRef.current = true
    },
    [frameFromClientX, setFrame],
  )

  // Attach window-level listeners so the drag continues even when the mouse
  // leaves the bar. Removed on mouseup. This fixes "lost capture" on fast drags.
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current) return
      const frame = frameFromClientX(e.clientX)
      if (frame !== null) setFrame(frame)
    }
    const onUp = () => {
      draggingRef.current = false
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [frameFromClientX, setFrame])

  return (
    <div className="flex items-center gap-3 px-4">
      <span className="text-xs text-gray-400 font-mono w-12">{formatTime(currentTime)}</span>
      <div
        ref={barRef}
        className="flex-1 h-2 bg-gray-800 rounded cursor-pointer relative"
        onMouseDown={handleMouseDown}
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
