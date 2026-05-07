/**
 * Skeleton placeholder with a subtle shimmer pass.
 *
 * The animation is keyframed in `styles/tokens.css` (`@keyframes shimmer`).
 * We render a horizontal gradient sized 200% of the element width and animate
 * `background-position` so the highlight band moves across the box ~once every
 * 1.6s. Restrained — no pulse, no opacity flicker.
 */

interface SkeletonProps {
  /** Tailwind height utility, e.g. `'h-4'`, `'h-8'`, `'h-[180px]'`. */
  h?: string
  /** Tailwind width utility, defaults to full width. */
  w?: string
  /** Tailwind rounding utility. */
  rounded?: string
  className?: string
}

export function Skeleton({
  h = 'h-4',
  w = 'w-full',
  rounded = 'rounded',
  className = '',
}: SkeletonProps) {
  // 200% gradient driven by the `shimmer` keyframes. Two surface stops
  // separated by a softly-lit highlight track give the band-passing-through
  // effect without ever fully clearing the surface.
  const style = {
    backgroundImage:
      'linear-gradient(90deg, var(--surface-2) 0%, var(--surface-3) 50%, var(--surface-2) 100%)',
    backgroundSize: '200% 100%',
    animation: 'shimmer 1.6s ease-in-out infinite',
  }
  return (
    <div
      role="status"
      aria-hidden="true"
      style={style}
      className={`${h} ${w} ${rounded} ${className}`}
    />
  )
}

interface SkeletonTextProps {
  /** Number of lines to render. The last line is rendered at 70% width. */
  lines?: number
  className?: string
}

export function SkeletonText({ lines = 3, className = '' }: SkeletonTextProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} h="h-3" w={i === lines - 1 ? 'w-7/12' : 'w-full'} />
      ))}
    </div>
  )
}
