import type { ReactNode } from 'react'

export type BadgeTone = 'ok' | 'warn' | 'error' | 'info' | 'neutral'

interface BadgeProps {
  tone: BadgeTone
  children: ReactNode
  /** Render a leading colored dot. */
  dot?: boolean
  className?: string
}

/** Token-driven background + foreground for each tone. */
function toneStyles(tone: BadgeTone): { background: string; color: string } {
  switch (tone) {
    case 'ok':
      return { background: 'var(--ok-bg)', color: 'var(--ok)' }
    case 'warn':
      return { background: 'var(--warn-bg)', color: 'var(--warn)' }
    case 'error':
      return { background: 'var(--error-bg)', color: 'var(--error)' }
    case 'info':
      return { background: 'var(--info-bg)', color: 'var(--info)' }
    case 'neutral':
      return { background: 'var(--surface-3)', color: 'var(--text-secondary)' }
  }
}

export function Badge({ tone, children, dot = false, className = '' }: BadgeProps) {
  const { background, color } = toneStyles(tone)
  return (
    <span
      style={{ backgroundColor: background, color }}
      className={
        `inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium ` +
        `rounded-full uppercase tracking-wide ${className}`
      }
    >
      {dot && (
        <span
          aria-hidden="true"
          style={{ backgroundColor: color }}
          className="inline-block h-1.5 w-1.5 rounded-full"
        />
      )}
      {children}
    </span>
  )
}
