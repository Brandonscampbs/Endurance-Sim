import type { ReactNode } from 'react'

interface EmptyStateProps {
  /** Defaults to a faded unicode glyph if not provided. NO emoji per project rule. */
  icon?: ReactNode
  title: string
  description?: string
  /** Optional CTA — typically a button. Rendered below the description. */
  action?: ReactNode
  className?: string
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className = '',
}: EmptyStateProps) {
  return (
    <div
      className={
        `flex flex-col items-center justify-center text-center px-6 py-12 ${className}`
      }
    >
      <div
        aria-hidden="true"
        className="text-3xl text-[var(--text-muted)] mb-3 leading-none select-none"
      >
        {icon ?? '◇'}
      </div>
      <div className="text-sm font-medium text-[var(--text-secondary)]">{title}</div>
      {description && (
        <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-[var(--text-tertiary)]">
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
