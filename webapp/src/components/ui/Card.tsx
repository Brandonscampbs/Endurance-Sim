import type { ReactNode } from 'react'

/**
 * Card primitive set. Composes as:
 *
 *   <Card>
 *     <CardHeader>
 *       <CardTitle>Per-lap energy</CardTitle>
 *       <CardActions>...</CardActions>
 *     </CardHeader>
 *     <CardBody>...</CardBody>
 *   </Card>
 *
 * Surfaces and borders are driven by tokens so the entire app re-themes via
 * `tokens.css` without touching component markup.
 */

interface CardProps {
  children: ReactNode
  className?: string
  /** `'raised'` swaps the surface to `--surface-2`, e.g. for popovers. */
  tone?: 'default' | 'raised'
}

export function Card({ children, className = '', tone = 'default' }: CardProps) {
  const surface = tone === 'raised' ? 'bg-[var(--surface-2)]' : 'bg-[var(--surface-1)]'
  return (
    <section
      className={
        `${surface} border border-[var(--border-subtle)] rounded-lg overflow-hidden ` +
        `text-[var(--text-primary)] ${className}`
      }
    >
      {children}
    </section>
  )
}

interface CardChildProps {
  children: ReactNode
  className?: string
}

export function CardHeader({ children, className = '' }: CardChildProps) {
  return (
    <header
      className={
        `flex items-center justify-between gap-3 px-5 py-3 ` +
        `border-b border-[var(--border-subtle)] ${className}`
      }
    >
      {children}
    </header>
  )
}

export function CardTitle({ children, className = '' }: CardChildProps) {
  return (
    <h3
      className={
        `text-sm font-semibold tracking-tight text-[var(--text-primary)] ${className}`
      }
    >
      {children}
    </h3>
  )
}

export function CardBody({ children, className = '' }: CardChildProps) {
  return <div className={`px-5 py-4 ${className}`}>{children}</div>
}

export function CardActions({ children, className = '' }: CardChildProps) {
  return (
    <div
      className={
        `flex items-center justify-end gap-2 px-5 py-3 ` +
        `border-t border-[var(--border-subtle)] ${className}`
      }
    >
      {children}
    </div>
  )
}
