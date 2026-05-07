import type { ReactNode } from 'react'
import { Card, CardHeader, CardBody } from './Card'
import { Skeleton } from './Skeleton'
import { EmptyState } from './EmptyState'

interface ChartShellEmptyState {
  title: string
  description?: string
}

interface ChartShellProps {
  title: string
  subtitle?: string
  loading?: boolean
  empty?: boolean
  emptyState?: ChartShellEmptyState
  /** Inner chart height in pixels. Defaults to 320. */
  height?: number
  /** Right-aligned actions in the header (e.g. legend toggles, exports). */
  actions?: ReactNode
  children: ReactNode
  className?: string
}

/**
 * ChartShell — a Card-wrapped chart container with consistent header,
 * loading skeleton, and empty state. Pages drop their Plotly/SVG/canvas
 * children into a fixed-height inner box so layout doesn't reflow when
 * the chart finishes loading.
 *
 * Render priority: loading > empty > children. The fixed height is held
 * across all three states to prevent layout shift.
 */
export function ChartShell({
  title,
  subtitle,
  loading = false,
  empty = false,
  emptyState,
  height = 320,
  actions,
  children,
  className = '',
}: ChartShellProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <div className="min-w-0">
          <h3 className="text-sm font-semibold tracking-tight text-[var(--text-primary)] truncate">
            {title}
          </h3>
          {subtitle && (
            <p className="mt-0.5 text-xs text-[var(--text-tertiary)] truncate">
              {subtitle}
            </p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </CardHeader>
      <CardBody className="px-4 py-4">
        <div style={{ height }} className="relative w-full">
          {loading ? (
            <Skeleton h="h-full" w="w-full" rounded="rounded-md" />
          ) : empty ? (
            <div className="flex h-full items-center justify-center">
              <EmptyState
                title={emptyState?.title ?? 'No data to display'}
                description={emptyState?.description}
              />
            </div>
          ) : (
            children
          )}
        </div>
      </CardBody>
    </Card>
  )
}
