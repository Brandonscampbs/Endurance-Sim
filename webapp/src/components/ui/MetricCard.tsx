import { Card, CardBody } from './Card'
import { Badge, type BadgeTone } from './Badge'
import { Skeleton } from './Skeleton'

export type MetricTone = 'ok' | 'warn' | 'error' | 'neutral'

interface MetricDelta {
  /** Pre-formatted delta string, e.g. `'+1.2s'` or `'-0.4%'`. */
  value: string
  tone: MetricTone
  /** Optional context, e.g. `'vs baseline'`. */
  label?: string
}

interface MetricBadge {
  text: string
  tone: MetricTone
}

interface MetricCardProps {
  label: string
  value: string | number
  unit?: string
  delta?: MetricDelta
  hint?: string
  badge?: MetricBadge
  loading?: boolean
  className?: string
}

/** Map our MetricTone → Badge's BadgeTone (Badge supports an extra `info` tone). */
function badgeTone(tone: MetricTone): BadgeTone {
  return tone
}

/** CSS variable for the delta text color. Neutral falls back to muted text. */
function deltaColor(tone: MetricTone): string {
  switch (tone) {
    case 'ok':
      return 'var(--ok)'
    case 'warn':
      return 'var(--warn)'
    case 'error':
      return 'var(--error)'
    case 'neutral':
      return 'var(--text-tertiary)'
  }
}

/**
 * MetricCard — generalized from the Verification page's `MetricCards.tsx`.
 *
 * Visual hierarchy (top → bottom):
 *   label (xs uppercase tracking)   ←→   optional badge
 *   value (3xl, semibold, tabular-nums)  unit (sm, secondary)
 *   delta (xs, color-coded)              hint (xs, tertiary)
 */
export function MetricCard({
  label,
  value,
  unit,
  delta,
  hint,
  badge,
  loading = false,
  className = '',
}: MetricCardProps) {
  return (
    <Card className={className}>
      <CardBody className="space-y-2">
        <div className="flex items-center justify-between gap-2 min-h-[18px]">
          <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-tertiary)]">
            {label}
          </span>
          {badge && <Badge tone={badgeTone(badge.tone)}>{badge.text}</Badge>}
        </div>

        <div className="flex items-baseline gap-1.5">
          {loading ? (
            <Skeleton h="h-8" w="w-24" />
          ) : (
            <>
              <span className="text-3xl font-semibold leading-none tracking-tight tabular-nums text-[var(--text-primary)]">
                {value}
              </span>
              {unit && (
                <span className="text-sm font-medium text-[var(--text-secondary)]">
                  {unit}
                </span>
              )}
            </>
          )}
        </div>

        {(delta || hint) && (
          <div className="flex items-center gap-2 text-xs leading-snug">
            {delta && !loading && (
              <span
                style={{ color: deltaColor(delta.tone) }}
                className="font-medium tabular-nums"
              >
                {delta.value}
                {delta.label && (
                  <span className="ml-1 font-normal text-[var(--text-tertiary)]">
                    {delta.label}
                  </span>
                )}
              </span>
            )}
            {hint && !delta && (
              <span className="text-[var(--text-tertiary)]">{hint}</span>
            )}
            {hint && delta && (
              <>
                <span className="text-[var(--text-muted)]" aria-hidden="true">
                  ·
                </span>
                <span className="text-[var(--text-tertiary)]">{hint}</span>
              </>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  )
}
