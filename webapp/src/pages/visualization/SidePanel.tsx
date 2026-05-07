import type { ReactNode } from 'react'
import type { VizFrame, VisualizationResponse } from '../../api/client'
import { formatSpeed, formatForce, formatTime } from '../../utils/formatters'
import { SkeletonText } from '../../components/ui/Skeleton'

// ---------- helpers ----------------------------------------------------------

/**
 * Map grip utilization (0..1) onto a status token.
 *
 * < 0.6 → comfortable, > 0.8 → at the edge / over.
 */
function gripColor(util: number): string {
  if (util < 0.6) return 'var(--ok)'
  if (util < 0.8) return 'var(--warn)'
  return 'var(--error)'
}

/**
 * Spread-free min/max for large centerline arrays; spreading ~40k elements
 * into Math.min/max blows the JS argument-count limit.
 */
function reduceExtents(arr: number[]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]
    if (!Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!Number.isFinite(lo)) lo = 0
  if (!Number.isFinite(hi)) hi = 1
  return [lo, hi]
}

function radToDeg(rad: number): number {
  return (rad * 180) / Math.PI
}

// ---------- atomic UI bits ---------------------------------------------------

interface SectionProps {
  label: string
  children: ReactNode
}

/** Section wrapper — small uppercase header + grid body. */
function Section({ label, children }: SectionProps) {
  return (
    <section className="px-3 py-2.5 border-b border-[var(--border-subtle)]">
      <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)] mb-2">
        {label}
      </h4>
      <div className="space-y-1.5">{children}</div>
    </section>
  )
}

interface RowProps {
  label: string
  /** Right-aligned value (string, number, or any node). */
  children: ReactNode
}

/** Two-column metric row: label left, value right. */
function Row({ label, children }: RowProps) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-xs text-[var(--text-secondary)]">{label}</span>
      <span className="text-sm tabular-nums text-[var(--text-primary)] text-right">
        {children}
      </span>
    </div>
  )
}

interface MeterProps {
  /** 0..100 (clamped). */
  pct: number
  color: string
}

/** Thin token-driven progress bar; used for throttle/brake/SOC. */
function Meter({ pct, color }: MeterProps) {
  const clamped = Math.max(0, Math.min(100, pct))
  return (
    <div className="h-1.5 bg-[var(--surface-3)] rounded overflow-hidden">
      <div
        className="h-full rounded transition-all duration-75"
        style={{ width: `${clamped}%`, backgroundColor: color }}
      />
    </div>
  )
}

// ---------- minimap ----------------------------------------------------------

function TrackMinimap({ frame, data }: { frame: VizFrame; data: VisualizationResponse }) {
  const [minX, maxX] = reduceExtents(data.track_centerline_x)
  const [minY, maxY] = reduceExtents(data.track_centerline_y)
  const range = Math.max(maxX - minX, maxY - minY) || 1

  // Convention: world X -> minimap x, world Y -> minimap y (no flip).
  // Matches the 3D TrackLine (which maps track Y -> world Z). SVG y grows
  // downward by default but we're content with that — the minimap shows the
  // same orientation as the 3D view with the camera looking toward +Y world-up.
  const points = data.track_centerline_x
    .map((x, i) => {
      const nx = ((x - minX) / range) * 100
      const ny = ((data.track_centerline_y[i] - minY) / range) * 100
      return `${nx},${ny}`
    })
    .join(' ')

  const cx = ((frame.x - minX) / range) * 100
  const cy = ((frame.y - minY) / range) * 100

  return (
    <svg
      viewBox="-10 -10 120 120"
      className="w-full bg-[var(--surface-3)] rounded"
    >
      <polyline
        points={points}
        fill="none"
        stroke="var(--text-muted)"
        strokeWidth="1.5"
      />
      <circle cx={cx} cy={cy} r="3" fill="var(--accent)" />
    </svg>
  )
}

// ---------- main component ---------------------------------------------------

interface Props {
  frame: VizFrame
  data: VisualizationResponse
}

const wheelLabels = ['FL', 'FR', 'RL', 'RR']

/**
 * Right-hand telemetry rail. Sections track the spec layout:
 *   Position · Driver inputs · Powertrain · Battery · Attitude · Wheel forces
 * Plus a track minimap pinned at the bottom that's been there since v1.
 */
export default function SidePanel({ frame, data }: Props) {
  return (
    <aside className="w-80 bg-[var(--surface-1)] border-l border-[var(--border-subtle)] overflow-y-auto">
      {/* Speed banner — biggest single piece of telemetry, kept at the top */}
      <div className="px-3 pt-3 pb-2">
        <div className="text-2xl font-semibold tabular-nums text-[var(--accent)]">
          {formatSpeed(frame.speed_kmh)}
        </div>
        <div className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">
          Current speed
        </div>
      </div>

      {/* Position */}
      <Section label="Position">
        <Row label="Time">{formatTime(frame.time_s)}</Row>
        <Row label="Distance">{frame.distance_m.toFixed(1)} m</Row>
        <Row label="X">{frame.x.toFixed(1)} m</Row>
        <Row label="Y">{frame.y.toFixed(1)} m</Row>
        <Row label="Heading">{radToDeg(frame.heading_rad).toFixed(1)}°</Row>
      </Section>

      {/* Driver inputs */}
      <Section label="Driver inputs">
        <div className="space-y-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-xs text-[var(--text-secondary)]">Throttle</span>
            <span className="text-sm tabular-nums text-[var(--text-primary)]">
              {frame.throttle_pct.toFixed(0)}%
            </span>
          </div>
          <Meter pct={frame.throttle_pct} color="var(--ok)" />
        </div>
        <div className="space-y-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-xs text-[var(--text-secondary)]">Brake</span>
            <span className="text-sm tabular-nums text-[var(--text-primary)]">
              {frame.brake_pct.toFixed(0)}%
            </span>
          </div>
          <Meter pct={frame.brake_pct} color="var(--error)" />
        </div>
        {frame.action && (
          <Row label="Action">
            <span className="text-[var(--text-secondary)] uppercase text-[10px] tracking-wide">
              {frame.action}
            </span>
          </Row>
        )}
      </Section>

      {/* Powertrain */}
      <Section label="Powertrain">
        <Row label="Motor RPM">{Math.round(frame.motor_rpm)}</Row>
        <Row label="Torque">{frame.motor_torque_nm.toFixed(1)} Nm</Row>
        <Row label="Mech power">
          {((frame.motor_torque_nm * frame.motor_rpm * 2 * Math.PI) / 60 / 1000).toFixed(1)}{' '}
          kW
        </Row>
      </Section>

      {/* Battery */}
      <Section label="Battery">
        <Row label="Voltage">{frame.pack_voltage_v.toFixed(1)} V</Row>
        <Row label="Current">{frame.pack_current_a.toFixed(1)} A</Row>
        <Row label="Pack power">
          {((frame.pack_voltage_v * frame.pack_current_a) / 1000).toFixed(2)} kW
        </Row>
        <div className="space-y-1 pt-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-xs text-[var(--text-secondary)]">Model SOC</span>
            <span className="text-sm tabular-nums text-[var(--text-primary)]">
              {frame.soc_pct.toFixed(1)}%
            </span>
          </div>
          <Meter pct={frame.soc_pct} color="var(--info)" />
        </div>
      </Section>

      {/* Attitude */}
      <Section label="Attitude">
        <Row label="Roll">{radToDeg(frame.roll_rad).toFixed(2)}°</Row>
        <Row label="Pitch">{radToDeg(frame.pitch_rad).toFixed(2)}°</Row>
      </Section>

      {/* Wheel forces */}
      <Section label="Wheel forces">
        <div className="grid grid-cols-2 gap-2">
          {frame.wheels.map((w, i) => {
            const total = Math.sqrt(w.fx ** 2 + w.fy ** 2)
            const utilPct = Math.min(100, w.grip_util * 100)
            const color = gripColor(w.grip_util)
            return (
              <div
                key={i}
                className="rounded p-2 bg-[var(--surface-2)] border border-[var(--border-subtle)]"
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-[10px] uppercase tracking-wide text-[var(--text-tertiary)]">
                    {wheelLabels[i]}
                  </span>
                  <span className="text-[11px] tabular-nums text-[var(--text-primary)]">
                    {formatForce(total)}
                  </span>
                </div>
                <div className="mt-1.5 h-1.5 bg-[var(--surface-3)] rounded overflow-hidden">
                  <div
                    className="h-full rounded transition-all duration-75"
                    style={{ width: `${utilPct}%`, backgroundColor: color }}
                  />
                </div>
                <div className="mt-1 text-[10px] tabular-nums" style={{ color }}>
                  {utilPct.toFixed(0)}% grip
                </div>
              </div>
            )
          })}
        </div>
      </Section>

      {/* Minimap pinned at the bottom */}
      <div className="p-3 mt-auto">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)] mb-2">
          Track position
        </h4>
        <TrackMinimap frame={frame} data={data} />
      </div>
    </aside>
  )
}

/**
 * Loading variant — exported so callers (currently only VisualizationPage)
 * can preserve the panel's width while frames are still being computed.
 *
 * Not consumed yet; kept here so importers don't need to know skeleton
 * markup details. Tree-shakes if unused.
 */
export function SidePanelSkeleton() {
  return (
    <aside className="w-80 bg-[var(--surface-1)] border-l border-[var(--border-subtle)] p-4">
      <SkeletonText lines={20} />
    </aside>
  )
}
