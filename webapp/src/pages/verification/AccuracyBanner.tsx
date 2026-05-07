import { useEffect, useState } from 'react'
import { useAllLaps } from '../../api/client'
import type { ValidationMetric } from '../../api/client'

const STORAGE_KEY = 'verification-banner-collapsed'

// Metrics we look for in the aggregate response. The first match wins so we
// gracefully handle the backend renaming "Charge used (net)" → "Net charge"
// without breaking the banner.
const TIME_METRIC_NAMES = ['Driving time', 'Lap time'] as const
const CHARGE_METRIC_NAMES = ['Charge used (net)', 'Net charge', 'Charge used'] as const
const ENERGY_METRIC_NAMES = [
  'Energy consumed (net)',
  'Net energy',
  'Energy consumed',
] as const
const SOC_METRIC_NAMES = ['Final SOC', 'Final state of charge'] as const

type MetricKind = 'time' | 'charge' | 'energy' | 'soc'

interface DisplayMetric {
  kind: MetricKind
  shortLabel: string
  raw: ValidationMetric
  /** Signed delta as percent: (sim − real) / real * 100. */
  signedDeltaPct: number
}

function findMetric(
  metrics: ValidationMetric[],
  candidates: readonly string[],
): ValidationMetric | undefined {
  for (const name of candidates) {
    const hit = metrics.find((m) => m.name === name)
    if (hit) return hit
  }
  return undefined
}

function signedDelta(m: ValidationMetric): number {
  if (Math.abs(m.real_value) < 1e-9) return 0
  return ((m.sim_value - m.real_value) / m.real_value) * 100
}

function buildDisplayMetrics(metrics: ValidationMetric[]): DisplayMetric[] {
  const out: DisplayMetric[] = []
  const time = findMetric(metrics, TIME_METRIC_NAMES)
  if (time) out.push({ kind: 'time', shortLabel: 'Lap time', raw: time, signedDeltaPct: signedDelta(time) })
  const charge = findMetric(metrics, CHARGE_METRIC_NAMES)
  if (charge) out.push({ kind: 'charge', shortLabel: 'Net charge', raw: charge, signedDeltaPct: signedDelta(charge) })
  const energy = findMetric(metrics, ENERGY_METRIC_NAMES)
  if (energy) out.push({ kind: 'energy', shortLabel: 'Net energy', raw: energy, signedDeltaPct: signedDelta(energy) })
  const soc = findMetric(metrics, SOC_METRIC_NAMES)
  if (soc) out.push({ kind: 'soc', shortLabel: 'Final SOC', raw: soc, signedDeltaPct: signedDelta(soc) })
  return out
}

// Tier thresholds per metric kind. Lap time deviates faster (5/10), the energy
// counters are noisier so we use 10/15 to match backend pass thresholds.
function dotColor(kind: MetricKind, absPct: number): string {
  const okWarn: Record<MetricKind, [number, number]> = {
    time: [5, 10],
    charge: [10, 15],
    energy: [10, 15],
    soc: [10, 15],
  }
  const [ok, warn] = okWarn[kind]
  if (absPct < ok) return 'bg-emerald-500'
  if (absPct < warn) return 'bg-amber-500'
  return 'bg-red-500'
}

function describeDelta(kind: MetricKind, signedPct: number): string {
  const abs = Math.abs(signedPct).toFixed(1)
  if (kind === 'time') {
    if (signedPct > 0) return `sim ${abs}% slower than telemetry`
    if (signedPct < 0) return `sim ${abs}% faster than telemetry`
    return 'sim matches telemetry'
  }
  if (signedPct > 0) return `sim ${abs}% higher than telemetry`
  if (signedPct < 0) return `sim ${abs}% lower than telemetry`
  return 'sim matches telemetry'
}

function readCollapsed(): boolean {
  if (typeof window === 'undefined') return false
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

function writeCollapsed(value: boolean): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, value ? 'true' : 'false')
  } catch {
    // Storage may be unavailable (private mode, quota); collapse state is
    // ephemeral in that case — not worth surfacing to the user.
  }
}

const DISCLAIMER =
  'Suitable for replay analysis and rough sensitivity. Not a predictive tune optimizer — the sim and the track were both fit to this telemetry.'

export default function AccuracyBanner() {
  const { data, error, isLoading } = useAllLaps()
  const [collapsed, setCollapsed] = useState<boolean>(false)

  useEffect(() => {
    setCollapsed(readCollapsed())
  }, [])

  const toggle = () => {
    setCollapsed((prev) => {
      const next = !prev
      writeCollapsed(next)
      return next
    })
  }

  if (isLoading) {
    return (
      <div
        aria-hidden="true"
        className="bg-gray-900 border border-gray-800 rounded-lg h-20 animate-pulse"
      />
    )
  }

  if (error || !data) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
        <p className="text-xs text-gray-500 italic">
          Couldn{'’'}t load accuracy summary.
        </p>
      </div>
    )
  }

  const display = buildDisplayMetrics(data.metrics)
  const worst =
    display.length > 0
      ? display.reduce((acc, m) =>
          Math.abs(m.signedDeltaPct) > Math.abs(acc.signedDeltaPct) ? m : acc,
        )
      : null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-gray-100">
            Sim vs Michigan 2025 telemetry
          </h3>
          {collapsed && worst && (
            <p className="mt-1 text-xs text-gray-400">
              <span
                aria-hidden="true"
                className={`inline-block h-2 w-2 rounded-full mr-2 align-middle ${dotColor(worst.kind, Math.abs(worst.signedDeltaPct))}`}
              />
              <span className="text-gray-300">{worst.shortLabel}:</span>{' '}
              <span className="tabular-nums">{describeDelta(worst.kind, worst.signedDeltaPct)}</span>
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={toggle}
          aria-expanded={!collapsed}
          className="text-xs text-gray-400 hover:text-gray-200 shrink-0"
        >
          {collapsed ? '▸ Show details' : '▾ Collapse'}
        </button>
      </div>

      {!collapsed && (
        <>
          {display.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-x-6 gap-y-2">
              {display.map((m) => (
                <div key={m.shortLabel} className="flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className={`inline-block h-2 w-2 rounded-full ${dotColor(m.kind, Math.abs(m.signedDeltaPct))}`}
                  />
                  <span className="text-xs text-gray-500">{m.shortLabel}:</span>
                  <span className="text-sm text-gray-200 tabular-nums">
                    {describeDelta(m.kind, m.signedDeltaPct)}
                  </span>
                </div>
              ))}
            </div>
          )}
          <p className="mt-3 text-xs text-gray-500 italic">{DISCLAIMER}</p>
        </>
      )}
    </div>
  )
}
