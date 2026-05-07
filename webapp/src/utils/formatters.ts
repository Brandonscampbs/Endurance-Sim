export function formatSpeed(kmh: number): string {
  return `${kmh.toFixed(1)} km/h`
}

export function formatTime(seconds: number): string {
  const min = Math.floor(seconds / 60)
  const sec = seconds % 60
  return `${min}:${sec.toFixed(1).padStart(4, '0')}`
}

export function formatEnergy(kwh: number): string {
  return `${kwh.toFixed(3)} kWh`
}

export function formatPercent(pct: number): string {
  return `${pct.toFixed(1)}%`
}

export function formatForce(n: number): string {
  return `${Math.round(n)} N`
}

const signedFixed = (digits: number) =>
  new Intl.NumberFormat('en-US', {
    signDisplay: 'exceptZero',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })

const SIGNED_1 = signedFixed(1)
const SIGNED_2 = signedFixed(2)
const SIGNED_3 = signedFixed(3)

/** Pre-formatted signed delta with the given precision and an optional unit suffix. */
export function formatSignedDelta(value: number, digits: 1 | 2 | 3, unit = ''): string {
  const fmt = digits === 1 ? SIGNED_1 : digits === 2 ? SIGNED_2 : SIGNED_3
  // Treat -0 / +0 within the printed precision as plain 0 so the badge reads neutral.
  const factor = 10 ** digits
  const snapped = Math.abs(value * factor) < 0.5 ? 0 : value
  return `${fmt.format(snapped)}${unit}`
}

/** Signed delta as percentage with one decimal place. */
export function formatSignedPct(pct: number): string {
  return formatSignedDelta(pct, 1, '%')
}

/** Total seconds rendered as `H:MM:SS` or `M:SS` (no fractional part). */
export function formatDurationHMS(totalSeconds: number): string {
  const sign = totalSeconds < 0 ? '-' : ''
  const s = Math.round(Math.abs(totalSeconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const pad = (n: number) => n.toString().padStart(2, '0')
  return h > 0 ? `${sign}${h}:${pad(m)}:${pad(sec)}` : `${sign}${m}:${pad(sec)}`
}
