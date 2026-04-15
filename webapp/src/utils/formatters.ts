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
