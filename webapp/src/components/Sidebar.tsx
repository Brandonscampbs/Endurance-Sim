import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import useSWR from 'swr'
import { rerunSimulation, API_BASE } from '../api/client'
import { Badge } from './ui/Badge'

const links = [
  { to: '/', label: 'Verification' },
  { to: '/visualization', label: 'Visualization' },
  { to: '/simulate', label: 'Simulate' },
]

interface HealthResponse {
  status: string
}

/**
 * Backend health pill. Polls `/api/health` every 30s. Green dot when the
 * backend is reachable and returns ok, red dot otherwise. We deliberately
 * disable the global `fetcher` toast for this URL by handling errors via
 * SWR's `error` state — a routine 30s health poll shouldn't spam toasts.
 */
function HealthIndicator() {
  // SWR catches the throw from `fetcher` and exposes it as `error`. The
  // `fetcher` itself emits a toast on error which we don't want here, so
  // we use a thin local fetch instead and convert non-OK to an error.
  const { data, error } = useSWR<HealthResponse>(
    `${API_BASE}/health`,
    async (url: string) => {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`status ${res.status}`)
      return res.json() as Promise<HealthResponse>
    },
    {
      refreshInterval: 30000,
      // Don't blow up on transient errors; just reflect them.
      shouldRetryOnError: false,
      revalidateOnFocus: false,
    },
  )

  const isUp = !error && data?.status === 'ok'
  return (
    <Badge tone={isUp ? 'ok' : 'error'} dot>
      {isUp ? 'Online' : 'Offline'}
    </Badge>
  )
}

export default function Sidebar() {
  const [rerunning, setRerunning] = useState(false)

  async function handleRerun() {
    if (rerunning) return
    setRerunning(true)
    try {
      await rerunSimulation()
    } catch {
      // `rerunSimulation` already surfaces a toast; swallow here so the
      // unhandled rejection doesn't bubble while still re-enabling the
      // button in `finally`.
    } finally {
      setRerunning(false)
    }
  }

  return (
    <aside className="w-56 shrink-0 bg-[var(--surface-1)] border-r border-[var(--border-subtle)] flex flex-col">
      <div className="p-4 border-b border-[var(--border-subtle)]">
        <div className="flex items-center justify-between gap-2">
          <h1 className="text-lg font-semibold tracking-tight text-[var(--text-primary)]">
            FSAE Sim
          </h1>
          <HealthIndicator />
        </div>
        <p className="text-xs text-[var(--text-tertiary)] mt-1">
          CT-16EV - Michigan 2025
        </p>
      </div>
      <nav aria-label="Main navigation" className="p-3 space-y-1">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `block px-3 py-2 rounded text-sm transition-colors ${
                isActive
                  ? 'bg-[var(--accent)]/15 text-[var(--accent)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-2)]'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto p-3 border-t border-[var(--border-subtle)]">
        <button
          onClick={handleRerun}
          disabled={rerunning}
          aria-busy={rerunning}
          className="w-full px-3 py-2 rounded text-sm font-medium transition-colors bg-[var(--accent)] hover:bg-[var(--accent-strong)] disabled:opacity-50 disabled:cursor-wait text-white"
        >
          {rerunning ? 'Rerunning...' : 'Rerun Simulation'}
        </button>
      </div>
    </aside>
  )
}
