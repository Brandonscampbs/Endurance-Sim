import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { rerunSimulation } from '../api/client'

const links = [
  { to: '/', label: 'Verification' },
  { to: '/visualization', label: 'Visualization' },
  { to: '/simulate', label: 'Simulate' },
]

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
    <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <h1 className="text-lg font-bold tracking-tight">FSAE Sim</h1>
        <p className="text-xs text-gray-500 mt-1">CT-16EV · Michigan 2025</p>
      </div>
      <nav aria-label="Main navigation" className="p-3 space-y-1">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `block px-3 py-2 rounded text-sm transition-colors ${
                isActive
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto p-3 border-t border-gray-800">
        <button
          onClick={handleRerun}
          disabled={rerunning}
          aria-busy={rerunning}
          className="w-full px-3 py-2 rounded text-sm font-medium transition-colors bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-wait text-white"
        >
          {rerunning ? 'Rerunning...' : 'Rerun Simulation'}
        </button>
      </div>
    </aside>
  )
}
