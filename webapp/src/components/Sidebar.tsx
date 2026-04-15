import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Validation' },
  { to: '/visualization', label: 'Visualization' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <h1 className="text-lg font-bold tracking-tight">FSAE Sim</h1>
        <p className="text-xs text-gray-500 mt-1">CT-16EV · Michigan 2025</p>
      </div>
      <nav className="p-3 space-y-1">
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
    </aside>
  )
}
