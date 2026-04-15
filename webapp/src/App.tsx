export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex">
      <aside className="w-56 bg-gray-900 border-r border-gray-800 p-4">
        <h1 className="text-lg font-bold mb-6">FSAE Sim</h1>
        <nav className="space-y-2">
          <div className="text-sm text-gray-400">Validation</div>
          <div className="text-sm text-gray-400">Visualization</div>
        </nav>
      </aside>
      <main className="flex-1 p-6">
        <p className="text-gray-500">App shell working.</p>
      </main>
    </div>
  )
}
