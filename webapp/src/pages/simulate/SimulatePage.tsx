import { useSimulateStore, BASELINE_PARAMS } from '../../stores/simulateStore'

export default function SimulatePage() {
  const { params, setMaxRpm, setMaxTorque, setSocPoint, resetToBaseline } = useSimulateStore()

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold">Simulate</h2>
        <p className="text-sm text-gray-400 mt-1">
          Change three tune parameters and re-run the Michigan 2025 endurance. Everything else
          (driver model, track, mass, aero, tires) stays fixed at the baseline.
        </p>
      </div>

      <div className="bg-amber-900/20 border border-amber-700/40 rounded px-4 py-3 text-sm text-amber-200">
        <strong>Stub page.</strong> The three-knob form is wired; the backend run endpoint is not
        yet implemented.
      </div>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-5">
        <h3 className="text-lg font-semibold">Tune parameters</h3>

        <div className="grid grid-cols-2 gap-5">
          <label className="flex flex-col gap-2">
            <span className="text-sm text-gray-400">
              Max motor RPM
              <span className="text-gray-600 ml-2">(baseline {BASELINE_PARAMS.max_rpm})</span>
            </span>
            <input
              type="number"
              value={params.max_rpm}
              onChange={(e) => setMaxRpm(Number(e.target.value))}
              min={1000}
              max={6500}
              step={50}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
            />
          </label>

          <label className="flex flex-col gap-2">
            <span className="text-sm text-gray-400">
              Max motor torque (Nm)
              <span className="text-gray-600 ml-2">(baseline {BASELINE_PARAMS.max_torque_nm})</span>
            </span>
            <input
              type="number"
              value={params.max_torque_nm}
              onChange={(e) => setMaxTorque(Number(e.target.value))}
              min={20}
              max={230}
              step={1}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
            />
          </label>
        </div>

        <div>
          <div className="text-sm text-gray-400 mb-2">SOC discharge map (pack current vs SOC)</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800">
                <th className="text-left py-2">SOC (%)</th>
                <th className="text-left py-2">Max pack current (A)</th>
              </tr>
            </thead>
            <tbody>
              {params.soc_discharge_map.map((pt, idx) => (
                <tr key={idx} className="border-b border-gray-800/50">
                  <td className="py-1.5">
                    <input
                      type="number"
                      value={pt.soc_pct}
                      onChange={(e) =>
                        setSocPoint(idx, { ...pt, soc_pct: Number(e.target.value) })
                      }
                      min={0}
                      max={100}
                      step={1}
                      className="w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1"
                    />
                  </td>
                  <td className="py-1.5">
                    <input
                      type="number"
                      value={pt.max_current_a}
                      onChange={(e) =>
                        setSocPoint(idx, { ...pt, max_current_a: Number(e.target.value) })
                      }
                      min={0}
                      max={200}
                      step={1}
                      className="w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex gap-3 pt-2">
          <button
            disabled
            title="Backend endpoint not yet implemented"
            className="px-4 py-2 rounded text-sm font-medium bg-emerald-700/40 text-emerald-200/60 cursor-not-allowed"
          >
            Run simulation
          </button>
          <button
            onClick={resetToBaseline}
            className="px-4 py-2 rounded text-sm font-medium bg-gray-800 hover:bg-gray-700 text-gray-200"
          >
            Reset to baseline
          </button>
        </div>
      </section>

      <section className="grid grid-cols-4 gap-4">
        {['Total time', 'Total energy', 'Min SOC', 'Completed 22 laps'].map((label) => (
          <div key={label} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="text-xs text-gray-500">{label}</div>
            <div className="text-2xl font-semibold text-gray-600 mt-1">—</div>
            <div className="text-xs text-gray-600 mt-1">vs baseline: —</div>
          </div>
        ))}
      </section>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-5">
        <h3 className="text-lg font-semibold mb-3">Per-lap results</h3>
        <div className="text-sm text-gray-500">Run a simulation to populate this table.</div>
      </section>

      <section className="bg-gray-900 border border-gray-800 rounded-lg p-5">
        <h3 className="text-lg font-semibold mb-3">Time series vs baseline</h3>
        <div className="text-sm text-gray-500">
          Pack power, SOC, and current overlaid with baseline will appear here.
        </div>
      </section>
    </div>
  )
}
