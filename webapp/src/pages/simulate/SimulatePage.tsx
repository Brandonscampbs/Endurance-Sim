import { Card, CardBody, CardHeader, CardTitle } from '../../components/ui'
import { useSimulateStore } from '../../stores/simulateStore'
import LapDeltaTable from './LapDeltaTable'
import RunNotes from './RunNotes'
import RunSummaryCards from './RunSummaryCards'
import SimulateEmpty from './SimulateEmpty'
import TimeSeriesOverlay from './TimeSeriesOverlay'
import TuneForm from './TuneForm'

export default function SimulatePage() {
  const status = useSimulateStore((s) => s.status)
  const result = useSimulateStore((s) => s.result)
  const error = useSimulateStore((s) => s.error)
  const runSimulation = useSimulateStore((s) => s.runSimulation)

  const hasResult = result !== null
  const isRunning = status === 'running'
  const showResults = hasResult || isRunning
  const showEmpty = status === 'idle' && !hasResult

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-2xl font-bold">Simulate</h2>
        <p className="text-sm text-[var(--text-tertiary)] mt-1">
          Change three tune parameters and re-run the Michigan 2025 endurance. Everything else
          (driver model, track, mass, aero, tires) stays fixed at the baseline.
        </p>
      </div>

      <TuneForm />

      {status === 'error' && (
        <Card>
          <CardHeader>
            <CardTitle>Run failed</CardTitle>
          </CardHeader>
          <CardBody className="space-y-3">
            <p className="text-sm text-[var(--error)]">
              {error ?? 'The simulation failed for an unknown reason.'}
            </p>
            <button
              type="button"
              onClick={() => void runSimulation()}
              style={{ backgroundColor: 'var(--accent)', color: '#fff' }}
              className="px-4 py-2 rounded text-sm font-medium"
            >
              Retry
            </button>
          </CardBody>
        </Card>
      )}

      {showResults && (
        <>
          <RunSummaryCards />
          <LapDeltaTable />
          <TimeSeriesOverlay />
          <RunNotes />
        </>
      )}

      {showEmpty && <SimulateEmpty />}
    </div>
  )
}
