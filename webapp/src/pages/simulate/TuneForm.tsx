import { useEffect, useMemo, useState } from 'react'
import { Card, CardActions, CardBody, CardHeader, CardTitle } from '../../components/ui'
import {
  BASELINE_PARAMS,
  useSimulateStore,
  type SocDischargePoint,
} from '../../stores/simulateStore'

const RPM_MIN = 1000
const RPM_MAX = 6500
const TORQUE_MIN = 20
const TORQUE_MAX = 230
const SOC_MIN = 0
const SOC_MAX = 100
const CURRENT_MIN = 0
const CURRENT_MAX = 200

interface FormErrors {
  rpm: string | null
  torque: string | null
  /** keyed by `${idx}-soc` or `${idx}-current` */
  soc: Record<string, string>
}

function validate(
  rpm: number,
  torque: number,
  socMap: SocDischargePoint[],
): FormErrors {
  const socErrors: Record<string, string> = {}
  const errors: FormErrors = {
    rpm:
      !Number.isFinite(rpm) || rpm < RPM_MIN || rpm > RPM_MAX
        ? `Must be ${RPM_MIN}-${RPM_MAX}`
        : null,
    torque:
      !Number.isFinite(torque) || torque < TORQUE_MIN || torque > TORQUE_MAX
        ? `Must be ${TORQUE_MIN}-${TORQUE_MAX} Nm`
        : null,
    soc: socErrors,
  }

  socMap.forEach((pt, i) => {
    if (!Number.isFinite(pt.soc_pct) || pt.soc_pct < SOC_MIN || pt.soc_pct > SOC_MAX) {
      socErrors[`${i}-soc`] = `SOC must be ${SOC_MIN}-${SOC_MAX}%`
    }
    if (
      !Number.isFinite(pt.max_current_a) ||
      pt.max_current_a < CURRENT_MIN ||
      pt.max_current_a > CURRENT_MAX
    ) {
      socErrors[`${i}-current`] = `Current must be ${CURRENT_MIN}-${CURRENT_MAX} A`
    }
    // Strictly descending SOC values across rows.
    if (i > 0 && pt.soc_pct >= socMap[i - 1].soc_pct) {
      socErrors[`${i}-soc`] = 'SOC must be strictly descending'
    }
    // Current non-decreasing as SOC increases (low-SOC current <= high-SOC current).
    if (i > 0 && pt.max_current_a > socMap[i - 1].max_current_a) {
      socErrors[`${i}-current`] = 'Current must not increase as SOC drops'
    }
  })

  return errors
}

function inputClass(invalid: boolean, extra = ''): string {
  return [
    'bg-[var(--surface-3)] border rounded px-3 py-1.5 text-sm tabular-nums',
    'text-[var(--text-primary)] focus:outline-none focus:border-[var(--border-strong)]',
    invalid
      ? 'border-[var(--error)]'
      : 'border-[var(--border-subtle)]',
    extra,
  ]
    .filter(Boolean)
    .join(' ')
}

function elapsedSeconds(start: number | null, now: number): string {
  if (start === null) return '0.0'
  return ((now - start) / 1000).toFixed(1)
}

export default function TuneForm() {
  const params = useSimulateStore((s) => s.params)
  const setMaxRpm = useSimulateStore((s) => s.setMaxRpm)
  const setMaxTorque = useSimulateStore((s) => s.setMaxTorque)
  const setSocPoint = useSimulateStore((s) => s.setSocPoint)
  const resetToBaseline = useSimulateStore((s) => s.resetToBaseline)
  const status = useSimulateStore((s) => s.status)
  const runStartedAt = useSimulateStore((s) => s.runStartedAt)
  const runSimulation = useSimulateStore((s) => s.runSimulation)

  const errors = useMemo(
    () => validate(params.max_rpm, params.max_torque_nm, params.soc_discharge_map),
    [params],
  )
  const isInvalid =
    errors.rpm !== null ||
    errors.torque !== null ||
    Object.keys(errors.soc).length > 0
  const isRunning = status === 'running'

  // Tick the elapsed timer while running. Pinned to runStartedAt so we don't
  // reschedule on unrelated re-renders.
  const [tick, setTick] = useState(Date.now())
  useEffect(() => {
    if (!isRunning) return
    const id = window.setInterval(() => setTick(Date.now()), 250)
    return () => window.clearInterval(id)
  }, [isRunning])

  const elapsed = elapsedSeconds(runStartedAt, tick)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tune parameters</CardTitle>
      </CardHeader>
      <CardBody className="space-y-5">
        <div className="grid grid-cols-2 gap-5">
          <label className="flex flex-col gap-1.5">
            <span className="text-xs uppercase tracking-wider text-[var(--text-tertiary)]">
              Max motor RPM
              <span className="ml-2 normal-case tracking-normal text-[var(--text-muted)]">
                (baseline {BASELINE_PARAMS.max_rpm})
              </span>
            </span>
            <input
              type="number"
              value={params.max_rpm}
              onChange={(e) => setMaxRpm(Number(e.target.value))}
              min={RPM_MIN}
              max={RPM_MAX}
              step={50}
              disabled={isRunning}
              aria-invalid={errors.rpm !== null}
              className={inputClass(errors.rpm !== null)}
            />
            {errors.rpm && (
              <span className="text-xs text-[var(--error)]">{errors.rpm}</span>
            )}
          </label>

          <label className="flex flex-col gap-1.5">
            <span className="text-xs uppercase tracking-wider text-[var(--text-tertiary)]">
              Max motor torque (Nm)
              <span className="ml-2 normal-case tracking-normal text-[var(--text-muted)]">
                (baseline {BASELINE_PARAMS.max_torque_nm})
              </span>
            </span>
            <input
              type="number"
              value={params.max_torque_nm}
              onChange={(e) => setMaxTorque(Number(e.target.value))}
              min={TORQUE_MIN}
              max={TORQUE_MAX}
              step={1}
              disabled={isRunning}
              aria-invalid={errors.torque !== null}
              className={inputClass(errors.torque !== null)}
            />
            {errors.torque && (
              <span className="text-xs text-[var(--error)]">{errors.torque}</span>
            )}
          </label>
        </div>

        <div>
          <div className="text-xs uppercase tracking-wider text-[var(--text-tertiary)] mb-2">
            BMS current limit map
          </div>
          <table className="w-full text-sm tabular-nums">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)] border-b border-[var(--border-subtle)]">
                <th className="text-left py-2 font-medium">Internal SOC (%)</th>
                <th className="text-left py-2 font-medium">Max pack current (A)</th>
              </tr>
            </thead>
            <tbody>
              {params.soc_discharge_map.map((pt, idx) => {
                const socErr = errors.soc[`${idx}-soc`]
                const curErr = errors.soc[`${idx}-current`]
                const rowBg = idx % 2 === 1 ? 'bg-[var(--surface-2)]/40' : ''
                return (
                  <tr key={idx} className={`border-b border-[var(--border-subtle)]/60 ${rowBg}`}>
                    <td className="py-1.5 pr-2">
                      <input
                        type="number"
                        value={pt.soc_pct}
                        onChange={(e) =>
                          setSocPoint(idx, { ...pt, soc_pct: Number(e.target.value) })
                        }
                        min={SOC_MIN}
                        max={SOC_MAX}
                        step={1}
                        disabled={isRunning}
                        aria-invalid={Boolean(socErr)}
                        className={inputClass(Boolean(socErr), 'w-24')}
                      />
                      {socErr && (
                        <div className="text-[11px] text-[var(--error)] mt-1">{socErr}</div>
                      )}
                    </td>
                    <td className="py-1.5">
                      <input
                        type="number"
                        value={pt.max_current_a}
                        onChange={(e) =>
                          setSocPoint(idx, { ...pt, max_current_a: Number(e.target.value) })
                        }
                        min={CURRENT_MIN}
                        max={CURRENT_MAX}
                        step={1}
                        disabled={isRunning}
                        aria-invalid={Boolean(curErr)}
                        className={inputClass(Boolean(curErr), 'w-24')}
                      />
                      {curErr && (
                        <div className="text-[11px] text-[var(--error)] mt-1">{curErr}</div>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="mt-2 text-[11px] italic text-[var(--text-tertiary)]">
            Approximated to a 2-knee taper before running. The full piecewise curve is
            summarised by a single threshold + slope; the response notes confirm what was applied.
          </p>
        </div>
      </CardBody>
      <CardActions>
        <button
          type="button"
          onClick={resetToBaseline}
          disabled={isRunning}
          className="px-4 py-2 rounded text-sm font-medium bg-[var(--surface-3)] hover:bg-[var(--surface-2)] text-[var(--text-secondary)] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reset to baseline
        </button>
        <button
          type="button"
          onClick={() => void runSimulation()}
          disabled={isRunning || isInvalid}
          style={{
            backgroundColor:
              isRunning || isInvalid ? 'var(--accent-soft)' : 'var(--accent)',
            color: isRunning || isInvalid ? 'var(--text-tertiary)' : '#fff',
          }}
          className="px-4 py-2 rounded text-sm font-medium disabled:cursor-not-allowed transition-colors"
        >
          {isRunning ? `Running… ${elapsed}s` : 'Run simulation'}
        </button>
      </CardActions>
    </Card>
  )
}
