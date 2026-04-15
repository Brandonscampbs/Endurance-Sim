import type { VizFrame, VisualizationResponse } from '../../api/client'
import { formatSpeed, formatForce, formatPercent, formatTime } from '../../utils/formatters'

function gripBg(util: number): string {
  if (util < 0.6) return 'bg-green-900/50 border-green-700'
  if (util < 0.8) return 'bg-yellow-900/50 border-yellow-700'
  return 'bg-red-900/50 border-red-700'
}

function BarGauge({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="w-8 h-24 bg-gray-800 rounded relative overflow-hidden">
      <div
        className="absolute bottom-0 w-full rounded-b transition-all duration-75"
        style={{ height: `${pct}%`, backgroundColor: color }}
      />
    </div>
  )
}

function TrackMinimap({ frame, data }: { frame: VizFrame; data: VisualizationResponse }) {
  const minX = Math.min(...data.track_centerline_x)
  const maxX = Math.max(...data.track_centerline_x)
  const minY = Math.min(...data.track_centerline_y)
  const maxY = Math.max(...data.track_centerline_y)
  const range = Math.max(maxX - minX, maxY - minY) || 1

  const points = data.track_centerline_x
    .map((x, i) => {
      const nx = ((x - minX) / range) * 100
      const ny = 100 - ((data.track_centerline_y[i] - minY) / range) * 100
      return `${nx},${ny}`
    })
    .join(' ')

  const cx = ((frame.x - minX) / range) * 100
  const cy = 100 - ((frame.y - minY) / range) * 100

  return (
    <svg viewBox="-10 -10 120 120" className="w-full bg-gray-800/50 rounded">
      <polyline
        points={points}
        fill="none"
        stroke="#4b5563"
        strokeWidth="1.5"
      />
      <circle cx={cx} cy={cy} r="3" fill="#22c55e" />
    </svg>
  )
}

interface Props {
  frame: VizFrame
  data: VisualizationResponse
}

export default function SidePanel({ frame, data }: Props) {
  const wheelLabels = ['FL', 'FR', 'RL', 'RR']

  return (
    <div className="w-64 bg-gray-900 border-l border-gray-800 flex flex-col overflow-y-auto">
      {/* Telemetry */}
      <div className="p-3 border-b border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Telemetry</h4>
        <div className="text-2xl font-bold text-green-400">{formatSpeed(frame.speed_kmh)}</div>
        <div className="grid grid-cols-2 gap-1 mt-2 text-xs">
          <div className="text-gray-400">RPM</div>
          <div className="text-right">{Math.round(frame.motor_rpm)}</div>
          <div className="text-gray-400">Torque</div>
          <div className="text-right">{frame.motor_torque_nm.toFixed(1)} Nm</div>
          <div className="text-gray-400">Voltage</div>
          <div className="text-right">{frame.pack_voltage_v.toFixed(0)} V</div>
          <div className="text-gray-400">Current</div>
          <div className="text-right">{frame.pack_current_a.toFixed(1)} A</div>
          <div className="text-gray-400">SOC</div>
          <div className="text-right">{frame.soc_pct.toFixed(1)}%</div>
          <div className="text-gray-400">Time</div>
          <div className="text-right">{formatTime(frame.time_s)}</div>
        </div>
        {/* SOC bar */}
        <div className="mt-2 h-2 bg-gray-800 rounded overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-75"
            style={{ width: `${frame.soc_pct}%` }}
          />
        </div>
      </div>

      {/* Driver Inputs */}
      <div className="p-3 border-b border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Driver Inputs</h4>
        <div className="flex items-end justify-center gap-6">
          <div className="text-center">
            <BarGauge value={frame.throttle_pct} max={100} color="#22c55e" />
            <div className="text-xs mt-1 text-gray-400">THR</div>
            <div className="text-xs font-mono">{frame.throttle_pct.toFixed(0)}%</div>
          </div>
          <div className="text-center">
            <BarGauge value={frame.brake_pct} max={100} color="#ef4444" />
            <div className="text-xs mt-1 text-gray-400">BRK</div>
            <div className="text-xs font-mono">{frame.brake_pct.toFixed(0)}%</div>
          </div>
        </div>
      </div>

      {/* Tire Forces */}
      <div className="p-3 border-b border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Tire Forces</h4>
        <div className="grid grid-cols-2 gap-2">
          {frame.wheels.map((w, i) => (
            <div key={i} className={`rounded p-2 border text-center text-xs ${gripBg(w.grip_util)}`}>
              <div className="text-gray-400 font-medium">{wheelLabels[i]}</div>
              <div className="font-mono">{formatForce(Math.sqrt(w.fx ** 2 + w.fy ** 2))}</div>
              <div className="text-gray-500">{formatPercent(w.grip_util * 100)} grip</div>
            </div>
          ))}
        </div>
      </div>

      {/* Minimap */}
      <div className="p-3 mt-auto">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Track Position</h4>
        <TrackMinimap frame={frame} data={data} />
      </div>
    </div>
  )
}
