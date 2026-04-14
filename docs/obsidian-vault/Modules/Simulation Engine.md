---
title: Simulation Engine
tags: [module, simulation, engine]
aliases: [engine, SimulationEngine]
---

# Simulation Engine

> [!summary]
> The central orchestrator that ties all physics models together — stepping through track segments, computing forces, resolving speeds, tracking battery state, and producing a detailed results DataFrame.

**Source:** `src/fsae_sim/sim/engine.py`

---

## Architecture

```mermaid
flowchart TD
    subgraph Inputs
        VC["VehicleConfig"]
        TRK["Track<br/>(~240 segments)"]
        STRAT["DriverStrategy"]
        BAT["BatteryModel<br/>(calibrated)"]
    end

    subgraph Engine["SimulationEngine"]
        PT["PowertrainModel"]
        DYN["VehicleDynamics"]
        LOOP["Segment Loop<br/>for each lap:<br/>  for each segment:<br/>    decide → resolve → step"]
    end

    subgraph Output
        RES["SimResult"]
        DF["states DataFrame<br/>24 columns per row"]
        MET["Summary Metrics<br/>time, energy, SOC, laps"]
    end

    VC --> PT & DYN
    TRK --> LOOP
    STRAT --> LOOP
    BAT --> LOOP
    PT --> LOOP
    DYN --> LOOP
    LOOP --> RES
    RES --> DF & MET

    style Engine fill:#5c1a3a,stroke:#a32d6d,color:#fff
```

---

## The Run Loop (Detailed)

```python
engine.run(num_laps=1, initial_soc_pct=95.0, initial_temp_c=25.0, initial_speed_ms=0.0)
```

```mermaid
flowchart TD
    INIT["Initialize State<br/>speed=0, soc=95%, temp=25°C"]
    INIT --> LLOOP["For each lap (0 to num_laps-1)"]
    LLOOP --> SLOOP["For each segment (0 to ~239)"]

    SLOOP --> DECIDE["1. Driver Decision<br/>strategy.decide(state, next 5 segments)<br/>→ THROTTLE / COAST / BRAKE"]

    DECIDE --> MODE{"Replay<br/>Strategy?"}

    MODE -->|"Yes"| REPLAY["2a. Use Recorded Speed<br/>exit_speed = target_speed(distance)<br/>torque = target_torque(distance)"]

    MODE -->|"No"| FORCE["2b. Force Balance<br/>drive_force, regen_force, resistance<br/>net_force = drive + regen - resist"]

    FORCE --> KINE["3. Kinematics<br/>v_exit² = v_entry² + 2ad<br/>clamp to corner speed limit"]

    REPLAY --> POWER
    KINE --> POWER

    POWER["4. Electrical Power<br/>P = τ × ω / η (motoring)<br/>P = τ × ω × η × η_regen (regen)"]

    POWER --> CURRENT["5. Pack Current<br/>I = P / V_pack"]

    CURRENT --> BMS["6. BMS Limit Check<br/>I_max = min(temp_limit, soc_taper, voltage_floor)<br/>clamp current if needed"]

    BMS --> STEP["7. Battery Step<br/>ΔSOC = -I·dt / (C·3600) × 100<br/>ΔT = I²R·dt / (m·cp)"]

    STEP --> RECORD["8. Record State<br/>append row to DataFrame"]

    RECORD --> TERM{"SOC ≤ 2%?<br/>Temp ≥ 65°C?"}
    TERM -->|"Yes"| DONE["Return SimResult<br/>(early termination)"]
    TERM -->|"No"| NEXT["Next Segment"]
    NEXT --> SLOOP

    SLOOP -->|"Lap complete"| LLOOP
    LLOOP -->|"All laps done"| DONE

    style DECIDE fill:#2d6da3,stroke:#fff,color:#fff
    style BMS fill:#a32d2d,stroke:#fff,color:#fff
    style DONE fill:#2da34a,stroke:#fff,color:#fff
```

---

## SimResult Structure

```mermaid
classDiagram
    class SimResult {
        +str config_name
        +str strategy_name
        +str track_name
        +DataFrame states
        +float total_time_s
        +float total_energy_kwh
        +float final_soc
        +int laps_completed
    }
```

### States DataFrame (24 columns)

| Group | Columns |
|-------|---------|
| **Position** | lap, segment_idx, time_s, distance_m |
| **Speed** | speed_ms, speed_kmh |
| **Battery** | soc_pct, pack_voltage_v, pack_current_a, cell_temp_c |
| **Powertrain** | motor_rpm, motor_torque_nm, electrical_power_w |
| **Forces** | drive_force_n, regen_force_n, resistance_force_n, net_force_n |
| **Timing** | segment_time_s |
| **Driver** | action, throttle_pct, brake_pct |
| **Track** | curvature, corner_speed_limit_ms, grade |

---

## Two Execution Modes

The engine handles [[Driver Strategies|ReplayStrategy]] differently from synthetic strategies:

| Aspect | Replay Mode | Force-Based Mode |
|--------|-------------|-----------------|
| Speed source | Recorded telemetry | Kinematic equation |
| Torque source | Recorded LVCU request | Computed from throttle × max_torque |
| Corner limits | Not applied (real driver already respected them) | Applied as speed clamp |
| Best for | Validation against real data | What-if analysis |

---

## Energy Accounting

Total energy consumed:

$$E_{total} = \sum_{segments} P_{electrical} \times \Delta t$$

Where:
- Positive $P$ = discharge (motoring) — adds to consumption
- Negative $P$ = charge (regen) — subtracts from consumption

Converted to kWh: $E_{kwh} = E_{total} / 3,600,000$

---

## Usage

```python
from fsae_sim.vehicle import VehicleConfig
from fsae_sim.vehicle.battery_model import BatteryModel
from fsae_sim.track import Track
from fsae_sim.driver.strategies import CoastOnlyStrategy
from fsae_sim.sim import SimulationEngine

# Setup
config = VehicleConfig.from_yaml("configs/ct16ev.yaml")
track = Track.from_telemetry("Real-Car-Data-And-Stats/2025 Endurance Data.csv")
battery = BatteryModel.from_config_and_data(config.battery, voltt_cell_csv)
strategy = CoastOnlyStrategy(dynamics)

# Run
engine = SimulationEngine(config, track, strategy, battery)
result = engine.run(num_laps=22, initial_soc_pct=95.0)

# Results
print(f"Total time: {result.total_time_s:.1f} s")
print(f"Energy used: {result.total_energy_kwh:.2f} kWh")
print(f"Final SOC: {result.final_soc:.1f}%")
print(f"Laps completed: {result.laps_completed}")
```

See also: [[System Overview]], [[Data Flow]], [[Driver Strategies]]
