---
title: Driver Strategies
tags: [module, driver, strategy]
aliases: [driver, strategy, control]
---

# Driver Strategies

> [!summary]
> Three driver control strategies determine throttle, coast, and brake behavior at each track segment. Strategies range from **replay** (reproduce real telemetry) to **synthetic** (algorithmic decisions).

**Source:** `src/fsae_sim/driver/strategy.py`, `src/fsae_sim/driver/strategies.py`

---

## Strategy Pattern

All strategies implement the same interface:

```mermaid
classDiagram
    class DriverStrategy {
        <<abstract>>
        +str name
        +decide(state, upcoming) ControlCommand*
    }

    class ReplayStrategy {
        +name = "replay"
        +target_speed(distance) float
        +target_torque(distance) float
        +decide(state, upcoming) ControlCommand
        +from_aim_data(df, start, end, lap_dist)$
    }

    class CoastOnlyStrategy {
        +name = "coast_only"
        +decide(state, upcoming) ControlCommand
    }

    class ThresholdBrakingStrategy {
        +name = "threshold_braking"
        +decide(state, upcoming) ControlCommand
    }

    DriverStrategy <|-- ReplayStrategy
    DriverStrategy <|-- CoastOnlyStrategy
    DriverStrategy <|-- ThresholdBrakingStrategy
```

### Decision Interface

```python
def decide(state: SimState, upcoming: list[Segment]) -> ControlCommand
```

- **Input:** Current simulation state + next 5 segments (lookahead)
- **Output:** One of three actions with intensity:

| Action | throttle_pct | brake_pct | Description |
|--------|-------------|-----------|-------------|
| THROTTLE | 0.0 — 1.0 | 0.0 | Apply motor torque |
| COAST | 0.0 | 0.0 | Zero torque, roll |
| BRAKE | 0.0 | 0.0 — 1.0 | Regen / friction braking |

---

## Strategy 1: ReplayStrategy

> [!info] Most Accurate for Validation
> Reproduces the real driver's behavior from AiM telemetry. This is the primary strategy for **validating** the simulation against recorded data.

### How It Works

```mermaid
flowchart LR
    AiM["AiM Telemetry<br/>throttle, brake,<br/>speed, torque<br/>vs. distance"] -->|"interpolate"| INT["Linear<br/>Interpolators"]
    INT -->|"query at<br/>current distance"| CMD["ControlCommand<br/>+ target speed<br/>+ target torque"]
```

- Builds interpolation functions from one recorded lap
- At each segment, queries throttle/brake/speed/torque at the current distance
- Speed comes directly from telemetry (not computed from forces)
- Wraps around `lap_distance_m` for multi-lap simulation

### Data Extraction

From AiM CSV:
- **Throttle:** `Throttle Pos / 100`, clamped to [0, 1]
- **Brake:** `max(FBrakePressure, RBrakePressure)`, normalized to 99th percentile
- **Speed:** `GPS Speed / 3.6` (km/h → m/s)
- **Torque:** `LVCU Torque Req`, capped at 85 Nm (inverter limit)

---

## Strategy 2: CoastOnlyStrategy

> [!note] Matches 2025 Team Approach
> The CT-16EV team used minimal braking — full throttle on straights, coasting into corners. This strategy models that behavior.

### Decision Logic

```mermaid
flowchart TD
    START["Current State"] --> CHECK{"speed > corner_limit<br/>- coast_margin?"}
    CHECK -->|Yes| COAST["COAST<br/>throttle=0, brake=0"]
    CHECK -->|No| THROTTLE["THROTTLE<br/>throttle=100%"]
```

- **Lookahead:** Finds minimum corner speed limit in next 5 segments
- **Coast margin:** 2.0 m/s default — starts coasting before reaching the limit
- **No braking at all** — relies on aerodynamic drag and rolling resistance to slow down

---

## Strategy 3: ThresholdBrakingStrategy

> [!tip] Most Realistic Synthetic Strategy
> Adds regenerative braking when coasting alone isn't enough to slow down for corners.

### Decision Logic

```mermaid
flowchart TD
    START["Current State"] --> CHECK1{"speed > corner_limit<br/>+ brake_threshold?"}
    CHECK1 -->|Yes| BRAKE["BRAKE<br/>brake = 50%"]
    CHECK1 -->|No| CHECK2{"speed > corner_limit<br/>- coast_margin?"}
    CHECK2 -->|Yes| COAST["COAST<br/>throttle=0, brake=0"]
    CHECK2 -->|No| THROTTLE["THROTTLE<br/>throttle=100%"]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coast_margin_ms` | 3.0 m/s | Start coasting this far before corner limit |
| `brake_threshold_ms` | 1.0 m/s | Brake if exceeding corner limit by this much |
| `brake_intensity` | 0.5 | Brake pedal fraction when applied |

---

## Strategy Comparison

```mermaid
xychart-beta
    title "Expected Behavior Comparison (conceptual)"
    x-axis "Track Distance (m)" [0, 100, 200, 300, 400, 500]
    y-axis "Speed (km/h)" 0 --> 80
    line "Replay" [30, 55, 65, 40, 35, 55]
    line "Coast Only" [30, 60, 70, 50, 45, 60]
    line "Threshold Brake" [30, 58, 68, 42, 38, 58]
```

| Strategy | Lap Time | Energy Use | Realism |
|----------|----------|------------|---------|
| Replay | Matches real | Matches real | Highest (is real) |
| Coast Only | Slower (over-speeds corners) | Lower (no regen) | Medium (2025 approach) |
| Threshold Braking | Moderate | Moderate (with regen) | Good (typical driver) |

---

## SimState (Driver Input)

The driver sees this snapshot at each decision point:

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Elapsed seconds |
| `distance` | float | Cumulative meters |
| `speed` | float | Current speed (m/s) |
| `soc` | float | Battery SOC (0-1) |
| `pack_voltage` | float | Terminal voltage (V) |
| `pack_current` | float | Current draw (A) |
| `cell_temp` | float | Cell temperature (°C) |
| `lap` | int | Current lap number |
| `segment_idx` | int | Current segment index |

See also: [[Simulation Engine]], [[Track Module]], [[Telemetry Data]]
