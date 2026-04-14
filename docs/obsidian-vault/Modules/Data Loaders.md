---
title: Data Loaders
tags: [module, data, loading]
aliases: [loader, CSV parser]
---

# Data Loaders

> [!summary]
> Two specialized CSV parsers for the project's data sources: AiM race telemetry and Voltt battery simulation exports.

**Source:** `src/fsae_sim/data/loader.py`

---

## AiM CSV Loader

### `load_aim_csv(path) → (metadata_dict, DataFrame)`

Parses the non-standard AiM Race Studio CSV export format:

```
"Format","AiM CSV File"           ┐
"Venue","MichiganS MI"            │ Metadata section
"Vehicle","CT-16EV"               │ (key-value pairs)
"Racer","EV Endurance 2025"       │
...                               ┘
                                  ← blank line
Time,GPS Speed,RPM,...            ← column headers
s,km/h,rpm,...                    ← units row
                                  ← blank line
"0.000","0.00","0.0",...          ← data (quoted)
```

**Returns:**
- `metadata`: dict with keys like `Format`, `Venue`, `Vehicle`, `Racer`
- `DataFrame`: all data columns, numeric types
  - `df.attrs['units']`: dict mapping column names to unit strings
  - `df.attrs['metadata']`: the metadata dict

> [!warning] Non-Standard Format
> AiM CSV is NOT standard CSV. It has a metadata preamble, a units row, and quoted numeric values. The parser handles all of this automatically.

---

## Voltt CSV Loader

### `load_voltt_csv(path) → DataFrame`

Parses Voltt battery simulation exports:

```
# Simulation ID: e73a8007-...    ┐
# Pack: 110S 4P                  │ Comment lines (# prefix)
# Cell: Molicel P45B             │
...                              ┘
Time [s],Voltage [V],SOC [%],... ← headers (with units in brackets)
0.0,4.185,100.0,...              ← data (unquoted)
```

**Columns returned:**

| Column | Unit | Description |
|--------|------|-------------|
| Time | s | Simulation time |
| Voltage | V | Terminal voltage |
| SOC | % | State of charge |
| Power | W | Electrical power |
| Current | A | Discharge current |
| Charge | Ah | Cumulative charge |
| OCV | V | Open-circuit voltage |
| Temperature | °C | Cell temperature |
| Heat Generation | W | Total heat |
| Cooling Power | W | Active cooling (0 for 2025) |
| Resistive Heat | W | I²R heating |
| Reversible Heat | W | Entropic heating |
| Hysteresis Heat | W | Hysteresis losses |

---

## Data File Locations

| File | Rows | Columns | Size |
|------|------|---------|------|
| `2025 Endurance Data.csv` | 37,196 | 114 | ~40 MB |
| `2025_Pack_cell.csv` | 18,264 | 13 | ~3.9 MB |
| `2025_Pack_pack.csv` | 18,264 | 13 | ~3.7 MB |
| `2026_Pack_cell.csv` | 18,264 | 13 | ~4.2 MB |
| `2026_Pack_pack.csv` | 18,264 | 13 | ~4.0 MB |

See also: [[Telemetry Data]], [[Battery Simulation Data]]
