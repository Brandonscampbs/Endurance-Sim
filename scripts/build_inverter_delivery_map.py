"""Build the inverter torque-delivery map from AiM telemetry.

The LVCU firmware computes a torque request and sends it to the Cascadia
CM200DX inverter. The inverter then applies its own current-limited
torque control and reports back the achieved motor shaft torque on the
``Torque Feedback`` channel. The two are not equal: across Michigan
2025, the inverter delivered ~88.4% of the LVCU's request on average,
with operating-point dependence on RPM and command magnitude.

This script bins the recorded telemetry by (motor RPM, LVCU torque
request) and writes the mean delivered torque per bin to a CSV that
``InverterDeliveryMap`` can load.

Output columns: rpm, command_nm, delivered_nm, n_samples.

Usage:
    python scripts/build_inverter_delivery_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.data.loader import load_cleaned_csv  # noqa: E402

TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
OUT_CSV = REPO / "Real-Car-Data-And-Stats" / "inverter_delivery_map.csv"

# Inverter saturates here (CM200DX with IQ=170A setting). LVCU may
# request beyond this, but the inverter cannot deliver more.
INVERTER_TORQUE_CAP_NM = 85.0

# Grid: motor RPM (0..3000 in 100-RPM steps) x LVCU command
# (0..85 Nm in 5-Nm steps). 31 x 18 = 558 bins.
RPM_GRID = np.arange(0, 3100, 100, dtype=float)
COMMAND_GRID = np.arange(0, 90, 5, dtype=float)


def main() -> None:
    _, df = load_cleaned_csv(str(TELEM))

    rpm = df["Motor RPM"].to_numpy(dtype=float)
    req_raw = df["LVCU Torque Req"].to_numpy(dtype=float)
    fb_raw = df["Torque Feedback"].to_numpy(dtype=float)

    # The replay strategy clips LVCU Req to +-85 Nm because that is the
    # inverter's mechanical ceiling; build the map on the same domain.
    req = np.clip(req_raw, 0.0, INVERTER_TORQUE_CAP_NM)
    # Torque Feedback occasionally pegs at 0xFFFF/10 = 6553.5 Nm during
    # CAN dropouts. Clip those out before averaging.
    fb = np.clip(fb_raw, 0.0, INVERTER_TORQUE_CAP_NM)

    # Restrict to powered samples (req > 0). Coast and brake events
    # produce zero command and we anchor command=0 -> delivered=0
    # explicitly below.
    mask = req > 0.5
    rpm_v = rpm[mask]
    req_v = req[mask]
    fb_v = fb[mask]
    print(f"Powered samples: {mask.sum():d} / {len(rpm):d}")

    # Bin index for each sample using digitize (0..len-1 inclusive).
    rpm_idx = np.clip(np.digitize(rpm_v, RPM_GRID) - 1, 0, len(RPM_GRID) - 1)
    cmd_idx = np.clip(
        np.digitize(req_v, COMMAND_GRID) - 1, 0, len(COMMAND_GRID) - 1,
    )

    delivered = np.full((len(RPM_GRID), len(COMMAND_GRID)), np.nan)
    counts = np.zeros_like(delivered, dtype=int)

    flat = rpm_idx * len(COMMAND_GRID) + cmd_idx
    sums = np.bincount(flat, weights=fb_v, minlength=delivered.size)
    cnts = np.bincount(flat, minlength=delivered.size)
    sums = sums.reshape(delivered.shape)
    cnts = cnts.reshape(delivered.shape)

    valid = cnts >= 5
    delivered[valid] = sums[valid] / cnts[valid]
    counts[:, :] = cnts

    # Anchor: when LVCU command is zero, the inverter delivers zero
    # torque (zero IQ command). Set the entire command=0 column to 0.
    delivered[:, 0] = 0.0

    # Fill sparse bins per RPM row by interpolating across the command
    # axis. Where a row has any valid samples, np.interp fills the gaps.
    # Where a row has no samples at all (very low RPM), copy the nearest
    # populated row.
    for i in range(len(RPM_GRID)):
        row = delivered[i, :]
        valid_cols = ~np.isnan(row)
        if valid_cols.sum() >= 2:
            delivered[i, :] = np.interp(
                np.arange(len(COMMAND_GRID)),
                np.where(valid_cols)[0],
                row[valid_cols],
            )

    # Any RPM row still entirely NaN: copy from the nearest filled row.
    rows_full = ~np.isnan(delivered).any(axis=1)
    if rows_full.any():
        good_rpm_idx = np.where(rows_full)[0]
        for i in range(len(RPM_GRID)):
            if not rows_full[i]:
                nearest = good_rpm_idx[np.argmin(np.abs(good_rpm_idx - i))]
                delivered[i, :] = delivered[nearest, :]
    else:
        raise RuntimeError(
            "No RPM row has full data; cannot build delivery map."
        )

    # Enforce monotonicity in command per RPM row (delivered torque
    # should not decrease as command increases). Use a cumulative-max
    # along the command axis to repair small statistical inversions.
    delivered = np.maximum.accumulate(delivered, axis=1)
    # Hard cap: delivered <= command (the inverter cannot amplify).
    delivered = np.minimum(delivered, COMMAND_GRID[np.newaxis, :])
    # And clip to the inverter's mechanical ceiling.
    delivered = np.clip(delivered, 0.0, INVERTER_TORQUE_CAP_NM)

    # Long-format CSV.
    rows: list[dict[str, float | int]] = []
    for i, rpm_v in enumerate(RPM_GRID):
        for j, cmd_v in enumerate(COMMAND_GRID):
            rows.append(
                {
                    "rpm": float(rpm_v),
                    "command_nm": float(cmd_v),
                    "delivered_nm": float(delivered[i, j]),
                    "n_samples": int(counts[i, j]),
                }
            )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} rows to {OUT_CSV}")

    # Sanity prints.
    print()
    print("Mean delivered/command ratio per RPM row (where command > 0):")
    for i, rpm_v in enumerate(RPM_GRID):
        cmd_pos = COMMAND_GRID[1:]
        del_pos = delivered[i, 1:]
        ratio = del_pos / cmd_pos
        print(
            f"  rpm={rpm_v:4.0f}  ratio mean={ratio.mean():.3f}  "
            f"min={ratio.min():.3f}  max={ratio.max():.3f}"
        )


if __name__ == "__main__":
    main()
