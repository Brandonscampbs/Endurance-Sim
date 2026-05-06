"""Compare current-driven and power-driven battery trace stepping.

This is a battery-path verification harness, not a driver-model validator.
It answers two questions for a measured voltage/current trace:

1. If measured current is applied to the battery model, does model voltage
   reproduce measured terminal power?
2. If measured terminal power is applied to the battery model, does the
   solved current/voltage reproduce the measured trace?
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fsae_sim.data.loader import load_cleaned_csv, load_voltt_csv  # noqa: E402
from fsae_sim.vehicle import VehicleConfig  # noqa: E402
from fsae_sim.vehicle.battery_model import BatteryModel  # noqa: E402

CONFIG = REPO / "configs" / "ct16ev.yaml"
TELEMETRY = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"
VOLTT_CELL = (
    REPO / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_cell.csv"
)
VOLTT_PACK = (
    REPO / "Real-Car-Data-And-Stats"
    / "About-Energy-Volt-Simulations-2025-Pack"
    / "2025_Pack_pack.csv"
)
OUT = REPO / "results"


def _positive_time_steps(time_s: np.ndarray) -> np.ndarray:
    dt = np.diff(time_s, prepend=time_s[0]).astype(float)
    dt[~np.isfinite(dt)] = 0.0
    dt[dt < 0.0] = 0.0
    return dt


def _finite_rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _finite_mae(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.abs(values)))


def _finite_max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _net_kwh(power_w: np.ndarray, dt_s: np.ndarray) -> float:
    return float(np.sum(power_w * dt_s) / 3.6e6)


def _temperature_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.startswith("Temperature"):
            return col
    raise ValueError("Voltt pack CSV has no Temperature column")


def _build_model(
    config_path: Path,
    voltt_cell_path: Path,
    *,
    telemetry_df: pd.DataFrame | None = None,
    telemetry_pack_calibration: bool = False,
) -> BatteryModel:
    vehicle = VehicleConfig.from_yaml(str(config_path))
    model = BatteryModel.from_config_and_data(vehicle.battery, str(voltt_cell_path))
    if telemetry_pack_calibration:
        if telemetry_df is None:
            raise ValueError("telemetry_pack_calibration requires telemetry_df")
        model.calibrate_pack_from_telemetry(
            telemetry_df,
            allow_same_run_validation=True,
        )
    return model


def _run_current_trace(
    model: BatteryModel,
    time_s: np.ndarray,
    measured_current_a: np.ndarray,
    soc0: float,
    temp0_c: float,
) -> dict[str, np.ndarray | float]:
    dt_s = _positive_time_steps(time_s)
    n = len(time_s)
    voltage_v = np.empty(n, dtype=float)
    power_w = np.empty(n, dtype=float)
    soc_pct = np.empty(n, dtype=float)
    temp_c = np.empty(n, dtype=float)

    soc = float(soc0)
    temp = float(temp0_c)
    for idx in range(n):
        current = float(measured_current_a[idx])
        voltage = model.pack_voltage(soc, current, time_s=float(time_s[idx]))
        voltage_v[idx] = voltage
        power_w[idx] = voltage * current
        soc_pct[idx] = soc
        temp_c[idx] = temp
        soc, temp, _ = model.step(
            current,
            float(dt_s[idx]),
            soc,
            temp,
            time_s=float(time_s[idx]),
        )

    return {
        "voltage_v": voltage_v,
        "current_a": np.asarray(measured_current_a, dtype=float),
        "power_w": power_w,
        "soc_pct": soc_pct,
        "temp_c": temp_c,
        "final_soc_pct": float(soc),
        "final_temp_c": float(temp),
    }


def _run_power_trace(
    model: BatteryModel,
    time_s: np.ndarray,
    measured_power_w: np.ndarray,
    soc0: float,
    temp0_c: float,
) -> dict[str, np.ndarray | float]:
    dt_s = _positive_time_steps(time_s)
    n = len(time_s)
    voltage_v = np.empty(n, dtype=float)
    current_a = np.empty(n, dtype=float)
    soc_pct = np.empty(n, dtype=float)
    temp_c = np.empty(n, dtype=float)

    soc = float(soc0)
    temp = float(temp0_c)
    for idx in range(n):
        power = float(measured_power_w[idx])
        soc_pct[idx] = soc
        temp_c[idx] = temp
        soc, temp, end_voltage, current = model.step_power(
            power,
            float(dt_s[idx]),
            soc,
            temp,
            time_s=float(time_s[idx]),
        )
        current_a[idx] = current
        voltage_v[idx] = (
            power / current
            if abs(current) > 1e-9
            else end_voltage
        )

    return {
        "voltage_v": voltage_v,
        "current_a": current_a,
        "power_w": np.asarray(measured_power_w, dtype=float),
        "soc_pct": soc_pct,
        "temp_c": temp_c,
        "final_soc_pct": float(soc),
        "final_temp_c": float(temp),
    }


def _summarize_trace(
    label: str,
    time_s: np.ndarray,
    measured_voltage_v: np.ndarray,
    measured_current_a: np.ndarray,
    measured_power_w: np.ndarray,
    measured_soc_pct: np.ndarray | None,
    measured_temp_c: np.ndarray | None,
    current_trace: dict[str, np.ndarray | float],
    power_trace: dict[str, np.ndarray | float],
) -> dict[str, object]:
    dt_s = _positive_time_steps(time_s)
    current_power_error = (
        np.asarray(current_trace["power_w"]) - measured_power_w
    )
    power_current_error = (
        np.asarray(power_trace["current_a"]) - measured_current_a
    )
    power_voltage_error = (
        np.asarray(power_trace["voltage_v"]) - measured_voltage_v
    )
    current_voltage_error = (
        np.asarray(current_trace["voltage_v"]) - measured_voltage_v
    )
    aligned_power_error = (
        np.asarray(power_trace["voltage_v"])
        * np.asarray(power_trace["current_a"])
        - measured_power_w
    )

    summary: dict[str, object] = {
        "label": label,
        "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) else 0.0,
        "samples": int(len(time_s)),
        "measured_net_kwh": _net_kwh(measured_power_w, dt_s),
        "measured_net_charge_ah": float(
            np.sum(measured_current_a * dt_s) / 3600.0
        ),
        "current_trace_net_kwh": _net_kwh(
            np.asarray(current_trace["power_w"]), dt_s,
        ),
        "power_trace_net_kwh": _net_kwh(
            np.asarray(power_trace["power_w"]), dt_s,
        ),
        "current_to_power_mae_w": _finite_mae(current_power_error),
        "current_to_power_rmse_w": _finite_rmse(current_power_error),
        "current_to_voltage_mae_v": _finite_mae(current_voltage_error),
        "current_to_voltage_rmse_v": _finite_rmse(current_voltage_error),
        "power_to_current_mae_a": _finite_mae(power_current_error),
        "power_to_current_rmse_a": _finite_rmse(power_current_error),
        "power_to_voltage_mae_v": _finite_mae(power_voltage_error),
        "power_to_voltage_rmse_v": _finite_rmse(power_voltage_error),
        "power_trace_alignment_max_abs_w": _finite_max_abs(
            aligned_power_error,
        ),
        "current_trace_final_soc_pct": float(current_trace["final_soc_pct"]),
        "power_trace_final_soc_pct": float(power_trace["final_soc_pct"]),
        "current_trace_final_temp_c": float(current_trace["final_temp_c"]),
        "power_trace_final_temp_c": float(power_trace["final_temp_c"]),
    }
    if measured_soc_pct is not None:
        measured_soc_delta = float(measured_soc_pct[0] - measured_soc_pct[-1])
        measured_charge_ah = float(np.sum(measured_current_a * dt_s) / 3600.0)
        summary["measured_final_soc_pct"] = float(measured_soc_pct[-1])
        summary["measured_soc_delta_pct"] = measured_soc_delta
        summary["effective_capacity_ah_from_measured_soc"] = (
            measured_charge_ah / (measured_soc_delta / 100.0)
            if measured_soc_delta > 1e-9 else float("nan")
        )
        summary["current_trace_final_soc_error_pct"] = (
            float(current_trace["final_soc_pct"]) - float(measured_soc_pct[-1])
        )
        summary["power_trace_final_soc_error_pct"] = (
            float(power_trace["final_soc_pct"]) - float(measured_soc_pct[-1])
        )
    if measured_temp_c is not None:
        summary["measured_final_temp_c"] = float(measured_temp_c[-1])
        summary["current_trace_final_temp_error_c"] = (
            float(current_trace["final_temp_c"]) - float(measured_temp_c[-1])
        )
        summary["power_trace_final_temp_error_c"] = (
            float(power_trace["final_temp_c"]) - float(measured_temp_c[-1])
        )
    return summary


def _telemetry_case(args: argparse.Namespace) -> dict[str, object]:
    _, telemetry_df = load_cleaned_csv(args.telemetry)
    time_s = telemetry_df["Time"].to_numpy(dtype=float)
    voltage = telemetry_df["Pack Voltage"].to_numpy(dtype=float)
    current = telemetry_df["Pack Current"].to_numpy(dtype=float)
    power = voltage * current
    soc = telemetry_df["State of Charge"].to_numpy(dtype=float)
    temp = telemetry_df["Pack Temp"].to_numpy(dtype=float)

    current_model = _build_model(
        args.config,
        args.voltt_cell,
        telemetry_df=telemetry_df,
        telemetry_pack_calibration=not args.no_telemetry_pack_calibration,
    )
    power_model = _build_model(
        args.config,
        args.voltt_cell,
        telemetry_df=telemetry_df,
        telemetry_pack_calibration=not args.no_telemetry_pack_calibration,
    )
    return _summarize_trace(
        "telemetry",
        time_s,
        voltage,
        current,
        power,
        soc,
        temp,
        _run_current_trace(current_model, time_s, current, soc[0], temp[0]),
        _run_power_trace(power_model, time_s, power, soc[0], temp[0]),
    )


def _voltt_case(args: argparse.Namespace) -> dict[str, object]:
    voltt_pack = load_voltt_csv(args.voltt_pack)
    time_s = voltt_pack["Time [s]"].to_numpy(dtype=float)
    voltage = voltt_pack["Voltage [V]"].to_numpy(dtype=float)
    current = -voltt_pack["Current [A]"].to_numpy(dtype=float)
    power = voltage * current
    soc = voltt_pack["SOC [%]"].to_numpy(dtype=float)
    temp = voltt_pack[_temperature_column(voltt_pack)].to_numpy(dtype=float)

    current_model = _build_model(args.config, args.voltt_cell)
    power_model = _build_model(args.config, args.voltt_cell)
    return _summarize_trace(
        "voltt_pack",
        time_s,
        voltage,
        current,
        power,
        soc,
        temp,
        _run_current_trace(current_model, time_s, current, soc[0], temp[0]),
        _run_power_trace(power_model, time_s, power, soc[0], temp[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--telemetry", type=Path, default=TELEMETRY)
    parser.add_argument("--voltt-cell", type=Path, default=VOLTT_CELL)
    parser.add_argument("--voltt-pack", type=Path, default=VOLTT_PACK)
    parser.add_argument(
        "--no-telemetry-pack-calibration",
        action="store_true",
        help="use Voltt-only pack OCV/R for telemetry trace comparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "battery_power_trace_compare_summary.json",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = {
            "config": str(args.config),
            "voltt_cell": str(args.voltt_cell),
            "voltt_pack": str(args.voltt_pack),
            "telemetry": str(args.telemetry),
            "telemetry_pack_calibration": (
                not args.no_telemetry_pack_calibration
            ),
            "cases": [_telemetry_case(args), _voltt_case(args)],
        }

    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
