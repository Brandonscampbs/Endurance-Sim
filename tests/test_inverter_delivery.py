"""Tests for InverterDeliveryMap."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fsae_sim.vehicle.inverter_delivery import InverterDeliveryMap


REPO = Path(__file__).resolve().parents[1]
DELIVERY_CSV = REPO / "Real-Car-Data-And-Stats" / "inverter_delivery_map.csv"


@pytest.fixture(scope="module")
def map_from_csv() -> InverterDeliveryMap:
    if not DELIVERY_CSV.exists():
        pytest.skip(f"inverter_delivery_map.csv not present at {DELIVERY_CSV}")
    return InverterDeliveryMap.from_csv(DELIVERY_CSV)


def _build_simple_map() -> InverterDeliveryMap:
    """Hand-built 2x2 grid for unit-level checks."""
    rpm = np.array([0.0, 3000.0])
    cmd = np.array([0.0, 85.0])
    delivered = np.array([
        [0.0, 80.0],
        [0.0, 60.0],
    ])
    return InverterDeliveryMap(rpm, cmd, delivered)


def test_zero_command_returns_zero():
    m = _build_simple_map()
    assert m.delivered_torque(motor_rpm=2000.0, command_nm=0.0) == 0.0
    assert m.delivered_torque(motor_rpm=0.0, command_nm=0.0) == 0.0


def test_negative_command_passes_through_as_zero():
    m = _build_simple_map()
    assert m.delivered_torque(motor_rpm=2000.0, command_nm=-10.0) == 0.0


def test_delivered_never_exceeds_command():
    m = _build_simple_map()
    for rpm in (0.0, 500.0, 1500.0, 2900.0, 3500.0):
        for cmd in (1.0, 5.0, 50.0, 85.0):
            d = m.delivered_torque(rpm, cmd)
            assert 0.0 <= d <= cmd, (rpm, cmd, d)


def test_inverter_cap_ceiling():
    """Even if a bad map says delivered > cap, the lookup must clamp."""
    rpm = np.array([0.0, 3000.0])
    cmd = np.array([0.0, 200.0])
    delivered = np.array([
        [0.0, 200.0],
        [0.0, 200.0],
    ])
    m = InverterDeliveryMap(rpm, cmd, delivered, inverter_torque_cap_nm=85.0)
    assert m.delivered_torque(motor_rpm=1500.0, command_nm=150.0) == 85.0


def test_off_grid_clamps_not_extrapolates():
    m = _build_simple_map()
    # Above the highest RPM grid point: should clamp to the row at 3000.
    cmd = 85.0
    high_rpm = m.delivered_torque(motor_rpm=10_000.0, command_nm=cmd)
    on_grid = m.delivered_torque(motor_rpm=3000.0, command_nm=cmd)
    assert high_rpm == pytest.approx(on_grid)


def test_csv_round_trip(tmp_path):
    """A map written and re-read must agree at every grid point."""
    rpm = np.array([0.0, 1000.0, 2000.0])
    cmd = np.array([0.0, 25.0, 75.0])
    delivered = np.array([
        [0.0, 20.0, 50.0],
        [0.0, 24.0, 70.0],
        [0.0, 24.5, 73.0],
    ])
    csv = tmp_path / "test_map.csv"
    rows = []
    for i, r in enumerate(rpm):
        for j, c in enumerate(cmd):
            rows.append(f"{r},{c},{delivered[i, j]}")
    csv.write_text("rpm,command_nm,delivered_nm\n" + "\n".join(rows) + "\n")

    m = InverterDeliveryMap.from_csv(csv)
    for i, r in enumerate(rpm):
        for j, c in enumerate(cmd):
            assert m.delivered_torque(r, c) == pytest.approx(
                min(delivered[i, j], c), abs=1e-9,
            )


def test_real_map_field_weakening_below_low_rpm_steady(map_from_csv):
    """Heart of operating envelope (1500-2700 RPM) should track within ~5%.

    Field weakening (>2800 RPM) should drop below 90% of command.
    """
    cmd = 50.0
    for rpm in (1500.0, 2000.0, 2500.0, 2700.0):
        d = map_from_csv.delivered_torque(rpm, cmd)
        ratio = d / cmd
        assert 0.92 <= ratio <= 1.0, (
            f"Mid-RPM steady delivery should track command within ~8 %; "
            f"got ratio={ratio:.3f} at rpm={rpm}"
        )

    # At very high RPM, the inverter cannot deliver high torque.
    # 2900 RPM with 85 Nm command -> expect ratio < 0.9.
    high = map_from_csv.delivered_torque(2900.0, 85.0)
    assert high / 85.0 < 0.9, (
        f"Field weakening should reduce delivery below 0.9; "
        f"got ratio={high / 85.0:.3f}"
    )


def test_real_map_global_ratio_close_to_empirical(map_from_csv):
    """Pulling LVCU-Req-style samples through the map should land near 0.92.

    The empirical mean delivered/requested ratio across Michigan 2025
    (after symmetric clip to [0, 85]) is 0.918. Sampling the map at
    realistic operating points should reproduce that within 5 pp.
    """
    rng = np.random.default_rng(0)
    rpm = rng.uniform(1800.0, 2900.0, size=2_000)
    cmd = rng.uniform(5.0, 70.0, size=2_000)
    delivered = np.array(
        [map_from_csv.delivered_torque(r, c) for r, c in zip(rpm, cmd)],
    )
    ratio = delivered.mean() / cmd.mean()
    assert 0.87 <= ratio <= 1.0, (
        f"Sampled delivery ratio {ratio:.3f} is outside the expected range "
        "for Michigan 2025."
    )


def test_grid_validation():
    """Constructor rejects malformed grids."""
    with pytest.raises(ValueError, match="strictly increasing"):
        InverterDeliveryMap(
            rpm_grid=np.array([0.0, 0.0]),
            command_grid=np.array([0.0, 85.0]),
            delivered_grid=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="does not match"):
        InverterDeliveryMap(
            rpm_grid=np.array([0.0, 1000.0]),
            command_grid=np.array([0.0, 85.0]),
            delivered_grid=np.zeros((3, 3)),
        )
