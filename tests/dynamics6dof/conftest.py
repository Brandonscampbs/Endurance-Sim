from __future__ import annotations

import pytest

from fsae_sim.dynamics6dof.oracle import OracleUnavailable, load_oracle


@pytest.fixture(scope="session")
def oracle():
    try:
        ora = load_oracle()
    except OracleUnavailable as exc:
        pytest.skip(f"fastest-lap oracle unavailable: {exc}")
    ora.create_kart()
    yield ora
    ora.delete_vehicle()
