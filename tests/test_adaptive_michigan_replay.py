"""Smoke test for scripts/validate_adaptive.py (Sub-task D).

The full quantitative regression lives in the script's ``__main__``
where it can print native-unit + percent deltas and a PASS/FAIL banner.
Pytest's job here is only to confirm the validation harness imports
and constructs end-to-end without exception so CI flags refactors that
break the public API.

Skipped if the Michigan telemetry CSV is not present (e.g. dev
checkouts without the data directory).
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
TELEM = REPO / "Real-Car-Data-And-Stats" / "CleanedEndurance.csv"


@pytest.mark.skipif(
    not TELEM.exists(), reason="Michigan telemetry CSV not present",
)
def test_validate_adaptive_script_imports():
    """The script must import without side effects."""
    # ``run_validation`` is the public function; we don't call it here
    # because it runs a full 22-lap sim (~few seconds) and writes
    # results to disk. The full run lives in the script's ``__main__``.
    import scripts.validate_adaptive as mod
    assert callable(mod.run_validation)
    # Constants the script reports against — these are the audit bars.
    assert mod.REAL_TIME_S == pytest.approx(1608.75, abs=0.01)
    assert mod.LAP_TIME_BAR_FRAC == pytest.approx(0.01)
    assert mod.NET_AH_BAR_FRAC == pytest.approx(0.02)
    assert mod.NET_KWH_BAR_FRAC == pytest.approx(0.02)
