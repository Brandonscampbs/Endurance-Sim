"""ctypes wrapper around fastest-lap's libfastestlapc-0.5.dll, used as a test oracle.

If the DLL is absent, load_oracle() raises OracleUnavailable and tests marked
@pytest.mark.oracle are skipped.
"""
from __future__ import annotations

import ctypes as _c
import os
import sys
from pathlib import Path
from typing import Sequence

_DEFAULT_DLL_PARENT = Path("C:/Users/brand/AppData/Local/Temp/fl-w10/v0.5")


class OracleUnavailable(RuntimeError):
    """Raised when the fastest-lap DLL cannot be loaded."""


def _locate_dll_root() -> Path:
    env = os.environ.get("FASTEST_LAP_ROOT")
    if env:
        return Path(env)
    return _DEFAULT_DLL_PARENT


def load_oracle() -> "Oracle":
    root = _locate_dll_root()
    bin_dir = root / "bin"
    include_dir = root / "include"
    if not bin_dir.exists() or not include_dir.exists():
        raise OracleUnavailable(f"fastest-lap DLL root not found at {root}")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(bin_dir))
    sys.path.insert(0, str(include_dir))
    try:
        import fastest_lap as _fl  # type: ignore
    except Exception as exc:
        raise OracleUnavailable(f"import fastest_lap failed: {exc}") from exc
    _fl.c_lib.vehicle_get_output.restype = _c.c_double
    _fl.set_print_level(0)
    return Oracle(_fl, root)


class Oracle:
    """Thin wrapper over fastest-lap's ctypes API for test use."""

    def __init__(self, fl_module, root: Path) -> None:
        self._fl = fl_module
        self.root = root

    @property
    def kart_xml(self) -> Path:
        return self.root / "database" / "vehicles" / "kart" / "roberto-lot-kart-2016.xml"

    def create_kart(self, name: str = "kart") -> None:
        self._fl.create_vehicle_from_xml(name, str(self.kart_xml))

    def delete_vehicle(self, name: str = "kart") -> None:
        try:
            self._fl.delete_vehicle(name)
        except Exception:
            pass

    def get_output(self, name: bytes, q: Sequence[float], u: Sequence[float], s: float, channel: str) -> float:
        q_arr = (_c.c_double * len(q))(*q)
        u_arr = (_c.c_double * len(u))(*u)
        return float(self._fl.c_lib.vehicle_get_output(name, q_arr, u_arr, _c.c_double(s), channel.encode()))
