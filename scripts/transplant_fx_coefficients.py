"""Transplant longitudinal (Fx) and combined-slip Pacejka coefficients.

Copies Fx and combined-slip coefficients from TTC Round 6 Hoosier R25B
.tir files (USE_MODE=4, has Fx data) into TTC Round 8 Hoosier LC0 .tir
files (USE_MODE=2, Fy-only, all Fx coefficients currently zero).

Scaling strategy:
  - PDX1 (peak Fx mu): scaled by |LC0_PDY1| / |R25B_PDY1| to preserve
    the donor's Fx/Fy ratio applied to the LC0's lateral grip level.
  - PKX1 (slip stiffness): scaled by |LC0_PKY1| / |R25B_PKY1| to
    preserve the stiffness ratio.
  - All other Fx shape/curvature/shift coefficients: transplanted
    directly (no scaling) since they define normalized curve shapes.
  - Combined-slip coefficients (RBX*, RCX*, REX*, RHX*, RBY*, RCY*,
    REY*, RHY*, RVY*): transplanted directly.

Usage:
    python scripts/transplant_fx_coefficients.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

R25B_DIR = PROJECT_ROOT / "Real-Car-Data-And-Stats" / "Round_6_Hoosier_R25B_18x7p5_10_on_7in"
LC0_DIR = PROJECT_ROOT / "Real-Car-Data-And-Stats" / "Tire Models from TTC"

PRESSURES = [8, 10, 12, 14]

R25B_TEMPLATE = "Round_6_Hoosier_R25B_18x7p5_10_on_7in_{psi}psi_PAC02_UM4.tir"
LC0_TEMPLATE = "Round_8_Hoosier_LC0_16x7p5_10_on_8in_{psi}psi_PAC02_UM2.tir"

# ---------------------------------------------------------------------------
# Coefficient lists
# ---------------------------------------------------------------------------

# Coefficients transplanted with per-pressure scaling
SCALED_COEFFICIENTS = {
    "PDX1": "pdy_ratio",   # scaled by |LC0_PDY1| / |R25B_PDY1|
    "PKX1": "pky_ratio",   # scaled by |LC0_PKY1| / |R25B_PKY1|
}

# Fx coefficients transplanted directly (no scaling)
DIRECT_FX_COEFFICIENTS = [
    "PCX1",
    "PDX2", "PDX3",
    "PEX1", "PEX2", "PEX3", "PEX4",
    "PKX2", "PKX3",
    "PHX1", "PHX2",
    "PVX1", "PVX2",
]

# Combined-slip from LONGITUDINAL_COEFFICIENTS section
COMBINED_LONGITUDINAL = [
    "RBX1", "RBX2",
    "RCX1",
    "REX1", "REX2",
    "RHX1",
]

# Combined-slip from LATERAL_COEFFICIENTS section
COMBINED_LATERAL = [
    "RBY1", "RBY2", "RBY3",
    "RCY1",
    "REY1", "REY2",
    "RHY1", "RHY2",
    "RVY1", "RVY2", "RVY3", "RVY4", "RVY5", "RVY6",
]


# ---------------------------------------------------------------------------
# .tir file coefficient reader
# ---------------------------------------------------------------------------

def read_tir_coefficient(text: str, key: str) -> float:
    """Extract a single coefficient value from .tir file text.

    Args:
        text: Full text content of a .tir file.
        key: Coefficient name (e.g. "PDX1", "PKY1").

    Returns:
        The numeric value of the coefficient.

    Raises:
        ValueError: If the key is not found in the file.
    """
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s+=\s+([^\s$!]+)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Coefficient {key!r} not found in .tir file")
    return float(match.group(1))


# ---------------------------------------------------------------------------
# .tir file coefficient writer
# ---------------------------------------------------------------------------

def replace_tir_coefficient(text: str, key: str, new_value: float) -> str:
    """Replace a coefficient value in .tir file text, preserving formatting.

    Matches lines like:
        PCX1                     =  0                       $Shape factor ...
    and replaces only the numeric value between '=' and '$' (or end of line),
    preserving the key name, spacing around '=', and the trailing comment.

    Args:
        text: Full text content of a .tir file.
        key: Coefficient name to update.
        new_value: New numeric value to write.

    Returns:
        Updated text with the coefficient value replaced.

    Raises:
        ValueError: If the key is not found in the file.
    """
    # Pattern: key, flexible whitespace, '=', flexible whitespace,
    # old numeric value, flexible whitespace before '$' comment or end of line.
    # We capture: (prefix with key and '= '), (old value), (trailing comment).
    pattern = re.compile(
        rf"^(\s*{re.escape(key)}\s+=\s+)"  # group 1: key through '= '
        rf"([^\s$!]+)"                       # group 2: old numeric value
        rf"(\s+\$.*)?$",                     # group 3: trailing $comment (optional)
        re.MULTILINE,
    )

    # Format the new value: use enough precision to preserve the data,
    # matching the general style (up to 6 significant digits).
    formatted = _format_value(new_value)

    def replacer(m: re.Match) -> str:
        prefix = m.group(1)
        comment = m.group(3) or ""
        # Pad value to maintain alignment with comment
        # Original files align $comment at roughly column 50.
        # prefix is ~26 chars, value + spacing should fill to ~50.
        target_width = 50 - len(prefix)
        padded_value = formatted.ljust(target_width)
        return f"{prefix}{padded_value}{comment.lstrip()}"

    result, count = pattern.subn(replacer, text, count=1)
    if count == 0:
        raise ValueError(f"Coefficient {key!r} not found in .tir file for replacement")
    return result


def _format_value(value: float) -> str:
    """Format a coefficient value matching .tir file conventions.

    Uses general format with up to 6 significant digits.
    Avoids unnecessary trailing zeros for clean output.
    """
    if value == 0.0:
        return "0"
    # Use 'g' format for compact representation with 6 sig figs
    return f"{value:.6g}"


# ---------------------------------------------------------------------------
# Main transplant logic
# ---------------------------------------------------------------------------

def transplant_single_pressure(psi: int) -> dict[str, float]:
    """Transplant Fx + combined-slip coefficients for one pressure variant.

    Args:
        psi: Tire inflation pressure (8, 10, 12, or 14).

    Returns:
        Dict of transplanted coefficient names and their final values.
    """
    r25b_path = R25B_DIR / R25B_TEMPLATE.format(psi=psi)
    lc0_path = LC0_DIR / LC0_TEMPLATE.format(psi=psi)

    print(f"\n{'='*70}")
    print(f"  Pressure: {psi} psi")
    print(f"  Donor:  {r25b_path.name}")
    print(f"  Target: {lc0_path.name}")
    print(f"{'='*70}")

    # Read source files
    r25b_text = r25b_path.read_text(encoding="utf-8", errors="replace")
    lc0_text = lc0_path.read_text(encoding="utf-8", errors="replace")

    # Read scaling reference values from both files
    r25b_pdy1 = read_tir_coefficient(r25b_text, "PDY1")
    r25b_pky1 = read_tir_coefficient(r25b_text, "PKY1")
    lc0_pdy1 = read_tir_coefficient(lc0_text, "PDY1")
    lc0_pky1 = read_tir_coefficient(lc0_text, "PKY1")

    pdy_ratio = abs(lc0_pdy1) / abs(r25b_pdy1)
    pky_ratio = abs(lc0_pky1) / abs(r25b_pky1)

    print(f"\n  Scaling ratios:")
    print(f"    |LC0_PDY1| / |R25B_PDY1| = {abs(lc0_pdy1):.5f} / {abs(r25b_pdy1):.5f} = {pdy_ratio:.6f}")
    print(f"    |LC0_PKY1| / |R25B_PKY1| = {abs(lc0_pky1):.5f} / {abs(r25b_pky1):.5f} = {pky_ratio:.6f}")

    ratios = {"pdy_ratio": pdy_ratio, "pky_ratio": pky_ratio}
    transplanted: dict[str, float] = {}

    # 1. Scaled coefficients (PDX1, PKX1)
    print(f"\n  Scaled coefficients:")
    for coeff, ratio_key in SCALED_COEFFICIENTS.items():
        donor_val = read_tir_coefficient(r25b_text, coeff)
        scale = ratios[ratio_key]
        new_val = donor_val * scale
        lc0_text = replace_tir_coefficient(lc0_text, coeff, new_val)
        transplanted[coeff] = new_val
        print(f"    {coeff}: {donor_val:.6g} * {scale:.6f} = {new_val:.6g}")

    # 2. Direct Fx coefficients (no scaling)
    print(f"\n  Direct Fx coefficients:")
    for coeff in DIRECT_FX_COEFFICIENTS:
        donor_val = read_tir_coefficient(r25b_text, coeff)
        lc0_text = replace_tir_coefficient(lc0_text, coeff, donor_val)
        transplanted[coeff] = donor_val
        print(f"    {coeff}: {donor_val:.6g}")

    # 3. Combined-slip from LONGITUDINAL section
    print(f"\n  Combined-slip (longitudinal section):")
    for coeff in COMBINED_LONGITUDINAL:
        donor_val = read_tir_coefficient(r25b_text, coeff)
        lc0_text = replace_tir_coefficient(lc0_text, coeff, donor_val)
        transplanted[coeff] = donor_val
        print(f"    {coeff}: {donor_val:.6g}")

    # 4. Combined-slip from LATERAL section
    print(f"\n  Combined-slip (lateral section):")
    for coeff in COMBINED_LATERAL:
        donor_val = read_tir_coefficient(r25b_text, coeff)
        lc0_text = replace_tir_coefficient(lc0_text, coeff, donor_val)
        transplanted[coeff] = donor_val
        print(f"    {coeff}: {donor_val:.6g}")

    # Write updated LC0 file
    lc0_path.write_text(lc0_text, encoding="utf-8")
    print(f"\n  Written: {lc0_path}")

    return transplanted


def verify_parsing() -> None:
    """Verify all updated LC0 .tir files still parse correctly."""
    # Add project src to path for import
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from fsae_sim.vehicle.tire_model import PacejkaTireModel

    print(f"\n{'='*70}")
    print("  Verification: parsing updated .tir files")
    print(f"{'='*70}")

    for psi in PRESSURES:
        lc0_path = LC0_DIR / LC0_TEMPLATE.format(psi=psi)
        tire = PacejkaTireModel(str(lc0_path))
        pdx1 = tire.longitudinal.get("PDX1", 0.0)
        pkx1 = tire.longitudinal.get("PKX1", 0.0)
        rbx1 = tire.longitudinal.get("RBX1", 0.0)
        rby1 = tire.lateral.get("RBY1", 0.0)
        print(
            f"  {psi} psi: PDX1={pdx1:.4f}, PKX1={pkx1:.4f}, "
            f"RBX1={rbx1:.4f}, RBY1={rby1:.4f}  -- OK"
        )


def main() -> None:
    """Run the full transplant pipeline."""
    print("Transplanting R25B Fx + combined-slip coefficients into LC0 .tir files")
    print(f"Donor directory:  {R25B_DIR}")
    print(f"Target directory: {LC0_DIR}")

    all_results: dict[int, dict[str, float]] = {}
    for psi in PRESSURES:
        all_results[psi] = transplant_single_pressure(psi)

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary: key transplanted values")
    print(f"{'='*70}")
    print(f"  {'Pressure':>8}  {'PDX1':>10}  {'PKX1':>10}  {'RBX1':>10}  {'RBY1':>10}")
    print(f"  {'-'*8:>8}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}")
    for psi in PRESSURES:
        r = all_results[psi]
        print(
            f"  {psi:>5} psi  {r['PDX1']:>10.4f}  {r['PKX1']:>10.4f}  "
            f"{r['RBX1']:>10.4f}  {r['RBY1']:>10.4f}"
        )

    # Verify files still parse
    verify_parsing()

    print("\nDone. All 4 LC0 .tir files updated and verified.")


if __name__ == "__main__":
    main()
