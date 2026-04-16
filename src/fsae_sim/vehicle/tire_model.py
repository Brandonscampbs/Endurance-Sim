"""Pacejka PAC2002 tire model for FSAE simulation.

Parses .tir files (PAC2002 format) and computes lateral force (Fy),
longitudinal force (Fx), combined forces via friction-circle coupling,
peak force magnitudes, and loaded radius.

The lateral model uses the full Magic Formula with load, camber, and
scaling-factor support.  Because the .tir files from the TTC Hoosier
LC0 dataset have all longitudinal coefficients zeroed (USE_MODE=2,
lateral-only test data), the longitudinal model mirrors the lateral
structure using |PDY1| for peak mu so that Fx has the same grip
envelope as Fy.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Union


class PacejkaTireModel:
    """PAC2002 Pacejka tire model loaded from a .tir file.

    Args:
        tir_path: Path to a PAC2002 .tir parameter file.

    Attributes:
        fnomin: Nominal vertical load (N).
        unloaded_radius: Free tyre radius (m).
        vertical_stiffness: Tyre vertical stiffness (N/m).
        lateral: Dict of lateral-force coefficients (PCY1, PDY1, ...).
        longitudinal: Dict of longitudinal-force coefficients (PCX1, ...).
        scaling: Dict of scaling factors (LFZO, LCX, LMUY, ...).
        loaded_radius_coeffs: Dict of loaded-radius coefficients (QV1, ...).
    """

    def __init__(self, tir_path: Union[str, Path]) -> None:
        self.tir_path = Path(tir_path)
        self.fnomin: float = 0.0
        self.unloaded_radius: float = 0.0
        self.vertical_stiffness: float = 0.0
        self.lateral: dict[str, float] = {}
        self.longitudinal: dict[str, float] = {}
        self.scaling: dict[str, float] = {}
        self.loaded_radius_coeffs: dict[str, float] = {}
        self._parse()

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        """Parse the .tir file into coefficient dictionaries.

        PAC2002 format rules:
        - Sections delimited by ``[SECTION_NAME]``.
        - Key-value lines: ``KEY = value $comment`` or ``KEY = value``.
        - Comment lines start with ``!`` or ``$``.
        - Values may be numeric or quoted strings (we only keep numerics).
        """
        text = self.tir_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        section = ""
        section_map = {
            "VERTICAL": "_vertical",
            "DIMENSION": "_dimension",
            "LATERAL_COEFFICIENTS": "_lateral",
            "LONGITUDINAL_COEFFICIENTS": "_longitudinal",
            "SCALING_COEFFICIENTS": "_scaling",
            "LOADED_RADIUS_COEFFICIENTS": "_loaded_radius",
        }

        # Temp storage for vertical and dimension sections
        _vertical: dict[str, float] = {}
        _dimension: dict[str, float] = {}
        _lateral: dict[str, float] = {}
        _longitudinal: dict[str, float] = {}
        _scaling: dict[str, float] = {}
        _loaded_radius: dict[str, float] = {}

        targets = {
            "_vertical": _vertical,
            "_dimension": _dimension,
            "_lateral": _lateral,
            "_longitudinal": _longitudinal,
            "_scaling": _scaling,
            "_loaded_radius": _loaded_radius,
        }

        kv_pattern = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s$!]+)"
        )

        for line in lines:
            stripped = line.strip()

            # Section header
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1]
                continue

            # Skip pure comments and empty lines
            if not stripped or stripped.startswith("!") or stripped.startswith("$"):
                continue

            # Skip shape table data lines (numeric-only, no key)
            if stripped[0].isdigit() or stripped[0] == "-":
                continue

            target_key = section_map.get(section)
            if target_key is None:
                continue

            match = kv_pattern.match(stripped)
            if match:
                key = match.group(1).upper()
                val_str = match.group(2).strip("'\"")
                try:
                    value = float(val_str)
                    targets[target_key][key] = value
                except ValueError:
                    pass  # skip non-numeric values

        # Assign scalar parameters
        self.fnomin = _vertical.get("FNOMIN", 0.0)
        self.vertical_stiffness = _vertical.get("VERTICAL_STIFFNESS", 0.0)
        self.unloaded_radius = _dimension.get("UNLOADED_RADIUS", 0.0)

        # Assign coefficient dicts
        self.lateral = _lateral
        self.longitudinal = _longitudinal
        self.scaling = _scaling
        self.loaded_radius_coeffs = _loaded_radius

    # ------------------------------------------------------------------
    # Helper: coefficient lookup with default
    # ------------------------------------------------------------------

    def _lat(self, key: str, default: float = 0.0) -> float:
        """Look up a lateral coefficient."""
        return self.lateral.get(key, default)

    def _lon(self, key: str, default: float = 0.0) -> float:
        """Look up a longitudinal coefficient."""
        return self.longitudinal.get(key, default)

    def _sc(self, key: str, default: float = 1.0) -> float:
        """Look up a scaling factor (defaults to 1.0)."""
        return self.scaling.get(key, default)

    def apply_grip_scale(self, scale: float) -> None:
        """Scale tire grip by multiplying LMUY and LMUX scaling factors.

        This is the standard Pacejka mechanism for calibrating TTC rig data
        to on-car grip. Scales peak force (D = mu * Fz) while preserving
        cornering stiffness (B compensates since B = Kya / (C * D)).

        Args:
            scale: Grip multiplier. 1.0 = no change, 0.5 = halve peak grip.
                Values should be positive.
        """
        self.scaling["LMUY"] = self.scaling.get("LMUY", 1.0) * scale
        self.scaling["LMUX"] = self.scaling.get("LMUX", 1.0) * scale

    # ------------------------------------------------------------------
    # Lateral force (Fy) -- PAC2002 pure side slip
    # ------------------------------------------------------------------

    def lateral_force(
        self,
        slip_angle_rad: float,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> float:
        """Compute pure lateral force Fy using PAC2002 Magic Formula.

        Args:
            slip_angle_rad: Tyre slip angle (rad). Positive = rightward slip.
            normal_load_n: Normal (vertical) load on the tyre (N). Must be > 0.
            camber_rad: Inclination (camber) angle (rad).

        Returns:
            Lateral force Fy (N). Sign follows the Pacejka convention:
            negative Fy for positive slip angle with the Hoosier LC0 fit
            (PDY1 < 0).
        """
        fz = max(normal_load_n, 1.0)  # guard against zero/negative load

        # Scaling factors
        lfzo = self._sc("LFZO")
        lcy = self._sc("LCY")
        lmuy = self._sc("LMUY")
        ley = self._sc("LEY")
        lky = self._sc("LKY")
        lhy = self._sc("LHY")
        lvy = self._sc("LVY")
        lkyg = self._sc("LKYG")

        # Nominal load and load increment
        fz0 = self.fnomin * lfzo
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        # Peak friction coefficient (muy)
        pdy1 = self._lat("PDY1")
        pdy2 = self._lat("PDY2")
        pdy3 = self._lat("PDY3")
        muy = (pdy1 + pdy2 * dfz) * (1.0 - pdy3 * camber_rad ** 2) * lmuy

        # Cornering stiffness (kya)
        pky1 = self._lat("PKY1")
        pky2 = self._lat("PKY2")
        pky3 = self._lat("PKY3")

        pky1_fz0 = pky1 * fz0
        pky2_fz0 = pky2 * fz0
        sin_arg = 2.0 * math.atan(fz / pky2_fz0) if abs(pky2_fz0) > 1e-9 else 0.0
        kya = (
            pky1_fz0
            * math.sin(sin_arg)
            * (1.0 - pky3 * abs(camber_rad))
            * lfzo
            * lky
        )

        # Shape factor (cy) and stiffness factor (by)
        pcy1 = self._lat("PCY1")
        cy = pcy1 * lcy
        denom = cy * muy * fz + 1e-6
        by = kya / denom

        # Horizontal shift (shy)
        phy1 = self._lat("PHY1")
        phy2 = self._lat("PHY2")
        phy3 = self._lat("PHY3")
        shy = (phy1 + phy2 * dfz) * lhy + phy3 * camber_rad * lkyg

        # Vertical shift (svy)
        pvy1 = self._lat("PVY1")
        pvy2 = self._lat("PVY2")
        pvy3 = self._lat("PVY3")
        pvy4 = self._lat("PVY4")
        svy = fz * (
            (pvy1 + pvy2 * dfz) * lvy
            + (pvy3 + pvy4 * dfz) * camber_rad
        ) * lmuy

        # Shifted slip angle
        alpha_star = slip_angle_rad + shy

        # Curvature factor (ey), clamped <= 1.0
        pey1 = self._lat("PEY1")
        pey2 = self._lat("PEY2")
        pey3 = self._lat("PEY3")
        pey4 = self._lat("PEY4")
        sign_a = 1.0 if alpha_star >= 0.0 else -1.0
        ey = (pey1 + pey2 * dfz) * (
            1.0 - (pey3 + pey4 * camber_rad) * sign_a
        ) * ley
        ey = min(ey, 1.0)

        # Magic Formula: Fy = D * sin(C * atan(B*x - E*(B*x - atan(B*x)))) + SV
        bx = by * alpha_star
        inner = bx - ey * (bx - math.atan(bx))
        fy = muy * fz * math.sin(cy * math.atan(inner)) + svy

        return fy

    # ------------------------------------------------------------------
    # Longitudinal force (Fx) -- symmetric mirror of lateral model
    # ------------------------------------------------------------------

    def longitudinal_force(
        self,
        slip_ratio: float,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> float:
        """Compute pure longitudinal force Fx.

        Since the .tir files have all Fx coefficients zeroed (USE_MODE=2,
        lateral-only TTC data), this uses a symmetric mirror of the lateral
        structure:
        - Peak mu (mux) = |PDY1 + PDY2*dfz|  (positive, symmetric)
        - Cornering stiffness magnitude from |kya|
        - No horizontal or vertical shift (symmetric about zero slip)
        - Same curvature factor structure

        Args:
            slip_ratio: Longitudinal slip ratio (-1..1). Positive = driving.
            normal_load_n: Normal load (N).
            camber_rad: Camber angle (rad).

        Returns:
            Longitudinal force Fx (N). Positive for positive slip ratio
            (driving force).
        """
        fz = max(normal_load_n, 1.0)

        lfzo = self._sc("LFZO")
        lmuy = self._sc("LMUY")  # reuse lateral scaling for mirrored model
        lcy = self._sc("LCY")
        ley = self._sc("LEY")
        lky = self._sc("LKY")

        fz0 = self.fnomin * lfzo
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        # Peak friction -- use absolute value of lateral muy (symmetric)
        pdy1 = self._lat("PDY1")
        pdy2 = self._lat("PDY2")
        pdy3 = self._lat("PDY3")
        mux = abs((pdy1 + pdy2 * dfz) * (1.0 - pdy3 * camber_rad ** 2)) * lmuy

        # Cornering stiffness magnitude
        pky1 = self._lat("PKY1")
        pky2 = self._lat("PKY2")
        pky3 = self._lat("PKY3")
        pky1_fz0 = pky1 * fz0
        pky2_fz0 = pky2 * fz0
        sin_arg = 2.0 * math.atan(fz / pky2_fz0) if abs(pky2_fz0) > 1e-9 else 0.0
        kx = abs(
            pky1_fz0
            * math.sin(sin_arg)
            * (1.0 - pky3 * abs(camber_rad))
            * lfzo
            * lky
        )

        # Shape and stiffness factors
        pcy1 = self._lat("PCY1")
        cx = pcy1 * lcy
        denom = cx * mux * fz + 1e-6
        bx_coeff = kx / denom

        # Curvature factor (symmetric: no sign dependency)
        pey1 = self._lat("PEY1")
        pey2 = self._lat("PEY2")
        ex = (pey1 + pey2 * dfz) * ley
        ex = min(ex, 1.0)

        # Magic Formula
        bk = bx_coeff * slip_ratio
        inner = bk - ex * (bk - math.atan(bk))
        fx = mux * fz * math.sin(cx * math.atan(inner))

        return fx

    # ------------------------------------------------------------------
    # Combined forces (friction-circle coupling)
    # ------------------------------------------------------------------

    def combined_forces(
        self,
        slip_angle_rad: float,
        slip_ratio: float,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> tuple[float, float]:
        """Compute combined Fx, Fy with friction-circle coupling.

        When the resultant of pure-slip Fx0 and Fy0 exceeds the tyre's
        peak force (friction circle), both forces are scaled down
        proportionally so the resultant lies on the circle.

        Args:
            slip_angle_rad: Slip angle (rad).
            slip_ratio: Longitudinal slip ratio.
            normal_load_n: Normal load (N).
            camber_rad: Camber angle (rad).

        Returns:
            Tuple of (Fx, Fy) in Newtons.
        """
        fx0 = self.longitudinal_force(slip_ratio, normal_load_n, camber_rad)
        fy0 = self.lateral_force(slip_angle_rad, normal_load_n, camber_rad)

        resultant = math.sqrt(fx0 ** 2 + fy0 ** 2)
        if resultant < 1e-9:
            return fx0, fy0

        # Peak force defines the friction circle radius
        peak_fx = self.peak_longitudinal_force(normal_load_n, camber_rad)
        peak_fy = self.peak_lateral_force(normal_load_n, camber_rad)

        # Elliptical friction circle: scale relative to each axis's peak
        if peak_fx < 1e-9 or peak_fy < 1e-9:
            return fx0, fy0

        # Normalized resultant on the friction ellipse
        norm_x = fx0 / peak_fx
        norm_y = fy0 / peak_fy
        norm_resultant = math.sqrt(norm_x ** 2 + norm_y ** 2)

        if norm_resultant <= 1.0:
            # Within the friction ellipse -- no scaling needed
            return fx0, fy0

        # Scale both forces to sit on the ellipse boundary
        scale = 1.0 / norm_resultant
        return fx0 * scale, fy0 * scale

    # ------------------------------------------------------------------
    # Peak force computation
    # ------------------------------------------------------------------

    def peak_lateral_force(
        self,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> float:
        """Peak lateral force magnitude via PAC2002 closed-form |D_y|.

        D_y = mu_y * Fz is the Magic Formula peak factor.  At the peak
        slip angle, ``C_y * atan(inner)`` reaches pi/2 and sin() = 1, so
        the peak of the full MF curve is exactly |D_y|.  Replacing the
        prior ``minimize_scalar`` search avoids the sign-asymmetric
        local-max selection the bounded Brent method suffered with
        nonzero SVy (see SIMULATOR_AUDIT_2026-04-16 C4/M11).
        """
        fz = max(normal_load_n, 1.0)
        fz0 = self.fnomin * self._sc("LFZO")
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        pdy1 = self._lat("PDY1")
        pdy2 = self._lat("PDY2")
        pdy3 = self._lat("PDY3")
        muy = (pdy1 + pdy2 * dfz) * (1.0 - pdy3 * camber_rad ** 2) * self._sc("LMUY")
        return abs(muy * fz)

    def peak_longitudinal_force(
        self,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> float:
        """Peak longitudinal force magnitude via PAC2002 closed-form |D_x|.

        Mirrors ``peak_lateral_force`` for the longitudinal Magic
        Formula: the peak of ``D * sin(C * atan(inner))`` is |D|.

        The current TTC .tir files have ``USE_MODE=2`` (lateral-only),
        so all PDX coefficients are zero.  ``longitudinal_force`` mirrors
        the lateral peak-mu expression ``|(PDY1 + PDY2*dfz)| * LMUY``,
        and this peak must match that same mu exactly to keep the two
        functions self-consistent.  If a PAC2002 parameter set with
        nonzero PDX1 is loaded (e.g., via ``transplant_fx_coefficients``),
        the orthodox PDX form is used instead.
        """
        fz = max(normal_load_n, 1.0)
        fz0 = self.fnomin * self._sc("LFZO")
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        pdx1 = self._lon("PDX1")
        pdx2 = self._lon("PDX2")
        pdx3 = self._lon("PDX3")
        if pdx1 != 0.0:
            # Orthodox PAC2002: use longitudinal coefficients and LMUX.
            mux = (
                (pdx1 + pdx2 * dfz)
                * (1.0 - pdx3 * camber_rad ** 2)
                * self._sc("LMUX")
            )
        else:
            # TTC USE_MODE=2 mirror: match the mu used inside
            # ``longitudinal_force``: |PDY1 + PDY2*dfz| * (1 - PDY3*gamma^2) * LMUY.
            pdy1 = self._lat("PDY1")
            pdy2 = self._lat("PDY2")
            pdy3 = self._lat("PDY3")
            mux = (
                abs((pdy1 + pdy2 * dfz) * (1.0 - pdy3 * camber_rad ** 2))
                * self._sc("LMUY")
            )
        return abs(mux * fz)

    # ------------------------------------------------------------------
    # Loaded radius
    # ------------------------------------------------------------------

    def loaded_radius(
        self,
        normal_load_n: float,
        speed_ms: float = 0.0,
    ) -> float:
        """Compute tyre loaded radius under vertical load.

        Uses a simple spring deflection model:
            r_loaded = r0 - Fz / kz

        where r0 is the unloaded radius and kz is vertical stiffness.
        Clamped to a minimum of 0.01 m.

        Args:
            normal_load_n: Normal load (N). Use 0 for free radius.
            speed_ms: Forward speed (m/s). Reserved for future centrifugal
                growth; currently unused beyond clamping.

        Returns:
            Loaded radius (m).
        """
        r0 = self.unloaded_radius
        kz = self.vertical_stiffness
        fz = max(normal_load_n, 0.0)

        if kz <= 0.0:
            return r0

        r_loaded = r0 - fz / kz
        return max(r_loaded, 0.01)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PacejkaTireModel(tir='{self.tir_path.name}', "
            f"Fz0={self.fnomin:.0f}N, R0={self.unloaded_radius:.4f}m, "
            f"kz={self.vertical_stiffness:.0f}N/m)"
        )
