"""Pacejka PAC2002 tire model for FSAE simulation.

Parses .tir files (PAC2002 format) and computes lateral force (Fy),
longitudinal force (Fx), combined forces via PAC2002 weighting functions,
peak force magnitudes, and loaded radius.

The lateral model uses the full Magic Formula with load, camber, and
scaling-factor support from TTC Round 8 Hoosier LC0 data.

The longitudinal model uses coefficients transplanted from TTC Round 6
Hoosier R25B (USE_MODE=4, Fx test data), scaled to match the LC0's
lateral grip envelope per-pressure.  Combined-slip uses PAC2002 weighting
functions (Gxa, Gyk, Svyk) from the R25B transplanted coefficients.
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

        NF-4: Non-numeric values that fail to parse are collected and
        emitted as a single ``UserWarning`` at end of parse so typos in
        coefficient files are not silently swallowed.  Required scalar
        keys (FNOMIN, UNLOADED_RADIUS, VERTICAL_STIFFNESS) are asserted
        positive.
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

        # NF-4: track keys whose values failed to parse as float
        skipped: list[tuple[str, str, str]] = []  # (section, key, raw_val)

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
                    # NF-4: record skipped non-numeric values for end-of-parse warning.
                    # Skip known-string keys (FILE_TYPE, TYRESIDE, UNITS) silently.
                    if key not in {
                        "FILE_TYPE",
                        "FILE_VERSION",
                        "FILE_FORMAT",
                        "TYRESIDE",
                        "LONGVL",
                        "LENGTH",
                        "FORCE",
                        "ANGLE",
                        "MASS",
                        "TIME",
                        "PROPERTY_FILE_FORMAT",
                        "TEST_CONDITIONS",
                        "USER_SUB_ID",
                        "COMMENT",
                    }:
                        skipped.append((section, key, val_str))

        # Assign scalar parameters
        self.fnomin = _vertical.get("FNOMIN", 0.0)
        self.vertical_stiffness = _vertical.get("VERTICAL_STIFFNESS", 0.0)
        self.unloaded_radius = _dimension.get("UNLOADED_RADIUS", 0.0)

        # Assign coefficient dicts
        self.lateral = _lateral
        self.longitudinal = _longitudinal
        self.scaling = _scaling
        self.loaded_radius_coeffs = _loaded_radius

        # NF-4: emit a single warning for all swallowed non-numeric values.
        if skipped:
            details = ", ".join(
                f"[{sec}] {key}={val!r}" for sec, key, val in skipped[:10]
            )
            warnings.warn(
                f"PacejkaTireModel: {len(skipped)} non-numeric coefficient "
                f"value(s) skipped while parsing {self.tir_path.name}: "
                f"{details}{' ...' if len(skipped) > 10 else ''}",
                UserWarning,
                stacklevel=2,
            )

        # NF-4: assert required scalar parameters are positive.  A missing
        # or zero value here would produce silent divide-by-zero or a
        # zero-radius tire in downstream physics; fail loud instead.
        assert self.fnomin > 0.0, (
            f"PacejkaTireModel({self.tir_path.name}): "
            f"FNOMIN must be > 0, got {self.fnomin}"
        )
        assert self.unloaded_radius > 0.0, (
            f"PacejkaTireModel({self.tir_path.name}): "
            f"UNLOADED_RADIUS must be > 0, got {self.unloaded_radius}"
        )
        assert self.vertical_stiffness > 0.0, (
            f"PacejkaTireModel({self.tir_path.name}): "
            f"VERTICAL_STIFFNESS must be > 0, got {self.vertical_stiffness}"
        )

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
        # NF-18: PAC2002 expects |PKY2| in the denominator.  Without abs(),
        # a negative PKY2 (the usual sign) flips the atan() and the
        # cornering-stiffness sign, breaking the lateral-force sign.
        denom_pky2 = abs(pky2_fz0)
        sin_arg = 2.0 * math.atan(fz / denom_pky2) if denom_pky2 > 1e-9 else 0.0
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

        # Curvature factor (ey), symmetrically clamped to [-1, 1] (NF-35)
        pey1 = self._lat("PEY1")
        pey2 = self._lat("PEY2")
        pey3 = self._lat("PEY3")
        pey4 = self._lat("PEY4")
        sign_a = 1.0 if alpha_star >= 0.0 else -1.0
        ey = (pey1 + pey2 * dfz) * (
            1.0 - (pey3 + pey4 * camber_rad) * sign_a
        ) * ley
        ey = max(-1.0, min(ey, 1.0))

        # Magic Formula: Fy = D * sin(C * atan(B*x - E*(B*x - atan(B*x)))) + SV
        bx = by * alpha_star
        inner = bx - ey * (bx - math.atan(bx))
        fy = muy * fz * math.sin(cy * math.atan(inner)) + svy

        return fy

    # ------------------------------------------------------------------
    # Longitudinal force (Fx) -- PAC2002 pure longitudinal slip
    # ------------------------------------------------------------------

    def longitudinal_force(
        self,
        slip_ratio: float,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> float:
        """Compute pure longitudinal force Fx using PAC2002 Magic Formula.

        Uses real longitudinal coefficients (PCX1, PDX1, PKX1, etc.)
        transplanted from TTC Round 6 R25B data and scaled to match
        the LC0's lateral grip envelope.

        When PDX1 is zero (TTC USE_MODE=2 lateral-only data), falls back
        to a symmetric mirror of the lateral peak-mu envelope so Fx
        remains physically reasonable rather than identically zero.

        Args:
            slip_ratio: Longitudinal slip ratio (-1..1). Positive = driving.
            normal_load_n: Normal load (N).
            camber_rad: Camber angle (rad).

        Returns:
            Longitudinal force Fx (N). Positive for driving, negative for braking.
        """
        fz = max(normal_load_n, 1.0)

        # Scaling factors
        lfzo = self._sc("LFZO")
        lcx = self._sc("LCX")
        lmux = self._sc("LMUX")
        lex = self._sc("LEX")
        lkx = self._sc("LKX")
        lhx = self._sc("LHX")
        lvx = self._sc("LVX")

        # Nominal load and normalized load increment
        fz0 = self.fnomin * lfzo
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        # Peak friction coefficient (mux)
        pdx1 = self._lon("PDX1")
        pdx2 = self._lon("PDX2")
        pdx3 = self._lon("PDX3")

        if pdx1 != 0.0:
            # Orthodox PAC2002 path -- use transplanted PDX/PKX/PCX data.
            mux = (pdx1 + pdx2 * dfz) * (1.0 - pdx3 * camber_rad ** 2) * lmux

            pkx1 = self._lon("PKX1")
            pkx2 = self._lon("PKX2")
            pkx3 = self._lon("PKX3")
            kxk = fz * (pkx1 + pkx2 * dfz) * math.exp(pkx3 * dfz) * lkx

            pcx1 = self._lon("PCX1")
            cx = pcx1 * lcx

            dx = mux * fz
            bx = kxk / (cx * dx + 1e-6)

            phx1 = self._lon("PHX1")
            phx2 = self._lon("PHX2")
            shx = (phx1 + phx2 * dfz) * lhx

            pvx1 = self._lon("PVX1")
            pvx2 = self._lon("PVX2")
            svx = fz * (pvx1 + pvx2 * dfz) * lvx * lmux

            kappa_x = slip_ratio + shx

            pex1 = self._lon("PEX1")
            pex2 = self._lon("PEX2")
            pex3 = self._lon("PEX3")
            pex4 = self._lon("PEX4")
            sign_k = 1.0 if kappa_x >= 0.0 else -1.0
            ex = (pex1 + pex2 * dfz + pex3 * dfz ** 2) * (1.0 - pex4 * sign_k) * lex
            # NF-35: symmetric clamp
            ex = max(-1.0, min(ex, 1.0))

            bk = bx * kappa_x
            inner = bk - ex * (bk - math.atan(bk))
            fx = dx * math.sin(cx * math.atan(inner)) + svx
            return fx

        # Fallback: TTC USE_MODE=2 -- mirror the lateral Magic Formula using
        # |PDY| for peak mu so Fx has the same grip envelope as Fy.  No
        # horizontal/vertical shift; symmetric about zero slip.
        pdy1 = self._lat("PDY1")
        pdy2 = self._lat("PDY2")
        pdy3 = self._lat("PDY3")
        mux = abs((pdy1 + pdy2 * dfz) * (1.0 - pdy3 * camber_rad ** 2)) * self._sc("LMUY")

        pky1 = self._lat("PKY1")
        pky2 = self._lat("PKY2")
        pky3 = self._lat("PKY3")
        pky1_fz0 = pky1 * fz0
        pky2_fz0 = pky2 * fz0
        # NF-18: divide-protected absolute denominator.
        denom_pky2 = abs(pky2_fz0)
        sin_arg = 2.0 * math.atan(fz / denom_pky2) if denom_pky2 > 1e-9 else 0.0
        kx = abs(
            pky1_fz0
            * math.sin(sin_arg)
            * (1.0 - pky3 * abs(camber_rad))
            * lfzo
            * self._sc("LKY")
        )

        pcy1 = self._lat("PCY1")
        cx = pcy1 * self._sc("LCY")
        denom = cx * mux * fz + 1e-6
        bx_coeff = kx / denom

        pey1 = self._lat("PEY1")
        pey2 = self._lat("PEY2")
        ex = (pey1 + pey2 * dfz) * self._sc("LEY")
        # NF-35: symmetric clamp
        ex = max(-1.0, min(ex, 1.0))

        bk = bx_coeff * slip_ratio
        inner = bk - ex * (bk - math.atan(bk))
        fx = mux * fz * math.sin(cx * math.atan(inner))
        return fx

    # ------------------------------------------------------------------
    # Combined forces (PAC2002 weighting functions)
    # ------------------------------------------------------------------

    def combined_forces(
        self,
        slip_angle_rad: float,
        slip_ratio: float,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> tuple[float, float]:
        """Compute combined Fx, Fy using PAC2002 weighting functions.

        Uses Gxa (slip-angle reduction of Fx) and Gyk (slip-ratio
        reduction of Fy) weighting functions from transplanted R25B
        combined-slip coefficients, plus kappa-induced side force Svyk.

        When combined-slip coefficients are zero (no data), the weighting
        functions evaluate to 1.0 and Svyk to 0.0, so pure-slip forces
        pass through unchanged.
        """
        fx0 = self.longitudinal_force(slip_ratio, normal_load_n, camber_rad)
        fy0 = self.lateral_force(slip_angle_rad, normal_load_n, camber_rad)

        fz = max(normal_load_n, 1.0)
        lfzo = self._sc("LFZO")
        fz0 = self.fnomin * lfzo
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        # === Gxa: slip-angle reduction of Fx ===
        rbx1 = self._lon("RBX1")
        rbx2 = self._lon("RBX2")
        rcx1 = self._lon("RCX1")
        rex1 = self._lon("REX1")
        rex2 = self._lon("REX2")
        rhx1 = self._lon("RHX1")
        lxal = self._sc("LXAL")

        bxa = rbx1 * math.cos(math.atan(rbx2 * slip_ratio)) * lxal
        cxa = rcx1
        exa = rex1 + rex2 * dfz
        # NF-35: symmetric clamp on curvature factor
        exa = max(-1.0, min(exa, 1.0))
        alpha_s = slip_angle_rad + rhx1

        bxa_as = bxa * alpha_s
        inner_xa = bxa_as - exa * (bxa_as - math.atan(bxa_as))
        gxa_num = math.cos(cxa * math.atan(inner_xa))

        bxa_sh = bxa * rhx1
        inner_xa0 = bxa_sh - exa * (bxa_sh - math.atan(bxa_sh))
        gxa_den = math.cos(cxa * math.atan(inner_xa0))

        gxa = gxa_num / gxa_den if abs(gxa_den) > 1e-9 else 1.0

        # === Gyk: slip-ratio reduction of Fy ===
        rby1 = self._lat("RBY1")
        rby2 = self._lat("RBY2")
        rby3 = self._lat("RBY3")
        rcy1 = self._lat("RCY1")
        rey1 = self._lat("REY1")
        rey2 = self._lat("REY2")
        rhy1 = self._lat("RHY1")
        rhy2 = self._lat("RHY2")
        lyka = self._sc("LYKA")

        byk = rby1 * math.cos(math.atan(rby2 * (slip_angle_rad - rby3))) * lyka
        cyk = rcy1
        eyk = rey1 + rey2 * dfz
        # NF-35: symmetric clamp
        eyk = max(-1.0, min(eyk, 1.0))
        kappa_s = slip_ratio + rhy1 + rhy2 * dfz

        byk_ks = byk * kappa_s
        inner_yk = byk_ks - eyk * (byk_ks - math.atan(byk_ks))
        gyk_num = math.cos(cyk * math.atan(inner_yk))

        sh_yk = rhy1 + rhy2 * dfz
        byk_sh = byk * sh_yk
        inner_yk0 = byk_sh - eyk * (byk_sh - math.atan(byk_sh))
        gyk_den = math.cos(cyk * math.atan(inner_yk0))

        gyk = gyk_num / gyk_den if abs(gyk_den) > 1e-9 else 1.0

        # === Svyk: kappa-induced side force ===
        rvy1 = self._lat("RVY1")
        rvy2 = self._lat("RVY2")
        rvy3 = self._lat("RVY3")
        rvy4 = self._lat("RVY4")
        rvy5 = self._lat("RVY5")
        rvy6 = self._lat("RVY6")
        lvyka = self._sc("LVYKA")
        lmuy = self._sc("LMUY")

        pdy1 = self._lat("PDY1")
        pdy2 = self._lat("PDY2")
        muy = abs(pdy1 + pdy2 * dfz) * lmuy

        dvyk = (
            muy * fz * (rvy1 + rvy2 * dfz + rvy3 * camber_rad)
            * math.cos(math.atan(rvy4 * slip_angle_rad))
        )
        svyk = dvyk * math.sin(rvy5 * math.atan(rvy6 * slip_ratio)) * lvyka

        fx = gxa * fx0
        fy = gyk * fy0 + svyk

        # Fallback: if the .tir has no combined-slip coefficients (RBX1 =
        # RBY1 = 0, LC0 USE_MODE=2 case), gxa and gyk both collapse to 1
        # so pure forces pass through unattenuated.  Apply a friction-
        # ellipse projection so combined slip still obeys the tire's
        # overall grip envelope.  This is a safety net, not a substitute
        # for proper weighting coefficients.
        if rbx1 == 0.0 and rby1 == 0.0:
            peak_fx = self.peak_longitudinal_force(normal_load_n, camber_rad)
            peak_fy = self.peak_lateral_force(normal_load_n, camber_rad)
            if peak_fx > 1e-9 and peak_fy > 1e-9:
                norm_resultant = math.sqrt(
                    (fx / peak_fx) ** 2 + (fy / peak_fy) ** 2
                )
                if norm_resultant > 1.0:
                    scale = 1.0 / norm_resultant
                    fx *= scale
                    fy *= scale

        return fx, fy

    # ------------------------------------------------------------------
    # Peak force computation (C4/M11 closed-form)
    # ------------------------------------------------------------------

    def peak_lateral_force(
        self,
        normal_load_n: float,
        camber_rad: float = 0.0,
    ) -> float:
        """Peak lateral force magnitude via PAC2002 closed-form |D_y|.

        D_y = mu_y * Fz is the Magic Formula peak factor.  At the peak
        slip angle, ``C_y * atan(inner)`` reaches pi/2 and sin() = 1, so
        the peak of the full MF curve is exactly |D_y|.  Replaces the
        prior ``minimize_scalar`` search that suffered sign-asymmetric
        local-max selection with nonzero SVy (C4/M11).
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
        Formula: the peak of ``D * sin(C * atan(inner))`` is |D|.  When
        transplanted PDX coefficients are present they are used; else
        the ``longitudinal_force`` lateral-mu mirror is used so the two
        functions stay self-consistent.
        """
        fz = max(normal_load_n, 1.0)
        fz0 = self.fnomin * self._sc("LFZO")
        dfz = (fz - fz0) / fz0 if fz0 > 0 else 0.0

        pdx1 = self._lon("PDX1")
        pdx2 = self._lon("PDX2")
        pdx3 = self._lon("PDX3")
        if pdx1 != 0.0:
            mux = (
                (pdx1 + pdx2 * dfz)
                * (1.0 - pdx3 * camber_rad ** 2)
                * self._sc("LMUX")
            )
        else:
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
