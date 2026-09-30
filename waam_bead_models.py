"""
waam_bead_models.py
====================
Bead geometry models derived from DoE experimental data.

Two databases:
  - Single-layer (SL): bead width on substrate / edge-compensation passes
  - Multilayer  (ML): steady-state bead height and width for main deposition

Working speed range (adaptive slicing): 3.0 – 8.0 mm/sec
  Corresponds to bead height range:     2.042 – 2.565 mm

Usage
-----
    from waam_bead_models import BeadModel
    bm = BeadModel()

    # Adaptive layer height -> required travel speed
    speed = bm.speed_for_height(2.3)       # mm/sec

    # Gap compensation -> required travel speed (single-layer pass)
    speed = bm.speed_for_width_sl(6.0)     # mm/sec

    # Gap compensation -> required travel speed (multilayer pass)
    speed = bm.speed_for_width_ml(6.5)     # mm/sec

    # Forward prediction
    h = bm.height_at_speed(5.5)            # mm
    w = bm.width_sl_at_speed(5.5)          # mm
    w = bm.width_ml_at_speed(5.5)          # mm
"""

import numpy as np
from numpy import interp

# ── Raw model tables (speed ascending) ───────────────────────────
_SPEEDS = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])

_HEIGHT_ML  = np.array([2.565, 2.521, 2.514, 2.386, 2.291, 2.262, 2.233, 2.219, 2.106, 2.042, 2.042])  # mm, multilayer steady-state
_WIDTH_SL   = np.array([7.345, 6.155, 6.134, 5.490, 5.243, 4.855, 4.733, 4.716, 4.708, 4.379, 4.379])  # mm, single-layer
_WIDTH_ML   = np.array([8.689, 7.281, 7.257, 6.495, 6.203, 5.743, 5.600, 5.579, 5.570, 5.180, 5.180])  # mm, multilayer (SL x 1.183)

# ── Physical bounds ───────────────────────────────────────────────
SPEED_MIN   = 3.0   # mm/sec  (thick layers)
SPEED_MAX   = 8.0   # mm/sec  (thin layers)
HEIGHT_MIN  = 2.042  # mm  (at SPEED_MAX)
HEIGHT_MAX  = 2.565  # mm  (at SPEED_MIN)
WIDTH_ML_MIN = float(_WIDTH_ML[-1])   # mm (narrowest, at SPEED_MAX)
WIDTH_ML_MAX = float(_WIDTH_ML[0])    # mm (widest, at SPEED_MIN)
WIDTH_SL_MIN = float(_WIDTH_SL[-1])
WIDTH_SL_MAX = float(_WIDTH_SL[0])


class BeadModel:
    """Invertible bead geometry model for WAAM adaptive slicing."""

    # ── Forward: speed -> geometry ────────────────────────────────
    def height_at_speed(self, speed_mms: float) -> float:
        """Predicted steady-state bead height (mm) at given travel speed."""
        s = np.clip(speed_mms, SPEED_MIN, SPEED_MAX)
        return float(interp(s, _SPEEDS, _HEIGHT_ML))

    def width_sl_at_speed(self, speed_mms: float) -> float:
        """Predicted single-layer bead width (mm) at given travel speed."""
        s = np.clip(speed_mms, SPEED_MIN, SPEED_MAX)
        return float(interp(s, _SPEEDS, _WIDTH_SL))

    def width_ml_at_speed(self, speed_mms: float) -> float:
        """Predicted multilayer bead width (mm) at given travel speed."""
        s = np.clip(speed_mms, SPEED_MIN, SPEED_MAX)
        return float(interp(s, _SPEEDS, _WIDTH_ML))

    # ── Inverse: geometry -> speed ────────────────────────────────
    def speed_for_height(self, target_height_mm: float) -> float:
        """
        Travel speed (mm/sec) required to achieve target bead height.
        Input is clipped to [HEIGHT_MIN, HEIGHT_MAX].
        Height decreases with speed, so inversion uses reversed arrays.
        """
        t = np.clip(target_height_mm, HEIGHT_MIN, HEIGHT_MAX)
        return float(interp(t, _HEIGHT_ML[::-1], _SPEEDS[::-1]))

    def speed_for_width_sl(self, target_width_mm: float) -> float:
        """Travel speed (mm/sec) for target single-layer bead width."""
        t = np.clip(target_width_mm, WIDTH_SL_MIN, WIDTH_SL_MAX)
        return float(interp(t, _WIDTH_SL[::-1], _SPEEDS[::-1]))

    def speed_for_width_ml(self, target_width_mm: float) -> float:
        """Travel speed (mm/sec) for target multilayer bead width."""
        t = np.clip(target_width_mm, WIDTH_ML_MIN, WIDTH_ML_MAX)
        return float(interp(t, _WIDTH_ML[::-1], _SPEEDS[::-1]))

    def __repr__(self):
        return (
            f"BeadModel(height {HEIGHT_MIN:.2f}–{HEIGHT_MAX:.2f} mm, "
            f"speed {SPEED_MIN}–{SPEED_MAX} mm/sec)"
        )
