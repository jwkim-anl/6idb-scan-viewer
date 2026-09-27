"""Shared look, and the two helpers, for the 3D windows.

``hkl_viewer.py`` and ``vti_viewer.py`` present the same scene -- a volume on a
black background with H/K/L axes, a yellow slice plane and one scalar bar -- so
the colours, the number format and the spin box live here instead of being kept
in step by hand.

Only PyQt5 and numpy are imported. A finished ``.vti`` needs neither
xrayutilities nor h5py to look at, and this module must not drag them in.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtWidgets

COLORMAPS = ("viridis", "plasma", "inferno", "magma", "turbo", "jet", "gray")
AXIS_NAMES = ("H", "K", "L")
BACKGROUND = "black"
AXIS_COLOR = "white"
# The plane widget draws in one colour; the default is far too dark on black.
SLICE_COLOR = "yellow"
# pyvista takes the scalar bar's text colour from the theme font, which is
# black -- the numbers vanish against the background unless it is set here.
SCALAR_BAR = {"color": AXIS_COLOR, "title": "intensity"}
# HKL is a small number that matters in its third decimal -- 5.321, not 5.3.
AXIS_NUMBER_FORMAT = "%.3f"


def spin(value, low, high, decimals=4, step=1.0, width=110):
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setFixedWidth(width)
    box.setKeyboardTracking(False)
    box.setValue(value)
    return box


def make_drawable(grid, fill=0.0):
    """Replace NaN in a loaded grid so a volume mapper can draw it.

    NaN marks cells the scan never reached. Handed to a GPU volume mapper it
    does not come out transparent -- it lights the whole empty region up -- so
    it has to become a value at the bottom of the colour scale, where the
    opacity ramp starts at zero.

    Edits the array in place, which on a large grid saves copying a few hundred
    megabytes, and returns ``(low, high)`` of the measured cells read before the
    NaN goes. The file on disk still holds the NaN.
    """
    values = grid.point_data["intensity"]
    if not np.isfinite(values).any():
        return (0.0, 1.0)
    span = (float(np.nanmin(values)), float(np.nanmax(values)))
    np.nan_to_num(values, copy=False, nan=fill, posinf=fill, neginf=fill)
    return span
