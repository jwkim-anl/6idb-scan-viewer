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
BACKGROUNDS = ("black", "white")


def theme(background: str) -> dict:
    """The colours that go with a background.

    Nothing here can be left to pyvista: it takes every label's colour from
    its own theme font, which is black, so on a black background the axis
    letters, the tick numbers and the scalar bar all vanish unless each is
    told otherwise. The slice handle needs the same care in reverse -- yellow
    reads well on black and disappears on white.
    """
    if background not in BACKGROUNDS:
        background = BACKGROUNDS[0]
    dark = background == "black"
    axis = "white" if dark else "black"
    return {
        "background": background,
        "axis": axis,
        "slice": "yellow" if dark else "#c1121f",
        "scalar_bar": {"color": axis, "title": "intensity"},
    }
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
