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
from PyQt5 import QtCore
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


class PlanePlacement(QtWidgets.QWidget):
    """Where the cutting plane sits: an offset, or a point it passes through.

    An offset along the normal is the natural handle for sweeping a plane
    through a map, and a hopeless one for landing on a feature whose H, K, L
    you already know. Both are here, kept in step: whichever is typed, the
    other is recomputed and shown, so the plane on screen always has both
    descriptions beside it.

    The widget holds no opinion about the data. The window gives it the
    centre of the map and the current normal through :meth:`refresh`, and
    asks for the plane back through :meth:`origin_and_offset`.
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, offset_step=0.005, parent=None):
        super().__init__(parent)
        self._updating = False
        self._centre = np.zeros(3)
        self._normal = np.array([0.0, 0.0, 1.0])

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.offset_spin = spin(0.0, -1e6, 1e6, 5, offset_step)
        self.offset_spin.setToolTip(
            "How far the plane sits from the centre of the map, along its\n"
            "normal. Ignored while the plane is placed by a point."
        )
        self.offset_spin.valueChanged.connect(self._offset_edited)
        layout.addRow("Offset along normal", self.offset_spin)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.point_spins = []
        for name in AXIS_NAMES:
            label = QtWidgets.QLabel(name)
            label.setToolTip(f"{name} of a point the plane runs through")
            box = spin(0.0, -1e6, 1e6, 4, offset_step, 62)
            box.setToolTip(
                f"{name} of a point the plane runs through.\n"
                "Typing here places the plane by the point."
            )
            box.valueChanged.connect(self._point_edited)
            row.addWidget(label)
            row.addWidget(box)
            self.point_spins.append(box)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        layout.addRow("Through point", holder)

        self.use_point_check = QtWidgets.QCheckBox("Place the plane by the point")
        self.use_point_check.setToolTip(
            "Put the plane through the H, K, L above instead of at an offset.\n"
            "The offset it works out to is shown either way."
        )
        self.use_point_check.toggled.connect(self._mode_changed)
        layout.addRow("", self.use_point_check)

        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setToolTip("The plane on screen, said both ways.")
        layout.addRow("", self.summary)

    # -- what the window asks for -----------------------------------------

    def uses_point(self) -> bool:
        return self.use_point_check.isChecked()

    def point(self) -> np.ndarray:
        return np.array([s.value() for s in self.point_spins], dtype=float)

    def offset(self) -> float:
        return float(self.offset_spin.value())

    def origin_and_offset(self, centre, normal):
        """``(origin, offset)`` of the plane, however it was placed."""
        from da6idb.hkl.slice import plane_origin

        if self.uses_point():
            return plane_origin(centre, normal, point=self.point())
        return plane_origin(centre, normal, offset=self.offset())

    def refresh(self, centre, normal):
        """Recompute whichever field the user is not driving, and say so.

        Called on every redraw, so moving the plane by its handle, by the
        normal, or by either box leaves the pair agreeing.
        """
        self._centre = np.asarray(centre, dtype=float)
        self._normal = np.asarray(normal, dtype=float)
        origin, offset = self.origin_and_offset(self._centre, self._normal)
        self._updating = True
        try:
            if self.uses_point():
                self.offset_spin.setValue(offset)
            else:
                for box, value in zip(self.point_spins, origin):
                    box.setValue(float(value))
        finally:
            self._updating = False
        self.summary.setText(
            "plane through  H %.4f  K %.4f  L %.4f\noffset %+.4f from the centre"
            % (origin[0], origin[1], origin[2], offset)
        )
        return origin, offset

    def set_offset(self, value):
        """Move the plane by offset, from a drag or a preset."""
        self._updating = True
        try:
            self.offset_spin.setValue(float(value))
            self.use_point_check.setChecked(False)
        finally:
            self._updating = False

    def set_step(self, step):
        self.offset_spin.setSingleStep(float(step))
        for box in self.point_spins:
            box.setSingleStep(float(step))

    def setEnabled(self, live):            # noqa: N802 -- Qt's own spelling
        super().setEnabled(bool(live))
        self.offset_spin.setEnabled(bool(live) and not self.uses_point())

    # -- internals ---------------------------------------------------------

    def _offset_edited(self):
        if not self._updating:
            self.changed.emit()

    def _point_edited(self):
        if self._updating:
            return
        # Typing an H, K, L is only ever meant one way.
        if not self.uses_point():
            self.use_point_check.setChecked(True)      # emits through _mode_changed
            return
        self.changed.emit()

    def _mode_changed(self, checked):
        self.offset_spin.setEnabled(self.isEnabled() and not checked)
        if not self._updating:
            self.changed.emit()
