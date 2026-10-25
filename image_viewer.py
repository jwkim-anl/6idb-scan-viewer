"""A 2D image-stack viewer for the ``pva*.h5`` files that accompany a master file.

``MyNXWriter`` puts the scan's scalar channels in ``*_master.hdf`` but leaves the
area-detector frames out of it -- ``.../streams/primary/lambda250k`` carries only
``EPOCH``/``time`` and no ``value``.  The pixels are written separately by the
pvapy streaming consumer into a sibling file, and the master records its name in
``entry/instrument/bluesky/metadata/pva_streaming``::

    file_name: pva_tise2_2_00006.h5
    file_path: ~/6idb-bits/<experiment directory>

``file_path`` is the beamline's path, so :func:`image_path_for_master` resolves
``file_name`` against the master's own directory and falls back to deriving
``pva_<stem without "_master">.h5`` when the metadata is missing.

The image file's layout (all paths are hard-coded here, matching the writer)::

    entry/data/data                (N, nx, ny)  the stack; ``pixel_order='xy'``,
                                   i.e. ``frame[x, y]`` -- the same convention as
                                   pyqtgraph's default col-major ImageItem
    entry/data/frame_timestamp     (N,)         epoch seconds
    entry/data/rois/<name>         (N, w, h)    pre-baked ROIs; the rectangle is in
                                   the ``x``/``y``/``w``/``h`` attributes
    entry/data/metadata/ca/<pv>    (N,)         one value per frame (eta, delta, cap, ...)

The frame count need not match the master scan's point count (the detector
usually starts a little early), so this window is deliberately self-contained:
every per-frame quantity it plots comes from the image file itself.

``entry/data/data`` is blosc-compressed, which needs ``hdf5plugin`` importable.
It is imported here if present; if it is not and the pixel read fails,
:class:`ImageData` raises a :class:`RuntimeError` that names the missing package.

Where the matching ``*_master.hdf`` sits beside the image file, its primary
stream is offered too, under a ``scan/`` prefix -- each frame taking the scan
point nearest it in time, since the detector free-runs and the two files rarely
have the same number of points.

Both ROI plots can **normalize** by any of the same quantities their axes offer
(a CA channel such as a monitor, or another ROI metric): the value axes are
divided by it point by point, after any background subtraction.

Any ROI can be marked as the **background**: its mean intensity per pixel is then
subtracted from every other ROI's sum (as ``background x pixel count``) and from
their mean/min/max, per frame and per file.

An **HKL 3D** button maps the frames into reciprocal space, taking the geometry
from the matching master file (see ``hkl_viewer.py``); it needs ``pyvista`` and
``xrayutilities``, and says so plainly when they are absent.

Several files can be opened at once: they are shown side by side in one window,
sharing a single set of ROIs, one colour scale and one frame control, so scans can
be compared directly. The frame control spans the longest file; a file that has run
out of frames keeps showing its last one.

Run standalone on one or more image files with::

    python image_viewer.py pva_tise2_2_00006.h5 [pva_tise2_2_00007.h5 ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

try:  # Registers the blosc/lz4/bitshuffle filters that entry/data/data needs.
    import hdf5plugin  # noqa: F401
    HAVE_HDF5PLUGIN = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_HDF5PLUGIN = False

import h5py
import pyqtgraph as pg

from da6idb.images import ImageData, MAX_TOTAL_BYTES
from da6idb.rois import Rect, series

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import viewer_state





# -- Locating the image file ------------------------------------------------






































# -- Reading the image file -------------------------------------------------




# -- Metrics ----------------------------------------------------------------

METRIC_KEYS = ["frame", "time", "sum", "mean", "min", "max", "comx", "comy"]
SINGLE_FRAME_KEYS = ["proj_x", "proj_y"]
NO_NORMALIZATION = "(none)"

DISPLAY_NAMES = {
    "frame": "Frame",
    "time": "Time (s)",
    "sum": "Sum",
    "mean": "Mean",
    "min": "Min",
    "max": "Max",
    "comx": "CoM X",
    "comy": "CoM Y",
    "proj_x": "Proj X (sum over Y)",
    "proj_y": "Proj Y (sum over X)",
}

AXIS_LABELS = {
    "frame": "Frame index",
    "time": "Time since first frame [s]",
    "sum": "ROI sum",
    "mean": "ROI mean",
    "min": "ROI min",
    "max": "ROI max",
    "comx": "ROI CoM X [px from ROI corner]",
    "comy": "ROI CoM Y [px from ROI corner]",
    "proj_x": "Sum over Y",
    "proj_y": "Sum over X",
    "pixel": "Pixel index within ROI",
}


def add_metric_item(combo: QtWidgets.QComboBox, key: str):
    combo.addItem(DISPLAY_NAMES.get(key, key), key)


def combo_key(combo: QtWidgets.QComboBox, default: str = "frame") -> str:
    data = combo.currentData()
    return str(data) if data is not None else (combo.currentText() or default)


def set_combo_key(combo: QtWidgets.QComboBox, key: str):
    index = combo.findData(key)
    if index >= 0:
        combo.setCurrentIndex(index)


def axis_label(key: str) -> str:
    return AXIS_LABELS.get(key, key)


# -- ROI geometry and per-frame reduction -----------------------------------
















# -- ROI geometry and per-frame reduction -----------------------------------
#
# The arithmetic lives in da6idb.rois now, against a plain rectangle, so a
# notebook gets the same numbers. These turn a pyqtgraph ROI into one.


def roi_is_rotated(roi) -> bool:
    try:
        return Rect.from_roi(roi).rotated
    except Exception:
        return False


def roi_bbox(roi, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """The ROI's axis-aligned pixel bounds ``(x0, x1, y0, y1)``, clipped to *shape*."""
    return Rect.from_roi(roi).bbox(shape)


def roi_sub_frame(frame, roi, image_item=None):
    """The pixels of one frame inside *roi*.

    *image_item* is accepted and ignored: the rectangle is resampled from its
    own geometry rather than through the scene, which is what lets the same
    call work outside a window.
    """
    return Rect.from_roi(roi).sub_frame(frame)


def roi_pixel_count(roi, shape, image_item=None, frame=None) -> int:
    """How many pixels *roi* covers -- the N in a ``background x N`` correction."""
    return Rect.from_roi(roi).pixel_count(shape, frame=frame)


def roi_series(stack, roi, image_item=None) -> dict:
    """Per-frame ROI metrics over the whole stack."""
    return series(stack, Rect.from_roi(roi))


# -- ROIs, shared across every open image -----------------------------------

ROI_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (80, 140, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

# Curve colours for the per-file traces in a shared ROI plot.
#
# Dark and saturated, for a white plot background. They used to be the
# pale set above, which reads on black and washes out on white -- and the
# background is white whenever this window was opened from the scan
# viewer, because pyqtgraph's background is a process-wide setting and
# that module sets it at import. The plots below now say which background
# they want rather than inheriting whichever module got imported first,
# so these only ever have to suit the one.
FILE_COLORS = [
    (20, 50, 150),       # dark blue
    (190, 40, 40),
    (20, 110, 50),
    (130, 50, 160),
    (200, 110, 0),
    (20, 120, 140),
]
# Where the 2D ROI plot's points start, and how much wider the ring
# marking the current frame is drawn, so it stays a ring at any size.
POINT_SIZE = 8
HIGHLIGHT_MARGIN = 6
PLOT_BACKGROUND = "w"
PLOT_FOREGROUND = "k"

# This window is the only one left that draws with pyqtgraph, so the global
# default belongs here. It used to be set by scan_viewer, which no longer
# imports pyqtgraph at all -- and while it was there, the image display came
# up light when this window was opened from the scan viewer and dark when it
# was opened on its own. Light either way now, which is the one most of this
# window already is.
pg.setConfigOption("background", PLOT_BACKGROUND)
pg.setConfigOption("foreground", PLOT_FOREGROUND)
pg.setConfigOptions(antialias=True)


class ContextRectROI(pg.RectROI):
    """One panel's drawn rectangle for a :class:`SharedRoi`.

    Locking freezes position and size: ``translatable`` stops the body being
    dragged, and :meth:`checkPointMove` -- the hook pyqtgraph's handles consult
    before every resize or rotate -- refuses the move.
    """

    def __init__(self, shared, panel, pos, size, pen):
        super().__init__(pos, size, pen=pen)
        self.shared = shared
        self.panel = panel
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.RightButton)
        self.addRotateHandle([1, 0], [0.5, 0.5])

    def checkPointMove(self, handle, pos, modifiers):
        return not self.shared.locked

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.shared.viewer.show_roi_menu(self.shared, QtGui.QCursor.pos())
            ev.accept()
            return
        super().mouseClickEvent(ev)


class SharedRoi(QtCore.QObject):
    """One logical ROI, drawn once per open image and kept in step.

    Every panel shows the same detector, so a rectangle means the same pixels in
    all of them. Dragging any one copy mirrors the geometry onto the others; the
    ROI plots listen to this object rather than to the individual rectangles.
    """

    geometryChanged = QtCore.pyqtSignal(object)
    geometryFinished = QtCore.pyqtSignal(object)

    def __init__(self, viewer, name: str, color, pos, size, angle: float = 0.0):
        super().__init__(viewer)
        self.viewer = viewer
        self.name = name
        self.color = color
        self.locked = False
        self.visible = True
        self.items: list[ContextRectROI] = []
        self._pos = (float(pos[0]), float(pos[1]))
        self._size = (float(size[0]), float(size[1]))
        self._angle = float(angle)
        self._mirroring = False

    # -- geometry, read the way roi_bbox()/roi_series() expect ------------

    def pos(self):
        return pg.Point(*self._pos)

    def size(self):
        return pg.Point(*self._size)

    def angle(self) -> float:
        return self._angle

    # -- items -------------------------------------------------------------

    def add_item(self, panel) -> ContextRectROI:
        item = ContextRectROI(
            self, panel, self._pos, self._size, pg.mkPen(self.color, width=2)
        )
        item.setAngle(self._angle, finish=False)
        item.sigRegionChanged.connect(self._on_item_changed)
        item.sigRegionChangeFinished.connect(self._on_item_finished)
        panel.image_view.addItem(item)
        self.items.append(item)
        self._apply_state(item)
        return item

    def item_for(self, panel) -> ContextRectROI | None:
        for item in self.items:
            if item.panel is panel:
                return item
        return None

    def remove_items(self):
        for item in self.items:
            item.panel.image_view.removeItem(item)
        self.items.clear()

    # -- keeping the copies in step ---------------------------------------

    def _write(self, item):
        item.blockSignals(True)
        try:
            item.setPos(pg.Point(*self._pos), finish=False)
            item.setSize(pg.Point(*self._size), finish=False)
            item.setAngle(self._angle, finish=False)
        finally:
            item.blockSignals(False)

    def _on_item_changed(self, item):
        if self._mirroring:
            return
        self._mirroring = True
        try:
            self._pos = (float(item.pos().x()), float(item.pos().y()))
            self._size = (float(item.size().x()), float(item.size().y()))
            self._angle = float(item.angle())
            for other in self.items:
                if other is not item:
                    self._write(other)
        finally:
            self._mirroring = False
        self.geometryChanged.emit(self)

    def _on_item_finished(self, _item):
        if self._mirroring:
            return
        self.geometryFinished.emit(self)

    def set_geometry(self, x: float, y: float, w: float, h: float, angle: float):
        self._pos = (float(x), float(y))
        self._size = (float(w), float(h))
        self._angle = float(angle)
        self._mirroring = True
        try:
            for item in self.items:
                self._write(item)
        finally:
            self._mirroring = False
        self.geometryChanged.emit(self)
        self.geometryFinished.emit(self)

    def swap_axes(self):
        """Mirror the rectangle across x=y, to follow a Transpose of the images."""
        if roi_is_rotated(self):
            return  # a rotated rectangle does not map onto a simple x/y swap
        (x, y), (w, h) = self._pos, self._size
        self.set_geometry(y, x, h, w, self._angle)

    # -- lock / visibility --------------------------------------------------

    def _apply_state(self, item):
        item.translatable = not self.locked
        item.resizable = not self.locked
        item.rotatable = not self.locked
        item.setAcceptedMouseButtons(
            QtCore.Qt.RightButton
            if self.locked
            else QtCore.Qt.LeftButton | QtCore.Qt.RightButton
        )
        item.setVisible(self.visible)
        for handle in item.getHandles():
            handle.setVisible(self.visible and not self.locked)

    def set_locked(self, locked: bool):
        self.locked = bool(locked)
        for item in self.items:
            self._apply_state(item)

    def set_visible(self, visible: bool):
        self.visible = bool(visible)
        for item in self.items:
            self._apply_state(item)


# -- One image, one panel ----------------------------------------------------


class ImagePanel(QtWidgets.QWidget):
    """The image area for a single file: title, ImageView, crosshair."""

    def __init__(self, viewer, data: ImageData, show_histogram: bool):
        super().__init__()
        self.viewer = viewer
        self.data = data
        self.frame_index = 0
        self._transposed = False
        self._transposed_stack: np.ndarray | None = None
        self._last_cursor: tuple[int, int] | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.title_label = QtWidgets.QLabel(data.path.name)
        self.title_label.setToolTip(str(data.path))
        layout.addWidget(self.title_label)

        self.plot_item = pg.PlotItem()
        self.image_view = pg.ImageView(view=self.plot_item)
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        if not show_histogram:
            # One shared colour scale, so only the first panel shows the sidebar.
            # Keep its space reserved though: if this panel's viewport were wider
            # than the first's, the aspect lock would give the two panels
            # different zooms and the images would not line up.
            histogram = self.image_view.ui.histogram
            policy = histogram.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            histogram.setSizePolicy(policy)
            histogram.hide()
        self.plot_item.setLabel("bottom", "X [pixels]")
        self.plot_item.setLabel("left", "Y [pixels]")
        self.plot_item.getViewBox().setAspectLocked(True)
        layout.addWidget(self.image_view, 1)

        self._setup_hover()

    # -- data ---------------------------------------------------------------

    def stack(self) -> np.ndarray:
        if not self._transposed:
            return self.data.stack
        if self._transposed_stack is None:
            self._transposed_stack = np.ascontiguousarray(
                self.data.stack.transpose(0, 2, 1)
            )
        return self._transposed_stack

    def set_transposed(self, transposed: bool):
        self._transposed = bool(transposed)

    def current_frame(self) -> np.ndarray:
        return self.stack()[self.frame_index]

    def image_item(self):
        return self.image_view.getImageItem()

    def set_frame(self, shared_index: int) -> bool:
        """Point this panel at *shared_index*, clamped to its own last frame.

        Returns whether the panel ran out of frames (i.e. the index was clamped).
        """
        last = self.data.n_frames - 1
        self.frame_index = int(np.clip(shared_index, 0, last))
        return shared_index > last

    def update_title(self, clamped: bool):
        text = f"{self.data.path.name}   [frame {self.frame_index}"
        if clamped:
            text += " — last"
        text += "]"
        self.title_label.setText(text)

    # -- crosshair ----------------------------------------------------------

    def _setup_hover(self):
        pen = pg.mkPen((255, 255, 0, 150), width=1)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hover_text = pg.TextItem("", color=(255, 255, 255))
        self._hover_text.setAnchor((0, 1))
        for item in (self._hline, self._vline, self._hover_text):
            item.setZValue(1000)
            self.plot_item.addItem(item, ignoreBounds=True)
        self._mouse_proxy = pg.SignalProxy(
            self.plot_item.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

    def _on_mouse_moved(self, event):
        pos = event[0]
        view_box = self.plot_item.getViewBox()
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        point = view_box.mapSceneToView(pos)
        x, y = int(np.floor(point.x())), int(np.floor(point.y()))
        frame = self.current_frame()
        if not (0 <= x < frame.shape[0] and 0 <= y < frame.shape[1]):
            self._last_cursor = None
            self._hover_text.setText("")
            return
        self._last_cursor = (x, y)
        self._vline.setPos(x + 0.5)
        self._hline.setPos(y + 0.5)
        self._hover_text.setPos(x, y)
        self.refresh_hover_text()

    def refresh_hover_text(self):
        if self._last_cursor is None:
            return
        x, y = self._last_cursor
        frame = self.current_frame()
        if not (0 <= x < frame.shape[0] and 0 <= y < frame.shape[1]):
            return
        value = float(frame[x, y])
        self._hover_text.setText(f"({x}, {y})  I={value:.6g}")
        self.viewer.show_readout(self, x, y, value)


# -- ROI plot docks ---------------------------------------------------------


class _RoiDockBase(QtWidgets.QDockWidget):
    """Shared series bookkeeping for the two ROI plot docks."""

    def __init__(self, viewer, roi: SharedRoi, title: str):
        super().__init__(title, viewer)
        self.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.viewer = viewer
        self.roi = roi

        # Dragging an ROI emits geometryChanged continuously; recomputing every
        # stack on each one would make the drag stutter, so coalesce.
        self._recompute_timer = QtCore.QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(150)
        self._recompute_timer.timeout.connect(self.recompute)

    def connect_signals(self):
        self.roi.geometryChanged.connect(self._schedule_recompute)
        self.roi.geometryFinished.connect(self._schedule_recompute)
        self.viewer.backgroundChanged.connect(self._schedule_recompute)
        self.viewer.frameChanged.connect(self.on_frame_changed)
        self.viewer.dataChanged.connect(self.recompute)

    def _schedule_recompute(self, *_args):
        self._recompute_timer.start()

    def background_per_pixel(self, panel) -> np.ndarray | None:
        """The background ROI's mean intensity per pixel, per frame, for *panel*.

        ``None`` when no background is set, or when this dock *is* the background
        ROI -- a background does not correct itself.
        """
        background = self.viewer.background_roi()
        if background is None or background is self.roi:
            return None
        return roi_series(
            panel.stack(), background, background.item_for(panel)
        )["mean"]

    def panel_series(self, panel) -> dict[str, np.ndarray]:
        """Every metric for this ROI over one panel's stack.

        With a background ROI set, its per-pixel mean ``b`` is taken off the
        pixel-value metrics: the sum loses ``b x (pixel count)`` and mean/min/max
        each lose ``b``. CoM is deliberately left on the raw pixels -- weighting a
        centroid by background-subtracted values goes unstable wherever the
        corrected sum passes through zero.
        """
        series = roi_series(panel.stack(), self.roi, self.roi.item_for(panel))

        background = self.background_per_pixel(panel)
        if background is not None:
            n_pixels = roi_pixel_count(
                self.roi,
                self.viewer.frame_shape(),
                self.roi.item_for(panel),
                panel.current_frame(),
            )
            series["sum"] = series["sum"] - background * n_pixels
            for key in ("mean", "min", "max"):
                series[key] = series[key] - background

        series["frame"] = np.arange(panel.data.n_frames, dtype=float)
        if panel.data.timestamps is not None:
            series["time"] = panel.data.timestamps
        series.update(panel.data.channels)
        return series

    def normalization_key(self) -> str:
        return combo_key(self.norm_select, NO_NORMALIZATION)

    def normalize(self, values: np.ndarray, series: dict) -> np.ndarray:
        """Divide *values* by the selected normalization channel, point by point.

        Points where the divisor is zero become NaN rather than infinity, so they
        simply leave a gap in the curve.
        """
        key = self.normalization_key()
        if key == NO_NORMALIZATION or key not in series:
            return values
        divisor = np.asarray(series[key], dtype=float)
        length = min(len(values), len(divisor))
        values, divisor = values[:length], divisor[:length]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(divisor != 0.0, values / divisor, np.nan)

    def normalize_scalar(self, values: np.ndarray, series: dict, index: int) -> np.ndarray:
        """Divide *values* by the normalization channel at one frame.

        Used for the single-frame projections, where the curve runs over pixels
        but the divisor is a per-frame quantity.
        """
        key = self.normalization_key()
        if key == NO_NORMALIZATION or key not in series:
            return values
        divisor = np.asarray(series[key], dtype=float)
        if index >= len(divisor) or divisor[index] == 0.0:
            return np.full_like(values, np.nan)
        return values / divisor[index]

    def normalization_note(self) -> str:
        key = self.normalization_key()
        if key == NO_NORMALIZATION:
            return ""
        return f" / {DISPLAY_NAMES.get(key, key)}"

    def refresh_norm_combo(self, keys: list[str]):
        current = combo_key(self.norm_select, NO_NORMALIZATION)
        self.norm_select.blockSignals(True)
        self.norm_select.clear()
        self.norm_select.addItem(NO_NORMALIZATION, NO_NORMALIZATION)
        for key in keys:
            add_metric_item(self.norm_select, key)
        set_combo_key(self.norm_select, current)
        if self.norm_select.currentIndex() < 0:
            self.norm_select.setCurrentIndex(0)
        self.norm_select.blockSignals(False)

    def background_note(self) -> str:
        """Suffix for axis labels when a background is being subtracted."""
        background = self.viewer.background_roi()
        if background is None:
            return ""
        if background is self.roi:
            return "  [this is the background ROI]"
        return f"  [− {background.name} background]"

    # -- frame scrubber ----------------------------------------------------

    def build_frame_slider(self) -> QtWidgets.QWidget:
        """A frame scrubber for the dock, shown in every mode."""
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.frame_label = QtWidgets.QLabel("frame 0")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(self.viewer.max_frames() - 1, 0))
        self.frame_slider.valueChanged.connect(
            lambda value: self.viewer.set_frame(int(value))
        )
        row.addWidget(self.frame_label)
        row.addWidget(self.frame_slider, 1)
        return widget

    def sync_frame_slider(self, index: int):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(self.viewer.max_frames() - 1, 0))
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"frame {index}")

    def refresh_combo(self, combo: QtWidgets.QComboBox, keys: list[str]):
        """Rebuild *combo* from *keys*, keeping the current selection if it survives."""
        current = combo_key(combo)
        combo.blockSignals(True)
        combo.clear()
        for key in keys:
            add_metric_item(combo, key)
        set_combo_key(combo, current)
        if combo.currentIndex() < 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    # Subclass hooks.
    def recompute(self):
        raise NotImplementedError

    def refresh_options(self):
        raise NotImplementedError

    def update_plot(self):
        raise NotImplementedError

    def on_frame_changed(self, index: int):
        raise NotImplementedError

    def retitle(self):
        raise NotImplementedError

    def closeEvent(self, event):
        self.viewer.forget_roi_dock(self)
        super().closeEvent(event)


class RoiPlotDock(_RoiDockBase):
    """The "ROI plot": one curve per open file, so scans can be compared.

    *Offset* stacks the curves -- curve *i* is drawn at ``y + i * offset`` -- which
    separates traces that would otherwise sit on top of each other. 0 overlays them.
    """

    def __init__(self, viewer, roi: SharedRoi):
        super().__init__(viewer, roi, f"ROI plot - {roi.name}")

        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.stats_label = QtWidgets.QLabel("ROI stats: -")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.radio_stack = QtWidgets.QRadioButton("Stack")
        self.radio_single = QtWidgets.QRadioButton("Single frame")
        self.radio_stack.setChecked(True)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.addButton(self.radio_stack, 0)
        self._mode_group.addButton(self.radio_single, 1)
        mode_row.addWidget(self.radio_stack)
        mode_row.addWidget(self.radio_single)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        axis_row = QtWidgets.QHBoxLayout()
        self.x_label = QtWidgets.QLabel("X:")
        self.x_select = QtWidgets.QComboBox()
        self.y_select = QtWidgets.QComboBox()
        axis_row.addWidget(self.x_label)
        axis_row.addWidget(self.x_select, 1)
        axis_row.addWidget(QtWidgets.QLabel("Y:"))
        axis_row.addWidget(self.y_select, 1)
        axis_row.addWidget(QtWidgets.QLabel("Norm:"))
        self.norm_select = QtWidgets.QComboBox()
        self.norm_select.setToolTip(
            "Divide Y by this, point by point (after any background subtraction).\n"
            "Same choices as the X and Y axes."
        )
        axis_row.addWidget(self.norm_select, 1)

        axis_row.addWidget(QtWidgets.QLabel("Offset:"))
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-1e12, 1e12)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setFixedWidth(120)
        self.offset_spin.setKeyboardTracking(False)
        self.offset_spin.setToolTip(
            "Separate the per-file curves: curve i is drawn at y + i × offset.\n"
            "0 overlays them again."
        )
        axis_row.addWidget(self.offset_spin)
        self.auto_offset_btn = QtWidgets.QPushButton("Auto")
        self.auto_offset_btn.setToolTip("Pick an offset from the curves' own span.")
        axis_row.addWidget(self.auto_offset_btn)
        layout.addLayout(axis_row)

        self.plot_item = pg.PlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        # Said here, not left to pyqtgraph's global setting: that is shared
        # with every other module in the process, so the colours below would
        # otherwise depend on whether the scan viewer had been imported.
        self.plot_widget.setBackground(PLOT_BACKGROUND)
        self.plot_item.getAxis("bottom").setPen(PLOT_FOREGROUND)
        self.plot_item.getAxis("left").setPen(PLOT_FOREGROUND)
        self.plot_item.getAxis("bottom").setTextPen(PLOT_FOREGROUND)
        self.plot_item.getAxis("left").setTextPen(PLOT_FOREGROUND)
        layout.addWidget(self.plot_widget, 1)

        self.frame_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen((80, 80, 80), width=1))
        self.plot_item.addItem(self.frame_line)

        # Kept visible in Single frame mode too -- that is the mode where
        # scrubbing frames matters most, since the plot shows one frame.
        layout.addWidget(self.build_frame_slider())

        self.setWidget(container)

        self.series_by_panel: list[dict[str, np.ndarray]] = []
        self.projections: list[tuple[np.ndarray, np.ndarray]] = []

        self.recompute()
        set_combo_key(self.x_select, "frame")
        set_combo_key(self.y_select, "sum")

        self.connect_signals()
        self.x_select.currentIndexChanged.connect(lambda _: self.update_plot())
        self.y_select.currentIndexChanged.connect(lambda _: self.update_plot())
        self.norm_select.currentIndexChanged.connect(lambda _: self.update_plot())
        self.offset_spin.valueChanged.connect(lambda _: self.update_plot())
        self.auto_offset_btn.clicked.connect(self._auto_offset)
        self.radio_stack.toggled.connect(lambda _: self._on_mode_changed())
        self.frame_line.sigPositionChanged.connect(self._on_line_moved)
        self._on_mode_changed()

    # -- data ---------------------------------------------------------------

    def recompute(self):
        self.series_by_panel = [self.panel_series(p) for p in self.viewer.panels]
        self.refresh_options()
        self.update_plot()

    def refresh_options(self):
        keys = self.viewer.metric_keys()
        self.refresh_combo(self.x_select, keys)
        self.refresh_combo(
            self.y_select, SINGLE_FRAME_KEYS if self._single_frame() else keys
        )
        self.refresh_norm_combo(keys)

    def _single_frame(self) -> bool:
        return self.radio_single.isChecked()

    def _on_mode_changed(self):
        single = self._single_frame()
        self.x_label.setVisible(not single)
        self.x_select.setVisible(not single)
        self.frame_line.setVisible(not single)
        self.refresh_combo(
            self.y_select,
            SINGLE_FRAME_KEYS if single else self.viewer.metric_keys(),
        )
        self.refresh_norm_combo(self.viewer.metric_keys())
        set_combo_key(self.y_select, "proj_x" if single else "sum")
        self.update_plot()

    def _compute_projections(self):
        self.projections = []
        for panel in self.viewer.panels:
            sub = roi_sub_frame(
                panel.current_frame(), self.roi, self.roi.item_for(panel)
            )
            if sub is None:
                self.projections.append((np.zeros(1), np.zeros(1)))
            else:
                # Frames are indexed [x, y]: summing over y leaves a profile along X.
                self.projections.append(
                    (sub.sum(axis=1).astype(float), sub.sum(axis=0).astype(float))
                )

    # -- curves --------------------------------------------------------------

    def _curves(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """``(label, x, y)`` for each file, before any offset is applied."""
        curves = []
        if self._single_frame():
            self._compute_projections()
            y_key = combo_key(self.y_select, "proj_x")
            for panel, series, (proj_x, proj_y) in zip(
                self.viewer.panels, self.series_by_panel, self.projections
            ):
                y = proj_x if y_key == "proj_x" else proj_y
                # The projection runs over pixels, so the divisor is this
                # frame's single value rather than a whole series.
                y = self.normalize_scalar(y, series, panel.frame_index)
                curves.append((panel.data.path.name, np.arange(len(y), dtype=float), y))
            return curves

        x_key = combo_key(self.x_select, "frame")
        y_key = combo_key(self.y_select, "sum")
        for panel, series in zip(self.viewer.panels, self.series_by_panel):
            if x_key not in series or y_key not in series:
                continue  # this file does not record that channel
            x = np.asarray(series[x_key], dtype=float)
            y = self.normalize(np.asarray(series[y_key], dtype=float), series)
            length = min(len(x), len(y))
            curves.append((panel.data.path.name, x[:length], y[:length]))
        return curves

    def _auto_offset(self):
        spans = [float(np.ptp(y)) for _label, _x, y in self._curves() if y.size]
        step = max(spans) * 1.1 if spans else 0.0
        if step > 0:
            # Round to 2 significant figures so the axis reads cleanly.
            magnitude = 10 ** (np.floor(np.log10(step)) - 1)
            step = float(np.ceil(step / magnitude) * magnitude)
        self.offset_spin.setValue(step)

    def update_plot(self):
        self.update_stats_label()
        self.plot_item.clear()
        # addLegend() installs on the PlotItem, which clear() empties, so the
        # legend is rebuilt alongside the curves.
        legend = self.plot_item.addLegend(offset=(-10, 10)) if len(self.viewer.panels) > 1 else None

        curves = self._curves()
        offset = self.offset_spin.value()
        for index, (label, x, y) in enumerate(curves):
            color = FILE_COLORS[index % len(FILE_COLORS)]
            shifted = y + index * offset
            item = self.plot_item.plot(
                x, shifted, pen=pg.mkPen(color, width=1), symbol="o", symbolSize=4,
                symbolBrush=color, symbolPen=None,
            )
            if legend is not None:
                suffix = f"  (+{index * offset:g})" if offset and index else ""
                legend.addItem(item, f"{label}{suffix}")

        note = self.normalization_note() + self.background_note()
        if self._single_frame():
            y_key = combo_key(self.y_select, "proj_x")
            self.plot_item.setLabel("bottom", axis_label("pixel"))
            self.plot_item.setLabel("left", axis_label(y_key) + note)
        else:
            self.plot_item.setLabel("bottom", axis_label(combo_key(self.x_select, "frame")))
            self.plot_item.setLabel(
                "left", axis_label(combo_key(self.y_select, "sum")) + note
            )
            self.plot_item.addItem(self.frame_line)
            self._move_line_to_frame(self.viewer.frame_index())

    def update_stats_label(self):
        parts = []
        for panel in self.viewer.panels:
            sub = roi_sub_frame(
                panel.current_frame(), self.roi, self.roi.item_for(panel)
            )
            total = "-" if sub is None else f"{np.sum(sub):.6g}"
            parts.append(f"{panel.data.path.name}: {total}")
        x0, x1, y0, y1 = roi_bbox(self.roi, self.viewer.frame_shape())
        self.stats_label.setText(
            f"{self.roi.name} [x={x0} y={y0} {x1 - x0}x{y1 - y0}]  "
            f"sum at this frame — " + ",  ".join(parts)
        )

    def _x_reference(self) -> np.ndarray:
        """The first file's X values, which the frame marker is positioned on."""
        if not self.series_by_panel:
            return np.zeros(1)
        series = self.series_by_panel[0]
        return np.asarray(
            series.get(combo_key(self.x_select, "frame"), series["frame"]), dtype=float
        )

    def _move_line_to_frame(self, index: int):
        x_data = self._x_reference()
        if len(x_data) == 0:
            return
        index = int(np.clip(index, 0, len(x_data) - 1))
        self.frame_line.blockSignals(True)
        self.frame_line.setPos(float(x_data[index]))
        self.frame_line.blockSignals(False)

    # -- frame sync ----------------------------------------------------------

    def on_frame_changed(self, index: int):
        self.sync_frame_slider(index)
        if self._single_frame():
            self.update_plot()
        else:
            self._move_line_to_frame(index)
            self.update_stats_label()

    def _on_line_moved(self):
        x_data = self._x_reference()
        if len(x_data) == 0:
            return
        # Snap to the nearest sample so the line always lands on a real frame.
        self.viewer.set_frame(
            int(np.argmin(np.abs(x_data - float(self.frame_line.value()))))
        )

    def retitle(self):
        self.setWindowTitle(f"ROI plot - {self.roi.name}")


class Roi2DPlotDock(_RoiDockBase):
    """The "ROI 2D plot" for one file: a scatter point per frame, Z as colour."""

    def __init__(self, viewer, roi: SharedRoi, panel: ImagePanel):
        super().__init__(viewer, roi, f"ROI 2D plot - {roi.name} - {panel.data.path.name}")
        self.panel = panel

        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.stats_label = QtWidgets.QLabel("ROI stats: -")
        layout.addWidget(self.stats_label)

        axis_row = QtWidgets.QHBoxLayout()
        self.x_select = QtWidgets.QComboBox()
        self.y_select = QtWidgets.QComboBox()
        self.z_select = QtWidgets.QComboBox()
        self.norm_select = QtWidgets.QComboBox()
        self.norm_select.setToolTip(
            "Divide Y and Z by this, point by point (after any background\n"
            "subtraction). Same choices as the X, Y and Z axes."
        )
        for label, combo in (
            ("X:", self.x_select),
            ("Y:", self.y_select),
            ("Z (colour):", self.z_select),
            ("Norm:", self.norm_select),
        ):
            axis_row.addWidget(QtWidgets.QLabel(label))
            axis_row.addWidget(combo, 1)
        layout.addLayout(axis_row)

        cmap_row = QtWidgets.QHBoxLayout()
        cmap_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap_select = QtWidgets.QComboBox()
        for name in available_colormaps():
            self.cmap_select.addItem(name, name)
        set_combo_key(self.cmap_select, "viridis")
        cmap_row.addWidget(self.cmap_select)
        cmap_row.addSpacing(12)
        self.z_min_spin = QtWidgets.QDoubleSpinBox()
        self.z_max_spin = QtWidgets.QDoubleSpinBox()
        for label, spin in (("Z min:", self.z_min_spin), ("Z max:", self.z_max_spin)):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setFixedWidth(110)
            spin.setEnabled(False)  # enabled only when Auto is off
            cmap_row.addWidget(QtWidgets.QLabel(label))
            cmap_row.addWidget(spin)
        self.z_auto_check = QtWidgets.QCheckBox("Auto")
        self.z_auto_check.setChecked(True)
        cmap_row.addWidget(self.z_auto_check)
        cmap_row.addSpacing(12)
        # How big a point should be depends on how many there are and how
        # close together they fall: a 51-point line scan wants them large
        # enough to read a colour off, a 600-frame map wants them small
        # enough not to cover each other.
        cmap_row.addWidget(QtWidgets.QLabel("Point size:"))
        self.point_size_spin = QtWidgets.QSpinBox()
        self.point_size_spin.setRange(1, 40)
        self.point_size_spin.setValue(POINT_SIZE)
        self.point_size_spin.setFixedWidth(60)
        self.point_size_spin.setToolTip(
            "Diameter of a plotted point, in pixels. The ring marking the\n"
            "current frame grows with it."
        )
        self.point_size_spin.valueChanged.connect(self._point_size_changed)
        cmap_row.addWidget(self.point_size_spin)
        cmap_row.addStretch(1)
        layout.addLayout(cmap_row)

        self.z_range_label = QtWidgets.QLabel("Z: -")
        layout.addWidget(self.z_range_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground(PLOT_BACKGROUND)
        for edge in ("bottom", "left"):
            self.plot_item.getAxis(edge).setPen(PLOT_FOREGROUND)
            self.plot_item.getAxis(edge).setTextPen(PLOT_FOREGROUND)
        layout.addWidget(self.plot_widget, 1)

        self.scatter = pg.ScatterPlotItem(size=POINT_SIZE, pen=pg.mkPen(None))
        # Black, not white: the ring marks the selected point and on a
        # white background a white ring is not there at all.
        self.highlight = pg.ScatterPlotItem(
            size=POINT_SIZE + HIGHLIGHT_MARGIN,
            pen=pg.mkPen("k", width=2), brush=pg.mkBrush(None)
        )
        self.plot_item.addItem(self.scatter)
        self.plot_item.addItem(self.highlight)

        layout.addWidget(self.build_frame_slider())

        self.setWidget(container)

        self.series: dict[str, np.ndarray] = {}
        self.recompute()
        set_combo_key(self.x_select, "frame")
        set_combo_key(self.y_select, "sum")
        set_combo_key(self.z_select, "max")

        self.connect_signals()
        for combo in (self.x_select, self.y_select, self.z_select,
                      self.norm_select, self.cmap_select):
            combo.currentIndexChanged.connect(lambda _: self.update_plot())
        self.z_min_spin.valueChanged.connect(lambda _: self.update_plot())
        self.z_max_spin.valueChanged.connect(lambda _: self.update_plot())
        self.z_auto_check.toggled.connect(self._on_auto_toggled)
        self.update_plot()

    def recompute(self):
        self.series = self.panel_series(self.panel)
        self.refresh_options()
        self.update_plot()

    def refresh_options(self):
        keys = [key for key in self.viewer.metric_keys() if key in self.series]
        for combo in (self.x_select, self.y_select, self.z_select):
            self.refresh_combo(combo, keys)
        self.refresh_norm_combo(keys)

    def series_for(self, key: str) -> np.ndarray:
        values = self.series.get(key)
        if values is None:
            values = self.series.get("frame", np.zeros(1))
        return np.asarray(values, dtype=float)

    def update_stats_label(self):
        sub = roi_sub_frame(
            self.panel.current_frame(), self.roi, self.roi.item_for(self.panel)
        )
        if sub is None:
            self.stats_label.setText("ROI stats: (empty)")
            return
        x0, x1, y0, y1 = roi_bbox(self.roi, self.viewer.frame_shape())
        self.stats_label.setText(
            f"{self.panel.data.path.name} — {self.roi.name} "
            f"[x={x0} y={y0} {x1 - x0}x{y1 - y0}]  "
            f"sum={np.sum(sub):.6g}  min={np.min(sub):.6g}  "
            f"max={np.max(sub):.6g}  mean={np.mean(sub):.6g}"
        )

    def _point_size_changed(self, size: int):
        """Resize the points, and the ring that marks the current frame.

        setSize on the items rather than a redraw: the data has not
        changed, and a 600-frame plot does not want recomputing to answer
        a question about how big a dot is.
        """
        self.scatter.setSize(int(size))
        self.highlight.setSize(int(size) + HIGHLIGHT_MARGIN)

    def _on_auto_toggled(self, checked: bool):
        self.z_min_spin.setEnabled(not checked)
        self.z_max_spin.setEnabled(not checked)
        self.update_plot()

    def _z_range(self, z_data: np.ndarray) -> tuple[float, float]:
        if self.z_auto_check.isChecked():
            finite = z_data[np.isfinite(z_data)] if z_data.size else z_data
            z_min = float(np.min(finite)) if finite.size else 0.0
            z_max = float(np.max(finite)) if finite.size else 1.0
            for spin, value in ((self.z_min_spin, z_min), (self.z_max_spin, z_max)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            return z_min, z_max
        return float(self.z_min_spin.value()), float(self.z_max_spin.value())

    def update_plot(self):
        self.update_stats_label()
        x_key = combo_key(self.x_select, "frame")
        y_key = combo_key(self.y_select, "sum")
        z_key = combo_key(self.z_select, "max")
        # X stays raw -- it is normally a position; Y and Z are the value axes.
        x_data = self.series_for(x_key)
        y_data = self.normalize(self.series_for(y_key), self.series)
        z_data = self.normalize(self.series_for(z_key), self.series)
        length = min(len(x_data), len(y_data), len(z_data))
        if length == 0:
            self.scatter.setData([], [])
            self.highlight.setData([], [])
            return
        x_data, y_data, z_data = x_data[:length], y_data[:length], z_data[:length]

        z_min, z_max = self._z_range(z_data)
        brushes = value_brushes(z_data, z_min, z_max, combo_key(self.cmap_select, "viridis"))
        self.scatter.setData(x=x_data, y=y_data, brush=brushes)

        index = int(np.clip(self.panel.frame_index, 0, length - 1))
        self.highlight.setData(x=[float(x_data[index])], y=[float(y_data[index])])

        mode = "auto" if self.z_auto_check.isChecked() else "manual"
        note = self.normalization_note() + self.background_note()
        self.z_range_label.setText(
            f"Z ({DISPLAY_NAMES.get(z_key, z_key)}{self.normalization_note()}): "
            f"{z_min:.6g} ... {z_max:.6g}  [{mode}]"
        )
        self.plot_item.setLabel("bottom", axis_label(x_key))
        self.plot_item.setLabel("left", axis_label(y_key) + note)

    def on_frame_changed(self, index: int):
        self.sync_frame_slider(index)
        self.update_plot()

    def retitle(self):
        self.setWindowTitle(
            f"ROI 2D plot - {self.roi.name} - {self.panel.data.path.name}"
        )


# -- Colormaps --------------------------------------------------------------

_COLORMAP_CANDIDATES = (
    "viridis", "plasma", "inferno", "magma", "turbo", "cividis",
    "gray", "hot", "jet", "CET-L9",
)
_LUT_CACHE: dict[str, np.ndarray] = {}


def get_colormap(name: str):
    """A pyqtgraph ColorMap by name, or ``None``.

    pyqtgraph only ships some of these itself -- ``gray``, ``hot`` and ``jet``
    come from matplotlib -- and ``get()`` raises rather than returning ``None``
    for a name it cannot find, so try both sources.
    """
    for source in (None, "matplotlib"):
        try:
            cmap = pg.colormap.get(name) if source is None else pg.colormap.get(name, source=source)
        except Exception:
            continue
        if cmap is not None:
            return cmap
    return None


def available_colormaps() -> list[str]:
    """The candidate colormaps this pyqtgraph build can actually supply."""
    names = [name for name in _COLORMAP_CANDIDATES if get_colormap(name) is not None]
    return names or ["viridis"]


def _lookup_table(name: str) -> np.ndarray | None:
    if name not in _LUT_CACHE:
        cmap = get_colormap(name)
        if cmap is None:
            return None
        try:
            _LUT_CACHE[name] = cmap.getLookupTable(nPts=256, alpha=False)
        except Exception:
            return None
    return _LUT_CACHE[name]


def value_brushes(values: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> list:
    """One brush per value, coloured by *cmap_name* over the ``[vmin, vmax]`` range."""
    span = vmax - vmin
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=vmin, posinf=vmax, neginf=vmin)
    fraction = np.clip((values - vmin) / (span if span > 0 else 1.0), 0.0, 1.0)
    lut = _lookup_table(cmap_name)
    if lut is None:
        level = (fraction * 255).astype(int)
        return [pg.mkBrush(int(v), int(v), int(v), 200) for v in level]
    index = (fraction * (len(lut) - 1)).astype(int)
    return [pg.mkBrush(int(r), int(g), int(b), 200) for r, g, b in lut[index]]


# -- The viewer window ------------------------------------------------------


class ImageViewerWindow(QtWidgets.QMainWindow):
    """One or more image stacks side by side, sharing ROIs, levels and frame."""

    frameChanged = QtCore.pyqtSignal(int)
    dataChanged = QtCore.pyqtSignal()
    # Raised when the background ROI is chosen, cleared, or dragged/resized --
    # every other ROI's numbers depend on it.
    backgroundChanged = QtCore.pyqtSignal()

    ROI_DOCK_WIDTH = 240

    def __init__(self, paths, parent=None, masters=None):
        super().__init__(parent)
        if isinstance(paths, (str, Path)):
            paths = [paths]
        paths = [Path(p) for p in paths]
        if not paths:
            raise RuntimeError("No image file to open.")
        # The scan browser resolved these pairs already; without them each
        # image has to guess its master back, which only works when the two
        # sit in the same directory.
        masters = list(masters) if masters else [None] * len(paths)
        if len(masters) != len(paths):
            masters = [None] * len(paths)

        self.datasets = [ImageData(path, master) for path, master in zip(paths, masters)]
        self._check_total_size()
        self._check_frame_shapes()

        names = ", ".join(d.path.name for d in self.datasets)
        self.setWindowTitle(f"Image viewer - {names}")
        self.resize(1250 if len(self.datasets) == 1 else 1700, 900)

        self._frame_index = 0
        self._transposed = False
        self._rois: list[SharedRoi] = []
        self._roi_docks: list[_RoiDockBase] = []
        self._hkl_windows: list = []
        self._roi_counter = 0
        self._background_roi: SharedRoi | None = None
        self._syncing_roi_table = False
        self._syncing_levels = False
        self._syncing_roi_geometry = False
        self._syncing_views = False
        self._ranged_on_show = False

        self._build_ui()
        self._restore_stored_rois()
        self._show_frame(0, auto_range=True, force_auto_levels=True)

    def _check_total_size(self):
        total = sum(d.stack.nbytes for d in self.datasets)
        if total > MAX_TOTAL_BYTES:
            listing = "\n".join(
                f"  {d.path.name}: {d.stack.nbytes / 1024 ** 3:.2f} GB" for d in self.datasets
            )
            raise RuntimeError(
                f"These {len(self.datasets)} files need "
                f"{total / 1024 ** 3:.1f} GB of memory together, more than this "
                f"viewer loads at once ({MAX_TOTAL_BYTES / 1024 ** 3:.0f} GB):\n\n"
                f"{listing}\n\nSelect fewer files."
            )

    def _check_frame_shapes(self):
        shapes = {d.frame_shape for d in self.datasets}
        if len(shapes) > 1:
            listing = "\n".join(
                f"  {d.path.name}: {d.frame_shape[0]} x {d.frame_shape[1]}"
                for d in self.datasets
            )
            raise RuntimeError(
                "These files have different frame sizes, so they cannot share "
                f"ROIs or be compared side by side:\n\n{listing}"
            )

    # -- construction -----------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        root.addLayout(self._build_control_row())
        root.addLayout(self._build_levels_row())

        self.info_label = QtWidgets.QLabel("")
        root.addWidget(self.info_label)

        # One panel per file, side by side and resizable.
        self.panel_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.panels: list[ImagePanel] = []
        for index, data in enumerate(self.datasets):
            panel = ImagePanel(self, data, show_histogram=(index == 0))
            self.panels.append(panel)
            self.panel_splitter.addWidget(panel)
        # Equal widths, so the aspect lock gives every panel the same zoom.
        self.panel_splitter.setSizes([1] * len(self.panels))
        self._link_panel_views()
        root.addWidget(self.panel_splitter, 1)

        root.addLayout(self._build_frame_row())

        self._build_roi_dock()
        self._build_view_menu()
        self._wire_level_signals()
        # The panels start greyscale until a colormap is pushed onto them, so
        # apply whatever the combo is showing rather than leaving them bare.
        self._apply_colormap(combo_key(self.cmap_combo, "viridis"))

        self.status = self.statusBar()
        self.status.showMessage(
            " | ".join(str(d.path) for d in self.datasets)
            if len(self.datasets) > 1
            else str(self.datasets[0].path)
        )

    def _link_panel_views(self):
        """Keep the panels showing the same region, without pyqtgraph's view link.

        ``setXLink``/``setYLink`` fight the aspect lock here: panel 0's viewport is
        narrower than the others (it carries the histogram), so each panel's
        aspect correction changes the range, the link pushes that to its
        neighbours, and they correct again -- the range runs away to millions of
        pixels. Copying the range across under a re-entrancy guard applies each
        correction exactly once instead.
        """
        for panel in self.panels:
            panel.plot_item.getViewBox().sigRangeChanged.connect(
                self._on_panel_range_changed
            )

    def _on_panel_range_changed(self, view_box, ranges):
        if self._syncing_views or len(self.panels) < 2:
            return
        self._syncing_views = True
        try:
            for panel in self.panels:
                other = panel.plot_item.getViewBox()
                if other is not view_box:
                    other.setRange(
                        xRange=ranges[0], yRange=ranges[1], padding=0, update=True
                    )
        finally:
            self._syncing_views = False

    def _build_control_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()

        self.add_roi_btn = QtWidgets.QPushButton("Add ROI")
        self.add_roi_btn.clicked.connect(lambda: self.add_roi())
        row.addWidget(self.add_roi_btn)

        self.hkl_btn = QtWidgets.QPushButton("HKL 3D…")
        self.hkl_btn.setToolTip(
            "Map this file's frames into reciprocal space, using the geometry\n"
            "recorded in the matching master file. Needs pyvista and xrayutilities."
        )
        self.hkl_btn.clicked.connect(self.open_hkl_window)
        row.addWidget(self.hkl_btn)

        row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmap_combo = QtWidgets.QComboBox()
        for name in available_colormaps():
            self.cmap_combo.addItem(name, name)
        set_combo_key(self.cmap_combo, "viridis")
        self.cmap_combo.currentIndexChanged.connect(
            lambda _: self._apply_colormap(combo_key(self.cmap_combo, "viridis"))
        )
        row.addWidget(self.cmap_combo)

        self.log_check = QtWidgets.QCheckBox("Log scale")
        self.log_check.toggled.connect(self._on_log_toggled)
        row.addWidget(self.log_check)

        self.aspect_check = QtWidgets.QCheckBox("Aspect 1:1")
        self.aspect_check.setChecked(True)
        self.aspect_check.toggled.connect(self._on_aspect_toggled)
        row.addWidget(self.aspect_check)

        self.transpose_check = QtWidgets.QCheckBox("Transpose")
        self.transpose_check.toggled.connect(self._on_transpose_toggled)
        row.addWidget(self.transpose_check)

        reset_btn = QtWidgets.QPushButton("Reset view")
        reset_btn.clicked.connect(self._reset_view)
        row.addWidget(reset_btn)

        # Closing the ROI dock takes its buttons with it, so there has to be a
        # way back that does not live inside the dock.
        self.panels_btn = QtWidgets.QToolButton()
        self.panels_btn.setText("Panels ▾")
        self.panels_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        row.addWidget(self.panels_btn)

        row.addStretch(1)
        return row

    def _build_levels_row(self) -> QtWidgets.QHBoxLayout:
        """Colour-scale controls: auto, or typed min/max in data units.

        One scale for every panel, so equal brightness means equal counts.
        """
        row = QtWidgets.QHBoxLayout()

        self.auto_levels_check = QtWidgets.QCheckBox("Auto levels")
        self.auto_levels_check.setChecked(True)
        row.addWidget(self.auto_levels_check)

        row.addWidget(QtWidgets.QLabel("Color min:"))
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        row.addWidget(self.vmin_spin)
        row.addWidget(QtWidgets.QLabel("max:"))
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        row.addWidget(self.vmax_spin)
        for spin in (self.vmin_spin, self.vmax_spin):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setFixedWidth(130)
            spin.setKeyboardTracking(False)  # apply on Enter/focus-out, not per digit
            spin.setToolTip(
                "Colour-scale limit in data units (raw counts, even with Log scale on),\n"
                "applied to every image. Typing a value switches off Auto levels."
            )

        full_btn = QtWidgets.QPushButton("Full range")
        full_btn.setToolTip("Set min/max to the combined data range of this frame.")
        full_btn.clicked.connect(self._levels_to_frame_range)
        row.addWidget(full_btn)

        self.levels_hint = QtWidgets.QLabel("")
        row.addWidget(self.levels_hint)
        row.addStretch(1)
        return row

    def _build_frame_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        multi = self.max_frames() > 1

        row.addWidget(QtWidgets.QLabel("Frame:"))
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.prev_btn.setFixedWidth(36)
        self.prev_btn.clicked.connect(lambda: self.set_frame(self._frame_index - 1))
        row.addWidget(self.prev_btn)

        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, max(self.max_frames() - 1, 0))
        self.frame_spin.setEnabled(multi)
        self.frame_spin.valueChanged.connect(self.set_frame)
        row.addWidget(self.frame_spin)

        row.addWidget(QtWidgets.QLabel(f"of {self.max_frames() - 1}"))

        self.next_btn = QtWidgets.QPushButton("▶")
        self.next_btn.setFixedWidth(36)
        self.next_btn.clicked.connect(lambda: self.set_frame(self._frame_index + 1))
        row.addWidget(self.next_btn)

        row.addSpacing(16)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.start_playback)
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_playback)
        self.pause_btn.setEnabled(False)
        self.play_btn.setEnabled(multi)
        row.addWidget(self.play_btn)
        row.addWidget(self.pause_btn)

        row.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(5)
        self.fps_spin.setEnabled(multi)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        row.addWidget(self.fps_spin)

        self.replay_check = QtWidgets.QCheckBox("Auto replay")
        self.replay_check.setChecked(True)
        self.replay_check.setEnabled(multi)
        row.addWidget(self.replay_check)

        row.addStretch(1)

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        return row

    def _build_roi_dock(self):
        dock = QtWidgets.QDockWidget("ROIs", self)
        dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        container = QtWidgets.QWidget(dock)
        container.setMinimumWidth(1)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.roi_table = QtWidgets.QTableWidget(0, 4)
        self.roi_table.setHorizontalHeaderLabels(["ROI", "Vis", "Lock", "Bkg"])
        self.roi_table.setMinimumWidth(1)
        self.roi_table.setMinimumHeight(90)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.roi_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.roi_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.roi_table.horizontalHeader().setStretchLastSection(False)
        self.roi_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        for column in (1, 2, 3):
            self.roi_table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents
            )
        self.roi_table.horizontalHeaderItem(3).setToolTip(
            "Use this ROI as the background: its mean intensity per pixel is\n"
            "subtracted from every other ROI's sum, mean, min and max."
        )
        self.roi_table.itemChanged.connect(self._on_roi_table_item_changed)
        self.roi_table.itemDoubleClicked.connect(self._on_roi_table_double_clicked)
        self.roi_table.itemSelectionChanged.connect(self._sync_roi_geometry_panel)
        layout.addWidget(self.roi_table, 1)

        layout.addWidget(self._build_roi_geometry_box())

        # Two columns of short buttons, so the dock stays narrow.
        buttons = [
            ("Add", lambda: self.add_roi()),
            ("Rename", lambda: self.rename_roi()),
            ("Delete", lambda: self.delete_roi()),
            ("Lock all", lambda: self.set_all_locked(True)),
            ("Unlock all", lambda: self.set_all_locked(False)),
            ("Plot", lambda: self.open_roi_plot()),
            ("2D plot", lambda: self.open_roi_2d_plot()),
        ]
        button_grid = QtWidgets.QGridLayout()
        button_grid.setContentsMargins(0, 0, 0, 0)
        button_grid.setSpacing(3)
        for index, (label, slot) in enumerate(buttons):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            button.setMinimumWidth(1)
            button_grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(button_grid)

        dock.setWidget(container)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.roi_dock = dock
        self.resizeDocks([dock], [self.ROI_DOCK_WIDTH], QtCore.Qt.Horizontal)

    def _build_roi_geometry_box(self) -> QtWidgets.QGroupBox:
        """Numeric position/size/angle entry for the selected ROI."""
        box = QtWidgets.QGroupBox("Geometry")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)

        specs = [
            ("x", "X:", 0, 0, (-99999.0, 99999.0), ""),
            ("y", "Y:", 0, 2, (-99999.0, 99999.0), ""),
            ("w", "W:", 1, 0, (0.01, 99999.0), ""),
            ("h", "H:", 1, 2, (0.01, 99999.0), ""),
            ("angle", "Angle:", 2, 0, (-360.0, 360.0), " °"),
        ]
        self.roi_geom_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, label, row, column, (low, high), suffix in specs:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setSuffix(suffix)
            spin.setKeyboardTracking(False)
            spin.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
            )
            spin.setMaximumWidth(84)
            spin.valueChanged.connect(self._on_roi_geometry_edited)
            grid.addWidget(QtWidgets.QLabel(label), row, column)
            grid.addWidget(spin, row, column + 1)
            self.roi_geom_spins[key] = spin

        self.roi_geom_hint = QtWidgets.QLabel("")
        self.roi_geom_hint.setWordWrap(True)
        self.roi_geom_hint.setMinimumWidth(1)
        box.setToolTip("Position and size in pixels; angle in degrees.")
        grid.addWidget(self.roi_geom_hint, 3, 0, 1, 4)
        return box

    def _build_view_menu(self):
        """A View menu (and Panels button) listing every dock, so a closed one
        can always be brought back."""
        self._view_menu = QtWidgets.QMenu("&View", self)
        self._view_menu.aboutToShow.connect(self._populate_view_menu)
        self.menuBar().addMenu(self._view_menu)
        self.panels_btn.setMenu(self._view_menu)
        self._populate_view_menu()

    def _live_docks(self) -> list[QtWidgets.QDockWidget]:
        """Docks that still exist.

        Not ``findChildren`` -- a plot dock closed a moment ago is deleted
        later (WA_DeleteOnClose), and until then it would still be listed and
        could be resurrected by "Show all panels".
        """
        docks = [getattr(self, "roi_dock", None)]
        docks.extend(self._roi_docks)
        return [dock for dock in docks if dock is not None]

    def _populate_view_menu(self):
        """Rebuild the menu each time -- ROI plot docks come and go."""
        self._view_menu.clear()
        docks = self._live_docks()
        for dock in docks:
            # toggleViewAction() is checkable and restores a closed dock.
            self._view_menu.addAction(dock.toggleViewAction())
        if docks:
            self._view_menu.addSeparator()
        self._view_menu.addAction("Show all panels", self.show_all_docks)

    def show_all_docks(self):
        for dock in self._live_docks():
            dock.setVisible(True)
        if getattr(self, "roi_dock", None) is not None:
            self.resizeDocks([self.roi_dock], [self.ROI_DOCK_WIDTH], QtCore.Qt.Horizontal)

    # -- data access ------------------------------------------------------

    def max_frames(self) -> int:
        return max(d.n_frames for d in self.datasets)

    def frame_shape(self) -> tuple[int, int]:
        shape = self.panels[0].stack().shape
        return int(shape[1]), int(shape[2])

    def frame_index(self) -> int:
        return self._frame_index

    def metric_keys(self) -> list[str]:
        """Base metrics plus every channel any open file records."""
        has_time = any(d.timestamps is not None for d in self.datasets)
        keys = [key for key in METRIC_KEYS if key != "time" or has_time]
        channels: set[str] = set()
        for data in self.datasets:
            channels.update(data.channels)
        return keys + sorted(channels)

    # Kept for single-file callers and tests.
    def stack(self) -> np.ndarray:
        return self.panels[0].stack()

    def current_frame(self) -> np.ndarray:
        return self.panels[0].current_frame()

    def image_item(self):
        return self.panels[0].image_item()

    # -- display ----------------------------------------------------------

    def _display_transform(self, frame: np.ndarray) -> np.ndarray:
        if self.log_check.isChecked():
            return np.log1p(np.maximum(frame, 0.0))
        return frame

    def _histogram_levels(self):
        try:
            return self.panels[0].image_view.getHistogramWidget().item.getLevels()
        except Exception:
            return None

    # -- colour-scale levels ----------------------------------------------
    #
    # The image items and the histogram work in the *display* domain (log1p'd
    # when Log scale is on); the spin boxes stay in the *data* domain, so a
    # number typed there always means raw counts. One set of levels drives
    # every panel.

    def _level_to_display(self, value: float) -> float:
        return float(np.log1p(max(value, 0.0))) if self.log_check.isChecked() else float(value)

    def _level_to_data(self, value: float) -> float:
        return float(np.expm1(value)) if self.log_check.isChecked() else float(value)

    def _wire_level_signals(self):
        self.auto_levels_check.toggled.connect(self._on_auto_levels_toggled)
        self.vmin_spin.valueChanged.connect(self._on_level_spin_changed)
        self.vmax_spin.valueChanged.connect(self._on_level_spin_changed)
        try:
            self.panels[0].image_view.getHistogramWidget().item.sigLevelsChanged.connect(
                self._on_histogram_changed
            )
        except Exception:
            pass  # older pyqtgraph: the spin boxes still drive the panels

    def _on_histogram_changed(self, _item=None):
        if self._syncing_levels:
            return
        levels = self._histogram_levels()
        if levels is not None:
            self._apply_levels_to_panels(levels, skip_first=True)
        self._sync_level_spins()

    def _apply_levels_to_panels(self, levels, skip_first: bool = False):
        """Put one pair of display-domain levels on every panel."""
        low, high = float(levels[0]), float(levels[1])
        self._syncing_levels = True
        try:
            for index, panel in enumerate(self.panels):
                if skip_first and index == 0:
                    continue  # it is the one that just changed
                panel.image_view.setLevels(min=low, max=high)
        finally:
            self._syncing_levels = False

    def _on_auto_levels_toggled(self, checked: bool):
        if checked:
            self._show_frame(self._frame_index, force_auto_levels=True)

    def _on_level_spin_changed(self, _value=None):
        if self._syncing_levels:
            return
        if self.auto_levels_check.isChecked():
            # Typing a limit is a request for manual control.
            self.auto_levels_check.blockSignals(True)
            self.auto_levels_check.setChecked(False)
            self.auto_levels_check.blockSignals(False)
        self._apply_manual_levels()

    def _apply_manual_levels(self):
        low = self._level_to_display(self.vmin_spin.value())
        high = self._level_to_display(self.vmax_spin.value())
        if high <= low:
            high = low + 1e-9  # pyqtgraph needs a non-empty range
        self._apply_levels_to_panels((low, high))

    def _sync_level_spins(self):
        """Mirror the shared levels into the spin boxes."""
        levels = self._histogram_levels()
        if levels is None:
            return
        self._syncing_levels = True
        try:
            self.vmin_spin.setValue(self._level_to_data(levels[0]))
            self.vmax_spin.setValue(self._level_to_data(levels[1]))
        finally:
            self._syncing_levels = False

    def _frame_data_range(self) -> tuple[float, float]:
        """The combined min/max of every panel's current frame, in data units."""
        frames = [panel.current_frame() for panel in self.panels]
        return float(min(f.min() for f in frames)), float(max(f.max() for f in frames))

    def _levels_to_frame_range(self):
        low, high = self._frame_data_range()
        self.auto_levels_check.blockSignals(True)
        self.auto_levels_check.setChecked(False)
        self.auto_levels_check.blockSignals(False)
        self._syncing_levels = True
        try:
            self.vmin_spin.setValue(low)
            self.vmax_spin.setValue(high)
        finally:
            self._syncing_levels = False
        self._apply_manual_levels()

    def _update_levels_hint(self):
        low, high = self._frame_data_range()
        label = "frame data" if len(self.panels) == 1 else "all frames"
        self.levels_hint.setText(f"({label}: {low:.6g} … {high:.6g})")

    # -- frames -------------------------------------------------------------

    def _show_frame(self, index: int, auto_range: bool = False, force_auto_levels: bool = False):
        index = int(np.clip(index, 0, self.max_frames() - 1))
        self._frame_index = index

        clamped = [panel.set_frame(index) for panel in self.panels]
        displays = [self._display_transform(panel.current_frame()) for panel in self.panels]

        if force_auto_levels or self.auto_levels_check.isChecked():
            low = float(min(d.min() for d in displays))
            high = float(max(d.max() for d in displays))
            if high <= low:
                high = low + 1e-9
            levels = (low, high)
            auto_histogram = True
        else:
            # Hold the levels explicitly, otherwise stepping frames or playing
            # back would rescale the contrast out from under the user.
            levels = self._histogram_levels()
            auto_histogram = False
            if levels is None:
                levels = (float(min(d.min() for d in displays)),
                          float(max(d.max() for d in displays)))

        self._syncing_levels = True
        try:
            for panel, display in zip(self.panels, displays):
                panel.image_view.setImage(
                    display,
                    autoLevels=False,
                    autoRange=auto_range,
                    autoHistogramRange=auto_histogram,
                    levels=levels,
                )
        finally:
            self._syncing_levels = False

        for panel, was_clamped in zip(self.panels, clamped):
            panel.update_title(was_clamped)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(index)
        self.frame_spin.blockSignals(False)
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < self.max_frames() - 1)

        self._update_info_label()
        self._update_levels_hint()
        self._sync_level_spins()
        for panel in self.panels:
            panel.refresh_hover_text()
        self.frameChanged.emit(index)

    def _refresh_image(self):
        self._show_frame(self._frame_index)

    def _update_info_label(self):
        nx, ny = self.frame_shape()
        parts = [f"{nx} x {ny} px", f"frame {self._frame_index} of {self.max_frames() - 1}"]
        if len(self.panels) == 1 and self.datasets[0].timestamps is not None:
            parts.append(f"t = {self.datasets[0].timestamps[self.panels[0].frame_index]:.3f} s")
        elif len(self.panels) > 1:
            parts.append(f"{len(self.panels)} files, ROIs shared")
        self.info_label.setText("   |   ".join(parts))

    def _apply_colormap(self, name: str):
        cmap = get_colormap(name)
        if cmap is None:
            self.statusBar().showMessage(f"Colormap {name!r} is not available.")
            return
        for panel in self.panels:
            panel.image_view.setColorMap(cmap)

    def _on_log_toggled(self, _checked: bool):
        # The display domain changes wholesale, so the old levels are meaningless.
        self._show_frame(self._frame_index, force_auto_levels=True)

    def _on_aspect_toggled(self, checked: bool):
        for panel in self.panels:
            panel.plot_item.getViewBox().setAspectLocked(checked)

    def _on_transpose_toggled(self, checked: bool):
        for panel in self.panels:
            panel.set_transposed(checked)
        for roi in self._rois:
            roi.swap_axes()
        self._show_frame(self._frame_index, auto_range=True, force_auto_levels=True)
        self._sync_roi_geometry_panel()
        self.dataChanged.emit()

    def _reset_view(self):
        self.panels[0].plot_item.getViewBox().autoRange()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._ranged_on_show:
            self._ranged_on_show = True
            # Auto-ranging during __init__ happens before Qt has laid the panels
            # out, so it measures a near-zero viewport and, with the aspect lock,
            # produces a wildly wrong range. Do it once the geometry is real.
            QtCore.QTimer.singleShot(0, self._reset_view)

    def show_readout(self, panel: ImagePanel, x: int, y: int, value: float):
        prefix = "" if len(self.panels) == 1 else f"{panel.data.path.name}:  "
        self.statusBar().showMessage(
            f"{prefix}x={x}  y={y}  value={value:.6g}   frame {panel.frame_index}"
        )

    # -- frame navigation -------------------------------------------------

    def set_frame(self, index: int):
        index = int(np.clip(index, 0, self.max_frames() - 1))
        if index == self._frame_index:
            return
        self._show_frame(index)

    def start_playback(self):
        if self.max_frames() < 2:
            return
        self.play_timer.start(max(1, int(1000 / max(1, self.fps_spin.value()))))
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)

    def pause_playback(self):
        self.play_timer.stop()
        self.play_btn.setEnabled(self.max_frames() > 1)
        self.pause_btn.setEnabled(False)

    def _on_fps_changed(self, fps: int):
        if self.play_timer.isActive():
            self.play_timer.start(max(1, int(1000 / max(1, fps))))

    def _advance_frame(self):
        nxt = self._frame_index + 1
        if nxt >= self.max_frames():
            if not self.replay_check.isChecked():
                self.pause_playback()
                return
            nxt = 0
        self._show_frame(nxt)

    # -- ROIs ---------------------------------------------------------------

    def _restore_stored_rois(self):
        """Recreate the rectangles the writer saved in ``entry/data/rois``.

        They come from the image file's own ``entry/data/rois`` when it has
        one, and otherwise from the master's baseline. Either way they describe
        the detector's own ROIs rather than anything the user drew here, so
        they start locked -- nudging one by accident would
        silently stop it matching the ROI the beamline recorded. With several
        files open the first file's set is used; they are shared by all panels.
        """
        for spec in self.datasets[0].stored_rois:
            self.add_roi(
                pos=(spec["x"], spec["y"]),
                size=(spec["w"], spec["h"]),
                angle=spec["angle"],
                name=spec["name"],
                select=False,
                locked=True,
            )
        if self._rois:
            self.roi_table.selectRow(0)

    def add_roi(
        self, pos=None, size=None, angle=0.0, name=None, select=True, locked=False
    ) -> SharedRoi:
        nx, ny = self.frame_shape()
        if pos is None or size is None:
            side_x, side_y = max(8, nx // 8), max(8, ny // 8)
            pos = (nx // 2 - side_x // 2, ny // 2 - side_y // 2)
            size = (side_x, side_y)
        self._roi_counter += 1
        if name is None:
            name = f"ROI{self._roi_counter}"

        color = ROI_COLORS[(self._roi_counter - 1) % len(ROI_COLORS)]
        roi = SharedRoi(self, name, color, pos, size, angle)
        for panel in self.panels:
            roi.add_item(panel)
        roi.geometryChanged.connect(self._on_roi_geometry_changed)
        self._rois.append(roi)
        self._append_roi_row(roi)
        roi.set_locked(locked)
        self._sync_roi_row(roi)
        if select:
            self.roi_table.selectRow(len(self._rois) - 1)
        return roi

    # -- ROI table ----------------------------------------------------------

    def _append_roi_row(self, roi: SharedRoi):
        self._syncing_roi_table = True
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)

        name_item = QtWidgets.QTableWidgetItem(roi.name)
        name_item.setForeground(QtGui.QBrush(QtGui.QColor(*roi.color)))
        self.roi_table.setItem(row, 0, name_item)

        states = ((1, roi.visible), (2, roi.locked), (3, roi is self._background_roi))
        for column, checked in states:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(
                (item.flags() | QtCore.Qt.ItemIsUserCheckable) & ~QtCore.Qt.ItemIsSelectable
            )
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.roi_table.setItem(row, column, item)
        self._syncing_roi_table = False

    def _sync_roi_row(self, roi: SharedRoi):
        """Push *roi*'s name/visibility/lock state into its table row."""
        if roi not in self._rois:
            return
        row = self._rois.index(roi)
        is_background = roi is self._background_roi
        self._syncing_roi_table = True
        self.roi_table.item(row, 0).setText(
            f"{roi.name}  (bkg)" if is_background else roi.name
        )
        states = ((1, roi.visible), (2, roi.locked), (3, is_background))
        for column, checked in states:
            item = self.roi_table.item(row, column)
            if item is not None:
                item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self._syncing_roi_table = False

    def _on_roi_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._syncing_roi_table:
            return
        row, column = item.row(), item.column()
        if not (0 <= row < len(self._rois)) or column not in (1, 2, 3):
            return
        checked = item.checkState() == QtCore.Qt.Checked
        roi = self._rois[row]
        if column == 1:
            self.set_roi_visible(roi, checked)
        elif column == 2:
            self.set_roi_locked(roi, checked)
        else:
            self.set_background_roi(roi if checked else None)

    def _on_roi_table_double_clicked(self, item: QtWidgets.QTableWidgetItem):
        if item.column() == 0:
            self.open_roi_plot(self._rois[item.row()])

    # -- ROI lock / visibility --------------------------------------------

    def set_roi_locked(self, roi: SharedRoi, locked: bool):
        """Freeze or release the ROI's position and size in every panel.

        A locked ROI keeps accepting right-clicks so its menu (and the unlock
        entry in it) stays reachable.
        """
        roi.set_locked(locked)
        self._sync_roi_row(roi)
        if roi is self.selected_roi():
            self._sync_roi_geometry_panel()

    def set_roi_visible(self, roi: SharedRoi, visible: bool):
        """Show or hide the ROI. Its plots keep updating -- they follow geometry."""
        roi.set_visible(visible)
        self._sync_roi_row(roi)

    def set_all_locked(self, locked: bool):
        for roi in self._rois:
            self.set_roi_locked(roi, locked)

    # -- background ROI -----------------------------------------------------

    def background_roi(self) -> SharedRoi | None:
        return self._background_roi

    def set_background_roi(self, roi: SharedRoi | None):
        """Designate (or clear) the ROI whose per-pixel mean is subtracted.

        Only one ROI can be the background, so picking a new one releases the
        previous. Every open plot recomputes.
        """
        if roi is self._background_roi:
            return
        previous = self._background_roi
        if previous is not None:
            previous.geometryChanged.disconnect(self._on_background_geometry_changed)
        self._background_roi = roi
        if roi is not None:
            # Moving or resizing the background changes every other ROI's
            # correction, so the plots have to follow it live.
            roi.geometryChanged.connect(self._on_background_geometry_changed)
        for other in (previous, roi):
            if other is not None:
                self._sync_roi_row(other)
        if roi is None:
            self.statusBar().showMessage("Background subtraction off.")
        else:
            self.statusBar().showMessage(
                f"{roi.name} is the background: its mean per pixel is subtracted "
                "from the other ROIs."
            )
        self.backgroundChanged.emit()

    def _on_background_geometry_changed(self, _roi):
        self.backgroundChanged.emit()

    def rois(self) -> list[SharedRoi]:
        """Every ROI on the image, in the order the table lists them.

        A copy of the list rather than the list itself: appending to it
        would leave the table, the plots and the stored specs behind.
        :meth:`add_roi` and :meth:`delete_roi` are how it changes.
        """
        return list(self._rois)

    def selected_roi(self) -> SharedRoi | None:
        row = self.roi_table.currentRow()
        if 0 <= row < len(self._rois):
            return self._rois[row]
        return None

    # -- numeric ROI geometry ---------------------------------------------

    def _sync_roi_geometry_panel(self):
        """Show the selected ROI's geometry; disable the fields while it is locked."""
        if self._syncing_roi_geometry:
            return
        roi = self.selected_roi()
        editable = roi is not None and not roi.locked
        for spin in self.roi_geom_spins.values():
            spin.setEnabled(editable)

        if roi is None:
            self.roi_geom_hint.setText("No ROI selected.")
            return
        self.roi_geom_hint.setText(
            f"{roi.name} is locked — untick Lock to edit." if roi.locked else roi.name
        )

        self._syncing_roi_geometry = True
        try:
            pos, size = roi.pos(), roi.size()
            self.roi_geom_spins["x"].setValue(float(pos.x()))
            self.roi_geom_spins["y"].setValue(float(pos.y()))
            self.roi_geom_spins["w"].setValue(float(size.x()))
            self.roi_geom_spins["h"].setValue(float(size.y()))
            self.roi_geom_spins["angle"].setValue(float(roi.angle()))
        finally:
            self._syncing_roi_geometry = False

    def _on_roi_geometry_edited(self, _value=None):
        if self._syncing_roi_geometry:
            return
        roi = self.selected_roi()
        if roi is None or roi.locked:
            return
        self._syncing_roi_geometry = True
        try:
            roi.set_geometry(
                self.roi_geom_spins["x"].value(),
                self.roi_geom_spins["y"].value(),
                self.roi_geom_spins["w"].value(),
                self.roi_geom_spins["h"].value(),
                self.roi_geom_spins["angle"].value(),
            )
        finally:
            self._syncing_roi_geometry = False

    def _on_roi_geometry_changed(self, roi: SharedRoi):
        """Keep the geometry fields in step when an ROI is dragged."""
        if self._syncing_roi_geometry:
            return
        if roi is self.selected_roi():
            self._sync_roi_geometry_panel()

    # -- ROI lifecycle ------------------------------------------------------

    def _require_roi(self, roi=None) -> SharedRoi | None:
        roi = roi or self.selected_roi()
        if roi is None:
            self.statusBar().showMessage("No ROI selected -- use 'Add ROI' first.")
        return roi

    def rename_roi(self, roi=None):
        roi = self._require_roi(roi)
        if roi is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename ROI", "Name:", text=roi.name
        )
        if not ok or not name.strip():
            return
        roi.name = name.strip()
        self._sync_roi_row(roi)
        for dock in self._roi_docks:
            if dock.roi is roi:
                dock.retitle()

    def delete_roi(self, roi=None):
        roi = self._require_roi(roi)
        if roi is None:
            return
        for dock in [d for d in self._roi_docks if d.roi is roi]:
            dock.close()
        if roi is self._background_roi:
            roi.geometryChanged.disconnect(self._on_background_geometry_changed)
            self._background_roi = None
            self.backgroundChanged.emit()
        index = self._rois.index(roi)
        roi.remove_items()
        self._rois.pop(index)
        self.roi_table.removeRow(index)
        self._sync_roi_geometry_panel()

    # -- ROI docks ----------------------------------------------------------

    def _existing_roi_dock(self, roi, dock_type, panel=None):
        for dock in self._roi_docks:
            if dock.roi is not roi or not isinstance(dock, dock_type):
                continue
            if panel is None or getattr(dock, "panel", None) is panel:
                return dock
        return None

    def open_roi_plot(self, roi=None):
        """One plot for this ROI holding every file's curve, for comparison."""
        roi = self._require_roi(roi)
        if roi is None:
            return
        existing = self._existing_roi_dock(roi, RoiPlotDock)
        if existing is not None:
            existing.setVisible(True)
            existing.raise_()
            return
        self._add_roi_dock(RoiPlotDock(self, roi), QtCore.Qt.BottomDockWidgetArea)

    def open_roi_2d_plot(self, roi=None):
        """One 2D plot per open file -- the axes are per-file, so they cannot share."""
        roi = self._require_roi(roi)
        if roi is None:
            return
        for panel in self.panels:
            existing = self._existing_roi_dock(roi, Roi2DPlotDock, panel)
            if existing is not None:
                existing.setVisible(True)
                existing.raise_()
                continue
            self._add_roi_dock(
                Roi2DPlotDock(self, roi, panel), QtCore.Qt.BottomDockWidgetArea
            )

    def _add_roi_dock(self, dock: _RoiDockBase, area):
        dock.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self._roi_docks.append(dock)
        self.addDockWidget(area, dock)
        dock.show()

    def forget_roi_dock(self, dock: _RoiDockBase):
        if dock in self._roi_docks:
            self._roi_docks.remove(dock)

    def show_roi_menu(self, roi: SharedRoi, global_pos):
        menu = QtWidgets.QMenu()
        actions = [
            ("Open ROI plot", lambda: self.open_roi_plot(roi)),
            ("Open ROI 2D plot", lambda: self.open_roi_2d_plot(roi)),
            (None, None),
            (
                "Unlock ROI" if roi.locked else "Lock ROI",
                lambda: self.set_roi_locked(roi, not roi.locked),
            ),
            (
                "Hide ROI" if roi.visible else "Show ROI",
                lambda: self.set_roi_visible(roi, not roi.visible),
            ),
            (
                "Clear background ROI"
                if roi is self._background_roi
                else "Use as background ROI",
                lambda: self.set_background_roi(
                    None if roi is self._background_roi else roi
                ),
            ),
            (None, None),
            ("Rename ROI", lambda: self.rename_roi(roi)),
            ("Delete ROI", lambda: self.delete_roi(roi)),
        ]
        for label, slot in actions:
            if label is None:
                menu.addSeparator()
            else:
                menu.addAction(label, slot)
        if roi in self._rois:
            self.roi_table.selectRow(self._rois.index(roi))
        menu.exec_(global_pos)

    # -- HKL 3D -------------------------------------------------------------

    def open_hkl_window(self):
        """Open the reciprocal-space map for the panel currently in front.

        hkl_viewer is imported here, not at module scope, so that this viewer
        keeps working on an installation without pyvista or xrayutilities.
        """
        try:
            from hkl_viewer import HklWindow
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "HKL 3D unavailable",
                "The 3D reciprocal-space map needs packages that are not "
                f"installed here:\n\n    {exc}\n\n"
                "Install them with:\n\n    pip install pyvista pyvistaqt xrayutilities",
            )
            return

        panel = self.panels[0]
        try:
            window = HklWindow(panel.data.path, parent=self)
        except (RuntimeError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Could not build the HKL map", str(exc))
            return
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda _=None, w=window: self._forget_hkl_window(w))
        # Qt destroys a window that nothing holds a reference to.
        self._hkl_windows.append(window)
        window.show()

    def _forget_hkl_window(self, window):
        self._hkl_windows = [w for w in self._hkl_windows if w is not window]

    # -- the panel as a dict -----------------------------------------------

    def _roi_state(self, roi: SharedRoi) -> dict:
        pos, size = roi.pos(), roi.size()
        return {
            "name": roi.name,
            "x": float(pos.x()),
            "y": float(pos.y()),
            "w": float(size.x()),
            "h": float(size.y()),
            "angle": float(roi.angle()),
            "locked": bool(roi.locked),
            "visible": bool(roi.visible),
        }

    def _about(self) -> dict:
        """What the window was given, as against what it was told to show."""
        background = self._background_roi
        return {
            "files": [str(d.path) for d in self.datasets],
            "masters": [str(d.master_path) if d.master_path else None
                        for d in self.datasets],
            "frames": int(self.max_frames()),
            "frame_shape": list(self.frame_shape()),
            "metrics": self.metric_keys(),
            "frame_range": list(self._frame_data_range()),
            "playing": bool(self.play_timer.isActive()),
            "background_roi": background.name if background else None,
            "open_plots": [
                {"roi": dock.roi.name,
                 "kind": "2d" if isinstance(dock, Roi2DPlotDock) else "curve"}
                for dock in self._roi_docks
            ],
            "hkl_windows": len(self._hkl_windows),
        }

    def state(self) -> dict:
        """Everything the panel says, as a dict :meth:`set_state` takes back.

        See :mod:`viewer_state` for what the pair is for and the rules
        they keep. Which files are open is *not* in here -- it is under
        ``"about"``, with the frame count and the detector's size --
        because this window takes its files in the constructor.

        The ROIs are, though. They are as much a setting as the
        colormap is: a rectangle drawn on one scan is the first thing
        anyone wants on the next one, and a list of them is how it
        travels.
        """
        return {
            "transpose": self.transpose_check.isChecked(),
            "log": self.log_check.isChecked(),
            "aspect": self.aspect_check.isChecked(),
            "colormap": combo_key(self.cmap_combo, "viridis"),
            "frame": int(self._frame_index),
            "level_min": float(self.vmin_spin.value()),
            "level_max": float(self.vmax_spin.value()),
            "auto_levels": self.auto_levels_check.isChecked(),
            "fps": int(self.fps_spin.value()),
            "replay": self.replay_check.isChecked(),
            "rois": [self._roi_state(roi) for roi in self._rois],
            "background": (self._background_roi.name
                           if self._background_roi is not None else None),
            viewer_state.ABOUT: self._about(),
        }

    def _set_rois(self, specs):
        """Make the ROI list match *specs*, matching the ones there by name.

        A rectangle whose name is already on the image is moved rather
        than duplicated, one the list does not mention is deleted, and
        the rest are drawn. Names are the identity here because that is
        what the ROI plots, the background and the table all show; the
        table keeps them in the order they were added, so a list that
        only reorders them changes nothing.
        """
        if not isinstance(specs, (list, tuple)):
            raise TypeError('rois is a list of {"name": ..., "x": ..., ...}')
        wanted = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise TypeError(f"rois[{index}] is a dict, not "
                                f"{type(spec).__name__}")
            missing = [key for key in ("name", "x", "y", "w", "h")
                       if key not in spec]
            if missing:
                raise ValueError(f"rois[{index}] has no "
                                 f"{', '.join(missing)}; an ROI takes "
                                 "name, x, y, w, h and may take "
                                 "angle, locked, visible")
            unknown = set(spec) - {"name", "x", "y", "w", "h",
                                   "angle", "locked", "visible"}
            if unknown:
                raise ValueError(f"rois[{index}] has no such key: "
                                 f"{', '.join(sorted(unknown))}")
            wanted.append(spec)
        names = [str(spec["name"]) for spec in wanted]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError("two ROIs cannot share a name: "
                             f"{', '.join(repeated)}")

        for roi in [roi for roi in self._rois if roi.name not in names]:
            self.delete_roi(roi)
        existing = {roi.name: roi for roi in self._rois}
        for spec, name in zip(wanted, names):
            geometry = (float(spec["x"]), float(spec["y"]),
                        float(spec["w"]), float(spec["h"]),
                        float(spec.get("angle", 0.0)))
            roi = existing.get(name)
            if roi is None:
                roi = self.add_roi(pos=geometry[:2], size=geometry[2:4],
                                   angle=geometry[4], name=name, select=False)
            else:
                roi.set_geometry(*geometry)
            self.set_roi_visible(roi, bool(spec.get("visible", True)))
            self.set_roi_locked(roi, bool(spec.get("locked", False)))

    def _set_background(self, name):
        if name is None:
            self.set_background_roi(None)
            return
        for roi in self._rois:
            if roi.name == str(name):
                self.set_background_roi(roi)
                return
        drawn = ", ".join(roi.name for roi in self._rois) or "none"
        raise ValueError(f"no ROI called {str(name)!r} to subtract as the "
                         f"background. ROIs: {drawn}")

    def set_state(self, state) -> list:
        """Drive the panel from a dict. Returns the settings applied.

        The order is the one the window needs rather than the one the
        dict happens to be in: Transpose swaps every ROI's axes, so it
        goes before the rectangles are placed; Log scale rescales the
        colour limits, so it goes before they are typed; and the limits
        go before the Auto tick that would overwrite them.
        """
        setters = {
            "transpose": self.transpose_check.setChecked,
            "log": self.log_check.setChecked,
            "aspect": self.aspect_check.setChecked,
            "colormap": lambda v: viewer_state.choose(
                self.cmap_combo, v, "colormap"),
            "frame": lambda v: viewer_state.within(
                self.frame_spin, v, "frame"),
            # Typing a limit switches Auto levels off, so a dict holding
            # both must mean the tick: it is applied last and wins.
            "level_min": lambda v: self.vmin_spin.setValue(float(v)),
            "level_max": lambda v: self.vmax_spin.setValue(float(v)),
            "auto_levels": self.auto_levels_check.setChecked,
            "fps": lambda v: viewer_state.within(self.fps_spin, v, "fps"),
            "replay": self.replay_check.setChecked,
            "rois": self._set_rois,
            "background": self._set_background,
        }
        return viewer_state.apply(state, setters)

    # -- teardown ---------------------------------------------------------

    def closeEvent(self, event):
        self.play_timer.stop()
        super().closeEvent(event)


def main():
    if len(sys.argv) < 2:
        print(f"usage: {Path(sys.argv[0]).name} <pva_image_file.h5> [more.h5 ...]")
        return 2
    app = QtWidgets.QApplication(sys.argv)
    try:
        window = ImageViewerWindow([Path(arg) for arg in sys.argv[1:]])
    except (RuntimeError, OSError) as exc:
        QtWidgets.QMessageBox.critical(None, "Could not open image file", str(exc))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
