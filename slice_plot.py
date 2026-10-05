"""The slice plane as a flat image, with real H/K/L axes.

The 3D view shows where the slice sits; this shows what is on it. The plane is
resampled onto a regular 2D lattice and drawn with matplotlib, so the axes carry
labels, tick numbers and a colour bar, and the cursor reads out the full H, K, L
of whatever it is over.

Which axes those are depends on the plane. A slice perpendicular to one of the
axes has the other two lying exactly along H, K or L, so the image is labelled
with them and the third is a constant printed above the plot -- an H-K map at
fixed L. For a plane at a general angle no axis of the image is an HKL axis, so
each is labelled with the direction it runs in, written as its [h k l]
components, and the cursor readout carries the real coordinates.

HKL is treated as a Cartesian frame here, exactly as the 3D window and the grid
itself do.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

from volume_style import AXIS_NAMES

# Moved into da6idb; re-exported for the rest of this module.
from da6idb.hkl.slice import (
    AXIS_TOLERANCE,
    MAX_PIXELS,
    _axis_index,
    average_along_normal,
    plane_basis,
    sample_plane,
    sample_volume,
    slab_offsets,
)

# How far below the top of the scale a log plot reaches when no floor is set.
# Fuzzy gridding leaves cells holding a millionth of a count -- numerical dust
# -- and a floor at the smallest positive value would spread the colours over
# twenty decades, squeezing everything real into the top fifth of the map.
LOG_DECADES = 5

# What the save dialog offers. Plain text and CSV are the lowest common
# denominator; .npz is what a Python session wants; HDF5 is what the rest of
# the beamline speaks.
EXPORT_FILTERS = (
    "Text, columns or matrix (*.txt *.dat)"
    ";;CSV (*.csv)"
    ";;NumPy archive (*.npz)"
    ";;HDF5 (*.h5 *.hdf5)"
)

CUT_NONE = "No cuts"
CUT_AXES = "Along the axes"
CUT_LINE = "Arbitrary line"














def _slug(text):
    """A label that is safe in a file name or an HDF5 key."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or "axis"


def _suffix_for(path: Path, selected: str) -> str:
    """Which writer to use: the extension if it says, else the chosen filter."""
    extension = path.suffix.lower()
    if extension in (".txt", ".dat", ".csv", ".npz", ".h5", ".hdf5"):
        return extension
    for marker, extension in (
        ("CSV", ".csv"), ("NumPy", ".npz"), ("HDF5", ".h5"), ("Text", ".txt")
    ):
        if selected.startswith(marker):
            return extension
    return ".txt"


def _edges(centres):
    """Cell edges for a row of sample centres, for imshow's extent."""
    centres = np.asarray(centres, float)
    if centres.size < 2:
        return float(centres[0] - 0.5), float(centres[0] + 0.5)
    half = (centres[1] - centres[0]) / 2.0
    return float(centres[0] - half), float(centres[-1] + half)


class SlicePlotWindow(QtWidgets.QMainWindow):
    """The current slice, flat, with axes, a colour bar and optional cuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slice")
        self.resize(820, 780)
        self._data = None
        self._layers = 1            # slab thickness, in cells, from the 3D window
        self._source = None         # the .vti these numbers came from
        self._normal = None
        self._cut = None            # crosshair for the axis cuts
        self._line = None           # ((x0, y0), (x1, y1)) for the arbitrary cut
        self._dragging = False
        self._art = None
        self._line_artist = None
        self._profile_artist = None
        self._profile_ax = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.log_check = QtWidgets.QCheckBox("Log intensity")
        self.log_check.setToolTip(
            "Colour by the logarithm of the intensity, which is usually the only\n"
            "way to see a weak feature beside a Bragg peak.\n"
            "Zero and unmeasured cells stay blank -- a log has nothing to say\n"
            f"about them. The scale runs {LOG_DECADES} decades below the top unless\n"
            "you set a positive Min in the 3D window."
        )
        self.log_check.toggled.connect(self._redraw)
        controls.addWidget(self.log_check)
        controls.addWidget(QtWidgets.QLabel("Cuts"))
        self.cut_combo = QtWidgets.QComboBox()
        self.cut_combo.addItems((CUT_NONE, CUT_AXES, CUT_LINE))
        self.cut_combo.setToolTip(
            "Along the axes: a profile down each axis of the image, through a\n"
            "crosshair you click to move.\n"
            "Arbitrary line: drag a line anywhere on the image and read the\n"
            "profile along it. One at a time, so each gets the width to be read."
        )
        self.cut_combo.currentIndexChanged.connect(self._cut_mode_changed)
        controls.addWidget(self.cut_combo)
        controls.addWidget(QtWidgets.QLabel("Cut width"))
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 999)
        self.width_spin.setValue(1)
        self.width_spin.setSuffix(" px")
        self.width_spin.setFixedWidth(84)
        self.width_spin.setToolTip(
            "Average this many rows, columns or parallel offsets into the cut.\n"
            "One pixel is the true profile; a few average the noise down."
        )
        self.width_spin.valueChanged.connect(self._redraw)
        controls.addWidget(self.width_spin)
        controls.addStretch(1)
        self.export_image_button = QtWidgets.QPushButton("Export image…")
        self.export_image_button.setToolTip(
            "Save the slice as a 2D array with its H/K/L axes."
        )
        self.export_image_button.clicked.connect(self.export_image)
        controls.addWidget(self.export_image_button)
        self.export_cut_button = QtWidgets.QPushButton("Export cut…")
        self.export_cut_button.setToolTip(
            "Save the cut as columns: position, H, K, L and intensity for every\n"
            "point along it."
        )
        self.export_cut_button.clicked.connect(self.export_cut)
        controls.addWidget(self.export_cut_button)
        layout.addLayout(controls)

        # The endpoints belong on screen, not only in the status bar where a
        # narrow window cuts them off.
        self.line_label = QtWidgets.QLabel("")
        self.line_label.setWordWrap(True)
        self.line_label.setStyleSheet(
            'font-family: Menlo, Monaco, "Courier New", monospace; color: gray;'
        )
        self.line_label.hide()
        layout.addWidget(self.line_label)

        self.figure = Figure(figsize=(6.4, 5.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, central)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        # A bare "monospace" is not a real family on macOS and Qt warns about
        # it on every launch; name actual fonts and fall back.
        self.coord_label = QtWidgets.QLabel(" ")
        self.coord_label.setStyleSheet(
            'font-family: Menlo, Monaco, "Courier New", monospace;'
        )
        layout.addWidget(self.coord_label)
        self.toolbar.set_message = self.coord_label.setText
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.status = self.statusBar()

    def _mode(self):
        return self.cut_combo.currentText()

    def _cut_mode_changed(self, _index=None):
        if self._mode() == CUT_LINE and self._line is None and self._data is not None:
            self._default_line()
        self._redraw()

    # -- taking the data ---------------------------------------------------

    def update_slice(self, grid, normal, origin, cmap="viridis", clim=None,
                     blank=0.0, layers=1, source=None):
        """Redraw for the plane the 3D window is showing.

        *layers* is the slab thickness in cells, set in the 3D window, so the
        image here is averaged over exactly what the 3D slice shows.
        """
        if grid is None:
            return
        self._layers = int(layers)
        self._source = source or self._source
        try:
            image, us, vs, u, v, corner = sample_plane(grid, normal, origin, layers=layers)
        except Exception as exc:
            self.status.showMessage(f"Could not sample the plane: {exc}")
            return

        normal = np.asarray(normal, float)
        normal = normal / np.linalg.norm(normal)

        # Each image axis is judged on its own: one can run exactly along an
        # HKL axis while the other does not, and then only that one gets real
        # H/K/L numbers on its ticks.
        def describe(direction, samples):
            index = _axis_index(direction)
            if index is None:
                return ("along [%.3f %.3f %.3f]" % tuple(direction), samples, None, 1.0)
            scale = float(direction[index])          # +1 or -1
            return (AXIS_NAMES[index], corner[index] + samples * scale, index, scale)

        x_label, x_coords, x_index, x_scale = describe(u, us)
        y_label, y_coords, y_index, y_scale = describe(v, vs)
        # imshow needs increasing axes; a mirrored basis would give decreasing.
        if x_coords[0] > x_coords[-1]:
            x_coords, image = x_coords[::-1], image[:, ::-1]
        if y_coords[0] > y_coords[-1]:
            y_coords, image = y_coords[::-1], image[::-1, :]

        axis = _axis_index(normal)
        if axis is not None:
            title = f"{AXIS_NAMES[axis]} = {float(corner[axis]):.4f}"
        else:
            title = (
                "normal [%.3f %.3f %.3f]  through  H %.4f  K %.4f  L %.4f"
                % (normal[0], normal[1], normal[2], *corner)
            )

        def to_hkl(x, y):
            along_u = (x - corner[x_index]) / x_scale if x_index is not None else x
            along_v = (y - corner[y_index]) / y_scale if y_index is not None else y
            return corner + along_u * u + along_v * v

        # Cells the scan never reached come back as the fill value; leave them
        # blank rather than painting them the bottom colour of the map.
        shown = np.ma.masked_invalid(image)
        shown = np.ma.masked_where(shown == blank, shown)

        self._normal = normal
        self._data = dict(
            image=shown,
            x=np.asarray(x_coords, float),
            y=np.asarray(y_coords, float),
            x_label=x_label,
            y_label=y_label,
            title=title,
            to_hkl=to_hkl,
            cmap=cmap,
            clim=clim,
        )
        # Keep the crosshair and the line where they were unless the plane has
        # moved out from under them.
        if self._cut is None or not self._inside(self._cut):
            self._cut = self._centre_point()
        if self._line is None or not all(self._inside(end) for end in self._line):
            self._default_line()
        self._redraw()

    # -- where things are --------------------------------------------------

    def _span(self):
        data = self._data
        return (data["x"][0], data["x"][-1], data["y"][0], data["y"][-1])

    def _inside(self, point):
        if self._data is None:
            return False
        x0, x1, y0, y1 = self._span()
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

    def _centre_point(self):
        x0, x1, y0, y1 = self._span()
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def _default_line(self):
        """A line across the middle, so there is something to see before a drag."""
        if self._data is None:
            return
        x0, x1, y0, y1 = self._span()
        mid = (y0 + y1) / 2.0
        pad = (x1 - x0) * 0.1
        self._line = ((x0 + pad, mid), (x1 - pad, mid))

    def _to_index(self, x, y):
        """Axis coordinates to fractional (row, column) in the image."""
        data = self._data
        xs, ys = data["x"], data["y"]
        column = float(np.interp(x, xs, np.arange(xs.size)))
        row = float(np.interp(y, ys, np.arange(ys.size)))
        return row, column

    # -- the cuts ----------------------------------------------------------

    def _profiles(self):
        """(x, horizontal cut, y, vertical cut) through the crosshair."""
        data = self._data
        image, xs, ys = data["image"], data["x"], data["y"]
        width = max(1, int(self.width_spin.value()))
        column = int(np.clip(np.abs(xs - self._cut[0]).argmin(), 0, xs.size - 1))
        row = int(np.clip(np.abs(ys - self._cut[1]).argmin(), 0, ys.size - 1))
        half = width // 2
        rows = slice(max(0, row - half), min(ys.size, max(0, row - half) + width))
        columns = slice(max(0, column - half), min(xs.size, max(0, column - half) + width))
        with np.errstate(invalid="ignore"):
            horizontal = np.ma.mean(image[rows, :], axis=0)
            vertical = np.ma.mean(image[:, columns], axis=1)
        return xs, horizontal, ys, vertical

    def _line_profile(self):
        """(distance along the line, intensity) for the arbitrary cut.

        Sampled along the segment at the image's own pixel pitch, and averaged
        over *width* parallel offsets either side of it.
        """
        from scipy.ndimage import map_coordinates

        data = self._data
        image = data["image"]
        (x0, y0), (x1, y1) = self._line
        start = np.array(self._to_index(x0, y0))          # (row, column)
        end = np.array(self._to_index(x1, y1))
        pixels = float(np.hypot(*(end - start)))
        count = int(np.clip(round(pixels) + 1, 2, MAX_PIXELS))
        steps = np.linspace(0.0, 1.0, count)
        along = start[:, None] + (end - start)[:, None] * steps[None, :]

        direction = end - start
        length = float(np.hypot(*direction))
        if length > 0:
            perpendicular = np.array([-direction[1], direction[0]]) / length
        else:
            perpendicular = np.array([0.0, 0.0])
        width = max(1, int(self.width_spin.value()))
        offsets = np.arange(width) - (width - 1) / 2.0

        # map_coordinates cannot see a mask, so hand it NaN and let NaN win.
        volume = image.filled(np.nan) if np.ma.isMaskedArray(image) else np.asarray(image)
        stack = []
        for offset in offsets:
            shifted = along + perpendicular[:, None] * offset
            stack.append(
                map_coordinates(
                    volume, shifted, order=1, mode="constant", cval=np.nan
                )
            )
        # A masked mean rather than nanmean: where the line runs right through
        # unmeasured space every offset is NaN, and nanmean warns about it on
        # every drag.
        values = np.ma.masked_invalid(np.vstack(stack)).mean(axis=0)

        # Distance in the plane's own units, which are HKL.
        distance = np.hypot(x1 - x0, y1 - y0) * steps
        return distance, np.ma.masked_invalid(values)

    # -- drawing -----------------------------------------------------------

    def _norm(self, data, clim):
        """Colour scaling, linear or logarithmic."""
        from matplotlib.colors import LogNorm
        from matplotlib.colors import Normalize

        low, high = (clim if clim is not None else (None, None))
        if not self.log_check.isChecked():
            return Normalize(vmin=low, vmax=high)
        # A log scale cannot show zero or less, so a floor has to come from
        # somewhere when the limit passed in is zero or negative.
        positive = data.compressed() if np.ma.isMaskedArray(data) else np.asarray(data)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        smallest = float(positive.min()) if positive.size else 1.0
        largest = float(positive.max()) if positive.size else 10.0
        if high is None or high <= 0:
            high = largest
        if low is None or low <= 0:
            # Never more than LOG_DECADES of range, and never below the data.
            low = max(smallest, high / (10.0 ** LOG_DECADES))
        if high <= low:
            high = low * 10.0
        return LogNorm(vmin=low, vmax=high)

    def _redraw(self):
        data = self._data
        if data is None:
            return
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        image = data["image"]
        mode = self._mode()
        self.figure.clear()
        self._line_artist = self._profile_artist = self._profile_ax = None
        ax = self.figure.add_subplot(111)
        self.ax = ax
        extent = (*_edges(data["x"]), *_edges(data["y"]))
        art = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=data["cmap"],
            interpolation="nearest",
            norm=self._norm(image, data["clim"]),
        )
        self._art = art
        ax.set_xlabel(data["x_label"])
        ax.set_ylabel(data["y_label"])
        ax.set_title(data["title"], fontsize=10)

        # append_axes keeps these lined up with the image even though it is
        # drawn at a fixed aspect, which a plain gridspec would not.
        divider = make_axes_locatable(ax)
        if mode == CUT_AXES:
            x, horizontal, y, vertical = self._profiles()
            ax_h = divider.append_axes("bottom", 1.15, pad=0.14, sharex=ax)
            ax_v = divider.append_axes("right", 1.15, pad=0.14, sharey=ax)
            ax_h.plot(x, horizontal, lw=0.9)
            ax_v.plot(vertical, y, lw=0.9)
            if self.log_check.isChecked():
                ax_h.set_yscale("log")
                ax_v.set_xscale("log")
            ax_h.set_xlabel(data["x_label"])
            ax_h.set_ylabel("intensity")
            ax_v.set_xlabel("intensity")
            ax_h.grid(alpha=0.3)
            ax_v.grid(alpha=0.3)
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
            ax_v.tick_params(labelleft=False)
            ax.axvline(self._cut[0], color="w", lw=0.7, alpha=0.7)
            ax.axhline(self._cut[1], color="w", lw=0.7, alpha=0.7)
            ax_h.axvline(self._cut[0], color="0.5", lw=0.7)
            ax_v.axhline(self._cut[1], color="0.5", lw=0.7)
        elif mode == CUT_LINE:
            (x0, y0), (x1, y1) = self._line
            self._line_artist = ax.plot(
                [x0, x1], [y0, y1], color="w", lw=1.1, marker="o", ms=3.5, alpha=0.9
            )[0]
            ax_l = divider.append_axes("bottom", 1.35, pad=0.42)
            distance, values = self._line_profile()
            self._profile_artist = ax_l.plot(distance, values, lw=0.9)[0]
            if self.log_check.isChecked():
                ax_l.set_yscale("log")
            ax_l.set_xlabel("distance along the line (HKL)")
            ax_l.set_ylabel("intensity")
            ax_l.grid(alpha=0.3)
            self._profile_ax = ax_l
        cax = divider.append_axes(
            "right", 0.13, pad=0.55 if mode == CUT_AXES else 0.16
        )
        self.figure.colorbar(art, cax=cax, label="intensity")

        def format_coord(x, y):
            h, k, l = data["to_hkl"](x, y)
            text = f"H {h:.4f}   K {k:.4f}   L {l:.4f}"
            value = self._value_at(x, y)
            return text if value is None else f"{text}   intensity {value:,.1f}"

        ax.format_coord = format_coord
        self.canvas.draw_idle()
        self._report()

    def _report(self):
        data = self._data
        image = data["image"]
        measured = int(image.count()) if np.ma.isMaskedArray(image) else image.size
        message = f"{image.shape[1]}×{image.shape[0]} pixels · {measured:,} measured"
        if self._layers > 1:
            message += f" · averaged over {self._layers} cells"
        mode = self._mode()
        if mode == CUT_AXES:
            h, k, l = data["to_hkl"](*self._cut)
            message += (
                f" · cuts through H {h:.4f}  K {k:.4f}  L {l:.4f}"
                f" ({self.width_spin.value()} px wide)"
            )
        elif mode == CUT_LINE:
            (x0, y0), (x1, y1) = self._line
            start = data["to_hkl"](x0, y0)
            end = data["to_hkl"](x1, y1)
            length = float(np.hypot(x1 - x0, y1 - y0))
            message += f" · line of length {length:.4f}, {self.width_spin.value()} px wide"
            self.line_label.setText(
                "line start   H %9.4f   K %9.4f   L %9.4f\n"
                "line end     H %9.4f   K %9.4f   L %9.4f     length %.4f"
                % (*start, *end, length)
            )
        self.line_label.setVisible(mode == CUT_LINE)
        self.export_cut_button.setEnabled(mode != CUT_NONE)
        self.status.showMessage(message)

    def _refresh_line(self):
        """Move the drawn line and its profile without rebuilding the figure."""
        if self._line_artist is None or self._profile_artist is None:
            return
        (x0, y0), (x1, y1) = self._line
        self._line_artist.set_data([x0, x1], [y0, y1])
        distance, values = self._line_profile()
        self._profile_artist.set_data(distance, values)
        self._profile_ax.relim()
        self._profile_ax.autoscale_view()
        self.canvas.draw_idle()
        self._report()

    # -- mouse -------------------------------------------------------------

    def _picking(self, event):
        """Whether this event is ours, rather than the toolbar's."""
        return (
            self._data is not None
            and event.inaxes is self.ax
            and not getattr(self.toolbar, "mode", "")
            and event.xdata is not None
            and event.ydata is not None
        )

    def _on_press(self, event):
        if not self._picking(event):
            return
        point = (float(event.xdata), float(event.ydata))
        if self._mode() == CUT_AXES:
            self._cut = point
            self._redraw()
        elif self._mode() == CUT_LINE:
            # Press starts a new line; it grows until the button comes back up.
            self._dragging = True
            self._line = (point, point)
            self._refresh_line()

    def _on_motion(self, event):
        if not self._dragging or not self._picking(event):
            return
        self._line = (self._line[0], (float(event.xdata), float(event.ydata)))
        self._refresh_line()

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        if self._picking(event):
            self._line = (self._line[0], (float(event.xdata), float(event.ydata)))
        start, end = self._line
        if np.hypot(end[0] - start[0], end[1] - start[1]) <= 0:
            # A bare click, not a drag: keep the line that was there.
            self._default_line()
        self._refresh_line()

    # -- export ------------------------------------------------------------

    def _provenance(self):
        """Lines describing where these numbers came from."""
        data = self._data
        lines = [
            f"source: {self._source}" if self._source else "source: unknown",
            f"plane: {data['title']}",
        ]
        if self._normal is not None:
            lines.append("normal: %.6f %.6f %.6f" % tuple(self._normal))
        if self._layers > 1:
            lines.append(f"averaged over {self._layers} cells along the normal")
        return lines

    def _hkl_along(self, xs, ys):
        """H, K, L at each (x, y) on the plane."""
        to_hkl = self._data["to_hkl"]
        points = np.array([to_hkl(float(x), float(y)) for x, y in zip(xs, ys)])
        return points[:, 0], points[:, 1], points[:, 2]

    def _cut_datasets(self):
        """[(name, note, {column: values}), ...] for whatever cut is showing.

        The first column is always called "position" rather than the axis it
        runs along: on an axis-aligned slice that name would be "H", colliding
        with the H of the H/K/L columns and quietly costing a column.
        """
        data = self._data
        mode = self._mode()
        if mode == CUT_AXES:
            xs, horizontal, ys, vertical = self._profiles()
            out = []
            h, k, l = self._hkl_along(xs, np.full(xs.size, self._cut[1]))
            out.append((
                f"along_{_slug(data['x_label'])}",
                f"position runs along {data['x_label']}",
                {"position": xs, "H": h, "K": k, "L": l,
                 "intensity": np.ma.filled(horizontal, np.nan)},
            ))
            h, k, l = self._hkl_along(np.full(ys.size, self._cut[0]), ys)
            out.append((
                f"along_{_slug(data['y_label'])}",
                f"position runs along {data['y_label']}",
                {"position": ys, "H": h, "K": k, "L": l,
                 "intensity": np.ma.filled(vertical, np.nan)},
            ))
            return out
        if mode == CUT_LINE:
            distance, values = self._line_profile()
            (x0, y0), (x1, y1) = self._line
            steps = np.linspace(0.0, 1.0, distance.size)
            xs = x0 + (x1 - x0) * steps
            ys = y0 + (y1 - y0) * steps
            h, k, l = self._hkl_along(xs, ys)
            note = (
                "position is the distance from the start of the line; "
                "line runs H %.6f K %.6f L %.6f -> H %.6f K %.6f L %.6f"
                % (h[0], k[0], l[0], h[-1], k[-1], l[-1])
            )
            return [(
                "line", note,
                {"position": distance, "H": h, "K": k, "L": l,
                 "intensity": np.ma.filled(values, np.nan)},
            )]
        return []

    def _ask(self, title, suggested):
        start = str(Path(self._source).with_suffix("")) + suggested if self._source else suggested
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self, title, start, EXPORT_FILTERS
        )
        if not path:
            return None, None
        path = Path(path)
        suffix = _suffix_for(path, selected or "")
        if path.suffix.lower() not in (".txt", ".dat", ".csv", ".npz", ".h5", ".hdf5"):
            path = path.with_suffix(suffix)
        return path, suffix

    def export_image(self):
        """Save the slice as a 2D array with its axes."""
        if self._data is None:
            return
        path, suffix = self._ask("Export the slice", "_slice.txt")
        if path is None:
            return
        data = self._data
        image = np.ma.filled(data["image"], np.nan)
        x, y = data["x"], data["y"]
        header = self._provenance() + [
            "x axis: %s  from %.6f to %.6f  (%d points, step %.6g)"
            % (data["x_label"], x[0], x[-1], x.size,
               (x[1] - x[0]) if x.size > 1 else 0.0),
            "y axis: %s  from %.6f to %.6f  (%d points, step %.6g)"
            % (data["y_label"], y[0], y[-1], y.size,
               (y[1] - y[0]) if y.size > 1 else 0.0),
            "rows run along y, columns along x; unmeasured cells are nan",
        ]
        try:
            if suffix in (".txt", ".dat"):
                np.savetxt(path, image, fmt="%.6g",
                           header="\n".join(header))
            elif suffix == ".csv":
                # First row the x values, first column the y values, so it
                # opens in a spreadsheet as the map it is.
                table = np.full((image.shape[0] + 1, image.shape[1] + 1), np.nan)
                table[0, 1:] = x
                table[1:, 0] = y
                table[1:, 1:] = image
                np.savetxt(path, table, fmt="%.6g", delimiter=",",
                           header="\n".join(header + [
                               "first row: %s values; first column: %s values"
                               % (data["x_label"], data["y_label"])]))
            elif suffix == ".npz":
                np.savez_compressed(
                    path, intensity=image, x=x, y=y,
                    x_label=data["x_label"], y_label=data["y_label"],
                    description="\n".join(header),
                )
            else:
                import h5py

                with h5py.File(path, "w") as handle:
                    handle.create_dataset("intensity", data=image, compression="gzip")
                    handle.create_dataset("x", data=x)
                    handle.create_dataset("y", data=y)
                    handle["x"].attrs["label"] = data["x_label"]
                    handle["y"].attrs["label"] = data["y_label"]
                    handle.attrs["description"] = "\n".join(header)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could not write", f"{path.name}\n\n{type(exc).__name__}: {exc}"
            )
            return
        self.status.showMessage(
            f"wrote {path.name} — {image.shape[1]}×{image.shape[0]} array plus axes"
        )

    def export_cut(self):
        """Save the cut as columns, H, K, L and intensity for every point."""
        if self._data is None:
            return
        sets = self._cut_datasets()
        if not sets:
            return
        path, suffix = self._ask("Export the cut", "_cut.txt")
        if path is None:
            return
        try:
            written = self._write_columns(path, suffix, sets)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could not write", f"{path.name}\n\n{type(exc).__name__}: {exc}"
            )
            return
        points = sum(len(next(iter(columns.values()))) for _n, _note, columns in sets)
        self.status.showMessage(
            f"wrote {', '.join(written)} — {points:,} points"
        )

    def _write_columns(self, path, suffix, sets):
        """One file of columns per cut for text, one file holding all otherwise."""
        provenance = self._provenance()
        written = []
        if suffix in (".txt", ".dat", ".csv"):
            delimiter = "," if suffix == ".csv" else " "
            for name, note, columns in sets:
                target = (
                    path if len(sets) == 1
                    else path.with_name(f"{path.stem}_{name}{path.suffix}")
                )
                table = np.column_stack([np.asarray(v, float) for v in columns.values()])
                np.savetxt(
                    target, table, fmt="%.6g", delimiter=delimiter,
                    header="\n".join(
                        provenance + [f"cut: {name}", note,
                                      f"cut width: {self.width_spin.value()} px",
                                      delimiter.join(columns.keys())]
                    ),
                )
                written.append(target.name)
        elif suffix == ".npz":
            payload = {"description": "\n".join(provenance)}
            for name, note, columns in sets:
                payload[f"{name}/note"] = np.array(note)
                for key, values in columns.items():
                    payload[f"{name}/{_slug(key)}"] = np.asarray(values, float)
            np.savez_compressed(path, **payload)
            written.append(path.name)
        else:
            import h5py

            with h5py.File(path, "w") as handle:
                handle.attrs["description"] = "\n".join(provenance)
                handle.attrs["cut_width_px"] = int(self.width_spin.value())
                for name, note, columns in sets:
                    group = handle.create_group(name)
                    group.attrs["note"] = note
                    for key, values in columns.items():
                        group.create_dataset(_slug(key), data=np.asarray(values, float))
            written.append(path.name)
        return written

    def _value_at(self, x, y):
        """The intensity under the cursor, read off the sampled image."""
        data = self._data
        if data is None:
            return None
        xs, ys, image = data["x"], data["y"], data["image"]
        column = int(np.abs(xs - x).argmin())
        row = int(np.abs(ys - y).argmin())
        if not (0 <= row < image.shape[0] and 0 <= column < image.shape[1]):
            return None
        value = image[row, column]
        return None if value is np.ma.masked else float(value)
