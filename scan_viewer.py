"""Browse a directory of Bluesky scans and plot them.

For each ``*_master.hdf`` file written by ``MyNXWriter``
(``callbacks/nexus_data_file_writer.py``), this shows:

- a scan listing on the left -- one row per scan giving its number, plan,
  motor, range, points and count time -- filled in as the directory is listed,
  without having to select anything. Only files carrying Bluesky scan metadata
  are listed; the sibling ``pva*.h5`` image files are not. Every column can be
  dragged to a different width. The leading part of whatever is selected -- the
  name up to the scan number, given once however many scans of that series are
  picked -- is shown above the plot. An "Img" tick marks the
  scans whose detector images sit beside them on disk. Click (or
  ctrl/shift-click for several) to open one or more scans; Refresh picks up
  new/changed files without losing the current selection,
- the ``entry/instrument/bluesky/streams/primary`` stream, chosen from one
  field table under the file list whose X / Y / Norm ticks drive the plot --
  X and Norm take a single field each, Y any number (none by default). An
  ``EPOCH`` entry (seconds past the UNIX epoch, one per point) joins the real
  fields wherever the file records it. X
  starts on the motor the scan moved, as named in its metadata -- the first
  selected scan decides -- and stays wherever you put it once you pick one,
  overlaid across every selected file, optionally normalized by dividing
  each Y field by a selected monitor field. A two-motor grid scan is instead
  drawn as a colour map over the two motors, with a colour scale; the "2D map"
  button turns that off to plot the raster as an ordinary X-Y curve, zoomable by dragging (via the
  plot toolbar's rectangle-zoom tool), with a peak-statistics table (max,
  min, half-max center and FWHM) below the plot for each curve shown, and
  a "Show dY/dX" toggle that overlays each curve's numerical derivative on
  a secondary Y axis,
- a "Baseline…" button opening a separate window with the
  ``entry/instrument/bluesky/streams/baseline`` stream for one selected file at
  a time, as a before/after table (the baseline stream records each field once
  before the scan and once after),
- a "Load Image" button that opens the detector frames belonging to the
  selected file(s) in a separate 2D image window (see ``image_viewer.py``).
  Selecting several files puts their images side by side in one window,
  sharing ROIs, so scans can be compared. The frames are not in the master
  file at all -- they live in a sibling ``pva*.h5`` written by the pvapy
  streaming consumer.

Each ``primary``/``baseline`` field is a subgroup holding ``EPOCH``, ``time``
and (for a scalar field) ``value`` datasets -- confirmed by inspecting a real
master file -- rather than a plain top-level dataset the way
``entry/data`` stores its (smaller) set of fields. ``lambda250k`` itself has
no ``value`` (its image lives elsewhere, linked separately) and is skipped.

``image_viewer.py`` must sit beside this file -- the image window is imported
from it.

Run with the ``6idb-bits`` environment's Python, with a display available::

    DISPLAY=localhost:10.0 \
        /home/beams18/USER6IDB/.conda/envs/6idb-bits/bin/python scan_viewer.py \
        [initial_directory]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg

from da6idb.files import file_prefix, list_hdf_files, resolve_image_file, scan_number
from da6idb import combine as _combine
from da6idb import export as _export
from da6idb import trends as _trends
from da6idb import fit as _fit
from da6idb.scan import (
    axis_edges,
    curve_stats,
    clipped_limits,
    default_x_field,
    scan_y_fields,
    format_scan_hkl,
    grid_stats,
    read_baseline,
    read_grid,
    read_primary,
    read_scan_info,
    scan_hkl,
    scan_x_field,
    stream_fields,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore
from PyQt5 import QtWidgets

# the one window-to-window import left, and it is about windows
from image_viewer import ImageViewerWindow


# Curves are drawn with pyqtgraph, which defaults to a black canvas; the rest
# of this window is light, and so is the matplotlib grid plot beside it.
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOptions(antialias=True)

# Enough distinct colours that a handful of files times a handful of fields
# stay tellable apart, in a fixed order so a curve keeps its colour across a
# redraw.
# pyqtgraph puts a grid line at every tick of every level, three deep, which
# inks about 15% of the plot and reads as a mesh. Two levels halves the lines
# and a lighter alpha softens what is left.
GRID_ALPHA = 0.12
GRID_LEVELS = 2

class SparseAxis(pg.AxisItem):
    """An axis offering only the coarser tick levels.

    The grid is drawn from the ticks, so trimming the finest level is what
    actually thins it; alpha alone only makes the same mesh fainter. Labels
    come from the top two levels, so they are unaffected.
    """

    def tickValues(self, minVal, maxVal, size):
        return super().tickValues(minVal, maxVal, size)[:GRID_LEVELS]


FIT_NONE = "None"
# What the statistics table always shows, before any fitted columns.
STATS_COLUMNS = ["Curve", "Max Y", "@X (max)", "Min Y", "@X (min)",
                 "Center", "FWHM"]

CURVE_COLORS = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
)
FILE_COLUMNS = ["Scan", "Img", "Plan", "Motor", "Range", "Pts", "Count [s]"]
# Starting widths. Every column is user-resizable from here; the motor name is
# the one that can run long, so it starts wide and is elided rather than
# allowed to shove the other columns aside.
FILE_COLUMN_WIDTHS = [58, 38, 72, 130, 118, 40, 66]
IMAGE_COLUMN = 1
MOTOR_COLUMN = 3
HAS_IMAGE, NO_IMAGE = "✓", "–"
FIELD_COLUMNS = ["Field", "X", "Y", "Norm"]
FIELD_X, FIELD_Y, FIELD_NORM = 1, 2, 3























































class HdfViewer(QtWidgets.QMainWindow):
    """Directory-of-HDF5-files browser with a primary-stream plot and baseline table."""

    def __init__(self, initial_directory: Path | None = None):
        super().__init__()
        self.setWindowTitle("6-ID-B HDF5 Scan Viewer")
        self.resize(1400, 900)

        self._settings = QtCore.QSettings("USER6IDB", "hdf5-viewer")
        self._directory: Path | None = None
        self._fields: list[str] = []
        self._last_paths: tuple[str, ...] = ()
        self._image_windows: list[ImageViewerWindow] = []
        self._syncing_fields = False
        self._colorbar = None
        self._last_curves: list = []
        self._last_mode = ""
        self._x_is_auto = True
        self._manual_x = ""
        # Y follows the scan's own hints until someone ticks a box.
        self._y_is_auto = True
        self._last_x_field: str | None = None
        self._last_monitor_field: str | None = None

        self._build_ui()

        start_dir = initial_directory
        if start_dir is None:
            remembered = self._settings.value("last_directory", "")
            if remembered and Path(remembered).is_dir():
                start_dir = Path(remembered)
        if start_dir is not None and start_dir.is_dir():
            self._set_directory(start_dir)

    # -- UI construction ------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open Directory…")
        open_btn.clicked.connect(self._choose_directory)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._reload_file_list)
        self.image_btn = QtWidgets.QPushButton("Load Image…")
        self.image_btn.setEnabled(False)
        self.image_btn.clicked.connect(self._open_image_viewers)
        self.baseline_btn = QtWidgets.QPushButton("Baseline…")
        self.baseline_btn.setToolTip(
            "Show the baseline (before/after) values for one selected file."
        )
        self.baseline_btn.clicked.connect(self._show_baseline_window)
        self.dir_label = QtWidgets.QLabel("No directory selected")
        top.addWidget(open_btn)
        top.addWidget(refresh_btn)
        top.addWidget(self.image_btn)
        top.addWidget(self.baseline_btn)
        top.addWidget(self.dir_label, 1)
        root.addLayout(top)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        # Left: the files, each with what its scan was.
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(
            QtWidgets.QLabel("HDF5 files (ctrl/shift-click for several):")
        )
        self.file_table = QtWidgets.QTableWidget(0, len(FILE_COLUMNS))
        self.file_table.setHorizontalHeaderLabels(FILE_COLUMNS)
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.file_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.file_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        file_header = self.file_table.horizontalHeader()
        # Interactive throughout: every column starts at a set width -- so a
        # long motor name does not push the others around -- and any of them
        # can be dragged wider or narrower. The last one takes up the slack.
        file_header.setStretchLastSection(True)
        for column, width in enumerate(FILE_COLUMN_WIDTHS):
            file_header.setSectionResizeMode(column, QtWidgets.QHeaderView.Interactive)
            self.file_table.setColumnWidth(column, width)
        self.file_table.setWordWrap(False)
        self.file_table.setTextElideMode(QtCore.Qt.ElideRight)
        row_header = self.file_table.verticalHeader()
        row_header.setMinimumSectionSize(16)
        row_header.setDefaultSectionSize(18)
        row_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.file_table.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.file_table)

        # The field chooser sits under the file list and drives all three axes.
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.addWidget(left)
        left_splitter.addWidget(self._build_field_panel())
        left_splitter.setSizes([520, 420])
        splitter.addWidget(left_splitter)

        splitter.addWidget(self._build_primary_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # The scan listing gets about a third of the window.
        total = self.width() or 1400
        splitter.setSizes([total // 3, total - total // 3])

        # Built now but shown only on demand, in its own window.
        self._baseline_panel = self._build_baseline_panel()
        self._baseline_window: QtWidgets.QDialog | None = None

        self.status = self.statusBar()

    def _build_field_panel(self) -> QtWidgets.QWidget:
        """One row per primary field, with X / Y / Norm ticks.

        X and Norm hold a single choice each -- ticking one clears the previous
        -- while any number of Y fields can be on at once.
        """
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            QtWidgets.QLabel("Fields — X (one), Y (any number), Norm (one):")
        )
        self.field_table = QtWidgets.QTableWidget(0, len(FIELD_COLUMNS))
        self.field_table.setHorizontalHeaderLabels(FIELD_COLUMNS)
        self.field_table.verticalHeader().setVisible(False)
        self.field_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.field_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.field_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        for column in range(1, len(FIELD_COLUMNS)):
            self.field_table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents
            )
        row_header = self.field_table.verticalHeader()
        row_header.setMinimumSectionSize(16)
        row_header.setDefaultSectionSize(18)
        row_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.field_table.itemChanged.connect(self._on_field_item_changed)
        layout.addWidget(self.field_table)
        return panel

    def _build_primary_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.file_label = QtWidgets.QLabel("No scan selected")
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.file_label)

        self.hkl_label = QtWidgets.QLabel("")
        self.hkl_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.hkl_label.setWordWrap(True)
        self.hkl_label.setStyleSheet("color: gray;")
        self.hkl_label.setToolTip(
            "The diffractometer's own psic_h/k/l, averaged over the scan.\n"
            "A spread is shown for any axis that moved -- on a rocking curve\n"
            "one of the three sweeps while the others hold still."
        )
        layout.addWidget(self.hkl_label)

        axis_row = QtWidgets.QHBoxLayout()
        self.axis_label = QtWidgets.QLabel("")
        axis_row.addWidget(self.axis_label, 1)
        self.grid_btn = QtWidgets.QPushButton("2D map")
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.setToolTip(
            "Draw a two-motor grid scan as a colour map.\n"
            "Untick to plot it as an ordinary X-Y curve instead.\n"
            "Only active when the selected scan is a grid."
        )
        self.grid_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.grid_btn)

        self.derivative_btn = QtWidgets.QPushButton("Show dY/dX")
        self.derivative_btn.setCheckable(True)
        self.derivative_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.derivative_btn)

        # Fitting. "None" rather than a tick, so the shape in use is always
        # legible: a tick plus a shape box would say the same thing twice
        # and could disagree.
        axis_row.addWidget(QtWidgets.QLabel("Fit"))
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItem(FIT_NONE)
        self.fit_combo.addItems(_fit.PEAK_SHAPES)
        self.fit_combo.setToolTip(
            "Fit this shape plus a sloping line to every plotted curve.\n"
            "The background is always fitted, never assumed flat."
        )
        self.fit_combo.currentIndexChanged.connect(self._update_plot)
        axis_row.addWidget(self.fit_combo)

        self.peaks_spin = QtWidgets.QSpinBox()
        self.peaks_spin.setRange(1, 4)
        self.peaks_spin.setPrefix("peaks ")
        self.peaks_spin.setToolTip(
            "How many copies of the shape to fit over one shared background.\n"
            "R2 cannot choose this for you -- the models are nested, so it\n"
            "only ever rises. A component too many divides one peak rather\n"
            "than failing, so each gets a row of its own to be judged."
        )
        self.peaks_spin.valueChanged.connect(self._update_plot)
        axis_row.addWidget(self.peaks_spin)

        self.trend_btn = QtWidgets.QPushButton("Trend")
        self.trend_btn.setCheckable(True)
        self.trend_btn.setToolTip(
            "Plot one point per scan instead of the curves: a fitted\n"
            "quantity against whatever was stepped between them.\n"
            "Needs a fit, and at least two scans selected."
        )
        self.trend_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.trend_btn)

        self.trend_x_combo = QtWidgets.QComboBox()
        self.trend_x_combo.setToolTip(
            "What to plot against. The baseline fields offered are the ones\n"
            "that actually moved across the selection, ranked by how far\n"
            "they moved relative to their own drift within a scan -- so the\n"
            "thing somebody deliberately stepped comes out at the top."
        )
        self.trend_x_combo.currentIndexChanged.connect(self._update_plot)
        axis_row.addWidget(self.trend_x_combo)

        self.trend_y_combo = QtWidgets.QComboBox()
        self.trend_y_combo.addItems(_trends.TREND_QUANTITIES)
        self.trend_y_combo.setToolTip("Which fitted quantity to follow.")
        self.trend_y_combo.currentIndexChanged.connect(self._update_plot)
        axis_row.addWidget(self.trend_y_combo)

        self.export_btn = QtWidgets.QPushButton("Export curves…")
        self.export_btn.setToolTip(
            "Write what is plotted as columns: an X and a Y per curve, with\n"
            "curves drawn against one sampling sharing their X column.\n"
            "The header records which scans and settings produced it."
        )
        self.export_btn.clicked.connect(self._export_curves)
        axis_row.addWidget(self.export_btn)

        self.clip_btn = QtWidgets.QPushButton("Clip map scale")
        self.clip_btn.setCheckable(True)
        self.clip_btn.setToolTip(
            "Leave the most extreme 0.5% at each end out of a colour map's\n"
            "scale. One or two wild cells -- a glitch, or the beam off the\n"
            "sample -- otherwise stretch it until everything else is one\n"
            "flat shade. Cells outside simply saturate, and the colour bar\n"
            "grows arrows to say so."
        )
        self.clip_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.clip_btn)

        self.average_btn = QtWidgets.QPushButton("Average repeats")
        self.average_btn.setCheckable(True)
        self.average_btn.setToolTip(
            "Average the scans in the selection that are one measurement\n"
            "repeated -- same plan, same motors, same sample position and\n"
            "temperature, same points. Scans that differ in any of those are\n"
            "left as separate curves, so a series does not quietly average\n"
            "into one clean curve that is not one measurement."
        )
        self.average_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.average_btn)

        self.fit_parts_btn = QtWidgets.QPushButton("Parts")
        self.fit_parts_btn.setCheckable(True)
        self.fit_parts_btn.setToolTip(
            "Draw the fitted background and each component on it, as well\n"
            "as the one summed model."
        )
        self.fit_parts_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.fit_parts_btn)
        layout.addLayout(axis_row)

        canvas_box = QtWidgets.QVBoxLayout()

        # Curves are pyqtgraph, the colour map stays matplotlib: one gives the
        # interaction, the other the colour bar and the print-quality output.
        # They take turns in a stack rather than fighting over the space.
        self.plot_stack = QtWidgets.QStackedWidget()
        self.curve_plot = pg.PlotWidget(axisItems={
            "bottom": SparseAxis(orientation="bottom"),
            "left": SparseAxis(orientation="left"),
        })
        self.curve_item = self.curve_plot.getPlotItem()
        self.curve_item.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        self.curve_item.addLegend(offset=(-10, 10), labelTextSize="8pt")
        self.curve_vb = self.curve_item.getViewBox()

        # dY/dX shares the X axis but needs its own Y scale, so it gets a
        # second view box hung off the right axis. Linking the X and keeping
        # the geometry in step is all that ties them together -- unlike a
        # matplotlib twin axis, neither one can drag the other's range about.
        self.deriv_vb = pg.ViewBox()
        self.curve_item.showAxis("right")
        self.curve_item.scene().addItem(self.deriv_vb)
        self.curve_item.getAxis("right").linkToView(self.deriv_vb)
        self.deriv_vb.setXLink(self.curve_item)
        self.curve_vb.sigResized.connect(self._sync_derivative_view)
        self.plot_stack.addWidget(self.curve_plot)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Kept so the grid's own readout can be swapped in and out.
        self._default_format_coord = self.ax.format_coord
        self._derivative_items: list = []
        # Gives the standard matplotlib "zoom to rectangle" (drag to zoom),
        # pan, and home/back/forward view-history controls.
        self.toolbar = NavigationToolbar2QT(self.canvas, panel)

        # The toolbar's built-in coordinate label lives inside the QToolBar, so
        # once the readout grows past the space left over Qt shunts it into the
        # overflow area and it vanishes -- which it did as soon as the labels
        # carried field names. Route the message to a label of our own, on the
        # row below, where it has the full width to itself.
        self.coord_label = QtWidgets.QLabel("")
        self.coord_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        if hasattr(self.toolbar, "locLabel"):
            self.toolbar.locLabel.hide()
        self.toolbar.set_message = self.coord_label.setText

        # The matplotlib toolbar belongs to the colour map, so it travels with
        # it; the curve plot has pyqtgraph's own wheel-zoom, drag-pan and
        # right-click menu instead.
        grid_page = QtWidgets.QWidget()
        grid_box = QtWidgets.QVBoxLayout(grid_page)
        grid_box.setContentsMargins(0, 0, 0, 0)
        grid_box.addWidget(self.toolbar)
        grid_box.addWidget(self.canvas, 1)
        self.plot_stack.addWidget(grid_page)

        canvas_box.addWidget(self.plot_stack, 1)
        canvas_box.addWidget(self.coord_label)

        # Both plots report the pointer to the same label.
        self._curve_mouse_proxy = pg.SignalProxy(
            self.curve_item.scene().sigMouseMoved, rateLimit=60,
            slot=self._on_curve_mouse_moved,
        )
        canvas_widget = QtWidgets.QWidget()
        canvas_widget.setLayout(canvas_box)
        layout.addWidget(canvas_widget, 1)

        layout.addWidget(QtWidgets.QLabel("Peak statistics (per plotted curve):"))
        self.stats_table = QtWidgets.QTableWidget(0, len(STATS_COLUMNS))
        self.stats_table.setHorizontalHeaderLabels(STATS_COLUMNS)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # Half the height it used to take. Compact rows keep roughly the same
        # number of curves visible in that space before scrolling.
        self.stats_table.setMaximumHeight(80)
        vertical_header = self.stats_table.verticalHeader()
        vertical_header.setMinimumSectionSize(16)
        vertical_header.setDefaultSectionSize(18)
        # defaultSectionSize only wins when the rows are not auto-sized.
        vertical_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        layout.addWidget(self.stats_table)
        return panel

    def _build_baseline_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Baseline (before/after) for file:"))
        self.baseline_combo = QtWidgets.QComboBox()
        self.baseline_combo.currentIndexChanged.connect(self._update_baseline)
        top.addWidget(self.baseline_combo, 1)
        layout.addLayout(top)

        self.baseline_table = QtWidgets.QTableWidget(0, 4)
        self.baseline_table.setHorizontalHeaderLabels(
            ["Field", "Before scan", "After scan", "Δ (after − before)"]
        )
        self.baseline_table.horizontalHeader().setStretchLastSection(True)
        self.baseline_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.baseline_table.setSortingEnabled(True)
        layout.addWidget(self.baseline_table)
        return panel

    # -- Directory / file list -------------------------------------------

    def _choose_directory(self):
        start = str(self._directory) if self._directory else ""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a directory of HDF5 files", start
        )
        if directory:
            self._set_directory(Path(directory))

    def _set_directory(self, directory: Path):
        self._directory = directory
        # A hand-picked X belongs to the run being looked at, not to the next
        # directory; deselecting on the way to another scan must not lose it,
        # so this is the one place automatic choice resumes.
        self._x_is_auto = True
        self._manual_x = ""
        # Y follows the scan's own hints until someone ticks a box.
        self._y_is_auto = True
        self.dir_label.setText(str(directory))
        self._settings.setValue("last_directory", str(directory))
        self._reload_file_list()

    def _reload_file_list(self):
        """List the directory, filling in each file's scan details as we go.

        The scan columns are read for every file up front rather than on
        selection, so the listing can be scanned for the run you want. Reads are
        cached per (path, mtime, size), so a Refresh only touches new or changed
        files.
        """
        if self._directory is None:
            self.file_table.setRowCount(0)
            return
        selected_names = {path.name for path in self._selected_paths()}
        self.file_table.blockSignals(True)
        self.file_table.setRowCount(0)
        # Only the scans: the sibling pva*.h5 image files carry no scan
        # metadata, and their frames are reached through "Load Image…" anyway.
        files = [p for p in list_hdf_files(self._directory) if read_scan_info(p)["is_scan"]]
        self.file_table.setRowCount(len(files))
        restore: list[int] = []
        for row, path in enumerate(files):
            for column, text in enumerate(self._scan_row(path)):
                item = QtWidgets.QTableWidgetItem(text)
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, str(path))
                elif column == IMAGE_COLUMN:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.file_table.setItem(row, column, item)
            if path.name in selected_names:
                restore.append(row)
        self._select_rows(restore)
        self.file_table.blockSignals(False)
        self.status.showMessage(f"{len(files)} scan(s) in {self._directory}.")
        self._on_selection_changed()

    def _select_rows(self, rows: list[int]):
        """Select every one of *rows*.

        Not ``selectRow`` in a loop: in ExtendedSelection mode each call
        *replaces* the selection, so restoring a multi-file selection after a
        Refresh would keep only the last row.
        """
        model = self.file_table.model()
        last_column = self.file_table.columnCount() - 1
        selection = QtCore.QItemSelection()
        for row in rows:
            selection.select(model.index(row, 0), model.index(row, last_column))
        self.file_table.selectionModel().select(
            selection,
            QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows,
        )

    @staticmethod
    def _scan_row(path: Path) -> list[str]:
        """One file's cells: its scan number, then what that scan was."""
        info = read_scan_info(path)
        motors = ", ".join(name for name, *_ in info["ranges"]) or ", ".join(info["motors"])
        span = ", ".join(
            f"{start:.6g} → {stop:.6g}" for _name, start, stop, _n in info["ranges"]
        )
        return [
            scan_number(path),
            HAS_IMAGE if resolve_image_file(path) is not None else NO_IMAGE,
            info["plan"],
            motors,
            span,
            "x".join(str(n) for n in info["shape"]) if len(info["shape"]) == 2
            else str(info["points"]),
            "" if info["count_time"] is None else f"{info['count_time']:.6g}",
        ]

    def _selected_paths(self) -> list[Path]:
        paths = []
        for row in sorted({index.row() for index in self.file_table.selectedIndexes()}):
            item = self.file_table.item(row, 0)
            if item is not None:
                paths.append(Path(item.data(QtCore.Qt.UserRole)))
        return paths

    # -- Selection handling -----------------------------------------------

    def _on_selection_changed(self):
        paths = self._selected_paths()
        self._update_file_label(paths)
        self._update_hkl_label(paths)
        self._update_image_button(paths)
        if not paths:
            self._fields = []
            self._syncing_fields = True
            self.field_table.setRowCount(0)
            self._syncing_fields = False
            self.baseline_combo.clear()
            self._update_plot()
            self._update_baseline()
            return
        self._refresh_field_lists(paths)
        self._refresh_baseline_combo(paths)
        self._update_plot()
        self._update_baseline()

    def _refresh_field_lists(self, paths: list[Path]):
        # Union of primary fields across every selected file, in the first
        # file's order (which, absent track_order, is alphabetical).
        primary_fields = stream_fields(paths[0], "primary")
        seen = set(primary_fields)
        for extra in paths[1:]:
            for name in stream_fields(extra, "primary"):
                if name not in seen:
                    primary_fields.append(name)
                    seen.add(name)

        # The scan that leads the selection decides the defaults.
        preferred_x = (
            scan_x_field(paths[0], primary_fields)
            or default_x_field(primary_fields)
            or ""
        )
        # The plan nominates one signal per detector -- the one worth looking
        # at out of a detector's half-dozen columns -- so open on that rather
        # than on an empty plot.
        preferred_y = scan_y_fields(paths[0], primary_fields)

        if primary_fields == self._fields:
            # Same fields as before, so the ticks stand -- but a different scan
            # may have been picked, and an X nobody chose by hand should follow
            # it to that scan's own motor.
            if self._x_is_auto and preferred_x and preferred_x != self._x_field():
                self._set_x_field(preferred_x)
            if self._y_is_auto and preferred_y != self._checked_y_fields():
                self._set_y_fields(preferred_y)
            return
        self._fields = primary_fields

        # Carry the current choices over to the new field set.
        current_x = self._manual_x if not self._x_is_auto else self._x_field()
        current_y = set(preferred_y if self._y_is_auto
                        else self._checked_y_fields())
        current_norm = self._monitor_field()
        if self._x_is_auto or current_x not in primary_fields:
            current_x = preferred_x
        if current_norm not in primary_fields:
            current_norm = ""

        self._syncing_fields = True
        try:
            self.field_table.setRowCount(0)
            self.field_table.setRowCount(len(primary_fields))
            for row, name in enumerate(primary_fields):
                label = QtWidgets.QTableWidgetItem(name)
                self.field_table.setItem(row, 0, label)
                states = (
                    (FIELD_X, name == current_x),
                    (FIELD_Y, name in current_y),
                    (FIELD_NORM, name == current_norm),
                )
                for column, checked in states:
                    item = QtWidgets.QTableWidgetItem()
                    item.setFlags(
                        (item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        & ~QtCore.Qt.ItemIsSelectable
                    )
                    item.setCheckState(
                        QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
                    )
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.field_table.setItem(row, column, item)
        finally:
            self._syncing_fields = False
        self.field_table.resizeColumnsToContents()
        self.field_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )

    # -- Field choices -------------------------------------------------------

    def _checked_in_column(self, column: int) -> list[str]:
        names = []
        for row in range(self.field_table.rowCount()):
            tick = self.field_table.item(row, column)
            label = self.field_table.item(row, 0)
            if tick is not None and label is not None and tick.checkState() == QtCore.Qt.Checked:
                names.append(label.text())
        return names

    def _set_x_field(self, name: str):
        """Move the X tick to *name* without counting as a hand-made choice."""
        self._syncing_fields = True
        try:
            for row in range(self.field_table.rowCount()):
                tick = self.field_table.item(row, FIELD_X)
                label = self.field_table.item(row, 0)
                if tick is None or label is None:
                    continue
                tick.setCheckState(
                    QtCore.Qt.Checked if label.text() == name else QtCore.Qt.Unchecked
                )
        finally:
            self._syncing_fields = False

    def _x_field(self) -> str:
        names = self._checked_in_column(FIELD_X)
        return names[0] if names else ""

    def _checked_y_fields(self) -> list[str]:
        return self._checked_in_column(FIELD_Y)

    def _set_y_fields(self, names: list[str]):
        """Tick exactly *names* in the Y column, leaving the rest alone."""
        wanted = set(names)
        self._syncing_fields = True
        try:
            for row in range(self.field_table.rowCount()):
                label = self.field_table.item(row, 0)
                tick = self.field_table.item(row, FIELD_Y)
                if label is None or tick is None:
                    continue
                tick.setCheckState(
                    QtCore.Qt.Checked if label.text() in wanted
                    else QtCore.Qt.Unchecked
                )
        finally:
            self._syncing_fields = False
        self._update_plot()

    def _monitor_field(self) -> str:
        """The normalization field, or ``""`` for none."""
        names = self._checked_in_column(FIELD_NORM)
        return names[0] if names else ""

    def _on_field_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._syncing_fields:
            return
        column = item.column()
        if column == FIELD_X:
            # Remember the choice by name, not by reading the widget back: an
            # empty selection clears the table, and the next scan would then
            # find no X to carry over. Unticking hands control back.
            label = self.field_table.item(item.row(), 0)
            chosen = item.checkState() == QtCore.Qt.Checked and label is not None
            self._manual_x = label.text() if chosen else ""
            self._x_is_auto = not chosen
        if column == FIELD_Y:
            # Any hand-picked Y stops the hints overriding it; clearing them
            # all hands the choice back to the scan.
            self._y_is_auto = not self._checked_y_fields()
        # X and Norm are single-choice: ticking one clears whatever held it.
        if column in (FIELD_X, FIELD_NORM) and item.checkState() == QtCore.Qt.Checked:
            self._syncing_fields = True
            try:
                for row in range(self.field_table.rowCount()):
                    other = self.field_table.item(row, column)
                    if other is not None and other is not item:
                        other.setCheckState(QtCore.Qt.Unchecked)
            finally:
                self._syncing_fields = False
        self._update_plot()

    def _update_hkl_label(self, paths: list[Path]):
        """Where the diffractometer was, averaged over each selected scan."""
        lines = []
        missing = 0
        for path in paths:
            values = scan_hkl(path)
            if values is None:
                missing += 1
                continue
            text = format_scan_hkl(values)
            lines.append(text if len(paths) == 1 else f"{scan_number(path)}:  {text}")
        if not lines:
            # Silent when nothing records it, rather than a standing complaint.
            self.hkl_label.setText(
                "" if not paths else "psic_h/k/l not recorded in this scan"
            )
            return
        if missing:
            lines.append(f"({missing} more without psic_h/k/l)")
        self.hkl_label.setText("        ".join(lines))

    def _update_file_label(self, paths: list[Path]):
        """Name the selected file(s) above the plot, by their leading part.

        Only what comes before the scan number is shown, and a series shared by
        several selected files is named once rather than repeated. The full
        names are on the tooltip.
        """
        if not paths:
            self.file_label.setText("No scan selected")
            self.file_label.setToolTip("")
            return
        prefixes: list[str] = []
        for path in paths:
            prefix = file_prefix(path)
            if prefix not in prefixes:
                prefixes.append(prefix)
        self.file_label.setText(",  ".join(prefixes))
        self.file_label.setToolTip("\n".join(str(path) for path in paths))

    def _update_axis_label(self):
        x_field = self._x_field() or "(pick one)"
        monitor = self._monitor_field()
        text = f"X: {x_field}    Y: {', '.join(self._checked_y_fields()) or '(none)'}"
        if monitor:
            text += f"    normalized by {monitor}"
        self.axis_label.setText(text)

    def _refresh_baseline_combo(self, paths: list[Path]):
        names = [p.name for p in paths]
        current = self.baseline_combo.currentText()
        self.baseline_combo.blockSignals(True)
        self.baseline_combo.clear()
        self.baseline_combo.addItems(names)
        self.baseline_combo.blockSignals(False)
        if current in names:
            self.baseline_combo.setCurrentText(current)

    # -- Image windows ------------------------------------------------------

    def _image_pairs(self, paths: list[Path]) -> list[tuple[Path, Path | None]]:
        return [(path, resolve_image_file(path)) for path in paths]

    def _resolved_images(self, paths: list[Path]) -> list[Path]:
        """The image files for *paths*, in selection order and without repeats.

        Two masters can point at the same image file, and a selection may mix a
        master with its own image file; either way it should appear once.
        """
        images: list[Path] = []
        for _master, image in self._image_pairs(paths):
            if image is not None and image not in images:
                images.append(image)
        return images

    def _update_image_button(self, paths: list[Path]):
        """Enable "Load Image…" only for selections that have image files."""
        found = self._resolved_images(paths)
        self.image_btn.setEnabled(bool(found))
        if not paths:
            self.image_btn.setToolTip("Select a master file to load its detector images.")
        elif not found:
            self.image_btn.setToolTip("No pva*.h5 image file next to the selected file(s).")
        elif len(found) == 1:
            self.image_btn.setToolTip(f"Open {found[0].name}")
        else:
            self.image_btn.setToolTip(
                f"Open {len(found)} images side by side, sharing ROIs:\n  "
                + "\n  ".join(image.name for image in found)
            )

    def _open_image_viewers(self):
        """Open the selected files' images -- several go side by side in one window."""
        paths = self._selected_paths()
        pairs = self._image_pairs(paths)
        missing = [master.name for master, image in pairs if image is None]
        images = self._resolved_images(paths)
        if not images:
            self.status.showMessage(f"No image file found for: {', '.join(missing)}")
            return

        masters = [master for master, image in pairs if image is not None]
        try:
            window = ImageViewerWindow(images, parent=self, masters=masters)
        except (RuntimeError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open images", str(exc))
            return

        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda _=None, w=window: self._forget_image_window(w))
        # Qt destroys a window that nothing holds a reference to.
        self._image_windows.append(window)
        window.show()

        message = f"Opened {len(images)} image(s) in one window."
        if missing:
            message += f"  No image file found for: {', '.join(missing)}"
        self.status.showMessage(message)

    def _forget_image_window(self, window):
        # Identity, not equality: this runs from destroyed(), by which point the
        # underlying C++ object is gone and comparing wrappers can raise.
        self._image_windows = [w for w in self._image_windows if w is not window]

    # -- Plotting -----------------------------------------------------------

    def _update_plot(self):
        paths = self._selected_paths()
        x_field = self._x_field()
        y_fields = self._checked_y_fields()
        monitor_field = self._monitor_field()
        normalize = bool(monitor_field)
        self._update_axis_label()

        # Preserve a manual zoom/pan across redraws triggered by checkbox or
        # dY/dX-toggle changes -- but only while the plot still shows the same
        # data plotted the same way (same files, X field and normalization),
        # otherwise old limits would be meaningless on the new data. Selecting
        # a different file is a different scan, so its range is autoscaled
        # afresh even when the X field happens to be the same one.
        # A two-motor raster can be shown either as a colour map or, with the
        # 2D map button off, as ordinary curves. Read it either way so the
        # button can tell the user whether there is a raster to switch to.
        grid = None
        if paths and y_fields:
            # One field only: a colour map has one colour axis. The hints of
            # a grid scan commonly nominate two detectors, which is right for
            # the curve plot and would otherwise leave the map button dead,
            # so the first is drawn and the title says it is the first.
            grid = read_grid(paths[0], y_fields[0], monitor_field)
        self.grid_btn.setEnabled(grid is not None)
        if not self.grid_btn.isChecked():
            grid = None
        mode = "grid" if grid is not None else "curves"

        path_key = tuple(str(p) for p in paths)
        same_view = (
            path_key == self._last_paths
            and x_field == self._last_x_field
            and monitor_field == self._last_monitor_field
            and mode == self._last_mode
        )
        prev_xlim = prev_ylim = None
        if same_view:
            if mode == "grid" and self.ax.lines or (mode == "grid" and self.ax.collections):
                if not self.ax.get_autoscalex_on():
                    prev_xlim = self.ax.get_xlim()
                if not self.ax.get_autoscaley_on():
                    prev_ylim = self.ax.get_ylim()
            elif mode == "curves" and self.curve_item.listDataItems():
                # pyqtgraph turns auto-range off for an axis the moment the
                # user zooms or pans it, which is the same signal matplotlib's
                # autoscale flag gives.
                auto_x, auto_y = self.curve_vb.autoRangeEnabled()
                x_range, y_range = self.curve_vb.viewRange()
                prev_xlim = None if auto_x else tuple(x_range)
                prev_ylim = None if auto_y else tuple(y_range)
        self._last_paths = path_key
        self._last_x_field = x_field
        self._last_monitor_field = monitor_field
        self._last_mode = mode

        # A colour bar owns its own axes, which ax.clear() does not touch.
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        self.ax.clear()
        self.ax.format_coord = self._default_format_coord
        self._clear_curves()
        if not paths or not x_field or not y_fields:
            self.plot_stack.setCurrentIndex(0)
            self.curve_item.setTitle(
                "Select file(s), an X field and at least one Y field."
            )
            self.stats_table.setRowCount(0)
            return

        if grid is not None:
            if self.plot_stack.currentIndex() != 1:
                self.coord_label.setText("")   # the curve plot's last readout
            self.plot_stack.setCurrentIndex(1)
            self._plot_grid(paths, grid, y_fields[0], monitor_field, prev_xlim, prev_ylim)
            return
        if self.plot_stack.currentIndex() != 0:
            self.coord_label.setText("")       # the colour map's last readout
        self.plot_stack.setCurrentIndex(0)

        if self.trend_btn.isChecked():
            self._draw_trend(paths, x_field, y_fields, monitor_field)
            return

        overlay = len(paths) > 1
        stats_rows: list[tuple[str, dict[str, float | None]]] = []
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        for path in paths:
            fields_needed = {x_field, *y_fields}
            if normalize:
                fields_needed.add(monitor_field)
            try:
                data = read_primary(path, list(fields_needed))
            except OSError as exc:
                self.status.showMessage(f"Could not read {path.name}: {exc}")
                continue
            if x_field not in data:
                continue
            x = data[x_field]
            monitor = data.get(monitor_field) if normalize else None
            if normalize and monitor is None:
                self.status.showMessage(
                    f"{path.name}: monitor field {monitor_field!r} not found -- plotting raw values."
                )
            for y_field in y_fields:
                if y_field not in data:
                    continue
                y = data[y_field]
                if monitor is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        y = np.where(monitor != 0, y / monitor, np.nan)
                label = f"{path.name}: {y_field}" if overlay else y_field
                stats_rows.append((label, curve_stats(x, y)))
                curves.append((label, x, y, path, y_field))

        if self.average_btn.isChecked() and len(paths) > 1:
            curves, stats_rows = self._average_repeats(curves, paths)
        curves = [(label, x, y) for label, x, y, *_rest in curves]

        shape = self.fit_combo.currentText()
        wanted = self.peaks_spin.value()
        fits: list[dict | None] = []
        if shape != FIT_NONE:
            rows: list[tuple[str, dict]] = []
            for (label, x, y), (_label, stats) in zip(curves, stats_rows):
                got = (_fit.peak_fit(x, y, shape, stats.get("fwhm"))
                       if wanted == 1 else
                       _fit.peak_fit_multi(x, y, shape, wanted))
                fits.append(got)
                rows.append((label, {**stats, **got}))
                # A component has no curve of its own, so its row carries
                # only what the fit says about it.
                for number, part in enumerate(got.get("fit_peaks") or [], 1):
                    rows.append((f"    peak {number}", part))
            stats_rows = rows

        for index, (label, x, y) in enumerate(curves):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            self.curve_item.plot(
                x, y, name=label,
                pen=pg.mkPen(color, width=1),
                symbol="o", symbolSize=4,
                symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color),
                # A normalised curve can divide by a zero monitor; break the
                # line at those rather than drawing through them.
                connect="finite",
            )

        for index, (got, (label, x, _y)) in enumerate(zip(fits, curves)):
            if got is None:
                continue
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)),
                               _fit.PEAK_FIT_PLOT_POINTS)
            model = _fit.peak_model(grid, shape, got)
            if model is None:
                continue
            self.curve_item.plot(grid, model,
                                 pen=pg.mkPen(color, width=2))
            if not self.fit_parts_btn.isChecked():
                continue
            for piece in _fit.peak_parts(grid, shape, got):
                self.curve_item.plot(
                    grid, piece,
                    pen=pg.mkPen(color, width=1, style=QtCore.Qt.DotLine))

        self.curve_item.setTitle(None)
        self.curve_item.setLabel("bottom", x_field)
        ylabel = ", ".join(y_fields)
        if normalize:
            ylabel += f" / {monitor_field}"
        self.curve_item.setLabel("left", ylabel)

        show_derivative = self.derivative_btn.isChecked()
        self.curve_item.getAxis("right").setStyle(showValues=show_derivative)
        self.curve_item.getAxis("right").setLabel(
            f"d/d{x_field}" if show_derivative else None
        )
        if show_derivative:
            for index, (label, x, y) in enumerate(curves):
                if x.size < 2:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    dy_dx = np.gradient(y, x)
                finite = np.isfinite(dy_dx)
                color = CURVE_COLORS[index % len(CURVE_COLORS)]
                item = pg.PlotDataItem(
                    x[finite], dy_dx[finite],
                    pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                )
                self.deriv_vb.addItem(item)
                self._derivative_items.append(item)
            self.deriv_vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._sync_derivative_view()

        # Auto-range unless the user had zoomed this same view, in which case
        # the range they chose is put back.
        if prev_xlim is None and prev_ylim is None:
            self.curve_vb.enableAutoRange()
        else:
            if prev_xlim is None:
                self.curve_vb.enableAutoRange(axis=pg.ViewBox.XAxis)
            else:
                self.curve_vb.setXRange(*prev_xlim, padding=0)
            if prev_ylim is None:
                self.curve_vb.enableAutoRange(axis=pg.ViewBox.YAxis)
            else:
                self.curve_vb.setYRange(*prev_ylim, padding=0)
        # Enabling auto-range only queues the recompute, so the view still
        # reads as the old one until Qt gets round to it. Resolve it here, so
        # a redraw leaves the range it is going to settle on.
        self.curve_vb.updateAutoRange()

        self._last_curves = curves
        self._update_stats_table(stats_rows)

    def _sync_derivative_view(self):
        """Keep the derivative's view box on top of the main one."""
        self.deriv_vb.setGeometry(self.curve_vb.sceneBoundingRect())
        self.deriv_vb.linkedViewChanged(self.curve_vb, self.deriv_vb.XAxis)

    def _clear_curves(self):
        """Empty the curve page, legend and derivative overlay alike."""
        self.curve_item.clear()
        for item in self._derivative_items:
            self.deriv_vb.removeItem(item)
        self._derivative_items = []
        legend = self.curve_item.legend
        if legend is not None:
            legend.clear()
        self.curve_item.setTitle(None)

    def _on_curve_mouse_moved(self, event):
        """Report the pointer, the way the matplotlib readout does."""
        if self.plot_stack.currentIndex() != 0:
            return
        pos = event[0]
        if not self.curve_item.sceneBoundingRect().contains(pos):
            self.coord_label.setText("")
            return
        point = self.curve_vb.mapSceneToView(pos)
        text = f"{self._x_field() or 'x'}={point.x():.6g}   y={point.y():.6g}"
        if self.derivative_btn.isChecked() and self._derivative_items:
            other = self.deriv_vb.mapSceneToView(pos)
            text += f"   d/dx={other.y():.6g}"
        self.coord_label.setText(text)

    def _plot_grid(self, paths, grid, value_field, monitor_field, prev_xlim, prev_ylim):
        """Draw a two-motor raster as a colour map with a scale bar."""
        label = value_field + (f" / {monitor_field}" if monitor_field else "")

        x_edges, y_edges = axis_edges(grid["x"]), axis_edges(grid["y"])
        low = high = None
        extend = "neither"
        if self.clip_btn.isChecked():
            low, high, extend = clipped_limits(grid["z"])
        mesh = self.ax.pcolormesh(x_edges, y_edges, grid["z"], shading="flat",
                                  vmin=low, vmax=high)

        self.ax.set_xlabel(grid["x_name"])
        self.ax.set_ylabel(grid["y_name"])
        title = f"{paths[0].name}  —  {len(grid['x'])} x {len(grid['y'])} grid"
        if len(paths) > 1:
            # Two colour maps cannot share one pair of axes.
            title += f"   (first of {len(paths)} selected)"
        if len(self._checked_y_fields()) > 1:
            title += f"   (showing {value_field})"
        missing = int(np.count_nonzero(~np.isfinite(grid["z"])))
        if missing:
            title += f"   [{missing} point(s) not measured]"
        if low is not None:
            title += f"   [scale {low:.6g} to {high:.6g}]"
        self.ax.set_title(title, fontsize=9)
        self.ax.grid(False)

        # Autoscale cannot be relied on here: pcolormesh adds a *collection*,
        # which relim() does not look at, so the X range was left at the empty
        # default while Y came out right. Set both ends from the cell edges.
        self.ax.set_xlim(x_edges[0], x_edges[-1])
        self.ax.set_ylim(y_edges[0], y_edges[-1])
        if prev_xlim is not None:
            self.ax.set_xlim(prev_xlim)
        if prev_ylim is not None:
            self.ax.set_ylim(prev_ylim)
        # Report the value under the pointer too. The stock readout gives
        # only x and y, so replace it outright rather than append to it.
        def format_coord(x, y):
            parts = [f"{grid['x_name']}={x:.6g}", f"{grid['y_name']}={y:.6g}"]
            column = int(np.searchsorted(x_edges, x) - 1)
            row = int(np.searchsorted(y_edges, y) - 1)
            if 0 <= column < len(grid["x"]) and 0 <= row < len(grid["y"]):
                value = grid["z"][row, column]
                parts.append(
                    f"{label}=" + ("no data" if not np.isfinite(value) else f"{value:.6g}")
                )
            return "   ".join(parts)

        self.ax.format_coord = format_coord

        # Lay the axes out *before* adding the colour bar. tight_layout()
        # afterwards gives the bar's strip back to the plot and the bar ends
        # up drawn on top of the map.
        self.figure.tight_layout()
        # The arrows say the scale does not reach its own data, rather than
        # leaving the clipping to be noticed.
        self._colorbar = self.figure.colorbar(mesh, ax=self.ax, extend=extend)
        self._colorbar.set_label(label)
        self.canvas.draw_idle()

        stats = grid_stats(grid)
        def at(point):
            return "" if point is None else f"{point[0]:.6g}, {point[1]:.6g}"
        self.stats_table.setRowCount(1)
        values = [
            label,
            "" if stats["max_z"] is None else f"{stats['max_z']:.6g}",
            at(stats["max_at"]),
            "" if stats["min_z"] is None else f"{stats['min_z']:.6g}",
            at(stats["min_at"]),
            "", "",
        ]
        for column, text in enumerate(values):
            self.stats_table.setItem(0, column, QtWidgets.QTableWidgetItem(text))
        self.stats_table.resizeColumnsToContents()

    def _refresh_trend_axes(self, paths):
        """Offer the scan number, then whatever moved across the selection."""
        choices = [_trends.TREND_X_SCAN] + _trends.baseline_trend_fields(paths)
        current = self.trend_x_combo.currentText()
        if [self.trend_x_combo.itemText(i)
                for i in range(self.trend_x_combo.count())] == choices:
            return
        self._syncing_fields = True
        try:
            self.trend_x_combo.clear()
            self.trend_x_combo.addItems(choices)
            if current in choices:
                self.trend_x_combo.setCurrentText(current)
        finally:
            self._syncing_fields = False

    def _draw_trend(self, paths, x_field, y_fields, monitor_field):
        """One point per scan: a fitted quantity against what was stepped."""
        self._refresh_trend_axes(paths)
        shape = self.fit_combo.currentText()
        if shape == FIT_NONE:
            self.curve_item.setTitle("Choose a fit shape to trend.")
            self.stats_table.setRowCount(0)
            self._last_curves = []
            return
        against = self.trend_x_combo.currentText() or _trends.TREND_X_SCAN
        key = _trends.TREND_QUANTITIES[self.trend_y_combo.currentText()]
        axis = _trends.trend_axis(paths, against)

        series: dict[str, list[tuple[float, float]]] = {}
        rows, skipped = [], 0
        for path in paths:
            where = axis.get(str(path))
            data = read_primary(path, sorted({x_field, *y_fields}))
            x = data.get(x_field)
            monitor = data.get(monitor_field) if monitor_field else None
            for y_field in y_fields:
                y = data.get(y_field)
                if x is None or y is None:
                    continue
                if monitor is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        y = np.where(monitor != 0, y / monitor, np.nan)
                if where is None:
                    skipped += 1
                    continue
                got = _fit.peak_fit(x, y, shape)
                value = got.get(key)
                if value is None:
                    continue
                series.setdefault(y_field, []).append((where, float(value)))
                rows.append((f"{scan_number(path)}: {y_field}", got))

        self._clear_curves()
        for index, (name, points) in enumerate(sorted(series.items())):
            points.sort()
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            self.curve_item.plot(
                [p[0] for p in points], [p[1] for p in points], name=name,
                pen=pg.mkPen(color, width=1), symbol="o", symbolSize=6,
                symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color))
        self.curve_item.setLabel("bottom", against)
        self.curve_item.setLabel("left", self.trend_y_combo.currentText())
        title = f"{sum(len(v) for v in series.values())} fitted point(s)"
        if skipped:
            title += f"   [{skipped} scan(s) have no {against}]"
        self.curve_item.setTitle(title)
        self.curve_vb.enableAutoRange(); self.curve_vb.updateAutoRange()
        self._last_curves = [
            (name, np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
            for name, pts in sorted(series.items())]
        self._update_stats_table(rows)

    def _export_curves(self):
        """Write the plotted curves as columns, with their provenance."""
        if not self._last_curves:
            QtWidgets.QMessageBox.information(
                self, "Nothing to export", "Plot some curves first.")
            return
        start = str((self._directory or Path(".")) / "curves.csv")
        name, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export the plotted curves", start, _export.EXPORT_FILTERS)
        if not name:
            return
        header = [
            f"scans: {', '.join(p.name for p in self._selected_paths())}",
            f"x: {self._x_field()}",
        ]
        if self._monitor_field():
            header.append(f"normalised by: {self._monitor_field()}")
        if self.average_btn.isChecked():
            header.append("repeats averaged")
        if self.fit_combo.currentText() != FIT_NONE:
            header.append(f"fit: {self.fit_combo.currentText()}, "
                          f"{self.peaks_spin.value()} peak(s)")
        try:
            written = _export.write_columns(
                name,
                [("curves", "", _export.table_columns(
                    self._last_curves, self._x_field() or "x"))],
                description=header,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"{type(exc).__name__}: {exc}")
            return
        self.statusBar().showMessage(
            "Wrote " + ", ".join(str(w) for w in written), 8000)

    def _average_repeats(self, curves, paths):
        """Collapse each set of repeats into one curve, per Y field.

        Grouped by what the scans are, not by what their numbers look like
        -- see :func:`da6idb.combine.scan_groups`. Within a group the
        points still have to line up: two scans of one thing do not step
        the same readback twice, so one whose X has drifted more than half
        a step is left out rather than smeared into the mean.
        """
        rank = {str(path): index for index, path in enumerate(paths)}
        groups = _combine.scan_groups(paths)
        belongs = {str(path): number
                   for number, group in enumerate(groups) for path in group}

        pooled: dict[tuple, list] = {}
        for label, x, y, path, y_field in curves:
            pooled.setdefault((belongs[str(path)], y_field), []).append(
                (rank[str(path)], label, x, y, path))

        out, rows = [], []
        for key in sorted(pooled, key=lambda k: (min(i for i, *_ in pooled[k]), k[1])):
            members = sorted(pooled[key])
            first = members[0]
            if len(members) == 1:
                _i, label, x, y, _p = first
                out.append((label, x, y))
                rows.append((label, curve_stats(x, y)))
                continue
            base_x = first[2]
            kept = [m for m in members if _combine.same_grid(base_x, m[2])]
            dropped = len(members) - len(kept)
            rows_n = max(m[2].size for m in kept)
            mean = _combine.stack_mean([m[3] for m in kept], rows_n)
            numbers = "+".join(scan_number(m[4]) for m in kept)
            label = f"{numbers}: {key[1]}"
            if dropped:
                label += f" ({dropped} off-grid left out)"
            out.append((label, base_x[:rows_n], mean))
            rows.append((label, curve_stats(base_x[:rows_n], mean)))
        return out, rows

    def _update_stats_table(self, rows: list[tuple[str, dict[str, float | None]]]):
        def fmt(value: float | None) -> str:
            return "" if value is None else f"{value:.6g}"

        shape = self.fit_combo.currentText()
        # The mixing gets a column only where the shape has one to report,
        # so three shapes of the four do not carry a column of blanks.
        fitted = ([] if shape == FIT_NONE
                  else _fit.fit_columns(shape in _fit.PEAK_MIX))
        keys = ["fit_center", "fit_fwhm", "fit_height", "fit_area", "fit_r2"]
        if shape in _fit.PEAK_MIX and fitted:
            keys.insert(2, "fit_eta")
        headings = STATS_COLUMNS + fitted
        if [self.stats_table.horizontalHeaderItem(c).text()
                for c in range(self.stats_table.columnCount())] != headings:
            self.stats_table.setColumnCount(len(headings))
            self.stats_table.setHorizontalHeaderLabels(headings)

        self.stats_table.setRowCount(len(rows))
        for row, (label, stats) in enumerate(rows):
            values = [
                label,
                fmt(stats.get("max_y")),
                fmt(stats.get("max_x")),
                fmt(stats.get("min_y")),
                fmt(stats.get("min_x")),
                fmt(stats.get("center")),
                fmt(stats.get("fwhm")),
            ] + [fmt(stats.get(key)) for key in keys[:len(fitted)]]
            for col, text in enumerate(values):
                self.stats_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        self.stats_table.resizeColumnsToContents()

    # -- Baseline -------------------------------------------------------

    def _show_baseline_window(self):
        """Open (or raise) the separate baseline window."""
        if self._baseline_window is None:
            window = QtWidgets.QDialog(self)
            window.setWindowTitle("Baseline (before / after)")
            window.resize(760, 560)
            layout = QtWidgets.QVBoxLayout(window)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.addWidget(self._baseline_panel)
            self._baseline_window = window
        # Show first: _update_baseline() does nothing while the window is
        # hidden, so filling the table before showing it would leave it empty.
        self._baseline_window.show()
        self._baseline_window.raise_()
        self._baseline_window.activateWindow()
        self._refresh_baseline_combo(self._selected_paths())
        self._update_baseline()


    def _update_baseline(self):
        window = getattr(self, "_baseline_window", None)
        if window is None or not window.isVisible():
            return  # nothing on screen to fill in; reading the file can wait
        name = self.baseline_combo.currentText()
        self.baseline_table.setRowCount(0)
        if not name:
            return
        path = next((p for p in self._selected_paths() if p.name == name), None)
        if path is None:
            return
        try:
            baseline = read_baseline(path)
        except OSError as exc:
            self.status.showMessage(f"Could not read baseline of {name}: {exc}")
            return

        self.baseline_table.setSortingEnabled(False)
        self.baseline_table.setRowCount(len(baseline))
        for row, field in enumerate(sorted(baseline)):
            before, after = baseline[field]
            is_scalar = isinstance(before, float) and isinstance(after, float)
            delta = f"{after - before:.6g}" if is_scalar else ""
            values = [
                field,
                f"{before:.6g}" if isinstance(before, float) else str(before),
                f"{after:.6g}" if isinstance(after, float) else str(after),
                delta,
            ]
            for col, text in enumerate(values):
                self.baseline_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        self.baseline_table.setSortingEnabled(True)
        self.baseline_table.resizeColumnsToContents()


def main():
    initial_directory = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = QtWidgets.QApplication(sys.argv)
    window = HdfViewer(initial_directory)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
