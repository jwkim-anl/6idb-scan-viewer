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


# Enough distinct colours that a handful of files times a handful of fields
# stay tellable apart, in a fixed order so a curve keeps its colour across a
# redraw.
# Faint: a grid is a reference, not a mesh drawn over the data.
GRID_ALPHA = 0.25
# How far apart stacked Y axes sit, in points. Wide enough for six figures
# and a label without the next one touching it.
AXIS_SPACING = 54
# How many fits to remember. One curve of one scan under one set of fit
# settings is one entry, so a selection of twenty scans with a couple of
# fields ticked fits inside this several times over.
FIT_CACHE_MAX = 512

FIT_NONE = "None"
AVERAGE_NONE, AVERAGE_REPEATS, AVERAGE_ALL = "None", "Repeats", "All selected"
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
        self._last_trend: list = []
        self._fit_cache: dict = {}
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

        # Two rows, not one: twelve controls side by side set the window's
        # minimum width wider than the plot needs. Display on the first,
        # everything about fitting on the second.
        fit_row = QtWidgets.QHBoxLayout()
        # On a line of its own above the buttons, with the full width to
        # itself: sharing the row cost the buttons space and still elided
        # the Y list as soon as more than a couple of fields were ticked.
        self.axis_label = QtWidgets.QLabel("")
        self.axis_label.setWordWrap(True)
        self.axis_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.axis_label)

        axis_row = QtWidgets.QHBoxLayout()
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
        fit_row.addWidget(QtWidgets.QLabel("Fit"))
        self.fit_combo = QtWidgets.QComboBox()
        self.fit_combo.addItem(FIT_NONE)
        self.fit_combo.addItems(_fit.PEAK_SHAPES)
        self.fit_combo.setToolTip(
            "Fit this shape plus a sloping line to every plotted curve.\n"
            "The background is always fitted, never assumed flat."
        )
        self.fit_combo.currentIndexChanged.connect(self._update_plot)
        fit_row.addWidget(self.fit_combo)

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
        fit_row.addWidget(self.peaks_spin)

        self.trend_btn = QtWidgets.QPushButton("Trend")
        self.trend_btn.setCheckable(True)
        self.trend_btn.setToolTip(
            "Plot one point per scan instead of the curves: a fitted\n"
            "quantity against whatever was stepped between them.\n"
            "Needs a fit, and at least two scans selected."
        )
        self.trend_btn.toggled.connect(self._update_plot)
        fit_row.addWidget(self.trend_btn)

        self.trend_x_combo = QtWidgets.QComboBox()
        self.trend_x_combo.setToolTip(
            "What to plot against. The baseline fields offered are the ones\n"
            "that actually moved across the selection, ranked by how far\n"
            "they moved relative to their own drift within a scan -- so the\n"
            "thing somebody deliberately stepped comes out at the top."
        )
        self.trend_x_combo.currentIndexChanged.connect(self._update_plot)
        fit_row.addWidget(self.trend_x_combo)

        self.trend_y_combo = QtWidgets.QComboBox()
        self.trend_y_combo.setToolTip(
            "Which fitted quantity to follow. With more than one peak each\n"
            "component is offered separately -- peak 1 is the leftmost in\n"
            "every scan, so a line follows one peak rather than jumping\n"
            "whenever two swap which is the taller."
        )
        self.trend_y_combo.currentIndexChanged.connect(self._update_plot)
        fit_row.addWidget(self.trend_y_combo)

        self.trend_export_btn = QtWidgets.QPushButton("Export trend…")
        self.trend_export_btn.setToolTip(
            "Write the points on the trend plot: the axis, and the fitted\n"
            "quantity for each Y field. The header records the fit they\n"
            "came from and the scans that were fitted."
        )
        self.trend_export_btn.setEnabled(False)
        self.trend_export_btn.clicked.connect(self._export_trend)
        fit_row.addWidget(self.trend_export_btn)

        self.separate_y_btn = QtWidgets.QPushButton("Separate Y")
        self.separate_y_btn.setCheckable(True)
        self.separate_y_btn.setToolTip(
            "Give every curve its own Y axis, stacked out to the right in\n"
            "the curve's own colour. For fields whose scales have nothing\n"
            "to do with each other -- a detector counting in the hundreds\n"
            "of thousands beside a capacitance of 0.9 -- which on one axis\n"
            "leaves the smaller one a flat line along the bottom."
        )
        self.separate_y_btn.toggled.connect(self._update_plot)
        axis_row.addWidget(self.separate_y_btn)

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

        axis_row.addWidget(QtWidgets.QLabel("Average"))
        self.average_combo = QtWidgets.QComboBox()
        self.average_combo.addItems((AVERAGE_NONE, AVERAGE_REPEATS, AVERAGE_ALL))
        self.average_combo.setToolTip(
            "Repeats: average only the scans that agree on plan, motors,\n"
            "sample position and temperature, so a series does not quietly\n"
            "become one clean curve that is not one measurement.\n\n"
            "All selected: average everything selected that shares a grid,\n"
            "however far apart the baseline says they sat. For a long run\n"
            "whose temperature drifts a little between repeats, which the\n"
            "grouping reads as a series and splits."
        )
        self.average_combo.currentIndexChanged.connect(self._update_plot)
        axis_row.addWidget(self.average_combo)

        self.fit_parts_btn = QtWidgets.QPushButton("Parts")
        self.fit_parts_btn.setCheckable(True)
        self.fit_parts_btn.setToolTip(
            "Draw the fitted background and each component on it, as well\n"
            "as the one summed model."
        )
        self.fit_parts_btn.toggled.connect(self._update_plot)
        fit_row.addWidget(self.fit_parts_btn)
        axis_row.addStretch(1)
        layout.addLayout(axis_row)
        fit_row.addStretch(1)
        layout.addLayout(fit_row)

        canvas_box = QtWidgets.QVBoxLayout()

        # Both pages are matplotlib: one toolkit, one look, and the same
        # zoom/pan/save toolbar and print-quality output on the curves as
        # on the colour map. They take turns in a stack rather than
        # fighting over the space.
        self.plot_stack = QtWidgets.QStackedWidget()

        self.curve_figure = Figure(figsize=(6, 4))
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        self.curve_ax = self.curve_figure.add_subplot(111)
        # dY/dX shares the X axis and wants its own Y scale.
        self.deriv_ax = self.curve_ax.twinx()
        self.deriv_ax.set_visible(False)
        # Created per redraw when "Separate Y" is on, and removed with the
        # curves: an axis outlives a clear() where a line does not.
        self._extra_axes: list = []
        self.curve_toolbar = NavigationToolbar2QT(self.curve_canvas, panel)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        # Kept so the grid's own readout can be swapped in and out.
        self._default_format_coord = self.ax.format_coord
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
        if hasattr(self.curve_toolbar, "locLabel"):
            self.curve_toolbar.locLabel.hide()
        self.curve_toolbar.set_message = self.coord_label.setText

        # A toolbar per page, each travelling with its own canvas.
        curve_page = QtWidgets.QWidget()
        curve_box = QtWidgets.QVBoxLayout(curve_page)
        curve_box.setContentsMargins(0, 0, 0, 0)
        curve_box.addWidget(self.curve_toolbar)
        curve_box.addWidget(self.curve_canvas, 1)
        self.plot_stack.addWidget(curve_page)

        grid_page = QtWidgets.QWidget()
        grid_box = QtWidgets.QVBoxLayout(grid_page)
        grid_box.setContentsMargins(0, 0, 0, 0)
        grid_box.addWidget(self.toolbar)
        grid_box.addWidget(self.canvas, 1)
        self.plot_stack.addWidget(grid_page)

        # The trend is not the same plot drawn differently -- it has one
        # point per scan where the other has one per point, and its axes
        # are nothing to do with the curves'. So it sits beside them
        # rather than taking their place, and you can watch a fit and the
        # trend it feeds at once. Draggable, because how much room it
        # wants depends on what is on it.
        self.trend_figure = Figure(figsize=(3.4, 4))
        self.trend_canvas = FigureCanvasQTAgg(self.trend_figure)
        self.trend_ax = self.trend_figure.add_subplot(111)
        self.trend_plot = QtWidgets.QWidget()
        trend_box = QtWidgets.QVBoxLayout(self.trend_plot)
        trend_box.setContentsMargins(0, 0, 0, 0)
        trend_box.addWidget(NavigationToolbar2QT(self.trend_canvas, panel))
        trend_box.addWidget(self.trend_canvas, 1)
        self.trend_plot.hide()

        plots = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        plots.addWidget(self.plot_stack)
        plots.addWidget(self.trend_plot)
        plots.setStretchFactor(0, 3)
        plots.setStretchFactor(1, 2)
        plots.setChildrenCollapsible(False)
        self._plot_splitter = plots
        canvas_box.addWidget(plots, 1)
        canvas_box.addWidget(self.coord_label)

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
            elif mode == "curves" and self.curve_ax.lines:
                # The autoscale flag is matplotlib's own record of whether
                # the user has zoomed or panned that axis.
                if not self.curve_ax.get_autoscalex_on():
                    prev_xlim = self.curve_ax.get_xlim()
                if not self.curve_ax.get_autoscaley_on():
                    prev_ylim = self.curve_ax.get_ylim()
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
            self.curve_ax.set_title(
                "Select file(s), an X field and at least one Y field.",
                fontsize=9)
            self.curve_canvas.draw_idle()
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

        if self.average_combo.currentText() != AVERAGE_NONE and len(paths) > 1:
            curves, stats_rows = self._average_repeats(curves, paths)
        curves = [(label, x, y) for label, x, y, *_rest in curves]

        shape = self.fit_combo.currentText()
        wanted = self.peaks_spin.value()
        fits: list[dict | None] = []
        if shape != FIT_NONE:
            rows: list[tuple[str, dict]] = []
            for (label, x, y), (_label, stats) in zip(curves, stats_rows):
                got = self._fitted(x, y, shape, wanted, stats.get("fwhm"))
                fits.append(got)
                rows.append((label, {**stats, **got}))
                # A component has no curve of its own, so its row carries
                # only what the fit says about it.
                for number, part in enumerate(got.get("fit_peaks") or [], 1):
                    rows.append((f"    peak {number}", part))
            stats_rows = rows

        # One axis per curve, or one axis for all of them. The first is
        # always the left-hand axis; the rest are stacked outwards to the
        # right, each tinted like the curve it belongs to so there is no
        # guessing which scale reads which line.
        separate = self.separate_y_btn.isChecked() and len(curves) > 1
        axes = [self.curve_ax]
        if separate:
            for index in range(1, len(curves)):
                extra = self.curve_ax.twinx()
                extra.spines["right"].set_position(
                    ("outward", AXIS_SPACING * (index - 1)))
                color = CURVE_COLORS[index % len(CURVE_COLORS)]
                extra.spines["right"].set_color(color)
                extra.tick_params(axis="y", colors=color, labelsize=8)
                extra.yaxis.label.set_color(color)
                self._extra_axes.append(extra)
                axes.append(extra)
        else:
            axes = [self.curve_ax] * len(curves)

        for index, (label, x, y) in enumerate(curves):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            # A normalised curve can divide by a zero monitor. matplotlib
            # breaks a line at NaN of its own accord, which is what is
            # wanted: the gap shows rather than being drawn through.
            axes[index].plot(x, y, label=label, color=color, lw=1,
                             marker="o", ms=3)
            if separate:
                axes[index].set_ylabel(label, fontsize=8)

        for index, (got, (label, x, _y)) in enumerate(zip(fits, curves)):
            if got is None:
                continue
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)),
                               _fit.PEAK_FIT_PLOT_POINTS)
            model = _fit.peak_model(grid, shape, got)
            if model is None:
                continue
            # On the curve's own axis, so a fit lies on the data it fitted
            # rather than wherever the left-hand scale happens to put it.
            axes[index].plot(grid, model, color=color, lw=1.6)
            if not self.fit_parts_btn.isChecked():
                continue
            for piece in _fit.peak_parts(grid, shape, got):
                axes[index].plot(grid, piece, color=color, lw=0.9, ls=":")

        self.curve_ax.set_xlabel(x_field)
        ylabel = ", ".join(y_fields)
        if normalize:
            ylabel += f" / {monitor_field}"
        if not separate:
            self.curve_ax.set_ylabel(ylabel)
        self.curve_ax.grid(True, alpha=GRID_ALPHA)
        if len(curves) > 1:
            # One curve is already named by the Y label; a legend for it
            # would only cover data. Gathered across the axes, since each
            # one knows only its own line.
            handles, labels = [], []
            for axis in dict.fromkeys(axes):
                got_h, got_l = axis.get_legend_handles_labels()
                handles += got_h
                labels += got_l
            if handles:
                self.curve_ax.legend(handles, labels, fontsize=7, loc="best")

        show_derivative = self.derivative_btn.isChecked()
        self.deriv_ax.set_visible(show_derivative)
        # One slot beyond the last separate axis: those occupy slots 0 to
        # n-1, so the derivative takes slot n rather than landing on top of
        # the last of them.
        self.deriv_ax.spines["right"].set_position(
            ("outward", AXIS_SPACING * len(self._extra_axes)))
        if show_derivative:
            for index, (label, x, y) in enumerate(curves):
                if x.size < 2:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    dy_dx = np.gradient(y, x)
                color = CURVE_COLORS[index % len(CURVE_COLORS)]
                self.deriv_ax.plot(x, dy_dx, color=color, lw=1, ls="--")
            self.deriv_ax.set_ylabel(f"d/d{x_field}")
            self.deriv_ax.relim()
            self.deriv_ax.autoscale_view()

        # Autoscale unless the user had zoomed this same view, in which
        # case the range they chose is put back.
        if prev_xlim is not None:
            self.curve_ax.set_xlim(prev_xlim)
        if prev_ylim is not None:
            self.curve_ax.set_ylim(prev_ylim)

        def report(x, y):
            text = f"{x_field or 'x'}={x:.6g}   y={y:.6g}"
            if show_derivative:
                # The pointer is over the left axis; say what that height
                # is on the right one too, which is the question a twin
                # axis always provokes.
                low, high = self.curve_ax.get_ylim()
                other_low, other_high = self.deriv_ax.get_ylim()
                if high > low:
                    scaled = other_low + (y - low) / (high - low) * (
                        other_high - other_low)
                    text += f"   d/dx={scaled:.6g}"
            return text

        self.curve_ax.format_coord = report
        self.curve_figure.tight_layout()
        self.curve_canvas.draw_idle()

        self._last_curves = curves
        self._update_stats_table(stats_rows)
        self._draw_trend(paths, x_field, y_fields, monitor_field)

    def _clear_curves(self):
        """Empty the curve page, legend and every extra Y axis alike."""
        for extra in self._extra_axes:
            extra.remove()
        self._extra_axes = []
        self.curve_ax.clear()
        self.deriv_ax.clear()
        # clear() resets a twin to a left-hand axis, which put the dY/dX
        # label back over the left one's. twinx() only sets this at birth.
        self.deriv_ax.yaxis.set_label_position("right")
        self.deriv_ax.yaxis.tick_right()
        self.deriv_ax.set_visible(False)

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
        # Block the box, not _syncing_fields: that flag guards the field
        # table's own handler, and _update_plot does not consult it -- so
        # refilling a box here would emit currentIndexChanged, call
        # _update_plot, and come straight back round to this line.
        blocked = QtCore.QSignalBlocker(self.trend_x_combo)
        try:
            self.trend_x_combo.clear()
            self.trend_x_combo.addItems(choices)
            if current in choices:
                self.trend_x_combo.setCurrentText(current)
        finally:
            del blocked

    def _fitted(self, x, y, shape, peaks, fwhm_seed=None):
        """The fit of this curve under these settings, computed once.

        Nothing the trend controls changes a fit. Which number it plots and
        which axis it plots against are both read out of a fit already made
        -- but each goes through a full redraw, and that refitted every
        selected scan twice over, once for the curves and once for the
        trend. Twenty-one scans under a three-peak pseudo-Voigt took 16.5 s
        to change the plotted quantity; cached it is immediate.

        Keyed on the numbers themselves rather than on the scan, so an
        averaged curve, a normalised one and the raw one are three
        different entries, and editing nothing cannot return a stale fit.
        The seed is not part of the key because it is derived from the same
        two arrays: it is passed in only to save recomputing it where the
        caller has it already.
        """
        key = (shape, int(peaks), x.tobytes(), y.tobytes())
        got = self._fit_cache.get(key)
        if got is not None:
            return got
        if int(peaks) == 1:
            if fwhm_seed is None:
                fwhm_seed = curve_stats(x, y).get("fwhm")
            got = _fit.peak_fit(x, y, shape, fwhm_seed)
        else:
            got = _fit.peak_fit_multi(x, y, shape, int(peaks))
        if len(self._fit_cache) >= FIT_CACHE_MAX:
            self._fit_cache.clear()
        self._fit_cache[key] = got
        return got

    def _refresh_trend_quantities(self):
        """Offer each component's numbers once there is more than one."""
        shape = self.fit_combo.currentText()
        choices = _trends.trend_choices(
            self.peaks_spin.value(), shape in _fit.PEAK_MIX)
        existing = [self.trend_y_combo.itemText(i)
                    for i in range(self.trend_y_combo.count())]
        if existing == list(choices):
            return choices
        current = self.trend_y_combo.currentText()
        # Block the box, not _syncing_fields: that flag guards the field
        # table's own handler, and _update_plot does not consult it -- so
        # refilling a box here would emit currentIndexChanged, call
        # _update_plot, and come straight back round to this line.
        blocked = QtCore.QSignalBlocker(self.trend_y_combo)
        try:
            self.trend_y_combo.clear()
            self.trend_y_combo.addItems(choices)
            if current in choices:
                self.trend_y_combo.setCurrentText(current)
            else:
                # Going from one peak to several, "Fit center" becomes
                # "Peak 1 center"; keep the quantity rather than resetting.
                tail = current.split(" ", 1)[-1] if current else ""
                for label in choices:
                    if label.split(" ", 2)[-1] == tail:
                        self.trend_y_combo.setCurrentText(label)
                        break
        finally:
            del blocked
        return choices

    def _draw_trend(self, paths, x_field, y_fields, monitor_field):
        """Fill the plot beside the curves: one point per scan.

        Drawn from the same fit the curves carry, so the point on the right
        is the peak on the left and the two cannot disagree.
        """
        if not self.trend_btn.isChecked():
            self.trend_plot.hide()
            self._last_trend = []
            self.trend_export_btn.setEnabled(False)
            return
        first_time = not self.trend_plot.isVisible()
        self.trend_plot.show()
        self._last_trend = []
        self.trend_export_btn.setEnabled(False)
        if first_time:
            # A splitter gives a pane its size hint, not its stretch factor,
            # until something drags it; a fresh plot's hint is small enough
            # that the trend came up a slot too narrow to read its own axis
            # labels. Set the split the first time it is shown and leave it
            # alone after, so a width chosen by hand survives a redraw.
            width = max(self._plot_splitter.width(), 600)
            self._plot_splitter.setSizes([int(width * 0.62), int(width * 0.38)])
        self.trend_ax.clear()
        shape = self.fit_combo.currentText()
        if shape == FIT_NONE or len(paths) < 2:
            self.trend_ax.set_title(
                "Choose a fit shape" if shape == FIT_NONE
                else "Select two or more scans", fontsize=9)
            self.trend_figure.tight_layout()
            self.trend_canvas.draw_idle()
            return
        self._refresh_trend_axes(paths)
        choices = self._refresh_trend_quantities()
        choice = choices.get(self.trend_y_combo.currentText())
        if choice is None:
            return
        wanted = self.peaks_spin.value()
        against = self.trend_x_combo.currentText() or _trends.TREND_X_SCAN
        axis = _trends.trend_axis(paths, against)

        series: dict[str, list[tuple[float, float]]] = {}
        skipped = 0
        for path in paths:
            where = axis.get(str(path))
            # The monitor has to be asked for, or normalising is quietly
            # dropped here while the curves and the table apply it, and the
            # point on the trend stops being the peak in the table.
            needed = {x_field, *y_fields} | ({monitor_field} if monitor_field
                                             else set())
            data = read_primary(path, sorted(needed))
            x = data.get(x_field)
            monitor = data.get(monitor_field) if monitor_field else None
            for y_field in y_fields:
                y = data.get(y_field)
                if x is None or y is None:
                    continue
                if where is None:
                    skipped += 1
                    continue
                if monitor is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        y = np.where(monitor != 0, y / monitor, np.nan)
                # Through the same door as the curves above, so the point
                # on the trend is the fit in the table and not a second
                # one that happened to be seeded differently.
                got = self._fitted(x, y, shape, wanted)
                value = _trends.trend_value(got, choice)
                if value is not None:
                    series.setdefault(y_field, []).append((where, float(value)))

        drawn = []
        shared = None
        for name, points in sorted(series.items()):
            points.sort()
            axis_values = [p[0] for p in points]
            if shared is None:
                shared = np.array(axis_values, dtype=float)
            # Hand the same array to every series that landed on the same
            # points, so the export writes one axis column rather than one
            # per series. Not the same thing as equal values -- a Y field
            # whose fit was refused on one scan has genuinely fewer points
            # and keeps an axis column of its own.
            x_values = (shared if list(shared) == axis_values
                        else np.array(axis_values, dtype=float))
            drawn.append((name, x_values, np.array([p[1] for p in points],
                                                   dtype=float)))
        self._last_trend = drawn
        self.trend_export_btn.setEnabled(bool(drawn))
        for index, (name, x_values, y_values) in enumerate(drawn):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            self.trend_ax.plot(x_values, y_values, label=name, color=color,
                               lw=1, marker="o", ms=5)
        self.trend_ax.set_xlabel(against)
        self.trend_ax.set_ylabel(self.trend_y_combo.currentText())
        self.trend_ax.grid(True, alpha=GRID_ALPHA)
        if len(drawn) > 1:
            self.trend_ax.legend(fontsize=7, loc="best")
        fitted = sum(len(v) for v in series.values())
        title = f"{fitted} fitted point(s) of {len(paths) * len(y_fields)}"
        if skipped:
            title += f"   [{skipped} with no {against}]"
        self.trend_ax.set_title(title, fontsize=9)
        self.trend_figure.tight_layout()
        self.trend_canvas.draw_idle()

    def _export_trend(self):
        """Write the trend's points, with the fit they came from."""
        if not self._last_trend:
            QtWidgets.QMessageBox.information(
                self, "Nothing to export", "The trend plot is empty.")
            return
        against = self.trend_x_combo.currentText() or _trends.TREND_X_SCAN
        start = str((self._directory or Path(".")) / "trend.csv")
        name, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export the trend", start, _export.EXPORT_FILTERS)
        if not name:
            return
        header = [
            f"trend: {self.trend_y_combo.currentText()} against {against}",
            f"fit: {self.fit_combo.currentText()}, "
            f"{self.peaks_spin.value()} peak(s)",
            f"scans: {', '.join(p.name for p in self._selected_paths())}",
            f"curve x: {self._x_field()}",
        ]
        if self._monitor_field():
            header.append(f"normalised by: {self._monitor_field()}")
        try:
            written = _export.write_columns(
                name,
                [("trend", "", _export.table_columns(self._last_trend, against))],
                description=header,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"{type(exc).__name__}: {exc}")
            return
        self.statusBar().showMessage(
            "Wrote " + ", ".join(str(w) for w in written), 8000)

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
        if self.average_combo.currentText() != AVERAGE_NONE:
            header.append(f"averaged: {self.average_combo.currentText().lower()}")
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
        if self.average_combo.currentText() == AVERAGE_ALL:
            # Told outright that these are repeats. The grid check below
            # still stands -- curves on different points cannot be averaged
            # whoever says they are the same measurement -- but what the
            # baseline says about where they sat no longer splits them.
            belongs = {str(path): 0 for path in paths}
        else:
            belongs = {str(path): number
                       for number, group in enumerate(_combine.scan_groups(paths))
                       for path in group}

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
            kept = [m for m in members if _combine.same_grid(first[2], m[2])]
            dropped = len(members) - len(kept)
            rows_n = max(m[2].size for m in kept)
            # The X of the longest of them, not of the first: they agree
            # over the overlap by construction, and the mean runs to the
            # longest, so taking the first's would leave the curve's X
            # short of its own Y whenever the first scan is the stopped one.
            base_x = max((m[2] for m in kept), key=np.size)
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
