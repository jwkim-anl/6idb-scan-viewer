"""A 3D viewer for the gridded reciprocal-space maps written by ``hkl_grid.py``.

The same scene as the HKL viewer -- a volume on black with H/K/L axes, an
arbitrary slice plane and an opacity ramp -- but it reads a finished ``.vti``
and nothing else. There is no conversion here and no detector data, so it opens
in about a second where the HKL viewer has to convert every pixel first, and it
needs neither xrayutilities nor h5py: a grid can be looked at on a machine that
could not have produced it.

    python vti_viewer.py pva_tise2_2_00152_hkl_grid.vti
    python vti_viewer.py /data/grids      # a folder of them
    python vti_viewer.py                  # empty, then open from the window

Any VTK ImageData works, not only the grids from ``hkl_grid.py``; the axes are
labelled H/K/L because that is what these files hold. Cells the scan never
reached are NaN in the file and are drawn fully transparent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from volume_style import AXIS_COLOR
from volume_style import AXIS_NAMES
from volume_style import AXIS_NUMBER_FORMAT
from volume_style import BACKGROUND
from volume_style import COLORMAPS
from volume_style import SCALAR_BAR
from volume_style import SLICE_COLOR
from volume_style import make_drawable
from volume_style import spin as _spin

INTENSITY = "intensity"
# Used until a file is open and the grid's own cell size can be read.
DEFAULT_OFFSET_STEP = 0.005


class VtiWindow(QtWidgets.QMainWindow):
    """A volume, a slice through it, and the controls for both."""

    def __init__(self, path=None, parent=None):
        super().__init__(parent)
        self.grid = None
        self.path: Path | None = None
        self._volume_actor = None
        self._slice_actor = None
        self._slice_widget = None
        self._slice_avg_actor = None
        self._slice_window = None
        self.folder: Path | None = None
        self.span = (0.0, 1.0)
        self._crop_cache = (None, None)
        self._updating = False
        self._needs_camera_reset = True

        self.setWindowTitle("VTI 3D viewer")
        self.resize(1450, 900)
        self._build_ui()
        if path is not None:
            path = Path(path)
            if path.is_dir():
                self.open_folder(path)
            else:
                self.load(path)
                # Having opened one, offer its neighbours.
                self.open_folder(path.parent)

    # -- construction -----------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(root)

        # The left side is its own splitter, so the list and the controls can
        # be traded off against each other as well as against the view.
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self._build_file_pane())

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = self._build_panel()
        scroll.setWidget(panel)
        # Wide enough to show the controls whole rather than clipping them, and
        # the splitter leaves it adjustable from there.
        wanted = panel.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 8
        scroll.setMinimumWidth(min(wanted, 520))
        left.addWidget(scroll)
        left.setStretchFactor(0, 0)
        left.setStretchFactor(1, 1)
        left.setSizes([200, 620])
        self.left_split = left
        root.addWidget(left)

        self.plot_host = QtWidgets.QWidget()
        host_layout = QtWidgets.QVBoxLayout(self.plot_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        reason = None
        # VTK needs a real OpenGL context. Asking for one under a windowless Qt
        # platform does not raise -- it takes the process down with it -- so
        # check before building the widget rather than trying and catching.
        platform = QtWidgets.QApplication.instance().platformName()
        if platform in ("offscreen", "minimal"):
            reason = f"no OpenGL context on the {platform!r} Qt platform"
        else:
            try:
                from pyvistaqt import QtInteractor

                self.plotter = QtInteractor(self.plot_host)
                host_layout.addWidget(self.plotter.interactor)
            except Exception as exc:  # pragma: no cover - depends on the install
                reason = str(exc)
        if self.plotter is None:
            label = QtWidgets.QLabel(
                f"3D view unavailable — {reason}.\n\n"
                "The file still loads; its shape is in the status bar."
            )
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setWordWrap(True)
            host_layout.addWidget(label)
        root.addWidget(self.plot_host)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([wanted + 16, max(900 - wanted, 600)])
        self.root_split = root

        menu = self.menuBar().addMenu("&File")
        action = menu.addAction("Open &folder…")
        action.setShortcut("Ctrl+O")
        action.triggered.connect(lambda: self.open_folder())
        action = menu.addAction("Open a single &file…")
        action.setShortcut("Ctrl+Shift+O")
        action.triggered.connect(self.open_dialog)
        menu.addSeparator()
        close = menu.addAction("&Close")
        close.setShortcut("Ctrl+W")
        close.triggered.connect(self.close)

        self.status = self.statusBar()
        self.status.showMessage(
            "Nothing open yet — use Open folder… or Open file… on the left."
        )
        self._slice_controls_enabled()

    def _build_file_pane(self) -> QtWidgets.QWidget:
        pane = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(pane)
        layout.setContentsMargins(6, 6, 6, 2)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        folder_button = QtWidgets.QPushButton("Open folder…")
        folder_button.setToolTip("List every .vti in a folder; click one to open it.")
        # clicked carries a bool, and open_folder takes an optional path --
        # connected directly, the button would pass False as the folder.
        folder_button.clicked.connect(lambda: self.open_folder())
        buttons.addWidget(folder_button)
        file_button = QtWidgets.QPushButton("Open file…")
        file_button.setToolTip(
            "Open one .vti. Its folder is listed as well, so its neighbours are\n"
            "a click away."
        )
        file_button.clicked.connect(self.open_dialog)
        buttons.addWidget(file_button)
        layout.addLayout(buttons)

        self.folder_label = QtWidgets.QLabel("no folder")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: gray;")
        layout.addWidget(self.folder_label)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.currentItemChanged.connect(self._file_chosen)
        layout.addWidget(self.file_list, 1)

        self.keep_check = QtWidgets.QCheckBox("Keep the current set-up")
        self.keep_check.setToolTip(
            "Carry everything over to the next file opened from the list --\n"
            "colormap, intensity range, HKL range, the slice and its averaging,\n"
            "and the camera. Off, each file opens on its own full extent.\n"
            "Leave it on to compare the same corner of reciprocal space across\n"
            "a set of scans."
        )
        self.keep_check.setChecked(True)
        layout.addWidget(self.keep_check)
        return pane

    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        self.file_label = QtWidgets.QLabel("no file")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: gray;")
        layout.addWidget(self.file_label)

        # -- visibility --
        box = QtWidgets.QGroupBox("Visibility")
        form = QtWidgets.QFormLayout(box)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentIndexChanged.connect(self._restyle)
        form.addRow("Colormap", self.cmap_combo)
        self.show_volume_check = QtWidgets.QCheckBox("Show volume")
        self.show_volume_check.setChecked(True)
        self.show_volume_check.toggled.connect(self.refresh_view)
        form.addRow(self.show_volume_check)

        self.show_slice_check = QtWidgets.QCheckBox("Show slice")
        self.show_slice_check.toggled.connect(self._slice_shown_toggled)
        form.addRow(self.show_slice_check)
        # An arbitrary plane: any normal, not just along an axis. Each component
        # is labelled with its axis, and the range is wide enough to type a
        # direction the way it is usually written -- [1 1 2] rather than a unit
        # vector -- since only the direction matters and it is normalised on use.
        normal_row = QtWidgets.QHBoxLayout()
        normal_row.setContentsMargins(0, 0, 0, 0)
        normal_row.setSpacing(2)
        self.slice_normal_spins = []
        for name, value in zip(AXIS_NAMES, (0.0, 0.0, 1.0)):
            label = QtWidgets.QLabel(name)
            label.setToolTip(f"{name} component of the plane normal")
            box_spin = _spin(value, -1e4, 1e4, 4, 0.1, 62)
            box_spin.setToolTip(
                f"{name} component of the plane normal.\n"
                "Type any direction -- [1 1 2] is fine, it does not have to be a\n"
                "unit vector -- or drag the yellow arrow in the view."
            )
            box_spin.valueChanged.connect(self._slice_moved)
            normal_row.addWidget(label)
            normal_row.addWidget(box_spin)
            self.slice_normal_spins.append(box_spin)
        normal_widget = QtWidgets.QWidget()
        normal_widget.setLayout(normal_row)
        form.addRow("Plane normal", normal_widget)
        self.slice_unit_label = QtWidgets.QLabel("")
        self.slice_unit_label.setToolTip(
            "The same direction, normalised -- what the plane actually uses."
        )
        form.addRow("", self.slice_unit_label)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        self._slice_preset_buttons = []
        for index, name in enumerate(AXIS_NAMES):
            button = QtWidgets.QPushButton(f"⊥{name}")
            button.setToolTip(f"Point the plane normal along {name}.")
            button.setFixedWidth(46)
            button.clicked.connect(lambda _=False, i=index: self._set_normal_axis(i))
            preset_row.addWidget(button)
            self._slice_preset_buttons.append(button)
        preset_widget = QtWidgets.QWidget()
        preset_widget.setLayout(preset_row)
        form.addRow("", preset_widget)

        # Five decimals, not four: the step below defaults to the grid's cell
        # size, and at four decimals a cell of 6.7e-04 could not be typed.
        self.slice_offset_spin = _spin(0.0, -1e6, 1e6, 5, DEFAULT_OFFSET_STEP)
        self.slice_offset_spin.setToolTip(
            "How far the plane sits from the centre of the map, along its normal."
        )
        self.slice_offset_spin.valueChanged.connect(self._slice_moved)
        form.addRow("Offset along normal", self.slice_offset_spin)
        self.slice_step_spin = _spin(DEFAULT_OFFSET_STEP, 1e-6, 1e4, 5, 1e-4)
        self.slice_step_spin.setToolTip(
            "How far one click of the arrows, or one arrow key, moves the plane.\n"
            "Set to the grid's cell size when a file is opened, so a click steps\n"
            "one cell -- the smallest move that shows new data. Type any value."
        )
        self.slice_step_spin.valueChanged.connect(self._offset_step_changed)
        form.addRow("Offset step", self.slice_step_spin)
        self.slice_avg_spin = QtWidgets.QSpinBox()
        self.slice_avg_spin.setRange(1, 999)
        self.slice_avg_spin.setValue(1)
        self.slice_avg_spin.setSuffix(" cells")
        self.slice_avg_spin.setFixedWidth(92)
        self.slice_avg_spin.setToolTip(
            "Average the slice over this many cells either side of the plane,\n"
            "along its normal. A plane has no thickness, so one cell is as noisy\n"
            "as the data gets; a few trade a little resolution across the plane\n"
            "for much less noise in it. The 2D window follows this too."
        )
        self.slice_avg_spin.valueChanged.connect(self._slice_average_changed)
        form.addRow("Average over", self.slice_avg_spin)
        self.slice_lock_check = QtWidgets.QCheckBox("Lock slice (hide handles)")
        self.slice_lock_check.setToolTip(
            "Freeze the plane and take the arrow and outline out of the view, so\n"
            "they stop covering the data. Untick to move it again."
        )
        self.slice_lock_check.toggled.connect(self._slice_lock_toggled)
        form.addRow(self.slice_lock_check)
        self.slice_plot_button = QtWidgets.QPushButton("Slice as 2D image…")
        self.slice_plot_button.setToolTip(
            "Open the plane flat, in its own window, with H/K/L axes, tick\n"
            "numbers and a colour bar. It follows the plane as you move it."
        )
        self.slice_plot_button.clicked.connect(self.show_slice_plot)
        form.addRow(self.slice_plot_button)
        layout.addWidget(box)

        # -- intensity --
        box = QtWidgets.QGroupBox("Intensity range")
        form = QtWidgets.QFormLayout(box)
        self.int_min_spin = _spin(0.0, -1e12, 1e12, 3, 10.0)
        self.int_max_spin = _spin(1.0, -1e12, 1e12, 3, 10.0)
        for widget in (self.int_min_spin, self.int_max_spin):
            widget.valueChanged.connect(self._intensity_edited)
        self.int_auto_check = QtWidgets.QCheckBox("Auto")
        self.int_auto_check.setChecked(True)
        self.int_auto_check.toggled.connect(self._intensity_auto_toggled)
        form.addRow("Min", self.int_min_spin)
        form.addRow("Max", self.int_max_spin)
        form.addRow(self.int_auto_check)
        layout.addWidget(box)

        # -- HKL range --
        box = QtWidgets.QGroupBox("HKL range")
        grid_layout = QtWidgets.QGridLayout(box)
        self.hkl_spins = {}
        for row, name in enumerate(AXIS_NAMES):
            grid_layout.addWidget(QtWidgets.QLabel(name), row, 0)
            low = _spin(0.0, -1e6, 1e6, 4, 0.01, 92)
            high = _spin(1.0, -1e6, 1e6, 4, 0.01, 92)
            for widget in (low, high):
                widget.valueChanged.connect(self._hkl_edited)
            grid_layout.addWidget(low, row, 1)
            grid_layout.addWidget(high, row, 2)
            self.hkl_spins[name] = (low, high)
        self.hkl_auto_check = QtWidgets.QCheckBox("Whole map")
        self.hkl_auto_check.setChecked(True)
        self.hkl_auto_check.setToolTip(
            "Untick to cut the volume down to the box below. The axes follow the\n"
            "box, so the numbers always describe what is drawn."
        )
        self.hkl_auto_check.toggled.connect(self._hkl_auto_toggled)
        grid_layout.addWidget(self.hkl_auto_check, 3, 0, 1, 3)
        layout.addWidget(box)

        layout.addStretch(1)
        return panel

    # -- loading ----------------------------------------------------------

    def open_folder(self, folder=None):
        """List the .vti files in a folder so they can be opened by clicking."""
        if folder is None:
            start = str(self.folder or (self.path.parent if self.path else ""))
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Open a folder of gridded maps", start
            )
            if not folder:
                return
        if not isinstance(folder, (str, Path)):
            return
        folder = Path(folder)
        self.folder = folder
        files = sorted(folder.glob("*.vti"), key=lambda q: q.name)
        self.folder_label.setText(
            f"{folder.name} — {len(files)} .vti file(s)" if files
            else f"{folder.name} — no .vti files here"
        )
        self.folder_label.setToolTip(str(folder))
        self._updating = True
        try:
            self.file_list.clear()
            for each in files:
                item = QtWidgets.QListWidgetItem(each.name)
                item.setData(QtCore.Qt.UserRole, str(each))
                size = each.stat().st_size / 1e6
                item.setToolTip(f"{each}\n{size:,.0f} MB")
                self.file_list.addItem(item)
            # Keep the highlight on the file already open, if it is in here.
            if self.path is not None:
                for row in range(self.file_list.count()):
                    if self.file_list.item(row).data(QtCore.Qt.UserRole) == str(self.path):
                        self.file_list.setCurrentRow(row)
                        break
        finally:
            self._updating = False
        self.status.showMessage(
            f"{folder}: {len(files)} .vti file(s)"
            + ("" if files else " — nothing to open")
        )

    def _file_chosen(self, current, _previous=None):
        if current is None or self._updating:
            return
        path = current.data(QtCore.Qt.UserRole)
        if path and Path(path) != self.path:
            self.load(path)

    def open_dialog(self):
        start = str(self.path.parent) if self.path else ""
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a gridded map", start, "VTK ImageData (*.vti);;All files (*)"
        )
        if not name:
            return
        self.load(name)
        if self.path is not None and self.path.parent != self.folder:
            self.open_folder(self.path.parent)

    def load(self, path):
        """Read a .vti and show it."""
        try:
            import pyvista as pv
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(self, "pyvista missing", str(exc))
            return
        path = Path(path)
        try:
            grid = pv.read(str(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could not read", f"{path.name}\n\n{type(exc).__name__}: {exc}"
            )
            return
        if not isinstance(grid, pv.ImageData):
            QtWidgets.QMessageBox.warning(
                self,
                "Not a regular grid",
                f"{path.name} holds {type(grid).__name__}, not VTK ImageData.\n"
                "This viewer draws regular grids; hkl_grid.py writes them.",
            )
            return
        if INTENSITY not in grid.point_data:
            names = list(grid.point_data.keys())
            if not names:
                QtWidgets.QMessageBox.warning(
                    self, "No data", f"{path.name} carries no point data to draw."
                )
                return
            # Some other writer's array: draw the first one under our own name.
            grid.point_data[INTENSITY] = grid.point_data[names[0]]

        # NaN marks cells nothing reached. A volume mapper does not render it
        # transparent -- it lights the whole empty region up -- so swap it for
        # the bottom of the colour scale. The file itself is not touched.
        keep = bool(self.keep_check.isChecked() and self.grid is not None)
        self.span = make_drawable(grid)
        self.grid, self.path = grid, path
        self._crop_cache = (None, None)
        # Carrying the set-up over means carrying the camera too: the point is
        # to flip between files and see the same thing from the same angle.
        self._needs_camera_reset = not keep
        self.setWindowTitle(f"VTI 3D viewer — {path.name}")

        measured = grid.field_data.get("filled_cells")
        cell = " ".join(f"{float(v):.1e}" for v in grid.spacing)
        detail = (
            f"{int(np.asarray(measured)[0]):,} of {int(np.asarray(measured)[1]):,} measured"
            if measured is not None and np.asarray(measured).size >= 2
            else f"{grid.n_points:,} cells"
        )
        self.file_label.setText(
            f"{path.name}\n{'×'.join(str(int(n)) for n in grid.dimensions)} cells, "
            f"step {cell}\n{detail}"
        )

        self._updating = True
        try:
            if not keep:
                # One click of the arrows moves the plane by one cell, which is
                # the smallest step that can show anything new.
                cell = float(np.min(np.abs(np.asarray(grid.spacing, float))))
                if np.isfinite(cell) and cell > 0:
                    self.slice_step_spin.setValue(cell)
                    self.slice_offset_spin.setSingleStep(cell)
                if self.int_auto_check.isChecked():
                    self.int_min_spin.setValue(self.span[0])
                    self.int_max_spin.setValue(self.span[1])
                self._fill_hkl_range()
                self._centre_slice()
            elif self.int_auto_check.isChecked():
                # Auto means auto: follow the new file's own range. Numbers
                # typed by hand are left alone.
                self.int_min_spin.setValue(self.span[0])
                self.int_max_spin.setValue(self.span[1])
        finally:
            self._updating = False
        self.refresh_view()
        if keep:
            self.status.showMessage(
                self.status.currentMessage() + "  ·  set-up carried over"
            )

    # -- panel reactions ---------------------------------------------------

    def _intensity_edited(self):
        if self._updating:
            return
        # Typing a limit is a request for manual control.
        if self.int_auto_check.isChecked():
            self._updating = True
            self.int_auto_check.setChecked(False)
            self._updating = False
        self._restyle()

    def _intensity_auto_toggled(self, checked):
        if self._updating:
            return
        if checked:
            self._updating = True
            self.int_min_spin.setValue(self.span[0])
            self.int_max_spin.setValue(self.span[1])
            self._updating = False
        self._restyle()

    def _hkl_edited(self):
        if self._updating:
            return
        if self.hkl_auto_check.isChecked():
            self._updating = True
            self.hkl_auto_check.setChecked(False)
            self._updating = False
        self._centre_slice()
        self.refresh_view()

    def _hkl_auto_toggled(self, checked):
        if self._updating:
            return
        if checked:
            self._updating = True
            self._fill_hkl_range()
            self._updating = False
        self._centre_slice()
        self.refresh_view()

    def _fill_hkl_range(self):
        if self.grid is None:
            return
        for name, (low, high) in zip(AXIS_NAMES, self.data_bounds()):
            self.hkl_spins[name][0].setValue(float(low))
            self.hkl_spins[name][1].setValue(float(high))

    def _offset_step_changed(self, value):
        """Retune the offset box's arrows; the plane itself does not move."""
        self.slice_offset_spin.setSingleStep(float(value))

    def _slice_controls_enabled(self):
        shown = self.show_slice_check.isChecked()
        locked = self.slice_lock_check.isChecked()
        live = shown and not locked
        for widget in self.slice_normal_spins:
            widget.setEnabled(live)
        for button in self._slice_preset_buttons:
            button.setEnabled(live)
        self.slice_offset_spin.setEnabled(live)
        self.slice_step_spin.setEnabled(live)
        self.slice_avg_spin.setEnabled(shown)
        self.slice_lock_check.setEnabled(shown)

    def _slice_shown_toggled(self, _checked=None):
        self._slice_controls_enabled()
        if self.plotter is None or self.grid is None:
            return
        # Only the slice changes; the volume behind it is untouched.
        self._draw_slice(take_bar=not self.show_volume_check.isChecked())
        self.plotter.render()
        self._update_slice_plot()

    def _slice_lock_toggled(self, _checked=None):
        self._slice_controls_enabled()
        widget = self._slice_widget
        if widget is None or self.plotter is None:
            return
        # The widget drives the cutter, so locking only takes its handles out
        # of the view -- the slice itself stays exactly where it is.
        try:
            widget.Off() if self.slice_lock_check.isChecked() else widget.On()
        except Exception:
            return
        self.plotter.render()

    def _set_normal_axis(self, axis: int):
        self._updating = True
        try:
            for index, widget in enumerate(self.slice_normal_spins):
                widget.setValue(1.0 if index == axis else 0.0)
        finally:
            self._updating = False
        self._slice_moved()

    def _centre_slice(self):
        self.slice_offset_spin.setValue(0.0)

    # -- geometry ----------------------------------------------------------

    def data_bounds(self):
        """((h0, h1), (k0, k1), (l0, l1)) of the loaded grid."""
        if self.grid is None:
            return None
        flat = self.grid.bounds
        return ((flat[0], flat[1]), (flat[2], flat[3]), (flat[4], flat[5]))

    def display_bounds(self):
        """The box the axes describe -- the crop if there is one, else the map."""
        bounds = self.data_bounds()
        if bounds is None:
            return None
        if not self.hkl_auto_check.isChecked():
            bounds = tuple(
                (self.hkl_spins[name][0].value(), self.hkl_spins[name][1].value())
                for name in AXIS_NAMES
            )
        flat = []
        for low, high in bounds:
            if high <= low:
                high = low + 1e-9
            flat.extend((float(low), float(high)))
        return flat

    def _centre(self):
        bounds = self.display_bounds()
        if bounds is None:
            return np.zeros(3)
        return np.array([(bounds[2 * i] + bounds[2 * i + 1]) / 2.0 for i in range(3)])

    def slice_plane(self):
        """(unit normal, origin) of the slice plane."""
        normal = np.array([widget.value() for widget in self.slice_normal_spins], float)
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            normal = np.array([0.0, 0.0, 1.0])
            length = 1.0
        normal = normal / length
        return normal, self._centre() + normal * self.slice_offset_spin.value()

    def _show_unit_normal(self):
        normal, _ = self.slice_plane()
        self.slice_unit_label.setText("unit:  %.3f, %.3f, %.3f" % tuple(normal))

    def visible_grid(self):
        """The grid to draw: cropped to the HKL box, or the whole thing.

        ``extract_subset`` copies, so the crop is kept until the box changes --
        redrawing for a colour change must not re-cut a 30-million-cell volume.
        """
        if self.grid is None:
            return None
        if self.hkl_auto_check.isChecked():
            return self.grid
        origin = np.asarray(self.grid.origin, float)
        spacing = np.asarray(self.grid.spacing, float)
        dims = np.asarray(self.grid.dimensions, int)
        low = np.array([self.hkl_spins[n][0].value() for n in AXIS_NAMES], float)
        high = np.array([self.hkl_spins[n][1].value() for n in AXIS_NAMES], float)
        spacing = np.where(spacing == 0, 1.0, spacing)
        first = np.clip(np.floor((low - origin) / spacing).astype(int), 0, dims - 1)
        last = np.clip(np.ceil((high - origin) / spacing).astype(int), 0, dims - 1)
        last = np.maximum(last, first)
        if np.all(first == 0) and np.all(last == dims - 1):
            # The box covers everything; extract_subset would copy the whole
            # grid to hand back what we already have.
            return self.grid
        voi = (first[0], last[0], first[1], last[1], first[2], last[2])
        if self._crop_cache[0] == voi:
            return self._crop_cache[1]
        try:
            cropped = self.grid.extract_subset(voi)
        except Exception:
            cropped = self.grid
        self._crop_cache = (voi, cropped)
        return cropped

    def _lookup_table(self, low, high):
        """A colormap whose opacity rises with intensity.

        Faint cells fade out instead of packing the view with haze, so the
        bright features stay visible inside the volume.
        """
        import pyvista as pv

        lut = pv.LookupTable(cmap=self.cmap_combo.currentText())
        # The ramp lives on the lookup table alone -- passing opacity= to
        # add_volume as well would be a second, competing source for it.
        lut.apply_opacity([0, 1])
        if high > low:
            lut.scalar_range = (float(low), float(high))
        return lut

    # -- drawing -----------------------------------------------------------

    def _plane_moved(self, widget, _event):
        """The plane widget was dragged: mirror it into the boxes.

        Only the boxes -- ``add_mesh_slice`` has already recut the volume
        through the VTK pipeline, and rebuilding the widget from here would
        pull it out from under the interaction that is still running.
        """
        if self._updating:
            return
        normal = np.asarray(widget.GetNormal(), dtype=float)
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            return
        normal = normal / length
        origin = np.asarray(widget.GetOrigin(), dtype=float)
        offset = float(np.dot(origin - self._centre(), normal))
        self._updating = True
        try:
            for spin, value in zip(self.slice_normal_spins, normal):
                spin.setValue(float(value))
            self.slice_offset_spin.setValue(offset)
        finally:
            self._updating = False
        self._show_unit_normal()
        self._apply_slab_average()
        self._update_slice_plot()

    def show_slice_plot(self):
        """Open (or raise) the flat view of the plane."""
        try:
            from slice_plot import SlicePlotWindow
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Slice plot unavailable", f"slice_plot needs: {exc}"
            )
            return
        if self._slice_window is None:
            self._slice_window = SlicePlotWindow(self)
        if not self.show_slice_check.isChecked():
            # Nothing to show otherwise, and the plane is where the 3D view
            # says it is -- so turn it on rather than open an empty window.
            self.show_slice_check.setChecked(True)
        self._slice_window.show()
        self._slice_window.raise_()
        self._slice_window.activateWindow()
        self._update_slice_plot()

    def _update_slice_plot(self):
        """Push the current plane to the 2D window, if it is open."""
        window = self._slice_window
        if window is None or not window.isVisible() or self.grid is None:
            return
        normal, origin = self.slice_plane()
        window.update_slice(
            self.visible_grid(),
            normal,
            origin,
            cmap=self.cmap_combo.currentText(),
            clim=(self.int_min_spin.value(), self.int_max_spin.value()),
            layers=int(self.slice_avg_spin.value()),
            source=self.path.name if self.path else None,
        )

    def _slice_average_changed(self, _value):
        if self._updating or self.plotter is None or self.grid is None:
            return
        self._apply_slab_average()
        self.plotter.render()
        self._update_slice_plot()

    def _apply_slab_average(self):
        """Show the slice averaged over a slab, rather than the bare plane.

        The cutter gives a plane with no thickness. For a thicker slice the
        volume is sampled at each of its points, once per cell of the slab, and
        the mean is drawn in its place -- the cutter's own actor is hidden
        rather than removed, so the widget that drives it stays alive.
        """
        if self.plotter is None:
            return
        try:
            self.plotter.remove_actor("slice_avg", reset_camera=False)
        except Exception:
            pass
        self._slice_avg_actor = None
        layers = int(self.slice_avg_spin.value())
        if self._slice_actor is None:
            return
        if layers <= 1 or not self.show_slice_check.isChecked():
            try:
                self._slice_actor.visibility = True
            except Exception:
                pass
            return
        try:
            from slice_plot import average_along_normal

            plane = self.plotter.plane_sliced_meshes[-1]
            if plane is None or plane.n_points == 0:
                return
            grid = self.visible_grid()
            normal, _origin = self.slice_plane()
            values = average_along_normal(grid, plane.points, normal, layers)
            averaged = plane.copy()
            # Unmeasured cells sit at the bottom of the scale, where the
            # opacity ramp starts at zero, exactly as in the volume.
            averaged.point_data[INTENSITY] = np.ma.filled(values, 0.0).astype("float32")
            low, high = self.int_min_spin.value(), self.int_max_spin.value()
            self._slice_actor.visibility = False
            self._slice_avg_actor = self.plotter.add_mesh(
                averaged,
                scalars=INTENSITY,
                cmap=self._lookup_table(low, high),
                clim=(low, high),
                name="slice_avg",
                show_scalar_bar=False,
                reset_camera=False,
            )
        except Exception as exc:
            self.status.showMessage(
                f"Slab average failed: {type(exc).__name__}: {exc}"
            )
            try:
                self._slice_actor.visibility = True
            except Exception:
                pass

    def _slice_moved(self):
        """The plane changed from the boxes: recut, and touch nothing else.

        Going through refresh_view for this would hand VTK the whole volume
        again and stand up a fresh set of axes -- seconds of work, and a
        visible flicker, because a plane moved by one step.
        """
        if self._updating:
            return
        self._show_unit_normal()
        if self.plotter is None or self.grid is None:
            return
        if not self.show_slice_check.isChecked():
            return
        widget = self._slice_widget
        if widget is None:
            self._draw_slice(take_bar=not self.show_volume_check.isChecked())
            self.plotter.render()
            return
        normal, origin = self.slice_plane()
        self._updating = True
        try:
            widget.SetNormal(*normal)
            widget.SetOrigin(*origin)
            # pyvista's cutter listens here; firing it recuts through the
            # pipeline that is already built rather than making a new one.
            # The guard keeps our own observer from writing the numbers back.
            widget.InvokeEvent("EndInteractionEvent")
        finally:
            self._updating = False
        self._apply_slab_average()
        self.plotter.render()
        self._update_slice_plot()

    def _clear_slice(self):
        """Take down the slice, its widget and the outline it brings with it."""
        for name in ("slice", "slice-outline", "slice_avg"):
            try:
                self.plotter.remove_actor(name, reset_camera=False)
            except Exception:
                pass
        try:
            self.plotter.clear_plane_widgets()
        except Exception:
            pass
        try:
            # add_mesh_slice appends to this every call; left alone it keeps a
            # reference to every slice ever cut.
            self.plotter.plane_sliced_meshes.clear()
        except Exception:
            pass

    def _draw_slice(self, take_bar=True):
        """Cut the volume with a plane.

        Unlocked, this is pyvista's own ``add_mesh_slice``: it drives a
        ``vtkCutter`` from the plane widget, so dragging recuts inside the VTK
        pipeline instead of re-slicing from Python on every event -- which is
        what makes it usable on a grid of tens of millions of cells. Locked,
        there is no widget to drive it, so the plane is cut once and drawn as a
        plain mesh.
        """
        if self.plotter is None:
            return
        self._clear_slice()
        self._slice_actor = None
        self._slice_avg_actor = None
        self._slice_widget = None
        if not self.show_slice_check.isChecked():
            return
        grid = self.visible_grid()
        if grid is None or grid.n_points == 0:
            return
        normal, origin = self.slice_plane()
        low, high = self.int_min_spin.value(), self.int_max_spin.value()
        try:
            self._slice_actor = self.plotter.add_mesh_slice(
                grid,
                normal=tuple(normal),
                origin=tuple(origin),
                widget_color=SLICE_COLOR,
                tubing=False,
                outline_opacity=0,
                implicit=True,
                normal_rotation=True,
                origin_translation=True,
                # 'always' recuts on every mouse-move event; 'end' moves the
                # handle freely and cuts once, when the drag finishes.
                interaction_event="end",
                # The default is off, and off puts vtkCutter on a slow
                # unthreaded path that builds general polygons: measured at
                # 1.18 s against 0.028 s here, for output identical to the
                # point. Leave this on.
                generate_triangles=True,
                scalars=INTENSITY,
                cmap=self._lookup_table(low, high),
                clim=(low, high),
                name="slice",
                show_scalar_bar=take_bar,
                scalar_bar_args=dict(SCALAR_BAR),
                reset_camera=False,
            )
        except Exception as exc:
            self.status.showMessage(f"Slice failed: {type(exc).__name__}: {exc}")
            return
        try:
            self._slice_widget = self.plotter.plane_widgets[-1]
            self._slice_widget.AddObserver("EndInteractionEvent", self._plane_moved)
            if self.slice_lock_check.isChecked():
                self._slice_widget.Off()
        except Exception:
            pass
        self._apply_slab_average()

    def _draw_volume(self, lut, low, high, take_bar):
        grid = self.visible_grid()
        if grid is None:
            return
        try:
            self._volume_actor = self.plotter.add_volume(
                grid,
                scalars=INTENSITY,
                cmap=lut,
                clim=(low, high),
                name="volume",
                show_scalar_bar=take_bar,
                scalar_bar_args=dict(SCALAR_BAR),
                reset_camera=False,
            )
        except Exception as exc:
            self.status.showMessage(
                f"Volume rendering failed: {type(exc).__name__}: {exc}"
            )
            return
        try:
            # Off by default, which means every frame of a rotation is rendered
            # at full quality. On, VTK coarsens the ray sampling while the
            # camera is moving and sharpens again once it stops.
            self._volume_actor.mapper.SetAutoAdjustSampleDistances(True)
        except Exception:
            pass

    def _restyle(self):
        """Recolour what is already drawn, without rebuilding it.

        Changing the colormap or the limits does not move a single cell, so
        there is no reason to hand the volume to VTK again. On a grid of tens
        of millions of cells that is the difference between a redraw you notice
        and one you do not -- about 15 ms against 1.3 s.
        """
        if self._updating:
            return
        if self.plotter is None or self.grid is None:
            self._report()
            return
        low = self.int_min_spin.value()
        high = self.int_max_spin.value()
        lut = self._lookup_table(low, high)
        restyled = False
        if self._volume_actor is not None:
            try:
                self._volume_actor.prop.apply_lookup_table(lut)
                restyled = True
            except Exception:
                pass
        for actor in (self._slice_actor, self._slice_avg_actor):
            if actor is None:
                continue
            try:
                actor.mapper.lookup_table = lut
                actor.mapper.scalar_range = (low, high)
                restyled = True
            except Exception:
                pass
        self._update_slice_plot()
        if restyled:
            self.plotter.render()
        else:
            # Nothing on screen to recolour -- fall back to a full draw.
            self.refresh_view()

    def refresh_view(self):
        if self._updating:
            return
        self._show_unit_normal()
        self._report()
        if self.plotter is None or self.grid is None:
            return

        self.plotter.clear()
        # clear() invalidates them; anything holding on would restyle a corpse.
        self._volume_actor = None
        self._slice_actor = None
        self._slice_avg_actor = None
        self.plotter.set_background(BACKGROUND)
        # color= is the *label* colour; left alone the H/K/L letters are drawn
        # in the theme font colour, which is black and so invisible here.
        self.plotter.add_axes(xlabel="H", ylabel="K", zlabel="L", color=AXIS_COLOR)
        bounds = self.display_bounds()

        low = self.int_min_spin.value()
        high = self.int_max_spin.value()
        lut = self._lookup_table(low, high)

        # One scalar bar between them: a second add_* with the same title
        # stacks another bar on top of the first.
        bar_taken = False
        if self.show_volume_check.isChecked():
            self._draw_volume(lut, low, high, take_bar=True)
            bar_taken = True
        # add_mesh_slice brings its own plane widget, so there is none to wire
        # up here -- and plotter.clear() above already took the old one down.
        self._draw_slice(take_bar=not bar_taken)

        # Drawn light so it reads on black.
        if bounds is not None:
            try:
                self.plotter.show_bounds(
                    bounds=bounds,
                    xtitle="H",
                    ytitle="K",
                    ztitle="L",
                    fmt=AXIS_NUMBER_FORMAT,
                    color=AXIS_COLOR,
                    grid=False,
                    location="outer",
                    ticks="both",
                    use_3d_text=False,
                )
            except Exception:
                pass
        if self._needs_camera_reset:
            try:
                self.plotter.reset_camera(bounds=bounds)
            except Exception:
                self.plotter.reset_camera()
            self._needs_camera_reset = False
        self._update_slice_plot()

    def _report(self):
        if self.grid is None:
            return
        grid = self.visible_grid()
        shown = "" if grid is self.grid else f" · showing {grid.n_points:,} of "
        dims = "×".join(str(int(n)) for n in grid.dimensions)
        cell = " ".join(f"{float(v):.1e}" for v in grid.spacing)
        (h0, h1), (k0, k1), (l0, l1) = (
            (grid.bounds[0], grid.bounds[1]),
            (grid.bounds[2], grid.bounds[3]),
            (grid.bounds[4], grid.bounds[5]),
        )
        total = "" if not shown else f"{self.grid.n_points:,} cells"
        self.status.showMessage(
            f"{self.path.name} · {dims} cells (step {cell}){shown}{total} · "
            f"H {h0:.4f}…{h1:.4f}  K {k0:.4f}…{k1:.4f}  L {l0:.4f}…{l1:.4f}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "vti",
        nargs="?",
        help="a .vti written by hkl_grid.py, or a folder of them (optional)",
    )
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    # The window comes up whether or not anything was named, so there is
    # something on screen to open a file from.
    window = VtiWindow(args.vti)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
