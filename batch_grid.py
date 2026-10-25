"""Grid a run of scans onto one common HKL lattice, as ``.vti`` files.

One scan at a time through the HKL viewer is fine for looking; comparing a
series means gridding them all the same way. This takes a folder of scans, a
list of scan numbers and an HKL range, and writes one ``.vti`` per scan into an
``output`` folder beside them. Scans are found by their master files, so it
does not matter whether the detector file is named ``pva_<scan>.h5``, named
after the scan alone, or off in a tree of its own that the master points to.

**The cell size is measured once, from the first scan in the list, and reused
for every scan after it.** Given the same HKL range as well, every grid then
has the same origin, the same spacing and the same shape, so they can be
subtracted, stacked or stepped through cell for cell. Measuring each scan
separately would give each its own lattice and quietly make them
incomparable.

The fuzzy footprint is *not* shared: it describes how far one sample of that
particular scan reaches, so each scan keeps its own. Sharing it would
over- or under-smooth any scan whose rocking step differs from the first.

    python batch_grid.py                 # opens empty
    python batch_grid.py /data/beamtime  # opens on that folder

From a script::

    from batch_grid import expand_scans, grid_scans
    grid_scans("/data/beamtime", expand_scans(152, 160, 2),
               hkl_range=(0.93, 1.11, -0.60, -0.36, 7.39, 7.66))
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from da6idb.batch import OUTPUT_DIRECTORY, expand_scans, find_scans, grid_scans, parse_scans
from da6idb.files import master_path_for_image
from da6idb.hkl.geometry import SETUP_FILENAME, read_geometry, read_setup
from da6idb.hkl.grid import measured_range
from volume_style import AXIS_NAMES
from volume_style import spin as _spin

import viewer_state












class BatchGridWindow(QtWidgets.QMainWindow):
    """Pick scans and a range, grid them all the same way."""

    def __init__(self, folder=None, parent=None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.found: dict = {}
        self._stop = False
        self._running = False

        self.setWindowTitle("Batch gridder")
        self.resize(760, 720)
        self._build_ui()
        if folder is not None:
            self.open_folder(folder)

    # -- construction -----------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # -- the folder --
        box = QtWidgets.QGroupBox("Scans")
        form = QtWidgets.QFormLayout(box)
        row = QtWidgets.QHBoxLayout()
        button = QtWidgets.QPushButton("Open folder…")
        button.setToolTip("The folder holding the scans' master files.")
        button.clicked.connect(lambda: self.open_folder())
        row.addWidget(button)
        self.folder_label = QtWidgets.QLabel("no folder")
        self.folder_label.setStyleSheet("color: gray;")
        row.addWidget(self.folder_label, 1)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        form.addRow(holder)

        self.setup_label = QtWidgets.QLabel("")
        self.setup_label.setWordWrap(True)
        self.setup_label.setStyleSheet("color: gray;")
        self.setup_label.setToolTip(
            f"A {SETUP_FILENAME} beside the data corrects the geometry for every\n"
            "scan in the folder. Edit and save it from the HKL viewer's\n"
            "Conversion setup window."
        )
        form.addRow(self.setup_label)

        self.found_label = QtWidgets.QLabel("")
        self.found_label.setWordWrap(True)
        self.found_label.setStyleSheet("color: gray;")
        form.addRow(self.found_label)

        # -- the list, and filling it from a range --
        fill = QtWidgets.QHBoxLayout()
        fill.setContentsMargins(0, 0, 0, 0)
        self.start_spin = QtWidgets.QSpinBox()
        self.end_spin = QtWidgets.QSpinBox()
        self.step_spin = QtWidgets.QSpinBox()
        for widget, low, high, value in (
            (self.start_spin, 0, 999999, 1),
            (self.end_spin, 0, 999999, 1),
            (self.step_spin, 1, 9999, 1),
        ):
            widget.setRange(low, high)
            widget.setValue(value)
            widget.setFixedWidth(84)
        for label, widget in (("first", self.start_spin), ("last", self.end_spin),
                              ("every", self.step_spin)):
            fill.addWidget(QtWidgets.QLabel(label))
            fill.addWidget(widget)
        fill_button = QtWidgets.QPushButton("Fill list")
        fill_button.setToolTip(
            "Write first…last, every n, into the list below.\\n"
            "Numbers with no file are reported and skipped."
        )
        fill_button.clicked.connect(self.fill_list)
        fill.addWidget(fill_button)
        fill.addStretch(1)
        holder = QtWidgets.QWidget()
        holder.setLayout(fill)
        form.addRow("From a range", holder)

        self.scans_edit = QtWidgets.QLineEdit()
        self.scans_edit.setPlaceholderText("152, 154, 157-160")
        self.scans_edit.setToolTip(
            "Scan numbers, comma or space separated. A hyphen gives a run:\\n"
            "157-160 means 157, 158, 159, 160."
        )
        self.scans_edit.textChanged.connect(self._describe_selection)
        form.addRow("Scan numbers", self.scans_edit)
        self.selection_label = QtWidgets.QLabel("")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet("color: gray;")
        form.addRow("", self.selection_label)
        layout.addWidget(box)

        # -- the range --
        box = QtWidgets.QGroupBox("HKL range")
        grid = QtWidgets.QGridLayout(box)
        self.hkl_spins = {}
        for row_index, name in enumerate(AXIS_NAMES):
            grid.addWidget(QtWidgets.QLabel(name), row_index, 0)
            low = _spin(0.0, -1e6, 1e6, 4, 0.01, 104)
            high = _spin(1.0, -1e6, 1e6, 4, 0.01, 104)
            grid.addWidget(low, row_index, 1)
            grid.addWidget(high, row_index, 2)
            self.hkl_spins[name] = (low, high)
        self.whole_check = QtWidgets.QCheckBox("Whole extent of each scan")
        self.whole_check.setToolTip(
            "Off, every grid covers the same box and they can be compared cell\\n"
            "for cell -- which is the point of gridding a series.\\n"
            "On, each scan gets its own extent: the cell size still matches, but\\n"
            "the grids have different origins and shapes."
        )
        self.whole_check.toggled.connect(self._range_enabled)
        grid.addWidget(self.whole_check, 3, 0, 1, 2)
        from_button = QtWidgets.QPushButton("From the first scan")
        from_button.setToolTip("Fill the box with the extent the first scan reaches.")
        from_button.clicked.connect(self.range_from_first)
        grid.addWidget(from_button, 3, 2)
        layout.addWidget(box)

        # -- output --
        box = QtWidgets.QGroupBox("Output")
        form = QtWidgets.QFormLayout(box)
        self.oversample_spin = _spin(1.0, 0.25, 8.0, 2, 0.25, 84)
        self.oversample_spin.setToolTip(
            "Scales the cell size measured from the first scan. 1 is a cell\\n"
            "holding about one sample; above 1 is coarser and lighter."
        )
        form.addRow("Cell size ×", self.oversample_spin)
        self.transmission_check = QtWidgets.QCheckBox(
            "Divide by filter transmission where it varies"
        )
        self.transmission_check.setChecked(True)
        self.transmission_check.setToolTip(
            "A scan that changes its attenuator records frames on different\n"
            "scales. Dividing each frame by its transmission puts them on the\n"
            "unattenuated beam, so the map does not step where the filter moved.\n"
            "Scans whose filter never moved are unaffected either way."
        )
        form.addRow(self.transmission_check)
        self.reuse_check = QtWidgets.QCheckBox("Skip scans already gridded for this range")
        self.reuse_check.setChecked(True)
        form.addRow(self.reuse_check)
        self.output_label = QtWidgets.QLabel("—")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet("color: gray;")
        form.addRow("Folder", self.output_label)
        layout.addWidget(box)

        # -- run --
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Grid them")
        self.run_button.clicked.connect(self.run)
        row.addWidget(self.run_button)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._request_stop)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            'font-family: Menlo, Monaco, "Courier New", monospace;'
        )
        layout.addWidget(self.log, 1)

        self.status = self.statusBar()
        self.status.showMessage("Open the folder holding the scan files.")
        self._range_enabled()

    # -- the folder and the list -------------------------------------------

    def open_folder(self, folder=None):
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Open the folder of scans", str(self.folder or "")
            )
            if not folder:
                return
        folder = Path(folder)
        self.folder = folder
        self.found = find_scans(folder)
        self.folder_label.setText(str(folder))
        self.output_label.setText(str(folder / OUTPUT_DIRECTORY))
        try:
            overrides = sorted(read_setup(folder))
        except ValueError as exc:
            self.setup_label.setText(f"{exc} — it will stop the run")
        else:
            self.setup_label.setText(
                f"{SETUP_FILENAME} overrides: " + ", ".join(overrides)
                if overrides else f"no {SETUP_FILENAME} — the files decide the geometry"
            )
        if self.found:
            numbers = sorted(self.found)
            self.found_label.setText(
                f"{len(numbers)} scan(s) here: {numbers[0]}–{numbers[-1]}"
            )
            self.start_spin.setValue(numbers[0])
            self.end_spin.setValue(numbers[-1])
        else:
            self.found_label.setText("no scans in this folder")
        self._describe_selection()

    def fill_list(self):
        """Write the start/end/step range into the list of scan numbers."""
        numbers = expand_scans(
            self.start_spin.value(), self.end_spin.value(), self.step_spin.value()
        )
        self.scans_edit.setText(", ".join(str(n) for n in numbers))

    def selected(self):
        try:
            return parse_scans(self.scans_edit.text())
        except ValueError:
            return []

    def _describe_selection(self):
        numbers = self.selected()
        if not numbers:
            self.selection_label.setText("")
            return
        missing = [n for n in numbers if n not in self.found]
        text = f"{len(numbers)} scan(s) chosen"
        if missing:
            text += f" — {len(missing)} with no file: " + ", ".join(
                str(n) for n in missing[:12]
            ) + ("…" if len(missing) > 12 else "")
        self.selection_label.setText(text)

    def _range_enabled(self):
        live = not self.whole_check.isChecked()
        for low, high in self.hkl_spins.values():
            low.setEnabled(live)
            high.setEnabled(live)

    def hkl_range(self):
        if self.whole_check.isChecked():
            return None
        out = []
        for name in AXIS_NAMES:
            low, high = (widget.value() for widget in self.hkl_spins[name])
            out.extend((float(low), float(high)))
        return tuple(out)

    def range_from_first(self):
        """Fill the HKL box from the extent of the first chosen scan."""
        numbers = [n for n in self.selected() if n in self.found]
        if not numbers:
            self.status.showMessage("No scan chosen that has a file.")
            return
        image = self.found[numbers[0]]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            geometry = read_geometry(image, master_path_for_image(image))
            low, high = measured_range(geometry)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could not read", f"{image.name}\\n\\n{type(exc).__name__}: {exc}"
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.whole_check.setChecked(False)
        for index, name in enumerate(AXIS_NAMES):
            self.hkl_spins[name][0].setValue(float(low[index]))
            self.hkl_spins[name][1].setValue(float(high[index]))
        self.status.showMessage(f"range taken from scan {numbers[0]}")

    # -- running ------------------------------------------------------------

    def _request_stop(self):
        self._stop = True
        self.status.showMessage("stopping after this frame…")

    def _note(self, text):
        self.log.appendPlainText(text)
        QtWidgets.QApplication.processEvents()

    def run(self):
        if self._running:
            return
        if self.folder is None:
            self.status.showMessage("Open a folder first.")
            return
        numbers = self.selected()
        if not numbers:
            self.status.showMessage("No scan numbers given.")
            return

        self._running = True
        self._stop = False
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, len(numbers))
        self.progress.setValue(0)
        self.log.clear()
        hkl = self.hkl_range()
        self._note(f"{len(numbers)} scan(s) -> {self.folder / OUTPUT_DIRECTORY}")
        self._note(
            "range: " + ("each scan's own extent" if hkl is None
                         else "  ".join(f"{a:.4f}…{b:.4f}" for a, b in zip(hkl[0::2], hkl[1::2])))
        )
        self._note("cell size: measured from the first scan, reused for the rest")
        try:
            overrides = sorted(read_setup(self.folder))
        except ValueError as exc:
            self._note(f"setup: {exc}")
            overrides = []
        if overrides:
            self._note(f"setup: {SETUP_FILENAME} overrides " + ", ".join(overrides))
        self._note("")

        done = [0]

        def on_start(number, image):
            self.progress.setFormat(f"scan {number}  (%v of %m)")
            self._note(
                f"scan {number}: {image.name}" if image is not None
                else f"scan {number}: no file"
            )

        def on_geometry(number, geometry):
            if geometry.transmission_varies():
                note = ("corrected" if geometry.normalize_transmission
                        else "NOT corrected")
                values = geometry.transmission
                good = values[np.isfinite(values) & (values > 0)]
                self._note(
                    f"    filter moved during this scan "
                    f"(x{good.max() / good.min():,.0f} range) — {note}"
                )

        def report(number, frame, total):
            self.progress.setFormat(f"scan {number} — frame {frame}/{total}  (%v of %m)")
            QtWidgets.QApplication.processEvents()
            return not self._stop

        def on_done(number, path, note):
            done[0] += 1
            self.progress.setValue(done[0])
            self._note(f"    {note}" if path is None else f"    {note} -> {path.name}")

        try:
            results = grid_scans(
                self.folder,
                numbers,
                hkl_range=hkl,
                oversample=self.oversample_spin.value(),
                reuse=self.reuse_check.isChecked(),
                normalize_transmission=self.transmission_check.isChecked(),
                on_start=on_start,
                on_geometry=on_geometry,
                on_done=on_done,
                progress=report,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Gridding failed", f"{type(exc).__name__}: {exc}"
            )
            results = []
        finally:
            self._running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress.setFormat("%v of %m")

        made = sum(1 for _n, path, note in results if path is not None and note == "written")
        kept = sum(1 for _n, path, note in results if path is not None and note == "reused")
        failed = [(n, note) for n, path, note in results if path is None]
        summary = f"{made} written"
        if kept:
            summary += f", {kept} already there"
        if failed:
            summary += f", {len(failed)} skipped"
        if len(results) < len(numbers):
            summary += f", {len(numbers) - len(results)} not reached"
        self._note("")
        self._note(f"done: {summary}")
        for number, note in failed:
            self._note(f"  scan {number}: {note}")
        self.status.showMessage(f"{summary} — {self.folder / OUTPUT_DIRECTORY}")

    # -- the panel as a dict -----------------------------------------------

    @staticmethod
    def _amount(spin, value, what):
        """Set a fractional spin box, or say so rather than clamp.

        :func:`viewer_state.within` does this for whole numbers; the
        cell size and the six range ends are the places here that take
        a fraction.
        """
        number = float(value)
        low, high = spin.minimum(), spin.maximum()
        if not low <= number <= high:
            raise ValueError(f"{what} must be {low:g}..{high:g}; got {number:g}")
        spin.setValue(number)

    def _about(self) -> dict:
        """What the folder holds, as against what it was told to grid."""
        numbers = self.selected()
        return {
            "folder": str(self.folder) if self.folder is not None else None,
            "output_folder": (str(self.folder / OUTPUT_DIRECTORY)
                              if self.folder is not None else None),
            "scans_available": sorted(self.found),
            "scans_chosen": numbers,
            "scans_missing": [n for n in numbers if n not in self.found],
            "setup": self.setup_label.text(),
            "running": bool(self._running),
        }

    def state(self) -> dict:
        """Everything the panel says, as a dict :meth:`set_state` takes back.

        See :mod:`viewer_state` for what the pair is for and the rules
        they keep. Which folder is open is *not* in here -- it is under
        ``"about"``, with the scans found in it -- because opening one
        is :meth:`open_folder`'s job, not a setting.

        ``scans`` is the line as it is written rather than the numbers
        it comes to, so ``"157-160"`` survives the trip as itself. Put
        a list back instead and it is written out as one.
        """
        return {
            "scans": self.scans_edit.text(),
            "first": int(self.start_spin.value()),
            "last": int(self.end_spin.value()),
            "every": int(self.step_spin.value()),
            "hkl_range": {name: [float(low.value()), float(high.value())]
                          for name, (low, high) in self.hkl_spins.items()},
            "whole_extent": self.whole_check.isChecked(),
            "oversample": float(self.oversample_spin.value()),
            "transmission": self.transmission_check.isChecked(),
            "reuse": self.reuse_check.isChecked(),
            viewer_state.ABOUT: self._about(),
        }

    def _set_scans(self, scans):
        """Write the list of scan numbers, from a line or from a list."""
        if isinstance(scans, (list, tuple)):
            text = ", ".join(str(int(number)) for number in scans)
        else:
            text = str(scans)
        # The window's own reader shrugs a bad line off as "nothing
        # chosen", which from here would look like it worked.
        if text.strip():
            parse_scans(text)
        self.scans_edit.setText(text)

    def _set_hkl_range(self, ranges):
        """Place the box the grids share, one axis or all three."""
        if not isinstance(ranges, dict):
            raise TypeError('hkl_range is {"H": [low, high], ...}, not '
                            f"{type(ranges).__name__}")
        unknown = set(ranges) - set(self.hkl_spins)
        if unknown:
            raise ValueError(f"no such axis: {', '.join(sorted(unknown))}. "
                             f"Axes: {', '.join(self.hkl_spins)}")
        for name, bounds in ranges.items():
            low, high = viewer_state.pair(bounds, f"the {name} range")
            if low >= high:
                raise ValueError(f"the {name} range runs low to high; got "
                                 f"{low:g} to {high:g}")
            self._amount(self.hkl_spins[name][0], low, f"{name} low")
            self._amount(self.hkl_spins[name][1], high, f"{name} high")

    def set_state(self, state) -> list:
        """Drive the panel from a dict. Returns the settings applied.

        The range goes in before the tick above it: ticking "whole
        extent" greys the six boxes out, so a dict holding both means
        the tick, and it is applied last and wins.

        Nothing here is applied while a run is going. The settings are
        read once, at the top of :meth:`run`, and the log and the
        progress bar are written against them for minutes afterwards --
        so a dict landing mid-run would change what the window says it
        is doing without changing what it does.
        """
        if self._running:
            raise RuntimeError("the gridder is running; stop it before "
                               "changing what it grids")
        setters = {
            "scans": self._set_scans,
            "first": lambda v: viewer_state.within(
                self.start_spin, v, "the first scan"),
            "last": lambda v: viewer_state.within(
                self.end_spin, v, "the last scan"),
            "every": lambda v: viewer_state.within(
                self.step_spin, v, "every"),
            "hkl_range": self._set_hkl_range,
            "whole_extent": self.whole_check.setChecked,
            "oversample": lambda v: self._amount(
                self.oversample_spin, v, "the cell size"),
            "transmission": self.transmission_check.setChecked,
            "reuse": self.reuse_check.setChecked,
        }
        done = viewer_state.apply(state, setters)
        self._range_enabled()
        return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", help="folder of scan files (optional)")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    window = BatchGridWindow(args.folder)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
