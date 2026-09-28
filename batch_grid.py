"""Grid a run of scans onto one common HKL lattice, as ``.vti`` files.

One scan at a time through the HKL viewer is fine for looking; comparing a
series means gridding them all the same way. This takes a folder of ``pva*.h5``
files, a list of scan numbers and an HKL range, and writes one ``.vti`` per
scan into an ``output`` folder beside them.

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

from hkl_grid import GridCancelled
from hkl_grid import grid_to_vti
from hkl_grid import measured_range
from hkl_grid import natural_step
from hkl_viewer import read_geometry
from image_viewer import master_path_for_image
from scan_viewer import scan_number
from volume_style import AXIS_NAMES
from volume_style import spin as _spin

OUTPUT_DIRECTORY = "output"


def expand_scans(start, end, step=1):
    """The scan numbers from *start* to *end* inclusive, every *step*.

    Given 152, 160, 2 this is [152, 154, 156, 158, 160]. The ends may be the
    other way round, and a step of zero is taken to mean one.
    """
    start, end, step = int(start), int(end), int(step)
    step = abs(step) or 1
    if end < start:
        start, end = end, start
    return list(range(start, end + 1, step))


def parse_scans(text):
    """Scan numbers from what someone typed: "152, 154 157-160"."""
    numbers = []
    for chunk in re.split(r"[,\s]+", str(text).strip()):
        if not chunk:
            continue
        span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if span:
            numbers.extend(expand_scans(span.group(1), span.group(2)))
        elif chunk.isdigit():
            numbers.append(int(chunk))
        else:
            raise ValueError(f"not a scan number: {chunk!r}")
    seen, unique = set(), []
    for number in numbers:          # keep the order given, drop repeats
        if number not in seen:
            seen.add(number)
            unique.append(number)
    return unique


def find_scans(folder):
    """``{scan number: image path}`` for the pva files in *folder*."""
    found = {}
    for path in sorted(Path(folder).glob("pva*.h5")):
        try:
            found[int(scan_number(path))] = path
        except ValueError:
            continue
    return found


def grid_scans(
    folder,
    numbers,
    hkl_range=None,
    oversample: float = 1.0,
    output=None,
    step=None,
    reuse: bool = True,
    on_start=None,
    on_done=None,
    progress=None,
):
    """Grid several scans onto one lattice. Returns ``[(number, path, note)]``.

    ``path`` is None for a scan that could not be done, and ``note`` says why.
    The cell size comes from the first scan that reads successfully unless
    *step* is given outright.
    """
    folder = Path(folder)
    found = find_scans(folder)
    out_dir = Path(output) if output is not None else folder / OUTPUT_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for number in numbers:
        image = found.get(int(number))
        if on_start is not None:
            on_start(number, image)
        if image is None:
            results.append((number, None, "no pva*.h5 with that scan number"))
            if on_done is not None:
                on_done(*results[-1])
            continue
        try:
            geometry = read_geometry(image, master_path_for_image(image))
            if step is None:
                # The first scan sets the lattice; everything after it is
                # measured onto the same one.
                step = natural_step(geometry, oversample=oversample)
            target = out_dir / f"{image.stem}_hkl_grid.vti"
            # Cheaper than asking whether the cached file matches: if grid_to_vti
            # reused it, it will not have touched it.
            stamp = target.stat().st_mtime if target.is_file() else None
            written = grid_to_vti(
                image,
                hkl_range=hkl_range,
                output=target,
                step=step,
                oversample=oversample,
                geometry=geometry,
                reuse=reuse,
                progress=(None if progress is None
                          else (lambda done, total, n=number: progress(n, done, total))),
            )
            reused = stamp is not None and written.stat().st_mtime == stamp
            results.append((number, written, "reused" if reused else "written"))
        except GridCancelled as stop:
            results.append((number, None, f"cancelled ({stop})"))
            if on_done is not None:
                on_done(*results[-1])
            break
        except Exception as exc:
            results.append((number, None, f"{type(exc).__name__}: {exc}"))
        if on_done is not None:
            on_done(*results[-1])
    return results


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
        button.setToolTip("The folder holding the pva*.h5 files and their masters.")
        button.clicked.connect(lambda: self.open_folder())
        row.addWidget(button)
        self.folder_label = QtWidgets.QLabel("no folder")
        self.folder_label.setStyleSheet("color: gray;")
        row.addWidget(self.folder_label, 1)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        form.addRow(holder)

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
        self.status.showMessage("Open the folder holding the pva*.h5 files.")
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
        if self.found:
            numbers = sorted(self.found)
            self.found_label.setText(
                f"{len(numbers)} scan(s) here: {numbers[0]}–{numbers[-1]}"
            )
            self.start_spin.setValue(numbers[0])
            self.end_spin.setValue(numbers[-1])
        else:
            self.found_label.setText("no pva*.h5 files in this folder")
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
        self._note("")

        done = [0]

        def on_start(number, image):
            self.progress.setFormat(f"scan {number}  (%v of %m)")
            self._note(
                f"scan {number}: {image.name}" if image is not None
                else f"scan {number}: no file"
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
                on_start=on_start,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", help="folder of pva*.h5 files (optional)")
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    window = BatchGridWindow(args.folder)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
