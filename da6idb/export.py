"""Writing a slice or a cut out to a file.

Text, CSV, a NumPy archive or HDF5, each carrying a header that says where the
numbers came from -- the source file, the plane, how thick a slab was averaged
-- so a file found again months later can still be read.

These used to be methods on a window, which meant a notebook could not export
anything without reimplementing them. They take plain arrays now.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SUFFIXES = (".txt", ".dat", ".csv", ".npz", ".h5", ".hdf5")


def slug(text) -> str:
    """A label that is safe in a file name or an HDF5 key."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or "axis"


def suffix_for(path, selected: str = "") -> str:
    """Which writer to use: the extension if it says, else the chosen filter."""
    extension = Path(path).suffix.lower()
    if extension in SUFFIXES:
        return extension
    for marker, extension in (
        ("CSV", ".csv"), ("NumPy", ".npz"), ("HDF5", ".h5"), ("Text", ".txt")
    ):
        if selected.startswith(marker):
            return extension
    return ".txt"


def _axis_note(label, values) -> str:
    values = np.asarray(values, dtype=float)
    step = (values[1] - values[0]) if values.size > 1 else 0.0
    return ("%s axis: %s  from %.6f to %.6f  (%d points, step %.6g)"
            % (label, label, values[0], values[-1], values.size, step))


def write_matrix(path, image, x, y, x_label="x", y_label="y",
                 description=(), suffix=None) -> Path:
    """A 2D array with its two axes. Unmeasured cells are written as NaN."""
    path = Path(path)
    suffix = suffix or suffix_for(path)
    image = np.ma.filled(np.ma.asarray(image), np.nan)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    header = list(description) + [
        _axis_note(x_label, x).replace("%s axis" % x_label, "x axis", 1),
        _axis_note(y_label, y).replace("%s axis" % y_label, "y axis", 1),
        "rows run along y, columns along x; unmeasured cells are nan",
    ]
    if suffix in (".txt", ".dat"):
        np.savetxt(path, image, fmt="%.6g", header="\n".join(header))
    elif suffix == ".csv":
        # First row the x values, first column the y values, so it opens in a
        # spreadsheet as the map it is.
        table = np.full((image.shape[0] + 1, image.shape[1] + 1), np.nan)
        table[0, 1:] = x
        table[1:, 0] = y
        table[1:, 1:] = image
        np.savetxt(path, table, fmt="%.6g", delimiter=",",
                   header="\n".join(header + [
                       f"first row: {x_label} values; first column: {y_label} values"]))
    elif suffix == ".npz":
        np.savez_compressed(path, intensity=image, x=x, y=y, x_label=x_label,
                            y_label=y_label, description="\n".join(header))
    else:
        import h5py

        with h5py.File(path, "w") as handle:
            handle.create_dataset("intensity", data=image, compression="gzip")
            handle.create_dataset("x", data=x)
            handle.create_dataset("y", data=y)
            handle["x"].attrs["label"] = x_label
            handle["y"].attrs["label"] = y_label
            handle.attrs["description"] = "\n".join(header)
    return path


# What a save dialog offers, in the order the writers below handle them.
# Here rather than in a window, so every window that exports agrees.
EXPORT_FILTERS = (
    "Text, columns or matrix (*.txt *.dat)"
    ";;CSV (*.csv)"
    ";;NumPy archive (*.npz)"
    ";;HDF5 (*.h5 *.hdf5)"
)


def table_columns(curves, x_label="x"):
    """``{heading: values}`` for *curves*, laid out side by side.

    *curves* is ``[(label, x, y), ...]``. Curves drawn against the **same X
    array** share one X column between them, so three detectors of one scan
    come out as ``psic_eta, det1, det2, det3`` rather than as three repeats
    of the same angles.

    The same array, not merely equal values, and the difference matters.
    Two scans of one thing read their motor at two different moments and
    differ in the fifth decimal: those are two honest X columns. Comparing
    values would weld them together, would weld any two one-point scans,
    and would never collapse an averaged curve, whose X can carry a NaN pad
    and so is not even equal to itself. Reading a scan's X once and handing
    the same array to each of its fields is what the caller already does,
    so identity answers exactly the question being asked -- were these
    drawn against one sampling.

    Columns come back at their own lengths; :func:`write_columns` squares
    them off.
    """
    columns: dict[str, object] = {}
    seen: dict[int, str] = {}
    used: set[str] = set()

    def unique(name):
        if name not in used:
            used.add(name)
            return name
        for n in range(2, 1000):
            candidate = f"{name}_{n}"
            if candidate not in used:
                used.add(candidate)
                return candidate
        return name

    for label, x, y in curves:
        key = id(x)
        if key not in seen:
            heading = unique(x_label if not seen else f"{x_label}_{len(seen) + 1}")
            seen[key] = heading
            columns[heading] = x
        columns[unique(label)] = y
    return columns


def write_columns(path, datasets, description=(), suffix=None, note_width=None):
    """Cuts as columns. Returns the names written.

    *datasets* is ``[(name, note, {column: values}), ...]``. Text and CSV take
    one file per cut, suffixing the name when there is more than one; the
    archive formats hold them all.
    """
    path = Path(path)
    suffix = suffix or suffix_for(path)
    provenance = list(description)
    written = []
    if suffix in (".txt", ".dat", ".csv"):
        delimiter = "," if suffix == ".csv" else " "
        for name, note, columns in datasets:
            target = (path if len(datasets) == 1
                      else path.with_name(f"{path.stem}_{name}{path.suffix}"))
            table = np.column_stack([np.asarray(v, float) for v in columns.values()])
            head = provenance + [f"cut: {name}", note]
            if note_width is not None:
                head.append(f"cut width: {note_width} px")
            head.append(delimiter.join(columns.keys()))
            np.savetxt(target, table, fmt="%.6g", delimiter=delimiter,
                       header="\n".join(head))
            written.append(target.name)
    elif suffix == ".npz":
        payload = {"description": "\n".join(provenance)}
        for name, note, columns in datasets:
            payload[f"{name}/note"] = np.array(note)
            for key, values in columns.items():
                payload[f"{name}/{slug(key)}"] = np.asarray(values, float)
        np.savez_compressed(path, **payload)
        written.append(path.name)
    else:
        import h5py

        with h5py.File(path, "w") as handle:
            handle.attrs["description"] = "\n".join(provenance)
            if note_width is not None:
                handle.attrs["cut_width_px"] = int(note_width)
            for name, note, columns in datasets:
                group = handle.create_group(name)
                group.attrs["note"] = note
                for key, values in columns.items():
                    group.create_dataset(slug(key), data=np.asarray(values, float))
        written.append(path.name)
    return written


def write_dataarray(array, path, description=()) -> Path:
    """Write a 2D :class:`xarray.DataArray`, taking its axes from the coords."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got dims {array.dims}")
    y_label, x_label = (str(d) for d in array.dims)
    notes = list(description) or [f"{array.name or 'intensity'} from da6idb"]
    notes += [f"{k}: {v}" for k, v in array.attrs.items() if np.ndim(v) == 0]
    return write_matrix(path, np.asarray(array.values),
                        np.asarray(array.coords[x_label].values),
                        np.asarray(array.coords[y_label].values),
                        x_label=x_label, y_label=y_label, description=notes)
