"""Reading a master file: its metadata, its streams and the numbers in them.

Everything a scan says about itself -- which plan ran, what moved, the channels
it recorded point by point, the baseline around it -- plus the derived
quantities the viewers show: peak statistics, a two-motor raster reshaped into
a map, where the diffractometer was pointing.

No Qt. A notebook imports this directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import h5py

METADATA_PATH = "entry/instrument/bluesky/metadata"

def _meta_text(group, key: str) -> str:
    item = group.get(key)
    return "" if item is None else _as_text(item[()])

def _as_text(value) -> str:
    """An h5py scalar -- bytes, numpy scalar or 0-d array -- as plain text."""
    value = np.asarray(value).reshape(-1)
    if value.size == 0:
        return ""
    item = value[0]
    return item.decode(errors="replace") if isinstance(item, bytes) else str(item)

def _yaml_items(text: str) -> list[str]:
    """The entries of a simple YAML block sequence (``- name`` per line).

    ``motors`` and ``detectors`` are stored as YAML text (``motors`` even keeps a
    ``!!python/tuple`` tag), so pull the items out directly rather than take a
    PyYAML dependency for two one-line lists.
    """
    return [
        line[2:].strip().strip("'\"")
        for line in text.splitlines()
        if line.startswith("- ")
    ]


def _hint_dimensions(text: str) -> list[str]:
    """The axes the ``dimensions`` hint names, outermost first.

    Bluesky hints one ``(fields, stream)`` tuple per scanned dimension, which
    PyYAML writes two or three levels deep with ``!!python/tuple`` tags::

        dimensions:
        - !!python/tuple
          - - psic_eta
          - primary

    Only the field names are wanted, so the trailing stream name -- the last
    leaf of each tuple -- is dropped. A plan that moved nothing writes an
    empty list there, which is not an axis and is left out.
    """
    groups: list[list[str]] = []
    collecting = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if not collecting:
            collecting = line.rstrip() == "dimensions:"
            continue
        if line.startswith("- "):
            groups.append([])               # one more (fields, stream) tuple
        elif not line[:1].isspace():
            break                           # a new top-level key ends the block
        item = line.strip()
        while item.startswith("- "):
            item = item[2:].strip()
        if item and not item.startswith("!!") and groups:
            groups[-1].append(item.strip("'\""))
    return [name for group in groups for name in group[:-1] if name != "[]"]


def _yaml_block(text: str, key: str) -> str:
    """The body under ``key:`` in a simple YAML mapping, as text.

    ``hints`` holds several sections in one blob, so :func:`_yaml_items` cannot
    be pointed at the whole thing -- it would collect the list items of every
    section at once. The body runs until the next key at column zero.
    """
    out, inside = [], False
    for line in text.splitlines():
        if inside:
            if not line.strip() or line[:1] not in (" ", "-"):
                break
            out.append(line)
        elif line.strip() == f"{key}:":
            inside = True
    return "\n".join(out)

def _half_max_crossing(
    x: np.ndarray, y: np.ndarray, half: float, start: int, stop: int, step: int
) -> float | None:
    """Linearly-interpolated X where *y* first crosses *half*, walking from
    *start* to (but not including) *stop* in steps of *step*."""
    i = start
    while i != stop:
        j = i + step
        yi, yj = y[i], y[j]
        if yi == half:
            return x[i]
        if (yi > half) != (yj > half):
            frac = (half - yi) / (yj - yi)
            return x[i] + frac * (x[j] - x[i])
        i = j
    return None

def _grid_axis_candidates(available: list[str], motor: str) -> list[str]:
    """Fields that could supply this motor's grid axis, best first.

    The readback comes first, because it is where the motor actually was. It is
    only usable when every row repeats its coordinates exactly, which a readback
    that wanders by a few counts will not; the ``*_user_setpoint`` companion
    holds the demanded value and always does. :func:`read_grid` tries them in
    this order and keeps the first that reconstructs a raster.
    """
    names = (motor, f"{motor}_user_setpoint", f"{motor}_setpoint")
    return [name for name in names if name in available]

PRIMARY_PATH = "entry/instrument/bluesky/streams/primary"

BASELINE_PATH = "entry/instrument/bluesky/streams/baseline"

# Offered alongside the real fields: each primary field carries its own EPOCH
# (seconds past the UNIX epoch, one per scan point), so absolute time can be an
# axis like anything else.
EPOCH_FIELD = "EPOCH"

# A plain 0, 1, 2 ... down the scan. A count plan moves nothing, so it has no
# motor to plot against and the fallback would otherwise settle on whichever
# motor happens to sort last -- a flat line at wherever it was parked.
INDEX_FIELD = "INDEX"

# Reading this opens the file, and the browser re-reads the whole selection on
# every selection change, so remember it per (path, mtime, size).
# The diffractometer's own reading of where it was pointing, per scan point.
# Older scans do not record it.
PSIC_HKL_FIELDS = ("psic_h", "psic_k", "psic_l")

_SCAN_HKL_CACHE: dict[tuple, tuple | None] = {}

_SCAN_INFO_CACHE: dict[tuple, dict] = {}

def _is_time_like(name: str) -> bool:
    return "time" in name.lower() or name == EPOCH_FIELD

def resolve_hint_field(name: str, fields: list[str]) -> str | None:
    """The recorded field a hinted name stands for, or None for none.

    A motor is hinted under its own name but may only have been written as a
    setpoint, so the readback is preferred and the setpoint accepted after
    it.

    ``time`` is Bluesky's pseudo-axis for a plan that moved nothing. That
    resolves to :data:`INDEX_FIELD` rather than :data:`EPOCH_FIELD`: EPOCH
    is absolute seconds, so a count scan plotted against it gets an axis
    reading 1790516140 to 1790516890 where the index reads 0 to 599. EPOCH
    is one tick away for a scan whose dwell actually varied.
    """
    if not name:
        return None
    if name == "time":
        for pseudo in (INDEX_FIELD, EPOCH_FIELD):
            if pseudo in fields:
                return pseudo
        return None
    for candidate in (name, f"{name}_user_setpoint", f"{name}_setpoint"):
        if candidate in fields:
            return candidate
    return None


def scan_y_fields(path: Path, fields: list[str]) -> list[str]:
    """The detectors the scan itself nominates to plot, from its ``hints``.

    The hints name the one signal per detector worth looking at --
    ``lambda250k_stats3_total`` out of that detector's half-dozen columns --
    so a scan can open on what it was run to measure instead of on nothing.
    Empty when the scan hints at nothing, or at fields it did not record.
    """
    chosen: list[str] = []
    for name in read_scan_info(path)["hints"]["detectors"]:
        resolved = resolve_hint_field(name, fields)
        if resolved is not None and resolved not in chosen:
            chosen.append(resolved)
    return chosen


def stream_fields(path: Path, stream: str) -> list[str]:
    """Field names under *stream* that carry a ``value`` dataset.

    h5py iterates group keys in name order (no ``track_order`` here), so this
    is also the order ``default_x_field`` treats as "last".
    """
    try:
        with h5py.File(path, "r") as f:
            group = f.get(f"entry/instrument/bluesky/streams/{stream}")
            if group is None:
                return []
            names = [
                name
                for name in group.keys()
                if isinstance(group[name], h5py.Group) and "value" in group[name]
            ]
            if _epoch_source(group) is not None:
                names.append(EPOCH_FIELD)
            if names:
                # Last, so default_x_field reaches for it before any parked
                # motor when the scan itself names none.
                names.append(INDEX_FIELD)
            return names
    except OSError:
        return []

def _epoch_source(group) -> str | None:
    """The field whose EPOCH stands for the scan's clock.

    Every field records its own EPOCH and they disagree by a few seconds --
    each device is stamped when it was read -- so one has to be picked. The
    first in the group's (alphabetical) order is used, consistently, rather
    than silently mixing time bases between axes.
    """
    for name in group.keys():
        item = group[name]
        if isinstance(item, h5py.Group) and "EPOCH" in item:
            return name
    return None

def _primary_point_count(group) -> int | None:
    """How many points the stream has: the most common length among its fields.

    Read off each dataset's shape rather than its contents, and by majority
    because a few fields (read-once presets and the like) are a different
    length from the scan.
    """
    counts: dict[int, int] = {}
    for name in group.keys():
        item = group[name]
        if not isinstance(item, h5py.Group) or "value" not in item:
            continue
        values = item["value"]
        if not isinstance(values, h5py.Dataset):
            continue
        try:
            size = int(values.size)
        except (AttributeError, TypeError, ValueError):
            continue
        if size >= 1:
            counts[size] = counts.get(size, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]

def read_primary(path: Path, fields: list[str]) -> dict[str, np.ndarray]:
    """Read the ``value`` array of each of *fields* from the primary stream."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        group = f.get(PRIMARY_PATH)
        if group is None:
            return out
        epoch_source = _epoch_source(group)
        for name in fields:
            if name == INDEX_FIELD:
                count = _primary_point_count(group)
                if count:
                    out[name] = np.arange(count, dtype=float)
                continue
            if name == EPOCH_FIELD:
                if epoch_source is not None:
                    out[name] = np.asarray(
                        group[epoch_source]["EPOCH"][()], dtype=float
                    )
                continue
            item = group.get(name)
            if item is not None and "value" in item:
                out[name] = np.asarray(item["value"][()], dtype=float)
    return out

def _scalarize(value: np.ndarray):
    """Return a plain float for a numeric size-1 array, else a short text summary.

    Some baseline fields are waveform-valued (e.g. a UB matrix or a
    pixel-center pair) rather than scalar, so ``value_start``/``value_end``
    are not always convertible with a bare ``float()``. Others are size-1
    but string-valued (e.g. a motor direction like ``b"y-"``), which
    ``float()`` also rejects.
    """
    value = np.asarray(value)
    if value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode(errors="replace")
        try:
            return float(item)
        except (TypeError, ValueError):
            return str(item)
    return f"array{value.shape}: {np.array2string(value, precision=4, threshold=6)}"

def read_baseline(path: Path) -> dict[str, tuple[object, object]]:
    """Return ``{field: (before, after)}`` from the baseline stream.

    ``before``/``after`` are ``float`` for a scalar field, or a text summary
    for an array-valued one (see :func:`_scalarize`).
    """
    out: dict[str, tuple[object, object]] = {}
    with h5py.File(path, "r") as f:
        group = f.get(BASELINE_PATH)
        if group is None:
            return out
        for name in group.keys():
            sub = group[name]
            if not isinstance(sub, h5py.Group):
                continue
            if "value_start" in sub and "value_end" in sub:
                before = _scalarize(sub["value_start"][()])
                after = _scalarize(sub["value_end"][()])
            elif "value" in sub and sub["value"].shape and sub["value"].shape[0] >= 2:
                value = np.asarray(sub["value"][()])
                before, after = _scalarize(value[0]), _scalarize(value[-1])
            else:
                continue
            out[name] = (before, after)
    return out

def read_scan_info(path: Path) -> dict:
    """What the scan was: plan, points, motors and the range they covered.

    The motor names come from the ``motors`` metadata, and each one's actual
    travel is read back from its own field in the primary stream -- the
    requested range in ``plan_args`` is not necessarily what the motor did.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        key = None
    if key is not None and key in _SCAN_INFO_CACHE:
        return _SCAN_INFO_CACHE[key]

    info = {
        "file": path.name,
        "is_scan": False,
        "plan": "",
        "points": "",
        "motors": [],
        # What the plan itself says is worth plotting: the axis of each
        # scanned dimension, and one signal per detector.
        "hints": {"dimensions": [], "detectors": []},
        "shape": [],
        "detectors": [],
        "ranges": [],          # (motor, first, last, n)
        "count_time": None,
        "started": "",
        "plan_args": "",
        "uid": "",
    }
    try:
        with h5py.File(path, "r") as f:
            meta = f.get(METADATA_PATH)
            if meta is not None:
                info["is_scan"] = True
                info["plan"] = _meta_text(meta, "plan_name")
                info["points"] = _meta_text(meta, "num_points")
                info["motors"] = _yaml_items(_meta_text(meta, "motors"))
                hints = _meta_text(meta, "hints")
                info["hints"] = {
                    "dimensions": _hint_dimensions(hints),
                    "detectors": _yaml_items(_yaml_block(hints, "detectors")),
                }
                # grid_scan records the raster size here; a 1-D plan has none.
                info["shape"] = [
                    int(n) for n in _yaml_items(_meta_text(meta, "shape"))
                    if n.lstrip("-").isdigit()
                ]
                info["detectors"] = _yaml_items(_meta_text(meta, "detectors"))
                info["plan_args"] = _meta_text(meta, "plan_args")
                info["uid"] = _meta_text(meta, "run_start_uid")

            primary = f.get(PRIMARY_PATH)
            if primary is not None:
                for motor in info["motors"]:
                    sub = primary.get(motor)
                    if sub is None or "value" not in sub:
                        continue
                    values = np.asarray(sub["value"][()], dtype=float).ravel()
                    if values.size:
                        info["ranges"].append(
                            (motor, float(values[0]), float(values[-1]), int(values.size))
                        )
                counter = primary.get("scaler_time")
                if counter is not None and "value" in counter:
                    seconds = np.asarray(counter["value"][()], dtype=float).ravel()
                    if seconds.size:
                        info["count_time"] = float(seconds[0])
                for name in primary:
                    field = primary[name]
                    stamp = field.get("time") if isinstance(field, h5py.Group) else None
                    if stamp is not None and "start_time_iso" in stamp.attrs:
                        info["started"] = _as_text(stamp.attrs["start_time_iso"])[:19]
                        break
    except OSError:
        pass

    if key is not None:
        _SCAN_INFO_CACHE[key] = info
    return info


# A counter channel is named by whoever wired it up ("Ion_Chamber_1"), so the
# name alone says nothing. The PV it was read from does: these are the EPICS
# motor-record and ophyd pseudo-positioner suffixes, and nothing that merely
# counts carries one.
_POSITIONER_SUFFIXES = (".RBV", ".VAL", ".position", ".target")


def _field_device(name: str, devices: list[str]) -> str:
    """The configured device a field name was prefixed with, if any."""
    best = ""
    for device in devices:
        if (name == device or name.startswith(f"{device}_")) and len(device) > len(best):
            best = device
    return best


def describe_fields(path: Path, stream: str = "primary") -> list[dict]:
    """Every field in *stream*, with what it is and the range it covered.

    This answers the question you have in front of an unfamiliar scan: which
    name goes in as x, as y, as the monitor. Nothing here is guessed from the
    name -- *role* comes from the plan's own ``motors`` list, the
    ``signal_type`` each field group carries and the PV it was read from, and
    *hinted* marks the detector the plan itself nominated in ``hints``.

    Returns one dict per field, the motor the scan moved first and the fields
    that never moved last.
    """
    path = Path(path)
    names = stream_fields(path, stream)
    if not names:
        return []
    info = read_scan_info(path)
    scanned = set(info.get("motors") or [])
    detectors = list(info.get("detectors") or [])

    # The hint names a device's signal; mark whichever field actually
    # carries it, which for a motor may be the setpoint.
    hinted = {resolve_hint_field(name, names)
              for name in info["hints"]["detectors"]}
    hinted.discard(None)
    attrs: dict[str, tuple[str, str, str]] = {}
    try:
        with h5py.File(path, "r") as f:
            group = f.get(f"entry/instrument/bluesky/streams/{stream}")
            for name in names:
                sub = group.get(name) if group is not None else None
                if not isinstance(sub, h5py.Group) or "value" not in sub:
                    continue
                attrs[name] = (
                    _as_text(sub.attrs.get("signal_type", "")),
                    _as_text(sub["value"].attrs.get("units", "")),
                    _as_text(sub["value"].attrs.get("source", "")),
                )
    except OSError:
        pass

    values = read_primary(path, names) if stream == "primary" else {}

    rows = []
    for name in names:
        signal, units, source = attrs.get(name, ("", "", ""))
        if name == INDEX_FIELD:
            role = "index"
        elif _is_time_like(name):
            role = "time"
        elif name in scanned:
            role = "scanned"
        elif source.endswith(_POSITIONER_SUFFIXES):
            role = "positioner"
        elif signal:
            role = signal
        elif _field_device(name, detectors):
            role = "detector"
        else:
            role = "other"
        row = {
            "field": name,
            "role": role,
            "hinted": name in hinted,
            "units": units,
            "varies": None,
            "min": None,
            "max": None,
            "source": source,
        }
        column = values.get(name)
        if column is not None and column.ndim == 1 and column.size:
            try:
                numbers = np.asarray(column, dtype=float)
            except (TypeError, ValueError):
                numbers = None
            if numbers is not None and np.isfinite(numbers).any():
                low, high = float(np.nanmin(numbers)), float(np.nanmax(numbers))
                row.update(varies=bool(high > low), min=low, max=high)
        rows.append(row)

    def order(row):
        if row["role"] == "scanned":
            rank = 0
        elif row["hinted"]:
            rank = 1
        elif row["varies"]:
            rank = 2
        else:
            rank = 3
        return (rank, row["field"])

    rows.sort(key=order)
    return rows

def default_x_field(fields: list[str]) -> str | None:
    """The last field, skipping trailing time-like ones (e.g. ``scaler_time``).

    Only a fallback: :func:`scan_x_field` asks the scan what it moved.
    """
    for name in reversed(fields):
        if not _is_time_like(name):
            return name
    return fields[-1] if fields else None

def scan_x_field(path: Path, fields: list[str]) -> str | None:
    """The field to plot against: the axis this scan actually scanned.

    Taken from the scan's own ``hints`` -- ``dimensions`` names the axis of
    each scanned dimension -- and failing that from the ``motors``
    metadata, rather than guessed from the field ordering. The readback is
    preferred over the setpoint. Returns ``None`` when the scan names no
    axis that was recorded.
    """
    info = read_scan_info(path)
    for name in (*info["hints"]["dimensions"], *info["motors"]):
        resolved = resolve_hint_field(name, fields)
        if resolved is not None:
            return resolved
    return None

def scan_hkl(path: Path):
    """Mean and spread of psic_h/k/l over the scan, or None if not recorded.

    The spread matters as much as the mean: on a rocking curve one of the three
    sweeps across the peak while the others hold still, and a bare average
    would read like a fixed position.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        key = None
    if key is not None and key in _SCAN_HKL_CACHE:
        return _SCAN_HKL_CACHE[key]

    result = None
    if set(PSIC_HKL_FIELDS) <= set(stream_fields(path, "primary")):
        try:
            data = read_primary(path, list(PSIC_HKL_FIELDS))
        except OSError:
            data = {}
        gathered = []
        for name in PSIC_HKL_FIELDS:
            values = np.asarray(data.get(name, ()), dtype=float).ravel()
            values = values[np.isfinite(values)]
            if values.size == 0:
                gathered = []
                break
            gathered.append((float(values.mean()), float(values.std())))
        result = tuple(gathered) if gathered else None
    if key is not None:
        _SCAN_HKL_CACHE[key] = result
    return result

def format_scan_hkl(values, decimals: int = 4) -> str:
    """``H 1.0135  K -0.4926 ± 0.0386  L 7.4840`` for one scan."""
    smallest = 0.5 * 10.0 ** (-decimals)
    parts = []
    for name, (mean, spread) in zip("HKL", values):
        text = f"{name} {mean:.{decimals}f}"
        if spread >= smallest:
            text += f" ± {spread:.{decimals}f}"
        parts.append(text)
    return "   ".join(parts)

def _build_grid(slow, fast, values, shape, slow_name, fast_name) -> dict | None:
    """Lay *values* onto the raster described by the two axis arrays, or ``None``."""
    n_points = min(len(slow), len(fast), len(values))
    if n_points < 4:
        return None
    slow, fast, values = slow[:n_points], fast[:n_points], values[:n_points]

    # return_inverse gives each point's row/column straight off, so a snaked
    # raster (alternate rows reversed) lands in the right place unaided.
    y_axis, rows = np.unique(slow, return_inverse=True)
    x_axis, columns = np.unique(fast, return_inverse=True)
    if len(x_axis) < 2 or len(y_axis) < 2:
        return None

    complete = len(x_axis) * len(y_axis) == n_points
    # A scan stopped part way has fewer rows than were asked for, so compare
    # against the declared raster rather than expecting a full rectangle: the
    # fast axis must be the declared width, the slow axis no taller.
    declared = (
        len(shape) == 2 and len(x_axis) == shape[1] and len(y_axis) <= shape[0]
    )
    if not (complete or declared):
        return None

    grid = np.full((len(y_axis), len(x_axis)), np.nan)
    grid[rows, columns] = values
    return {
        "x_name": fast_name,
        "y_name": slow_name,
        "x": x_axis,
        "y": y_axis,
        "z": grid,
        "points": n_points,
    }

def read_grid(
    path: Path, value_field: str, monitor_field: str = ""
) -> dict | None:
    """*value_field* laid out over a two-motor grid scan, or ``None``.

    ``None`` means "not a raster": a two-motor ``inner_product`` scan moves both
    motors together, so its axes have as many levels as there are points and the
    product test in :func:`_build_grid` fails, which is exactly what tells the
    two plans apart without relying on the plan name.
    """
    info = read_scan_info(path)
    if len(info["motors"]) != 2:
        return None

    available = stream_fields(path, "primary")
    slow_options = _grid_axis_candidates(available, info["motors"][0])
    fast_options = _grid_axis_candidates(available, info["motors"][1])
    if not slow_options or not fast_options or value_field not in available:
        return None

    wanted = set(slow_options) | set(fast_options) | {value_field}
    if monitor_field:
        wanted.add(monitor_field)
    try:
        data = read_primary(path, list(wanted))
    except OSError:
        return None
    if value_field not in data:
        return None

    values = data[value_field]
    monitor = data.get(monitor_field) if monitor_field else None
    if monitor is not None:
        length = min(len(values), len(monitor))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(
                monitor[:length] != 0, values[:length] / monitor[:length], np.nan
            )

    # Readback first, setpoint only if the readback will not form a raster.
    for slow_name in slow_options:
        for fast_name in fast_options:
            if slow_name not in data or fast_name not in data:
                continue
            grid = _build_grid(
                data[slow_name], data[fast_name], values,
                info["shape"], slow_name, fast_name,
            )
            if grid is not None:
                return grid
    return None

def axis_edges(values: np.ndarray) -> np.ndarray:
    """Cell boundaries around the grid's sample positions.

    Midway between neighbours, with the outer two extended by half their own
    spacing, so uneven steps keep their proportions.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    middles = (v[1:] + v[:-1]) / 2.0
    return np.concatenate(([2 * v[0] - middles[0]], middles, [2 * v[-1] - middles[-1]]))

def curve_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """Max/min (with position), and FWHM center/width, for one plotted curve.

    ``center``/``fwhm`` come from linearly-interpolated crossings of the
    half-max level (halfway between the curve's min and max) either side of
    the peak -- the same convention tools like SPEC's peak search use --
    rather than a fit, so they are ``None`` when the peak sits at an end of
    the data (no crossing on that side).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    stats: dict[str, float | None] = {
        "max_y": None,
        "max_x": None,
        "min_y": None,
        "min_x": None,
        "center": None,
        "fwhm": None,
    }
    if x.size == 0:
        return stats

    imax = int(np.argmax(y))
    imin = int(np.argmin(y))
    stats["max_y"], stats["max_x"] = y[imax], x[imax]
    stats["min_y"], stats["min_x"] = y[imin], x[imin]

    half = (y[imax] + y[imin]) / 2.0
    left = _half_max_crossing(x, y, half, imax, 0, -1)
    right = _half_max_crossing(x, y, half, imax, len(x) - 1, 1)
    if left is not None and right is not None:
        stats["center"] = (left + right) / 2.0
        stats["fwhm"] = abs(right - left)
    return stats

def grid_stats(grid: dict) -> dict[str, float | None]:
    """Max/min of a grid and the (x, y) they sit at."""
    z = grid["z"]
    stats = {"max_z": None, "max_at": None, "min_z": None, "min_at": None}
    if not np.any(np.isfinite(z)):
        return stats
    flat_max = int(np.nanargmax(z))
    flat_min = int(np.nanargmin(z))
    for key, flat in (("max", flat_max), ("min", flat_min)):
        row, column = np.unravel_index(flat, z.shape)
        stats[f"{key}_z"] = float(z[row, column])
        stats[f"{key}_at"] = (float(grid["x"][column]), float(grid["y"][row]))
    return stats
