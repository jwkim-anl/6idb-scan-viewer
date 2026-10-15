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

def _dwell(seconds: np.ndarray) -> float | None:
    """The counting time per point in *seconds*, as the one number it mostly is.

    The commonest reading, not the first. A scan whose dwell is not the same
    at every point is rare -- 32 of the 1012 scans in the two test
    directories -- and in every one of those it is a stray point or two
    rather than a change of plan: scan 83 reads 0.2 s at 44 of its 45 points
    and 0.0 at the first, scan 172 reads 5.0 at 108 of 109. The first point
    is the one that strays, being the one the scaler was still settling into,
    so reading the dwell off the front reports 0.0 for a scan that counted
    for 0.2 s throughout -- which is what 18 of those 1012 would have done.

    No rounding before counting: the repeated readings are bit-identical in
    all 231 scans here that record one, so the only thing rounding could
    merge is strays, which are outvoted anyway.
    """
    seconds = seconds[np.isfinite(seconds)]
    if seconds.size == 0:
        return None
    values, counts = np.unique(seconds, return_counts=True)
    return float(values[np.argmax(counts)])


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
                    info["count_time"] = _dwell(seconds)
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

# How far a point cloud has to stand off its own best-fit line before it is
# worth drawing as a map. Measured after each axis is divided by its own span,
# so a perfectly straight sweep scores 0 and a filled raster of the order of 1.
# The scans this rejects score 1e-16, so the exact cut hardly matters.
SCATTER_MIN_SPREAD = 0.01


def spans_an_area(x, y) -> bool:
    """Whether these points cover a patch of the plane, or just a line.

    Two motors named by a scan do not make it a raster. A scan can sweep both
    together along a path -- two coupled voltages, a pair of angles held in
    some relation -- and then every point lies on one line. There are no
    triangles to be made of those, so matplotlib's tripcolor hands them to
    qhull and qhull fails outright: "singular input data".

    Each axis is divided by its own span before the shape is measured, so a
    fine motor played against a coarse one is not mistaken for a line.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3:
        return False
    x_span, y_span = float(np.ptp(x)), float(np.ptp(y))
    if x_span <= 0.0 or y_span <= 0.0:
        return False                     # one of the two never moved
    points = np.column_stack(
        ((x - x.mean()) / x_span, (y - y.mean()) / y_span)
    )
    largest, smallest = np.linalg.svd(points, compute_uv=False)[:2]
    if largest <= 0.0:
        return False
    return bool(smallest / largest >= SCATTER_MIN_SPREAD)


def read_scatter(
    path: Path, value_field: str, monitor_field: str = ""
) -> dict | None:
    """The points of a two-motor scan where they were actually measured.

    For a plan whose grid is not a lattice. A fly grid steps one axis and
    flies the other, so the flown positions differ from row to row -- and
    so do the row lengths. On the example scans one row holds 11 points
    and the next 12, the rows alternate direction, and 125 points come
    back where the plan declared 121; two others declared 441 and recorded
    372 and 377. There is no rectangle to reshape into. What there is, is
    a set of ``(x, y, value)`` where the motors were, and that is what
    this returns -- nothing is snapped to a lattice or resampled, because
    the irregularity is the measurement rather than noise in it.

    ``None`` when the scan names fewer than two motors, did not record
    them, or moved them together along a line rather than over an area --
    see :func:`spans_an_area`. :func:`read_grid` is the one to try first: a
    scan that really is a raster is better drawn as one, since a lattice
    says which cells were never visited and a cloud of points cannot.
    """
    info = read_scan_info(path)
    if len(info["motors"]) != 2:
        return None
    available = stream_fields(path, "primary")
    slow_options = _grid_axis_candidates(available, info["motors"][0])
    fast_options = _grid_axis_candidates(available, info["motors"][1])
    if not slow_options or not fast_options:
        return None

    wanted = {slow_options[0], fast_options[0], value_field}
    if monitor_field:
        wanted.add(monitor_field)
    try:
        data = read_primary(path, list(wanted))
    except OSError:
        return None
    slow_name, fast_name = slow_options[0], fast_options[0]
    if not {slow_name, fast_name, value_field} <= set(data):
        return None

    values = data[value_field]
    monitor = data.get(monitor_field) if monitor_field else None
    if monitor is not None:
        length = min(len(values), len(monitor))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(
                monitor[:length] != 0, values[:length] / monitor[:length], np.nan
            )
    # The reader does not truncate the columns against each other, and a
    # fly scan is exactly where they come back different lengths.
    rows = min(data[slow_name].size, data[fast_name].size, values.size)
    if rows < 3:            # fewer than a triangle: nothing to draw
        return None
    x = np.asarray(data[fast_name][:rows], dtype=float)
    y = np.asarray(data[slow_name][:rows], dtype=float)
    # Two motors moved together is a curve, not a map, and asking for a map
    # of it fails in qhull rather than anywhere this could explain.
    if not spans_an_area(x, y):
        return None
    return {
        "x_name": fast_name,
        "y_name": slow_name,
        "x": x,
        "y": y,
        "z": np.asarray(values[:rows], dtype=float),
        "points": int(rows),
        "scattered": True,
    }


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

# Where a map's colour clipping starts: on a 1500-cell raster, the eight
# most extreme cells either way. Small enough that a map with nothing wrong
# with it barely changes, which is what a starting point should do; one or
# two cells are what usually wreck a scale.
COLOR_CLIP_PERCENT = 0.5


# A point whose monitor has collapsed to less than this fraction of the
# column's own median was taken with no beam, so none of its channels
# mean anything -- see :func:`lone_dropouts`.
BEAM_GONE_FRACTION = 0.05

def clipped_limits(
    z: np.ndarray, percent: float = COLOR_CLIP_PERCENT
) -> tuple[float | None, float | None, str]:
    """Colour limits leaving out the most extreme *percent* at each end.

    A handful of wild cells -- a spot where the beam ran off the sample, a
    detector glitch -- stretch a map's colour scale over a range nothing else
    occupies, leaving the rest of it one flat shade. Cutting the scale back to
    where the bulk of the data lives brings that structure back; the cells
    outside simply saturate.

    How much to cut is asked for rather than worked out, which is the point.
    Rules that decide for themselves what is an outlier (Tukey's fences on the
    interquartile range, say) read the top of a sharp peak as one, because a
    peaked map legitimately has a long tail above its upper quartile. On the
    sample-position rasters this beamline lines up on -- ``huber_euler_z`` by
    ``huber_euler_y`` against a detector total, where the peak *is* the result
    -- those fences clip most of the peak away: measured over the twelve of
    them, 3x IQR left scan 468 with 7% of its height, 232 with 14%, 228 with
    32%. A number someone typed can be wrong, but it cannot be surprising,
    and the same number gives the same answer on every map it is applied to.

    *percent* is a percentile, so what it costs depends on how the values are
    distributed, not on how extreme they are: 0.5 on a 1500-cell raster is the
    eight cells at each end whatever they hold. Nothing stops it from being
    large -- a map still washed out after a fraction of a percent may well
    need several -- but past a few percent it is no longer only spikes going.

    Returns the two limits and which ends they actually cut, in the wording
    :meth:`Figure.colorbar` wants for its ``extend`` arrows, so a clipped
    scale says on the bar that it is clipped. ``(None, None, "neither")``
    means leave the limits alone.
    """
    finite = np.asarray(z, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return None, None, "neither"
    low, high = (float(v) for v in np.percentile(finite, [percent, 100.0 - percent]))
    if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
        # Flat, or so nearly flat that the percentiles coincide.
        return None, None, "neither"
    return low, high, scale_extend(finite, low, high)


def lone_dropouts(monitor: np.ndarray) -> np.ndarray:
    """Mask the points where the beam was gone for that point alone.

    A dropout is a point whose *monitor* has collapsed to less than
    :data:`BEAM_GONE_FRACTION` of the column's own median. Every channel goes
    with it -- on scan 239's row 367 the monitor reads 5372 against a median
    of 1.85e6, ``LockDC`` 5502 against 713000 -- so the point carries no
    measurement at all, and in the reduction it drags a whole energy bin with
    it: 239's XMuCD spans 0.511 with that row in and 0.0262 with it out.

    Unlike :func:`clipped_limits` above this needs no amount typed into it,
    and the difference is worth stating because the two sit side by side. The
    clip is a judgement -- where a peak stops being signal is not written in
    the data. This is not: a monitor measures the beam arriving, and a beam
    that is arriving at 2% of normal has gone. The two populations are forty
    apart with nothing between them, so the threshold is a reading rather than
    a choice.

    Only *isolated* dropouts are masked -- both neighbours must be live. A run
    of them is not a spike but a dump, and the scan should show it: of the 24
    scans in these directories carrying a dead point, 17 carry nothing but
    single ones and 7 were taken while the beam was down, scan 244 for 265
    consecutive rows. Deleting that would hide a quarter of the run behind a
    curve that looked fine. Length, not depth, is what tells the two apart --
    scan 89's one dead row is 5% of a 20-point scan and still a spike.

    Neighbours are the rows either side *in acquisition order*, so this wants
    one file's column at a time: run two files together and the join would
    stand one scan's last row next to another's first.

    What this deliberately will not catch is a spike the monitor knew nothing
    about. Reduced index 80 of scans 239 and 242 (E = 7.26907) is a 13-sigma
    one-point excursion with the monitor at 0.91 of median and all four raw
    rows agreeing to four decimals -- a monochromator glitch, reproduced in
    every repeat. It is data, and it stays.
    """
    values = np.asarray(monitor, dtype=float)
    gone = np.zeros(values.shape, dtype=bool)
    if values.ndim != 1 or values.size < 3:
        return gone
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return gone
    median = float(np.median(finite))
    if not np.isfinite(median) or median <= 0.0:
        # No scale to judge against: a monitor that is flat zero, or counting
        # down, says nothing about which of its points had beam.
        return gone
    dead = np.isfinite(values) & (values < BEAM_GONE_FRACTION * median)
    padded = np.concatenate(([False], dead, [False]))
    return dead & ~padded[:-2] & ~padded[2:]


def scale_extend(z: np.ndarray, low: float | None, high: float | None) -> str:
    """Which ends of *z* fall outside ``(low, high)``, as colour-bar arrows.

    In the wording :meth:`Figure.colorbar` wants for ``extend``, so a scale
    that does not reach its own data says so on the bar rather than leaving it
    to be noticed. ``None`` for a limit is matplotlib's "fit the data", which
    by definition cuts nothing off at that end.

    Asked separately from the limits because the limits move after they are
    first worked out: :func:`centered_limits` widens a range, and cells that
    saturated against the clipped one need not saturate against the wider. An
    arrow left over from the narrower scale points at data that is not out
    there.
    """
    finite = np.asarray(z, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "neither"
    cuts = (
        low is not None and low > float(finite.min()),
        high is not None and high < float(finite.max()),
    )
    return {
        (True, True): "both", (True, False): "min",
        (False, True): "max", (False, False): "neither",
    }[cuts]


def centered_limits(
    z: np.ndarray, low: float | None, high: float | None
) -> tuple[float | None, float | None]:
    """``(low, high)`` widened until zero sits exactly in the middle.

    A diverging colour map is read by where its middle colour falls, and left
    to the data that lands wherever the range happens to centre. On scan 156's
    XMCD the clipped range is -0.0009 to +0.0036, which puts white at +0.0013:
    a map with signal everywhere, drawn so that most of it reads as none. Zero
    is where white belongs, because zero is what the map is a difference from.

    Widening to whichever end reaches further costs the shorter end some of
    its contrast, and buys a map that can be read without consulting the bar.
    Either limit may be ``None``, meaning the data's own extreme.

    Only ever widens, so no cell that was inside the scale leaves it -- but
    the arrows still have to be recomputed, which is :func:`scale_extend`.
    """
    finite = np.asarray(z, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return low, high
    lo = float(finite.min()) if low is None else low
    hi = float(finite.max()) if high is None else high
    reach = max(abs(lo), abs(hi))
    # Every cell is zero, and there is no scale left to centre on it.
    return (-reach, reach) if reach > 0 else (low, high)


def grid_stats(grid: dict) -> dict[str, float | None]:
    """Max/min of a map and the (x, y) they sit at.

    Takes either shape of map: a lattice from :func:`read_grid`, whose z
    is a rectangle indexed by row and column, or the points as measured
    from :func:`read_scatter`, where x, y and z run alongside each other
    and the position is simply read off at the same index.
    """
    z = grid["z"]
    stats = {"max_z": None, "max_at": None, "min_z": None, "min_at": None}
    if not np.any(np.isfinite(z)):
        return stats
    for key, flat in (("max", int(np.nanargmax(z))),
                      ("min", int(np.nanargmin(z)))):
        if np.ndim(z) == 1:
            stats[f"{key}_z"] = float(z[flat])
            stats[f"{key}_at"] = (float(grid["x"][flat]), float(grid["y"][flat]))
            continue
        row, column = np.unravel_index(flat, z.shape)
        stats[f"{key}_z"] = float(z[row, column])
        stats[f"{key}_at"] = (float(grid["x"][column]), float(grid["y"][row]))
    return stats
