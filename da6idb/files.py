"""Finding the files a scan is made of, and pairing them up.

A scan is a master ``.hdf`` and, usually, a detector ``.h5`` beside it -- but
"beside" has stopped being reliable. Newer masters name the detector file
outright, in a tree of their own, and newer detector files follow no naming
convention at all. Everything needed to get from one to the other lives here,
and nothing here needs Qt.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import h5py

IMAGE_DATA_PATH = "entry/data/data"

ROI_GROUP_PATH = "entry/data/rois"

PVA_STREAMING_PATH = "entry/instrument/bluesky/metadata/pva_streaming"

MASTER_PRIMARY_PATH = "entry/instrument/bluesky/streams/primary"

MASTER_BASELINE_PATH = "entry/instrument/bluesky/streams/baseline"

# Newer masters name the detector's own file outright, instead of leaving it
# to the pva_* naming convention. The value is one "detector: path" per line.
BLUESKY_METADATA_PATH = "entry/instrument/bluesky/metadata"

DETECTOR_PATH_KEYS = ("detectors_file_relative_path", "detectors_file_full_path")

MASTER_PATH_KEY = "master_file_path"

# The detector's ROIs as the master records them, e.g.
# lambda250k_rois_roi1_min_x. The detector's name varies, so it is matched
# rather than assumed.
MASTER_ROI_PATTERN = re.compile(
    r"^(?P<detector>.+)_rois_roi(?P<number>\d+)_(?P<field>min_x|min_y|size_x|size_y)$"
)

# How far up from an image file to look for its master. Newer scans put the
# image in a separate tree (…/lambda/<experiment>/scan.h5) from the master
# (…/<experiment>/scan_master.hdf), so beside-it is no longer enough.
MASTER_SEARCH_DEPTH = 3

# Resolution opens the file to read its metadata, and the browser re-resolves
# the whole selection on every selection change, so remember the answer per
# (path, mtime, size) -- cheap locally, but worth it over a beamline mount.
_RESOLVE_CACHE: dict[tuple, "Path | None"] = {}

def _decode(value) -> str:
    """Return *value* -- an h5py scalar that may be bytes or a 0-d array -- as text."""
    value = np.asarray(value).reshape(-1)
    if value.size == 0:
        return ""
    item = value[0]
    if isinstance(item, bytes):
        return item.decode(errors="replace")
    return str(item)

def _metadata_text(master: Path, key: str) -> str | None:
    try:
        with h5py.File(master, "r") as f:
            group = f.get(BLUESKY_METADATA_PATH)
            if group is None or key not in group:
                return None
            return _decode(group[key][()])
    except OSError:
        return None

def detector_files_from_metadata(master: Path) -> list[str]:
    """The detector file paths the master records, in the order it lists them.

    The value reads ``lambda250k: /some/where/scan_00001.h5``, one detector per
    line; older masters write an empty ``{}`` there, which yields nothing.
    """
    found: list[str] = []
    for key in DETECTOR_PATH_KEYS:
        text = _metadata_text(master, key)
        if not text:
            continue
        for line in text.splitlines():
            _name, sep, value = line.partition(":")
            value = (value if sep else line).strip().strip("'\"")
            if value and value not in ("{}", "{", "}") and value not in found:
                found.append(value)
    return found

def _experiment_base(master: Path) -> Path | None:
    """The directory the master's recorded relative path is measured from.

    ``master_file_path`` is itself relative -- ``test_26_3/test_00001_master.hdf``
    -- so stripping it from where the file actually sits gives the root the
    detector's relative path is written against, wherever the tree has since
    been copied to.
    """
    recorded = _metadata_text(master, MASTER_PATH_KEY)
    if not recorded:
        return None
    tail = Path(recorded.strip())
    actual = Path(master).resolve()
    if tail.is_absolute() or not actual.as_posix().endswith(tail.as_posix()):
        return None
    base = actual
    for _ in tail.parts:
        base = base.parent
    return base

def _pva_name_from_metadata(master: Path) -> str | None:
    """The ``file_name:`` recorded in the master's ``pva_streaming`` YAML block."""
    try:
        with h5py.File(master, "r") as f:
            item = f.get(PVA_STREAMING_PATH)
            if item is None:
                return None
            text = _decode(item[()])
    except OSError:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "file_name":
            name = value.strip().strip("'\"")
            if name:
                return name
    return None

def image_path_for_master(master: Path) -> Path | None:
    """The image file belonging to *master*, or ``None`` if none is next to it.

    Tried in turn: the detector file the master names -- where it points, then
    relative to the experiment root, then by name beside the master -- the
    ``file_name`` in the older ``pva_streaming`` block, and last the
    ``<master stem minus "_master">`` -> ``pva_<stem>.h5`` convention. Newer
    masters name a file that follows no convention at all
    (``test_00001.h5`` beside ``test_00001_master.hdf``), which only the
    recorded name can find.
    """
    master = Path(master)
    candidates: list[Path] = []

    # What the master says the detector wrote, tried where it points and then
    # where the data actually is. A path recorded at the beamline stays valid
    # there and is simply missing anywhere else, which costs one stat call.
    base = _experiment_base(master)
    for declared in detector_files_from_metadata(master):
        declared_path = Path(declared)
        if declared_path.is_absolute():
            candidates.append(declared_path)
        else:
            if base is not None:
                candidates.append(base / declared_path)
            candidates.append(master.parent / declared_path)
        # Failing that, the same file name beside the master: the usual state
        # of a directory copied off the beamline.
        candidates.append(master.parent / declared_path.name)

    recorded = _pva_name_from_metadata(master)
    if recorded:
        candidates.append(master.parent / Path(recorded).name)

    stem = master.stem
    if stem.endswith("_master"):
        stem = stem[: -len("_master")]
    # "" last: newer writers drop the pva_ prefix entirely and name the file
    # after the scan alone, so tise2_2_00353_master.hdf sits beside plain
    # tise2_2_00353.h5. Tried after the prefixed forms, so a directory holding
    # both still prefers the one the old convention named.
    for prefix in ("pva_", "pva", ""):
        for suffix in (".h5", ".hdf5", ".hdf"):
            candidates.append(master.parent / f"{prefix}{stem}{suffix}")

    for candidate in candidates:
        # Guessing at names this broadly can land on a file that merely shares
        # one, so take only a candidate that actually holds an image stack
        # rather than handing the viewer something it cannot open.
        if candidate.is_file() and is_image_file(candidate):
            return candidate
    return None

def master_path_for_image(image: Path) -> Path | None:
    """The master file this image file belongs to.

    Beside the image first. Failing that, a short walk up the tree, trying each
    ancestor and each ancestor's copy of the image's own directory name -- which
    is where the master sits when the detector writes into a parallel tree. The
    name is specific enough (``<scan>_master.hdf``) that the search stays safe;
    callers that already know the pair should pass it rather than rely on this.
    """
    image = Path(image)
    stem = image.stem
    for prefix in ("pva_", "pva"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    names = [f"{stem}_master{suffix}" for suffix in (".hdf", ".hdf5", ".h5")]

    directories = [image.parent]
    here = image.parent.resolve()
    leaf = here.name          # the image's own folder, looked for in each ancestor
    for _ in range(MASTER_SEARCH_DEPTH):
        parent = here.parent
        if parent == here:
            break
        directories.append(parent)
        directories.append(parent / leaf)   # the same folder in a sibling tree
        here = parent

    seen = set()
    for directory in directories:
        if directory in seen:
            continue
        seen.add(directory)
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None

def read_master_rois(master: Path) -> list[dict]:
    """The detector's ROIs as recorded in the master's baseline.

    Newer image files carry no ``entry/data/rois`` of their own, so this is
    the only record of where the detector's ROIs were. The numbers are the
    same ones the older writer put in the image file -- a corner and a size in
    pixels -- so they come back in the same shape.
    """
    found: dict[tuple[str, int], dict] = {}
    try:
        with h5py.File(master, "r") as f:
            group = f.get(MASTER_BASELINE_PATH)
            if group is None:
                return []
            for name in group.keys():
                match = MASTER_ROI_PATTERN.match(name)
                if match is None:
                    continue
                item = group[name]
                source = "value_start" if "value_start" in item else "value"
                if source not in item:
                    continue
                try:
                    value = float(np.asarray(item[source][()]).ravel()[0])
                except (IndexError, TypeError, ValueError):
                    continue
                key = (match["detector"], int(match["number"]))
                found.setdefault(key, {})[match["field"]] = value
    except OSError:
        return []

    rois = []
    for (_detector, number), parts in sorted(found.items(), key=lambda kv: kv[0][1]):
        if not all(k in parts for k in ("min_x", "min_y", "size_x", "size_y")):
            continue          # half a rectangle is no rectangle
        rois.append(
            {
                "name": f"ROI{number}",
                "x": parts["min_x"],
                "y": parts["min_y"],
                "w": max(1.0, parts["size_x"]),
                "h": max(1.0, parts["size_y"]),
                "angle": 0.0,
            }
        )
    return rois

def is_image_file(path: Path) -> bool:
    """Whether *path* is itself an image file (i.e. it holds an image stack)."""
    try:
        with h5py.File(path, "r") as f:
            return isinstance(f.get(IMAGE_DATA_PATH), h5py.Dataset)
    except OSError:
        return False

def resolve_image_file(path: Path) -> Path | None:
    """The image file to open for *path*, which may be a master or an image file.

    Selecting a ``pva*.h5`` directly in the file list is the obvious way to ask
    for it, so that opens itself rather than hunting for a partner.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        key = None
    if key is not None and key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[key]

    resolved = image_path_for_master(path)
    if resolved is None and is_image_file(path):
        resolved = path
    if key is not None:
        _RESOLVE_CACHE[key] = resolved
    return resolved

HDF_SUFFIXES = (".hdf", ".h5", ".hdf5")

def scan_number(path: Path) -> str:
    """The scan number in a file name: ``00008`` from ``tise2_2_00008_master.hdf``.

    The last run of digits wins, so the prefix's own digits (``tise2_2``) are
    ignored and the sibling ``pva_tise2_2_00008.h5`` reports the same number.
    """
    stem = path.stem
    if stem.endswith("_master"):
        stem = stem[: -len("_master")]
    digits = re.findall(r"\d+", stem)
    return digits[-1] if digits else stem

def file_prefix(path: Path) -> str:
    """The part of a file name in front of the scan number: ``tise2_2``.

    Everything from the scan number onwards is dropped, so the files of one
    series all reduce to the same string.
    """
    stem = path.stem
    if stem.endswith("_master"):
        stem = stem[: -len("_master")]
    digits = list(re.finditer(r"\d+", stem))
    if not digits:
        return stem
    return stem[: digits[-1].start()].rstrip("_-. ") or stem

def list_hdf_files(directory: Path) -> list[Path]:
    """Return the HDF5 files directly inside *directory*, sorted by name."""
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in HDF_SUFFIXES
    )
