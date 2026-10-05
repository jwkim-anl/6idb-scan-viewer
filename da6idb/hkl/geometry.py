"""The instrument description that turns a detector pixel into H, K, L.

Angles, circle directions, the UB matrix, the detector's size and where its
centre sits -- gathered from the master, the image file and a folder's own
``hkl_setup.json``, in that order of precedence, with each value remembering
where it came from. The attenuator's transmission comes along too, since a
scan that changed its filter needs correcting before anything else is done
with the intensities.

No Qt. A notebook imports this directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import h5py

from da6idb.files import IMAGE_DATA_PATH, MASTER_PRIMARY_PATH, master_path_for_image
from da6idb.images import FRAME_TIMESTAMP_PATH, nearest_index

HKL_PATH = "entry/data/metadata/HKL"

BASELINE_PATH = "entry/instrument/bluesky/streams/baseline"

# Per-frame EPICS readings the consumer stored beside the frames.
CA_PATH = "entry/data/metadata/ca"

# The attenuator's transmission, as a fraction of the unattenuated beam.
IMAGE_TRANSMISSION = "filter_trans"

MASTER_TRANSMISSION = "filters_transmission"

# A folder of scans shares a diffractometer, so a correction to the geometry
# belongs beside the data rather than being retyped per scan. This file, if it
# is there, is read after both HDF5 files and wins over them.
SETUP_FILENAME = "hkl_setup.json"

# The order the beamline turns them in, as stated by the instrument.
SAMPLE_MOTORS = ("psic_mu", "psic_eta", "psic_chi", "psic_phi")

DETECTOR_MOTORS = ("psic_nu", "psic_delta")

SAMPLE_LABELS = ("Mu", "Eta", "Chi", "Phi")

DETECTOR_LABELS = ("Nu", "Delta")

AXIS_CHOICES = ("x+", "x-", "y+", "y-", "z+", "z-")

def _text(value) -> str:
    """An h5py scalar -- bytes, numpy scalar or 0-d array -- as plain text."""
    flat = np.asarray(value).reshape(-1)
    if flat.size == 0:
        return ""
    item = flat[0]
    return item.decode(errors="replace") if isinstance(item, bytes) else str(item)

def _first_float(value) -> float | None:
    flat = np.asarray(value, dtype=float).reshape(-1)
    return float(flat[0]) if flat.size else None

@dataclass
class HklGeometry:
    """Everything the conversion needs, and where each part came from."""

    image_path: Path
    master_path: Path | None
    frame_shape: tuple[int, int]
    n_frames: int

    energy_kev: float = 20.0
    distance: float = 1000.0
    center_pixel: tuple[float, float] = (0.0, 0.0)
    pixel_direction: tuple[str, str] = ("y-", "z-")
    detector_size: tuple[float, float] = (28.38, 28.38)

    ub: np.ndarray = field(default_factory=lambda: np.eye(3))
    primary_beam: tuple[float, float, float] = (1.0, 0.0, 0.0)
    inplane: tuple[float, float, float] = (0.0, 1.0, 0.0)
    surface: tuple[float, float, float] = (0.0, 0.0, 1.0)

    sample_dirs: list[str] = field(default_factory=lambda: ["z+", "y-", "x+", "y-"])
    detector_dirs: list[str] = field(default_factory=lambda: ["z+", "y-"])
    # (n_circles, n_frames) degrees
    sample_angles: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    detector_angles: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))

    # Attenuator transmission per frame, when the file records it, and whether
    # the intensities should be divided by it.
    transmission: np.ndarray | None = None
    normalize_transmission: bool = False

    sources: dict = field(default_factory=dict)
    frame_epoch: np.ndarray | None = None

    def transmission_varies(self) -> bool:
        """Whether the attenuator moved during the scan."""
        if self.transmission is None:
            return False
        good = self.transmission[np.isfinite(self.transmission) & (self.transmission > 0)]
        return good.size > 1 and not np.allclose(good, good[0])

    def frame_scale(self, frame: int) -> float:
        """What to multiply a frame's counts by to undo the attenuator.

        Frames taken through different filters are not on the same scale, so a
        map built from them has steps in it wherever the attenuator moved.
        Dividing by the transmission puts every frame on the unattenuated one.
        """
        if not self.normalize_transmission or self.transmission is None:
            return 1.0
        if not 0 <= frame < self.transmission.size:
            return 1.0
        value = float(self.transmission[frame])
        return 1.0 / value if np.isfinite(value) and value > 0 else 1.0

    def energy_ev(self) -> float:
        return float(self.energy_kev) * 1000.0

    def pixel_widths(self) -> tuple[float, float]:
        return (
            self.detector_size[0] / self.frame_shape[0],
            self.detector_size[1] / self.frame_shape[1],
        )

def _read_image_geometry(image: Path, geometry: HklGeometry):
    """Fill from the image file's own HKL metadata (the complete description)."""
    with h5py.File(image, "r") as f:
        hkl = f.get(HKL_PATH)
        if hkl is None:
            return
        mark = geometry.sources

        for index in range(1, 5):
            group = hkl.get(f"SAMPLE_CIRCLE_AXIS_{index}")
            if group is not None and "DIRECTION_AXIS" in group:
                geometry.sample_dirs[index - 1] = _text(group["DIRECTION_AXIS"][()])
                mark["sample_dirs"] = "image"
        for index in range(1, 3):
            group = hkl.get(f"DETECTOR_CIRCLE_AXIS_{index}")
            if group is not None and "DIRECTION_AXIS" in group:
                geometry.detector_dirs[index - 1] = _text(group["DIRECTION_AXIS"][()])
                mark["detector_dirs"] = "image"

        # Spelled both ways in the wild -- older writers dropped the T.
        vectors = {
            "primary_beam": ("PRIMARY_BEAM_DIRECTION", "PRIMARY_BEAM_DIRECITON"),
            "inplane": ("INPLANE_REFERENCE_DIRECTION", "INPLANE_REFERENCE_DIRECITON"),
            "surface": (
                "SAMPLE_SURFACE_NORMAL_DIRECTION",
                "SAMPLE_SURFACE_NORMAL_DIRECITON",
            ),
        }
        for attribute, names in vectors.items():
            for name in names:
                group = hkl.get(name)
                if group is None:
                    continue
                try:
                    values = [
                        _first_float(group[f"AXIS_NUMBER_{i}"][()]) for i in (1, 2, 3)
                    ]
                except KeyError:
                    continue
                if all(v is not None for v in values):
                    setattr(geometry, attribute, tuple(values))
                    mark[attribute] = "image"
                break

        detector = hkl.get("DETECTOR_SETUP")
        if detector is not None:
            if "SIZE" in detector:
                size = np.asarray(detector["SIZE"][()], dtype=float).ravel()
                if size.size >= 2:
                    geometry.detector_size = (float(size[0]), float(size[1]))
                    mark["detector_size"] = "image"
            if "CENTER_CHANNEL_PIXEL" in detector:
                center = np.asarray(
                    detector["CENTER_CHANNEL_PIXEL"][()], dtype=float
                ).ravel()
                if center.size >= 2:
                    geometry.center_pixel = (float(center[0]), float(center[1]))
                    mark["center_pixel"] = "image"
            if "DISTANCE" in detector:
                distance = _first_float(detector["DISTANCE"][()])
                if distance:
                    geometry.distance = distance
                    mark["distance"] = "image"
            directions = [
                _text(detector[key][()])
                for key in ("PIXEL_DIRECTION_1", "PIXEL_DIRECTION_2")
                if key in detector
            ]
            if len(directions) == 2:
                geometry.pixel_direction = (directions[0], directions[1])
                mark["pixel_direction"] = "image"

        spec = hkl.get("SPEC")
        if spec is not None:
            if "UB_MATRIX_VALUE" in spec:
                values = np.asarray(spec["UB_MATRIX_VALUE"][()], dtype=float).ravel()
                if values.size >= 9:
                    geometry.ub = values[:9].reshape(3, 3)
                    mark["ub"] = "image"
            if "ENERGY_VALUE" in spec:
                energy = _first_float(spec["ENERGY_VALUE"][()])
                if energy:
                    geometry.energy_kev = energy
                    mark["energy_kev"] = "image"

        # Per-frame angles, as recorded alongside the frames themselves.
        sample, detector_pos = [], []
        for index in range(1, 5):
            group = hkl.get(f"SAMPLE_CIRCLE_AXIS_{index}")
            if group is not None and "POSITION" in group:
                sample.append(np.asarray(group["POSITION"][()], dtype=float).ravel())
        for index in range(1, 3):
            group = hkl.get(f"DETECTOR_CIRCLE_AXIS_{index}")
            if group is not None and "POSITION" in group:
                detector_pos.append(
                    np.asarray(group["POSITION"][()], dtype=float).ravel()
                )
        if len(sample) == 4 and len(detector_pos) == 2:
            geometry.sample_angles = np.vstack(sample)
            geometry.detector_angles = np.vstack(detector_pos)
            mark["angles"] = "image"

def _read_master_geometry(
    master: Path, frame_epoch, geometry: HklGeometry, take_angles: bool
):
    """Override with the master's own record -- the beamline's source of truth."""
    with h5py.File(master, "r") as f:
        mark = geometry.sources
        baseline = f.get(BASELINE_PATH)
        if baseline is not None:

            def start(name):
                group = baseline.get(name)
                if group is None or "value_start" not in group:
                    return None
                return group["value_start"][()]

            center = start("hkl_pvs_center_pixel")
            if center is not None:
                flat = np.asarray(center, dtype=float).ravel()
                if flat.size >= 2:
                    geometry.center_pixel = (float(flat[0]), float(flat[1]))
                    mark["center_pixel"] = "master"
            distance = start("hkl_pvs_distance")
            if distance is not None and _first_float(distance):
                geometry.distance = _first_float(distance)
                mark["distance"] = "master"
            one, two = start("hkl_pvs_pixel_direction1"), start("hkl_pvs_pixel_direction2")
            if one is not None and two is not None:
                geometry.pixel_direction = (_text(one), _text(two))
                mark["pixel_direction"] = "master"
            ub = start("hkl_pvs_ub_matrix")
            if ub is not None:
                flat = np.asarray(ub, dtype=float).ravel()
                if flat.size >= 9:
                    geometry.ub = flat[:9].reshape(3, 3)
                    mark["ub"] = "master"
            energy = start("energy")
            if energy is not None and _first_float(energy):
                geometry.energy_kev = _first_float(energy)
                mark["energy_kev"] = "master"

        if not take_angles:
            return
        primary = f.get(MASTER_PRIMARY_PATH)
        if primary is None or frame_epoch is None:
            return
        angles, clock = {}, None
        for name in (*SAMPLE_MOTORS, *DETECTOR_MOTORS):
            group = primary.get(name)
            if group is None or "value" not in group:
                return  # an incomplete set is no use; keep the image file's
            angles[name] = np.asarray(group["value"][()], dtype=float).ravel()
            if clock is None and "EPOCH" in group:
                epoch = np.asarray(group["EPOCH"][()], dtype=float).ravel()
                # The clock has to actually advance. A motor that did not move
                # keeps the timestamp of whenever it was last read -- days
                # earlier, and identical for every point -- which looks like a
                # valid time but collapses every frame onto one scan point.
                if (
                    epoch.size > 1
                    and epoch[0] > 1e9              # later than 2001, so a real time
                    and epoch[-1] > epoch[0]        # and it moves
                    and np.all(np.diff(epoch) >= 0)
                ):
                    clock = epoch
        if clock is None or any(v.size != clock.size for v in angles.values()):
            return
        index = nearest_index(np.asarray(frame_epoch, dtype=float), clock)
        geometry.sample_angles = np.vstack([angles[n][index] for n in SAMPLE_MOTORS])
        geometry.detector_angles = np.vstack([angles[n][index] for n in DETECTOR_MOTORS])
        mark["angles"] = "master"

def _as_floats(value, count):
    values = [float(v) for v in np.asarray(value, dtype=float).ravel()]
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {len(values)}")
    return tuple(values)

def _as_strings(value, count):
    values = [str(v) for v in value]
    if len(values) != count:
        raise ValueError(f"expected {count} entries, got {len(values)}")
    return values

# What a setup file may override, and how each value is read back. Angles are
# deliberately not here: they are per-frame and belong to the scan, not to the
# instrument.
SETUP_FIELDS = {
    "energy_kev": lambda v: float(v),
    "distance": lambda v: float(v),
    "center_pixel": lambda v: _as_floats(v, 2),
    "detector_size": lambda v: _as_floats(v, 2),
    "pixel_direction": lambda v: tuple(_as_strings(v, 2)),
    "ub": lambda v: np.asarray(v, dtype=float).reshape(3, 3),
    "sample_dirs": lambda v: _as_strings(v, 4),
    "detector_dirs": lambda v: _as_strings(v, 2),
    "primary_beam": lambda v: _as_floats(v, 3),
    "inplane": lambda v: _as_floats(v, 3),
    "surface": lambda v: _as_floats(v, 3),
}

def setup_path(folder) -> Path:
    """Where the per-folder setup lives."""
    return Path(folder) / SETUP_FILENAME

def read_setup(folder):
    """The overrides for *folder*, as ``{field: raw value}``. Empty if none.

    Raises ValueError if the file is there but unreadable, so a typo is
    reported rather than silently ignored.
    """
    path = setup_path(folder)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name}: expected an object of field: value")
    return {k: v for k, v in loaded.items() if k in SETUP_FIELDS}

def apply_setup(geometry: HklGeometry, folder=None):
    """Lay the folder's setup over *geometry*. Returns the fields it changed."""
    folder = Path(folder) if folder is not None else geometry.image_path.parent
    values = read_setup(folder)
    applied = []
    for name, convert in SETUP_FIELDS.items():
        if name not in values:
            continue
        try:
            setattr(geometry, name, convert(values[name]))
        except Exception as exc:
            raise ValueError(f"{SETUP_FILENAME}: {name}: {exc}") from exc
        geometry.sources[name] = "setup"
        applied.append(name)
    return applied

def write_setup(folder, values) -> Path:
    """Write *values* as the folder's setup, or remove it when empty."""
    path = setup_path(folder)
    if not values:
        path.unlink(missing_ok=True)
        return path
    payload = {}
    for name in SETUP_FIELDS:                 # a stable, readable order
        if name in values:
            value = values[name]
            payload[name] = (
                np.asarray(value).tolist() if isinstance(value, np.ndarray)
                else list(value) if isinstance(value, (tuple, list))
                else value
            )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

def setup_differences(geometry: HklGeometry):
    """What in *geometry* differs from what its files alone say.

    This is what a Save writes: only the values actually changed, so a setup
    file does not quietly pin everything else as well.
    """
    plain = read_geometry(
        geometry.image_path, geometry.master_path, use_setup=False
    )
    changed = {}
    for name in SETUP_FIELDS:
        mine, theirs = getattr(geometry, name), getattr(plain, name)
        if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
            same = np.allclose(np.asarray(mine, float), np.asarray(theirs, float))
        elif isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
            same = bool(np.isclose(mine, theirs))
        elif all(isinstance(v, (int, float)) for v in np.atleast_1d(mine).tolist()):
            same = np.allclose(np.asarray(mine, float), np.asarray(theirs, float))
        else:
            same = list(mine) == list(theirs)
        if not same:
            changed[name] = mine
    return changed

def read_transmission(image: Path, master: Path | None, n_frames: int,
                      frame_epoch=None):
    """Attenuator transmission for each frame, or None if nothing records it.

    The image file stores one reading per frame, which is what a frame was
    actually taken through; the master samples at scan points and misses the
    transients the free-running detector catches, so it is only a fallback.
    """
    try:
        with h5py.File(image, "r") as f:
            ca = f.get(CA_PATH)
            if ca is not None and IMAGE_TRANSMISSION in ca:
                values = np.asarray(ca[IMAGE_TRANSMISSION], dtype=float).ravel()
                if values.size:
                    return _fit_frames(values, n_frames), "image"
    except Exception:
        pass
    if master is None:
        return None, "missing"
    try:
        with h5py.File(master, "r") as f:
            primary = f.get(MASTER_PRIMARY_PATH)
            if primary is None or MASTER_TRANSMISSION not in primary:
                return None, "missing"
            values = np.asarray(primary[MASTER_TRANSMISSION]["value"], dtype=float).ravel()
            if not values.size:
                return None, "missing"
            if frame_epoch is not None and "time" in primary[MASTER_TRANSMISSION]:
                clock = np.asarray(primary[MASTER_TRANSMISSION]["time"], dtype=float)
                if clock.size == values.size:
                    index = nearest_index(np.asarray(frame_epoch, dtype=float), clock)
                    return values[index], "master"
            return _fit_frames(values, n_frames), "master"
    except Exception:
        return None, "missing"

def _fit_frames(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Size a per-frame column to the frames, padding with the last reading."""
    if values.size == n_frames:
        return values
    if values.size > n_frames:
        return values[:n_frames]
    return np.concatenate([values, np.repeat(values[-1], n_frames - values.size)])

def read_geometry(
    image: Path, master: Path | None = None, angles_from_master: bool = False,
    use_setup: bool = True,
) -> HklGeometry:
    """The conversion parameters for *image*, master first where it has them.

    The angles are the exception. The image file records a ``POSITION`` with each
    frame, while the master records one per scan point, on a read-back clock that
    lags the exposure -- matching those onto the frames shifted every frame of a
    rocking curve by one whole step, which would move the entire map. So the
    per-frame values win when they exist; *angles_from_master* forces the
    resampled ones, and the window exposes the choice.
    """
    image = Path(image)
    with h5py.File(image, "r") as f:
        data = f.get(IMAGE_DATA_PATH)
        if data is None:
            raise RuntimeError(f"{image.name} has no image stack at '{IMAGE_DATA_PATH}'.")
        shape = data.shape
        stamps = f.get(FRAME_TIMESTAMP_PATH)
        frame_epoch = (
            None if stamps is None else np.asarray(stamps[()], dtype=float).ravel()
        )

    geometry = HklGeometry(
        image_path=image,
        master_path=master if master is not None else master_path_for_image(image),
        frame_shape=(int(shape[1]), int(shape[2])),
        n_frames=int(shape[0]),
    )
    # Sized for the stack from the outset: the dataclass default is a single
    # column, and a file carrying no angles anywhere would otherwise index past
    # the end of it on frame 1 rather than say what was missing.
    geometry.sample_angles = np.zeros((4, geometry.n_frames))
    geometry.detector_angles = np.zeros((2, geometry.n_frames))

    _read_image_geometry(image, geometry)
    has_frame_angles = geometry.sources.get("angles") == "image"
    if geometry.master_path is not None:
        try:
            _read_master_geometry(
                geometry.master_path,
                frame_epoch,
                geometry,
                take_angles=angles_from_master or not has_frame_angles,
            )
        except OSError:
            pass
    geometry.frame_epoch = frame_epoch

    transmission, source = read_transmission(
        image, geometry.master_path, geometry.n_frames, frame_epoch
    )
    geometry.transmission = transmission
    geometry.sources["transmission"] = source
    # A scan that changed its attenuator has frames on different scales, and a
    # map built from them steps wherever the filter moved. Correct it by
    # default; a scan with one filter throughout is unaffected either way.
    geometry.normalize_transmission = geometry.transmission_varies()

    if use_setup:
        # Last word: a correction the folder carries beats both files.
        apply_setup(geometry)

    if "angles" not in geometry.sources:
        # Nothing to convert against. Kept openable so the geometry can still
        # be inspected and typed in, but the map itself means nothing.
        geometry.sources["angles"] = "missing"
    return geometry
