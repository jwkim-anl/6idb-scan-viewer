"""Turn a ``pva*.h5`` detector stack into a 3D map of reciprocal space.

Every pixel of every frame is a point in (H, K, L), fixed by the diffractometer
angles at that frame, the detector geometry and the sample's UB matrix. The
conversion is ``xrayutilities``' area-detector Q conversion::

    qconv = xu.experiment.QConversion(sample_dirs, detector_dirs, primary_beam)
    hxrd  = xu.HXRD(inplane, surface, en=energy_eV, qconv=qconv)
    hxrd.Ang2Q.init_area(pixel_dir1, pixel_dir2, cch1=, cch2=, Nch1=, Nch2=,
                         pwidth1=, pwidth2=, distance=, roi=)
    H, K, L = hxrd.Ang2Q.area(mu, eta, chi, phi, nu, delta, UB=ub)

Where the parameters come from
------------------------------
The ``*_master.hdf`` beside the image file is the primary source, as the
beamline records it there:

- its **baseline** holds ``hkl_pvs_center_pixel``, ``hkl_pvs_distance``,
  ``hkl_pvs_pixel_direction1``/``2``, ``hkl_pvs_ub_matrix`` and ``energy``;
- its **primary** holds the per-frame angles ``psic_mu``, ``psic_eta``,
  ``psic_chi``, ``psic_phi`` (sample, in that order) and ``psic_nu``,
  ``psic_delta`` (detector).

The circle *directions*, the beam/reference vectors and the detector's physical
size are not in the master at all, so those come from the image file's own
``entry/data/metadata/HKL`` group, which also serves as the fallback for
anything the master is missing. Every value is editable afterwards.

The detector free-runs, so the two files rarely have the same number of points
(601 frames against 600 scan points is typical); the angles are matched onto the
frames by nearest timestamp, the same way ``image_viewer`` matches scan channels.

``pyvista`` and ``xrayutilities`` are imported at module level, so this module is
imported lazily -- the scan and image viewers must keep working without them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import xrayutilities as xu
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from volume_style import AXIS_COLOR
from volume_style import AXIS_NAMES
from volume_style import AXIS_NUMBER_FORMAT
from volume_style import BACKGROUND
from volume_style import COLORMAPS
from volume_style import SCALAR_BAR
from volume_style import SLICE_COLOR
from volume_style import spin as _spin

from image_viewer import MASTER_PRIMARY_PATH
from image_viewer import master_path_for_image
from image_viewer import nearest_index

IMAGE_DATA_PATH = "entry/data/data"
FRAME_TIMESTAMP_PATH = "entry/data/frame_timestamp"
HKL_PATH = "entry/data/metadata/HKL"
BASELINE_PATH = "entry/instrument/bluesky/streams/baseline"
# Per-frame EPICS readings the consumer stored beside the frames.
CA_PATH = "entry/data/metadata/ca"
# The attenuator's transmission, as a fraction of the unattenuated beam.
IMAGE_TRANSMISSION = "filter_trans"
MASTER_TRANSMISSION = "filters_transmission"

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


# -- reading the geometry ----------------------------------------------------


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
    image: Path, master: Path | None = None, angles_from_master: bool = False
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

    if "angles" not in geometry.sources:
        # Nothing to convert against. Kept openable so the geometry can still
        # be inspected and typed in, but the map itself means nothing.
        geometry.sources["angles"] = "missing"
    return geometry


# -- the conversion ----------------------------------------------------------


def build_converter(geometry: HklGeometry):
    """The xrayutilities area-detector converter for this geometry."""
    qconv = xu.experiment.QConversion(
        list(geometry.sample_dirs),
        list(geometry.detector_dirs),
        list(geometry.primary_beam),
    )
    hxrd = xu.HXRD(
        list(geometry.inplane),
        list(geometry.surface),
        en=geometry.energy_ev(),
        qconv=qconv,
    )
    rows, columns = geometry.frame_shape
    width1, width2 = geometry.pixel_widths()
    hxrd.Ang2Q.init_area(
        geometry.pixel_direction[0],
        geometry.pixel_direction[1],
        cch1=geometry.center_pixel[0],
        cch2=geometry.center_pixel[1],
        Nch1=rows,
        Nch2=columns,
        pwidth1=width1,
        pwidth2=width2,
        distance=geometry.distance,
        roi=[0, rows, 0, columns],
    )
    return hxrd


def frame_hkl(geometry: HklGeometry, hxrd, frame: int):
    """(H, K, L) for every pixel of one frame, each shaped like the detector."""
    angles = [
        *[float(row[frame]) for row in geometry.sample_angles],
        *[float(row[frame]) for row in geometry.detector_angles],
    ]
    return hxrd.Ang2Q.area(*angles, UB=geometry.ub)


@dataclass
class HklCloud:
    """The reduced point cloud handed to the plotter."""

    points: np.ndarray            # (n, 3) H K L
    intensity: np.ndarray         # (n,)
    frames_used: int
    total_pixels: int
    kept_before_cap: int

    def bounds(self):
        if self.points.size == 0:
            return None
        low = self.points.min(axis=0)
        high = self.points.max(axis=0)
        return tuple(zip(low.tolist(), high.tolist()))


def convert(
    geometry: HklGeometry,
    first: int = 0,
    last: int | None = None,
    step: int = 1,
    threshold: float = 0.0,
    max_points: int = 2_000_000,
    progress=None,
) -> HklCloud:
    """Map the selected frames into (H, K, L), keeping the interesting pixels.

    Every pixel of every frame is a point, which for a long scan is far more
    than can be held or drawn -- 601 frames of 516x516 is 160 million. So each
    frame is reduced as it is converted: pixels at or below *threshold* are
    dropped, and if what survives still exceeds *max_points* the brightest are
    kept. Reducing per frame keeps the full set from ever being in memory.
    """
    last = geometry.n_frames - 1 if last is None else min(last, geometry.n_frames - 1)
    first = max(0, min(first, last))
    frames = range(first, last + 1, max(1, step))

    hxrd = build_converter(geometry)
    rows, columns = geometry.frame_shape
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    total_pixels = 0
    kept = 0

    with h5py.File(geometry.image_path, "r") as f:
        stack = f[IMAGE_DATA_PATH]
        for count, frame in enumerate(frames):
            image = np.asarray(stack[frame], dtype=np.float32).reshape(rows, columns)
            total_pixels += image.size
            # Threshold on the counts as recorded: it is a statement about the
            # detector, so it must not move when the attenuator does.
            keep = image > threshold
            if not np.any(keep):
                continue
            scale = geometry.frame_scale(frame)
            h, k, l = frame_hkl(geometry, hxrd, frame)
            points = np.column_stack(
                (
                    np.asarray(h, dtype=np.float32)[keep],
                    np.asarray(k, dtype=np.float32)[keep],
                    np.asarray(l, dtype=np.float32)[keep],
                )
            )
            values = image[keep]
            if scale != 1.0:
                values = values * np.float32(scale)
            chunks.append((points, values))
            kept += points.shape[0]
            if progress is not None and not progress(count + 1, len(frames)):
                break

    if not chunks:
        empty = np.zeros((0, 3), dtype=np.float32)
        return HklCloud(empty, np.zeros(0, dtype=np.float32), len(frames), total_pixels, 0)

    points = np.concatenate([c[0] for c in chunks])
    intensity = np.concatenate([c[1] for c in chunks])
    if max_points and points.shape[0] > max_points:
        # argpartition beats a full sort: the order among the kept points is
        # irrelevant, only that they are the brightest ones.
        brightest = np.argpartition(intensity, -max_points)[-max_points:]
        points, intensity = points[brightest], intensity[brightest]
    return HklCloud(points, intensity, len(frames), total_pixels, kept)


# -- shared widget helpers ---------------------------------------------------


def _axis_combo(value):
    combo = QtWidgets.QComboBox()
    combo.addItems(AXIS_CHOICES)
    if value in AXIS_CHOICES:
        combo.setCurrentText(value)
    combo.setFixedWidth(70)
    return combo


# -- the conversion setup, in its own window ---------------------------------


class ConversionSetupDialog(QtWidgets.QDialog):
    """Every parameter that decides where a pixel lands in reciprocal space.

    Kept apart from the main window: this is the instrument description, set
    once and rarely touched, while the controls beside the plot are about
    looking at the result.
    """

    applied = QtCore.pyqtSignal()

    def __init__(self, geometry: HklGeometry, parent=None):
        super().__init__(parent)
        self.geometry_data = geometry
        self.setWindowTitle("HKL conversion setup")
        self.resize(420, 760)

        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)
        self._scroll = scroll
        scroll.setWidget(self._build_panel())

        buttons = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply and recompute")
        apply_button.clicked.connect(self._apply)
        reset_button = QtWidgets.QPushButton("Reset to file")
        reset_button.clicked.connect(self._reset)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.hide)
        buttons.addWidget(apply_button)
        buttons.addWidget(reset_button)
        buttons.addWidget(close_button)
        outer.addLayout(buttons)

    # -- building ---------------------------------------------------------

    def _group(self, title, grid=False):
        box = QtWidgets.QGroupBox(title)
        inner = QtWidgets.QGridLayout(box) if grid else QtWidgets.QFormLayout(box)
        inner.setContentsMargins(6, 6, 6, 6)
        return box, inner

    def _tagged(self, label: str, key: str) -> str:
        """Label carrying where the value came from, so edits stay traceable."""
        source = self.geometry_data.sources.get(key)
        return f"{label}  [{source}]" if source else label

    def _build_panel(self) -> QtWidgets.QWidget:
        geometry = self.geometry_data
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        origin = QtWidgets.QLabel(
            f"image: {geometry.image_path.name}\nmaster: "
            f"{geometry.master_path.name if geometry.master_path else '(none found)'}"
        )
        origin.setWordWrap(True)
        layout.addWidget(origin)

        box, form = self._group("Beam and detector")
        self.energy_spin = _spin(geometry.energy_kev, 0.1, 1000.0, 6, 0.1)
        form.addRow(self._tagged("Energy [keV]", "energy_kev"), self.energy_spin)
        self.distance_spin = _spin(geometry.distance, 0.1, 1e6, 3, 1.0)
        form.addRow(self._tagged("Distance [mm]", "distance"), self.distance_spin)
        self.cch1_spin = _spin(geometry.center_pixel[0], -1e5, 1e5, 2)
        self.cch2_spin = _spin(geometry.center_pixel[1], -1e5, 1e5, 2)
        form.addRow(self._tagged("Centre pixel 1", "center_pixel"), self.cch1_spin)
        form.addRow("Centre pixel 2", self.cch2_spin)
        self.size1_spin = _spin(geometry.detector_size[0], 0.001, 1e5, 4)
        self.size2_spin = _spin(geometry.detector_size[1], 0.001, 1e5, 4)
        form.addRow(self._tagged("Detector size 1 [mm]", "detector_size"), self.size1_spin)
        form.addRow("Detector size 2 [mm]", self.size2_spin)
        self.pdir1_combo = _axis_combo(geometry.pixel_direction[0])
        self.pdir2_combo = _axis_combo(geometry.pixel_direction[1])
        form.addRow(self._tagged("Pixel direction 1", "pixel_direction"), self.pdir1_combo)
        form.addRow("Pixel direction 2", self.pdir2_combo)
        layout.addWidget(box)

        box, grid = self._group("UB matrix", grid=True)
        self.ub_spins = []
        for row in range(3):
            for column in range(3):
                spin = _spin(float(geometry.ub[row, column]), -1e4, 1e4, 6, 0.01, 92)
                grid.addWidget(spin, row, column)
                self.ub_spins.append(spin)
        layout.addWidget(box)

        box, form = self._group("Circle directions")
        self.sample_combos, self.detector_combos = [], []
        for index, label in enumerate(SAMPLE_LABELS):
            combo = _axis_combo(geometry.sample_dirs[index])
            self.sample_combos.append(combo)
            name = f"Sample {index + 1} ({label})"
            form.addRow(self._tagged(name, "sample_dirs") if index == 0 else name, combo)
        for index, label in enumerate(DETECTOR_LABELS):
            combo = _axis_combo(geometry.detector_dirs[index])
            self.detector_combos.append(combo)
            name = f"Detector {index + 1} ({label})"
            form.addRow(self._tagged(name, "detector_dirs") if index == 0 else name, combo)
        layout.addWidget(box)

        box, grid = self._group("Reference vectors", grid=True)
        self.vector_spins = {}
        for row, (attribute, label) in enumerate(
            (("primary_beam", "Beam"), ("inplane", "In-plane"), ("surface", "Surface"))
        ):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            spins = []
            for column in range(3):
                spin = _spin(float(getattr(geometry, attribute)[column]), -1e4, 1e4, 4, 1.0, 74)
                grid.addWidget(spin, row, column + 1)
                spins.append(spin)
            self.vector_spins[attribute] = spins
        layout.addWidget(box)

        box, form = self._group("Frames and reduction")
        self.first_spin = QtWidgets.QSpinBox()
        self.first_spin.setRange(0, max(geometry.n_frames - 1, 0))
        self.last_spin = QtWidgets.QSpinBox()
        self.last_spin.setRange(0, max(geometry.n_frames - 1, 0))
        self.last_spin.setValue(geometry.n_frames - 1)
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, max(geometry.n_frames, 1))
        form.addRow("First frame", self.first_spin)
        form.addRow("Last frame", self.last_spin)
        form.addRow("Frame step", self.step_spin)
        self.threshold_spin = _spin(0.0, -1e12, 1e12, 3, 10.0)
        self.threshold_spin.setToolTip("Pixels at or below this are never converted.")
        form.addRow("Intensity threshold", self.threshold_spin)
        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(1000, 50_000_000)
        self.max_points_spin.setSingleStep(100_000)
        self.max_points_spin.setValue(2_000_000)
        self.max_points_spin.setToolTip("If more survive the threshold, the brightest are kept.")
        form.addRow("Max converted points", self.max_points_spin)
        self.master_angles_check = QtWidgets.QCheckBox("Angles from master")
        self.master_angles_check.setToolTip(
            "Off: the per-frame angles the image file stores with each frame.\n"
            "On: the master's scan-point angles, matched onto frames by time --\n"
            "a read-back clock that can sit a whole scan step behind."
        )
        form.addRow(self.master_angles_check)

        self.transmission_check = QtWidgets.QCheckBox("Divide by filter transmission")
        self.transmission_check.setToolTip(
            "A scan that changes its attenuator records frames on different\n"
            "scales, and the map steps wherever the filter moved. Dividing each\n"
            "frame by its transmission puts them all on the unattenuated beam.\n"
            "Ticked by default when the transmission actually varies."
        )
        self.transmission_check.setChecked(bool(geometry.normalize_transmission))
        self.transmission_check.toggled.connect(self._transmission_toggled)
        form.addRow(self.transmission_check)
        self.transmission_label = QtWidgets.QLabel("")
        self.transmission_label.setWordWrap(True)
        self.transmission_label.setStyleSheet("color: gray;")
        form.addRow("", self.transmission_label)
        self._describe_transmission()
        layout.addWidget(box)
        layout.addStretch(1)
        return panel

    # -- applying ---------------------------------------------------------

    @staticmethod
    def _typed(spin, current):
        """The widget's value, unless it is only a rounded view of *current*.

        A spin box shows a fixed number of decimals, so reading every box back
        into the geometry would quantise whatever the files hold -- the UB comes
        out of the master with more digits than six. Taking the widget's number
        only when it differs from the rounded form of what is already there
        leaves untouched values at full precision, and still picks up an edit
        of a single digit.
        """
        current = float(current)
        return current if spin.value() == round(current, spin.decimals()) else spin.value()

    def apply_to_geometry(self):
        geometry = self.geometry_data
        take = self._typed
        geometry.energy_kev = take(self.energy_spin, geometry.energy_kev)
        geometry.distance = take(self.distance_spin, geometry.distance)
        geometry.center_pixel = (
            take(self.cch1_spin, geometry.center_pixel[0]),
            take(self.cch2_spin, geometry.center_pixel[1]),
        )
        geometry.detector_size = (
            take(self.size1_spin, geometry.detector_size[0]),
            take(self.size2_spin, geometry.detector_size[1]),
        )
        geometry.pixel_direction = (
            self.pdir1_combo.currentText(),
            self.pdir2_combo.currentText(),
        )
        geometry.ub = np.array(
            [take(spin, geometry.ub.ravel()[index])
             for index, spin in enumerate(self.ub_spins)]
        ).reshape(3, 3)
        geometry.sample_dirs = [c.currentText() for c in self.sample_combos]
        geometry.detector_dirs = [c.currentText() for c in self.detector_combos]
        for attribute, spins in self.vector_spins.items():
            current = getattr(geometry, attribute)
            setattr(geometry, attribute,
                    tuple(take(spin, current[index]) for index, spin in enumerate(spins)))

    def conversion_arguments(self) -> dict:
        return {
            "first": self.first_spin.value(),
            "last": self.last_spin.value(),
            "step": self.step_spin.value(),
            "threshold": self.threshold_spin.value(),
            "max_points": self.max_points_spin.value(),
        }

    def _describe_transmission(self):
        """Say what the file records, so the checkbox is not a mystery."""
        values = self.geometry_data.transmission
        if values is None or not values.size:
            self.transmission_check.setEnabled(False)
            self.transmission_label.setText("no transmission recorded in either file")
            return
        distinct = np.unique(values[np.isfinite(values) & (values > 0)])
        where = self.geometry_data.sources.get("transmission", "?")
        if distinct.size <= 1:
            only = distinct[0] if distinct.size else float("nan")
            self.transmission_label.setText(
                f"constant at {only:.4g} ({where}) — the attenuator did not move,"
                " so this changes nothing but the scale"
            )
        else:
            self.transmission_label.setText(
                f"{distinct.size} values from {distinct.min():.3g} to {distinct.max():.3g}"
                f" ({where}) — a factor of {distinct.max() / distinct.min():,.0f}"
                " between the weakest and strongest filter"
            )

    def _transmission_toggled(self, checked):
        self.geometry_data.normalize_transmission = bool(checked)

    def wants_master_angles(self) -> bool:
        return self.master_angles_check.isChecked()

    def _apply(self):
        self.applied.emit()

    def _reset(self):
        """Back to the values read from the files."""
        fresh = read_geometry(self.geometry_data.image_path, self.geometry_data.master_path)
        for attribute in (
            "energy_kev", "distance", "center_pixel", "detector_size",
            "pixel_direction", "ub", "sample_dirs", "detector_dirs",
            "primary_beam", "inplane", "surface", "sample_angles",
            "detector_angles", "sources",
        ):
            setattr(self.geometry_data, attribute, getattr(fresh, attribute))
        self._scroll.setWidget(self._build_panel())
        self.applied.emit()


# -- the 3D window -----------------------------------------------------------

class HklWindow(QtWidgets.QMainWindow):
    """A 3D map of reciprocal space.

    The panel beside the plot is about *looking* -- colour, what is drawn, which
    intensities and which corner of reciprocal space. The instrument description
    that decides where each pixel lands lives in its own setup window.
    """

    def __init__(self, image_path, master_path=None, parent=None):
        super().__init__(parent)
        self.geometry_data = read_geometry(Path(image_path), master_path)
        self.cloud: HklCloud | None = None
        self.grid = None                 # the loaded .vti, when one has been made
        self.grid_path: Path | None = None
        self.grid_range: tuple | None = None
        self._updating = False
        # Only a fresh conversion re-frames the view; toggling a checkbox must
        # not throw away the angle the user has set up.
        self._needs_camera_reset = True

        self.setWindowTitle(f"HKL 3D — {self.geometry_data.image_path.name}")
        self.resize(1450, 900)
        self.setup_dialog = ConversionSetupDialog(self.geometry_data, self)
        self.setup_dialog.applied.connect(self.recompute)
        self._build_ui()
        self.recompute()

    # -- construction -----------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setWidget(self._build_display_panel())
        self._slice_controls_enabled()
        root.addWidget(scroll)

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
                "The conversion still runs; the numbers are in the status bar."
            )
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setWordWrap(True)
            host_layout.addWidget(label)
        root.addWidget(self.plot_host, 1)

        self.status = self.statusBar()

    def _build_display_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        setup_button = QtWidgets.QPushButton("Conversion setup…")
        setup_button.setToolTip(
            "Geometry, UB, energy, circles, frame range — everything that decides\n"
            "where a pixel lands in reciprocal space."
        )
        setup_button.clicked.connect(self.show_setup)
        layout.addWidget(setup_button)

        # -- visibility --
        box = QtWidgets.QGroupBox("Visibility")
        form = QtWidgets.QFormLayout(box)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentIndexChanged.connect(self.refresh_view)
        form.addRow("Colormap", self.cmap_combo)
        self.show_points_check = QtWidgets.QCheckBox("Show points")
        self.show_points_check.setChecked(True)
        self.show_points_check.toggled.connect(self.refresh_view)
        form.addRow(self.show_points_check)
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
            spin = _spin(value, -1e4, 1e4, 4, 0.1, 62)
            spin.setToolTip(
                f"{name} component of the plane normal.\n"
                "Type any direction -- [1 1 2] is fine, it does not have to be a\n"
                "unit vector -- or drag the yellow arrow in the view."
            )
            spin.valueChanged.connect(self.refresh_view)
            normal_row.addWidget(label)
            normal_row.addWidget(spin)
            self.slice_normal_spins.append(spin)
        normal_widget = QtWidgets.QWidget()
        normal_widget.setLayout(normal_row)
        form.addRow("Plane normal", normal_widget)
        self.slice_unit_label = QtWidgets.QLabel("")
        self.slice_unit_label.setToolTip("The same direction, normalised -- what the plane actually uses.")
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

        self.slice_offset_spin = _spin(0.0, -1e6, 1e6, 4, 0.005)
        self.slice_offset_spin.setToolTip(
            "How far the plane sits from the centre of the map, along its normal."
        )
        self.slice_offset_spin.valueChanged.connect(self.refresh_view)
        form.addRow("Offset along normal", self.slice_offset_spin)
        self.slice_width_spin = _spin(0.01, 1e-6, 1e6, 5, 0.005)
        self.slice_width_spin.valueChanged.connect(self.refresh_view)
        form.addRow("Slice thickness", self.slice_width_spin)
        self.slice_lock_check = QtWidgets.QCheckBox("Lock slice (hide handles)")
        self.slice_lock_check.setToolTip(
            "Freeze the plane where it is and take the arrow and outline out of\n"
            "the view, so they stop covering the data. Untick to move it again."
        )
        self.slice_lock_check.toggled.connect(self._slice_lock_toggled)
        form.addRow(self.slice_lock_check)
        self.point_size_spin = _spin(5.0, 0.5, 50.0, 1, 0.5)
        self.point_size_spin.valueChanged.connect(self.refresh_view)
        form.addRow("Point size", self.point_size_spin)
        layout.addWidget(box)

        # -- intensity --
        box = QtWidgets.QGroupBox("Intensity range")
        form = QtWidgets.QFormLayout(box)
        self.int_min_spin = _spin(0.0, -1e12, 1e12, 3, 10.0)
        self.int_max_spin = _spin(1.0, -1e12, 1e12, 3, 10.0)
        for spin in (self.int_min_spin, self.int_max_spin):
            spin.valueChanged.connect(self._intensity_edited)
        self.int_auto_check = QtWidgets.QCheckBox("Auto")
        self.int_auto_check.setChecked(True)
        self.int_auto_check.toggled.connect(self._intensity_auto_toggled)
        form.addRow("Min", self.int_min_spin)
        form.addRow("Max", self.int_max_spin)
        form.addRow(self.int_auto_check)
        layout.addWidget(box)

        # -- HKL range --
        box = QtWidgets.QGroupBox("HKL range")
        grid = QtWidgets.QGridLayout(box)
        self.hkl_spins = {}
        for row, name in enumerate(AXIS_NAMES):
            grid.addWidget(QtWidgets.QLabel(name), row, 0)
            low = _spin(0.0, -1e6, 1e6, 4, 0.01, 92)
            high = _spin(1.0, -1e6, 1e6, 4, 0.01, 92)
            for spin in (low, high):
                spin.valueChanged.connect(self._hkl_edited)
            grid.addWidget(low, row, 1)
            grid.addWidget(high, row, 2)
            self.hkl_spins[name] = (low, high)
        self.hkl_auto_check = QtWidgets.QCheckBox("Whole map")
        self.hkl_auto_check.setChecked(True)
        self.hkl_auto_check.toggled.connect(self._hkl_auto_toggled)
        grid.addWidget(self.hkl_auto_check, 3, 0, 1, 3)
        layout.addWidget(box)

        # -- downsampling --
        box = QtWidgets.QGroupBox("Downsampling")
        form = QtWidgets.QFormLayout(box)
        self.downsample_check = QtWidgets.QCheckBox("On")
        self.downsample_check.setChecked(True)
        self.downsample_check.setToolTip(
            "Draw at most the number below, taking every Nth point.\n"
            "Off sends the whole cloud to the renderer, which can be slow."
        )
        self.downsample_check.toggled.connect(self.refresh_view)
        form.addRow(self.downsample_check)
        self.display_points_spin = QtWidgets.QSpinBox()
        self.display_points_spin.setRange(1000, 20_000_000)
        self.display_points_spin.setSingleStep(50_000)
        self.display_points_spin.setValue(500_000)
        self.display_points_spin.valueChanged.connect(self.refresh_view)
        form.addRow("Max drawn", self.display_points_spin)
        layout.addWidget(box)

        # -- gridded volume --
        box = QtWidgets.QGroupBox("Gridded volume")
        form = QtWidgets.QFormLayout(box)
        self.grid_button = QtWidgets.QPushButton("Grid && load (.vti)")
        self.grid_button.setToolTip(
            "Bin the pixels onto a regular HKL grid over the range above and\n"
            "save it beside the image file as a .vti.\n"
            "The cell size is taken from the data: no finer than the spacing\n"
            "the scan actually sampled, which differs per axis because the\n"
            "rocking step is much coarser than the detector pixels.\n"
            "An existing grid for the same range is loaded instead of redone."
        )
        self.grid_button.clicked.connect(self.grid_and_load)
        form.addRow(self.grid_button)
        self.detail_spin = _spin(1.0, 0.25, 8.0, 2, 0.25, 72)
        self.detail_spin.setToolTip(
            "Scales the automatic cell size. 1 is a cell holding about one\n"
            "sample, which is as fine as the scan can justify. Below 1 gives a\n"
            "finer grid at several times the memory; above 1 a coarser, lighter\n"
            "one. Each value is cached separately."
        )
        form.addRow("Cell size ×", self.detail_spin)
        self.show_volume_check = QtWidgets.QCheckBox("Show volume")
        self.show_volume_check.setEnabled(False)
        self.show_volume_check.setToolTip(
            "Draw the grid instead of the point cloud. Rendering a loaded grid\n"
            "costs a fraction of re-converting and drawing every pixel."
        )
        self.show_volume_check.toggled.connect(self._volume_shown_toggled)
        form.addRow(self.show_volume_check)
        self.grid_label = QtWidgets.QLabel("none")
        self.grid_label.setWordWrap(True)
        self.grid_label.setStyleSheet("color: gray;")
        form.addRow(self.grid_label)
        layout.addWidget(box)

        layout.addStretch(1)
        return panel

    # -- panel reactions --------------------------------------------------

    def show_setup(self):
        self.setup_dialog.show()
        self.setup_dialog.raise_()
        self.setup_dialog.activateWindow()

    def _intensity_edited(self):
        if self._updating:
            return
        # Typing a limit is a request for manual control.
        self.int_auto_check.blockSignals(True)
        self.int_auto_check.setChecked(False)
        self.int_auto_check.blockSignals(False)
        self.refresh_view()

    def _intensity_auto_toggled(self, checked):
        if checked:
            self._fill_intensity_range()
        self.refresh_view()

    def _hkl_edited(self):
        if self._updating:
            return
        self.hkl_auto_check.blockSignals(True)
        self.hkl_auto_check.setChecked(False)
        self.hkl_auto_check.blockSignals(False)
        self.refresh_view()

    def _hkl_auto_toggled(self, checked):
        if checked:
            self._fill_hkl_range()
        self.refresh_view()

    def _slice_controls_enabled(self):
        """Grey out what cannot be used: the plane's own geometry while it is
        locked, and everything about it while the slice is hidden."""
        shown = self.show_slice_check.isChecked()
        movable = shown and not self.slice_lock_check.isChecked()
        for spin in (*self.slice_normal_spins, self.slice_offset_spin):
            spin.setEnabled(movable)
        for widget in self._slice_preset_buttons:
            widget.setEnabled(movable)
        self.slice_width_spin.setEnabled(shown)
        self.slice_lock_check.setEnabled(shown)

    def _slice_shown_toggled(self, _checked=None):
        self._slice_controls_enabled()
        self.refresh_view()

    def _slice_lock_toggled(self, _checked=None):
        self._slice_controls_enabled()
        self.refresh_view()

    def _set_normal_axis(self, axis: int):
        self._updating = True
        try:
            for index, spin in enumerate(self.slice_normal_spins):
                spin.setValue(1.0 if index == axis else 0.0)
            self.slice_offset_spin.setValue(0.0)
        finally:
            self._updating = False
        self.refresh_view()

    def slice_plane(self):
        """(unit normal, origin) of the cutting plane, in HKL.

        Whatever is typed is treated as a direction: only its orientation is
        used, so [1 1 2] and its unit form give the same plane.
        """
        normal = np.array([s.value() for s in self.slice_normal_spins], dtype=float)
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            normal, length = np.array([0.0, 0.0, 1.0]), 1.0
        normal = normal / length
        centre = self._cloud_centre()
        return normal, centre + normal * self.slice_offset_spin.value()

    def _show_unit_normal(self):
        normal, _ = self.slice_plane()
        self.slice_unit_label.setText(
            "unit:  %.3f, %.3f, %.3f" % tuple(normal)
        )

    def _cloud_centre(self):
        bounds = None if self.cloud is None else self.cloud.bounds()
        if bounds is None:
            return np.zeros(3)
        return np.array([(low + high) / 2.0 for low, high in bounds])

    def _fill_intensity_range(self):
        if self.cloud is None or self.cloud.intensity.size == 0:
            return
        self._updating = True
        try:
            self.int_min_spin.setValue(float(self.cloud.intensity.min()))
            self.int_max_spin.setValue(float(self.cloud.intensity.max()))
        finally:
            self._updating = False

    def _fill_hkl_range(self):
        bounds = None if self.cloud is None else self.cloud.bounds()
        if bounds is None:
            return
        self._updating = True
        try:
            for name, (low, high) in zip(AXIS_NAMES, bounds):
                self.hkl_spins[name][0].setValue(float(low))
                self.hkl_spins[name][1].setValue(float(high))
        finally:
            self._updating = False

    def _centre_slice(self):
        """Put the plane through the middle of the map with a sane thickness."""
        bounds = None if self.cloud is None else self.cloud.bounds()
        if bounds is None:
            return
        span = max(high - low for low, high in bounds)
        self._updating = True
        try:
            self.slice_offset_spin.setValue(0.0)
            self.slice_width_spin.setValue(max(float(span) / 50, 1e-5))
        finally:
            self._updating = False

    # -- compute and draw --------------------------------------------------

    def recompute(self):
        """Re-read the setup window, convert, and redraw."""
        geometry = self.geometry_data
        wanted = self.setup_dialog.wants_master_angles()
        if wanted != (geometry.sources.get("angles") == "master"):
            refreshed = read_geometry(
                geometry.image_path, geometry.master_path, angles_from_master=wanted
            )
            geometry.sample_angles = refreshed.sample_angles
            geometry.detector_angles = refreshed.detector_angles
            geometry.sources["angles"] = refreshed.sources.get("angles")

        self.setup_dialog.apply_to_geometry()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.cloud = convert(geometry, **self.setup_dialog.conversion_arguments())
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.status.showMessage(f"Conversion failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "HKL conversion failed", str(exc))
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        if self.int_auto_check.isChecked():
            self._fill_intensity_range()
        if self.hkl_auto_check.isChecked():
            self._fill_hkl_range()
        self._centre_slice()
        self._needs_camera_reset = True
        self.refresh_view()

    def visible_cloud(self):
        """The points that survive the HKL box and the downsample stride."""
        if self.cloud is None or self.cloud.points.size == 0:
            return np.zeros((0, 3), np.float32), np.zeros(0, np.float32), 0
        points, intensity = self.cloud.points, self.cloud.intensity
        if not self.hkl_auto_check.isChecked():
            keep = np.ones(points.shape[0], dtype=bool)
            for axis, name in enumerate(AXIS_NAMES):
                low, high = (s.value() for s in self.hkl_spins[name])
                keep &= (points[:, axis] >= low) & (points[:, axis] <= high)
            points, intensity = points[keep], intensity[keep]
        in_range = points.shape[0]
        if self.downsample_check.isChecked():
            limit = self.display_points_spin.value()
            if in_range > limit:
                # Every Nth point, so the cloud thins evenly instead of losing
                # one whole corner of reciprocal space.
                stride = int(np.ceil(in_range / limit))
                points, intensity = points[::stride], intensity[::stride]
        return points, intensity, in_range

    def _lookup_table(self, low, high):
        """A colormap whose opacity rises with intensity.

        Faint points fade out instead of packing the view with haze, so the
        bright features stay visible inside the cloud.
        """
        import pyvista as pv

        lut = pv.LookupTable(cmap=self.cmap_combo.currentText())
        # The ramp lives on the lookup table alone -- passing opacity= to
        # add_mesh as well would be a second, competing source for it.
        lut.apply_opacity([0, 1])
        if high > low:
            lut.scalar_range = (float(low), float(high))
        return lut

    def display_bounds(self):
        """The box the axes describe: the whole map, whatever is drawn in it.

        Taken from the converted cloud rather than from the points currently on
        screen, so hiding the points or thinning them down to a slice leaves the
        H/K/L axes reading exactly as they did.
        """
        bounds = None if self.cloud is None else self.cloud.bounds()
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

    def _on_plane_dragged(self, normal, origin):
        """The plane widget moved: mirror it into the boxes and redraw."""
        if self._updating:
            return
        normal = np.asarray(normal, dtype=float)
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            return
        normal = normal / length
        offset = float(np.dot(np.asarray(origin, dtype=float) - self._cloud_centre(), normal))
        self._updating = True
        try:
            for spin, value in zip(self.slice_normal_spins, normal):
                spin.setValue(float(value))
            self.slice_offset_spin.setValue(offset)
        finally:
            self._updating = False
        # The guard above suppressed refresh_view, which is what normally keeps
        # this in step, so update it here rather than leave a stale reading.
        self._show_unit_normal()
        self._draw_slab()

    def _draw_slab(self, take_bar=None):
        """Just the slab actor -- used by the widget so dragging stays smooth.

        *take_bar* says whether the slab should carry the scalar bar; left out
        (as it is on the drag path) it takes it only when nothing else has.
        """
        if take_bar is None:
            take_bar = not (
                self.show_points_check.isChecked()
                or (self.show_volume_check.isChecked() and self.grid is not None)
            )
        if self.plotter is None:
            return
        import pyvista as pv

        try:
            self.plotter.remove_actor("slice", reset_camera=False)
        except Exception:
            pass
        if not self.show_slice_check.isChecked():
            return
        points, intensity, _ = self.visible_cloud()
        if points.shape[0] == 0:
            return
        normal, origin = self.slice_plane()
        distance = (points - origin) @ normal
        inside = np.abs(distance) <= self.slice_width_spin.value() / 2.0
        if not np.any(inside):
            return
        low, high = self.int_min_spin.value(), self.int_max_spin.value()
        slab = pv.PolyData(points[inside].astype(np.float32))
        slab["intensity"] = intensity[inside].astype(np.float32)
        self.plotter.add_mesh(
            slab,
            scalars="intensity",
            cmap=self._lookup_table(low, high),
            clim=(low, high),
            point_size=self.point_size_spin.value() * 1.6,
            name="slice",
            show_scalar_bar=take_bar,
            scalar_bar_args=dict(SCALAR_BAR),
            reset_camera=False,
        )

    def requested_hkl_range(self):
        """The box to grid: the panel's numbers, or None for the whole map."""
        if self.hkl_auto_check.isChecked():
            return None
        out = []
        for name in AXIS_NAMES:
            low, high = (spin.value() for spin in self.hkl_spins[name])
            out.extend((float(low), float(high)))
        return tuple(out)

    def grid_and_load(self):
        """Bin the pixels onto a regular HKL grid and draw that instead.

        The grid is written beside the image file. Asking again for a range
        already on disk reloads it rather than repeating the work.
        """
        try:
            from hkl_grid import GridCancelled
            from hkl_grid import make_drawable
            from hkl_grid import ensure_grid
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Gridding unavailable", f"hkl_grid could not be imported: {exc}"
            )
            return

        hkl_range = self.requested_hkl_range()
        progress = QtWidgets.QProgressDialog(
            "Binning frames onto the HKL grid…", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("HKL grid")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        def report(done, total):
            progress.setMaximum(total)
            progress.setValue(done)
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()

        # The geometry the setup window is currently holding, edits included,
        # and the same frame range the cloud is using -- not a fresh read of
        # the files, which would quietly ignore both.
        arguments = self.setup_dialog.conversion_arguments()
        self.grid_button.setEnabled(False)
        try:
            grid, path, regridded = ensure_grid(
                self.geometry_data.image_path,
                self.geometry_data.master_path,
                hkl_range=hkl_range,
                geometry=self.geometry_data,
                oversample=self.detail_spin.value(),
                first=arguments["first"],
                last=arguments["last"],
                frame_step=arguments["step"],
                progress=report,
            )
        except GridCancelled:
            self.status.showMessage("Gridding cancelled — nothing was written.")
            return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Gridding failed", f"{type(exc).__name__}: {exc}"
            )
            return
        finally:
            progress.close()
            self.grid_button.setEnabled(True)

        # NaN marks the cells the scan never reached. The volume mapper does not
        # render it transparent -- it lights the whole empty region up -- so swap
        # it for the bottom of the colour scale, in place to avoid copying a few
        # hundred megabytes. The file on disk keeps the NaN.
        span = make_drawable(grid)
        self.grid, self.grid_path, self.grid_range = grid, path, hkl_range
        filled, total = (int(v) for v in grid.field_data["filled_cells"])
        self.grid_label.setText(
            "{name}\n{dims} cells, step {step}\n{filled:,} measured ({pct:.0f}%){how}".format(
                name=path.name,
                dims="×".join(str(int(n)) for n in grid.dimensions),
                step=" ".join(f"{float(v):.1e}" for v in grid.spacing),
                filled=filled,
                pct=100.0 * filled / max(total, 1),
                how="" if regridded else "\nreused the file already on disk",
            )
        )

        self._updating = True
        self.show_volume_check.setEnabled(True)
        self.show_volume_check.setChecked(True)
        # The cloud and the grid hold the same data; drawing both would only
        # cost time, and the grid is the reason for gridding.
        self.show_points_check.setChecked(False)
        if self.int_auto_check.isChecked():
            self.int_min_spin.setValue(span[0])
            self.int_max_spin.setValue(span[1])
        self._updating = False
        self.refresh_view()

    def _volume_shown_toggled(self, _shown):
        if not self._updating:
            self.refresh_view()

    def _draw_volume(self, lut, low, high, take_bar):
        """Draw the loaded grid as a volume.

        Cells the scan never reached were NaN; they are drawn from the NaN-free
        copy at the bottom of the colour scale, where the opacity ramp starts at
        zero, so the empty region stays invisible.
        """
        try:
            self.plotter.add_volume(
                self.grid,
                scalars="intensity",
                cmap=lut,
                clim=(low, high),
                name="volume",
                show_scalar_bar=take_bar,
                scalar_bar_args=dict(SCALAR_BAR),
                reset_camera=False,
            )
        except Exception as exc:
            self.status.showMessage(f"Volume rendering failed: {type(exc).__name__}: {exc}")

    def refresh_view(self):
        if self._updating:
            return
        points, intensity, in_range = self.visible_cloud()
        self._show_unit_normal()
        self._report(points.shape[0], in_range)
        if self.plotter is None:
            return
        import pyvista as pv

        self.plotter.clear()
        self.plotter.set_background(BACKGROUND)
        # color= is the *label* colour; left alone the H/K/L letters are drawn
        # in the theme font colour, which is black and so invisible here.
        self.plotter.add_axes(
            xlabel="H", ylabel="K", zlabel="L", color=AXIS_COLOR
        )
        bounds = self.display_bounds()
        showing_volume = self.show_volume_check.isChecked() and self.grid is not None
        if points.shape[0] == 0 and not showing_volume:
            return

        low = self.int_min_spin.value()
        high = self.int_max_spin.value()
        lut = self._lookup_table(low, high)

        if self.show_points_check.isChecked():
            mesh = pv.PolyData(points.astype(np.float32))
            mesh["intensity"] = intensity.astype(np.float32)
            self.plotter.add_mesh(
                mesh,
                scalars="intensity",
                cmap=lut,
                clim=(low, high),
                point_size=self.point_size_spin.value(),
                name="points",
                show_scalar_bar=True,   # first drawn owns the bar; see below
                scalar_bar_args=dict(SCALAR_BAR),
                reset_camera=False,
            )

        # One scalar bar between them: a second add_* with the same title
        # stacks another bar on top of the first.
        bar_taken = self.show_points_check.isChecked()
        if showing_volume:
            self._draw_volume(lut, low, high, take_bar=not bar_taken)
            bar_taken = True

        self._draw_slab(take_bar=not bar_taken)

        # plotter.clear() leaves widgets behind, so without this every redraw
        # would add another plane on top of the last -- several arrows at stale
        # normals, and a drag grabbing whichever one VTK happened to hit.
        try:
            self.plotter.clear_plane_widgets()
        except Exception:
            pass
        if self.show_slice_check.isChecked() and not self.slice_lock_check.isChecked():
            normal, origin = self.slice_plane()
            try:
                self.plotter.add_plane_widget(
                    callback=self._on_plane_dragged,
                    normal=tuple(normal),
                    origin=tuple(origin),
                    bounds=bounds,
                    color=SLICE_COLOR,
                    implicit=True,
                    normal_rotation=True,
                    origin_translation=True,
                    outline_opacity=0,
                    tubing=False,
                    # Creating the widget would otherwise fire the callback and
                    # redraw the slab for a plane that has not moved.
                    test_callback=False,
                )
            except Exception:
                pass

        # Pinned to the whole map, and drawn light so it reads on black.
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

    def _report(self, drawn: int, in_range: int):
        cloud = self.cloud
        if cloud is None:
            return
        if cloud.points.size == 0:
            self.status.showMessage(
                f"{cloud.frames_used} frame(s), {cloud.total_pixels/1e6:.1f} M pixels — "
                "nothing above the conversion threshold."
            )
            return
        (h0, h1), (k0, k1), (l0, l1) = cloud.bounds()
        if self.geometry_data.sources.get("angles") == "missing":
            self.status.showMessage(
                "No diffractometer angles found in either file — every frame is "
                "being treated as 0°, so this map is not meaningful. Check that "
                "the master file is beside the image file."
            )
            return
        if self.show_volume_check.isChecked() and self.grid is not None:
            filled, total = (int(v) for v in self.grid.field_data["filled_cells"])
            dims = "×".join(str(int(n)) for n in self.grid.dimensions)
            step = " ".join(f"{float(v):.1e}" for v in self.grid.spacing)
            self.status.showMessage(
                f"{cloud.frames_used} frame(s) · gridded volume {dims} "
                f"(step {step}) · {filled:,} of {total:,} cells measured "
                f"({100.0 * filled / max(total, 1):.0f}%) · {self.grid_path.name}"
            )
            return
        self.status.showMessage(
            f"{cloud.frames_used} frame(s) · drawing {drawn:,} of {in_range:,} in range "
            f"({len(cloud.points):,} converted from {cloud.total_pixels/1e6:.1f} M pixels) · "
            f"H {h0:.4f}…{h1:.4f}  K {k0:.4f}…{k1:.4f}  L {l0:.4f}…{l1:.4f} · "
            f"angles from {self.geometry_data.sources.get('angles')}"
        )


def main():
    if len(sys.argv) < 2:
        print(f"usage: {Path(sys.argv[0]).name} <pva_image_file.h5> [master.hdf]")
        return 2
    app = QtWidgets.QApplication(sys.argv)
    master = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    window = HklWindow(Path(sys.argv[1]), master)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
