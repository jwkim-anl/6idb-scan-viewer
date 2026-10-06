"""The detector's image stack, and everything recorded alongside it.

One ``pva*.h5`` holds the frames, the per-frame EPICS readings, the frame
clock and sometimes the detector's ROIs. The master's own channels are
resampled onto those frames here too, since a free-running detector rarely
takes exactly as many frames as the scan took points.

No Qt. A notebook imports this directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # Registers the blosc/lz4/bitshuffle filters that entry/data/data needs.
    import hdf5plugin  # noqa: F401
    HAVE_HDF5PLUGIN = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_HDF5PLUGIN = False

import h5py

HDF5PLUGIN_HINT = (
    "The image stack is blosc-compressed and could not be decoded.\n\n"
    "This needs the 'hdf5plugin' package importable in the Python running this "
    "viewer:\n\n    pip install hdf5plugin\n\n"
    "(or conda install -c conda-forge hdf5plugin)"
)

from da6idb.files import (
    IMAGE_DATA_PATH,
    MASTER_PRIMARY_PATH,
    ROI_GROUP_PATH,
    master_path_for_image,
    read_master_rois,
)

FRAME_TIMESTAMP_PATH = "entry/data/frame_timestamp"

CA_GROUP_PATHS = ("entry/data/metadata/ca", "entry/data/metadata/motor_positions")

# Master-file channels are offered alongside the image file's own, under this
# prefix so their provenance is plain and they cannot collide.
SCAN_PREFIX = "scan/"

# Refuse rather than thrash: the stack is read into memory in one go (a 110 MB
# 103x516x516 stack decodes in well under a second).
MAX_STACK_BYTES = 8 * 1024 ** 3

# Several files are held at once when they are compared side by side.
MAX_TOTAL_BYTES = 8 * 1024 ** 3

def nearest_index(frame_epoch: np.ndarray, scan_epoch: np.ndarray) -> np.ndarray:
    """For each frame, the scan point closest to it in time.

    Shared with ``hkl_viewer``, which matches the master's diffractometer angles
    onto the frames the same way.
    """
    right = np.clip(np.searchsorted(scan_epoch, frame_epoch), 1, scan_epoch.size - 1)
    left = right - 1
    take_left = frame_epoch - scan_epoch[left] <= scan_epoch[right] - frame_epoch
    return np.where(take_left, left, right)

def _scan_time_base(group) -> np.ndarray | None:
    """A primary field's EPOCH that is a real, advancing clock.

    Not every field has one: an ``EpicsSignal`` that was never re-read keeps the
    EPICS zero (631152000.0, i.e. 1990) for every point, which places nothing in
    time. One good clock is used for the whole stream, so those fields are
    carried along by it rather than dropped.
    """
    for name in group.keys():
        item = group[name]
        if not isinstance(item, h5py.Group) or "EPOCH" not in item:
            continue
        epoch = np.asarray(item["EPOCH"][()], dtype=float).ravel()
        if (
            epoch.size > 1
            and epoch[0] > 1e9                      # later than 2001, so a real time
            and np.all(np.diff(epoch) >= 0)
            and epoch[-1] > epoch[0]
        ):
            return epoch
    return None

def evenly_spaced_index(n_frames: int, n_scan_points: int) -> np.ndarray:
    """Map each frame onto a scan point by position, not by time.

    Used only when the image file has no per-frame timestamp to match against
    (see ``read_scan_channels``), so frame 0 takes scan point 0, the last frame
    takes the last scan point, and the rest are spread evenly in between.
    """
    if n_frames <= 1:
        return np.zeros(max(n_frames, 0), dtype=int)
    positions = np.arange(n_frames) * (n_scan_points - 1) / (n_frames - 1)
    return np.round(positions).astype(int)

def _scan_point_count(group) -> int | None:
    """How many points the master's primary stream has.

    Used in place of ``_scan_time_base`` when there is no frame timestamp to
    read a count off of a real clock: the most common length among the
    stream's own value arrays, since a few fields (e.g. read-once presets) can
    be shorter or longer than the scan itself.
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
            # The length is in the shape; reading the array itself would pull
            # every field of the stream off disk only to measure it.
            size = int(values.size)
        except (AttributeError, TypeError, ValueError):
            continue
        if size > 1:
            counts[size] = counts.get(size, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]

def read_scan_channels(
    image: Path, frame_epoch, n_frames: int | None = None, master: Path | None = None
) -> dict[str, np.ndarray]:
    """The master file's primary fields, resampled onto the image's frames.

    The detector free-runs, so the two files rarely have the same number of
    points -- 103 frames against 101 scan points is typical, the camera starting
    a moment early. Each frame therefore takes the value of the scan point
    nearest it in time; frames outside the scan hold the first or last point.

    When the image file carries no per-frame timestamp at all, there is no
    clock to match against, so frames are instead spread evenly across the
    master's scan points by position (see ``evenly_spaced_index``) -- a
    rougher alignment, but the best available without a time base.
    """
    have_epoch = frame_epoch is not None and len(frame_epoch) > 0
    if not have_epoch and not n_frames:
        return {}
    if master is None:
        master = master_path_for_image(image)
    if master is None:
        return {}
    try:
        with h5py.File(master, "r") as f:
            group = f.get(MASTER_PRIMARY_PATH)
            if group is None:
                return {}

            if have_epoch:
                frame_epoch = np.asarray(frame_epoch, dtype=float)
                scan_epoch = _scan_time_base(group)
                if scan_epoch is None:
                    return {}
                index = nearest_index(frame_epoch, scan_epoch)

                # Refuse to pair files that do not describe the same stretch
                # of time -- a mismatched master would otherwise be resampled
                # into plausible-looking nonsense.
                spacing = float(np.median(np.diff(scan_epoch))) or 1.0
                if float(np.median(np.abs(frame_epoch - scan_epoch[index]))) > 5 * spacing:
                    return {}
                n_scan_points = scan_epoch.size
            else:
                n_scan_points = _scan_point_count(group)
                if not n_scan_points:
                    return {}
                index = evenly_spaced_index(n_frames, n_scan_points)

            channels: dict[str, np.ndarray] = {}
            for name in group.keys():
                item = group[name]
                if not isinstance(item, h5py.Group) or "value" not in item:
                    continue
                try:
                    values = np.asarray(item["value"][()], dtype=float).ravel()
                except (TypeError, ValueError):
                    continue
                if values.size == n_scan_points:
                    channels[SCAN_PREFIX + name] = values[index]
            return channels
    except OSError:
        return {}

class ImageData:
    """The whole contents of one ``pva*.h5`` file, held in memory.

    ``stack`` is ``(n_frames, nx, ny)`` float32 -- indexed ``[frame, x, y]``, the
    file's own ``pixel_order='xy'`` convention, which is also what pyqtgraph's
    default (col-major) ``ImageItem`` expects, so frames need no transposing to
    display the right way round.
    """

    def __init__(self, path: Path, master: Path | None = None):
        self.path = Path(path)
        # Newer scans keep the image in a different tree from its master, so
        # the beside-it reverse lookup no longer finds it. The caller usually
        # knows the pair already; guessing is only the fallback.
        self.master_path = Path(master) if master is not None else master_path_for_image(self.path)
        with h5py.File(self.path, "r") as f:
            dset = f.get(IMAGE_DATA_PATH)
            if dset is None or not isinstance(dset, h5py.Dataset):
                raise RuntimeError(
                    f"{self.path.name} has no image stack at '{IMAGE_DATA_PATH}'."
                )
            if dset.ndim not in (2, 3):
                raise RuntimeError(
                    f"'{IMAGE_DATA_PATH}' is {dset.ndim}-dimensional; "
                    "only 2D images and 3D stacks are supported."
                )
            if dset.nbytes > MAX_STACK_BYTES:
                raise RuntimeError(
                    f"{self.path.name} holds {dset.nbytes / 1024 ** 3:.1f} GB of image "
                    f"data, more than this viewer loads at once "
                    f"({MAX_STACK_BYTES / 1024 ** 3:.0f} GB)."
                )
            try:
                stack = np.asarray(dset[()], dtype=np.float32)
            except OSError as exc:
                if not HAVE_HDF5PLUGIN:
                    raise RuntimeError(HDF5PLUGIN_HINT) from exc
                raise RuntimeError(f"Could not read the image stack: {exc}") from exc

            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]
            self.stack = stack
            n_frames = stack.shape[0]

            self.epoch = self._read_epoch(f, n_frames)
            self.timestamps = (
                None if self.epoch is None else self.epoch - self.epoch[0]
            )
            self.channels = self._read_channels(f, n_frames)
            self.stored_rois = self._read_stored_rois(f)
        if not self.stored_rois and self.master_path is not None:
            # Nothing in the image file: newer writers leave the rectangles to
            # the master's baseline instead.
            self.stored_rois = read_master_rois(self.master_path)

        # Offered next to the image file's own channels, on the same frames.
        self.channels.update(
            read_scan_channels(self.path, self.epoch, n_frames, master=self.master_path)
        )

    @property
    def n_frames(self) -> int:
        return int(self.stack.shape[0])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self.stack.shape[1]), int(self.stack.shape[2])

    @staticmethod
    def _read_epoch(f: h5py.File, n_frames: int) -> np.ndarray | None:
        """Absolute frame times, needed to line the frames up with the scan."""
        item = f.get(FRAME_TIMESTAMP_PATH)
        if item is None:
            return None
        stamps = np.asarray(item[()], dtype=float).ravel()
        if stamps.size != n_frames or stamps.size == 0:
            return None
        return stamps

    @staticmethod
    def _read_channels(f: h5py.File, n_frames: int) -> dict[str, np.ndarray]:
        """Per-frame scalar channels (the CA metadata), keyed by name."""
        channels: dict[str, np.ndarray] = {}
        for group_path in CA_GROUP_PATHS:
            group = f.get(group_path)
            if not isinstance(group, h5py.Group):
                continue
            for name in group.keys():
                if name in channels:
                    continue
                item = group.get(name)
                if not isinstance(item, h5py.Dataset):
                    continue
                try:
                    values = np.asarray(item[()], dtype=float).ravel()
                except (OSError, TypeError, ValueError):
                    continue
                if values.size == n_frames:
                    channels[name] = values
        return channels

    @staticmethod
    def _read_stored_rois(f: h5py.File) -> list[dict]:
        """The rectangles saved in ``entry/data/rois`` (x/y/w/h attributes)."""
        group = f.get(ROI_GROUP_PATH)
        if not isinstance(group, h5py.Group):
            return []
        rois = []
        for name in sorted(group.keys()):
            item = group.get(name)
            if not isinstance(item, h5py.Dataset):
                continue  # e.g. the 'info' subgroup
            attrs = item.attrs
            if not all(key in attrs for key in ("x", "y", "w", "h")):
                continue
            rois.append(
                {
                    "name": str(name),
                    "x": float(attrs["x"]),
                    "y": float(attrs["y"]),
                    "w": max(1.0, float(attrs["w"])),
                    "h": max(1.0, float(attrs["h"])),
                    "angle": float(attrs.get("angle", 0.0)),
                }
            )
        return rois
