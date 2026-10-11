"""Shared pieces for the characterisation tests.

These tests exist to pin what the code does *today*, before the analysis
package pulls the data layer out from under the windows. They are deliberately
about values, not about whether a call raises: a refactor that moves a function
unchanged has to reproduce these numbers exactly, and anything that does not is
either a bug being introduced or a behaviour change that should be deliberate.

Only ``tise2_2_00152`` ships with the repository, so everything else is built
here -- small, deterministic, and shaped like the real files -- which also lets
the awkward cases be covered: a scan with no frame timestamp, a count plan, a
two-motor raster, an attenuator that moves mid-scan, a master that names its
detector file somewhere else, and ROIs recorded in the baseline rather than in
the image file.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden"

# The one pair that travels with the repository.
def _example(name: str) -> Path:
    """The example scan, wherever it is kept.

    Beside the code or under ``data/``: the tests that use it guard on
    :func:`have_example`, so a wrong guess here does not fail them -- it
    quietly turns a dozen of them into no-ops, which is worse.
    """
    for folder in (ROOT, ROOT / "data"):
        if (folder / name).is_file():
            return folder / name
    return ROOT / name


EXAMPLE_MASTER = _example("tise2_2_00152_master.hdf")
EXAMPLE_IMAGE = _example("pva_tise2_2_00152.h5")

PRIMARY = "entry/instrument/bluesky/streams/primary"
BASELINE = "entry/instrument/bluesky/streams/baseline"
METADATA = "entry/instrument/bluesky/metadata"


def have_example() -> bool:
    return EXAMPLE_MASTER.is_file() and EXAMPLE_IMAGE.is_file()


# -- building files that look like the real ones ----------------------------


# What the real files put on each field, and what describe_fields reads to
# tell a motor readback from a counter channel.
FIELD_ATTRS = {
    "lambda250k_stats1_total": {"source": "PV:s6lambda1:Stats1:Total_RBV"},
    "Ion_Chamber_1": {"signal_type": "detector", "source": "PV:6idb1:scaler1.S2"},
    "scaler_time": {"source": "PV:6idb1:scaler1.T"},
    "psic_eta": {"units": "degrees", "source": "PV:6idb1:m17.RBV"},
    "psic_chi": {"units": "degrees", "source": "PV:6idb1:m19.RBV"},
    "psic_h": {"source": "PY:psic_h.position"},
}

HINTS = "detectors:\n- lambda250k_stats1_total\nmonitor: Time\nscan_type: ascan\n"


def _stream_field(group, name, values, epoch):
    item = group.create_group(name)
    value = item.create_dataset("value", data=np.asarray(values))
    item.create_dataset("time", data=np.asarray(epoch) - epoch[0])
    item.create_dataset("EPOCH", data=np.asarray(epoch))
    attrs = FIELD_ATTRS.get(name, {})
    if "signal_type" in attrs:
        item.attrs["signal_type"] = attrs["signal_type"]
    for key in ("units", "source"):
        if key in attrs:
            value.attrs[key] = attrs[key]


def _baseline_field(group, name, value):
    item = group.create_group(name)
    item.create_dataset("value", data=np.asarray([value, value]))
    item.create_dataset("value_start", data=value)
    item.create_dataset("value_end", data=value)
    item.create_dataset("time", data=np.zeros(2))
    item.create_dataset("EPOCH", data=np.full(2, 1.79e9))


def write_master(
    path,
    points=11,
    plan="rel_scan",
    motors=("psic_eta",),
    epoch0=1.79e9,
    extra_primary=None,
    baseline=None,
    metadata=None,
):
    """A master shaped like the real ones, with a motor and a detector."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch = epoch0 + np.arange(points, dtype=float)
    with h5py.File(path, "w") as f:
        primary = f.create_group(PRIMARY)
        # a peak, so curve statistics have something to find
        centre = (points - 1) / 2.0
        fields = {
            "lambda250k_stats1_total":
                1000.0 * np.exp(-0.5 * ((np.arange(points) - centre) / 1.5) ** 2),
            "Ion_Chamber_1": np.full(points, 2.0),
            "scaler_time": np.full(points, 0.1),
        }
        for index, motor in enumerate(motors):
            fields[motor] = np.linspace(10.0 + index, 11.0 + index, points)
        # Explicit values win, so a caller can give a motor a real trajectory
        # rather than the straight ramp the default would write.
        fields.update({k: np.asarray(v, dtype=float)
                       for k, v in (extra_primary or {}).items()})
        for name, values in fields.items():
            _stream_field(primary, name, values, epoch)

        base = f.create_group(BASELINE)
        _baseline_field(base, "energy", 20.0)
        for name, value in (baseline or {}).items():
            _baseline_field(base, name, value)

        meta = f.create_group(METADATA)
        meta.create_dataset("plan_name", data=np.bytes_(plan))
        meta.create_dataset("num_points", data=np.bytes_(str(points)))
        motor_yaml = "!!python/tuple\n" + "".join(f"- {m}\n" for m in motors)
        meta.create_dataset("motors", data=np.bytes_(motor_yaml))
        meta.create_dataset("detectors", data=np.bytes_("- lambda250k\n"))
        if "hints" not in (metadata or {}):
            meta.create_dataset("hints", data=np.bytes_(HINTS))
        for name, value in (metadata or {}).items():
            meta.create_dataset(name, data=np.bytes_(value))
    return path


def write_image(path, frames=13, shape=(16, 16), epoch0=1.79e9, timestamps=True,
                rois=True, filter_trans=None, seed=0):
    """An image file shaped like a pva one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stack = rng.integers(0, 100, size=(frames, *shape)).astype(np.uint32)
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/data/data", data=stack)
        if timestamps:
            f.create_dataset("entry/data/frame_timestamp",
                             data=epoch0 + np.arange(frames, dtype=float) * 0.9)
        ca = f.create_group("entry/data/metadata/ca")
        ca.create_dataset("imroi1", data=np.arange(frames, dtype=float))
        ca.create_dataset("eta", data=np.linspace(10.0, 11.0, frames))
        if filter_trans is not None:
            ca.create_dataset("filter_trans",
                              data=np.asarray(filter_trans, dtype=float))
        if rois:
            group = f.create_group("entry/data/rois")
            for index, (x, y, w, h) in enumerate(
                [(2, 3, 5, 4), (6, 1, 3, 3), (0, 0, 16, 2), (8, 8, 4, 4)], start=1
            ):
                item = group.create_dataset(f"ROI{index}", data=np.zeros(1))
                item.attrs.update(x=x, y=y, w=w, h=h, angle=0.0)
    return path


class Sandbox:
    """A temporary directory of generated files, removed on exit."""

    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(prefix="da6idb-test-"))
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


# -- golden values -----------------------------------------------------------


def _plain(value):
    """Make a value comparable and JSON-able, rounding floats to 10 digits."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 10)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, Path):
        return value.name
    if value is None:
        return None
    return str(value)


def golden_path(name) -> Path:
    return GOLDEN / f"{name}.json"


def check(name, value, record=False):
    """Compare *value* with what was recorded, or record it the first time.

    Running with ``DA6IDB_RECORD=1`` writes the file instead of comparing,
    which is how the goldens are made and the only way they should change.
    """
    plain = _plain(value)
    path = golden_path(name)
    if record or os.environ.get("DA6IDB_RECORD"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plain, indent=1, sort_keys=True) + "\n")
        return True
    if not path.is_file():
        raise AssertionError(
            f"no golden for {name!r}; run with DA6IDB_RECORD=1 to write one"
        )
    expected = json.loads(path.read_text())
    if plain != expected:
        raise AssertionError(
            f"{name} changed\n  expected: {json.dumps(expected)[:400]}\n"
            f"  actual  : {json.dumps(plain)[:400]}"
        )
    return True
