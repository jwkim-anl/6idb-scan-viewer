"""Turning detector frames into a cloud of points in reciprocal space.

``xrayutilities`` does the trigonometry; this wraps it in the geometry the
files describe, and reduces each frame as it is converted -- a six-hundred
frame scan is 160 million pixels, far more than can be held at once, so the
dim ones are dropped and the brightest kept as it goes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xrayutilities as xu

import h5py

from da6idb.files import IMAGE_DATA_PATH
from da6idb.hkl.geometry import HklGeometry

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
