"""Rectangles on the detector, and what they reduce each frame to.

A region of interest is a corner, a size and an angle in detector pixels.
Unrotated it is a plain slice of the stack, which is both exact and fast;
rotated it has to be resampled, which is the only part that needs care.

The viewers used to do this against a pyqtgraph ROI object. The numbers are
the same either way, but here they come from a plain rectangle, so a notebook
can ask for them too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRICS = ("sum", "mean", "min", "max", "comx", "comy")

# A size within this of a whole number of pixels is that many pixels.
_SIZE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Rect:
    """A region of interest: corner, size and angle, all in detector pixels.

    The angle turns the rectangle about its corner, which is what the viewers'
    handles do.
    """

    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0

    @classmethod
    def from_roi(cls, roi) -> "Rect":
        """From a pyqtgraph ROI, so the windows can hand one straight over."""
        pos, size = roi.pos(), roi.size()
        return cls(float(pos.x()), float(pos.y()),
                   float(size.x()), float(size.y()), float(roi.angle()))

    @classmethod
    def from_stored(cls, spec: dict) -> "Rect":
        """From the dictionaries the files record."""
        return cls(float(spec["x"]), float(spec["y"]),
                   float(spec["w"]), float(spec["h"]), float(spec.get("angle", 0.0)))

    @property
    def rotated(self) -> bool:
        return abs(float(self.angle)) % 360.0 > 1e-6

    def corners(self) -> np.ndarray:
        """The four corners ``(x, y)``, going round, for drawing the rectangle.

        The same two axes :meth:`sub_frame` walks, so what is drawn is what is
        summed -- including the way the angle turns the rectangle about its
        corner rather than its centre.
        """
        radians = np.deg2rad(float(self.angle))
        along = np.array([np.cos(radians), np.sin(radians)])
        across = np.array([-np.sin(radians), np.cos(radians)])
        origin = np.array([float(self.x), float(self.y)])
        return np.array([
            origin,
            origin + self.width * along,
            origin + self.width * along + self.height * across,
            origin + self.height * across,
        ])

    def sample_shape(self, shape) -> tuple[int, int]:
        """How many samples :meth:`sub_frame` takes along each axis.

        Unrotated that is the clipped integer box; rotated it is one sample
        per pixel along each edge, counted from the geometry -- see
        :meth:`sub_frame` for why it cannot come from anywhere else.
        """
        if not self.rotated:
            x0, x1, y0, y1 = self.bbox(shape)
            return x1 - x0, y1 - y0
        return (max(1, int(np.ceil(self.width - _SIZE_TOLERANCE))),
                max(1, int(np.ceil(self.height - _SIZE_TOLERANCE))))

    def sample_points(self, shape):
        """Frame coordinates of every sample, shaped like the sub-image.

        Two arrays laid out like :meth:`sub_frame`'s result, so a per-pixel
        quantity -- a scattering angle, say -- lines up with the counts.
        """
        columns, rows = self.sample_shape(shape)
        u = np.arange(columns, dtype=float)[:, None]
        v = np.arange(rows, dtype=float)[None, :]
        # unrotated, to_frame keeps the two axes separate; broadcast so both
        # come back laid out like the sub-image
        x, y = self.to_frame(u, v, shape)
        return np.broadcast_arrays(np.asarray(x), np.asarray(y))

    def to_frame(self, u, v, shape):
        """Where ROI coordinates land on the frame, as ``(x, y)`` pixels.

        :func:`reduce_frame` reports a centre of mass in the sub-image's own
        coordinates, which is not where it sits on the detector. This undoes
        that, by the same route :meth:`sub_frame` took: a rotated rectangle
        is walked along its own axes from its corner, an unrotated one is a
        slice, so its origin is the clipped integer corner rather than the
        fractional one.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if not self.rotated:
            x0, _, y0, _ = self.bbox(shape)
            return x0 + u, y0 + v
        radians = np.deg2rad(float(self.angle))
        along = np.array([np.cos(radians), np.sin(radians)])
        across = np.array([-np.sin(radians), np.cos(radians)])
        return (self.x + u * along[0] + v * across[0],
                self.y + u * along[1] + v * across[1])

    def bbox(self, shape) -> tuple[int, int, int, int]:
        """Axis-aligned pixel bounds ``(x0, x1, y0, y1)``, clipped to *shape*."""
        nx, ny = int(shape[0]), int(shape[1])
        x0 = max(0, min(int(np.floor(self.x)), nx))
        x1 = max(x0, min(int(np.ceil(self.x + self.width)), nx))
        y0 = max(0, min(int(np.floor(self.y)), ny))
        y1 = max(y0, min(int(np.ceil(self.y + self.height)), ny))
        return x0, x1, y0, y1

    def pixel_count(self, shape, frame=None) -> int:
        """How many pixels it covers -- the N in a ``background x N`` correction."""
        if self.rotated:
            sub = self.sub_frame(frame) if frame is not None else None
            return int(sub.size) if sub is not None else 0
        x0, x1, y0, y1 = self.bbox(shape)
        return max(0, (x1 - x0) * (y1 - y0))

    def sub_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """The pixels of one frame inside the rectangle.

        Unrotated this is a slice, deliberately: resampling it instead would
        round the fractional edges differently and make a number disagree with
        itself depending on which way it was asked for. Rotated, the rectangle
        is walked along its own axes and the frame sampled where it lands.
        """
        frame = np.asarray(frame)
        if not self.rotated:
            x0, x1, y0, y1 = self.bbox(frame.shape)
            if x1 <= x0 or y1 <= y0:
                return None
            return np.asarray(frame[x0:x1, y0:y1], dtype=np.float32)

        from scipy.ndimage import map_coordinates

        # One sample per pixel along each edge, starting at the corner.
        #
        # The tolerance is not decoration. The viewers used to get this count
        # from pyqtgraph, whose own arithmetic returns a size a few parts in
        # 1e16 above the integer at some angles and exactly on it at others,
        # so a plain ceiling gave an 8x6 rectangle 48 samples at 0, 30, 60 and
        # 90 degrees and 63 at 15 and 45 -- the same rectangle over the same
        # pixels, its sum jumping about 30% with nothing but the angle. The
        # count has to come from the geometry, which is what this does.
        columns, rows = self.sample_shape(frame.shape)
        radians = np.deg2rad(self.angle)
        along = np.array([np.cos(radians), np.sin(radians)])
        across = np.array([-np.sin(radians), np.cos(radians)])
        u = np.arange(columns, dtype=float)
        v = np.arange(rows, dtype=float)
        grid = (np.array([self.x, self.y])
                + u[:, None, None] * along
                + v[None, :, None] * across)
        sub = map_coordinates(
            np.asarray(frame, dtype=float),
            [grid[..., 0].ravel(), grid[..., 1].ravel()],
            order=1, mode="constant", cval=0.0,
        ).reshape(columns, rows)
        return None if sub.size == 0 else sub.astype(np.float32)


def reduce_frame(sub: np.ndarray):
    """``(sum, mean, min, max, comx, comy)`` of one sub-image indexed ``[x, y]``."""
    total = float(np.sum(sub))
    weight = total if total != 0.0 else 1.0
    comx = float((np.arange(sub.shape[0]) @ sub.sum(axis=1)) / weight)
    comy = float((sub.sum(axis=0) @ np.arange(sub.shape[1])) / weight)
    return (total, float(np.mean(sub)), float(np.min(sub)),
            float(np.max(sub)), comx, comy)


def empty_series(n_frames: int) -> dict[str, np.ndarray]:
    """Every metric, all zero -- what a rectangle off the detector reduces to."""
    zeros = np.zeros(int(n_frames), dtype=float)
    return {key: zeros.copy() for key in METRICS}


def sub_frames(stack: np.ndarray, rect: Rect):
    """Each frame's pixels inside *rect*, laid out like :meth:`Rect.sample_points`.

    Yields ``None`` for a frame the rectangle misses entirely. Unrotated this
    is a slice per frame, rotated it resamples, exactly as :func:`series`
    does -- the two must agree pixel for pixel.
    """
    stack = np.asarray(stack)
    n_frames = int(stack.shape[0])
    if rect.rotated:
        for index in range(n_frames):
            yield rect.sub_frame(stack[index])
        return
    x0, x1, y0, y1 = rect.bbox((stack.shape[1], stack.shape[2]))
    for index in range(n_frames):
        yield (None if x1 <= x0 or y1 <= y0
               else np.asarray(stack[index, x0:x1, y0:y1], dtype=np.float32))


def series(stack: np.ndarray, rect: Rect) -> dict[str, np.ndarray]:
    """Every metric, for every frame.

    An unrotated rectangle is one slice of the whole stack, so the reductions
    run over all frames at once; a rotated one is resampled frame by frame.
    """
    stack = np.asarray(stack)
    n_frames = int(stack.shape[0])

    if rect.rotated:
        out = empty_series(n_frames)
        for index in range(n_frames):
            sub = rect.sub_frame(stack[index])
            if sub is None:
                continue
            for key, value in zip(METRICS, reduce_frame(sub)):
                out[key][index] = value
        return out

    x0, x1, y0, y1 = rect.bbox((stack.shape[1], stack.shape[2]))
    if x1 <= x0 or y1 <= y0:
        return empty_series(n_frames)
    sub = stack[:, x0:x1, y0:y1]
    sums = sub.sum(axis=(1, 2), dtype=np.float64)
    weights = np.where(sums != 0.0, sums, 1.0)
    return {
        "sum": sums,
        "mean": sub.mean(axis=(1, 2), dtype=np.float64),
        "min": sub.min(axis=(1, 2)).astype(float),
        "max": sub.max(axis=(1, 2)).astype(float),
        "comx": (sub.sum(axis=2, dtype=np.float64) @ np.arange(x1 - x0)) / weights,
        "comy": (sub.sum(axis=1, dtype=np.float64) @ np.arange(y1 - y0)) / weights,
    }
