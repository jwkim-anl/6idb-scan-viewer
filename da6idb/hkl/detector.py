"""The scattering angle a detector pixel sees, from xrayutilities' geometry.

Two theta is the angle between the incident beam and the direction from the
sample to the pixel. The detector circle reads the angle of one pixel -- the
centre -- so a feature anywhere else is at a different angle, and on a
2D detector that offset is not along one axis only.

xrayutilities already knows where every pixel is once the detector arm has
turned: ``QConversion.getDetectorPos`` applies the rotations and
``VecAngle`` takes the angle against the beam. Nothing about the mounting is
assumed here -- circle directions, pixel directions, centre pixel, distance
and pixel size all arrive through the same converter the HKL conversion is
built on, so both answer from one description of the instrument.

A pixel's position is affine in its two indices, so the whole frame follows
exactly from a 2x2 corner, and a centre of mass lands on its own angle
rather than the nearest pixel's.
"""

from __future__ import annotations

import numpy as np

from da6idb.hkl.geometry import DETECTOR_MOTORS, HklGeometry

# psic_nu then psic_delta, as DETECTOR_MOTORS has them.
DELTA_CIRCLE = DETECTOR_MOTORS.index("psic_delta")


def circle_delta(geometry: HklGeometry) -> np.ndarray:
    """What the delta circle read on each frame.

    The angle of the centre pixel, which is what the motor is set to -- not
    the angle of whatever the detector was actually looking at.
    """
    return _per_frame(geometry)[DELTA_CIRCLE]


def _per_frame(geometry: HklGeometry) -> np.ndarray:
    """The detector circle readings, one column per frame."""
    angles = np.asarray(geometry.detector_angles, dtype=float)
    if angles.ndim != 2 or angles.shape[0] != len(DETECTOR_MOTORS):
        raise ValueError(
            f"expected {len(DETECTOR_MOTORS)} detector circles, "
            f"got {np.shape(geometry.detector_angles)}"
        )
    return np.broadcast_to(angles, (angles.shape[0], int(geometry.n_frames)))


def detector_basis(hxrd, detector_angles) -> tuple[np.ndarray, ...]:
    """Pixel (0, 0) and one step along each pixel axis, in lab coordinates.

    ``getDetectorPos`` returns the direction to each pixel with the arm
    rotations applied. That position is affine in the pixel indices, so
    these three vectors give any pixel, whole or fractional, exactly.
    """
    corner = np.asarray(
        hxrd.Ang2Q.getDetectorPos(*detector_angles, dim=2, roi=[0, 2, 0, 2]),
        dtype=float,
    )
    origin = corner[:, 0, 0]
    return origin, corner[:, 1, 0] - origin, corner[:, 0, 1] - origin


def two_theta_at(geometry: HklGeometry, x, y, hxrd=None) -> np.ndarray:
    """The scattering angle at pixel ``(x, y)`` on each frame, in degrees.

    *x* and *y* are detector pixels indexed as the frame is stored, and may
    be fractional -- a centre of mass usually is. Give one position, or one
    per frame.

    The detector basis is rebuilt only when the circles move, so a scan that
    leaves the detector still costs one call however many frames it has.
    """
    from xrayutilities.math import VecAngle

    if hxrd is None:
        from da6idb.hkl.convert import build_converter

        hxrd = build_converter(geometry)

    beam = np.asarray(geometry.primary_beam, dtype=float)
    angles = _per_frame(geometry)
    n_frames = angles.shape[1]
    x = np.broadcast_to(np.asarray(x, dtype=float), (n_frames,))
    y = np.broadcast_to(np.asarray(y, dtype=float), (n_frames,))

    out = np.empty(n_frames, dtype=float)
    seen: dict[tuple, tuple] = {}
    for frame in range(n_frames):
        position = angles[:, frame]
        key = tuple(np.round(position, 9))
        basis = seen.get(key)
        if basis is None:
            basis = seen[key] = detector_basis(hxrd, position)
        origin, step_x, step_y = basis
        here = origin + x[frame] * step_x + y[frame] * step_y
        out[frame] = float(VecAngle(here, beam, deg=True))
    return out
