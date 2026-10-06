"""Turning a detector pixel into the angle the detector circle would need.

A feature does not generally sit at the centre of the detector, so the circle
reading is not its scattering angle: the pixel's own offset has to be added.
These scans are vertical scattering, so the circle in question is delta, and
the offset is the angle the pixel subtends at the sample.

Which way round that goes is not assumed here. Rotating the detector about a
circle whose axis is *d* carries a point on the beam *b* along ``d x b``, so
the pixel direction with a component along that is the one delta moves the
image across, and the sign of that component says whether a higher pixel
index means a higher angle or a lower one. Both the circle axis and the pixel
directions come out of the file, so a beamline that mounts its detector the
other way round still gets the right answer.

No Qt, no xrayutilities. A notebook imports this directly.
"""

from __future__ import annotations

import numpy as np

from da6idb.hkl.geometry import DETECTOR_MOTORS, HklGeometry

# psic_nu then psic_delta, as DETECTOR_MOTORS has them.
DELTA_CIRCLE = DETECTOR_MOTORS.index("psic_delta")

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def direction_vector(text: str) -> np.ndarray:
    """A direction the files write as ``"z+"`` or ``"y-"``, as a unit vector."""
    name = str(text).strip().lower()
    if not name or name[0] not in _AXES:
        raise ValueError(f"{text!r} is not a direction like 'z+' or 'y-'")
    sign = -1.0 if name[1:2] == "-" else 1.0
    return sign * np.array(_AXES[name[0]], dtype=float)


def delta_pixel_axis(geometry: HklGeometry) -> tuple[int, float]:
    """Which frame axis delta sweeps the image along, and which way.

    Returns ``(axis, sign)``. *axis* indexes the frame as it is stored, so 0
    goes with ``comx`` and 1 with ``comy``; *sign* is ``+1`` when a higher
    pixel index means a higher delta and ``-1`` when it means a lower one.
    """
    beam = np.asarray(geometry.primary_beam, dtype=float)
    norm = np.linalg.norm(beam)
    if norm == 0:
        raise ValueError("the primary beam direction is zero")
    sweep = np.cross(direction_vector(geometry.detector_dirs[DELTA_CIRCLE]),
                     beam / norm)
    overlap = [float(np.dot(direction_vector(p), sweep))
               for p in geometry.pixel_direction]
    axis = int(np.argmax(np.abs(overlap)))
    if abs(overlap[axis]) < 0.5:
        raise ValueError(
            "neither detector axis follows the delta circle: directions "
            f"{tuple(geometry.pixel_direction)} against {sweep.tolist()}"
        )
    return axis, float(np.sign(overlap[axis]))


def pixel_offset(geometry: HklGeometry, pixel) -> np.ndarray:
    """Degrees between *pixel* and the pixel the delta circle reads.

    *pixel* is a position along the axis :func:`delta_pixel_axis` picks, in
    detector pixels. The distance and the detector size are both read from
    the same ``DETECTOR_SETUP``, so whatever length unit they are in cancels.
    """
    axis, sign = delta_pixel_axis(geometry)
    offset = (np.asarray(pixel, dtype=float)
              - float(geometry.center_pixel[axis])) * geometry.pixel_widths()[axis]
    return sign * np.degrees(np.arctan2(offset, float(geometry.distance)))


def circle_delta(geometry: HklGeometry) -> np.ndarray:
    """What the delta circle read on each frame."""
    angles = np.asarray(geometry.detector_angles, dtype=float)
    if angles.ndim != 2 or angles.shape[0] <= DELTA_CIRCLE:
        raise ValueError("this geometry records no delta circle")
    return np.broadcast_to(angles[DELTA_CIRCLE].ravel(),
                           (int(geometry.n_frames),)).astype(float)


def delta_angles(geometry: HklGeometry, pixel) -> np.ndarray:
    """The delta a feature sitting at *pixel* is actually scattered through.

    The circle's own reading for each frame, plus that frame's pixel offset.
    *pixel* is one position or one per frame.
    """
    return circle_delta(geometry) + pixel_offset(geometry, pixel)
