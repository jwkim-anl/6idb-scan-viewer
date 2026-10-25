"""Cutting a plane through a gridded map.

Sampling the volume on a regular lattice in an arbitrary plane, averaging over
a slab of cells either side of it, and the basis that decides which way the
image's own axes run.

No Qt, no matplotlib. A notebook imports this directly.
"""

from __future__ import annotations

import numpy as np

from da6idb.hkl.grid import make_drawable  # noqa: F401

# How close a normal has to be to an axis before the slice counts as
# axis-aligned and the image is labelled H/K/L outright.
AXIS_TOLERANCE = 1e-6

# Cap on the sampled image, so an enormous map cannot ask for a 20000-pixel side.
MAX_PIXELS = 2000

def _axis_index(vector, tolerance=AXIS_TOLERANCE):
    """Which HKL axis this unit vector lies along, or None."""
    vector = np.asarray(vector, float)
    for axis in range(3):
        if abs(abs(vector[axis]) - 1.0) <= tolerance:
            return axis
    return None

def plane_basis(normal):
    """Two in-plane directions for *normal*, preferring the HKL axes.

    When the normal is along an axis the other two axes span the plane exactly,
    and they are returned in H, K, L order so an L slice is drawn H across and
    K up. Otherwise any orthonormal pair will do.
    """
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    axis = _axis_index(normal)
    if axis is not None:
        keep = [a for a in range(3) if a != axis]
        basis = np.eye(3)
        return basis[keep[0]], basis[keep[1]]
    # Start from whichever axis the normal leans on least.
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = seed - normal * float(np.dot(seed, normal))
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v / np.linalg.norm(v)

def plane_origin(centre, normal, offset=0.0, point=None):
    """Where the cutting plane sits, from an offset or from a point on it.

    An offset is measured from *centre* along the unit normal, which is fine
    for sweeping a plane through a map but awkward when you already know the
    H, K, L you want to cut through: naming the *point* places the plane
    directly, and the plane is then the one through it.

    Returns ``(origin, offset)`` either way -- the second being how far the
    plane ended up from *centre* -- so a viewer can always show both
    descriptions of the same plane, whichever was typed.
    """
    normal = np.asarray(normal, float)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("the plane normal is zero")
    normal = normal / length
    centre = np.asarray(centre, float)
    if point is None:
        return centre + normal * float(offset), float(offset)
    point = np.asarray(point, float)
    return point, float(np.dot(point - centre, normal))


def slab_offsets(layers):
    """Offsets, in cells, of a slab of *layers* planes centred on zero."""
    layers = max(1, int(layers))
    return np.arange(layers) - (layers - 1) / 2.0

def sample_volume(grid, points):
    """Trilinear values of *grid* at arbitrary HKL points; NaN off the grid."""
    from scipy.ndimage import map_coordinates

    origin = np.asarray(grid.origin, float)
    spacing = np.asarray(grid.spacing, float)
    spacing = np.where(spacing == 0, 1.0, spacing)
    dims = tuple(int(n) for n in grid.dimensions)
    index = (np.asarray(points, float).reshape(-1, 3) - origin) / spacing
    volume = np.asarray(grid.point_data["intensity"]).reshape(dims, order="F")
    return map_coordinates(volume, index.T, order=1, mode="constant", cval=np.nan)

def average_along_normal(grid, points, normal, layers=1, step=None):
    """Average *grid* over a slab of *layers* cells centred on *points*.

    A plane has no thickness, so a slice of one cell is as noisy as the data
    gets. Averaging a few cells either side along the normal trades a little
    resolution across the plane for a great deal less noise in it.
    """
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    if step is None:
        step = float(np.min(np.abs(np.asarray(grid.spacing, float))))
    points = np.asarray(points, float).reshape(-1, 3)
    if int(layers) <= 1:
        return np.ma.masked_invalid(sample_volume(grid, points))
    stack = [
        sample_volume(grid, points + normal * (offset * step))
        for offset in slab_offsets(layers)
    ]
    return np.ma.masked_invalid(np.vstack(stack)).mean(axis=0)

def line_cut(grid, start, end, samples=None, width=1, normal=None, step=None):
    """Intensity along the segment from *start* to *end*, both in HKL.

    The two points name the cut outright, which is what you want when the
    line runs between features whose H, K, L you know -- dragging one into
    place by eye cannot be written down or repeated.

    *width* averages that many parallel lines, spaced *step* apart and
    centred on the segment, the way a slab average thickens a plane. Their
    offset direction is perpendicular to the line and, when *normal* is
    given, inside that plane, so widening a cut on a slice stays on the
    slice instead of wandering out of it.

    Returns ``(distance, points, values)``: distance from *start* in HKL
    units, the ``(n, 3)`` H, K, L of each sample, and the intensity there,
    NaN wherever the scan did not reach.
    """
    start = np.asarray(start, float).reshape(3)
    end = np.asarray(end, float).reshape(3)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 0:
        raise ValueError("the two ends of the cut are the same point")
    unit = direction / length
    if step is None:
        step = float(np.min(np.abs(np.asarray(grid.spacing, float))))
    step = float(step) if step > 0 else 1.0

    count = int(samples) if samples else int(np.ceil(length / step)) + 1
    count = int(np.clip(count, 2, MAX_PIXELS))
    distance = np.linspace(0.0, length, count)
    points = start[None, :] + distance[:, None] * unit[None, :]

    width = max(1, int(width))
    if width == 1:
        values = np.ma.filled(
            np.ma.masked_invalid(sample_volume(grid, points)), np.nan)
        return distance, points, values

    if normal is not None:
        across = np.cross(np.asarray(normal, float), unit)
    else:
        # Any perpendicular will do: start from the axis the line leans on
        # least, so the subtraction cannot collapse to zero.
        seed = np.eye(3)[int(np.argmin(np.abs(unit)))]
        across = seed - unit * float(np.dot(seed, unit))
    size = float(np.linalg.norm(across))
    if size < 1e-12:
        raise ValueError("the cut runs along its own plane normal")
    across = across / size

    stack = [
        sample_volume(grid, points + across[None, :] * (offset * step))
        for offset in slab_offsets(width)
    ]
    # Masked, not nanmean: where the line leaves the measured volume every
    # offset is NaN, and nanmean warns on each one.
    values = np.ma.filled(
        np.ma.masked_invalid(np.vstack(stack)).mean(axis=0), np.nan)
    return distance, points, np.asarray(values, float)


def sample_plane(grid, normal, origin, pixels=None, layers=1):
    """Resample *grid* on the plane. Returns (image, us, vs, u, v, origin).

    ``image`` is indexed [row, column], columns running along ``u`` and rows
    along ``v``; ``us`` and ``vs`` are the sample positions as distances from
    ``origin`` along each. Points off the grid come back NaN. With *layers*
    above one the plane is averaged over that many cells along its normal.
    """
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    origin = np.asarray(origin, float)
    u, v = plane_basis(normal)

    grid_origin = np.asarray(grid.origin, float)
    spacing = np.asarray(grid.spacing, float)
    dims = np.asarray(grid.dimensions, int)
    spacing = np.where(spacing == 0, 1.0, spacing)

    # How far the volume reaches along u and v, from its eight corners.
    lows = grid_origin
    highs = grid_origin + (dims - 1) * spacing
    corners = np.array(np.meshgrid(*zip(lows, highs), indexing="ij")).reshape(3, -1).T
    offsets = corners - origin
    along_u = offsets @ u
    along_v = offsets @ v
    step = float(np.min(spacing)) if pixels is None else None

    def axis_samples(low, high):
        size = max(high - low, float(np.min(spacing)))
        count = int(np.ceil(size / step)) + 1 if step else int(pixels)
        count = int(np.clip(count, 2, MAX_PIXELS))
        return np.linspace(low, high, count)

    us = axis_samples(along_u.min(), along_u.max())
    vs = axis_samples(along_v.min(), along_v.max())

    # Every pixel's HKL, then the same in fractional grid indices.
    points = (
        origin
        + us[None, :, None] * u[None, None, :]
        + vs[:, None, None] * v[None, None, :]
    )
    shape = points.shape[:2]
    image = np.ma.filled(
        average_along_normal(grid, points, normal, layers=layers), np.nan
    ).reshape(shape)

    return image, us, vs, u, v, origin


def plane_array(grid, normal=(0, 0, 1), offset=0.0, point=None, layers=1,
                pixels=None):
    """A plane through *grid* as a labelled 2D array.

    :func:`sample_plane` with its answer named: an axis-aligned plane is
    labelled H, K or L outright and carries real reciprocal-space
    coordinates, so a feature's position can be read straight off it. A
    tilted one keeps distances along its own two in-plane directions,
    since no single HKL axis describes them.

    The attributes record both ways of placing the plane -- the *offset*
    it ended up at and the *point* it passes through -- whichever was
    given. Unmeasured cells are NaN, as everywhere else.
    """
    import xarray as xr

    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    origin, offset = plane_origin(grid.center, normal, offset=offset,
                                  point=point)
    image, us, vs, u, v, corner = sample_plane(grid, normal, origin,
                                               pixels=pixels, layers=layers)

    def axis(direction, samples):
        index = _axis_index(direction)
        if index is None:
            return ("along [%.3f %.3f %.3f]" % tuple(direction), samples)
        return ("HKL"[index], corner[index] + samples * float(direction[index]))

    x_name, x_values = axis(u, us)
    y_name, y_values = axis(v, vs)
    return xr.DataArray(
        image, dims=(y_name, x_name),
        coords={y_name: y_values, x_name: x_values}, name="intensity",
        attrs={"normal": list(normal), "offset": float(offset),
               "point": [float(c) for c in origin], "layers": int(layers)})
