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
