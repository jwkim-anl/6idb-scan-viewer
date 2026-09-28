"""Bin a pva detector stack onto a regular HKL grid and save it as ``.vti``.

The viewers draw reciprocal space as a point cloud, which keeps every pixel's
exact position but is expensive to rebuild and awkward to analyse. This turns
the same data into a regular grid -- mean intensity per cell, unmeasured cells
left as NaN -- written as a VTK ImageData file that ParaView, pyvista or any
analysis script can open directly.

From the command line::

    python hkl_grid.py pva_tise2_2_00152.h5
    python hkl_grid.py pva_tise2_2_00152.h5 --range 0.93 1.11 -0.60 -0.36 7.39 7.66

From analysis code::

    from hkl_grid import grid_to_vti
    path = grid_to_vti("pva_tise2_2_00152.h5",
                       hkl_range=(0.93, 1.11, -0.60, -0.36, 7.39, 7.66))

The master file is found beside the image file unless one is passed.

Resolution
----------
Two different numbers matter here, and using one for the other is the easy
mistake.

At every pixel the three sampling directions -- along a detector row, along a
column, and from one frame to the next -- span a small parallelepiped in
reciprocal space.

- Its **volume**, the Jacobian determinant ``|(drow x dcol) . dframe|``, is the
  HKL volume each sample speaks for. A cell of that volume holds about one
  sample, so the default cell is the cube of it.
- Its **axis-aligned extent** is how far one sample's influence reaches along
  each axis. That is the footprint each sample is spread over, not the cell
  size.

On the example scan the volume gives a cell of 6.7e-4 while the extent is
3.6e-4, 1.3e-3, 5.6e-4 -- very different, because the parallelepiped is
strongly skewed. Consecutive frames are 1.2e-3 apart measured perpendicular to
the detector surface while pixels within a frame are 4.9e-4 apart, so the
frames are about 2.5x further apart than the detector samples.

That gap is why simple nearest-cell binning cannot be made fine. At a 6.7e-4
cell it leaves roughly one empty layer between every pair of frame surfaces:
83% as many empty cells inside the shell as filled ones. Spreading each sample
over its footprint instead -- fuzzy gridding, via
:class:`xrayutilities.FuzzyGridder3D` -- drops that to 3.3% at the same cell
size, and is several times faster. So the grid is fine *and* continuous.

Attenuator
----------
A scan that changes its filter mid-way records frames on different scales, and
the map steps wherever the attenuator moved -- on one test scan the same angle
was measured twice, 44 times brighter through one filter than the other.
Dividing each frame by its transmission puts them all on the unattenuated
beam, and the two readings then agree to a few percent. This happens by
default when the transmission varies, and not at all when it does not.

Unmeasured cells
----------------
Most of the bounding box is never visited -- about 23% of cells hold data on
the example scan -- because each frame maps to a curved surface and the swept
volume is a skewed shell, not a box. That is geometry, not a resolution
problem, and coarsening does not fix it. Unmeasured cells are written as NaN so
analysis can mask them; anything drawing the grid must replace them first,
because NaN handed to a GPU volume mapper lights up the whole empty region.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import xrayutilities as xu

from volume_style import make_drawable  # noqa: F401  (re-exported)

from hkl_viewer import HklGeometry
from hkl_viewer import IMAGE_DATA_PATH
from hkl_viewer import build_converter
from hkl_viewer import frame_hkl
from hkl_viewer import read_geometry

import h5py

# The gridder holds a float64 value and weight for every cell, so 100 M cells
# is about 1.6 GB. Beyond this the cell is coarsened rather than the machine
# swapped; raise it if you have the memory and want the detail.
MAX_CELLS = 100_000_000


class GridCancelled(Exception):
    """Raised when a progress callback asked to stop, so nothing is written."""


def geometry_fingerprint(geometry: HklGeometry) -> str:
    """A short digest of everything that decides where a pixel lands.

    Stored in the ``.vti`` so a cached grid is not reused after the UB, the
    energy, the detector geometry or the angles have been edited -- the HKL
    range alone would not notice.
    """
    parts = [
        repr(geometry.energy_kev),
        repr(geometry.distance),
        repr(tuple(geometry.center_pixel)),
        repr(tuple(geometry.pixel_direction)),
        repr(tuple(geometry.detector_size)),
        repr(tuple(geometry.primary_beam)),
        repr(tuple(geometry.inplane)),
        repr(tuple(geometry.surface)),
        repr(list(geometry.sample_dirs)),
        repr(list(geometry.detector_dirs)),
        np.asarray(geometry.ub, dtype=float).tobytes().hex(),
        np.asarray(geometry.sample_angles, dtype=float).tobytes().hex(),
        np.asarray(geometry.detector_angles, dtype=float).tobytes().hex(),
        repr(bool(geometry.normalize_transmission)),
        ""
        if geometry.transmission is None
        else np.asarray(geometry.transmission, dtype=float).tobytes().hex(),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _sampling_vectors(geometry: HklGeometry, frame=None, stride: int = 8):
    """The three sampling directions at many pixels of one frame.

    Returns ``(row, col, between)``, each ``(n_pixels, 3)`` in HKL, or
    ``between=None`` for a single-frame scan that sweeps no volume.
    """
    hxrd = build_converter(geometry)
    if frame is None:
        frame = geometry.n_frames // 2
    frame = int(np.clip(frame, 0, geometry.n_frames - 1))

    here = np.stack([np.asarray(a) for a in frame_hkl(geometry, hxrd, frame)])
    row = (here[:, 1:, :] - here[:, :-1, :])[:, ::stride, ::stride]
    col = (here[:, :, 1:] - here[:, :, :-1])[:, ::stride, ::stride]
    rows = min(row.shape[1], col.shape[1])
    cols = min(row.shape[2], col.shape[2])
    row = row[:, :rows, :cols].reshape(3, -1).T
    col = col[:, :rows, :cols].reshape(3, -1).T

    other_frame = frame + 1 if frame + 1 < geometry.n_frames else frame - 1
    if other_frame < 0:
        return row, col, None
    other = np.stack([np.asarray(a) for a in frame_hkl(geometry, hxrd, other_frame)])
    between = (other - here)[:, ::stride, ::stride][:, :rows, :cols].reshape(3, -1).T
    return row, col, between


def sampling_cell(geometry: HklGeometry, frame=None, stride: int = 8):
    """What the sampling supports, in HKL units.

    Returns ``(volume_per_sample, frame_gap, in_plane)``: the Jacobian volume
    each sample speaks for, the perpendicular separation between consecutive
    frames, and the spacing between neighbouring pixels within a frame.
    """
    row, col, between = _sampling_vectors(geometry, frame, stride)
    normal = np.cross(row, col)                      # area vector of one pixel
    area = np.linalg.norm(normal, axis=1)
    good = area > 0
    in_plane = float(np.median(np.sqrt(area[good]))) if np.any(good) else float("nan")
    if between is None:
        return float("nan"), float("nan"), in_plane
    # |det J| = |(row x col) . frame|; over the in-plane area that leaves the
    # part of the frame step which actually departs from the surface.
    volume = np.abs(np.einsum("ij,ij->i", normal, between))
    gap = float(np.median(volume[good] / area[good])) if np.any(good) else float("nan")
    return float(np.median(volume)), gap, in_plane


def sampling_footprint(geometry: HklGeometry, frame=None, stride: int = 8) -> np.ndarray:
    """How far one sample reaches along each axis -- the fuzzy spread width.

    The axis-aligned extent of the sampling parallelepiped. This is emphatically
    not the cell size: the parallelepiped is skewed, so its bounding box is
    several times its volume.
    """
    row, col, between = _sampling_vectors(geometry, frame, stride)
    directions = [row, col] + ([] if between is None else [between])
    return np.array(
        [
            max(float(np.median(np.abs(d[:, axis]))) for d in directions)
            for axis in range(3)
        ]
    )


def natural_step(geometry: HklGeometry, oversample: float = 1.0, frame=None) -> np.ndarray:
    """The cell size the sampling supports: a cube holding about one sample.

    *oversample* scales it -- above 1 for fewer, better-filled cells, below 1
    for a finer grid. Because samples are spread over their footprint, a finer
    cell stays continuous instead of breaking into layers.
    """
    volume, _gap, in_plane = sampling_cell(geometry, frame=frame)
    if not np.isfinite(volume) or volume <= 0:
        side = in_plane                      # a single frame, or a degenerate one
    else:
        side = volume ** (1.0 / 3.0)
    return np.full(3, side * float(oversample))


def _resolve_range(geometry, hkl_range, hxrd, sample_frames=8):
    """The volume to grid: what was asked for, else the extent of the data."""
    if hkl_range is not None:
        values = np.array(hkl_range, dtype=float)
        return values[0::2], values[1::2]
    indices = np.unique(
        np.linspace(0, geometry.n_frames - 1, min(sample_frames, geometry.n_frames))
        .astype(int)
    )
    lows, highs = [], []
    for index in indices:
        hkl = np.stack([np.asarray(a) for a in frame_hkl(geometry, hxrd, index)])
        flat = hkl.reshape(3, -1)
        lows.append(flat.min(axis=1))
        highs.append(flat.max(axis=1))
    return np.min(lows, axis=0), np.max(highs, axis=0)


def measured_range(geometry: HklGeometry, sample_frames: int = 8):
    """The HKL box the scan actually reaches, as ``(low, high)`` triples.

    Useful for choosing a range to give a whole batch before gridding any of it.
    """
    return _resolve_range(geometry, None, build_converter(geometry), sample_frames)


def default_output(image: Path, hkl_range=None, oversample: float = 1.0) -> Path:
    """Where the grid for this file, range and detail level is kept."""
    image = Path(image)
    detail = "" if float(oversample) == 1.0 else f"_d{float(oversample):g}"
    if hkl_range is None:
        return image.with_name(f"{image.stem}_hkl_grid{detail}.vti")
    tag = "_".join(f"{v:.4g}" for v in hkl_range)
    return image.with_name(f"{image.stem}_hkl_{tag}{detail}.vti")


def matches_range(path: Path, hkl_range, tolerance=1e-6) -> bool:
    """Whether an existing .vti was made for this HKL range."""
    try:
        import pyvista as pv

        stored = pv.read(str(path)).field_data.get("hkl_range")
    except Exception:
        return False
    if stored is None:
        return False
    if hkl_range is None:
        return True
    return bool(
        np.allclose(np.asarray(stored, float), np.asarray(hkl_range, float), atol=tolerance)
    )


def grid_to_vti(
    image: str | Path,
    master: str | Path | None = None,
    hkl_range=None,
    output: str | Path | None = None,
    step=None,
    width=None,
    oversample: float = 1.0,
    first: int = 0,
    last: int | None = None,
    frame_step: int = 1,
    max_cells: int = MAX_CELLS,
    reuse: bool = True,
    progress=None,
    geometry: HklGeometry | None = None,
    normalize_transmission: bool | None = None,
) -> Path:
    """Bin *image* onto a regular HKL grid and write it as a ``.vti``.

    ``hkl_range`` is ``(h_min, h_max, k_min, k_max, l_min, l_max)``; without it
    the whole measured extent is used. ``step`` overrides the automatic cell
    size and ``width`` the fuzzy footprint. Returns the path written -- or the
    existing one, when *reuse* is set and a grid for the same range is on disk.
    """
    import pyvista as pv

    image = Path(image)
    out = (
        Path(output) if output is not None
        else default_output(image, hkl_range, oversample)
    )
    if reuse and out.is_file() and matches_range(out, hkl_range):
        return out

    if geometry is None:
        geometry = read_geometry(image, Path(master) if master is not None else None)
    if normalize_transmission is not None:
        geometry.normalize_transmission = bool(normalize_transmission)
    hxrd = build_converter(geometry)
    low, high = _resolve_range(geometry, hkl_range, hxrd)
    span = np.maximum(high - low, 1e-12)

    if step is None:
        step = natural_step(geometry, oversample=oversample)
    step = np.asarray(step, dtype=float)
    step = np.where(np.isfinite(step) & (step > 0), step, span / 128.0)
    if width is None:
        width = sampling_footprint(geometry)
    width = np.asarray(width, dtype=float)
    width = np.where(np.isfinite(width) & (width > 0), width, step)

    # The gridder's axes run lo..hi inclusive over n points, so n-1 gaps.
    shape = np.maximum(np.round(span / step).astype(int) + 1, 2)
    if shape.prod() > max_cells:
        factor = float(np.cbrt(shape.prod() / max_cells))
        step = step * factor
        shape = np.maximum(np.round(span / step).astype(int) + 1, 2)

    gridder = xu.FuzzyGridder3D(int(shape[0]), int(shape[1]), int(shape[2]))
    gridder.dataRange(
        low[0], high[0], low[1], high[1], low[2], high[2], fixed=True
    )
    gridder.KeepData(True)                   # accumulate across the frame loop

    last = geometry.n_frames - 1 if last is None else min(last, geometry.n_frames - 1)
    frames = range(max(0, first), last + 1, max(1, frame_step))
    rows, columns = geometry.frame_shape
    spread = tuple(float(w) for w in width)

    with h5py.File(image, "r") as handle:
        stack = handle[IMAGE_DATA_PATH]
        for number, frame in enumerate(frames):
            values = np.asarray(stack[frame], dtype=np.float64).reshape(rows, columns)
            # Frames taken through different filters are not on the same scale;
            # dividing by the transmission puts them on the unattenuated beam,
            # so the map does not step where the attenuator moved.
            scale = geometry.frame_scale(frame)
            if scale != 1.0:
                values = values * scale
            hkl = frame_hkl(geometry, hxrd, frame)
            gridder(
                np.asarray(hkl[0], dtype=np.float64).ravel(),
                np.asarray(hkl[1], dtype=np.float64).ravel(),
                np.asarray(hkl[2], dtype=np.float64).ravel(),
                values.ravel(),
                width=spread,
            )
            if progress is not None and not progress(number + 1, len(frames)):
                # Writing now would cache a half-filled grid under a name that
                # claims to cover the whole range.
                raise GridCancelled(f"stopped after {number + 1} of {len(frames)} frames")

    mean = np.asarray(gridder.data, dtype=np.float32)
    weight = np.asarray(getattr(gridder, "_gnorm", None), dtype=np.float32)
    if weight.shape != mean.shape:               # normalisation turned off
        weight = (mean != 0).astype(np.float32)
    measured = weight > 0
    mean[~measured] = np.nan

    axes = (
        np.asarray(gridder.xaxis, dtype=float),
        np.asarray(gridder.yaxis, dtype=float),
        np.asarray(gridder.zaxis, dtype=float),
    )
    grid = pv.ImageData(
        dimensions=tuple(int(n) for n in mean.shape),
        spacing=tuple(
            float(a[1] - a[0]) if a.size > 1 else float(step[i]) for i, a in enumerate(axes)
        ),
        origin=tuple(float(a[0]) for a in axes),
    )
    # VTK runs the first axis fastest; the gridder's array is (H, K, L) in C order.
    grid.point_data["intensity"] = mean.ravel(order="F")
    # The accumulated weight is not saved: NaN already marks the cells nothing
    # reached, and keeping it more than doubles the file and the time to read
    # it back (2.9 M distinct floats compress badly, 215 MB against 94 MB).
    requested = (
        np.asarray(hkl_range, dtype=float)
        if hkl_range is not None
        else np.array([low[0], high[0], low[1], high[1], low[2], high[2]])
    )
    grid.field_data["hkl_range"] = requested
    grid.field_data["step"] = np.asarray(grid.spacing, dtype=float)
    grid.field_data["width"] = width
    grid.field_data["source_image"] = np.array([image.name])
    grid.field_data["source_master"] = np.array(
        [geometry.master_path.name if geometry.master_path else ""]
    )
    grid.field_data["frames"] = np.array([len(frames)])
    grid.field_data["frame_range"] = np.array(
        [first, -1 if last is None else last, frame_step]
    )
    grid.field_data["geometry"] = np.array([geometry_fingerprint(geometry)])
    grid.field_data["oversample"] = np.array([float(oversample)])
    grid.field_data["transmission_corrected"] = np.array(
        [1 if geometry.normalize_transmission else 0]
    )
    grid.field_data["filled_cells"] = np.array([int(measured.sum()), int(measured.size)])
    grid.save(str(out))
    return out


def ensure_grid(
    image: str | Path,
    master: str | Path | None = None,
    hkl_range=None,
    output: str | Path | None = None,
    reuse: bool = True,
    geometry: HklGeometry | None = None,
    oversample: float = 1.0,
    **kwargs,
):
    """Return ``(grid, path, regridded)`` for this file and HKL range.

    A cached ``.vti`` is reused when the HKL range, the frame range and the
    geometry it was made for all match what is being asked for now; otherwise
    the data is gridded and written. Unlike calling :func:`grid_to_vti` then
    :func:`load_vti`, the cached file is read once.
    """
    image = Path(image)
    out = (
        Path(output) if output is not None
        else default_output(image, hkl_range, oversample)
    )
    if reuse and out.is_file():
        try:
            grid = load_vti(out)
            stored = grid.field_data.get("hkl_range")
            same_range = stored is not None and (
                hkl_range is None
                or np.allclose(
                    np.asarray(stored, float), np.asarray(hkl_range, float), atol=1e-6
                )
            )
            want = np.array(
                [
                    kwargs.get("first", 0),
                    -1 if kwargs.get("last") is None else kwargs["last"],
                    kwargs.get("frame_step", 1),
                ]
            )
            got = grid.field_data.get("frame_range")
            same_frames = got is None or np.array_equal(np.asarray(got), want)
            detail = grid.field_data.get("oversample")
            same_detail = detail is None or float(np.asarray(detail).ravel()[0]) == float(oversample)
            mark = grid.field_data.get("geometry")
            same_geometry = (
                geometry is None
                or mark is None
                or str(np.asarray(mark).ravel()[0]) == geometry_fingerprint(geometry)
            )
            if same_range and same_frames and same_geometry and same_detail:
                return grid, out, False
        except Exception:
            pass  # unreadable or half-written: grid it again
    path = grid_to_vti(image, master, hkl_range, out, reuse=False,
                       geometry=geometry, oversample=oversample, **kwargs)
    return load_vti(path), path, True


def load_vti(path: str | Path):
    """Read a grid written by :func:`grid_to_vti`."""
    import pyvista as pv

    return pv.read(str(path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="pva*.h5 detector stack")
    parser.add_argument("--master", help="master .hdf (found beside the image by default)")
    parser.add_argument(
        "--range", nargs=6, type=float, metavar=("Hmin", "Hmax", "Kmin", "Kmax", "Lmin", "Lmax")
    )
    parser.add_argument("--output", help="output .vti (named after the image by default)")
    parser.add_argument(
        "--oversample", type=float, default=1.0,
        help="scale the automatic cell size: <1 finer, >1 coarser (default 1)",
    )
    parser.add_argument(
        "--no-transmission",
        action="store_true",
        help="do not divide intensities by the attenuator transmission "
             "(it is divided by default on a scan whose filter moved)",
    )
    parser.add_argument("--max-cells", type=int, default=MAX_CELLS)
    parser.add_argument("--force", action="store_true", help="re-grid even if a cached .vti exists")
    args = parser.parse_args(argv)

    out = grid_to_vti(
        args.image,
        master=args.master,
        hkl_range=tuple(args.range) if args.range else None,
        output=args.output,
        oversample=args.oversample,
        normalize_transmission=False if args.no_transmission else None,
        max_cells=args.max_cells,
        reuse=not args.force,
        progress=lambda done, total: (
            print(f"\r  frame {done}/{total}", end="", flush=True) or True
        ),
    )
    print()
    grid = load_vti(out)
    filled, total = grid.field_data["filled_cells"]
    spacing = ", ".join(f"{float(v):.3e}" for v in grid.spacing)
    width = ", ".join(f"{float(v):.3e}" for v in grid.field_data["width"])
    print(f"wrote {out}")
    print(f"  dimensions {grid.dimensions}  cell ({spacing})")
    print(f"  fuzzy footprint ({width})")
    print(f"  {filled:,} of {total:,} cells measured ({100 * filled / total:.1f}%)")
    if int(np.asarray(grid.field_data["transmission_corrected"]).ravel()[0]):
        print("  intensities divided by the attenuator transmission")
    return 0


if __name__ == "__main__":
    sys.exit(main())
