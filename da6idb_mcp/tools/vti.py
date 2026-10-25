"""Gridded maps already on disk: what the VTI viewer reads.

`hkl_*` grids a scan; these read the ``.vti`` that came out, whoever
wrote it -- an earlier `hkl_grid`, a `batch_grid_scans` run, or the
windows themselves. A map is a file, so this namespace needs no scan
numbers and no detector files, and works just as well on a map copied
from another machine.

The tools are the same shapes as the HKL ones -- a plane, a line -- cut
through a loaded volume instead of one gridded on demand, which makes
them fast: the file is read once and kept. `vti_export_image` and
`vti_export_cut` write the same text, CSV, NumPy or HDF5 files the
viewer's export buttons do, so a cut can leave here and go into a paper.

There is no 3D render. That needs an OpenGL context the server has not
got; `viewer_open("vti", path)` opens the real window for one.
"""

from typing import Optional

from da6idb_mcp import cache, render, resolve
from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool


def _axis_label(name):
    return f"{name} [r.l.u.]" if name in ("H", "K", "L") else str(name)


def _provenance(path, extra=()):
    """The header every exported file carries: where the numbers came from."""
    return [f"source: {path}", *extra]


def _stats(values):
    """Min, max, mean over the measured cells alone."""
    import numpy as np

    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"measured": 0}
    return {"measured": int(finite.size), "min": float(finite.min()),
            "max": float(finite.max()), "mean": float(finite.mean())}


#: Said when a plane or a line came back entirely NaN, which almost always
#: means it was placed in the coordinates of a different map.
NOTHING_MEASURED = ("nothing measured anywhere on it -- this is usually a "
                    "plane or a line placed outside the map; vti_info gives "
                    "its H, K, L bounds")


@tool()
def vti_list(where: Optional[str] = None) -> Reply:
    """Every gridded map in a folder, with its size on disk.

    Looks in the folder itself and in its ``output`` subfolder, which is
    where `batch_grid_scans` puts a run's maps. The names that come back
    are what `vti_info`, `vti_slice` and the rest take.
    """
    folder = resolve.folder(where)
    found = resolve.vti_files(folder)
    files = [{"name": p.name,
              "path": str(p),
              "megabytes": round(p.stat().st_size / 1e6, 1)}
             for p in found]
    return Reply(data={"folder": str(folder), "count": len(files),
                       "files": files},
                 note="" if files else "nothing gridded here yet; hkl_grid or "
                                       "batch_grid_scans writes these")


@tool()
def vti_info(path: str, where: Optional[str] = None) -> Reply:
    """A map's lattice and everything the gridding run recorded in it.

    The bounds are the H, K, L box it covers and the spacing is its cell,
    which together say whether a slice of it will resolve what you are
    after. ``attributes`` is the field data `hkl_grid` wrote -- the range
    asked for, the frames used, the oversample, whether the attenuator
    correction was applied -- so a map found months later still explains
    itself.
    """
    import numpy as np

    file = resolve.vti(path, where)
    grid = cache.grid(file)
    values = np.asarray(grid.point_data["intensity"])

    attributes = {}
    for name in grid.field_data.keys():
        entry = np.asarray(grid.field_data[name]).ravel()
        attributes[name] = (entry.tolist() if entry.size > 1
                            else render.to_jsonable(entry[0]))

    dims = [int(n) for n in grid.dimensions]
    return Reply(data={
        "path": str(file),
        "dimensions": dims,
        "cells": int(np.prod(dims)),
        "spacing": [float(s) for s in grid.spacing],
        "origin": [float(o) for o in grid.origin],
        "bounds": {axis: [float(grid.bounds[2 * i]),
                          float(grid.bounds[2 * i + 1])]
                   for i, axis in enumerate("HKL")},
        "centre": [float(c) for c in grid.center],
        "intensity": _stats(values),
        "attributes": attributes,
    }, note="dimensions run H, K, L; unmeasured cells are NaN")


def _slice_of(file, normal, offset, point, layers):
    """The plane itself, as a labelled array -- shared by slice and export."""
    from da6idb.hkl.slice import plane_array

    aim = [0.0, 0.0, 1.0] if normal is None else [float(v) for v in normal]
    return plane_array(cache.grid(file), normal=aim, offset=float(offset),
                       point=point, layers=int(layers))


@tool()
def vti_slice(path: str, normal: Optional[list] = None, offset: float = 0.0,
              point: Optional[list] = None, layers: int = 1, log: bool = True,
              cmap: str = "viridis", where: Optional[str] = None) -> Reply:
    """A plane through a map on disk, as a picture and as numbers.

    `hkl_slice` without the gridding: the same plane, cut through a file
    that already exists, so it answers in well under a second however
    long the map took to build.

    *normal* is an H, K, L direction, ``[0, 0, 1]`` by default -- the
    constant-L plane, drawn H across and K up. Place it by *offset* from
    the middle of the map, or by *point*, an H, K, L it must pass
    through. *layers* averages that many cells of thickness along the
    normal, which is how a weak feature is pulled out of the noise.
    """
    import numpy as np

    file = resolve.vti(path, where)
    array = _slice_of(file, normal, offset, point, layers)
    y_name, x_name = (str(d) for d in array.dims)
    values = array.values

    fig = render.plane(
        values, array[x_name].values, array[y_name].values,
        x_label=_axis_label(x_name), y_label=_axis_label(y_name),
        title=(f"{file.name} — normal {array.attrs['normal']}, "
               f"offset {array.attrs['offset']:.4g}, {layers} layer(s)"),
        log=log, cmap=cmap)

    peak = {}
    if np.isfinite(values).any():
        row, column = np.unravel_index(int(np.nanargmax(values)), values.shape)
        peak = {x_name: float(array[x_name].values[column]),
                y_name: float(array[y_name].values[row]),
                "intensity": float(values[row, column])}

    return Reply(data={
        "path": str(file),
        "normal": array.attrs["normal"],
        "offset": array.attrs["offset"],
        "through": array.attrs["point"],
        "layers": array.attrs["layers"],
        "axes": {"x": x_name, "y": y_name},
        "shape": list(values.shape),
        "intensity": _stats(values),
        "brightest": peak,
    }, figure=fig,
        note=NOTHING_MEASURED if not peak else
             "brightest is in HKL; pass it to vti_cut as an end, or back "
             "here as point=")


@tool()
def vti_cut(path: str, start: list, end: list, samples: Optional[int] = None,
            width: int = 1, normal: Optional[list] = None, log: bool = False,
            where: Optional[str] = None) -> Reply:
    """Intensity along the line between two H, K, L points of a map.

    Named by its two ends rather than dragged across a window, so the
    same cut can be run again on the next map and written down. *width*
    averages that many parallel lines either side; give the slice's
    *normal* too and that widening stays in the plane instead of
    wandering out of it.
    """
    import numpy as np

    from da6idb.hkl.slice import line_cut

    file = resolve.vti(path, where)
    distance, points, values = line_cut(
        cache.grid(file), start, end, samples=samples, width=int(width),
        normal=normal)

    fig, ax = render.figure()
    ax.plot(distance, values, lw=1.3)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("distance from start [r.l.u.]")
    ax.set_ylabel("intensity")
    ax.grid(alpha=0.3)
    ax.set_title(f"{file.name} — {list(start)} to {list(end)}",
                 fontsize="medium")

    peak = {}
    if np.isfinite(values).any():
        top = int(np.nanargmax(values))
        peak = {"distance": float(distance[top]),
                "H": float(points[top, 0]), "K": float(points[top, 1]),
                "L": float(points[top, 2]), "intensity": float(values[top])}

    return Reply(data={
        "path": str(file),
        "start": [float(v) for v in start], "end": [float(v) for v in end],
        "width": int(width),
        "samples": int(distance.size),
        "length": float(distance[-1]),
        "intensity": _stats(values),
        "brightest": peak,
        "distance": distance,
        "values": values,
    }, figure=fig,
        note=NOTHING_MEASURED if not peak else
             "NaN where the line left the measured volume")


@tool()
def vti_export_image(path: str, output: Optional[str] = None,
                     normal: Optional[list] = None, offset: float = 0.0,
                     point: Optional[list] = None, layers: int = 1,
                     where: Optional[str] = None) -> Reply:
    """Write a plane out as a file, the way the viewer's export button does.

    The format follows *output*'s extension: ``.txt``/``.dat`` a plain
    matrix, ``.csv`` the same with the axes down the first row and
    column, ``.npz`` a NumPy archive, ``.h5``/``.hdf5`` a dataset each
    for intensity, x and y. Every one carries a header saying which map
    and which plane it came from.

    The plane is placed exactly as in `vti_slice`. Without *output* the
    file lands in the server's scratch folder and the reply says where.
    """
    from da6idb import export

    file = resolve.vti(path, where)
    array = _slice_of(file, normal, offset, point, layers)
    target = resolve.artefact(output, ".txt", stem=f"{file.stem}_slice")

    written = export.write_dataarray(
        array, target,
        description=_provenance(file, [
            f"normal: {array.attrs['normal']}",
            f"offset: {array.attrs['offset']:.6g}",
            f"through: {array.attrs['point']}",
            f"layers averaged: {array.attrs['layers']}",
        ]))

    y_name, x_name = (str(d) for d in array.dims)
    return Reply(data={
        "path": str(written),
        "shape": list(array.shape),
        "axes": {"x": x_name, "y": y_name},
        "offset": array.attrs["offset"],
    }, artefacts=[written],
        note="rows run along y, columns along x; unmeasured cells are nan")


@tool()
def vti_export_cut(path: str, start: list, end: list, output: Optional[str] = None,
                   samples: Optional[int] = None, width: int = 1,
                   normal: Optional[list] = None,
                   where: Optional[str] = None) -> Reply:
    """Write a line cut out as columns: distance, H, K, L and intensity.

    The cut is the one `vti_cut` draws, in the format the viewer's export
    button writes -- the extension of *output* chooses it, as for
    `vti_export_image`. Four columns plus the distance mean the file says
    where in reciprocal space every point was, not only how far along the
    line, so it can be replotted against any axis.
    """
    import numpy as np

    from da6idb import export
    from da6idb.hkl.slice import line_cut

    file = resolve.vti(path, where)
    distance, points, values = line_cut(
        cache.grid(file), start, end, samples=samples, width=int(width),
        normal=normal)
    target = resolve.artefact(output, ".txt", stem=f"{file.stem}_cut")

    note = ("position is the distance from the start of the line; "
            "line runs H %.6f K %.6f L %.6f -> H %.6f K %.6f L %.6f"
            % (*points[0], *points[-1]))
    names = export.write_columns(
        target,
        [("line", note, {"position": distance, "H": points[:, 0],
                         "K": points[:, 1], "L": points[:, 2],
                         "intensity": values})],
        description=_provenance(file),
        note_width=int(width) if width > 1 else None)

    return Reply(data={
        "path": str(target),
        "files": names,
        "samples": int(distance.size),
        "measured": int(np.isfinite(values).sum()),
    }, artefacts=[target], note=note)
