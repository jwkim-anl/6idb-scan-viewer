"""What the HKL viewer does: geometry, the cloud, the lattice, planes.

The order these are meant to be used in is the order they are listed.
`hkl_sources` says where every conversion parameter came from and
`hkl_extent` says what box the scan reached; `hkl_grid` bins the frames
onto a lattice and writes it as a ``.vti``; `hkl_slice` and `hkl_cut`
then cut that same lattice. Slicing before gridding works, but it grids
at full resolution first, which is the slow way round.

There is no 3D render here. A volume needs an OpenGL context, which a
server does not have -- `viewer_open("vti", ...)` is how you see one.
What this namespace gives instead is planes and lines through the same
data, which is what the numbers come off anyway.

The instrument description itself is `image_geometry`: it belongs to the
detector file rather than to reciprocal space, and the two namespaces
should not both own it.
"""

from typing import Optional, Union

from da6idb_mcp import cache, jobs, render, resolve
from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool

Scan = Union[int, str]


def _map(scan, where, angles_from_master=False):
    master = resolve.scan_master(scan, where)
    return cache.hklmap(master, angles_from_master=bool(angles_from_master))


def _axis_label(name):
    return f"{name} [r.l.u.]" if name in ("H", "K", "L") else str(name)


@tool()
def hkl_sources(scan: Scan, angles_from_master: bool = False,
                where: Optional[str] = None) -> Reply:
    """Where each conversion parameter came from: file, master, or setup.

    The first thing to check when a map lands somewhere impossible. A
    parameter sourced from ``default`` was guessed because nothing in the
    files said otherwise, and is the usual culprit; one sourced from
    ``setup`` came from the folder's `hkl_setup_read` overrides.
    """
    hkl = _map(scan, where, angles_from_master)
    return Reply(data={
        "file": str(hkl.path),
        "master": str(hkl.master) if hkl.master else None,
        "frames": hkl.geometry.n_frames,
        "sources": hkl.sources(),
    })


@tool()
def hkl_setup_read(where: Optional[str] = None) -> Reply:
    """The folder's geometry overrides, and which fields may be overridden.

    These live in one ``hkl_setup.json`` beside the data and apply to
    every scan in the folder, which is what you want when the detector
    distance or the UB matrix was wrong for the whole beamtime.
    """
    from da6idb.hkl.geometry import SETUP_FIELDS, read_setup, setup_path

    folder = resolve.folder(where)
    values = read_setup(folder)
    return Reply(data={
        "file": str(setup_path(folder)),
        "overrides": values,
        "fields": list(SETUP_FIELDS),
    }, note="" if values else "nothing is overridden here")


@tool()
def hkl_setup_write(values: dict, merge: bool = True,
                    where: Optional[str] = None) -> Reply:
    """Override geometry for a whole folder, after checking the values.

    Each field is run through the same converter the reader uses before
    anything is written, so a mis-shaped UB or a misspelt field name is
    refused here rather than halfway through the next gridding run.
    *merge* keeps the overrides already there; pass false to replace them.

    Every map cached in memory is dropped afterwards, since they were all
    built on the old geometry.
    """
    from da6idb.hkl.geometry import SETUP_FIELDS, read_setup, write_setup

    folder = resolve.folder(where)
    unknown = [name for name in values if name not in SETUP_FIELDS]
    if unknown:
        raise ValueError(
            f"not setup fields: {', '.join(unknown)}. "
            f"Fields: {', '.join(SETUP_FIELDS)}")

    for name, value in values.items():
        try:
            SETUP_FIELDS[name](value)
        except Exception as problem:
            raise ValueError(f"{name}: {problem}") from problem

    merged = {**read_setup(folder), **values} if merge else dict(values)
    path = write_setup(folder, merged)
    cache.clear()
    return Reply(data={"overrides": merged}, artefacts=[path],
                 note="cached maps dropped; the next hkl call re-reads the "
                      "geometry")


@tool()
def hkl_setup_delete(where: Optional[str] = None) -> Reply:
    """Remove a folder's geometry overrides and go back to what the files say."""
    from da6idb.hkl.geometry import read_setup, write_setup

    folder = resolve.folder(where)
    had = read_setup(folder)
    write_setup(folder, {})
    cache.clear()
    return Reply(data={"removed": had},
                 note="nothing was overridden" if not had else
                      "back to the values in the files")


@tool()
def hkl_extent(scan: Scan, angles_from_master: bool = False,
               where: Optional[str] = None) -> Reply:
    """The H, K, L box this scan actually reaches.

    Use it to choose the *hkl_range* for `hkl_grid`, or to give a whole
    batch one range before gridding any of it. A range outside this box
    grids empty cells; a range well inside it is how you get fine
    sampling on one peak without a lattice the size of the whole map.
    """
    hkl = _map(scan, where, angles_from_master)
    frame = hkl.extent()
    flat = [float(v) for pair in zip(frame["low"], frame["high"]) for v in pair]
    return Reply(data={"extent": frame, "hkl_range": flat},
                 note="hkl_range is the same box flattened, ready to pass on")


@tool()
def hkl_resolution(scan: Scan, angles_from_master: bool = False,
                   where: Optional[str] = None) -> Reply:
    """The finest cell this scan's sampling can honestly support.

    ``cell`` is the step `hkl_grid` picks by default, a cube holding
    about one sample; *oversample* scales it, above one for coarser and
    better-filled cells, below one for a finer lattice. Going much finer
    than ``footprint_*`` buys nothing real -- it interpolates the gaps
    between frames rather than measuring them.
    """
    hkl = _map(scan, where, angles_from_master)
    return Reply(data={"resolution": hkl.resolution()})


@tool()
def hkl_cloud(scan: Scan, threshold: float = 0.0, max_points: int = 400_000,
              log: bool = True, angles_from_master: bool = False,
              where: Optional[str] = None) -> Reply:
    """Every surviving pixel as a point in reciprocal space, projected.

    The cloud before any binning: each detector pixel converted to H, K,
    L and kept if it is above *threshold*. The picture is the three
    projections of it, which between them say where the intensity is and
    whether the conversion looks sane, without a lattice having been
    built at all.

    *threshold* is the lever that matters -- at zero this is every pixel
    of every frame, and a few hundred counts is usually enough to leave
    only the features.
    """
    import numpy as np

    hkl = _map(scan, where, angles_from_master)
    cloud = hkl.cloud(threshold=threshold, max_points=int(max_points))
    if cloud.empty:
        return Reply(data={"points": 0},
                     note=f"nothing above {threshold}; lower the threshold")

    weights = cloud["intensity"].to_numpy(dtype=float)
    fig, _unused = render.figure((8.4, 3.0))
    fig.clear()
    axes = fig.subplots(1, 3)
    for ax, (first, second) in zip(axes, (("H", "K"), ("H", "L"), ("K", "L"))):
        ax.hexbin(cloud[first], cloud[second], C=weights, reduce_C_function=np.sum,
                  gridsize=90, cmap="viridis",
                  bins="log" if log else None, linewidths=0)
        ax.set_xlabel(_axis_label(first))
        ax.set_ylabel(_axis_label(second))
    fig.suptitle(f"{hkl.path.name} — {len(cloud):,} points above {threshold:g}",
                 fontsize="medium")

    return Reply(data={
        "points": len(cloud),
        "threshold": threshold,
        "intensity": {"min": float(weights.min()), "max": float(weights.max()),
                      "total": float(weights.sum())},
        "centre_of_mass": {
            axis: float(np.average(cloud[axis], weights=weights))
            for axis in "HKL"},
        "range": {axis: [float(cloud[axis].min()), float(cloud[axis].max())]
                  for axis in "HKL"},
    }, figure=fig, note="colour is summed intensity per bin, not point density")


@tool()
def hkl_grid(scan: Scan, hkl_range: Optional[list] = None,
             oversample: float = 1.0, output: Optional[str] = None,
             reuse: bool = True, wait_seconds: Optional[float] = None,
             angles_from_master: bool = False,
             where: Optional[str] = None) -> Reply:
    """Bin the frames onto a regular lattice and write it as a ``.vti``.

    The slow one, and the one everything else leans on: `hkl_slice` and
    `hkl_cut` cut whatever lattice this last built, and `vti_*` reads the
    file it wrote. A matching grid already on disk is reused and comes
    back in a second, so calling this before slicing is cheap insurance.

    *hkl_range* is ``[h_min, h_max, k_min, k_max, l_min, l_max]``, from
    `hkl_extent`; without it the whole measured box is gridded.
    *oversample* scales the cell: above one coarser and quicker, below
    one finer. If the run outlasts *wait_seconds* you get a job id
    instead of the answer -- the work carries on either way.
    """
    hkl = _map(scan, where, angles_from_master)
    target = None if output is None else str(resolve.artefact(output, ".vti"))

    def work(job):
        array = hkl.grid(hkl_range=hkl_range, oversample=float(oversample),
                         output=target, reuse=reuse, progress=job.progress())
        path = array.attrs.get("path")
        filled = int((array.notnull()).sum())
        return Reply(data={
            "path": path,
            "dimensions": list(array.shape),
            "dims": [str(d) for d in array.dims],
            "cells": int(array.size),
            "measured_cells": filled,
            "filled_fraction": round(filled / max(1, array.size), 4),
            "range": {str(d): [float(array[d].min()), float(array[d].max())]
                      for d in array.dims},
            "max": float(array.max()),
        }, artefacts=[path] if path else [],
            note="hkl_slice and hkl_cut now cut this lattice")

    return jobs.run("hkl_grid", work, wait_seconds)


@tool()
def hkl_slice(scan: Scan, normal: Optional[list] = None, offset: float = 0.0,
              point: Optional[list] = None, layers: int = 1, log: bool = True,
              cmap: str = "viridis", wait_seconds: Optional[float] = None,
              angles_from_master: bool = False,
              where: Optional[str] = None) -> Reply:
    """A plane through the gridded map, as a picture and as numbers.

    *normal* is an H, K, L direction, ``[0, 0, 1]`` by default, which is
    the constant-L plane drawn H across and K up. Place it either by
    *offset* -- how far along that normal from the middle of the map --
    or by *point*, an H, K, L it must pass through, which is usually what
    you actually know. *layers* averages that many cells of thickness,
    and is the cheapest way to pull a weak feature out of the noise.

    Cuts the lattice `hkl_grid` last built for this scan; with none, it
    grids at full resolution first, which is why this can turn into a job.
    """
    aim = [0.0, 0.0, 1.0] if normal is None else [float(v) for v in normal]
    hkl = _map(scan, where, angles_from_master)

    def work(job):
        import numpy as np

        array = hkl.slice(normal=aim, offset=float(offset), point=point,
                          layers=int(layers))
        y_name, x_name = (str(d) for d in array.dims)
        values = array.values
        fig = render.plane(
            values, array[x_name].values, array[y_name].values,
            x_label=_axis_label(x_name), y_label=_axis_label(y_name),
            title=(f"{hkl.path.name} — normal {aim}, "
                   f"offset {array.attrs['offset']:.4g}, {layers} layer(s)"),
            log=log, cmap=cmap)

        finite = values[np.isfinite(values)]
        peak = {}
        if finite.size:
            where_max = np.unravel_index(int(np.nanargmax(values)), values.shape)
            peak = {x_name: float(array[x_name].values[where_max[1]]),
                    y_name: float(array[y_name].values[where_max[0]]),
                    "intensity": float(values[where_max])}
        return Reply(data={
            "normal": array.attrs["normal"],
            "offset": array.attrs["offset"],
            "through": array.attrs["point"],
            "layers": array.attrs["layers"],
            "axes": {"x": x_name, "y": y_name},
            "shape": list(values.shape),
            "measured_cells": int(finite.size),
            "brightest": peak,
        }, figure=fig,
            note="brightest is in HKL, ready to pass to hkl_cut or as a point")

    return jobs.run("hkl_slice", work, wait_seconds)


@tool()
def hkl_cut(scan: Scan, start: list, end: list, samples: Optional[int] = None,
            width: int = 1, normal: Optional[list] = None, log: bool = False,
            wait_seconds: Optional[float] = None,
            angles_from_master: bool = False,
            where: Optional[str] = None) -> Reply:
    """Intensity along the line between two H, K, L points.

    Named by its two ends rather than dragged into place, so the same cut
    can be repeated on the next scan and written into a paper. *width*
    averages that many parallel lines either side; pass the slice's
    *normal* as well to keep that widening inside the plane instead of
    letting it wander out of it.

    Cuts the same lattice `hkl_slice` does.
    """
    hkl = _map(scan, where, angles_from_master)

    def work(job):
        import numpy as np

        frame = hkl.cut(start=start, end=end, samples=samples,
                        width=int(width), normal=normal)
        values = frame["intensity"].to_numpy(dtype=float)

        fig, ax = render.figure()
        ax.plot(frame["distance"], values, lw=1.3)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("distance from start [r.l.u.]")
        ax.set_ylabel("intensity")
        ax.grid(alpha=0.3)
        ax.set_title(f"{hkl.path.name} — {start} to {end}", fontsize="medium")

        peak = {}
        if np.isfinite(values).any():
            top = int(np.nanargmax(values))
            peak = {"distance": float(frame["distance"][top]),
                    "H": float(frame["H"][top]), "K": float(frame["K"][top]),
                    "L": float(frame["L"][top]),
                    "intensity": float(values[top])}
        return Reply(data={
            "start": frame.attrs["start"], "end": frame.attrs["end"],
            "width": frame.attrs["width"],
            "samples": len(frame),
            "measured": int(np.isfinite(values).sum()),
            "brightest": peak,
            # As columns rather than as the frame's records: a cut is a few
            # hundred points, and naming H, K, L and the distance once each
            # instead of once per point is most of the reply.
            "distance": frame["distance"].to_numpy(dtype=float),
            "intensity": values,
        }, figure=fig,
            note="intensity against distance from start, NaN where the line "
                 "left the measured volume; H, K, L run from start to end")

    return jobs.run("hkl_cut", work, wait_seconds)
