"""Gridding a run of scans onto one common lattice, and the jobs that takes.

The point of a batch is the word *common*: the cell size is measured
once, from the first scan that reads, and every scan after it is binned
onto the same lattice, so with one HKL range the maps share origin,
spacing and shape and can be subtracted from each other. Gridding the
same scans one at a time with `hkl_grid` does not give you that.

This talks to :mod:`da6idb.batch` directly and never opens the batch
window; `viewer_open("batch", ...)` is for when a person wants to watch
one run.

A run of a dozen scans is minutes of work, so `batch_grid_scans` hands
back a job id when it outlasts *wait_seconds* -- `job_status` for how far
it has got and `job_cancel` to stop it, which it does between frames
rather than at the end.
"""

from typing import Optional, Union

from da6idb_mcp import jobs, resolve
from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool

Scans = Union[str, list, int]


def _numbers(scans) -> list:
    """Scan numbers from a list or from ``"152, 154 157-160"``."""
    from da6idb.batch import parse_scans

    if isinstance(scans, str):
        wanted = parse_scans(scans)
    elif isinstance(scans, int):
        wanted = [scans]
    else:
        wanted = [int(n) for n in scans]
    if not wanted:
        raise ValueError(f"{scans!r} named no scans. Try '152-160' or "
                         f"'152, 154, 157'")
    return wanted


@tool()
def batch_find(where: Optional[str] = None) -> Reply:
    """Which scans in a folder have a detector file, and are already gridded.

    What the batch window lists. A scan missing from here has a master
    but no frames that could be found, and gridding it will fail with
    that as its note -- better learnt now than eleven scans into a run.
    """
    from da6idb.batch import OUTPUT_DIRECTORY, find_scans

    folder = resolve.folder(where)
    found = find_scans(folder)
    gridded = {p.name: p for p in resolve.vti_files(folder)}

    rows = []
    for number in sorted(found):
        image = found[number]
        target = f"{image.stem}_hkl_grid.vti"
        rows.append({"scan": int(number), "image": image.name,
                     "gridded": target in gridded,
                     "grid": str(gridded[target]) if target in gridded else None})
    done = sum(1 for row in rows if row["gridded"])
    return Reply(data={
        "folder": str(folder),
        "output_folder": str(folder / OUTPUT_DIRECTORY),
        "scans": len(rows),
        "gridded": done,
        "found": rows,
    }, note=f"{done} of {len(rows)} already gridded; those are reused unless "
            f"reuse=false")


@tool()
def batch_grid_scans(scans: Scans, hkl_range: Optional[Union[list, str]] = None,
                     oversample: float = 1.0, reuse: bool = True,
                     normalize_transmission: bool = True,
                     output: Optional[str] = None,
                     wait_seconds: Optional[float] = None,
                     where: Optional[str] = None) -> Reply:
    """Grid several scans onto one lattice. The long one.

    *scans* is a list of numbers or the range text the window takes,
    ``"152-160"`` or ``"152, 154, 157"``.

    *hkl_range* is ``[h_min, h_max, k_min, k_max, l_min, l_max]``; pass
    ``"first"`` to take the box from the first scan's own extent, which
    is the window's "range from first scan" button and the usual way to
    give a series one box. Left out, each scan is gridded over whatever
    it reached, and the maps will not line up.

    *oversample* above one gives coarser, better-filled cells and a much
    quicker run; below one a finer lattice. Maps already on disk are
    reused unless *reuse* is false. *output* is a folder, defaulting to
    ``output`` beside the data.

    Every scan is reported whether it worked or not: ``path`` is null for
    one that failed and ``note`` says why, so a single unreadable scan
    does not cost the run.
    """
    from da6idb import batch

    folder = resolve.folder(where)
    numbers = _numbers(scans)
    out_dir = None if output is None else str(resolve.artefact(output))

    wanted = hkl_range
    taken_from = None
    if isinstance(wanted, str):
        if wanted.strip().lower() != "first":
            raise ValueError(f"hkl_range={wanted!r}: pass six numbers, or "
                             f"'first' to take the box from the first scan")
        from da6idb.files import master_path_for_image
        from da6idb.hkl.geometry import read_geometry
        from da6idb.hkl.grid import measured_range

        found = batch.find_scans(folder)
        have = [n for n in numbers if n in found]
        if not have:
            raise ValueError(f"none of {numbers} has a detector file in "
                             f"{folder}; batch_find lists the ones that do")
        image = found[have[0]]
        low, high = measured_range(read_geometry(image,
                                                 master_path_for_image(image)))
        wanted = [float(v) for pair in zip(low, high) for v in pair]
        taken_from = have[0]
    elif wanted is not None:
        wanted = [float(v) for v in wanted]

    def work(job):
        results = batch.grid_scans(
            folder, numbers, hkl_range=wanted, oversample=float(oversample),
            output=out_dir, reuse=reuse,
            normalize_transmission=normalize_transmission,
            progress=job.progress())

        rows = [{"scan": int(number), "path": str(path) if path else None,
                 "note": note} for number, path, note in results]
        written = [row["path"] for row in rows if row["path"]]
        failed = [row for row in rows if not row["path"]]
        return Reply(data={
            "folder": str(folder),
            "asked": len(numbers),
            "gridded": len(written),
            "failed": len(failed),
            "hkl_range": wanted,
            "results": rows,
        }, artefacts=written,
            note=(f"range taken from scan {taken_from}; " if taken_from else "")
                 + ("every scan gridded" if not failed else
                    "see note on each scan that has no path"))

    return jobs.run("batch_grid_scans", work, wait_seconds)


@tool()
def job_status(job: str) -> Reply:
    """How a long call is getting on, or its answer now that it has finished.

    A gridding tool that outlasted its wait returned a job id; this is
    where the result turns up. ``progress`` is the fraction of frames
    done. The answer is handed over once -- a second call gets the
    numbers again but not the picture.
    """
    found = jobs.get(job)
    if found.state == "running":
        return Reply(data=found.status(),
                     note=f"still running; job_cancel('{found.id}') to stop it")
    return jobs.collect(found)


@tool()
def job_cancel(job: str) -> Reply:
    """Stop a long call between frames, keeping what it has already written.

    Cancellation is real rather than advisory: the gridding loop checks
    after every frame, so this takes effect in well under a second, and
    the half-finished ``.vti`` is discarded rather than left to be
    mistaken for a complete one.
    """
    found = jobs.get(job)
    if found.finished is not None:
        return Reply(data=found.status(), note="it had already finished")
    found.cancel()
    found.wait(5.0)
    return Reply(data=found.status(),
                 note="asked to stop" if found.state == "running" else "stopped")


@tool()
def job_list() -> Reply:
    """Every long call this server has started, running or finished."""
    listing = jobs.listing()
    running = [row for row in listing if row["state"] == "running"]
    return Reply(data={"jobs": listing, "running": len(running)},
                 note="" if listing else "nothing has needed a job yet")
