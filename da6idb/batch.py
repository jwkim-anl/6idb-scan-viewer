"""Gridding a run of scans onto one common lattice.

The cell size is measured once, from the first scan, and reused for the rest:
with a common HKL range the grids then share origin, spacing and shape, which
is the whole point of doing a series together.

No Qt. A notebook imports this directly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from da6idb.files import image_path_for_master, master_path_for_image, scan_number
from da6idb.hkl.geometry import read_geometry
from da6idb.hkl.grid import GridCancelled, grid_to_vti, natural_step

OUTPUT_DIRECTORY = "output"

def expand_scans(start, end, step=1):
    """The scan numbers from *start* to *end* inclusive, every *step*.

    Given 152, 160, 2 this is [152, 154, 156, 158, 160]. The ends may be the
    other way round, and a step of zero is taken to mean one.
    """
    start, end, step = int(start), int(end), int(step)
    step = abs(step) or 1
    if end < start:
        start, end = end, start
    return list(range(start, end + 1, step))

def parse_scans(text):
    """Scan numbers from what someone typed: "152, 154 157-160"."""
    numbers = []
    for chunk in re.split(r"[,\s]+", str(text).strip()):
        if not chunk:
            continue
        span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", chunk)
        if span:
            numbers.extend(expand_scans(span.group(1), span.group(2)))
        elif chunk.isdigit():
            numbers.append(int(chunk))
        else:
            raise ValueError(f"not a scan number: {chunk!r}")
    seen, unique = set(), []
    for number in numbers:          # keep the order given, drop repeats
        if number not in seen:
            seen.add(number)
            unique.append(number)
    return unique

def find_scans(folder):
    """``{scan number: image path}`` for the scans in *folder*.

    Keyed off the masters, not off the image files. Where the detector wrote
    is no longer something a name in this directory can answer: a master may
    record a relative path into a parallel tree
    (``../lambda/<run>/test_00020.h5``), or sit beside a file named after the
    scan alone with no ``pva`` about it. :func:`image_path_for_master` knows
    every one of those routes -- it is what the image viewer resolves with --
    and a glob for ``pva*.h5`` knows exactly one.

    Image files with no master beside them are picked up afterwards, so a
    folder of bare ``pva*.h5`` goes on working as it did.

    Two series sharing a directory can collide on a scan number -- see
    :func:`~da6idb.files.scan_tags` -- and one key can only hold one of them.
    The last in sorted order wins, which is what the glob did before.
    """
    folder = Path(folder)
    found: dict[int, Path] = {}
    for master in sorted(folder.glob("*_master.*")):
        image = image_path_for_master(master)
        if image is None:
            continue
        try:
            found[int(scan_number(master))] = image
        except ValueError:
            continue
    for path in sorted(folder.glob("pva*.h5")):
        try:
            found.setdefault(int(scan_number(path)), path)
        except ValueError:
            continue
    return found

def grid_scans(
    folder,
    numbers,
    hkl_range=None,
    oversample: float = 1.0,
    output=None,
    step=None,
    reuse: bool = True,
    normalize_transmission: bool = True,
    on_start=None,
    on_geometry=None,
    on_done=None,
    progress=None,
):
    """Grid several scans onto one lattice. Returns ``[(number, path, note)]``.

    ``path`` is None for a scan that could not be done, and ``note`` says why.
    The cell size comes from the first scan that reads successfully unless
    *step* is given outright. With *normalize_transmission* each scan's frames
    are divided by its attenuator transmission where that varies, which is
    decided per scan -- a series can mix scans that moved their filter with
    scans that did not.
    """
    folder = Path(folder)
    found = find_scans(folder)
    out_dir = Path(output) if output is not None else folder / OUTPUT_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for number in numbers:
        image = found.get(int(number))
        if on_start is not None:
            on_start(number, image)

        if image is None:
            results.append((number, None, "no image file for that scan number"))
            if on_done is not None:
                on_done(*results[-1])
            continue
        try:
            geometry = read_geometry(image, master_path_for_image(image))
            # read_geometry turns this on where the filter moved; switching it
            # off here is a decision for the whole batch.
            if not normalize_transmission:
                geometry.normalize_transmission = False
            if on_geometry is not None:
                on_geometry(number, geometry)
            if step is None:
                # The first scan sets the lattice; everything after it is
                # measured onto the same one.
                step = natural_step(geometry, oversample=oversample)
            target = out_dir / f"{image.stem}_hkl_grid.vti"
            # Cheaper than asking whether the cached file matches: if grid_to_vti
            # reused it, it will not have touched it.
            stamp = target.stat().st_mtime if target.is_file() else None
            written = grid_to_vti(
                image,
                hkl_range=hkl_range,
                output=target,
                step=step,
                oversample=oversample,
                geometry=geometry,
                reuse=reuse,
                progress=(None if progress is None
                          else (lambda done, total, n=number: progress(n, done, total))),
            )
            reused = stamp is not None and written.stat().st_mtime == stamp
            results.append((number, written, "reused" if reused else "written"))
        except GridCancelled as stop:
            results.append((number, None, f"cancelled ({stop})"))
            if on_done is not None:
                on_done(*results[-1])
            break
        except Exception as exc:
            results.append((number, None, f"{type(exc).__name__}: {exc}"))
        if on_done is not None:
            on_done(*results[-1])
    return results
