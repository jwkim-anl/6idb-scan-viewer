"""Keeping the expensive objects around between tool calls.

A model works in small steps -- list the scans, then one curve, then the
statistics of the same curve, then a frame -- and every one of those would
otherwise re-read the same file. The caches here make that sequence cost
one read.

Everything is keyed on a *stamp*: the resolved path with its modification
time and size, which is what :class:`da6idb.hkl.grid.LoadedGrids` already
uses. Reusing that is not only less code, it means a file rewritten on disk
is a cache miss by construction -- so a regrid, or a detector still writing,
is picked up without anyone having to remember to invalidate anything.

The limits are about memory, not file handles: ``ImageData`` reads its
stack and closes the file, so what six cached ``Images`` cost is six
stacks, which is why that limit is the small one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from da6idb_mcp import config


def stamp(path):
    """``(path, mtime, size)``, or None if the file has gone."""
    from da6idb.hkl.grid import LoadedGrids

    return LoadedGrids.stamp(Path(path))


class _Lru:
    """A small thread-safe LRU."""

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, build):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                self.hits += 1
                return self._items[key]
            self.misses += 1

        # Built outside the lock: reading a 100 MB stack must not stop
        # another tool reading a curve it already has.
        value = build()

        with self._lock:
            if key in self._items:        # someone else got there first
                self._items.move_to_end(key)
                return self._items[key]
            self._items[key] = value
            while len(self._items) > self.limit:
                self._items.popitem(last=False)
            return value

    def clear(self):
        with self._lock:
            self._items.clear()

    def status(self) -> dict:
        with self._lock:
            return {"held": len(self._items), "limit": self.limit,
                    "hits": self.hits, "misses": self.misses}


_scansets = _Lru(config.SCANSET_CACHE)
_scans = _Lru(config.SCAN_CACHE)
_images = _Lru(config.IMAGES_CACHE)
_hklmaps = _Lru(config.SCAN_CACHE)

_grids = None
#: ``LoadedGrids`` keeps a plain list and does no locking of its own, which
#: is right for a viewer on one thread and not enough for a server on
#: several, so every touch of it goes through this.
_grids_lock = threading.RLock()


def grids():
    """The one grid cache, shared by every tool that loads a ``.vti``."""
    global _grids
    with _grids_lock:
        if _grids is None:
            from da6idb.hkl.grid import LoadedGrids

            _grids = LoadedGrids(max_cells=config.GRID_CELLS)
        return _grids


def _folder_signature(directory: Path):
    """What the folder's HDF files look like right now.

    A scan appearing mid-beamtime has to invalidate the ScanSet, and a
    folder's own mtime does not reliably change when a file inside it is
    rewritten, so the names and stamps are the key.
    """
    from da6idb.files import list_hdf_files

    try:
        return tuple(stamp(p) for p in list_hdf_files(directory))
    except OSError:
        return ()


def scanset(directory):
    directory = Path(directory)
    key = (directory, _folder_signature(directory))

    def build():
        from da6idb import open_folder

        return open_folder(directory)

    return _scansets.get(key, build)


def scan(master):
    master = Path(master)

    def build():
        from da6idb import open_scan

        return open_scan(master)

    return _scans.get(stamp(master), build)


def _image_key(master: Path):
    """Key an image-derived object on the detector file *and* the master.

    The detector file is the large one and two masters can name the same
    one, but the master carries the angles, so both have to be in the key.
    """
    from da6idb.files import resolve_image_file

    image = resolve_image_file(master)
    if image is None:
        raise FileNotFoundError(f"no detector file found for {master.name}")
    return stamp(image), stamp(master)


def images(master):
    master = Path(master)
    key = _image_key(master)

    def build():
        return scan(master).images()

    return _images.get(key, build)


def hklmap(master, **kwargs):
    master = Path(master)
    key = (*_image_key(master), tuple(sorted(kwargs.items())))

    def build():
        return scan(master).hkl(**kwargs)

    return _hklmaps.get(key, build)


def grid(path):
    """A loaded ``.vti``, through the shared grid cache."""
    from da6idb.hkl.grid import load_vti

    path = Path(path)
    with _grids_lock:
        found = grids().get(path)
    if found is not None:
        return found

    value = load_vti(path)         # outside the lock: this is the slow part
    with _grids_lock:
        grids().put(path, value, int(value.n_cells))
    return value


def forget(path) -> None:
    """Drop one ``.vti`` from the grid cache.

    A rewritten file is a miss anyway -- the stamp changed -- so this is
    only about reclaiming the memory promptly after a regrid, rather than
    holding a map nobody will ask for again until it ages out.
    """
    key = stamp(path)
    store = grids()
    with _grids_lock:
        store._items = [item for item in store._items if item[0] != key]


def status() -> dict:
    with _grids_lock:
        store = grids()
        cells = sum(item[2] for item in store._items)
        held = len(store._items)
    return {
        "scansets": _scansets.status(),
        "scans": _scans.status(),
        "images": _images.status(),
        "hkl_maps": _hklmaps.status(),
        "grids": {"held": held, "cells": cells, "cell_limit": config.GRID_CELLS},
    }


def clear() -> dict:
    """Let go of everything, returning what was held before."""
    before = status()
    for store in (_scansets, _scans, _images, _hklmaps):
        store.clear()
    with _grids_lock:
        grids().clear()
    return before
