"""Reading and processing the HDF5 files written at APS 6-ID-B.

Everything here works in a plain Python session: no Qt, no windows. The
viewers in this repository are built on top of it.

    from da6idb import open_folder

    folder = open_folder("/data/beamtime")
    folder.summary()

The notebook objects need pandas and xarray; the reading modules underneath
(:mod:`da6idb.scan`, :mod:`da6idb.images`, :mod:`da6idb.hkl`) need neither,
and are importable on their own.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["Scan", "ScanSet", "Images", "HklMap", "open_folder", "open_scan"]


def __getattr__(name):
    # Imported on first use, so `import da6idb` costs nothing until the
    # notebook layer is actually wanted.
    if name in __all__:
        from da6idb import session

        return getattr(session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
