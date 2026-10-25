"""Turning what a model typed into a path on this disk.

A model will say ``scan=152`` and expect to be understood, or hand back a
path it read out of an earlier reply, or name a ``.vti`` by its bare
filename. All three arrive here and leave as a resolved :class:`Path`.

The errors matter as much as the successes: a tool that fails with "no such
scan" and then lists the twenty-one numbers that do exist saves the model a
round trip, so :func:`scan_master` does exactly that.
"""

from __future__ import annotations

from pathlib import Path

from da6idb_mcp import config


class ResolveError(ValueError):
    """A path or scan number that could not be made sense of."""


def _within_roots(path: Path) -> Path:
    if not config.ROOTS:
        return path
    for root in config.ROOTS:
        if path == root or root in path.parents:
            return path
    allowed = ", ".join(str(r) for r in config.ROOTS)
    raise ResolveError(
        f"{path} is outside the folders this server may read ({allowed}). "
        f"Set DA6IDB_MCP_ROOTS to change that."
    )


def folder(where=None) -> Path:
    """A data folder, defaulting to the one the server was pointed at."""
    if where is None or where == "":
        fallback = config.default_folder()
        if fallback is None:
            raise ResolveError(
                "no folder given and no default is set; pass folder=..., or "
                "start the server with DA6IDB_MCP_FOLDER"
            )
        where = fallback
    path = Path(where).expanduser()
    if not path.is_absolute():
        base = config.default_folder()
        path = (base / path) if base else path.resolve()
    path = path.resolve()
    if not path.is_dir():
        raise ResolveError(f"{path} is not a folder")
    return _within_roots(path)


def scan_master(scan, where=None) -> Path:
    """The master file for *scan*: a number, a filename, or a full path.

    The number lookup goes through :class:`da6idb.session.ScanSet`, so what
    counts as a scan here is exactly what counts as one in the viewers --
    including its ``is_scan`` test, which keeps a stray HDF file in the
    folder from being offered as scan 0.
    """
    from da6idb_mcp import cache

    # A path, given outright. Checked before the number so a folder of
    # files named "00152.hdf" cannot be mistaken for a scan number.
    text = str(scan).strip()
    direct = Path(text).expanduser()
    if direct.suffix.lower() in {".hdf", ".h5", ".hdf5"} and direct.is_file():
        return _within_roots(direct.resolve())

    directory = folder(where)
    scans = cache.scanset(directory)

    try:
        number = int(text)
    except ValueError:
        candidate = directory / text
        if candidate.is_file():
            return _within_roots(candidate.resolve())
        raise ResolveError(
            f"{scan!r} is neither a scan number nor a file in {directory}"
        ) from None

    if number not in scans.numbers:
        have = ", ".join(str(n) for n in scans.numbers) or "none"
        raise ResolveError(f"no scan {number} in {directory}. Scans there: {have}")
    return _within_roots(Path(scans[number].path).resolve())


def scan_masters(scans, where=None) -> list[Path]:
    """Several scans: a list, or the range syntax the batch window takes.

    ``"151-160"`` and ``"151, 153 158"`` both work, through
    :func:`da6idb.batch.parse_scans`, so what a model can type here is what
    a user can type in the window. A list of numbers or names also works.
    Either way the order given is kept and repeats are dropped.
    """
    from da6idb.batch import parse_scans

    if isinstance(scans, str):
        wanted = parse_scans(scans)
        if not wanted:
            raise ResolveError(
                f"{scans!r} named no scans. Try '151-160', '151,153' or "
                f"'151-160:2'"
            )
    elif isinstance(scans, (int, Path)):
        wanted = [scans]
    else:
        wanted = list(scans)
    if not wanted:
        raise ResolveError("no scans given")
    return [scan_master(one, where) for one in wanted]


def vti(path, where=None) -> Path:
    """A gridded ``.vti``: a full path, or a name inside a folder."""
    direct = Path(str(path)).expanduser()
    if direct.is_file():
        return _within_roots(direct.resolve())

    if direct.is_absolute():
        raise ResolveError(f"no such file: {direct}")

    directory = folder(where)
    for candidate in (directory / direct,
                      directory / "output" / direct,
                      directory / f"{direct}.vti",
                      directory / "output" / f"{direct}.vti"):
        if candidate.is_file():
            return _within_roots(candidate.resolve())

    have = ", ".join(sorted(p.name for p in vti_files(directory))) or "none"
    raise ResolveError(f"no {path!r} under {directory}. Gridded files: {have}")


def vti_files(directory: Path) -> list[Path]:
    """Every ``.vti`` in a folder and in its ``output`` subfolder."""
    from da6idb.batch import OUTPUT_DIRECTORY

    found = list(directory.glob("*.vti"))
    found += list((directory / OUTPUT_DIRECTORY).glob("*.vti"))
    return sorted(set(found))


def artefact(path=None, suffix: str = ".png", stem: str = "da6idb") -> Path:
    """Where to write something the reply only points at."""
    if path is not None:
        out = Path(str(path)).expanduser()
        if not out.is_absolute():
            out = config.ARTEFACTS / out
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    import time

    config.ARTEFACTS.mkdir(parents=True, exist_ok=True)
    return config.ARTEFACTS / f"{stem}-{time.strftime('%H%M%S')}{suffix}"
