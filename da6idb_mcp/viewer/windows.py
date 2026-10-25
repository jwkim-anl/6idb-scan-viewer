"""The five windows as a catalogue, readable from either process.

Nothing here imports PyQt5. The server needs to know which kinds exist,
what each one opens and which argument it takes -- to describe the tool,
to resolve a model's ``scan=152`` to a path *before* a GUI process is
spawned, and to answer "no window called ..." without one -- and the
server is the process that must never import Qt. So the catalogue is
data, and :func:`build`, which is called only in the host, is the one
thing here that reaches for the module it names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kind:
    """One sort of window: how to build it, and what it wants built from."""

    name: str
    module: str
    factory: str
    #: Which of the four constructor shapes below it takes.
    takes: str
    about: str

    @property
    def title(self) -> str:
        return self.about.split(".")[0]


KINDS = {
    kind.name: kind for kind in (
        Kind("scan", "scan_viewer", "HdfViewer", "folder",
             "The scan browser. Lists a folder's scans and plots any field "
             "against any other -- rocking curves, fits, maps, trends."),
        Kind("image", "image_viewer", "ImageViewerWindow", "files",
             "The detector panel. Plays the frames of one or more scans, "
             "with ROIs that can be dragged, plotted and compared."),
        Kind("hkl", "hkl_viewer", "HklWindow", "pair",
             "Reciprocal space for one scan. Converts the frames to a point "
             "cloud or a gridded volume and cuts slices through it."),
        Kind("vti", "vti_viewer", "VtiWindow", "vti",
             "A gridded volume already on disk. The same 3D view as the HKL "
             "window, opening in a second rather than a minute."),
        Kind("batch", "batch_grid", "BatchGridWindow", "folder",
             "The batch gridder. Grids a run of scans onto one shared "
             "reciprocal-space box so they can be compared cell for cell."),
    )
}

#: What :func:`da6idb_mcp.viewer.client.targets` must hand each shape.
WANTS = {
    "folder": "a folder of scans",
    "files": "one or more scans",
    "pair": "one scan",
    "vti": "a gridded .vti file",
}


def kind(name):
    """A :class:`Kind` by name, or an error listing the five."""
    try:
        return KINDS[str(name)]
    except KeyError:
        raise ValueError(
            f"no window kind called {name!r}. Kinds: {', '.join(KINDS)}"
        ) from None


def build(name, paths, masters=()):
    """Construct a window. **Host process only** -- this imports Qt.

    *paths* and *masters* are absolute paths as strings, already resolved
    and checked by the server; the host does not second-guess them.
    """
    import importlib

    wanted = kind(name)
    cls = getattr(importlib.import_module(wanted.module), wanted.factory)
    paths = [Path(p) for p in paths]
    masters = [Path(m) if m else None for m in masters]

    if wanted.takes in {"folder", "vti"}:
        return cls(paths[0])
    if wanted.takes == "files":
        return cls(paths, masters=masters or None)
    if wanted.takes == "pair":
        return cls(paths[0], masters[0] if masters else None)
    raise ValueError(f"{wanted.takes!r} is not a constructor shape")


def catalogue() -> list:
    """The five, as the plain data a tool puts in its reply."""
    return [{"kind": k.name, "opens": WANTS[k.takes], "about": k.about}
            for k in KINDS.values()]
