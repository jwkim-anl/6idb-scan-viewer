"""The live layer: real viewer windows, driven from the other end of a pipe.

Four modules, split by which process they run in.

:mod:`~da6idb_mcp.viewer.windows` is the catalogue of the five windows and
imports nothing heavy, so **both** processes can read it.
:mod:`~da6idb_mcp.viewer.client` runs in the server: it spawns the GUI
process, resolves a tool's arguments to paths before anything is spawned,
and turns a dead host into one clear sentence. :mod:`~da6idb_mcp.viewer.host`
is that GUI process, and :mod:`~da6idb_mcp.viewer.bridge` is what gets a call
from its socket onto the Qt main thread and the answer back again.

Only ``host`` and ``bridge`` import PyQt5, and the server imports neither.
That is the whole point of the split: a VTK fault in a 3D view is a
segfault rather than an exception, and in one process it would take the
headless tools down with it.
"""
