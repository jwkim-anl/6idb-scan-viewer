"""The server's own state, and the way out of its tool surface.

`cache_status` and `cache_clear` are about memory: this server is meant to
be left running for a whole beamtime, and one gridded map is 83 million
cells, so what it is holding is a fair question to be able to ask.

`da6idb_eval` is the other kind of tool entirely -- the one that answers
the question nobody wrote a tool for. See :mod:`da6idb_mcp.evaluate` for
what it will and will not do.
"""

from typing import Optional

from da6idb_mcp import cache, evaluate
from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool


@tool()
def cache_status() -> Reply:
    """What the server is holding in memory, and how well it is paying off.

    Each cache reports what it holds, its limit, and its hits and misses
    since the server started. Grids are counted in cells rather than
    entries, because that is what the memory actually is: one map of
    359x422x550 is 83 million cells, so two of them are most of the
    allowance.
    """
    held = cache.status()
    grids = held["grids"]
    share = (100.0 * grids["cells"] / grids["cell_limit"]
             if grids["cell_limit"] else 0.0)
    return Reply(data=held,
                 note=f"{grids['held']} gridded map(s), {share:.0f}% of the "
                      f"cell allowance; cache_clear lets go of everything")


@tool()
def cache_clear() -> Reply:
    """Let go of every cached scan, image stack and gridded map.

    Nothing is lost: every one of them is a file on disk and is re-read on
    the next call that wants it. Worth doing after a long run of gridding,
    or before one, if the machine is tight for memory. What it was holding
    comes back as the answer.
    """
    before = cache.clear()
    grids = before["grids"]
    return Reply(data={"released": before},
                 note=f"released {grids['held']} map(s), "
                      f"{grids['cells']:,} cells, and every cached scan")


@tool()
def da6idb_eval(code: str, reset: bool = False,
                timeout: Optional[float] = None) -> Reply:
    """Run a Python snippet against the same objects the other tools use.

    For the question no tool was written for. ``np``, ``pd``, ``xr``,
    ``plt``, ``da6idb`` and its submodules (``scan``, ``images``, ``hkl``,
    ``fit``, ``trends``, ``batch``, ``export``, ``files``, ``rois``) are
    already bound, along with ``open_scan``, ``open_folder``, ``resolve``
    and ``cache`` -- use ``cache.scan(path)`` and ``cache.images(path)``
    rather than opening files again, and the object another tool just
    built is the one you get.

    The value of the last expression comes back, whatever it is -- a
    DataFrame arrives as records, an array as its shape and range -- and
    anything printed comes with it. Leave a matplotlib figure open and it
    is returned as a picture, so ``plt.plot(...)`` is enough to see
    something. Names persist between calls; pass ``reset=True`` for a
    clean namespace.

    An exception is reported, not raised: the traceback is part of the
    answer. *timeout* is advisory and reported back, since a thread that
    will not stop cannot be stopped. Qt and the viewer modules are
    refused outright -- a window in this process would put a segfault in
    it, which is what the separate `viewer_*` tools exist to avoid.
    ``pyvista`` itself is fine, and is how ``hkl.grid.load_vti`` reads a
    gridded file.
    """
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    outcome = evaluate.run(code, reset=bool(reset))
    value = outcome["value"]

    # A snippet that plots usually ends on plt.plot(...), whose value is
    # a list of Line2D nobody wants to read. The figure is the answer.
    figure = value if isinstance(value, plt.Figure) else None
    if figure is not None:
        value = None
    else:
        fresh = [n for n in plt.get_fignums() if n not in before]
        if fresh:
            figure = plt.figure(fresh[-1])
            if not isinstance(value, (int, float, str, bool, type(None))):
                value = None

    data = {"value": value, "defined": evaluate.defined()}
    if outcome["printed"].strip():
        data["printed"] = outcome["printed"]
    if outcome["failed"]:
        data["failed"] = outcome["failed"]

    note = f"{outcome['seconds']:g} s"
    if timeout is not None:
        note += (f" (timeout={float(timeout):g} s is advisory; nothing was "
                 "interrupted)")
    if outcome["failed"]:
        note += "; the snippet raised -- see failed and printed"
    return Reply(data=data, figure=figure, note=note)
