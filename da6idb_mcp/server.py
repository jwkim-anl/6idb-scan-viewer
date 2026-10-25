"""The MCP server itself: the one place that imports ``mcp``.

Everything else in this package is ordinary Python that can be imported,
called and tested without the protocol anywhere in sight -- see
:mod:`da6idb_mcp.tools`. This module is the thin edge that turns the
registry into a server:

    python -m da6idb_mcp                      # stdio, for a client to spawn
    python -m da6idb_mcp --folder /data/6idb  # ... with a default folder
    python -m da6idb_mcp --no-eval --no-live  # ... locked down

Two things happen here that are worth knowing. The *live* tools are
advertised unless ``--no-live`` says otherwise, **not** probed for at
start-up: starting a GUI process to find out whether one can be started
would cost every client a spawn it may never need, and the tools answer
the question themselves, in one sentence, the first time one is called.
And the viewer host is shut down on the way out, because it is a child
process holding windows open; a server that exits without it leaves a
Qt process behind on the user's machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from da6idb_mcp import config, tools

#: What the client sees in its server list.
NAME = "da6idb"

INSTRUCTIONS = """\
Reads and analyses X-ray diffraction data from beamline 6-ID-B: scans
recorded by bluesky as HDF5 masters with a detector file beside them,
and the reciprocal-space maps built from them.

Four headless namespaces, which need no window and always work:

  scan_*   a scan's fields, curves, peak statistics, fits, maps,
           baseline, hkl position, and trends across several scans
  image_*  detector frames, ROIs and their curves, projections, the
           two-theta of a feature, the geometry a frame was taken with
  hkl_*    reciprocal-space extent, resolution, a point cloud, a
           gridded map, a slice through one, a line cut
  vti_*    the same reads against a map already gridded to a .vti file

Tools that draw return the numbers and the picture together.

  viewer_* open the real PyQt windows and drive them. viewer_state
           reads a window's settings as a dict and viewer_set_state
           takes the same dict back, so change one key and leave the
           rest. Only these can render 3D volumes, which need a GPU.
           On a machine with no desktop session they say so and the
           headless tools are unaffected.

  da6idb_eval runs a Python snippet against the same loaded objects the
           other tools use, for the question no tool was written for.

Scans are named by number (152) and resolved in a folder: pass
where="/path/to/data", or start the server with --folder. scan_list
says what a folder holds.\
"""


def build(include_live: bool = True):
    """An MCP server with every tool attached. Returns it and the names."""
    from da6idb_mcp import _sdk

    mcp = _sdk.server_class()(NAME, instructions=INSTRUCTIONS)
    names = tools.register(mcp, include_live=include_live)
    return mcp, names


def parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="da6idb-mcp",
        description="An MCP server over the 6-ID-B analysis package.")
    parser.add_argument("--folder", metavar="PATH",
                        help="where a bare scan number is looked up")
    parser.add_argument("--roots", metavar="PATH", action="append",
                        help="limit every tool to this folder (repeatable)")
    parser.add_argument("--no-eval", action="store_true",
                        help="switch the da6idb_eval escape hatch off")
    parser.add_argument("--no-live", action="store_true",
                        help="do not advertise the viewer_* tools at all")
    parser.add_argument("--host-platform", metavar="NAME",
                        help="Qt platform for the viewer process "
                             "(offscreen, say, on a machine with no display)")
    parser.add_argument("--list", action="store_true",
                        help="print the tool names and exit")
    return parser.parse_args(argv)


def apply(options) -> None:
    """Put the command line into :mod:`da6idb_mcp.config`.

    The flags are the environment variables under another name, and the
    flag wins: a client's config file spawns this with arguments, which
    is harder to get wrong than exporting ``DA6IDB_MCP_*`` around it.
    """
    if options.folder:
        config.DEFAULT_FOLDER = str(Path(options.folder).expanduser())
    if options.roots:
        config.ROOTS = tuple(Path(r).expanduser().resolve()
                             for r in options.roots)
    if options.no_eval:
        config.ALLOW_EVAL = False
    if options.host_platform:
        config.HOST_PLATFORM = options.host_platform


def main(argv=None) -> int:
    options = parse(argv)
    apply(options)

    # Agg before anything draws: tool bodies make matplotlib figures on
    # worker threads, and an interactive backend would try for a window
    # in a process whose whole design is not to have one.
    os.environ.setdefault("MPLBACKEND", "agg")

    if options.list:
        registry = tools.load()
        for name in registry.names():
            if options.no_live and registry[name].live:
                continue
            print(name)
        return 0

    try:
        mcp, names = build(include_live=not options.no_live)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from da6idb_mcp import _sdk

    print(f"da6idb-mcp: {len(names)} tools"
          + (" (headless only)" if options.no_live else "")
          + f", mcp {_sdk.version() or '?'}", file=sys.stderr)
    try:
        mcp.run()
    finally:
        _shut_the_windows()
    return 0


def _shut_the_windows() -> None:
    """Take the GUI process with us, if one was ever started.

    Imported here rather than at the top so that a server which never
    opens a window never loads the client either.
    """
    from da6idb_mcp.viewer import client

    host = client.host()
    if host.running:
        host.stop()
