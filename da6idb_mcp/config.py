"""Everything the server reads from its environment, in one place.

The defaults are the ones that suit a server an agent leaves running all
day, which is not always what suits an interactive window: the grid cache
below is a quarter of what :mod:`da6idb.hkl.grid` gives the viewers, because
a viewer is quit when the beamtime ends and this is not.
"""

from __future__ import annotations

import os
from pathlib import Path

# -- where the data is -------------------------------------------------------

#: Folders a tool may reach. Empty means anywhere: the server runs as the
#: user, with the user's own privileges, and pretending otherwise would be
#: theatre. Set ``DA6IDB_MCP_ROOTS`` (colon-separated) to mean it.
ROOTS: tuple[Path, ...] = tuple(
    Path(part).expanduser().resolve()
    for part in os.environ.get("DA6IDB_MCP_ROOTS", "").split(os.pathsep)
    if part.strip()
)

#: Resolves a bare scan number when no folder is named.
DEFAULT_FOLDER = os.environ.get("DA6IDB_MCP_FOLDER") or None


def default_folder() -> Path | None:
    if DEFAULT_FOLDER:
        return Path(DEFAULT_FOLDER).expanduser().resolve()
    return ROOTS[0] if len(ROOTS) == 1 else None


# -- caches ------------------------------------------------------------------

#: Gridded cells held in memory, against ``da6idb``'s own 200M (~800 MB).
#: Lower, because a viewer is quit when the beamtime ends and a server is
#: not -- but not much lower: one map in ``data/`` is 359x422x550, 83M cells
#: on its own, so anything under about 100M would evict the previous scan's
#: map every time and leave a model paying 2 s to reload what it just had.
GRID_CELLS = int(os.environ.get("DA6IDB_MCP_GRID_CELLS", 150_000_000))

SCANSET_CACHE = int(os.environ.get("DA6IDB_MCP_SCANSETS", 4))
SCAN_CACHE = int(os.environ.get("DA6IDB_MCP_SCANS", 32))
#: Small: each one is a whole stack in memory, 101 x 516 x 516 for the
#: example scan, and a 2k detector would be twenty times that.
IMAGES_CACHE = int(os.environ.get("DA6IDB_MCP_IMAGES", 6))

# -- pictures ----------------------------------------------------------------

#: A returned PNG is re-rendered smaller until it fits. Every picture crosses
#: the wire into a context window, so this is a real budget, not a formality.
MAX_PNG_BYTES = int(os.environ.get("DA6IDB_MCP_MAX_PNG", 180_000))
FIGURE_DPI = int(os.environ.get("DA6IDB_MCP_DPI", 110))
FIGURE_SIZE = (7.0, 4.6)

#: Where tools write artefacts they do not inline (full-resolution images,
#: exported columns) when the caller names no path of its own.
ARTEFACTS = Path(
    os.environ.get("DA6IDB_MCP_ARTEFACTS")
    or Path(os.environ.get("TMPDIR", "/tmp")) / "da6idb-mcp"
).expanduser()

# -- work --------------------------------------------------------------------

#: Headless tool bodies run in threads; two at once keeps a long grid from
#: starving a quick curve without letting numpy thrash.
COMPUTE_SLOTS = int(os.environ.get("DA6IDB_MCP_COMPUTE_SLOTS", 2))

#: Seconds a gridding tool waits before handing back a job id instead.
DEFAULT_WAIT = float(os.environ.get("DA6IDB_MCP_WAIT", 20))

#: Finished jobs are kept this long so a poll after the fact still works.
JOB_TTL = float(os.environ.get("DA6IDB_MCP_JOB_TTL", 3600))

#: ``da6idb_eval`` can be switched off entirely for a shared deployment.
ALLOW_EVAL = os.environ.get("DA6IDB_MCP_NO_EVAL", "").lower() not in {
    "1", "true", "yes", "on"}

# -- the viewer host ---------------------------------------------------------

#: How long to wait for the GUI process to say hello before deciding there
#: is no desktop session.
HOST_STARTUP = float(os.environ.get("DA6IDB_MCP_HOST_STARTUP", 20))

#: The GUI process is started with ``QT_QPA_PLATFORM`` cleared, so that a
#: server launched under ``offscreen`` still puts real windows on a real
#: screen. Setting this puts a platform back deliberately -- ``offscreen``
#: is how the live layer is tested on a machine with no display.
HOST_PLATFORM = os.environ.get("DA6IDB_MCP_HOST_PLATFORM") or ""

#: Read-only viewer commands run several at a time: asking what four
#: windows are showing should not be four round trips deep in a queue.
#: Writing ones go one at a time, which the host enforces as well.
VIEWER_READ_SLOTS = int(os.environ.get("DA6IDB_MCP_VIEWER_READ_SLOTS", 4))

#: Having decided that, how long to believe it before trying again.
NO_DISPLAY_TTL = float(os.environ.get("DA6IDB_MCP_NO_DISPLAY_TTL", 60))

#: Per-command ceilings in the host, by command prefix. A timeout does not
#: cancel the queued work -- it only stops this call waiting for it.
TIMEOUTS = {
    "viewer_state": 15.0,
    "viewer_set_state": 60.0,
    "viewer_screenshot": 20.0,
    "viewer_list": 10.0,
    "viewer_close": 15.0,
    "viewer_open": 120.0,
    "viewer_call": 900.0,
}
DEFAULT_TIMEOUT = 60.0


def timeout_for(command: str) -> float:
    return TIMEOUTS.get(command, DEFAULT_TIMEOUT)
