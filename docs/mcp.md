# `da6idb_mcp` — the viewers as tools a model can call

Everything the five windows do is reachable from Python already; this makes it
reachable from a conversation. Ask for scan 152's rocking curve and you get the
peak statistics *and* the plot. Ask for an L-slice of a gridded map and you get
the image. Ask it to open the real `image_viewer` window and it opens on your
screen, and you can keep driving it from the chat while you watch.

52 tools in seven namespaces, one server.

```bash
pip install -e ".[notebook,hkl,mcp]"      # headless tools
pip install -e ".[all]"                   # + the real windows
```

`mcp` is the one new dependency. Nothing else in the repository needs it —
`python -m da6idb_mcp --list` and the whole test suite run without it.

Either generation of the SDK will do. 2.0 renamed `FastMCP` to `MCPServer` and
moved it to `mcp.server.mcpserver`, so a machine set up last year and one set
up today disagree about where the class lives; `da6idb_mcp/_sdk.py` tries both
names, and nothing else in the package imports the SDK at all. The parts used
here — the constructor, `add_tool`, `run`, `Image`, `ToolError` — did not
change across the rename. Run against 2.3; the 1.x path is the same four
calls under the older name.

## Registering it

The server speaks stdio, so a client spawns it. Point it at the Python that has
the package — on this machine the conda environment, not the system one.

**Claude Code** — `.mcp.json` at the repository root does this already:

```bash
claude mcp list          # da6idb should be there
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "da6idb": {
      "command": "/path/to/envs/DashPVA/bin/python",
      "args": ["-m", "da6idb_mcp", "--folder", "/path/to/your/data"],
      "cwd": "/path/to/bluesky_hdf"
    }
  }
}
```

`--folder` is where a bare scan number is looked up, so you can say "scan 152"
rather than a path. Without it, pass `where="/path/to/data"` to any tool.

| flag | |
|---|---|
| `--folder PATH` | where a bare scan number is resolved |
| `--roots PATH` | refuse any path outside this folder (repeatable) |
| `--no-eval` | switch the `da6idb_eval` escape hatch off |
| `--no-live` | do not advertise the `viewer_*` tools at all |
| `--host-platform NAME` | Qt platform for the window process, e.g. `offscreen` |
| `--list` | print the tool names and exit; needs no `mcp` installed |

Each flag has a `DA6IDB_MCP_*` environment variable behind it (see
`da6idb_mcp/config.py`); the flag wins, because a config file that spawns the
server with arguments is harder to get wrong than exporting variables around it.

## The namespaces

| | |
|---|---|
| `scan_*` (11) | `list` `info` `fields` `curve` `statistics` `fit` `map` `baseline` `hkl_position` `trend` `export` |
| `image_*` (10) | `info` `frame` `channels` `rois` `roi_rect` `roi_curve` `roi_series` `projection` `two_theta` `geometry` |
| `hkl_*` (10) | `sources` `setup_read` `setup_write` `setup_delete` `extent` `resolution` `cloud` `grid` `slice` `cut` |
| `vti_*` (6) | `list` `info` `slice` `cut` `export_image` `export_cut` |
| `batch_*` (2), jobs (3) | `find` `grid_scans`; `job_status` `job_cancel` `job_list` |
| `viewer_*` (7) | `list` `open` `close` `state` `set_state` `screenshot` `call` |
| admin (3) | `cache_status` `cache_clear` `da6idb_eval` |

Nothing is reimplemented: each tool wraps a function `docs/api.md` already
lists. Tools that draw return the numbers and the picture together, so a model
sees the plot it asked for rather than a path to one.

## Two layers, two processes

The headless tools compute through `da6idb` and draw with matplotlib/Agg. They
need no window and no desktop session, and they are the ones you want for
almost everything.

The `viewer_*` tools open the actual PyQt windows. They run in a **separate
process**, started the first time one is called and kept for the session. That
is not tidiness:

- a VTK/OpenGL fault in the 3D views is a **segfault, not an exception** — in
  one process it would take the headless tools down with it;
- Qt **`abort()`s** rather than raising when it cannot reach a display, so a
  machine with no desktop session cannot be handled with `try`/`except`;
- the gridding windows call `processEvents()` during long runs, so GUI work and
  a minutes-long `grid_to_vti` in one process stutter each other.

On a machine with no display the host dies, and every `viewer_*` tool says so in
one sentence — remembered for a minute, so six live calls do not spawn six
doomed processes. The headless tools are unaffected.

The live layer is six generic tools rather than one per control.
`viewer_state` gives a window's settings as a flat dict and `viewer_set_state`
takes the same dict back, so you change one key and leave the rest:

```
viewer_open(kind="image", target=152)
viewer_state(window="image-1")        -> {"frame": 0, "colormap": "viridis", ...}
viewer_set_state(window="image-1", frame=37, log_scale=True)
viewer_screenshot(window="image-1")
```

`w.set_state(w.state())` is a no-op for all five windows, and the test suite
round-trips each one in its own subprocess to keep that true.

A read (`viewer_state`, `viewer_screenshot`, `viewer_list`) is served *while* the
window is in the middle of a long gridding run; a write is refused with what it
is busy with and for how long. So you can watch a batch run progress.

## Long work

`hkl_grid` and `batch_grid_scans` take `wait_seconds=20`. Finish inside it — the
common case, since a cached `.vti` is reused in well under a second — and the
answer comes straight back. Otherwise you get a job id for `job_status` and
`job_cancel`; cancellation is real, through the `progress` callback that
`grid_to_vti` already has.

## `da6idb_eval`

For the question no tool was written for. A snippet runs with `np`, `pd`, `xr`,
`plt`, every `da6idb` submodule and `cache` bound — the same `Scan` and `Images`
objects the curated tools built, not a second copy. The last expression's value
comes back, stdout with it, a left-open figure as a picture. Names persist
between calls.

It is as privileged as your own shell, because that is what it is: same files,
same process, same account. Two limits:

- `--no-eval` switches it off entirely, for a server several people share;
- it will not import PyQt5, pyvistaqt, pyqtgraph or the viewer modules, checked
  three ways. That one rule is what keeps the two-process guarantee true. Plain
  `pyvista` is fine, and is how `hkl.grid.load_vti` reads a `.vti`.

An exception is reported as data, with the snippet's own line numbers, not
raised as a tool failure.

## Memory

A server is meant to be left running for a whole beamtime, and one gridded map
is 83 million cells. Scans, image stacks and grids are cached on
`(path, mtime, size)`, so a regrid on disk is automatically a miss; grids share
one budget of 150 million cells. `cache_status` says what is held and how well
it is paying off, `cache_clear` lets go of all of it — nothing is lost, since
every one of them is a file.

## Tests

```bash
pytest tests          # or: python tests/run.py
```

`tests/test_mcp.py` is 23 of them, pinned to numbers in `tests/golden/mcp/`
rather than asking whether a call raises. Each window's state round-trip runs in
its own subprocess, because a graphics fault is a segfault and not something a
test runner can catch. There is a no-display test that forces the host to fail
and checks every live tool gives the one sentence and every headless tool still
works.
