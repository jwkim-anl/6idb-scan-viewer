"""Pinning the MCP server: the tool surface, the escape hatch, the windows.

Three layers, and they fail in three different ways, so they are tested
three different ways:

* the **headless** tools are ordinary functions over ``data/`` -- pinned
  to numbers, like the rest of this suite;
* the **round trip** ``w.set_state(w.state())`` is run in a subprocess
  per window, one window per process on purpose: two of the five build a
  VTK ``QtInteractor``, and a graphics fault there is a segfault rather
  than an exception, which would otherwise take the whole suite with it;
* the **live** layer already lives in a subprocess -- that is its whole
  design -- so it is driven from here directly, with the host forced
  offscreen so that running the tests does not throw real windows onto
  the user's screen.

The live tests skip themselves when no host can be started, which is the
same condition the server has to survive in the field, and is checked
here too: with the host made to fail, every ``viewer_*`` tool says one
clear sentence and every headless tool still works.
"""

import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Pipe
from pathlib import Path

if __name__ == "__main__":
    # Run as a script, this file is the subprocess end of the round-trip
    # test below, and nothing has put the repository on its path yet --
    # conftest does that for the suite, and there is no conftest here.
    _here = Path(__file__).resolve().parent
    for _directory in (_here, _here.parent):
        if str(_directory) not in sys.path:
            sys.path.insert(0, str(_directory))

from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, ROOT, Sandbox, check, have_example

from da6idb_mcp import cache, config, evaluate, tools
from da6idb_mcp.viewer import client

#: The folder the example scan is in, whichever of the two it is.
WHERE = str(EXAMPLE_MASTER.parent)

SCAN = 152
Y = "lambda250k_stats1_total"
GRID = "pva_tise2_2_00152_hkl_grid_d4.vti"

# A server started under this suite has QT_QPA_PLATFORM=offscreen, which
# the client deliberately scrubs from the host's environment so that a
# headless server can still put real windows on a real screen. Here that
# would mean windows appearing while the tests run, so it is put back.
if not config.HOST_PLATFORM:
    config.HOST_PLATFORM = "offscreen"

tools.load()


def _grid_file() -> Path:
    return Path(WHERE) / GRID


def _sandbox_folder(folder: Path) -> Path:
    """A folder holding nothing but the example scan, by symlink.

    The folder-wide tools -- ``scan_list``, ``batch_find``, ``vti_list``
    -- answer about everything they can see, and what ``data/`` can see
    is whatever the user left there. Pointing them at a folder with one
    known scan in it is the only way their counts mean anything.
    """
    for source in (EXAMPLE_MASTER, EXAMPLE_IMAGE):
        os.symlink(source.resolve(), folder / source.name)
    return folder


# -- the surface itself ------------------------------------------------------


def test_the_registry_holds_every_tool_once():
    check("mcp/tools", sorted(tools.registry.names()))
    check("mcp/lanes", {name: spec.lane
                        for name, spec in tools.registry.specs.items()})
    live = sorted(n for n, s in tools.registry.specs.items() if s.live)
    check("mcp/live_tools", live)
    assert all(n.startswith("viewer_") for n in live)


def test_every_tool_can_be_turned_into_a_schema():
    """What FastMCP reads off a tool body has to be there to be read.

    A string annotation -- which is what ``from __future__ import
    annotations`` in a tool module would leave behind -- is the failure
    this catches, and it would otherwise only show up as a bad schema
    advertised to a model at run time.
    """
    for name, spec in tools.registry.specs.items():
        assert inspect.getdoc(spec.fn), f"{name} has no docstring"
        assert spec.lane in {"compute", "viewer", "viewer_read"}, name
        signature = inspect.signature(spec.fn)
        for parameter in signature.parameters.values():
            assert parameter.kind not in (parameter.VAR_POSITIONAL,
                                          parameter.VAR_KEYWORD), \
                f"{name}({parameter.name}) cannot be described in a schema"
            assert parameter.annotation is not inspect.Parameter.empty, \
                f"{name}({parameter.name}) is not annotated"
            assert not isinstance(parameter.annotation, str), \
                f"{name}({parameter.name}) has a string annotation"


class _Recorder:
    """What FastMCP does with a tool, as far as this package can see.

    ``mcp`` itself is an optional dependency -- the whole point of the
    ``[mcp]`` extra -- so the suite must not require it to check that
    registration works. ``add_tool`` is the entire interface
    :func:`tools.register` uses, so standing in for it tests the real
    path rather than a parallel one.
    """

    def __init__(self):
        self.added = {}

    def add_tool(self, fn, name=None, title=None, description=None):
        self.added[name] = (fn, title, description)


def test_registering_attaches_every_tool_with_its_own_signature():
    """The shim has to carry the body's parameters, or the model is blind.

    FastMCP builds the advertised JSON schema from the callable it is
    handed; a wrapper left as ``**kwargs`` would register all 52 tools
    as taking nothing at all, and every one of them would still import
    cleanly.
    """
    recorder = _Recorder()
    added = tools.register(recorder, include_live=True)

    assert added == tools.registry.names()
    assert set(recorder.added) == set(added)
    for name in added:
        shim, _title, description = recorder.added[name]
        body = tools.registry[name].fn
        wrapper, inner = inspect.signature(shim), inspect.signature(body)
        assert list(wrapper.parameters.values()) == \
            list(inner.parameters.values()), name
        assert description == inspect.getdoc(body), name
        assert inspect.iscoroutinefunction(shim), name

        # And no return annotation, anywhere FastMCP might look for one:
        # the bodies all say ``-> Reply``, but the shim hands back the
        # content blocks made out of that Reply, so an output schema
        # built from the annotation would reject every single answer.
        assert wrapper.return_annotation is inspect.Signature.empty, name
        assert "return" not in getattr(shim, "__annotations__", {}), name

    headless = _Recorder()
    without = tools.register(headless, include_live=False)
    assert not any(n.startswith("viewer_") for n in without)
    assert len(added) - len(without) == sum(
        1 for spec in tools.registry.specs.values() if spec.live)


def test_the_command_line_says_what_it_sets():
    """The flags are the config module under another name.

    A client's config file spawns the server with arguments, so these
    are the only knobs most installations ever touch -- and getting one
    of them silently wrong means a server that reads the wrong folder,
    or leaves the escape hatch open on a shared machine.
    """
    from da6idb_mcp import server

    before = (config.DEFAULT_FOLDER, config.ROOTS, config.ALLOW_EVAL,
              config.HOST_PLATFORM)
    try:
        options = server.parse(["--folder", str(ROOT), "--roots", str(ROOT),
                                "--no-eval", "--host-platform", "offscreen"])
        server.apply(options)
        assert config.DEFAULT_FOLDER == str(ROOT)
        assert config.ROOTS == (ROOT.resolve(),)
        assert config.ALLOW_EVAL is False
        assert config.HOST_PLATFORM == "offscreen"

        # And the default is to change nothing, so a bare `python -m
        # da6idb_mcp` is the environment's own settings.
        plain = server.parse([])
        assert (plain.folder, plain.roots, plain.no_eval, plain.no_live,
                plain.host_platform) == (None, None, False, False, None)
    finally:
        (config.DEFAULT_FOLDER, config.ROOTS, config.ALLOW_EVAL,
         config.HOST_PLATFORM) = before


def test_listing_the_tools_needs_nothing_installed():
    """``--list`` is the one thing that works before ``pip install mcp``.

    Which is what makes it the right way to check an installation: the
    server a client spawns imports the same modules, so a name coming
    out here means the tool modules load in that interpreter.
    """
    done = subprocess.run(
        [sys.executable, "-m", "da6idb_mcp", "--list"],
        cwd=str(ROOT), env=dict(os.environ, MPLBACKEND="agg"),
        capture_output=True, timeout=300)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-800:]
    listed = done.stdout.decode().split()
    assert listed == tools.registry.names()

    headless = subprocess.run(
        [sys.executable, "-m", "da6idb_mcp", "--list", "--no-live"],
        cwd=str(ROOT), env=dict(os.environ, MPLBACKEND="agg"),
        capture_output=True, timeout=300)
    assert "viewer_open" not in headless.stdout.decode().split()


# -- headless: the numbers ---------------------------------------------------


def test_scan_tools_answer_with_the_pinned_numbers():
    if not have_example():
        return
    info = tools.result("scan_info", scan=SCAN, where=WHERE)
    check("mcp/scan_info", {"number": info["number"], "plan": info["plan"],
                            "motors": info["motors"], "points": info["points"],
                            "master": Path(info["master"]).name,
                            "image_file": Path(info["image_file"]).name})

    stats = tools.result("scan_statistics", scan=SCAN, where=WHERE, y=Y)
    row = stats["records"][0]
    assert row["max_y"] == 41531.0
    assert round(row["max_x"], 6) == 21.691766
    assert round(row["center"], 6) == 21.722856
    assert round(row["fwhm"], 6) == 0.088734
    check("mcp/scan_statistics", row)

    curve = tools.result("scan_curve", scan=SCAN, where=WHERE, y=Y)
    assert curve["rows"] == 101
    assert curve["records"][0][Y] == 18152.0
    assert curve["figure"] is True

    fit = tools.result("scan_fit", scan=SCAN, where=WHERE, y=Y)
    check("mcp/scan_fit", fit["fit"])

    hkl = tools.result("scan_hkl_position", scan=SCAN, where=WHERE)
    axes = {r["axis"]: r for r in hkl["hkl"]["records"]}
    assert round(axes["H"]["mean"], 6) == 1.013464
    assert round(axes["K"]["mean"], 6) == -0.492560
    assert round(axes["K"]["spread"], 6) == 0.038620
    assert round(axes["L"]["mean"], 6) == 7.484036
    check("mcp/scan_hkl_position", hkl["hkl"]["records"])

    fields = tools.result("scan_fields", scan=SCAN, where=WHERE)
    check("mcp/scan_fields", [r["field"] for r in fields["fields"]["records"]])
    assert fields["fields"]["records"][0]["role"] == "scanned"

    baseline = tools.result("scan_baseline", scan=SCAN, where=WHERE)
    assert baseline["fields"]["rows"] == 220

    with Sandbox() as folder:
        written = tools.result("scan_export", scan=SCAN, where=WHERE, y=Y,
                               path=str(folder / "curve.dat"))
        assert written["rows"] == 101
        assert (folder / "curve.dat").is_file()
        check("mcp/scan_export", written["columns"])


def test_image_tools_answer_with_the_pinned_numbers():
    if not have_example():
        return
    info = tools.result("image_info", scan=SCAN, where=WHERE)
    assert info["frames"] == 101
    assert info["frame_shape"] == [516, 516]
    check("mcp/image_rois", info["rois"]["records"])

    frame = tools.result("image_frame", scan=SCAN, where=WHERE, frame=50)
    assert frame["figure"] is True
    check("mcp/image_frame", {k: frame[k]
                              for k in ("frame", "shape", "min", "max", "total")})

    curve = tools.result("image_roi_curve", scan=SCAN, where=WHERE, roi=0)
    assert curve["label"] == "ROI1.sum"
    assert curve["curve"]["0"] == 18152.0

    series = tools.result("image_roi_series", scan=SCAN, where=WHERE, roi=0)
    check("mcp/image_roi_series", series["series"]["records"][0])

    projection = tools.result("image_projection", scan=SCAN, where=WHERE,
                              mode="max")
    check("mcp/image_projection", projection["brightest_pixel"])
    assert projection["figure"] is True

    two_theta = tools.result("image_two_theta", scan=SCAN, where=WHERE, roi=0)
    assert round(two_theta["two_theta_deg"], 6) == 46.792798

    geometry = tools.result("image_geometry", scan=SCAN, where=WHERE)["geometry"]
    check("mcp/image_geometry", {k: v for k, v in geometry.items()
                                 if not str(k).endswith("_path")})

    channels = tools.result("image_channels", scan=SCAN, where=WHERE)
    assert "eta" in channels["channels"]["columns"]


def test_hkl_tools_answer_with_the_pinned_numbers():
    if not have_example():
        return
    extent = tools.result("hkl_extent", scan=SCAN, where=WHERE)
    low_high = {r["index"]: (r["low"], r["high"])
                for r in extent["extent"]["records"]}
    assert round(low_high["H"][0], 6) == 0.919476
    assert round(low_high["H"][1], 6) == 1.159641
    check("mcp/hkl_extent", extent["extent"]["records"])

    check("mcp/hkl_resolution",
          tools.result("hkl_resolution", scan=SCAN, where=WHERE)["resolution"])
    check("mcp/hkl_sources",
          tools.result("hkl_sources", scan=SCAN, where=WHERE)["sources"])

    cloud = tools.result("hkl_cloud", scan=SCAN, where=WHERE, threshold=100.0)
    assert cloud["points"] == 84419
    check("mcp/hkl_cloud", {"points": cloud["points"],
                            "intensity": cloud["intensity"],
                            "centre_of_mass": cloud["centre_of_mass"]})

    # The range the plan pinned, at the oversampling data/ already holds
    # a file for, so this reuses rather than regrids.
    grid = tools.result("hkl_grid", scan=SCAN, where=WHERE, oversample=4.0,
                        hkl_range=[0.95, 1.05, -0.55, -0.45, 7.45, 7.55])
    assert grid["dimensions"] == [38, 38, 38]
    check("mcp/hkl_grid", {k: grid[k] for k in
                           ("dimensions", "dims", "cells", "measured_cells",
                            "filled_fraction", "max")})

    # offset is measured from the centre of the map, not along the axis
    # in absolute units -- the thing a model gets wrong first.
    sliced = tools.result("hkl_slice", scan=SCAN, where=WHERE, normal=[0, 0, 1])
    assert sliced["figure"] is True
    check("mcp/hkl_slice", {k: sliced[k] for k in
                            ("normal", "offset", "layers", "axes", "shape",
                             "measured_cells")})

    cut = tools.result("hkl_cut", scan=SCAN, where=WHERE,
                       start=[0.95, -0.5, 7.5], end=[1.05, -0.5, 7.5])
    check("mcp/hkl_cut", {"samples": cut["samples"], "measured": cut["measured"],
                          "brightest": cut["brightest"]})

    setup = tools.result("hkl_setup_read", where=WHERE)
    check("mcp/hkl_setup_fields", setup["fields"])


def test_vti_tools_read_a_gridded_file():
    if not have_example() or not _grid_file().is_file():
        return
    info = tools.result("vti_info", path=GRID, where=WHERE)
    assert info["dimensions"] == [91, 106, 138]
    check("mcp/vti_info", {k: info[k] for k in
                           ("dimensions", "cells", "spacing", "origin",
                            "bounds", "centre", "intensity")})

    sliced = tools.result("vti_slice", path=GRID, where=WHERE, normal=[0, 0, 1])
    assert sliced["figure"] is True
    check("mcp/vti_slice", {k: sliced[k] for k in
                            ("normal", "offset", "layers", "axes", "shape",
                             "intensity")})

    cut = tools.result("vti_cut", path=GRID, where=WHERE,
                       start=[0.95, -0.5, 7.5], end=[1.05, -0.5, 7.5])
    check("mcp/vti_cut", {k: cut[k] for k in
                          ("samples", "length", "intensity", "brightest")})

    with Sandbox() as folder:
        shot = tools.result("vti_export_image", path=GRID, where=WHERE,
                            output=str(folder / "slice.png"))
        assert (folder / "slice.png").is_file()
        assert shot["shape"] == [107, 91]
        written = tools.result("vti_export_cut", path=GRID, where=WHERE,
                               start=[0.95, -0.5, 7.5], end=[1.05, -0.5, 7.5],
                               output=str(folder / "cut.dat"))
        assert (folder / "cut.dat").is_file()
        check("mcp/vti_export_cut", sorted(written))


def test_the_folder_tools_see_exactly_what_is_there():
    """Counted against a folder holding one scan, so the counts mean something."""
    if not have_example():
        return
    with Sandbox() as folder:
        _sandbox_folder(folder)
        listed = tools.result("scan_list", where=str(folder))
        check("mcp/scan_list", listed["scans"]["records"])

        found = tools.result("batch_find", where=str(folder))
        assert (found["scans"], found["gridded"]) == (1, 0)
        check("mcp/batch_find", found["found"])

        empty = tools.result("vti_list", where=str(folder))
        assert (empty["count"], empty["files"]) == (0, [])
        assert "hkl_grid" in empty["note"]

    jobs = tools.result("job_list")
    assert jobs["running"] == 0


def test_the_caches_report_and_release_what_they_hold():
    if not have_example():
        return
    tools.call("scan_statistics", scan=SCAN, where=WHERE, y=Y)
    held = tools.result("cache_status")
    assert held["scans"]["held"] >= 1
    assert held["grids"]["cell_limit"] == config.GRID_CELLS
    check("mcp/cache_keys", sorted(k for k in held if k != "note"))

    released = tools.result("cache_clear")
    assert released["released"]["scans"]["held"] >= 1
    after = tools.result("cache_status")
    assert after["scans"]["held"] == 0
    assert after["grids"]["cells"] == 0


# -- the escape hatch --------------------------------------------------------


def test_eval_returns_the_last_expression_and_what_it_printed():
    out = tools.result("da6idb_eval", code="print('hello')\n1 + 1", reset=True)
    assert out["value"] == 2
    assert out["printed"].strip() == "hello"
    assert out["defined"] == []

    # Names persist between calls; that is most of what makes it usable.
    tools.call("da6idb_eval", code="kept = 41531.0")
    again = tools.result("da6idb_eval", code="kept")
    assert again["value"] == 41531.0
    assert "kept" in again["defined"]

    figure = tools.result("da6idb_eval", code="plt.figure()\nplt.plot([1, 2, 3])")
    assert figure["figure"] is True

    fresh = tools.result("da6idb_eval", code="sorted(globals())[:0]", reset=True)
    assert fresh["defined"] == []


def test_eval_shares_the_objects_the_tools_built():
    """The point of the namespace: the same ``Scan``, not a second read."""
    if not have_example():
        return
    tools.call("cache_clear")
    tools.call("scan_statistics", scan=SCAN, where=WHERE, y=Y)
    misses = tools.result("cache_status")["scans"]["misses"]

    out = evaluate.run(
        f"s = cache.scan(resolve.scan_master({SCAN}, {WHERE!r}))\n"
        "type(s).__name__", reset=True)
    assert out["failed"] is None, out["printed"]
    assert out["value"] == "Scan"
    assert tools.result("cache_status")["scans"]["misses"] == misses


def test_eval_refuses_qt_however_it_is_asked():
    """Three ways in, and the one a cached module could slip past.

    ``__import__`` is checked separately from ``sys.meta_path`` because
    a finder is not consulted for a module already in ``sys.modules``,
    and in a server that has been up all day most things are. PyQt5 is
    in this process by the time the whole suite has run, which is
    exactly the case that used to get through.
    """
    for code in ("import PyQt5",                        # read off the source
                 "from pyqtgraph import PlotWidget",
                 "__import__('PyQt5.QtWidgets')",       # the snippet's import
                 "import importlib\n"
                 "importlib.import_module('pyvistaqt')",  # the meta-path guard
                 "import da6idb_mcp.viewer.windows"):
        try:
            evaluate.run(code)
        except evaluate.Forbidden as exc:
            assert "viewer_" in str(exc)
        else:
            raise AssertionError(f"{code!r} was allowed")
    # and the guard does not outlive the snippet that installed it
    assert not [f for f in sys.meta_path if isinstance(f, evaluate._Guard)]


def test_eval_allows_the_pyvista_the_headless_tools_themselves_use():
    """The ban is on windows, not on reading a ``.vti``.

    ``da6idb.hkl.grid`` imports pyvista inside ``load_vti``, which is
    what ``vti_info`` does every time it is called -- so a snippet
    reaching for ``hkl.grid`` must not be refused the import its own
    namespace entry depends on.
    """
    if not have_example() or not _grid_file().is_file():
        return
    out = evaluate.run("import pyvista\npyvista.__name__", reset=True)
    assert out["failed"] is None, out["printed"]
    assert out["value"] == "pyvista"

    out = evaluate.run(
        f"grid = hkl.grid.load_vti({str(_grid_file())!r})\n"
        "type(grid).__name__")
    assert out["failed"] is None, out["printed"]
    check("mcp/eval_load_vti", out["value"])


def test_eval_reports_a_failure_as_data_with_the_snippet_s_own_lines():
    out = evaluate.run("def inner(n):\n    return 100 / n\n\ninner(0)")
    assert out["failed"] == "ZeroDivisionError: division by zero"
    assert evaluate.SOURCE_NAME in out["printed"]
    assert "return 100 / n" in out["printed"]
    assert "evaluate.py" not in out["printed"]


def test_the_import_guard_binds_one_thread_only():
    """A live tool importing the viewer must not be caught by an eval.

    The guard is installed in ``sys.meta_path``, which is global; only
    the thread check keeps it from refusing imports it was never meant
    to see. This is the test that the check is really there.
    """
    outcome = {}

    def slowly():
        outcome["eval"] = evaluate.run(
            "import time\ntime.sleep(1.0)\n'done'", reset=True)

    worker = threading.Thread(target=slowly, daemon=True)
    worker.start()
    time.sleep(0.15)
    installed = [f for f in sys.meta_path if isinstance(f, evaluate._Guard)]
    try:
        # The import under test, on this thread, while that one is guarded.
        assert installed, "the snippet finished too soon to prove anything"
        module = importlib.import_module("da6idb_mcp.viewer.bridge")
        assert module is not None
    finally:
        worker.join(timeout=30)
    assert outcome["eval"]["value"] == "done"


# -- the five windows, one subprocess each -----------------------------------

#: Which windows can be built here, and what each needs to exist first.
WINDOWS = ("scan", "image", "batch", "vti", "hkl")


def _roundtrip_in_a_subprocess(which: str) -> dict:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                       MPLBACKEND="agg")
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), which],
        cwd=str(ROOT), env=environment, capture_output=True, timeout=300)
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = "\n".join(done.stderr.decode("utf-8", "replace").splitlines()[-8:])
    raise AssertionError(
        f"the {which} window said nothing (exit {done.returncode})\n{tail}")


def test_every_window_round_trips_its_own_state():
    """``w.set_state(w.state())`` changes nothing, for all five windows.

    The one test that keeps the live layer honest: ``viewer_set_state``
    is the only way a model writes to a window, and it is only safe to
    offer if what comes out goes back in unchanged.
    """
    if not have_example():
        return
    for which in WINDOWS:
        if which == "vti" and not _grid_file().is_file():
            continue
        out = _roundtrip_in_a_subprocess(which)
        assert out["differences"] == [], f"{which}: {out['differences']}"
        assert out["applied"] == len(out["settings"]), which
        assert "colourmap" in out["unknown"], which
        check(f"mcp/state_{which}", out["settings"])


# -- the live layer ----------------------------------------------------------


def _live_or_skip():
    """The host, or ``None`` on a machine that cannot start one."""
    try:
        client.host().connect()
    except client.NoDisplay:
        return None
    return client.host()


def _state(window) -> dict:
    """A window's settings, as the window itself reports them.

    ``.data`` rather than ``tools.result``: the payload a model sees
    carries the reply's *note* alongside the settings, and feeding that
    straight back to ``viewer_set_state`` would -- rightly -- be refused
    as a setting that does not exist.
    """
    return tools.call("viewer_state", window=window).data


def _stop_the_host():
    try:
        tools.call("viewer_close")
    except Exception:                                    # noqa: BLE001
        pass
    client.host().stop()


def test_the_live_layer_opens_and_drives_a_real_window():
    if not have_example() or _live_or_skip() is None:
        return
    try:
        idle = tools.result("viewer_list")
        assert len(idle["kinds"]) == 5

        opened = tools.result("viewer_open", kind="image", target=str(SCAN),
                              where=WHERE)
        window = opened["window"]
        assert opened["kind"] == "image"
        assert opened["has_3d"] is False

        # state -> set_state -> state, through the socket this time
        before = _state(window)
        settings = {k: v for k, v in before.items() if k != "about"}
        applied = tools.result("viewer_set_state", window=window,
                               settings=settings)
        assert len(applied["applied"]) == len(settings)
        assert _state(window) == before

        tools.call("viewer_set_state", window=window, settings={"frame": 7})
        assert _state(window)["frame"] == 7

        try:
            tools.call("viewer_set_state", window=window,
                       settings={"colourmap": 1})
        except client.ViewerError as exc:
            assert "colormap" in str(exc)     # the real key, offered back
        else:
            raise AssertionError("an unknown setting was accepted")

        # viewer_call: the verbs settings cannot express
        tools.call("viewer_call", window=window, method="add_roi",
                   kwargs={"pos": [100, 100], "size": [40, 40], "name": "peak"})
        added = [roi for roi in _state(window)["rois"]
                 if roi["name"] == "peak"]
        assert added and (added[0]["x"], added[0]["w"]) == (100.0, 40.0)

        try:
            tools.call("viewer_call", window=window, method="explode")
        except client.ViewerError as exc:
            assert "add_roi" in str(exc)      # its own methods, offered back
        else:
            raise AssertionError("a method that is not there was called")

        shot = tools.call("viewer_screenshot", window=window)
        assert shot.picture[:4] == b"\x89PNG"
        assert 0 < len(shot.picture) <= config.MAX_PNG_BYTES

        closed = tools.result("viewer_close")
        assert closed["open"] == []
        assert window in closed["closed"]
    finally:
        _stop_the_host()


def test_a_read_is_served_while_the_window_is_busy_and_a_write_is_not():
    """The re-entrancy guarantee, which is not a theory about Qt.

    ``batch_grid`` calls ``processEvents`` while it grids, so a queued
    bridge task really is dispatched mid-run. That is what makes a
    screenshot of a running job possible -- and what makes refusing a
    *writing* call necessary, since it would change what the run is
    doing underneath it.

    It really grids, which is the only way to get a run long enough to
    interrupt: this rewrites ``output/pva_tise2_2_00152_hkl_grid.vti``
    beside the data, which is a derived file the window itself would
    rewrite anyway.
    """
    if not have_example() or _live_or_skip() is None:
        return
    try:
        opened = tools.result("viewer_open", kind="batch", target=WHERE)
        window = opened["window"]
        # Finer than the default so the run lasts long enough to be
        # interrupted, and with the cache off so it really runs.
        tools.call("viewer_set_state", window=window,
                   settings={"scans": str(SCAN), "reuse": False,
                             "oversample": 2.0})

        ran = {}
        host = client.host()

        def grid():
            try:
                ran["value"] = host.call("call", window=window, method="run",
                                         timeout=600)
            except Exception as exc:                     # noqa: BLE001
                ran["error"] = repr(exc)

        worker = threading.Thread(target=grid, daemon=True)
        worker.start()

        running = False
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not running:
            running = _state(window)["about"]["running"]
            if not running:
                time.sleep(0.05)
        assert running, "the run was over before a read could be served"

        shot = tools.call("viewer_screenshot", window=window)
        assert len(shot.picture) > 0

        try:
            tools.call("viewer_set_state", window=window,
                       settings={"reuse": True})
        except client.ViewerError as exc:
            assert "busy" in str(exc).lower()
        else:
            raise AssertionError("a writing call landed mid-run")

        worker.join(timeout=600)
        assert "error" not in ran, ran.get("error")
    finally:
        _stop_the_host()


def test_no_display_is_one_sentence_and_the_headless_tools_carry_on():
    """What the server does on a machine with no desktop session.

    The host is made to fail by naming a Qt platform that does not
    exist, which is the same shape of failure as no display at all: the
    process aborts rather than raising, and the only evidence is its
    exit code and whatever it wrote on the way down.
    """
    if not have_example():
        return
    client.host().stop()
    was = config.HOST_PLATFORM
    config.HOST_PLATFORM = "there-is-no-such-platform"
    try:
        started = time.monotonic()
        said = []
        for name, arguments in (("viewer_open", {"kind": "scan",
                                                 "target": WHERE}),
                                ("viewer_state", {"window": "scan1"}),
                                ("viewer_screenshot", {"window": "scan1"})):
            try:
                tools.call(name, **arguments)
            except client.NoDisplay as exc:
                said.append(str(exc).splitlines()[0])
            else:
                raise AssertionError(f"{name} worked without a display")

        assert len(set(said)) == 1, said
        assert "headless" in said[0] or "exited" in said[0]
        # Remembered rather than respawned: three tools, one dead process.
        assert time.monotonic() - started < config.HOST_STARTUP

        statistics = tools.result("scan_statistics", scan=SCAN, where=WHERE,
                                  y=Y)
        assert statistics["records"][0]["max_y"] == 41531.0
    finally:
        config.HOST_PLATFORM = was
        client.host().stop()


#: A host that answers the hello and then dies, like Qt aborting on the
#: first real window. Written as a stub rather than by breaking the real
#: host because the point is the timing: up long enough to be connected
#: to, gone by the time it is asked for anything.
_DIES_AFTER_HELLO = """\
import os, sys
from multiprocessing.connection import Connection
link = Connection(int(sys.argv[1]))
link.send({"id": 0, "ok": True, "value": {"windows": []}})
link.recv()
sys.stderr.write("qt.qpa.screen: could not connect to display\\n")
sys.stderr.flush()
os.abort()
"""


def test_a_host_that_dies_after_the_hello_says_the_same_sentence():
    """The failure the designed error message was written for.

    ``NoDisplay`` used to be reached only from start-up, and a host can
    fail later than that: ``QApplication`` constructs and the hello goes
    out, and the abort comes when cocoa is actually asked for a window.
    That is the ordinary case on a headless server, and it surfaced as a
    bare ``the viewer process has gone (EOFError)`` -- true, unhelpful,
    and not the sentence every other no-display path gives.
    """
    host = client.Host()

    def spawn():
        host._log = tempfile.TemporaryFile()
        ours, theirs = Pipe(duplex=True)
        try:
            host._process = subprocess.Popen(
                [sys.executable, "-c", _DIES_AFTER_HELLO,
                 str(theirs.fileno())],
                pass_fds=(theirs.fileno(),), stdin=subprocess.DEVNULL,
                stdout=host._log, stderr=host._log)
        finally:
            theirs.close()
        host._connection = ours
        while not ours.poll(0.05):
            pass

    host._spawn = spawn
    try:
        try:
            host.call("open", timeout=10, kind="scan")
        except client.NoDisplay as exc:
            said = str(exc)
        else:
            raise AssertionError("a dead host answered")

        assert said.splitlines()[0].startswith("the viewer process exited")
        assert "headless" in said
        # The log tail comes with it: without that there is nothing to
        # tell a missing display from any other reason for the abort.
        assert "could not connect to display" in said

        # And it is remembered, so the next tool answers from the
        # diagnosis rather than starting another host that cannot live.
        started = time.monotonic()
        try:
            host.call("list", timeout=10)
        except client.NoDisplay as again:
            assert str(again) == said
        else:
            raise AssertionError("it tried again and succeeded")
        assert time.monotonic() - started < 1.0
    finally:
        host.stop()


def test_a_failure_the_tool_saw_coming_reaches_the_model_in_words():
    """Half the error messages in this package are wasted without this.

    The SDK treats an escaping exception as a crash: it logs the
    traceback its end and tells the model only ``Error executing tool
    scan_info``. So the shim raises the failures it anticipated as the
    SDK's own ``ToolError``, which carries the text through -- and the
    list of what counts as anticipated stays short, because a genuine
    bug *should* be reported as a crash rather than as advice.
    """
    try:
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError:
            return                      # no SDK: nothing to serve with
    import anyio

    from da6idb_mcp import server

    mcp, _names = server.build()

    async def ask():
        try:
            await mcp.call_tool("scan_info", {"scan": 9999, "where": WHERE})
        except ToolError as exc:
            return str(exc)
        return ""

    said = anyio.run(ask)
    assert "9999" in said, said
    assert "Scans there" in said, said


# -- the subprocess side of the round trip -----------------------------------


def _build(which, folder: Path):
    """One window of each kind, with enough loaded to have a state."""
    if which == "scan":
        from scan_viewer import HdfViewer
        window = HdfViewer(folder)
        window.select_scans([f"{SCAN:05d}"])
        return window
    if which == "image":
        from image_viewer import ImageViewerWindow
        window = ImageViewerWindow([EXAMPLE_IMAGE])
        window.add_roi(pos=(100, 100), size=(40, 40), name="peak")
        return window
    if which == "batch":
        from batch_grid import BatchGridWindow
        window = BatchGridWindow(folder)
        window.scans_edit.setText(str(SCAN))
        return window
    if which == "vti":
        from vti_viewer import VtiWindow
        return VtiWindow(folder / GRID)
    if which == "hkl":
        from hkl_viewer import HklWindow
        return HklWindow(EXAMPLE_IMAGE)
    raise SystemExit(f"no window called {which!r}")


def _differences(before, after, path=""):
    """Every leaf the two dicts disagree on, named by its path."""
    out = []
    for key in sorted(set(before) | set(after)):
        here = f"{path}{key}"
        if key not in before or key not in after:
            out.append(f"{here}: only in {'before' if key in before else 'after'}")
            continue
        one, two = before[key], after[key]
        if isinstance(one, dict) and isinstance(two, dict):
            out.extend(_differences(one, two, f"{here}."))
        elif one != two:
            out.append(f"{here}: {one!r} -> {two!r}")
    return out


def _roundtrip(which) -> int:
    from PyQt5 import QtWidgets

    application = QtWidgets.QApplication(sys.argv[:1])
    window = _build(which, Path(WHERE))

    before = window.state()
    applied = window.set_state(before)
    after = window.state()

    # A partial dict leaves what it does not mention alone, and an
    # invented key says so rather than being quietly dropped.
    name = next(n for n in before if n != "about")
    window.set_state({name: before[name]})
    try:
        window.set_state({"colourmap": 1})
    except ValueError as exc:
        unknown = str(exc)
    else:
        unknown = ""

    print("RESULT " + json.dumps({
        "which": which,
        "settings": sorted(n for n in before if n != "about"),
        "applied": len(applied),
        "differences": _differences(before, after),
        "unknown": unknown,
    }))
    # The window outlives *application* on purpose: pyqtgraph's view
    # boxes unregister themselves as they are destroyed, which needs the
    # application still there to do it against.
    window.close()
    assert application is not None
    return 0


if __name__ == "__main__":
    sys.exit(_roundtrip(sys.argv[1] if len(sys.argv) > 1 else "scan"))
