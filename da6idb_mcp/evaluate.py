"""The escape hatch: run a snippet against the same objects the tools use.

Forty-odd tools cover what is asked for often. They cannot cover what a
beamline scientist thinks of at two in the morning, and a server that
answers "there is no tool for that" to a question Python answers in three
lines is the wrong kind of careful.

So: a namespace with ``np``, ``pd``, ``xr``, ``plt``, every ``da6idb``
submodule, and -- the part that matters -- :mod:`da6idb_mcp.cache`, so a
snippet gets the *same* ``Scan`` and ``Images`` objects the curated tools
built rather than re-reading the files. Names persist between calls, so a
map built in one snippet is still there in the next.

Two rules, and they are different in kind:

* it can be switched off entirely, ``DA6IDB_MCP_NO_EVAL=1``, for a server
  that several people share;
* it will not import PyQt5, pyvistaqt, pyqtgraph or the viewer modules.
  That is not timidity about GUIs. The whole design rests on no Qt
  living in the server process -- Qt aborts rather than raises, and a
  render window's fault is a segfault -- so a snippet that imported one
  could take the server down with every headless tool in it. Plain
  ``pyvista`` and ``vtk`` are *not* banned: they are already here, since
  that is what reads and writes a ``.vti``, and refusing them would
  refuse ``hkl.grid.load_vti`` to the snippet the namespace binds it
  for. The ban is checked three ways -- in the source before anything
  runs, in ``sys.meta_path`` while it runs, and in the snippet's own
  ``__import__``, which is the one a cached module cannot slip past.
  Reaching into ``sys.modules`` by hand still gets there; this stops the
  mistake, not a determined author, who is the user anyway.

Otherwise it is as privileged as the user's own shell, which is what it
is: the same files, the same process, the same account. The *timeout* is
reported rather than enforced, because Python cannot kill a thread that
will not stop, and pretending otherwise would be a lie in a docstring.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import io
import linecache
import threading
import time
import traceback

from da6idb_mcp import config

#: Never importable from a snippet: everything that would put Qt in this
#: process, directly or by importing something that does.
#:
#: Plain ``pyvista``, ``vtk`` and ``vtkmodules`` are deliberately *not*
#: here. They are already in this process -- ``da6idb.hkl.grid`` imports
#: pyvista to read and write ``.vti`` files, which is what ``vti_info``
#: and ``hkl_grid`` do all day -- and banning them would refuse a
#: snippet the very module the namespace binds as ``hkl.grid``. What
#: cannot survive here is a *render window*, and that arrives with
#: ``pyvistaqt``, not with the array classes.
FORBIDDEN = (
    "PyQt5", "PyQt6", "PySide2", "PySide6", "sip", "shiboken2",
    "pyvistaqt", "pyqtgraph",
    "da6idb_mcp.viewer",
    "scan_viewer", "image_viewer", "hkl_viewer", "vti_viewer",
    "batch_grid", "slice_plot", "viewer_state",
)

#: Said the same way wherever the refusal comes from.
REFUSAL = (
    "da6idb_eval will not import {name}: Qt must stay out of the server "
    "process, which is what keeps a window's graphics fault from taking "
    "the headless tools down with it. The viewer_* tools drive the real "
    "windows, in the process built to survive them."
)

#: Submodules bound by name in the namespace, so ``hkl.grid.load_vti``
#: and ``batch.parse_scans`` work without an import line.
SUBMODULES = ("batch", "combine", "export", "files", "fit", "images",
              "rois", "scan", "session", "trends", "hkl")

_HKL = ("convert", "detector", "geometry", "grid", "slice")

#: The file name a snippet's frames carry, in tracebacks and in
#: ``linecache``, which is what makes its own lines show up in them.
SOURCE_NAME = "<da6idb_eval>"


class Forbidden(ImportError):
    """A snippet tried to import something that must stay out of here."""


def _is_forbidden(name: str) -> bool:
    return any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)


class _Guard:
    """A meta-path finder that refuses the banned modules, on one thread.

    On one thread because this is installed globally for the few
    milliseconds a snippet runs, and a live tool on another thread may
    legitimately be importing :mod:`da6idb_mcp.viewer` at that moment.
    Refusing *its* import would be a bug caused entirely by the guard.
    """

    def __init__(self, thread_id: int):
        self.thread = thread_id

    def find_module(self, name, path=None):      # pragma: no cover - legacy
        self.find_spec(name, path)
        return None

    def find_spec(self, name, path=None, target=None):
        if threading.get_ident() == self.thread and _is_forbidden(name):
            raise Forbidden(REFUSAL.format(name=name))
        return None

    def __enter__(self):
        import sys

        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        import sys

        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self)
        return False


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """``__import__`` for a snippet, refusing the banned names.

    The finder above is not consulted for a module already in
    ``sys.modules``, and by the time a snippet runs, a long-lived server
    has imported a great deal. So the ban is also applied here, where
    the cache cannot route around it -- which matters most for exactly
    the case it is written for: the host process has Qt loaded, and a
    QWidget built on this thread would abort the server.
    """
    if level == 0 and _is_forbidden(name):
        raise Forbidden(REFUSAL.format(name=name))
    return builtins.__import__(name, globals, locals, fromlist, level)


def _static_check(tree) -> None:
    """Say no before running, where the source says what it will import.

    The guard above is the real enforcement; this is the good error
    message, and it arrives before any of the snippet has run.
    """
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        for name in names:
            if _is_forbidden(name):
                raise Forbidden(REFUSAL.format(name=name))


def _build_namespace() -> dict:
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    import da6idb
    from da6idb_mcp import cache, config as configuration, render, resolve
    from da6idb_mcp import tools as tool_module

    space = {
        "__name__": "da6idb_eval",
        # Everything builtins has, with one name replaced; functions the
        # snippet defines inherit these globals, so they are covered too.
        "__builtins__": dict(vars(builtins), __import__=_guarded_import),
        "np": np, "pd": pd, "plt": plt,
        "da6idb": da6idb,
        # The two things a snippet is most likely to want first.
        "open_scan": da6idb.open_scan, "open_folder": da6idb.open_folder,
        # The point of the exercise: the objects the tools already built.
        "cache": cache, "resolve": resolve, "render": render,
        "config": configuration,
        "call": tool_module.call, "result": tool_module.result,
    }
    with contextlib.suppress(ImportError):
        import xarray as xr

        space["xr"] = xr
    from pathlib import Path

    space["Path"] = Path

    for name in SUBMODULES:
        with contextlib.suppress(ImportError):
            space[name] = importlib.import_module(f"da6idb.{name}")
    hkl = space.get("hkl")
    for name in _HKL:
        with contextlib.suppress(ImportError):
            setattr(hkl, name, importlib.import_module(f"da6idb.hkl.{name}"))
    return space


#: Kept between calls, so a snippet can build on the last one.
_namespace: dict = {}
_provided: set = set()
_namespace_lock = threading.RLock()


def namespace(reset: bool = False) -> dict:
    global _namespace, _provided
    with _namespace_lock:
        if reset or not _namespace:
            _namespace = _build_namespace()
            _provided = set(_namespace)
        return _namespace


def defined() -> list:
    """Names a snippet has left behind, ignoring the ones we handed it."""
    with _namespace_lock:
        return sorted(name for name in _namespace
                      if name not in _provided and not name.startswith("_"))


def _snippet_traceback(exc) -> str:
    """The traceback with this module's own frames taken off the top.

    Nobody debugging a snippet wants to read about ``exec`` and the
    guard; they want the line they wrote. ``linecache`` is primed with
    the source above, so that line is actually shown.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    mine = [frame for frame in frames if frame.filename == SOURCE_NAME]
    return "".join(["Traceback (most recent call last):\n"]
                   + traceback.format_list(mine or frames)
                   + traceback.format_exception_only(type(exc), exc))


def _split(source: str):
    """The body to exec and the trailing expression to eval, if there is one."""
    tree = ast.parse(source, filename=SOURCE_NAME, mode="exec")
    _static_check(tree)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
        ast.fix_missing_locations(last)
    return tree, last


def run(source: str, reset: bool = False) -> dict:
    """Run *source*; return its value, what it printed, and how long it took.

    Raises :class:`Forbidden` when eval is switched off or the snippet
    reaches for Qt, and lets every other exception through as a formatted
    traceback in the result -- a failing snippet is information, not a
    tool failure, and the model should see the line number.
    """
    if not config.ALLOW_EVAL:
        raise Forbidden(
            "da6idb_eval is switched off on this server (DA6IDB_MCP_NO_EVAL). "
            "The other tools are unaffected.")

    space = namespace(reset=reset)
    tree, last = _split(source)
    # So a traceback can show the line that failed: there is no file for
    # the usual machinery to go and read.
    linecache.cache[SOURCE_NAME] = (len(source), None,
                                    source.splitlines(True), SOURCE_NAME)
    printed = io.StringIO()
    started = time.monotonic()
    value = None
    failure = None

    with _Guard(threading.get_ident()):
        with contextlib.redirect_stdout(printed):
            try:
                exec(compile(tree, SOURCE_NAME, "exec"), space)
                if last is not None:
                    value = eval(compile(last, SOURCE_NAME, "eval"), space)
            except Forbidden:
                raise
            except BaseException as exc:                  # noqa: BLE001
                failure = "".join(traceback.format_exception_only(
                    type(exc), exc)).strip()
                printed.write("\n" + _snippet_traceback(exc))

    return {
        "value": value,
        "printed": printed.getvalue(),
        "failed": failure,
        "seconds": round(time.monotonic() - started, 3),
    }
