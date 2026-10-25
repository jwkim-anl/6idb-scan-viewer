"""What the host can be asked to do, and what it does about it.

Every command arrives on a worker thread, is handed to the Qt main
thread by the :mod:`~da6idb_mcp.viewer.bridge`, and comes back as plain
data -- numbers, strings, PNG bytes -- because the only things that can
cross the socket are the things ``pickle`` understands, and a widget is
not one of them.

The registry of open windows lives here. It is not only what this
process opened: a scan viewer opens detector panels of its own, and an
image panel opens HKL windows, so :meth:`Desk.adopt` sweeps Qt's own
list of top-level widgets and gives a name to any of the five it finds.
A model that opens a scan browser, clicks through to a frame and then
asks for the state of what appeared is asking a reasonable question.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from da6idb_mcp import config, render
from da6idb_mcp.viewer import windows
from da6idb_mcp.viewer.bridge import Bridge

#: Which kind a window belongs to, by the name of its class.
BY_CLASS = {kind.factory: name for name, kind in windows.KINDS.items()}

#: Commands that may run while something slower is still going.
READ_ONLY = {"hello", "list", "state", "screenshot"}

#: Beyond this, a screenshot is scaled down rather than sent whole.
SCREENSHOT_WIDTH = 1400


def _encode(pixmap, budget: int) -> bytes:
    """A pixmap as PNG bytes, narrowed until it fits *budget*.

    A window grabbed on a retina screen is 3000 px across and several
    megabytes; what reaches the model has to be readable, not faithful.
    """
    if pixmap.width() > SCREENSHOT_WIDTH:
        pixmap = pixmap.scaledToWidth(SCREENSHOT_WIDTH,
                                      QtCore.Qt.SmoothTransformation)
    for _ in range(5):
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        data = bytes(buffer.data())
        if len(data) <= budget or pixmap.width() <= 500:
            return data
        pixmap = pixmap.scaledToWidth(int(pixmap.width() * 0.75),
                                      QtCore.Qt.SmoothTransformation)
    return data


def _methods(window) -> list:
    """The window's own public methods, without Qt's three hundred.

    ``dir()`` on a ``QMainWindow`` is mostly ``setContentsMargins`` and
    its friends, which nobody came here for. What is interesting is what
    the viewer module itself defined, so the classes Qt contributed are
    passed over -- by the module they come from, since that is the one
    thing that distinguishes them reliably.
    """
    names = set()
    for cls in type(window).__mro__:
        if getattr(cls, "__module__", "").startswith(("PyQt5", "sip",
                                                      "builtins")):
            continue
        names.update(name for name, value in vars(cls).items()
                     if not name.startswith("_") and callable(value))
    return sorted(names)


def _alive(window) -> bool:
    """Whether Qt still has the widget this Python object wraps."""
    try:
        window.objectName()
    except RuntimeError:
        return False
    return True


class Desk:
    """The open windows, and the commands that act on them."""

    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self.windows: dict = {}
        self._counts: dict = {}

    # -- naming -------------------------------------------------------------

    def _name(self, kind: str) -> str:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        return f"{kind}{self._counts[kind]}"

    def _register(self, kind: str, window) -> str:
        name = self._name(kind)
        self.windows[name] = window
        return name

    def adopt(self) -> None:
        """Name any of the five that appeared without being asked for.

        Also forgets the ones Qt has destroyed, which is the other half
        of the same sweep: a window the user closed should stop being
        offered, and its Python wrapper is all that is left of it.
        """
        self.windows = {name: window for name, window in self.windows.items()
                        if _alive(window)}
        known = set(map(id, self.windows.values()))
        for widget in QtWidgets.QApplication.topLevelWidgets():
            kind = BY_CLASS.get(type(widget).__name__)
            if kind is not None and id(widget) not in known and _alive(widget):
                self._register(kind, widget)

    def _window(self, name):
        self.adopt()
        try:
            return self.windows[str(name)]
        except KeyError:
            open_now = ", ".join(self.windows) or "none"
            raise ValueError(
                f"no window called {name!r}. Open: {open_now}"
            ) from None

    # -- the commands themselves (all on the Qt main thread) ----------------

    def _do_hello(self):
        return {"pid": os.getpid(), "kinds": list(windows.KINDS)}

    def _describe(self, name, window) -> dict:
        return {
            "window": name,
            "kind": BY_CLASS.get(type(window).__name__, "?"),
            "title": window.windowTitle(),
            "visible": bool(window.isVisible()),
            "has_3d": getattr(window, "plotter", None) is not None,
            # So that viewer_call can be used without first calling it
            # wrong and reading the error.
            "methods": _methods(window),
        }

    def _do_list(self):
        self.adopt()
        return [self._describe(name, window)
                for name, window in self.windows.items()]

    def _do_open(self, kind, paths, masters=(), show=True):
        window = windows.build(kind, paths, masters)
        name = self._register(kind, window)
        if show:
            window.show()
            window.raise_()
            window.activateWindow()
        # Anything the new window opened for itself -- a scan browser is
        # given a folder and may put a detector panel up with it -- gets
        # a name now rather than on the next call.
        self.adopt()
        return self._describe(name, window)

    def _do_close(self, window=None):
        if window is None:
            self.adopt()
            wanted = list(self.windows)
        else:
            self._window(window)       # raises, with the list, if unknown
            wanted = [str(window)]
        closed = []
        for name in wanted:
            widget = self.windows.pop(name, None)
            if widget is not None and _alive(widget):
                widget.close()
                closed.append(name)
        return {"closed": closed, "open": list(self.windows)}

    def _do_state(self, window):
        widget = self._window(window)
        if not hasattr(widget, "state"):
            raise ValueError(f"{type(widget).__name__} has no settings to read")
        return widget.state()

    def _do_set_state(self, window, state):
        widget = self._window(window)
        if not hasattr(widget, "set_state"):
            raise ValueError(f"{type(widget).__name__} has no settings to set")
        return {"applied": widget.set_state(state)}

    def _do_screenshot(self, window, region=None):
        widget = self._window(window)
        plotter = getattr(widget, "plotter", None)
        if region is None:
            region = "view" if plotter is not None else "window"
        if region not in {"view", "window"}:
            raise ValueError(f"region is 'view' or 'window', not {region!r}")

        if region == "view":
            if plotter is None:
                raise ValueError(
                    f"{window} has no 3D view; ask for region='window'")
            # Grabbing a GL widget gives black on macOS, so the render
            # window is asked for its own picture. It writes a file, and
            # that file is then put through the same budget as the rest.
            with tempfile.TemporaryDirectory() as folder:
                shot = Path(folder) / "view.png"
                plotter.screenshot(str(shot))
                pixmap = QtGui.QPixmap(str(shot))
        else:
            pixmap = widget.grab()

        if pixmap.isNull():
            raise RuntimeError(f"nothing was drawn for {window}")
        return {
            "window": str(window),
            "region": region,
            "size": [pixmap.width(), pixmap.height()],
            "png": _encode(pixmap, config.MAX_PNG_BYTES),
        }

    def _do_call(self, window, method, args=(), kwargs=None):
        """Call a method on a window -- the actions settings cannot express.

        Starting a batch run, exporting the curves on screen, filling the
        HKL box from the first scan: each is a verb, not a setting, and
        each already exists as a method because a button needed it.
        """
        widget = self._window(window)
        name = str(method)
        if name.startswith("_"):
            raise ValueError(
                f"{name} is private to {type(widget).__name__}; the methods "
                "meant to be called from outside do not start with _"
            )
        fn = getattr(widget, name, None)
        if not callable(fn):
            raise ValueError(f"{type(widget).__name__} has no method {name}. "
                             f"Methods: {', '.join(_methods(widget))}")
        value = fn(*list(args), **dict(kwargs or {}))
        return {"window": str(window), "method": name,
                "returned": render.to_jsonable(value)}

    # -- dispatch (worker threads) -----------------------------------------

    def handle(self, command: str, args: dict):
        """Run one request, on the Qt main thread, and return its value."""
        fn = getattr(self, f"_do_{command}", None)
        if fn is None:
            offered = ", ".join(sorted(
                n[4:] for n in dir(self) if n.startswith("_do_")))
            raise ValueError(f"no command {command!r}. Commands: {offered}")
        if command == "hello":
            return fn()
        return self.bridge.call(
            command,
            lambda: fn(**args),
            read_only=command in READ_ONLY,
            timeout=config.timeout_for(f"viewer_{command}"),
        )
