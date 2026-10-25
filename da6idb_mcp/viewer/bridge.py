"""Getting a call from a socket onto the Qt main thread, and back.

Qt owns the main thread of the host process: a widget may only be
touched from it, and the windows here go further than that -- pyqtgraph
and VTK both keep state that assumes it. Meanwhile the socket has to be
read by somebody, and that somebody cannot be the thread sitting in
``app.exec_()``.

So: a reader thread reads requests, and every request is handed to the
main thread as a queued signal carrying a :class:`~concurrent.futures.Future`.
A ``Future`` rather than a queue of results, so that an exception in a
window arrives at the caller as *that* exception, with its type and
message, rather than as a timeout nobody can explain.

Two things this has to get right that a first draft would not:

* **A long command must not deafen the host.** ``batch_grid`` and the HKL
  conversion call ``QApplication.processEvents()`` as they work, which is
  what keeps their progress bars alive -- and which means a queued task
  *will* be dispatched in the middle of one. That is a feature here: it
  is how a screenshot of a gridding run in progress is possible at all.
  Each request therefore waits on its own thread, so the reader is never
  the one blocked.
* **And must not be interfered with.** A command that would change what
  the running one is doing is refused, by name and by how long it has
  been going, rather than queued behind it or run underneath it.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures

from PyQt5 import QtCore


class Busy(RuntimeError):
    """A command that writes, arriving while another one is running."""


class _Task:
    __slots__ = ("command", "fn", "read_only", "future")

    def __init__(self, command, fn, read_only):
        self.command = command
        self.fn = fn
        self.read_only = read_only
        self.future = futures.Future()


class Bridge(QtCore.QObject):
    """Runs callables on the Qt main thread for whoever asks.

    Built on the main thread, called from any other.
    """

    _submit = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Queued, explicitly: the default would be a direct call whenever
        # the sender happened to be the main thread, which is exactly the
        # case this class exists to make impossible to get wrong.
        self._submit.connect(self._run, QtCore.Qt.QueuedConnection)
        self._lock = threading.Lock()
        self._busy = None

    # -- the caller's side (any thread) ------------------------------------

    def busy(self):
        """``(command, seconds)`` if a writing command is running."""
        with self._lock:
            if self._busy is None:
                return None
            command, started = self._busy
            return command, time.monotonic() - started

    def call(self, command, fn, read_only=False, timeout=60.0):
        """Run *fn* on the main thread and return what it returned."""
        running = self.busy()
        if running is not None and not read_only:
            name, seconds = running
            raise Busy(
                f"the window is busy with {name}, started {seconds:.0f} s "
                f"ago; {command} would change what it is doing. Wait for it, "
                f"or ask for something read-only -- viewer_state, "
                f"viewer_list and viewer_screenshot are served meanwhile."
            )

        task = _Task(command, fn, read_only)
        self._submit.emit(task)
        try:
            return task.future.result(timeout)
        except futures.TimeoutError:
            raise TimeoutError(
                f"{command} has not finished after {timeout:g} s. It is "
                f"still running in the window -- nothing was cancelled, and "
                f"the next call may well find it done."
            ) from None

    # -- the Qt main thread's side -----------------------------------------

    def _run(self, task):
        if not task.read_only:
            with self._lock:
                self._busy = (task.command, time.monotonic())
        try:
            task.future.set_result(task.fn())
        except BaseException as exc:         # noqa: BLE001 -- it is relayed
            task.future.set_exception(exc)
        finally:
            if not task.read_only:
                with self._lock:
                    self._busy = None
