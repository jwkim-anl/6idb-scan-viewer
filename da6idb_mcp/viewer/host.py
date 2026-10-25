"""The GUI process: a Qt event loop with a socket attached.

Started by :mod:`~da6idb_mcp.viewer.client`, never by hand: the one
argument is the number of a file descriptor the client passed down, and
there is nothing to connect to without it::

    python -m da6idb_mcp.viewer.host <fd>

The main thread does what a Qt program's main thread does and nothing
else. A reader thread takes requests off the socket and hands each to a
small pool, so that one slow command -- a batch gridding run is minutes
-- does not stop the next one being read; the bridge decides which of
them may actually proceed while it is going.

The windows are the repository's own modules, imported here and nowhere
in the server. If importing them fails, or Qt cannot reach a display and
aborts outright, this process dies and its standard error is what the
client shows the model. That is the arrangement: everything that can
take a process down with it happens in this one.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection

#: Requests served at once. Generous: all but one of them are waiting.
WORKERS = 8


def _reply_for(request_id, exc) -> dict:
    """A failure as something that is certain to survive the pickle."""
    return {
        "id": request_id,
        "ok": False,
        "type": type(exc).__name__,
        "error": str(exc) or repr(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def serve(connection, desk, app) -> None:
    """Read requests until the server goes away, then quit the loop."""
    writing = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=WORKERS,
                              thread_name_prefix="da6idb-host")

    def send(reply):
        with writing:
            try:
                connection.send(reply)
            except (EOFError, OSError, BrokenPipeError):
                pass
            except Exception as exc:
                # Almost always something in the value that will not
                # pickle. The call did happen; say so, rather than leave
                # the caller waiting for a reply that cannot be sent.
                connection.send(_reply_for(reply.get("id"), exc))

    def run(request):
        try:
            value = desk.handle(request["command"], request.get("args") or {})
        except BaseException as exc:                      # noqa: BLE001
            send(_reply_for(request.get("id"), exc))
        else:
            send({"id": request.get("id"), "ok": True, "value": value})

    try:
        while True:
            try:
                request = connection.recv()
            except (EOFError, OSError):
                break
            if request == "bye":
                break
            pool.submit(run, request)
    finally:
        pool.shutdown(wait=False)
        app.quit()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or not argv[0].isdigit():
        print("usage: python -m da6idb_mcp.viewer.host <fd>", file=sys.stderr)
        return 2
    descriptor = int(argv[0])

    # The repository's window modules sit beside the package, not in it.
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)

    from PyQt5 import QtWidgets

    from da6idb_mcp.viewer.bridge import Bridge
    from da6idb_mcp.viewer.commands import Desk

    app = QtWidgets.QApplication(sys.argv[:1])
    # Closing the last window must not end the process: a model may well
    # close one window and open another a minute later, and respawning
    # costs it the Qt start-up and every window it had.
    app.setQuitOnLastWindowClosed(False)

    desk = Desk(Bridge())
    connection = Connection(descriptor)
    # The first thing sent, and the client's signal that Qt came up and
    # the window modules imported: everything that can kill this process
    # has already had its chance by now.
    connection.send({"id": 0, "ok": True, "value": desk.handle("hello", {})})

    threading.Thread(target=serve, args=(connection, desk, app),
                     name="da6idb-host-reader", daemon=True).start()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
