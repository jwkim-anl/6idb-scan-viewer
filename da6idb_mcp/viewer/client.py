"""The server's end: start the GUI process, talk to it, survive its death.

One host process, started the first time a live tool is called and kept
for the rest of the session. They talk over a ``multiprocessing``
``Pipe`` -- a socket pair, one end inherited by the child -- which
pickles dicts and PNG bytes without a protocol to invent. A pair rather
than a named socket in ``$TMPDIR`` for three reasons: nothing is bound,
so a sandbox that forbids binding does not forbid this; there is no
file to leave behind or to guess the name of; and the only process that
can ever connect is the one we handed the descriptor to, which is a
stronger guarantee than an auth key.

Replies are matched to requests by number and delivered by a reader
thread, which is not ceremony: the host serves a screenshot *while* a
batch gridding run is going, so the second answer sent is routinely the
first one to arrive.

When there is no desktop session the host does not raise, it dies --
Qt calls ``abort()`` rather than returning an error when it cannot talk
to the window server. So failure is measured by the process going away,
its standard error is kept, and every live tool repeats it in one
sentence. That verdict is remembered for a minute, so a model that tries
six live tools in a row does not spawn six doomed processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent import futures
from multiprocessing.connection import Pipe
from pathlib import Path

from da6idb_mcp import config

#: Added to the host's own ceiling before this end gives up, so that the
#: host's more informative timeout is the one that gets reported.
GRACE = 5.0


class ViewerError(RuntimeError):
    """Something the window said no to. *kind* is what it raised there."""

    def __init__(self, message, kind="RuntimeError", detail=""):
        super().__init__(message)
        self.kind = kind
        self.detail = detail


class NoDisplay(RuntimeError):
    """The GUI process could not be started, so no window can be opened."""


def _child_environment() -> dict:
    """The host's environment: like ours, but able to open a window.

    The server may well have been started with ``QT_QPA_PLATFORM`` set to
    ``offscreen`` -- the test suite does exactly that -- and inheriting it
    would give windows nobody can see. Setting ``DA6IDB_MCP_HOST_PLATFORM``
    puts it back deliberately, which is how the live layer is tested.
    """
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    if config.HOST_PLATFORM:
        env["QT_QPA_PLATFORM"] = config.HOST_PLATFORM
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


class Host:
    """The viewer process, however many times it has to be started."""

    def __init__(self):
        self._lock = threading.RLock()
        self._connection = None
        self._process = None
        self._log = None
        self._pending: dict = {}
        self._id = 0
        self._refused = None          # (when, why) -- a remembered failure
        self.hello: dict = {}

    # -- starting -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return (self._connection is not None and self._process is not None
                and self._process.poll() is None)

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _spawn(self):
        # Unnamed: the log is only ever read back through this handle, by
        # _why_dead, and a named one would need removing by hand -- which
        # the old mkdtemp here did not do, leaving a directory per spawn.
        self._log = tempfile.TemporaryFile()
        ours, theirs = Pipe(duplex=True)
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "da6idb_mcp.viewer.host",
                 str(theirs.fileno())],
                cwd=str(self._repo_root()), env=_child_environment(),
                pass_fds=(theirs.fileno(),),
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log,
            )
        finally:
            # Ours only from here: were this end left open, the reader
            # thread would never see the pipe close when the host died.
            theirs.close()
        self._connection = ours

        # Qt start-up, the window modules and -- on a bad machine -- the
        # abort all happen before the first byte. Watching the process
        # too means a host that cannot reach a display is reported in a
        # second rather than at the end of the startup allowance.
        deadline = time.monotonic() + config.HOST_STARTUP
        while not ours.poll(0.05):
            if self._process.poll() is not None:
                raise NoDisplay(self._why_dead())
            if time.monotonic() > deadline:
                raise NoDisplay(
                    f"the viewer process did not answer within "
                    f"{config.HOST_STARTUP:g} s. Headless tools are "
                    f"unaffected."
                )

    def _hello(self) -> dict:
        """The host's first message, or why there is never going to be one.

        A readable pipe is not a speaking one: ``poll`` is true at
        end-of-file as well as on data, and end-of-file is precisely what
        a host that aborted on its way up leaves behind. So the first
        read is also the second half of the startup check.
        """
        try:
            return self._unwrap(self._connection.recv(), 0)
        except (EOFError, OSError):
            if self._process is not None:
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            raise NoDisplay(self._why_dead()) from None

    def _why_dead(self) -> str:
        """The one sentence a model gets when there is no display."""
        tail = ""
        if self._log is not None:
            try:
                self._log.flush()
                self._log.seek(0)
                tail = self._log.read().decode("utf-8", "replace").strip()
            except OSError:
                tail = ""
        tail = "\n".join(tail.splitlines()[-12:])
        code = self._process.poll() if self._process else None
        return (
            f"the viewer process exited ({code}) before it could open a "
            "window. This machine has no desktop session, or Qt cannot "
            "reach it. Every headless tool still works -- scan_*, image_*, "
            "hkl_* and vti_* need no window at all."
            + (f"\n\n{tail}" if tail else "")
        )

    def _died(self) -> NoDisplay:
        """Diagnose a host that has gone away mid-conversation, and forget it.

        :meth:`_why_dead` is the sentence :meth:`connect` gives for a host
        that never started; this is the other half of the same story. Qt
        ``abort()``s rather than raising when cocoa cannot reach a desktop
        session, and that abort does not always come at start-up: the
        ``QApplication`` can be built and the hello sent, and the process
        die only when a window is actually shown. Without this, the most
        ordinary failure there is -- a headless server, a plain ssh session
        -- arrived as a bare *"the viewer process has gone (EOFError)"*
        instead of the one sentence the live layer promises. The diagnosis
        is remembered in ``_refused`` as well, so the next live tool answers
        from it rather than spawning another doomed process.
        """
        with self._lock:
            if self._process is not None:
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            why = self._why_dead()
            self._refused = (time.monotonic(), why)
            self._kill()
        return NoDisplay(why)

    def _kill(self):
        for closer in (lambda: self._connection and self._connection.close(),
                       lambda: self._process and self._process.kill(),
                       lambda: self._log and self._log.close()):
            try:
                closer()
            except Exception:
                pass
        self._connection = self._process = self._log = None

    def connect(self):
        """The live connection, starting the host if there is not one."""
        with self._lock:
            if self.running:
                return self._connection
            if self._refused is not None:
                when, why = self._refused
                if time.monotonic() - when < config.NO_DISPLAY_TTL:
                    raise NoDisplay(why)
                self._refused = None

            self._kill()
            self._pending.clear()
            try:
                self._spawn()
                # The hello is read here rather than by the reader
                # thread, which is not fussiness: the thread delivers a
                # reply to whoever is already waiting for its number,
                # and nobody is yet waiting for this one. So read it,
                # then start the thread, and every reply after this one
                # has a claimant.
                self.hello = self._hello()
            except NoDisplay as exc:
                self._refused = (time.monotonic(), str(exc))
                self._kill()
                raise

            threading.Thread(target=self._read, args=(self._connection,),
                             name="da6idb-replies", daemon=True).start()
            return self._connection

    # -- talking ------------------------------------------------------------

    def _read(self, connection):
        """Deliver replies to whoever is waiting for them, until the end."""
        reason = "the viewer process has gone"
        while True:
            try:
                reply = connection.recv()
            except (EOFError, OSError) as exc:
                reason = f"{reason} ({type(exc).__name__})"
                break
            waiting = self._pending.pop(reply.get("id"), None)
            if waiting is not None and not waiting.done():
                waiting.set_result(reply)
        for waiting in list(self._pending.values()):
            if not waiting.done():
                # kind="gone" so that `call` can tell this apart from a
                # window that merely said no, and go and find out why.
                waiting.set_exception(ViewerError(reason, kind="gone"))
        self._pending.clear()

    def _expect(self, request_id) -> futures.Future:
        waiting: futures.Future = futures.Future()
        self._pending[request_id] = waiting
        return waiting

    def _unwrap(self, reply, request_id):
        if reply.get("ok"):
            return reply.get("value")
        raise ViewerError(reply.get("error", "the window refused"),
                          kind=reply.get("type", "RuntimeError"),
                          detail=reply.get("traceback", ""))

    def call(self, command: str, timeout=None, **args):
        """Run one command in the host and return what it gave back."""
        connection = self.connect()
        timeout = (config.timeout_for(f"viewer_{command}")
                   if timeout is None else float(timeout))
        with self._lock:
            self._id += 1
            request_id = self._id
            waiting = self._expect(request_id)
            try:
                connection.send({"id": request_id, "command": command,
                                 "args": args})
            except (EOFError, OSError) as exc:
                self._pending.pop(request_id, None)
                raise self._died() from exc
        try:
            reply = waiting.result(timeout + GRACE)
        except futures.TimeoutError:
            self._pending.pop(request_id, None)
            raise TimeoutError(
                f"the window has not answered {command} in {timeout:g} s. "
                "It was not cancelled; try viewer_list to see whether it is "
                "still working."
            ) from None
        except ViewerError as exc:
            if exc.kind != "gone":
                raise
            raise self._died() from exc
        return self._unwrap(reply, request_id)

    def stop(self):
        """Shut the host down. The next live tool starts a fresh one."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.send("bye")
                except Exception:
                    pass
            was = self._process.pid if self._process else None
            self._kill()
            self._refused = None
            return was


_host = Host()


def host() -> Host:
    """The one host process this server talks to."""
    return _host


# -- resolving, before anything is spawned -----------------------------------


def targets(kind_name, target=None, where=None):
    """What a window should be built from, as absolute paths.

    Done here rather than in the host for two reasons: the folder rules
    and the "no scan 153 -- here are the ones there are" errors live in
    :mod:`da6idb_mcp.resolve` and should not be written twice, and a
    mistyped scan number should cost a sentence rather than a GUI
    process started and thrown away.
    """
    from da6idb.files import image_path_for_master

    from da6idb_mcp import resolve
    from da6idb_mcp.viewer import windows

    kind = windows.kind(kind_name)

    if kind.takes == "folder":
        return [str(resolve.folder(target if target is not None else where))], []
    if kind.takes == "vti":
        if target is None:
            raise ValueError(
                "a vti window opens a gridded file: pass target='...vti', or "
                "vti_list to see what has been gridded")
        return [str(resolve.vti(target, where))], []

    if target is None:
        raise ValueError(f"a {kind_name} window opens {windows.WANTS[kind.takes]}"
                         "; pass target=152, say")
    wanted = (resolve.scan_masters(target, where) if kind.takes == "files"
              else [resolve.scan_master(target, where)])

    paths, masters = [], []
    for master in wanted:
        image = image_path_for_master(master)
        if image is None:
            raise ValueError(
                f"{master.name} has no detector file beside it, so there are "
                "no frames for this window to show")
        paths.append(str(image))
        masters.append(str(master))
    return paths, masters
