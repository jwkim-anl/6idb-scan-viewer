"""Work that may outlast one tool call.

Gridding a scan takes between half a second and several minutes, and
which it will be is not knowable before starting: a cached ``.vti`` that
already matches comes back at once, a fresh 150-cubed lattice does not.
So the gridding tools start the work, wait a little, and either return
the answer or a job id -- the model is never left holding a call that
might block for ten minutes, and never made to poll for one that took a
second.

Cancellation is real, not advisory, because the package already supports
it: :func:`da6idb.hkl.grid.grid_to_vti` takes a *progress* callback and
raises :class:`~da6idb.hkl.grid.GridCancelled` the moment it returns
false. :meth:`Job.watch` is that callback.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import dataclass, field

from da6idb_mcp import config
from da6idb_mcp.render import Reply

_counter = itertools.count(1)
_jobs: dict = {}
_lock = threading.Lock()


@dataclass
class Job:
    """One piece of background work and everything known about it."""

    id: str
    name: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    state: str = "running"
    note: str = ""
    fraction: float | None = None
    result: object = None
    error: str = ""
    _stop: threading.Event = field(default_factory=threading.Event)
    _done: threading.Event = field(default_factory=threading.Event)

    # -- for the worker --------------------------------------------------

    def watch(self, fraction=None, note: str = "") -> bool:
        """Report progress and ask whether to carry on.

        Shaped as ``da6idb``'s own *progress* callback: returning false is
        what makes a grid stop, so passing this straight in is all the
        cancellation support a long call needs.
        """
        if fraction is not None:
            try:
                self.fraction = float(fraction)
            except (TypeError, ValueError):
                pass
        if note:
            self.note = str(note)
        return not self._stop.is_set()

    def progress(self, label: str = ""):
        """A callback shaped the way ``da6idb``'s long calls expect one.

        :func:`~da6idb.hkl.grid.grid_to_vti` calls ``progress(done, total)``
        and :func:`~da6idb.batch.grid_scans` wraps that as
        ``progress(scan, done, total)``; both stop on a false return. One
        adapter serves both so neither caller has to care which it got.
        """

        def report(*args) -> bool:
            if len(args) == 3:
                scan, done, total = args
                where = f"scan {scan}, frame {done} of {total}"
            else:
                done, total = args
                where = f"frame {done} of {total}"
            fraction = (done / total) if total else None
            return self.watch(fraction, f"{label}{where}")

        return report

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    # -- for the caller --------------------------------------------------

    def cancel(self) -> None:
        self._stop.set()

    def wait(self, seconds: float) -> bool:
        """Block up to *seconds*. True if it has finished."""
        return self._done.wait(max(0.0, float(seconds)))

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def status(self) -> dict:
        return {
            "job": self.id,
            "tool": self.name,
            "state": self.state,
            "elapsed_s": round(self.elapsed, 2),
            "progress": self.fraction,
            "note": self.note,
            "error": self.error,
        }


def _sweep() -> None:
    """Forget finished jobs nobody came back for."""
    cutoff = time.time() - config.JOB_TTL
    for key, job in list(_jobs.items()):
        if job.finished is not None and job.finished < cutoff:
            _jobs.pop(key, None)


def submit(name: str, work) -> Job:
    """Start *work* on a thread. It is called with the :class:`Job`."""
    job = Job(id=f"{name}-{next(_counter)}", name=name)

    def run():
        try:
            job.result = work(job)
            job.state = "cancelled" if job.cancelled else "done"
        except Exception as problem:
            # GridCancelled arrives here when the stop flag was set, and
            # that is a cancellation rather than a failure.
            if job.cancelled and type(problem).__name__ == "GridCancelled":
                job.state = "cancelled"
            else:
                job.state = "failed"
                job.error = f"{type(problem).__name__}: {problem}"
                job.note = traceback.format_exc(limit=3)
        finally:
            job.finished = time.time()
            job.fraction = 1.0 if job.state == "done" else job.fraction
            job._done.set()

    with _lock:
        _sweep()
        _jobs[job.id] = job
    threading.Thread(target=run, name=job.id, daemon=True).start()
    return job


def collect(job: Job) -> Reply:
    """A finished job's answer, as the tool that started it would return it.

    The figure is handed out once and then let go: the tool decorator
    renders and closes it, so a second poll would otherwise find an empty
    canvas and quietly return a blank picture.
    """
    if job.state == "failed":
        raise RuntimeError(f"{job.name} failed. {job.error}")
    if job.state == "cancelled":
        return Reply(data=job.status(), note="cancelled; nothing was written")

    reply = job.result
    if not isinstance(reply, Reply):
        return Reply(data=reply)
    out = Reply(data=reply.data, figure=reply.figure, note=reply.note,
                artefacts=list(reply.artefacts))
    reply.figure = None
    return out


def run(name: str, work, wait_seconds=None) -> Reply:
    """Start *work*, wait a little, and answer with whichever came first.

    The common case is that the work was quick -- a grid already on disk
    reloads in a second -- and then this is indistinguishable from having
    called it directly. Only the genuinely long runs turn into a job id.
    """
    wait = config.DEFAULT_WAIT if wait_seconds is None else float(wait_seconds)
    job = submit(name, work)
    if job.wait(wait):
        return collect(job)
    return Reply(
        data=job.status(),
        note=(f"still going after {wait:g} s and left running. "
              f"job_status('{job.id}') for the answer, "
              f"job_cancel('{job.id}') to stop it"),
    )


def get(job_id: str) -> Job:
    with _lock:
        job = _jobs.get(str(job_id))
    if job is None:
        known = ", ".join(sorted(_jobs)) or "none running"
        raise KeyError(f"no job {job_id!r}. Jobs: {known}")
    return job


def listing() -> list:
    with _lock:
        return [job.status() for job in _jobs.values()]
