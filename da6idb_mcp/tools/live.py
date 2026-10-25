"""The real windows: open them, read them, drive them, photograph them.

Everything else in this server computes. These seven tools operate the
five PyQt5 viewers themselves, in a separate process with a desktop
session -- which is the only way to get a true 3D volume render, and the
only way to leave something on screen for the person at the beamline to
go on using after the conversation ends.

Deliberately generic. A window's controls are read and written as one
dict by `viewer_state` and `viewer_set_state`, rather than through forty
setters: the dict is symmetric, so what comes out goes back in, and a
window that gains a control gains it here for nothing. `viewer_call`
covers the verbs a dict cannot hold -- *run*, *export*, *fill* -- by
calling the window's own method, the same one its button calls.

Nothing here works on a machine with no display, and every one of these
tools will say so plainly rather than hang. The headless namespaces
(`scan_*`, `image_*`, `hkl_*`, `vti_*`, `batch_*`) need no window at all.
"""

from typing import Optional

from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool
from da6idb_mcp.viewer import client, windows


def _host():
    return client.host()


@tool(lane="viewer_read", live=True)
def viewer_list() -> Reply:
    """The windows open right now, and the kinds that can be opened.

    Every other tool here names a window by the short name this gives --
    ``scan1``, ``image2`` -- including windows nobody asked for: opening
    a scan browser and clicking a scan puts a detector panel up, and it
    appears here as soon as it exists.

    Asking costs nothing when no window is open: the GUI process is not
    started until something is actually opened.
    """
    host = _host()
    if not host.running:
        return Reply(data={"running": False, "windows": [],
                           "kinds": windows.catalogue()},
                     note="no viewer process yet; viewer_open starts one")
    open_now = host.call("list")
    return Reply(data={"running": True, "count": len(open_now),
                       "windows": open_now, "kinds": windows.catalogue()},
                 note="" if open_now else "the viewer process is up but has "
                                          "no windows open")


@tool(lane="viewer", live=True)
def viewer_open(kind: str, target: Optional[str] = None,
                where: Optional[str] = None, show: bool = True) -> Reply:
    """Open one of the five viewer windows on the screen.

    *kind* is ``scan``, ``image``, ``hkl``, ``vti`` or ``batch``;
    `viewer_list` describes what each is for. *target* is what it opens:
    a folder for ``scan`` and ``batch`` (the default folder when
    omitted), a scan number for ``image`` and ``hkl`` -- ``image`` takes
    several, as ``"152, 155"`` or ``"152-158"`` -- and a ``.vti`` file
    name for ``vti``.

    The window stays open and keeps working after this returns, so it is
    worth remembering that the person at the beamline can see it. An
    ``hkl`` window converts its scan on opening, which takes a minute or
    so for a large one; a ``vti`` window shows an existing map at once.
    """
    paths, masters = client.targets(kind, target, where)
    opened = _host().call("open", kind=kind, paths=paths, masters=masters,
                          show=bool(show))
    return Reply(data=opened,
                 note=f"use window={opened['window']!r} for the other "
                      f"viewer_* tools"
                      + ("" if opened["has_3d"] else
                         "; it has no 3D view, so screenshots are of the "
                         "whole window"))


@tool(lane="viewer", live=True)
def viewer_close(window: Optional[str] = None) -> Reply:
    """Close one window, or every window when none is named.

    The GUI process itself stays up, so opening the next window is quick.
    Nothing is saved on the way out: a window is a view of files on disk,
    and anything a tool wrote is already written.
    """
    done = _host().call("close", window=window)
    still = ", ".join(done["open"]) or "none"
    return Reply(data=done,
                 note=f"closed {len(done['closed'])}; still open: {still}")


@tool(lane="viewer_read", live=True)
def viewer_state(window: str) -> Reply:
    """Every control of a window, as a dict `viewer_set_state` takes back.

    The settable keys are the window's controls -- which fields are
    plotted, the colour map and its limits, the slice plane, the ROIs,
    the HKL box -- and ``about`` is the read-only part: which files are
    open, what the data allows, whether a run is going. Put the dict back
    unchanged and nothing moves; change one key and only that changes.

    Served even while the window is busy with something long, so this is
    how to find out where a gridding run has got to.
    """
    state = _host().call("state", window=window)
    settings = [key for key in state if key != "about"]
    return Reply(data=state,
                 note=f"{len(settings)} settings: {', '.join(settings)}")


@tool(lane="viewer", live=True)
def viewer_set_state(window: str, settings: dict) -> Reply:
    """Drive a window by handing it part of its own state dict.

    Pass only the keys to change. Each is written to its control exactly
    as a person moving it would, so the window redraws, recomputes and
    re-exports off the back of it just the same.

    An unknown key is refused with the list of real ones rather than
    quietly dropped, and a value outside what a control allows is refused
    with the range -- so a rejection here is information, not a failure.
    """
    done = _host().call("set_state", window=window, state=dict(settings))
    return Reply(data=done,
                 note=f"applied: {', '.join(done['applied']) or 'nothing'}")


@tool(lane="viewer_read", live=True)
def viewer_screenshot(window: str, region: Optional[str] = None) -> Reply:
    """A picture of a window as it is on screen right now.

    *region* is ``"window"`` for the whole thing, controls included, or
    ``"view"`` for the 3D render alone in the two windows that have one
    -- which is the only way to see a real volume rendering, since that
    needs a graphics context this server has not got. The default is the
    3D view where there is one and the whole window otherwise.

    Taken even mid-run, so it is also the way to watch a long job.
    """
    shot = _host().call("screenshot", window=window, region=region)
    width, height = shot["size"]
    return Reply(data={"window": shot["window"], "region": shot["region"],
                       "size": shot["size"]},
                 picture=shot["png"],
                 note=f"{width}x{height} as grabbed; scaled down to fit "
                      f"the reply")


@tool(lane="viewer", live=True)
def viewer_call(window: str, method: str, args: Optional[list] = None,
                kwargs: Optional[dict] = None) -> Reply:
    """Call a method on a window: the actions settings cannot express.

    What `viewer_set_state` is to the controls, this is to the buttons.
    The useful ones:

    * ``batch``: ``fill_list`` (write the scan list from the From/To
      boxes), ``range_from_first`` (set the HKL box from the first scan),
      ``run`` -- minutes, and it holds the window while it goes;
    * ``scan``: ``export_curves(path)``, ``export_trend(path)``,
      ``select_scans([152, 155])``;
    * ``hkl``: ``load(image_path, master_path)``, ``recompute()``;
    * ``image``: ``add_roi(pos, size, name)``, ``set_frame(n)``.

    Pass a path to any export and it writes there silently; leave it out
    and a file dialog opens for the person at the screen, which is not
    what you want from here. A method name that does not exist comes back
    with the list of the ones that do.
    """
    done = _host().call("call", window=window, method=method,
                        args=list(args or []), kwargs=dict(kwargs or {}))
    return Reply(data=done,
                 note="the window may still be redrawing; viewer_screenshot "
                      "shows what it looks like now")
