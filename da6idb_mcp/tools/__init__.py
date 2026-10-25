"""The registry every tool is written against, and the one decorator.

A tool body is an ordinary synchronous function that returns a
:class:`~da6idb_mcp.render.Reply`. :func:`tool` records it here; nothing
about MCP happens until :func:`register` is handed a FastMCP instance by
:mod:`da6idb_mcp.server`. That is what lets the whole tool surface be
imported, called and tested in an environment with no ``mcp`` installed::

    from da6idb_mcp import tools
    tools.load()
    reply = tools.call("scan_statistics", scan=152, where="data")

Writing a tool, three rules:

* name the function after the tool -- ``def scan_statistics(...)`` -- so
  that grepping for what a model asked for finds the code;
* **do not** put ``from __future__ import annotations`` in a tool module,
  and annotate with real objects (``int``, ``str | None``, ``float``). The
  schema MCP advertises is read off these, and a string annotation would
  have to be resolved in a module this one cannot see;
* give it a docstring written for the model: what it returns, in what
  units, and which argument it will get wrong if unguided.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field

#: Tool modules, in the order their tools should be advertised.
MODULES = ("scan", "image", "hkl", "vti", "batch", "live", "admin")


@dataclass
class Spec:
    """One tool: the function, and how the server should run it."""

    name: str
    fn: object
    lane: str = "compute"
    #: Live tools are hidden when no GUI host can be started.
    live: bool = False
    title: str = ""

    @property
    def description(self) -> str:
        return inspect.getdoc(self.fn) or ""


@dataclass
class Registry:
    specs: dict = field(default_factory=dict)
    loaded: bool = False

    def add(self, spec: Spec) -> None:
        if spec.name in self.specs:
            raise RuntimeError(f"two tools are called {spec.name!r}")
        self.specs[spec.name] = spec

    def names(self) -> list:
        return list(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name) -> bool:
        return name in self.specs

    def __getitem__(self, name) -> Spec:
        try:
            return self.specs[name]
        except KeyError:
            near = ", ".join(sorted(n for n in self.specs if n[:4] == name[:4]))
            raise KeyError(
                f"no tool {name!r}" + (f". Did you mean: {near}" if near else "")
            ) from None


registry = Registry()


def tool(lane: str = "compute", live: bool = False, title: str = ""):
    """Register a tool body.

    *lane* decides which capacity limiter the server runs it under:
    ``"compute"`` has two slots, so one long gridding call cannot starve a
    quick curve, and ``"viewer"`` has one, because the GUI host has a
    single event loop and serialising here gives a better error than
    queueing there. ``"viewer_read"`` is the exception to that: asking a
    window what it is showing, or for a picture of it, is served by the
    host even mid-run, so queueing those behind a gridding call here
    would throw away the one thing the bridge works hardest to allow.
    *live* marks a tool that needs that host at all.
    """

    def decorate(fn):
        registry.add(Spec(name=fn.__name__, fn=fn, lane=lane, live=live,
                          title=title))
        return fn

    return decorate


def load() -> Registry:
    """Import every tool module once, so the registry is complete."""
    import importlib

    if not registry.loaded:
        for name in MODULES:
            importlib.import_module(f"da6idb_mcp.tools.{name}")
        registry.loaded = True
    return registry


# -- calling, without MCP in the way -----------------------------------------


def call(name: str, **kwargs):
    """Run a tool body and get its :class:`Reply`. For tests and eval."""
    load()
    return registry[name].fn(**kwargs)


def result(name: str, **kwargs) -> dict:
    """Run a tool and get the plain payload, with any figure dropped.

    The shape a test pins: all the numbers, none of the PNG bytes.
    """
    from da6idb_mcp import render

    reply = call(name, **kwargs)
    payload = reply.payload()
    if reply.figure is not None:
        payload["figure"] = True
        render.close(reply.figure)
    elif reply.picture:
        payload["figure"] = True
        payload["png_bytes"] = len(reply.picture)
    return payload


# -- the MCP side, reached only when actually serving ------------------------


def _content(reply):
    """A reply as the list of content blocks FastMCP understands.

    Returning the dict and the picture as a list lets FastMCP do the
    encoding -- the JSON text block and the base64 image block -- rather
    than this file building protocol objects that move between versions.
    """
    from da6idb_mcp import _sdk, render

    Image = _sdk.image_class()

    blocks = [reply.payload()]
    if reply.figure is not None:
        try:
            blocks.append(Image(data=render.png(reply.figure), format="png"))
        finally:
            render.close(reply.figure)
    elif reply.picture:
        # Already PNG, and already within budget: a screenshot is encoded
        # in the host, where the pixmap it came from lives.
        blocks.append(Image(data=reply.picture, format="png"))
    return blocks


def _expected():
    """The exception types whose message is an answer, not a bug.

    A tool that cannot find scan 152, is pointed outside its roots, or is
    told there is no desktop session has not malfunctioned: it has worked
    out something the model needs to read and act on. Those are raised to
    the SDK as its own ``ToolError``, which sends the text through. Any
    other exception is a defect in this package, and the SDK is right to
    log the traceback its end and tell the model only that the tool
    failed -- so the list stays short and deliberate rather than becoming
    ``except Exception``.
    """
    from da6idb_mcp.evaluate import Forbidden
    from da6idb_mcp.viewer.client import NoDisplay, ViewerError

    return (ValueError,        # ResolveError, and every bad argument
            KeyError,          # an unknown field, scan or window
            FileNotFoundError,
            TimeoutError,      # a window that has not answered
            NoDisplay, ViewerError, Forbidden)


def _shim(spec, limiters):
    """An async wrapper that runs the body off the event loop.

    The signature is copied over deliberately: FastMCP reads the JSON
    schema it advertises off the wrapped function, so the model sees the
    tool body's own parameters, defaults and docstring rather than
    ``**kwargs``.

    The *return* annotation is just as deliberately thrown away. Every
    body is declared ``-> Reply``, and from 1.10 FastMCP turns a return
    annotation into an output schema and then validates against it --
    which this would fail, because what the wrapper actually hands back
    is the list of content blocks ``_content`` built out of that Reply.
    No annotation, no output schema, and the blocks go out as they are.

    The expected failures are translated on the way out, for the reason
    :func:`_expected` gives: all the trouble those messages go to -- *no
    scan 152 in data/. Scans there: 148, 150, 152* -- is wasted if the
    model only ever sees ``Error executing tool scan_curve``.
    """
    import anyio.to_thread

    from da6idb_mcp import _sdk
    from da6idb_mcp.render import Reply

    ToolError = _sdk.tool_error_class()
    expected = _expected()
    fn = spec.fn

    @functools.wraps(fn)
    async def run(**kwargs):
        limiter = limiters[spec.lane]
        bound = functools.partial(fn, **kwargs)
        try:
            reply = await anyio.to_thread.run_sync(bound, limiter=limiter)
        except expected as exc:
            raise ToolError(str(exc) or type(exc).__name__) from exc
        if not isinstance(reply, Reply):
            reply = Reply(data=reply)
        return _content(reply)

    run.__signature__ = inspect.signature(fn).replace(
        return_annotation=inspect.Signature.empty)
    # functools.wraps copied the body's annotations wholesale, and
    # get_type_hints reads this dict rather than the signature.
    run.__annotations__ = {name: annotation
                           for name, annotation in fn.__annotations__.items()
                           if name != "return"}
    return run


def register(mcp, include_live: bool = True) -> list:
    """Attach every registered tool to a FastMCP server. Returns the names."""
    import anyio

    from da6idb_mcp import config

    load()
    limiters = {
        "compute": anyio.CapacityLimiter(config.COMPUTE_SLOTS),
        "viewer": anyio.CapacityLimiter(1),
        "viewer_read": anyio.CapacityLimiter(config.VIEWER_READ_SLOTS),
    }

    added = []
    for name, spec in registry.specs.items():
        if spec.live and not include_live:
            continue
        mcp.add_tool(_shim(spec, limiters), name=name, title=spec.title or None,
                     description=spec.description)
        added.append(name)
    return added
