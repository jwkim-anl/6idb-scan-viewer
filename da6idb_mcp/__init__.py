"""An MCP server over the 6-ID-B analysis package and its five viewers.

Two layers, deliberately kept apart:

* the **headless** tools compute through :mod:`da6idb` and return numbers
  together with a rendered PNG, so a model asking for a rocking curve sees
  the curve as well as its statistics;
* the **live** tools open the real PyQt windows and steer them, in a
  separate process (:mod:`da6idb_mcp.viewer`) that this one spawns.

Importing this package costs nothing: ``mcp`` itself, Qt and VTK are all
reached lazily, so the tool bodies and the tests run in an environment that
has only the analysis package installed.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["Reply", "registry"]


def __getattr__(name):
    if name == "Reply":
        from da6idb_mcp.render import Reply

        return Reply
    if name == "registry":
        from da6idb_mcp.tools import registry

        return registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
