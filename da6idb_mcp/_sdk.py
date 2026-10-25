"""Where the MCP SDK keeps the two classes this package needs.

The SDK renamed ``FastMCP`` to ``MCPServer`` in 2.0 and moved it from
``mcp.server.fastmcp`` to ``mcp.server.mcpserver``. Both names are in
the wild -- a beamline machine installed last year has 1.x, one set up
today gets 2.x -- and the parts this package actually touches did not
change at all across that rename:

    MCPServer(name, instructions=...)
    .add_tool(fn, name=..., title=..., description=...)
    .run()                        # stdio
    Image(data=..., format="png")
    ToolError("...")              # a failure the model should read

So the rename is the whole of the incompatibility, and two import
attempts are the whole of the fix. Everything else in the package asks
here rather than importing the SDK itself, which also keeps the import
in one place -- it is an optional dependency, and the tool modules must
stay importable without it for ``--list`` and for the tests.
"""

from __future__ import annotations

#: Told to the user when neither name is there.
MISSING = (
    "da6idb-mcp needs the MCP SDK: pip install 'da6idb[mcp]'\n"
    "  (the tools themselves do not -- `python -m da6idb_mcp --list` "
    "works without it.)"
)


def server_class():
    """``MCPServer`` on 2.x, ``FastMCP`` on 1.x. Raises ImportError if neither."""
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError:
        raise ImportError(MISSING) from None


def image_class():
    """The SDK's image wrapper, from whichever module holds it."""
    try:
        from mcp.server.mcpserver import Image

        return Image
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import Image

        return Image
    except ImportError:
        raise ImportError(MISSING) from None


class ToolFailed(RuntimeError):
    """Stand-in for the SDK's ``ToolError`` where there is no SDK.

    Registration is tested without ``mcp`` installed, and the shim built
    by :func:`da6idb_mcp.tools._shim` names the class it will raise as it
    is built. Nothing serves over the protocol in that situation, so what
    the class *is* does not matter -- only that there is one.
    """


def tool_error_class():
    """The exception that puts its own message in front of the model.

    Everything else the SDK catches is treated as a crash: it logs the
    traceback its end and sends ``Error executing tool <name>`` and
    nothing more, which is the right answer for a bug and a useless one
    for *"no scan 152 in data/"*. Tools raise this for the failures they
    saw coming -- see :func:`da6idb_mcp.tools._shim`.
    """
    try:
        from mcp.server.mcpserver.exceptions import ToolError

        return ToolError
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp.exceptions import ToolError

        return ToolError
    except ImportError:
        return ToolFailed


def version() -> str:
    """The installed SDK's version, for the start-up line. '' if unknown."""
    import importlib.metadata as metadata

    try:
        return metadata.version("mcp")
    except Exception:                                    # noqa: BLE001
        return ""
