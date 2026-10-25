"""``python -m da6idb_mcp`` -- start the server on stdio.

The whole of it is in :mod:`da6idb_mcp.server`; this is the name a
client's config file points at.
"""

import sys

from da6idb_mcp.server import main

if __name__ == "__main__":
    sys.exit(main())
