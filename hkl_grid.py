"""Moved into :mod:`da6idb.hkl.grid`.

Kept so the windows, the batch gridder and anything already importing this
can go on doing so; the code itself lives in the package now.
"""

from __future__ import annotations

import sys

from da6idb.hkl.grid import (  # noqa: F401
    GridCancelled,
    MAX_CELLS,
    _resolve_range,
    _sampling_vectors,
    default_output,
    ensure_grid,
    geometry_fingerprint,
    grid_to_vti,
    load_vti,
    main,
    make_drawable,
    matches_range,
    measured_range,
    natural_step,
    sampling_cell,
    sampling_footprint,
)


if __name__ == "__main__":
    sys.exit(main())
