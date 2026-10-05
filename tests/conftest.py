"""Makes the suite work under pytest as well as `python tests/run.py`.

The tests import their helpers and the modules under test by plain name, so
both directories go on the path, and the Qt and matplotlib backends are set
before anything imports them.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "agg")
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
