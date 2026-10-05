"""Run the characterisation tests without needing pytest installed.

pytest will collect these as they are; this is only so they can be run today.
``DA6IDB_RECORD=1 python tests/run.py`` records the golden values instead of
checking them.
"""

import importlib
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "agg")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def main(argv):
    wanted = argv[1:] if len(argv) > 1 else None
    modules = sorted(p.stem for p in HERE.glob("test_*.py"))
    if wanted:
        modules = [m for m in modules if any(w in m for w in wanted)]
    passed = failed = 0
    failures = []
    started = time.time()
    for name in modules:
        module = importlib.import_module(name)
        tests = [n for n in dir(module) if n.startswith("test_")]
        for test in sorted(tests):
            try:
                getattr(module, test)()
                passed += 1
                print(f"  .  {name}.{test}")
            except Exception as exc:
                failed += 1
                failures.append((f"{name}.{test}", exc, traceback.format_exc()))
                print(f"  F  {name}.{test}  -- {type(exc).__name__}: {exc}")
    print(f"\n{passed} passed, {failed} failed in {time.time() - started:.1f}s")
    for name, _exc, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
