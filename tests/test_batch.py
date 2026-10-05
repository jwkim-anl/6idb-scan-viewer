"""Pinning the scan-number arithmetic and the batch gridder."""

import numpy as np
from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example

import batch_grid as bg


def test_expand_scans():
    check("batch/expand", {
        "152,160,2": bg.expand_scans(152, 160, 2),
        "160,152,3": bg.expand_scans(160, 152, 3),
        "5,5,0": bg.expand_scans(5, 5, 0),
        "1,4,1": bg.expand_scans(1, 4, 1),
    })


def test_parse_scans():
    check("batch/parse", {
        "a": bg.parse_scans("152, 154 157-160"),
        "b": bg.parse_scans("7-9, 8"),
        "c": bg.parse_scans("  3  "),
    })
    for bad in ("banana", "1, two"):
        try:
            bg.parse_scans(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be refused")


def test_find_scans():
    import shutil
    if not have_example():
        return
    with Sandbox() as d:
        shutil.copy(EXAMPLE_IMAGE, d)
        shutil.copy(EXAMPLE_MASTER, d)
        found = bg.find_scans(d)
        assert set(found) == {152}
        assert found[152].name == EXAMPLE_IMAGE.name


def test_batch_shares_one_lattice():
    """The point of the batch: every grid on the same cells."""
    import shutil
    if not have_example():
        return
    import hkl_grid as hg
    with Sandbox() as d:
        shutil.copy(EXAMPLE_IMAGE, d)
        shutil.copy(EXAMPLE_MASTER, d)
        results = bg.grid_scans(d, [152, 999], oversample=6.0,
                                hkl_range=(0.95, 1.10, -0.60, -0.40, 7.40, 7.65))
        notes = {number: note for number, _path, note in results}
        check("batch/notes", notes)
        written = [p for _n, p, _note in results if p is not None]
        assert len(written) == 1
        assert written[0].parent.name == "output"
        grid = hg.load_vti(written[0])
        check("batch/dimensions", list(grid.dimensions))
