"""Pinning what counts as a repeat, and how repeats are put together."""

import numpy as np
from helpers import Sandbox, check, have_example, write_master

from da6idb import combine


def test_curve_step_is_the_typical_gap_not_the_span():
    even = np.linspace(0.0, 1.0, 11)
    assert np.isclose(combine.curve_step(even), 0.1)
    # a list scan need not step evenly, and one long jump back to the start
    # would make every real step look small beside it
    ragged = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    assert np.isclose(combine.curve_step(ragged), 0.1)
    assert combine.curve_step(np.full(5, 2.0)) == 0.0     # nothing moved
    assert combine.curve_step(np.array([1.0])) == 0.0


def test_same_grid_allows_a_readback_wobble_but_not_a_shift():
    base = np.linspace(0.0, 1.0, 11)
    rng = np.random.default_rng(0)
    assert combine.same_grid(base, base + rng.normal(0, 1e-4, base.size))
    # half a step is the limit: past it, point i is nearer the other's i+1
    assert combine.same_grid(base, base + 0.049)
    assert not combine.same_grid(base, base + 0.051)
    # a scan stopped part way is the same grid as far as it got
    assert combine.same_grid(base, base[:6])
    assert not combine.same_grid(base, base[:1])
    assert not combine.same_grid(np.full(5, 1.0), np.full(5, 1.0))


def test_combine_reads_b_onto_a_and_stops_at_the_overlap():
    x_a = np.linspace(0.0, 1.0, 11)
    x_b = x_a[2:] + 0.013          # offset, and starting later
    y_a, y_b = np.ones(x_a.size) * 10.0, np.ones(x_b.size) * 3.0
    x, total = combine.combine_curves(x_a, y_a, x_b, y_b, subtract=False)
    assert np.allclose(x, x_a)
    # outside B there is no second scan, so the sum stops rather than
    # holding B's first reading flat across it
    assert np.isnan(total[:3]).all()
    assert np.allclose(total[3:], 13.0)
    _x, less = combine.combine_curves(x_a, y_a, x_b, y_b, subtract=True)
    assert np.allclose(less[3:], 7.0)

    # a scan running downwards still works: np.interp needs increasing xp
    # and says nothing when it is not
    down = x_a[::-1]
    _x, back = combine.combine_curves(x_a, y_a, down, np.ones(down.size) * 3.0,
                                      subtract=False)
    assert np.isfinite(back).sum() == x_a.size
    assert np.allclose(back, 13.0)
    assert combine.combine_name("A", "B", True) == "A - B"


def test_stack_mean_averages_what_each_row_actually_has():
    rows = combine.stack_mean([np.array([1.0, 2.0, 3.0]),
                               np.array([3.0, 4.0])], 3)
    # the short piece is padded, not recycled, so row 2 is the first alone
    assert np.allclose(rows[:2], [2.0, 3.0])
    assert np.isclose(rows[2], 3.0)


def test_a_count_plan_is_nobodys_repeat():
    with Sandbox() as d:
        stepped = write_master(d / "s_00001_master.hdf", points=5)
        counted = write_master(d / "s_00002_master.hdf", points=5,
                               plan="count", motors=())
        assert combine.scan_signature(stepped) == ("rel_scan", ("psic_eta",))
        # nothing stepped, so there is no shape to repeat
        assert combine.scan_signature(counted) is None
        # and it is left on its own rather than pooled with the others
        groups = combine.scan_groups([stepped, counted])
        assert [len(g) for g in groups] == [1, 1]


def test_the_example_folder_splits_repeats_from_a_series():
    """152, 155 and 158 look alike and are not repeats: the sample warmed."""
    if not have_example():
        return
    from pathlib import Path
    folder = Path("tise2_2_00152_master.hdf").parent
    names = ("tise2_2_00152", "tise2_2_00155", "tise2_2_00158",
             "tise2_2_00353", "tise2_2_00354", "tise2_2_00355")
    paths = [folder / f"{n}_master.hdf" for n in names]
    if not all(p.is_file() for p in paths):
        return
    groups = [sorted(p.stem.replace("_master", "") for p in group)
              for group in combine.scan_groups(paths)]
    check("combine/example_groups", sorted(groups))
    # the three 353-355 sit at one temperature and one position: repeats
    assert ["tise2_2_00353", "tise2_2_00354", "tise2_2_00355"] in groups
    # 152/155/158 stepped about a kelvin apart, so each stands alone
    for name in ("tise2_2_00152", "tise2_2_00155", "tise2_2_00158"):
        assert [name] in groups, name
