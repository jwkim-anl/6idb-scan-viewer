"""Pinning what is read out of a master file, and the numbers derived from it."""

import numpy as np
from helpers import EXAMPLE_MASTER, Sandbox, check, have_example, write_master

from da6idb import scan as sv


def test_stream_fields_and_pseudo_fields():
    with Sandbox() as d:
        master = write_master(d / "s_00001_master.hdf", points=7)
        fields = sv.stream_fields(master, "primary")
        check("scan/stream_fields", fields)
        assert sv.EPOCH_FIELD in fields and sv.INDEX_FIELD in fields
        # INDEX last, so the fallback reaches it before any parked motor
        assert fields[-1] == sv.INDEX_FIELD
        assert sv.stream_fields(master, "nosuchstream") == []


def test_index_matches_the_scan_length():
    with Sandbox() as d:
        master = write_master(d / "s_00002_master.hdf", points=9)
        data = sv.read_primary(master, [sv.INDEX_FIELD, "lambda250k_stats1_total"])
        assert np.array_equal(data[sv.INDEX_FIELD], np.arange(9.0))
        assert data["lambda250k_stats1_total"].size == 9


def test_x_field_choice():
    with Sandbox() as d:
        moved = write_master(d / "s_00003_master.hdf", motors=("psic_eta",))
        counted = write_master(d / "s_00004_master.hdf", plan="count", motors=())
        fields = sv.stream_fields(moved, "primary")
        assert sv.scan_x_field(moved, fields) == "psic_eta"
        counted_fields = sv.stream_fields(counted, "primary")
        assert sv.scan_x_field(counted, counted_fields) is None
        # with no motor named, the fallback takes the index rather than a
        # motor that never moved
        assert sv.default_x_field(counted_fields) == sv.INDEX_FIELD


def test_scan_info():
    with Sandbox() as d:
        master = write_master(d / "s_00005_master.hdf", points=11)
        info = sv.read_scan_info(master)
        check("scan/info", {k: info.get(k) for k in
                            ("is_scan", "plan", "points", "motors", "shape")})


def test_curve_statistics():
    x = np.linspace(-5, 5, 101)
    y = np.exp(-0.5 * (x / 1.2) ** 2) * 500 + 3
    check("scan/curve_stats", sv.curve_stats(x, y))


def test_curve_statistics_on_degenerate_input():
    flat = sv.curve_stats(np.array([0.0, 1.0]), np.array([2.0, 2.0]))
    check("scan/curve_stats_flat", flat)
    empty = sv.curve_stats(np.array([]), np.array([]))
    check("scan/curve_stats_empty", empty)


def test_axis_edges():
    check("scan/axis_edges_even", sv.axis_edges(np.array([0.0, 1.0, 2.0, 3.0])))
    check("scan/axis_edges_uneven", sv.axis_edges(np.array([0.0, 1.0, 4.0])))
    check("scan/axis_edges_single", sv.axis_edges(np.array([2.0])))


def test_two_motor_raster_is_recognised():
    with Sandbox() as d:
        slow = np.repeat(np.arange(3.0), 4)
        fast = np.tile(np.arange(4.0), 3)
        value = slow * 10 + fast
        master = write_master(
            d / "s_00006_master.hdf", points=12, plan="grid_scan",
            motors=("m_slow", "m_fast"),
            extra_primary={"m_slow": slow, "m_fast": fast, "det": value},
        )
        grid = sv.read_grid(master, "det", None)
        assert grid is not None
        check("scan/grid", {k: grid[k] for k in ("x", "y", "z", "x_name", "y_name")})
        check("scan/grid_stats", sv.grid_stats(grid))


def test_a_line_scan_is_not_a_raster():
    with Sandbox() as d:
        master = write_master(d / "s_00007_master.hdf", points=11)
        assert sv.read_grid(master, "lambda250k_stats1_total", None) is None


def test_example_master_values():
    if not have_example():
        return
    fields = sv.stream_fields(EXAMPLE_MASTER, "primary")
    check("scan/example_fields", sorted(fields))
    info = sv.read_scan_info(EXAMPLE_MASTER)
    check("scan/example_info", {k: info.get(k) for k in
                                ("is_scan", "plan", "points", "motors", "count_time")})
    check("scan/example_x_field", sv.scan_x_field(EXAMPLE_MASTER, fields))
    check("scan/example_hkl", sv.scan_hkl(EXAMPLE_MASTER))
    data = sv.read_primary(EXAMPLE_MASTER, ["psic_eta", "lambda250k_stats1_total"])
    check("scan/example_curve_stats",
          sv.curve_stats(data["psic_eta"], data["lambda250k_stats1_total"]))
