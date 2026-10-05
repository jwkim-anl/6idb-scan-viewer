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


def test_describe_fields_says_what_each_field_is():
    """The table you read before choosing x, y and the monitor."""
    with Sandbox() as d:
        path = write_master(d / "s_00007_master.hdf", points=9,
                            extra_primary={"psic_h": np.full(9, 1.5)})
        rows = sv.describe_fields(path)
        check("scan/describe_rows", rows)
        # the motor first, the detector the plan nominated next
        assert [r["field"] for r in rows][:2] == ["psic_eta",
                                                  "lambda250k_stats1_total"]
        roles = {r["field"]: r["role"] for r in rows}
        assert roles["psic_eta"] == "scanned"        # from the motors metadata
        assert roles["Ion_Chamber_1"] == "detector"  # from its signal_type
        assert roles["psic_h"] == "positioner"       # from its PV suffix
        assert roles["scaler_time"] == "time"
        assert roles["INDEX"] == "index"
        hinted = [r["field"] for r in rows if r["hinted"]]
        assert hinted == ["lambda250k_stats1_total"]
        still = [r["field"] for r in rows if r["varies"] is False]
        assert "Ion_Chamber_1" in still and "psic_h" in still


def test_describe_fields_survives_a_file_with_no_stream():
    with Sandbox() as d:
        path = write_master(d / "s_00007_master.hdf", points=5)
        assert sv.describe_fields(path, "baseline_does_not_exist") == []


def test_yaml_block_takes_one_section_of_the_hints():
    text = ("detectors:\n- a_total\n- Ion Chamber 4\n"
            "dimensions:\n- !!python/tuple\n  - - psic_eta\n  - primary\n"
            "monitor: Time\n")
    assert sv._yaml_items(sv._yaml_block(text, "detectors")) == [
        "a_total", "Ion Chamber 4"]
    # without the split, every list item in the blob would come back at once
    assert len(sv._yaml_items(text)) > 2
