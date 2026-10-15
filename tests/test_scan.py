"""Pinning what is read out of a master file, and the numbers derived from it."""

import numpy as np
from helpers import EXAMPLE_MASTER, ROOT, Sandbox, check, have_example, write_master

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


def test_resolve_hint_field_finds_what_was_actually_recorded():
    fields = ["psic_eta", "psic_chi_user_setpoint", "det_total", "EPOCH", "INDEX"]
    assert sv.resolve_hint_field("psic_eta", fields) == "psic_eta"
    # hinted under its own name, recorded only as a setpoint
    assert sv.resolve_hint_field("psic_chi", fields) == "psic_chi_user_setpoint"
    assert sv.resolve_hint_field("nobody", fields) is None
    assert sv.resolve_hint_field("", fields) is None
    # the pseudo-axis of a plan that moved nothing: the index, not the clock
    assert sv.resolve_hint_field("time", fields) == "INDEX"
    assert sv.resolve_hint_field("time", ["EPOCH"]) == "EPOCH"
    assert sv.resolve_hint_field("time", ["det_total"]) is None


def test_hint_dimensions_reads_the_nested_tuples():
    text = ("detectors:\n- a_total\n"
            "dimensions:\n"
            "- !!python/tuple\n  - - bmag_sx\n    - bmag_sy\n  - primary\n"
            "- !!python/tuple\n  - - psic_chi\n  - primary\n"
            "monitor: Time\n")
    assert sv._hint_dimensions(text) == ["bmag_sx", "bmag_sy", "psic_chi"]
    # a plan that moved nothing writes an empty list, which is not an axis
    nothing = "dimensions:\n- !!python/tuple\n  - []\n  - primary\n"
    assert sv._hint_dimensions(nothing) == []
    assert sv._hint_dimensions("detectors:\n- a_total\n") == []


def test_scan_y_fields_come_from_the_plans_own_hints():
    hints = ("detectors:\n- lambda250k_stats1_total\n- Ion_Chamber_1\n"
             "- never_recorded\ndimensions:\n- !!python/tuple\n"
             "  - - psic_eta\n  - primary\n")
    with Sandbox() as d:
        path = write_master(d / "s_00007_master.hdf", points=5,
                            metadata={"hints": hints})
        fields = sv.stream_fields(path, "primary")
        info = sv.read_scan_info(path)
        assert info["hints"]["dimensions"] == ["psic_eta"]
        assert "never_recorded" in info["hints"]["detectors"]
        # only the ones the file actually holds come back, in hint order
        assert sv.scan_y_fields(path, fields) == [
            "lambda250k_stats1_total", "Ion_Chamber_1"]
        # and the axis comes from the dimensions hint
        assert sv.scan_x_field(path, fields) == "psic_eta"


def test_a_scan_that_hints_at_nothing_nominates_nothing():
    with Sandbox() as d:
        path = write_master(d / "s_00007_master.hdf", points=5,
                            metadata={"hints": "monitor: Time\n"})
        fields = sv.stream_fields(path, "primary")
        assert sv.scan_y_fields(path, fields) == []
        assert sv.read_scan_info(path)["hints"] == {"dimensions": [],
                                                    "detectors": []}
        # with no hints the motors metadata still names the axis
        assert sv.scan_x_field(path, fields) == "psic_eta"


def test_the_example_scans_nominate_their_own_signals():
    if not have_example():
        return
    fields = sv.stream_fields(EXAMPLE_MASTER, "primary")
    check("scan/example_hints", {
        "x": sv.scan_x_field(EXAMPLE_MASTER, fields),
        "y": sv.scan_y_fields(EXAMPLE_MASTER, fields),
    })


def test_dwell_is_the_commonest_reading_not_the_first():
    # the first point is the one the scaler was still settling into
    assert sv._dwell(np.array([0.0] + [0.2] * 44)) == 0.2
    assert sv._dwell(np.array([5.0] * 108 + [1.0])) == 5.0
    assert sv._dwell(np.array([0.2, 0.2, 0.2])) == 0.2
    assert sv._dwell(np.array([])) is None
    assert sv._dwell(np.array([np.nan, np.nan])) is None


def test_clipping_brings_a_map_back_from_two_wild_cells():
    rng = np.random.default_rng(0)
    z = rng.normal(100.0, 5.0, (30, 50))
    z[7, 12] = 50_000.0          # a detector glitch
    z[20, 3] = -9_000.0
    low, high, extend = sv.clipped_limits(z)
    assert 80.0 < low < 95.0 and 105.0 < high < 125.0
    assert extend == "both"      # the bar says it is cutting both ends
    # a map with nothing wrong with it is barely touched
    clean = rng.normal(100.0, 5.0, (30, 50))
    low, high, _e = sv.clipped_limits(clean)
    assert abs(low - clean.min()) < 6.0 and abs(high - clean.max()) < 6.0
    assert sv.scale_extend(z, None, None) == "neither"
    assert sv.scale_extend(z, float(z.min()), 1000.0) == "max"


def test_centered_limits_put_zero_in_the_middle():
    z = np.linspace(-0.0009, 0.0036, 50)
    low, high = sv.centered_limits(z, -0.0009, 0.0036)
    assert np.isclose(low, -high)                  # white lands on zero
    assert high >= 0.0036 and low <= -0.0009       # and nothing is cut
    # it only ever widens, so data already centred is left alone
    assert sv.centered_limits(z, -1.0, 1.0) == (-1.0, 1.0)


def test_lone_dropouts_catch_the_point_the_beam_missed():
    monitor = np.full(40, 1.85e6)
    monitor[17] = 5372.0                            # the beam was gone
    flagged = sv.lone_dropouts(monitor)
    assert list(np.flatnonzero(flagged)) == [17]
    # a monitor that simply drifts is not a dropout
    assert not sv.lone_dropouts(np.linspace(1.0e6, 0.6e6, 40)).any()
    assert not sv.lone_dropouts(np.full(40, 1.85e6)).any()


def test_read_scatter_keeps_the_positions_a_fly_grid_recorded():
    """A flown axis lands where it lands; nothing here snaps it."""
    if not (ROOT / "data" / "test_26_3").is_dir():
        return
    folder = ROOT / "data" / "test_26_3"
    path = folder / "test_00014_master.hdf"
    if not path.is_file():
        return
    field = "lambda250k_stats5_total"
    # a fly grid is not a lattice, and read_grid is right to refuse it
    assert sv.read_grid(path, field) is None
    got = sv.read_scatter(path, field)
    assert got["scattered"] and got["x_name"] == "cryo_x"
    assert got["y_name"] == "cryo_y"
    # x, y and z run alongside each other, one entry per measurement
    assert got["x"].shape == got["y"].shape == got["z"].shape
    assert got["points"] == got["x"].size

    # the plan declared 11 x 11 = 121 and the scan recorded 125: there is
    # no rectangle here, which is the whole reason this function exists
    assert sv.read_scan_info(path)["shape"] == [11, 11]
    assert got["points"] != 11 * 11

    # the stepped axis holds exact levels, the flown one does not
    assert np.unique(np.round(got["y"], 6)).size == 11
    assert np.unique(np.round(got["x"], 6)).size > 11
    check("scan/fly_grid", {
        "points": got["points"],
        "rows": int(np.unique(np.round(got["y"], 6)).size),
        "x_span": [round(float(got["x"].min()), 4), round(float(got["x"].max()), 4)],
    })


def test_grid_stats_reads_either_shape_of_map():
    lattice = {"x": np.array([0.0, 1.0]), "y": np.array([10.0, 20.0]),
               "z": np.array([[1.0, 5.0], [2.0, np.nan]])}
    stats = sv.grid_stats(lattice)
    assert stats["max_z"] == 5.0 and stats["max_at"] == (1.0, 10.0)
    assert stats["min_z"] == 1.0 and stats["min_at"] == (0.0, 10.0)

    # the same question of points as measured, where the position is read
    # off at the same index rather than unravelled
    points = {"x": np.array([0.3, 0.7, 0.5]), "y": np.array([9.0, 9.5, 9.2]),
              "z": np.array([1.0, 5.0, np.nan]), "scattered": True}
    stats = sv.grid_stats(points)
    assert stats["max_z"] == 5.0 and stats["max_at"] == (0.7, 9.5)
    assert stats["min_z"] == 1.0 and stats["min_at"] == (0.3, 9.0)
    assert sv.grid_stats({"x": np.array([0.0]), "y": np.array([0.0]),
                          "z": np.array([np.nan])})["max_z"] is None


def test_two_motors_swept_together_are_not_a_map():
    """A scan can move two motors along one line; that is a curve, not a map.

    Scans 6, 9, 11 and 83 sweep a pair of coupled voltages together, so every
    point sits on a single straight line. Asking matplotlib to colour the
    triangles between them reaches qhull, which fails with "singular input
    data" -- so the map has to be refused here, where the reason is known.
    """
    # the geometry, independent of any file
    straight = np.linspace(0.0, 1.0, 50)
    assert not sv.spans_an_area(straight, 2.0 * straight - 3.0)   # a diagonal
    assert not sv.spans_an_area(straight, np.full(50, 7.0))       # y never moved
    assert not sv.spans_an_area(np.zeros(50), straight)           # x never moved
    assert not sv.spans_an_area([0.0, 1.0], [0.0, 1.0])           # too few points

    # a fine axis against a coarse one is a map, not a line: each axis is
    # judged against its own span, so the units cannot decide this
    coarse, fine = np.meshgrid(np.linspace(0, 1000, 8), np.linspace(0, 1e-4, 8))
    assert sv.spans_an_area(coarse.ravel(), fine.ravel())

    if not have_example():
        return
    for number in ("00006", "00009", "00011", "00083"):
        path = ROOT / "data" / f"tise2_2_{number}_master.hdf"
        if not path.is_file():
            continue
        info = sv.read_scan_info(path)
        assert len(info["motors"]) == 2        # two motors named ...
        field = info["hints"]["detectors"][0]
        assert sv.read_grid(path, field) is None
        # ... and still no map to be drawn of them
        assert sv.read_scatter(path, field) is None

    # the fly grid in the same suite must not be caught by this
    fly = ROOT / "data" / "test_26_3" / "test_00014_master.hdf"
    if fly.is_file():
        assert sv.read_scatter(fly, "lambda250k_stats5_total") is not None
