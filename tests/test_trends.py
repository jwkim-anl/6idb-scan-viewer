"""Pinning which baseline fields count as 'what was stepped'."""

import numpy as np
from helpers import Sandbox, check, have_example, write_master

from da6idb import trends


def _scan(path, points=5, **baseline):
    return write_master(path, points=points, baseline=baseline)


def test_one_scan_has_nothing_to_be_plotted_against():
    with Sandbox() as d:
        assert trends.baseline_trend_fields([_scan(d / "s_00001_master.hdf")]) == []
        assert trends.baseline_trend_fields([]) == []


def test_the_field_that_was_stepped_ranks_above_the_one_that_drifted():
    with Sandbox() as d:
        paths = []
        for index, (stepped, drifting) in enumerate(
                [(10.0, 1.0000), (20.0, 1.0004), (30.0, 0.9997)], start=1):
            paths.append(_scan(d / f"s_{index:05d}_master.hdf",
                               stepped=stepped, drifting=drifting,
                               parked=7.0))
        ranked = trends.baseline_trend_fields(paths)
        assert ranked[0] == "stepped"
        # a field that never moved is not an axis
        assert "parked" not in ranked
        # and the drifting one is either below it or below the bar entirely
        if "drifting" in ranked:
            assert ranked.index("drifting") > ranked.index("stepped")


def test_trend_axis_reads_the_scan_number_or_a_baseline_field():
    with Sandbox() as d:
        a = _scan(d / "s_00007_master.hdf", stepped=10.0)
        b = _scan(d / "s_00009_master.hdf", stepped=20.0)
        numbers = trends.trend_axis([a, b], trends.TREND_X_SCAN)
        assert numbers == {str(a): 7.0, str(b): 9.0}
        field = trends.trend_axis([a, b], "stepped")
        assert field == {str(a): 10.0, str(b): 20.0}
        # a scan that cannot supply one is left out, not given a value
        assert trends.trend_axis([a, b], "never_recorded") == {}


def test_the_example_series_names_what_was_stepped():
    """152, 155 and 158 are a temperature series; the ranking should say so."""
    if not have_example():
        return
    from pathlib import Path
    paths = [Path(f"tise2_2_{n:05d}_master.hdf") for n in (152, 155, 158)]
    if not all(p.is_file() for p in paths):
        return
    ranked = trends.baseline_trend_fields(paths)
    assert "temp340_control_setpoint" in ranked
    axis = trends.trend_axis(paths, "temp340_control_setpoint")
    # the setpoint was stepped in whole kelvin, which is what deliberate
    # looks like beside a readback that drifts in the fourth decimal
    assert sorted(axis.values()) == [162.0, 164.0, 166.0]
    check("trends/example_ranked", ranked[:6])

    # three true repeats stepped nothing: only the ring current moves
    repeats = [Path(f"tise2_2_{n:05d}_master.hdf") for n in (353, 354, 355)]
    if all(p.is_file() for p in repeats):
        assert "temp340_control_setpoint" not in trends.baseline_trend_fields(repeats)
