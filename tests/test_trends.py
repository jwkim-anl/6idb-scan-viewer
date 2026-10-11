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


def test_trend_choices_split_by_component_once_there_are_several():
    one = trends.trend_choices(1)
    assert list(one) == ["Fit center", "Fit FWHM", "Fit height", "Fit area"]
    assert one["Fit center"] == (None, "fit_center")
    # the mixing only for a shape that fits one
    assert "Fit eta" in trends.trend_choices(1, mixed=True)

    two = trends.trend_choices(2)
    assert list(two)[:2] == ["Peak 1 center", "Peak 1 FWHM"]
    assert two["Peak 2 area"] == (1, "fit_area")
    # the one quantity of a several-peak fit that is the fit's, not a peak's
    assert trends.TREND_TOTAL_AREA in two
    assert len(trends.trend_choices(3)) == 3 * 4 + 1


def test_trend_value_reads_a_component_or_the_whole_fit():
    from da6idb import fit
    x = np.linspace(-5, 5, 201)
    gauss = fit.PEAK_SHAPES["Gauss"][0]
    stats = fit.peak_fit_multi(
        x, 900.0 * gauss(x, -1.5, 1.0) + 600.0 * gauss(x, 1.8, 1.4) + 2.0 * x,
        "Gauss", 2)
    choices = trends.trend_choices(2)
    assert np.isclose(trends.trend_value(stats, choices["Peak 1 center"]), -1.5,
                      atol=1e-5)
    assert np.isclose(trends.trend_value(stats, choices["Peak 2 center"]), 1.8,
                      atol=1e-5)
    # the total is the components added, not the fit's own blank cell
    total = trends.trend_value(stats, choices[trends.TREND_TOTAL_AREA])
    parts = sum(p["fit_area"] for p in stats["fit_peaks"])
    assert np.isclose(total, parts)
    assert stats["fit_area"] is None       # which is why it cannot come from there

    # a refused fit, and a component the fit does not have, are gaps not errors
    assert trends.trend_value({}, choices["Peak 1 center"]) is None
    assert trends.trend_value(stats, trends.trend_choices(3)["Peak 3 center"]) is None
    assert trends.trend_value(stats, (0, "fit_eta")) is None   # Gauss has none


def test_a_single_peak_fit_still_reads_through_the_same_door():
    from da6idb import fit
    x = np.linspace(-5, 5, 201)
    stats = fit.peak_fit(
        x, 1000.0 * fit.PEAK_SHAPES["Gauss"][0](x, 0.7, 1.4) + 3.0 * x, "Gauss")
    choices = trends.trend_choices(1)
    assert np.isclose(trends.trend_value(stats, choices["Fit center"]), 0.7,
                      atol=1e-6)
    assert np.isclose(trends.trend_value(stats, choices["Fit area"]),
                      stats["fit_area"])
