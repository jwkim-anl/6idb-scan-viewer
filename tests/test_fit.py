"""Pinning the peak fitting: known peaks in, the same numbers out."""

import numpy as np
from helpers import check

from da6idb import fit


def _line(x, slope=3.0, offset=50.0):
    return slope * x + offset


def test_every_shape_recovers_a_peak_it_made_itself():
    """The sharpest check a fit can be given: no noise, so it must be exact."""
    x = np.linspace(-5, 5, 201)
    for shape, (peak, _area) in fit.PEAK_SHAPES.items():
        y = 1000.0 * peak(x, 0.7, 1.4) + _line(x)
        got = fit.peak_fit(x, y, shape)
        assert np.isclose(got["fit_center"], 0.7, atol=1e-6), shape
        assert np.isclose(got["fit_fwhm"], 1.4, atol=1e-6), shape
        assert np.isclose(got["fit_height"], 1000.0, rtol=1e-6), shape
        assert np.isclose(got["fit_slope"], 3.0, atol=1e-6), shape
        assert np.isclose(got["fit_offset"], 50.0, atol=1e-6), shape
        assert got["fit_r2"] > 0.999999, shape


def test_the_area_is_the_integral_of_the_shape():
    """Each shape carries a closed form; check it against a quadrature.

    Over the whole line, not a window: the Lorentzian's tails fall only as
    1/x squared, so integrating it between -40 and 40 is already 1.1% short
    of its area and would fail a check the closed form passes exactly.
    """
    from scipy.integrate import quad
    x = np.linspace(-5, 5, 401)
    for shape, (peak, _area) in fit.PEAK_SHAPES.items():
        got = fit.peak_fit(x, 1000.0 * peak(x, 0.7, 1.4) + _line(x), shape)
        integral, _error = quad(lambda v: 1000.0 * peak(v, 0.7, 1.4),
                                -np.inf, np.inf, limit=400)
        assert np.isclose(got["fit_area"], integral, rtol=1e-6), (
            shape, got["fit_area"], integral)


def test_a_fitted_mixing_is_reported_only_where_there_is_one():
    x = np.linspace(-5, 5, 201)
    mixed = fit.PEAK_MIX["Pseudo-Voigt"][0]
    for eta in (0.2, 0.8):
        y = 1000.0 * mixed(x, 0.7, 1.4, eta) + _line(x)
        got = fit.peak_fit(x, y, "Pseudo-Voigt")
        assert np.isclose(got["fit_eta"], eta, atol=5e-3), (eta, got["fit_eta"])
    # a shape without a mixing reports none, and loses the column
    assert fit.peak_fit(x, 1000.0 * fit.PEAK_SHAPES["Gauss"][0](x, 0.7, 1.4)
                        + _line(x), "Gauss")["fit_eta"] is None
    assert "Fit eta" not in fit.fit_columns(False)
    assert fit.fit_columns(True).index("Fit eta") == 2


def test_two_peaks_come_back_in_order_of_centre():
    x = np.linspace(-5, 5, 201)
    gauss = fit.PEAK_SHAPES["Gauss"][0]
    y = (900.0 * gauss(x, -1.5, 1.0) + 600.0 * gauss(x, 1.8, 1.4)
         + _line(x, 2.0, 40.0))
    got = fit.peak_fit_multi(x, y, "Gauss", 2)
    peaks = got["fit_peaks"]
    assert len(peaks) == 2
    # leftmost first, whichever is the taller, so a trend follows one peak
    assert peaks[0]["fit_center"] < peaks[1]["fit_center"]
    assert np.isclose(peaks[0]["fit_center"], -1.5, atol=1e-5)
    assert np.isclose(peaks[1]["fit_center"], 1.8, atol=1e-5)
    assert np.isclose(peaks[0]["fit_height"], 900.0, rtol=1e-5)
    assert np.isclose(peaks[1]["fit_height"], 600.0, rtol=1e-5)
    # the line is shared, and the curve's own row stays blank
    assert np.isclose(got["fit_slope"], 2.0, atol=1e-6)
    assert got["fit_center"] is None and got["fit_fwhm"] is None


def test_the_drawn_model_is_the_fit():
    x = np.linspace(-5, 5, 201)
    gauss = fit.PEAK_SHAPES["Gauss"][0]
    y = 1000.0 * gauss(x, 0.7, 1.4) + _line(x)
    one = fit.peak_fit(x, y, "Gauss")
    assert np.allclose(fit.peak_model(x, "Gauss", one), y, atol=1e-9)

    two = fit.peak_fit_multi(
        x, 900.0 * gauss(x, -1.5, 1.0) + 600.0 * gauss(x, 1.8, 1.4) + _line(x),
        "Gauss", 2)
    drawn = fit.peak_model(x, "Gauss", two)
    assert np.allclose(
        drawn, 900.0 * gauss(x, -1.5, 1.0) + 600.0 * gauss(x, 1.8, 1.4) + _line(x),
        atol=1e-9)
    # the pieces: a background and one line per component, each drawn on it
    assert len(fit.peak_parts(x, "Gauss", two)) == 3
    # nothing to draw is None rather than a raise, so a caller need not check
    assert fit.peak_model(x, "Gauss", {}) is None
    assert fit.peak_model(x, "Nonsense", one) is None


def test_a_fit_is_refused_rather_than_invented():
    x = np.linspace(-5, 5, 201)
    rng = np.random.default_rng(1)
    # a sloping line with noise on it is not a peak
    noise = fit.peak_fit(x, _line(x) + rng.normal(0, 5.0, x.size), "Gauss")
    assert all(noise[k] is None for k in noise), noise
    gauss = fit.PEAK_SHAPES["Gauss"][0]
    y = 1000.0 * gauss(x, 0.7, 1.4) + _line(x)
    assert fit.peak_fit(x[:4], y[:4], "Gauss")["fit_center"] is None
    assert fit.peak_fit(x, y, "Nonsense")["fit_center"] is None
    assert fit.peak_fit(np.ones(20), y[:20], "Gauss")["fit_center"] is None
    # NaN padding, which an averaged curve carries, is dropped not fatal
    padded_x, padded_y = np.r_[x, np.nan], np.r_[y, np.nan]
    assert np.isclose(fit.peak_fit(padded_x, padded_y, "Gauss")["fit_center"],
                      0.7, atol=1e-6)


def test_r2_is_taken_over_the_peak_not_the_slope():
    """A sloping line would otherwise explain the variance by itself."""
    x = np.linspace(-5, 5, 201)
    rng = np.random.default_rng(2)
    gauss = fit.PEAK_SHAPES["Gauss"][0]
    # a modest peak on a steep background: against the raw curve the slope
    # alone would score well, so the number has to ignore it
    y = 60.0 * gauss(x, 0.7, 1.4) + _line(x, 300.0, 50.0) + rng.normal(0, 3, x.size)
    got = fit.peak_fit(x, y, "Gauss")
    raw = 1.0 - (np.sum((y - fit.peak_model(x, "Gauss", got)) ** 2)
                 / np.sum((y - y.mean()) ** 2))
    assert raw > 0.999            # the slope makes the raw number meaningless
    assert got["fit_r2"] < 0.99   # over the peak alone it is honest
    check("fit/r2_over_the_peak", round(float(got["fit_r2"]), 4))
