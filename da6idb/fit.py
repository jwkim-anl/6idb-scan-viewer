"""Fitting a peak, or several, to one curve.

Ported from the 6-ID-C fork of ``scan_viewer``, where it grew against real
rocking curves; the numerics are unchanged, only the imports and the two
column constants moved here with them.

The model is always a peak shape plus a sloping line. The background is
never optional: on those rocking curves it ran 1754 counts down to 476
across a single scan, and fitting the slope rather than a flat pedestal is
what buys the accuracy. Every value comes back ``None`` when no fit worth
trusting was found, which is far more often than it looks like it should
be, and deliberately.

numpy and scipy only -- no Qt, no pandas. A notebook can fit a curve with
this as readily as the window can.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from da6idb.scan import curve_stats

# The headings the fitted columns carry, and the mixing's own, which is
# shown only for a shape that has one to report.
STATS_FIT_COLUMNS = [
    "Fit center", "Fit FWHM", "Fit height", "Fit area", "R2",
]
STATS_MIX_COLUMN = "Fit eta"


def _peak_gauss(x: np.ndarray, centre: float, fwhm: float) -> np.ndarray:
    return np.exp(-4.0 * np.log(2.0) * ((x - centre) / fwhm) ** 2)


def _peak_lorentzian(x: np.ndarray, centre: float, fwhm: float) -> np.ndarray:
    return 1.0 / (1.0 + 4.0 * ((x - centre) / fwhm) ** 2)


def _peak_lorentzian2(x: np.ndarray, centre: float, fwhm: float) -> np.ndarray:
    root = 4.0 * (np.sqrt(2.0) - 1.0) * ((x - centre) / fwhm) ** 2
    return 1.0 / (1.0 + root) ** 2


# Lorentzian and Gaussian at one shared FWHM, which is the whole trick of
# the pseudo-Voigt: both are already unit height at the centre and already
# half height at centre +/- fwhm/2, so any mix of them is too, and the
# blend keeps that FWHM without a correction term. *eta* is the Lorentzian
# fraction -- 1 is the pure Lorentzian, 0 the pure Gauss -- and it is
# fitted rather than assumed, which is what PEAK_MIX below arranges.
def _peak_pseudo_voigt_mixed(
    x: np.ndarray, centre: float, fwhm: float, eta: float
) -> np.ndarray:
    return (
        eta * _peak_lorentzian(x, centre, fwhm)
        + (1.0 - eta) * _peak_gauss(x, centre, fwhm)
    )


# The same shape with the mixing held at a half: where the fit starts, and
# what it falls back to when a free eta will not stand. Kept as a function
# of three parameters of its own because that is the signature PEAK_SHAPES
# promises and that every shape without a mixing has.
def _peak_pseudo_voigt(
    x: np.ndarray, centre: float, fwhm: float
) -> np.ndarray:
    return _peak_pseudo_voigt_mixed(x, centre, fwhm, 0.5)


def _pseudo_voigt_area(eta: float) -> float:
    """The mix's own integral, at unit height and unit FWHM.

    Linear in *eta*, because the integral of a sum is the sum of the
    integrals: no third closed form is needed beyond the two above, and a
    fitted mixing reaches the "Fit area" column without a quadrature
    anywhere. At eta a half this is the constant the fixed shape carried
    before the mixing was fitted, to the last bit -- halving is exact.
    """
    return float(
        eta * np.pi / 2.0
        + (1.0 - eta) * np.sqrt(np.pi / (4.0 * np.log(2.0)))
    )


# Each shape is paired with the integral of its own function over the whole
# line, at unit height and unit FWHM, so that a new shape cannot be added
# with its area quietly left out. These are closed forms, checked against a
# trapezoid of the function itself: the Gaussian and the square agree to
# nine figures, the Lorentzian only to four, the rest being tail that a
# numeric check truncated at 4000 half-widths has still not collected. That
# same heavy tail is what the "Fit area" column carries -- see peak_fit.
PEAK_SHAPES = {
    "Gauss": (_peak_gauss, float(np.sqrt(np.pi / (4.0 * np.log(2.0))))),
    "Lorentzian": (_peak_lorentzian, float(np.pi / 2.0)),
    "Lorentzian squared": (
        _peak_lorentzian2,
        float(np.pi / (4.0 * np.sqrt(np.sqrt(2.0) - 1.0))),
    ),
    # Half of each of the two above it. Its area comes from the same
    # function the fitted mixing uses, so the two cannot drift apart.
    "Pseudo-Voigt": (_peak_pseudo_voigt, _pseudo_voigt_area(0.5)),
}
# The shapes that carry a mixing parameter of their own, paired with the
# area factor as a function of it. Being listed here is what tells both
# fitters to spend a fourth parameter a component on that shape; a shape
# absent from it is solved on three, exactly as all four were before, and
# only a shape listed here is given the "Fit eta" column.
#
# Both fitters also solve the fixed shape and keep whichever scores
# better, so the extra parameter cannot cost anything, and a free eta is
# refused outright unless it lands clear of its two ends: a mixing resting
# on 0 or 1 is the solver saying "pure Gauss" or "pure Lorentzian" rather
# than measuring a blend, and both of those are in the combo already.
#
# Measured on the scans the tooltips name. It is kept where it buys
# something -- scan 31's derivative at two peaks goes 0.996363 to
# 0.997166 at eta 0.690 and 0.802, scan 34 goes 0.998357 to 0.999409 at
# 0.851 and 0.073 -- and refused where it does not: scan 32 at two peaks
# runs its first component to the 1.000 rail, and scan 109 drives one
# component under the step size. Those two fall back to the fixed half,
# report precisely the fit they reported before and leave the eta cell
# blank, which is the honest reading and is why the fallback exists at
# all rather than letting one of the three cases "Multi" is for decline.
PEAK_MIX = {
    "Pseudo-Voigt": (_peak_pseudo_voigt_mixed, _pseudo_voigt_area),
}
# How far clear of either of its two ends a fitted mixing has to land
# before it is read as a measurement rather than as the solver resting
# against a wall. Small on purpose, because the wall is the only thing
# being detected: a bounded least squares that wants to be outside its
# box stops hard against it, and scan 32 at two peaks stops at one ulp
# below 1 -- a plain "strictly less than 1" lets that straight through
# and prints 1.000 as though it had been measured. A mixing genuinely
# near a pure shape is a different statement and is nowhere near this
# close; it is reported as whatever number it is.
PEAK_MIX_RAIL = 1e-6
# Below this many finite points a five-parameter fit is not measuring
# anything: over 200 six-point curves of pure noise, five parameters scored
# a median R2 of 0.71, one in twenty above 0.99 and the best a flat 1.0000.
# Seven leaves two degrees of freedom, which is thin but honest, and the
# sanity gate below is what actually does the rejecting.
PEAK_FIT_MIN_POINTS = 7
# How far above the data's own range a fitted peak may stand before it is
# read as a solver artefact rather than a signal. Generous on purpose --
# a real peak sits inside the data range, so ten times it is already far
# past anything physical -- but finite, which is the point: see peak_fit.
PEAK_FIT_MAX_HEIGHT = 10.0
# How prominent a bump must be, as a fraction of the data's own Y range,
# before find_peaks offers it as a starting point for one of "Multi"'s
# components. A seed and nothing more: a bump it passes over can still be
# fitted -- scan 34's second peak is not a local maximum at all, and is
# found by the padding in _peak_seeds -- and one it offers can still be
# turned down by the gate afterwards.
PEAK_SEED_PROMINENCE = 0.05
# Bounded least squares walks to its answer in much smaller steps than the
# unbounded fit peak_fit does, and with three parameters a component there
# are a good many more of them to take. Generous rather than measured: the
# failure mode is a RuntimeError and a blank row, and even eight parameters
# over fifty points is a few milliseconds.
PEAK_FIT_MULTI_MAXFEV = 60000
# What the count beside "Multi" offers. It starts at two because one peak
# is what the button being off already means, so a "1" in the list would
# be a second way of saying the same thing. Two and three are measured on
# the scans the tooltips name; four is allowed by the same arithmetic and
# held in check by the minimum-points rule, which asks three more readings
# for every component.
PEAK_COUNTS = ["2", "3", "4"]


def _peak_mix_refine(
    xs: np.ndarray,
    ys: np.ndarray,
    mixed,
    start: tuple[float, float, float, float, float],
    span: float,
    spread: float,
    step: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Re-solve one fitted peak with the shape's mixing set free.

    *mixed* is the shape's four-parameter form and *start* is what
    :func:`peak_fit` already has -- height, centre, FWHM, slope, offset.
    The mixing is seeded at the half the fixed shape assumes, so the
    six-parameter solve starts from the five-parameter answer with only
    the mixing left to find.

    Returns the six refined parameters, or ``None`` for "keep the answer
    you had". That happens when the solver gives up, when the refined
    peak fails any of the four geometric tests ``peak_fit`` has just
    applied -- a freed mixing can trade width against shape and walk out
    of them -- or when the mixing does not land strictly between its own
    two ends, which PEAK_MIX explains.

    Unbounded, like the fit it refines. Bounding it would switch
    ``curve_fit`` from Levenberg-Marquardt to a trust region and move
    every other parameter along with the mixing; the ends are tested
    here afterwards instead, exactly as the geometry already is.
    """
    height, centre, fwhm, line_slope, line_offset = start

    def model(grid, height, centre, fwhm, line_slope, line_offset, eta):
        return (
            height * mixed(grid, centre, fwhm, eta)
            + line_slope * grid
            + line_offset
        )

    try:
        # As peak_fit: catch_warnings only saves and restores the filter
        # state, and the simplefilter is what silences the covariance and
        # overflow warnings scipy prints straight to stderr.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _covariance = curve_fit(
                model,
                xs,
                ys,
                p0=[height, centre, fwhm, line_slope, line_offset, 0.5],
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(popt)):
        return None
    height, centre, fwhm, line_slope, line_offset, eta = (
        float(value) for value in popt
    )
    if not xs[0] <= centre <= xs[-1]:
        return None
    if not 0.0 < fwhm <= span:
        return None
    if abs(height) > PEAK_FIT_MAX_HEIGHT * spread:
        return None
    if fwhm < step:
        return None
    # Nothing holds this solve inside the two ends, so a mixing that
    # wants to be outside them lands outside and is refused here. The
    # same margin as the bounded path all the same, so that the two
    # report the same shapes as measured and the same ones as pure.
    if not PEAK_MIX_RAIL <= eta <= 1.0 - PEAK_MIX_RAIL:
        return None
    return height, centre, fwhm, line_slope, line_offset, eta


def peak_fit(
    x: np.ndarray,
    y: np.ndarray,
    shape: str,
    fwhm_seed: float | None = None,
) -> dict[str, float | None]:
    """Fit *shape* plus a sloping line to one curve.

    The model is ``height * peak(x; center, fwhm) + slope * x + offset``,
    five parameters. The background is always sloping and never optional:
    on the rocking curves in TiSe2 (32 to 38) it runs 1754 counts down to
    476 across scan 32 alone, and fitting it lifts the mean R2 of the whole
    curve from 0.957 to 0.980 for a Gaussian -- 0.976 for a flat offset, so
    it is the slope that is bought and not merely a pedestal. (Those three
    are measured against the raw curve, which is not how ``fit_r2`` below
    is reckoned; they compare background models, not fits.) ``height`` is
    measured above that line, which is why it comes out below the curve's
    own maximum (46664 against 51863 on scan 32).

    Every value is ``None`` when no trustworthy fit came back, which the
    table shows as a blank row and the export writes as ``(none)``. That
    happens a great deal more often than it might seem it should, and
    deliberately: see the gate below.

    *fwhm_seed* is ``curve_stats``' crossing width when it found one. It is
    passed in rather than recomputed because the caller has just worked it
    out for the same curve.
    """
    stats: dict[str, float | None] = {
        "fit_center": None,
        "fit_fwhm": None,
        "fit_height": None,
        "fit_area": None,
        "fit_eta": None,
        "fit_r2": None,
        "fit_slope": None,
        "fit_offset": None,
    }
    entry = PEAK_SHAPES.get(shape)
    if entry is None:
        return stats
    peak, area_factor = entry

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Both, as curve_stats masks: an averaged curve carries a NaN pad in X
    # and a combined one carries NaN outside the overlap.
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < PEAK_FIT_MIN_POINTS:
        return stats
    # Sorted into locals, never in place, for the reason curve_stats gives:
    # the caller's x is shared between a scan's curves and is what the
    # export groups its columns by. Sorting matters here too -- 171 and 173
    # step downwards, and the end points below are read as first and last.
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    span = float(xs[-1] - xs[0])
    # Every reading at one motor position: no span to fit along, and the
    # model grid the caller draws would be 400 copies of one point.
    if span <= 0.0:
        return stats

    # Seeded off the straight line through the end points. The largest
    # residual from it gives both the height and the position, and because
    # that residual keeps its sign a dip needs no special case at all --
    # it simply seeds a negative height and fits one.
    slope = (ys[-1] - ys[0]) / span
    offset = ys[0] - slope * xs[0]
    residual = ys - (slope * xs + offset)
    tallest = int(np.argmax(np.abs(residual)))
    width = fwhm_seed if fwhm_seed and fwhm_seed > 0.0 else span / 10.0

    def model(grid, height, centre, fwhm, line_slope, line_offset):
        return height * peak(grid, centre, fwhm) + line_slope * grid + line_offset

    try:
        # catch_warnings on its own only saves and restores the filter
        # state -- it suppresses nothing, and a flat curve prints
        # "OptimizeWarning: Covariance of the parameters could not be
        # estimated" straight to stderr through it. The simplefilter is
        # what silences that, and the RuntimeWarning from an overflowing
        # np.exp on a bad step. This file neither prints nor logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _covariance = curve_fit(
                model,
                xs,
                ys,
                p0=[residual[tallest], xs[tallest], width, slope, offset],
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        return stats
    height, centre, fwhm, line_slope, line_offset = (float(v) for v in popt)

    # The gate. An unbounded five-parameter fit returns nonsense without
    # raising and without complaining: over 400 curves of pure noise, 76
    # came back with a negative width and 33 with a centre outside the data
    # entirely, and the table would have printed those to six figures. The
    # four tests are geometric and cost nothing.
    #
    # The last two are not tidiness. A tall narrow spike between two
    # readings passes the first two happily, and the plot cannot survive
    # it: matplotlib's relim() takes the model line as readily as the data,
    # so the Y axis scales to the spike and the real curve flattens onto
    # the axis. Before these two the worst model spanned 6.6e28 times the
    # data's own Y range; after them, under ten.
    if not np.all(np.isfinite(popt)):
        return stats
    if not xs[0] <= centre <= xs[-1]:
        return stats
    if not 0.0 < fwhm <= span:
        return stats
    spread = float(ys.max() - ys.min())
    if spread <= 0.0 or abs(height) > PEAK_FIT_MAX_HEIGHT * spread:
        return stats
    # A peak narrower than the step size was never resolved -- the model is
    # interpolating between two readings, not measuring a width.
    step = float(np.median(np.diff(xs)))
    if fwhm < step:
        return stats

    # The mixing, for a shape that has one. Refined from the five
    # parameters above rather than solved alongside them from scratch, so
    # that a shape without a mixing goes through exactly the code it
    # always did, and so that a pseudo-Voigt whose mixing will not stand
    # still reports the fit it would have reported anyway -- the answer
    # already in hand is the fallback, at no extra solve.
    eta: float | None = None
    mixed = PEAK_MIX.get(shape)
    # One more parameter wants one more reading, keeping the two degrees
    # of freedom PEAK_FIT_MIN_POINTS was chosen to leave.
    if mixed is not None and xs.size >= PEAK_FIT_MIN_POINTS + 1:
        refined = _peak_mix_refine(
            xs, ys, mixed[0],
            (height, centre, fwhm, line_slope, line_offset),
            span, spread, step,
        )
        if refined is not None:
            height, centre, fwhm, line_slope, line_offset, eta = refined
            area_factor = mixed[1](eta)
    profile = (
        peak(xs, centre, fwhm) if eta is None
        else mixed[0](xs, centre, fwhm, eta)
    )

    # R2 against the background-subtracted data, not the raw curve. With a
    # sloping line in the model the line otherwise explains most of the
    # variance by itself and the number stops discriminating: pure noise
    # scored 0.62 to 0.87 that way, which a reader would take for a real
    # fit. Taken over the peak alone, the same noise sits around 0.29 and
    # the rocking curves stay above 0.96.
    base = line_slope * xs + line_offset
    target = ys - base
    ss_total = float(np.sum((target - target.mean()) ** 2))
    if ss_total > 0.0:
        ss_residual = float(np.sum((target - height * profile) ** 2))
        stats["fit_r2"] = 1.0 - ss_residual / ss_total
    stats["fit_center"] = centre
    stats["fit_fwhm"] = fwhm
    stats["fit_height"] = height
    # Blank for every shape without a mixing, and for one whose mixing was
    # refused: in both the model drawn is the three-parameter shape, and a
    # printed 0.5 would read as a measurement when it is an assumption.
    stats["fit_eta"] = eta
    # The integrated intensity of the peak alone, the sloping line taken
    # off: height x FWHM x the shape's own integral, which is exact rather
    # than a sum over the drawn points. Deliberately the whole line and not
    # just the scanned range, so that the same peak measured over a wider
    # sweep reports the same intensity and the repeats stay comparable --
    # which is the whole reason for fitting them. The price is paid by the
    # Lorentzian, 6 to 11% of whose area on scans 32 to 38 lies in tails
    # beyond where anybody actually measured; the Gaussian extrapolates
    # nothing worth printing and its square under 1%. A dip fits a negative
    # height and so reports a negative area, exactly as the height column
    # does, rather than pretending to an intensity it does not have.
    stats["fit_area"] = height * fwhm * area_factor
    stats["fit_slope"] = line_slope
    stats["fit_offset"] = line_offset
    return stats


def fit_columns(mixed: bool) -> list[str]:
    """The fit headings, the mixing's among them only where it is fitted.

    One function so that the headings and the cells under them cannot be
    built from two different ideas of the order -- ``_set_stats_columns``
    and ``_update_stats_table`` both come here.
    """
    if not mixed:
        return list(STATS_FIT_COLUMNS)
    at = STATS_FIT_COLUMNS.index("Fit FWHM") + 1
    return (
        STATS_FIT_COLUMNS[:at]
        + [STATS_MIX_COLUMN]
        + STATS_FIT_COLUMNS[at:]
    )


def _blank_stats() -> dict[str, float | None]:
    """A statistics row with every curve column empty.

    For the per-peak rows "Multi" adds: a component of a fit has no curve
    of its own, so it has no maximum, no crossing width and no trapezoid.
    Built by asking ``curve_stats`` about no data rather than by listing
    its keys here, because ``_update_stats_table`` reads those columns by
    subscript -- a row missing one raises rather than showing a blank --
    and a list of them kept by hand is a list that drifts.
    """
    return curve_stats(np.empty(0), np.empty(0))


def _peak_seeds(
    xs: np.ndarray,
    ys: np.ndarray,
    peaks: int,
    step: float,
    peak,
    slope: float,
    offset: float,
    rises_first: bool,
) -> list[tuple[float, float, float]]:
    """Starting height, centre and width for each of *peaks* components.

    ``find_peaks`` is run over the data and over its negative, so a dip
    seeds as readily as a rise. That is the whole of the derivative case:
    the derivative of a peak is a lobe up then a lobe down, and seeding
    only maxima finds one of the two -- which is exactly the one-lobe trap
    the "Fit" tooltip has warned about on scans 32 to 38.

    Candidates are taken in order of prominence, and *rises_first* says
    whether every maximum is ranked ahead of every dip or the two signs
    are pooled. Neither order is right on its own, which is why the
    caller tries both and keeps the fit that scores better; the sort
    below says what each of them gets wrong.

    The widths come from ``find_peaks``' own ``widths`` rather than a
    fraction of the scan, and that is not a refinement. A tenth of the
    range on scan 31's derivative is about twice the true lobe width and
    wider than the gap between the lobes; seeded that way the solver
    walked off and returned no fit at all.
    """
    spread = float(np.max(ys) - np.min(ys))
    reach = float(xs[-1] - xs[0])
    found: list[tuple[int, float, float, float, float]] = []
    for rank, sign in ((0, 1.0), (1, -1.0)):
        # width=1 is the loosest setting that makes find_peaks report a
        # width at all; prominence is what keeps it off the noise.
        index, props = find_peaks(
            sign * ys, prominence=PEAK_SEED_PROMINENCE * spread, width=1
        )
        for i, rise, samples in zip(
            index, props["prominences"], props["widths"]
        ):
            found.append(
                (
                    rank,
                    float(rise),
                    sign * float(rise),
                    float(xs[i]),
                    max(float(samples) * step, 2.0 * step),
                )
            )
    # Each ordering has one shape it gets wrong, and they are not the
    # same shape, which is why both are tried.
    #
    # Pooling the signs on prominence alone sounds even-handed and is
    # not. The valley between two separate peaks is about as prominent as
    # the smaller of the two, so a single pool decides between them on a
    # margin of about a percent: a clean pair of Gaussians at 0.3 and
    # 0.65 loses on exactly that -- valley 59.05 against peak 58.30 --
    # and the fit then hangs a wide negative component off the left edge
    # and scores R2 0.955 with every gate below satisfied.
    #
    # Ranking every rise first settles that the way round the data means
    # it, and starves the derivative case instead, wherever the positive
    # side has a candidate to spare. Scan 35's derivative offers two
    # rises and two dips, and its second lobe is the most prominent thing
    # in the scan: taken rises first it is passed over for a 1.7e5 bump,
    # both components come back positive and R2 falls to 0.415, below
    # even the one-lobe fit this is all meant to replace.
    found.sort(
        key=lambda candidate: (
            candidate[0] if rises_first else 0,
            -candidate[1],
        )
    )
    seeds = [(h, c, w) for _rank, _rise, h, c, w in found[:peaks]]

    # Fewer candidates than asked for is the interesting case rather than
    # an error, and it is why the count has to be the user's to set: scan
    # 34's second peak hides under the falling flank of the first, so it
    # is not a local maximum and no peak finder will ever report it.
    #
    # The shortfall is seeded where the data departs furthest from what
    # has been seeded so far -- peak_fit's own trick of taking the largest
    # residual, applied once per missing component instead of once. Where
    # that residual keeps its sign a dip seeds a negative height and fits
    # one, exactly as it does there. Spreading the spare seeds evenly
    # across the scan instead was tried and is worse than it looks: on
    # scan 34 it lands an even third of the way along, to the *left* of
    # the one real candidate, and the fit settles for a 645-count
    # satellite out at 44.80 (R2 0.9818) rather than the shoulder at
    # 44.74 that is actually there (0.9926).
    width = (
        float(np.median([w for _k, _r, _h, _c, w in found])) if found else 0.0
    )
    if width <= 0.0:
        width = max(reach / 10.0, 2.0 * step)
    while len(seeds) < peaks:
        settled = slope * xs + offset
        for height, centre, fwhm in seeds:
            settled = settled + height * peak(xs, centre, fwhm)
        residual = ys - settled
        apart = int(np.argmax(np.abs(residual)))
        # A residual of nothing left to explain seeds no position worth
        # having, and two components on one point will not converge.
        if not np.isfinite(residual[apart]) or residual[apart] == 0.0:
            seeds.append((spread, float(xs[xs.size // 2]), width))
        else:
            seeds.append((float(residual[apart]), float(xs[apart]), width))
    return seeds


def _peak_solve(
    xs: np.ndarray,
    ys: np.ndarray,
    peaks: int,
    step: float,
    peak,
    seeds: list[tuple[float, float, float]],
    slope: float,
    offset: float,
    mixed=None,
) -> tuple[
    float | None,
    list[tuple[float, float, float, float | None]],
    float,
    float,
] | None:
    """Solve one seeding and gate the answer, or ``None`` if it will not do.

    Returns ``(r2, [(centre, fwhm, height, eta), ...], slope, offset)``
    with the components ordered by centre. ``None`` is this seeding
    declined: the solver gave up, or came back with something the gates
    refuse, and the caller is free to try another seeding or to report no
    fit at all.

    *mixed* is the shape's four-parameter form for a shape that has one,
    and giving it is what spends a fourth parameter a component on the
    mixing; left out, every component is solved on three and its ``eta``
    comes back ``None``. The caller solves both ways and keeps the better
    answer -- see PEAK_MIX and ``peak_fit_multi``.

    Expects *xs* sorted, *step* and the data's span and range positive,
    and warnings already silenced -- ``peak_fit_multi`` sees to all four.
    """
    span = float(xs[-1] - xs[0])
    spread = float(ys.max() - ys.min())
    # The stride through the solver's flat parameter vector, which is
    # what the mixing costs: everything below indexes components by it.
    width = 3 if mixed is None else 4
    guess: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for height, centre, fwhm in seeds:
        guess += [height, centre, fwhm]
        # Bounded, where peak_fit only checks afterwards. With several
        # components free at once there is always somewhere for a runaway
        # to hide, so the solver is held to the same geometry the gate
        # below tests: inside the data, and no taller than
        # PEAK_FIT_MAX_HEIGHT times the data's range.
        #
        # The width is the one that deliberately falls short of its gate,
        # at a tenth of a step rather than a step. A bound is a place the
        # solver is allowed to sit, so a bound at the gate is a gate that
        # can never fire -- and it did not: scan 171's derivative, which
        # is too ragged for its own step size and which peak_fit declines
        # outright, came back as two components pinned at exactly one
        # step and R2 0.17. Left room beneath it, the solver goes under,
        # the strict test below catches it, and the two paths agree
        # again. The bound is kept off zero only because a width of
        # nothing divides by nothing in every shape in PEAK_SHAPES.
        lower += [-PEAK_FIT_MAX_HEIGHT * spread, xs[0], step / 10.0]
        upper += [PEAK_FIT_MAX_HEIGHT * spread, xs[-1], span]
        if mixed is not None:
            # Seeded at the half the fixed shape assumes, so the mixed
            # solve starts from the shape the fixed one would have
            # fitted, and bounded by the two pure shapes it blends --
            # outside them one of the halves turns negative and the
            # profile stops being a peak. Unlike the width above, these
            # two are deliberately set *at* the gate rather than beyond
            # it, and the margin that lets the gate still fire is kept
            # on the gate's side instead -- see PEAK_MIX_RAIL. A mixing
            # resting on either wall is the solver asking for a pure
            # Gauss or a pure Lorentzian, which is what gets caught.
            guess.append(0.5)
            lower.append(0.0)
            upper.append(1.0)
    guess += [slope, offset]
    lower += [-np.inf, -np.inf]
    upper += [np.inf, np.inf]
    # curve_fit raises ValueError on a seed outside its bounds, and a
    # padded seed on a short scan can be: reach/10 is under one step
    # wherever there are fewer than ten readings.
    guess = [min(max(g, lo), hi) for g, lo, hi in zip(guess, lower, upper)]

    def model(grid, *free):
        total = free[-2] * grid + free[-1]
        for k in range(peaks):
            block = free[width * k:width * k + width]
            total = total + block[0] * (
                peak(grid, block[1], block[2]) if mixed is None
                else mixed(grid, block[1], block[2], block[3])
            )
        return total

    try:
        popt, _covariance = curve_fit(
            model,
            xs,
            ys,
            p0=guess,
            bounds=(lower, upper),
            maxfev=PEAK_FIT_MULTI_MAXFEV,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(popt)):
        return None
    line_slope, line_offset = float(popt[-2]), float(popt[-1])

    found: list[tuple[float, float, float, float | None]] = []
    for k in range(peaks):
        height = float(popt[width * k])
        centre = float(popt[width * k + 1])
        fwhm = float(popt[width * k + 2])
        eta = None if mixed is None else float(popt[width * k + 3])
        # peak_fit's four geometric tests, once per component, and the
        # whole seeding declined when any one of them fails. Declined
        # whole rather than per component because the model is drawn as a
        # sum: one runaway takes the Y axis with it however sane its
        # neighbours are, which is the matplotlib relim() failure the
        # height test was put there to stop. A solver sitting exactly on
        # a bound trips the same tests, since the bounds are that
        # geometry.
        if not xs[0] <= centre <= xs[-1]:
            return None
        if not 0.0 < fwhm <= span:
            return None
        if abs(height) > PEAK_FIT_MAX_HEIGHT * spread:
            return None
        if fwhm < step:
            return None
        # And the mixing clear of its two ends, where the four above are
        # the geometry. A component resting on 0 or 1 declines the whole
        # mixed seeding, which costs nothing: peak_fit_multi solves the
        # fixed shape alongside this one and reports that instead. Scan
        # 32 at two peaks is the measured case.
        if eta is not None and not (
            PEAK_MIX_RAIL <= eta <= 1.0 - PEAK_MIX_RAIL
        ):
            return None
        found.append((centre, fwhm, height, eta))
    # By centre, so "peak 1" is the leftmost -- see peak_fit_multi. By
    # the centre alone and not the whole tuple, so that the mixing, which
    # is None for every shape without one, is never compared.
    found.sort(key=lambda component: component[0])

    # R2 against the background-subtracted data, for the reason peak_fit
    # gives: with a sloping line in the model the line explains most of
    # the variance by itself and the number stops discriminating. One R2
    # for the whole model, since the components were fitted together and
    # none of them has an R2 of its own.
    base = line_slope * xs + line_offset
    target = ys - base
    ss_total = float(np.sum((target - target.mean()) ** 2))
    r2: float | None = None
    if ss_total > 0.0:
        shaped = np.zeros_like(xs)
        for centre, fwhm, height, eta in found:
            shaped = shaped + height * (
                peak(xs, centre, fwhm) if eta is None
                else mixed(xs, centre, fwhm, eta)
            )
        r2 = 1.0 - float(np.sum((target - shaped) ** 2)) / ss_total
    return r2, found, line_slope, line_offset


def peak_fit_multi(
    x: np.ndarray,
    y: np.ndarray,
    shape: str,
    peaks: int,
) -> dict[str, float | None]:
    """Fit *peaks* copies of *shape* over one shared sloping line.

    The model is ``sum(height_k * peak(x; centre_k, fwhm_k))`` plus
    ``slope * x + offset``: three parameters a component and two for the
    line, which the components share because one background under several
    peaks is what the data has.

    Beside the keys ``peak_fit`` fills, this adds ``fit_peaks`` -- a list
    of one row per component, **ordered by centre**. That ordering is what
    makes a trend across scans mean anything: "peak 1" is then always the
    leftmost, and its line follows the same physical peak through a set of
    repeats even where the two swap which is the taller.

    The four per-peak values on the returned dict itself stay ``None``, so
    the curve's own row shows five blank cells and the numbers appear on
    the peak rows underneath it. ``fit_slope``/``fit_offset`` are kept,
    because the shared line is what ``peak_model`` needs to draw the sum.

    Two things this cannot do, both measured and both in the tooltips.

    R2 cannot choose *peaks* for you. The models are nested, so it only
    ever rises: on scan 32, a single clean rocking curve, one peak scores
    0.97832 and two score 0.99761.

    And a component too many does not announce itself by failing. Asked
    for two, a single peak is *divided* rather than fitted alongside a
    rejected spare, and the division need not be lopsided: scan 32 comes
    back as a broad base (FWHM 0.0914) under a sharp core (0.0347) at
    much the same place, and neither of those is the 0.0447 that one peak
    reports and that the crossings agree with. The derivative of a knife
    edge divides too -- scan 101 grows a negative component at -6.4542
    and its main width moves 0.02095 to 0.02340, 12%, which is more than
    the drift the whole trend exists to show. Every component is therefore
    given a row of its own and the gate below stays geometric: a broad
    base is as sane a shape as a peak, and no test here can know it was
    not wanted.
    """
    stats: dict[str, float | None] = {
        "fit_center": None,
        "fit_fwhm": None,
        "fit_height": None,
        "fit_area": None,
        "fit_eta": None,
        "fit_r2": None,
        "fit_slope": None,
        "fit_offset": None,
        "fit_peaks": None,
    }
    entry = PEAK_SHAPES.get(shape)
    if entry is None or peaks < 1:
        return stats
    peak, area_factor = entry
    mixed = PEAK_MIX.get(shape)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Both, as peak_fit and curve_stats mask: an averaged curve carries a
    # NaN pad in X and a combined one carries NaN outside the overlap.
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    # Three parameters a component against peak_fit's seven points for
    # five, written so that one peak gives PEAK_FIT_MIN_POINTS back exactly.
    if x.size < PEAK_FIT_MIN_POINTS + 3 * (peaks - 1):
        return stats
    # Sorted into locals, never in place, for the reason curve_stats gives:
    # the caller's x is shared between a scan's curves and is what the
    # export groups its columns by.
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    span = float(xs[-1] - xs[0])
    spread = float(ys.max() - ys.min())
    if span <= 0.0 or spread <= 0.0:
        return stats
    # The lower bound on every width below, so it has to be positive or
    # the bounds themselves are malformed. Zero where more than half the
    # readings repeat an X position.
    step = float(np.median(np.diff(xs)))
    if step <= 0.0:
        return stats

    try:
        # As peak_fit: catch_warnings only saves and restores the filter
        # state, and it is the simplefilter that silences the covariance
        # and overflow warnings scipy prints straight to stderr. polyfit
        # is inside it too, for its RankWarning on a flat end. This file
        # neither prints nor logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The background is seeded from a line through both ends of
            # the scan rather than peak_fit's chord between the two end
            # points. A derivative sits at about zero at either end, so
            # that chord is mostly noise, and seeding the offset from it
            # put scan 31's fit in the wrong basin. A tenth of the points
            # each side, never fewer than two.
            edge = max(2, xs.size // 10)
            slope, offset = np.polyfit(
                np.concatenate((xs[:edge], xs[-edge:])),
                np.concatenate((ys[:edge], ys[-edge:])),
                1,
            )
            # Both the orderings _peak_seeds offers, keeping whichever
            # fits better, because each of them has a shape it gets
            # wrong and they are not the same shape -- the comment there
            # names the two scans. Comparing R2 between these two is
            # sound where comparing it between peak counts is not: this
            # is one model with one parameter count, solved twice from
            # different starting points, so the better number is the
            # better fit of the same thing and not a reward for more
            # freedom.
            seedings: list[list[tuple[float, float, float]]] = []
            for rises_first in (False, True):
                seeds = _peak_seeds(
                    xs, ys, peaks, step, peak,
                    float(slope), float(offset), rises_first,
                )
                # The two orderings agree whenever find_peaks offers no
                # more candidates than were asked for, which is most
                # scans; solving the same seeding twice is only slow.
                if seeds not in seedings:
                    seedings.append(seeds)
            # And, for a shape with a mixing, every one of those
            # seedings solved twice more: once with the mixing free and
            # once with it held at a half, keeping whichever scores
            # best. The free solves are tried first so that a tie keeps
            # the measured mixing over the assumed one.
            #
            # Scored against each other rather than simply trusted
            # because the free model contains the fixed one: nesting
            # promises that the better optimum exists, not that this
            # solver walked to it, and a freed mixing is one more
            # direction in which to get lost. The caution is the same
            # one the two seedings already get, and it is free -- where
            # the mixing is turned down the fixed answer is in hand.
            attempts = [None]
            # A fourth parameter a component asks a reading a component
            # more, on the same terms the minimum above is written to.
            if (
                mixed is not None
                and xs.size >= PEAK_FIT_MIN_POINTS + 4 * peaks - 3
            ):
                attempts.insert(0, mixed[0])
            best = None
            best_score = -np.inf
            for mix in attempts:
                for seeds in seedings:
                    got = _peak_solve(
                        xs, ys, peaks, step, peak, seeds,
                        float(slope), float(offset), mix,
                    )
                    if got is None:
                        continue
                    # A fit whose R2 could not be formed -- the
                    # background alone accounts for the data -- is still
                    # a fit, and so beats nothing while losing to
                    # anything scored.
                    score = -np.inf if got[0] is None else got[0]
                    if best is None or score > best_score:
                        best, best_score = got, score
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return stats
    if best is None:
        return stats
    r2, found, line_slope, line_offset = best

    components: list[dict[str, float | None]] = []
    for centre, fwhm, height, eta in found:
        component = _blank_stats()
        component.update(
            {
                "fit_center": centre,
                "fit_fwhm": fwhm,
                "fit_height": height,
                # Each component's own integrated intensity, on the same
                # terms peak_fit states: height x width x the shape's
                # integral over the whole line, the shared background
                # off. The integral is the mixing's where one was
                # fitted, so two components of one fit can take two
                # different factors -- they are two different shapes.
                "fit_area": height * fwhm * (
                    area_factor if eta is None else mixed[1](eta)
                ),
                # Blank unless this component's own mixing was fitted
                # and kept, which peak_fit says more about.
                "fit_eta": eta,
                "fit_r2": r2,
                "fit_slope": line_slope,
                "fit_offset": line_offset,
            }
        )
        components.append(component)
    stats["fit_slope"] = line_slope
    stats["fit_offset"] = line_offset
    stats["fit_peaks"] = components
    return stats


# How many points the drawn model is evaluated at. Enough that a narrow
# peak still reads as a curve rather than a polygon.
PEAK_FIT_PLOT_POINTS = 400


def _peak_shaped(
    peak, mixed, stats: dict[str, float | None], x: np.ndarray
) -> np.ndarray:
    """One fitted row's unit-height profile, its own mixing applied.

    *stats* is a whole fit or one of its components. ``fit_eta`` is
    there and not ``None`` only where a mixing was fitted and kept, so a
    shape that has none -- and a pseudo-Voigt whose mixing was refused
    and which fell back to the fixed half -- goes through *peak* exactly
    as it did before there was a mixing to fit.
    """
    eta = stats.get("fit_eta")
    if mixed is not None and eta is not None:
        return mixed[0](x, stats["fit_center"], stats["fit_fwhm"], eta)
    return peak(x, stats["fit_center"], stats["fit_fwhm"])


def peak_model(
    x: np.ndarray, shape: str, stats: dict[str, float | None]
) -> np.ndarray | None:
    """Evaluate over *x* the fit ``peak_fit`` returned in *stats*.

    ``None`` when there is no fit to draw, so the caller can ask without
    checking first. Uses ``.get`` throughout because a row that was not the
    one fitted -- the curve while dY/dX is shown, the derivative while it is
    not -- and any colour map never carry these keys at all.

    A ``peak_fit_multi`` result is drawn as the one summed model its
    components make, not as a line apiece: they were fitted together over
    a single background, and several lines in one colour over one curve
    would read as several fits rather than one.
    """
    entry = PEAK_SHAPES.get(shape)
    if entry is None:
        return None
    peak, _area_factor = entry
    mixed = PEAK_MIX.get(shape)
    components = stats.get("fit_peaks")
    if components:
        total = stats["fit_slope"] * x + stats["fit_offset"]
        for component in components:
            total = total + component["fit_height"] * _peak_shaped(
                peak, mixed, component, x
            )
        return total
    if stats.get("fit_center") is None:
        return None
    return (
        stats["fit_height"] * _peak_shaped(peak, mixed, stats, x)
        + stats["fit_slope"] * x
        + stats["fit_offset"]
    )


def peak_parts(
    x: np.ndarray, shape: str, stats: dict[str, float | None]
) -> list[np.ndarray]:
    """The pieces the fit in *stats* is made of, for "Parts" to draw.

    The fitted background first, then one line a component -- each drawn
    **on** that background rather than from zero, so a component lies
    where the data it is claiming lies. From zero they would be readable
    against each other and nowhere near the curve: scan 109's background
    is about 800 counts, so every component would float a screen below
    its own peak.

    Empty where there is nothing to add to what ``peak_model`` already
    drew. A single-component fit is its background plus its one peak, so
    the component line would be the summed model over again, in the same
    colour, and only the background is worth a line of its own.
    """
    entry = PEAK_SHAPES.get(shape)
    if entry is None or stats.get("fit_slope") is None:
        return []
    peak, _area_factor = entry
    mixed = PEAK_MIX.get(shape)
    base = stats["fit_slope"] * x + stats["fit_offset"]
    components = stats.get("fit_peaks") or []
    parts = [base]
    if len(components) > 1:
        for component in components:
            parts.append(
                base
                + component["fit_height"]
                * _peak_shaped(peak, mixed, component, x)
            )
    return parts


