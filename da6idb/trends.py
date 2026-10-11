"""What was stepped across a set of scans, and plotting against it.

A row of scans that look alike is usually a series: something was changed
between them. The baseline records where everything stood when each ran, so
the answer is in the files -- the trick is finding which of a few hundred
fields actually moved, rather than drifted.

Ported from the 6-ID-C fork of ``scan_viewer``. The ranking is unchanged;
the plumbing around it is ours, the fork's being built on curve dicts this
package does not have.

numpy only -- no Qt, no pandas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from da6idb.files import scan_number
from da6idb.scan import read_baseline

# The trend's X axis when it is not a baseline field, and the default.
TREND_X_SCAN = "Scan number"
# What counts as "deliberately moved" when deciding which baseline fields
# are worth offering as that axis. A field's noise is taken as how far it
# drifted *within* a scan -- the gap between the readings before and after
# -- plus a floor proportional to its own size, since a field that never
# moves at all reports no drift and would otherwise divide by zero.
BASELINE_TREND_NOISE = 1e-4
BASELINE_TREND_SNR = 10.0

_BASELINE_CACHE: dict[tuple, dict[str, tuple[float, float]]] = {}
_TREND_FIELD_CACHE: dict[tuple, list[str]] = {}


def _baseline_numbers(path: Path) -> dict[str, tuple[float, float]]:
    """Every numeric baseline field of *path*, as its before/after pair.

    Numeric fields only: a baseline holds motor directions and UB matrices
    alongside the readbacks, and :func:`_scalarize` hands those back as text.
    A field the scan did not record, or recorded as text, is left out rather
    than filled in, so a caller can tell "not measured" from a value.

    Remembered per (path, mtime, size) the way :func:`reduce_channels` is,
    and the whole baseline at once rather than the fields asked for:
    :func:`read_baseline` reads all 322 of them whatever is wanted, three
    callers here want three different handfuls, and every redraw asks again
    for every selected scan. Keying on the field list too would open each
    file three times over for one answer. ``OSError`` gives ``{}``, which
    leaves the scan ungroupable rather than taking the window down.

    Both ends are kept rather than averaged away because the gap between
    them is the only noise estimate one scan carries on its own, which is
    what :func:`baseline_trend_fields` ranks by.
    """
    path = Path(path)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        key = None
    if key is not None and key in _BASELINE_CACHE:
        return _BASELINE_CACHE[key]
    try:
        baseline = read_baseline(path)
    except OSError:
        baseline = {}
    numbers: dict[str, tuple[float, float]] = {}
    for name, pair in baseline.items():
        try:
            numbers[name] = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError):
            continue
    if key is not None:
        _BASELINE_CACHE[key] = numbers
    return numbers


def baseline_trend_fields(paths: list[Path]) -> list[str]:
    """Baseline fields worth trending against, most deliberate first.

    The answer to "what was stepped across these scans": every numeric field
    the whole selection recorded, whose motion over the selection clears the
    noise bar BASELINE_TREND_SNR and BASELINE_TREND_NOISE describe, best
    first. Fewer than two scans gives ``[]`` -- one point has nothing to be
    plotted against.

    Ordered by how far each moved relative to its own noise rather than
    alphabetically, so the motor somebody deliberately drove comes out at
    the top: scans 32 to 39 put huber_euler_ana_eta third of 19, behind only
    its own setpoint and the huber_hp alias that reads the same encoder.

    The ranking is of one selection against itself, so it says nothing about
    a wider one. Over all 146 TiSe2 scans at once eta falls below the bar
    altogether, correctly -- somewhere in those 146 it was moving *during* a
    scan, and a field that will not hold still inside one scan cannot be the
    axis that scan sits at.
    """
    paths = list(paths)
    if len(paths) < 2:
        return []
    # Keyed on the same (path, mtime, size) the readings underneath are, and
    # not on the names alone: a scan still being written to is one the
    # listing offers like any other, and on the names alone it would keep
    # whatever ranking its first few points happened to earn. Worth a cache
    # at all because the ranking is 14 ms over a 27 scan selection and every
    # redraw asks for it again, where the readings behind it are already
    # remembered and cost nothing the second time.
    stamps = []
    for name in sorted(str(path) for path in paths):
        try:
            stat = Path(name).stat()
            stamps.append((name, stat.st_mtime, stat.st_size))
        except OSError:
            stamps.append((name, None, None))
    key = tuple(stamps)
    if key in _TREND_FIELD_CACHE:
        return _TREND_FIELD_CACHE[key]
    pairs = [_baseline_numbers(path) for path in paths]
    common = set(pairs[0]).intersection(*pairs[1:]) if pairs else set()
    scored: list[tuple[float, str]] = []
    for name in common:
        readings = [reading[name] for reading in pairs]
        if not all(
            np.isfinite(before) and np.isfinite(after)
            for before, after in readings
        ):
            continue
        values = [(before + after) / 2.0 for before, after in readings]
        across = max(values) - min(values)
        if across <= 0.0:
            continue
        within = max(abs(after - before) for before, after in readings)
        floor = within + BASELINE_TREND_NOISE * max(
            abs(value) for value in values
        )
        if floor <= 0.0 or across / floor > BASELINE_TREND_SNR:
            scored.append((across / floor if floor > 0.0 else np.inf, name))
    # By score, then by name, so that the aliases reading one encoder -- and
    # they are legion here -- come out in a stable order rather than
    # whichever the set happened to yield.
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    fields = [name for _score, name in scored]
    _TREND_FIELD_CACHE[key] = fields
    return fields


# The quantity a trend follows, when the whole fit is one peak.
TREND_TOTAL_AREA = "Total area"


def trend_choices(peaks: int = 1, mixed: bool = False) -> dict:
    """What a trend can follow, as ``{label: (component, fit key)}``.

    One peak offers the fit's own numbers and *component* is ``None``. Two
    or more offer each component's separately -- "Peak 1 center" and so on
    -- because that is the question a multi-peak trend is asked: not how
    the fit as a whole drifted but how each peak did.

    The components are numbered as :func:`da6idb.fit.peak_fit_multi`
    returns them, in order of centre, so "Peak 1" is the leftmost in every
    scan. That is what makes a trend of it mean anything: the line follows
    one peak through the set rather than jumping whenever two swap which
    is the taller.

    :data:`TREND_TOTAL_AREA` is offered alongside them, being the one
    quantity of a several-peak fit that belongs to the fit rather than to
    any component. *mixed* adds the mixing, for the shape that fits one.
    """
    keys = [("center", "fit_center"), ("FWHM", "fit_fwhm")]
    if mixed:
        keys.append(("eta", "fit_eta"))
    keys += [("height", "fit_height"), ("area", "fit_area")]
    if int(peaks) <= 1:
        return {f"Fit {name}": (None, key) for name, key in keys}
    out = {}
    for index in range(int(peaks)):
        for name, key in keys:
            out[f"Peak {index + 1} {name}"] = (index, key)
    out[TREND_TOTAL_AREA] = (None, "fit_area_total")
    return out


def trend_value(stats: dict, choice) -> float | None:
    """Read one :func:`trend_choices` entry out of a fit, or None.

    ``None`` wherever the fit has nothing to say -- it was refused, or it
    came back with fewer components than were asked for -- so a scan that
    would not fit simply leaves a gap in the line.
    """
    component, key = choice
    parts = stats.get("fit_peaks") or []
    if key == "fit_area_total":
        areas = [part.get("fit_area") for part in parts] or [stats.get("fit_area")]
        return (float(sum(areas)) if areas and all(a is not None for a in areas)
                else None)
    if component is None:
        value = stats.get(key)
        return None if value is None else float(value)
    if component >= len(parts):
        return None
    value = parts[component].get(key)
    return None if value is None else float(value)


def trend_axis(paths, against: str) -> dict[str, float]:
    """Where each scan sits on the trend axis, keyed by ``str(path)``.

    :data:`TREND_X_SCAN` reads the number out of the file name; anything
    else is a baseline field, taken as the mean of its readings before and
    after -- the scan sat between them. A scan that cannot supply a value
    is left out rather than given one, since a point with no x has nowhere
    to go.
    """
    out: dict[str, float] = {}
    for path in paths:
        path = Path(path)
        if against == TREND_X_SCAN:
            digits = scan_number(path).lstrip("0")
            if digits.isdigit():
                out[str(path)] = float(digits)
            continue
        pair = _baseline_numbers(path).get(against)
        if pair is None:
            continue
        before, after = pair
        if np.isfinite(before) and np.isfinite(after):
            out[str(path)] = (before + after) / 2.0
    return out
