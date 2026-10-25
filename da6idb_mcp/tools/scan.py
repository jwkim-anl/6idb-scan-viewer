"""What the scan browser does: curves, peak statistics, fits and trends.

Deliberately no ``from __future__ import annotations`` -- the schema a
model is shown is read off these annotations, and a string would have to
be resolved in a module the registry cannot see.
"""

from typing import Optional, Union

from da6idb_mcp import cache, render, resolve
from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool

#: Accepted everywhere a y field is asked for.
Fields = Union[str, list]

SHAPES = ("Gauss", "Lorentzian", "Lorentzian squared", "Pseudo-Voigt")


def _wanted(y) -> list:
    return [y] if isinstance(y, str) else list(y)


def _curve_figure(curve, title, log=False):
    fig, ax = render.figure()
    for column in curve.columns:
        ax.plot(curve.index, curve[column], marker=".", lw=1.2, label=str(column))
    ax.set_xlabel(str(curve.index.name or "x"))
    if log:
        ax.set_yscale("log")
    if len(curve.columns) > 1:
        ax.legend(fontsize="small")
    else:
        ax.set_ylabel(str(curve.columns[0]))
    ax.set_title(title, fontsize="medium")
    ax.grid(alpha=0.3)
    return fig


@tool()
def scan_list(where: Optional[str] = None) -> Reply:
    """Every scan in a folder, with its plan, motors, range and point count.

    The first call in almost any session: the `scan` number from this
    table is what every other scan tool takes. *where* defaults to the
    folder the server was started on.
    """
    directory = resolve.folder(where)
    scans = cache.scanset(directory)
    return Reply(data={"folder": str(directory), "scans": scans.summary()})


@tool()
def scan_info(scan: Union[int, str], where: Optional[str] = None) -> Reply:
    """One scan's metadata: plan, motors, points, and what hangs off it.

    Says whether a detector file was found, which is what decides if the
    `image_*` and `hkl_*` tools can be used on this scan at all.
    """
    from da6idb.files import resolve_image_file

    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    image = resolve_image_file(master)
    return Reply(data={
        "number": one.number,
        "plan": one.plan,
        "motors": one.motors,
        "points": one.points,
        "master": str(master),
        "image_file": str(image) if image else None,
        "info": one.info,
    }, note=None if image else "no detector file beside this master")


@tool()
def scan_fields(scan: Union[int, str], stream: str = "primary",
                where: Optional[str] = None) -> Reply:
    """Every field that can be plotted, with its role, units and range.

    Read this before choosing a `y`. `role` marks the scanned motor and
    the detector channels; `hinted` marks the detector the plan itself
    nominated, which is the sensible default y. A channel that barely
    varies and stays positive -- an ion chamber -- is your `normalise`.
    """
    master = resolve.scan_master(scan, where)
    frame = cache.scan(master).fields(stream)
    hinted = [str(n) for n in frame.index[frame["hinted"].astype(bool)]]
    return Reply(
        data={"stream": stream, "fields": frame, "hinted": hinted},
        note=f"suggested y: {hinted[0]}" if hinted else "",
    )


@tool()
def scan_curve(scan: Union[int, str], y: Fields, x: Optional[str] = None,
               normalise: Optional[str] = None, log: bool = False,
               where: Optional[str] = None) -> Reply:
    """One or more detector channels against the scanned motor, plotted.

    With no *x* the scan's own motor is used. *normalise* divides by a
    monitor channel, which is how these curves are read at the beamline;
    the resulting column is named ``y/monitor``.
    """
    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    curve = one.curve(x=x, y=_wanted(y), normalise=normalise)
    fig = _curve_figure(curve, f"scan {one.number}", log=log)
    return Reply(data=curve, figure=fig)


@tool()
def scan_statistics(scan: Union[int, str], y: Fields, x: Optional[str] = None,
                    normalise: Optional[str] = None,
                    where: Optional[str] = None) -> Reply:
    """Peak statistics per channel: maximum, minimum, centre and width.

    Read straight off the curve -- the centre is its centre of mass and
    the width a half-maximum crossing -- with no model assumed. For fitted
    numbers with a background removed, use `scan_fit`.
    """
    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    return Reply(data=one.statistics(y=_wanted(y), x=x, normalise=normalise))


def _fit_one(curve, column, shape, peaks, seed):
    from da6idb import fit as fitting

    xs = curve.index.to_numpy(dtype=float)
    ys = curve[column].to_numpy(dtype=float)
    if int(peaks) > 1:
        return fitting.peak_fit_multi(xs, ys, shape, int(peaks))
    return fitting.peak_fit(xs, ys, shape, fwhm_seed=seed)


@tool()
def scan_fit(scan: Union[int, str], y: str, shape: str = "Gauss",
             peaks: int = 1, x: Optional[str] = None,
             normalise: Optional[str] = None,
             where: Optional[str] = None) -> Reply:
    """Fit a peak and a sloping background, and plot the model over the data.

    *shape* is one of Gauss, Lorentzian, Lorentzian squared, Pseudo-Voigt.
    The background is always a sloping line and is not optional, so the
    fitted height sits below the curve's own maximum. With *peaks* above
    one the components come back in `fit_peaks`, ordered by centre.

    Every value comes back null when no trustworthy fit was found; that is
    a real answer, not an error, and happens often by design.
    """
    from da6idb import fit as fitting

    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {', '.join(SHAPES)}")
    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    curve = one.curve(x=x, y=y, normalise=normalise)
    column = curve.columns[0]
    stats = one.statistics(y=y, x=x, normalise=normalise)
    seed = float(stats.loc[column, "fwhm"]) if "fwhm" in stats else None

    result = _fit_one(curve, column, shape, peaks, seed)

    xs = curve.index.to_numpy(dtype=float)
    fig = _curve_figure(curve, f"scan {one.number} — {shape}")
    model = fitting.peak_model(xs, shape, result)
    if model is not None:
        fig.axes[0].plot(xs, model, lw=1.6, color="crimson", label="fit")
        for part in fitting.peak_parts(xs, shape, result):
            fig.axes[0].plot(xs, part, lw=0.9, ls="--", alpha=0.7)
        fig.axes[0].legend(fontsize="small")

    return Reply(
        data={"column": str(column), "shape": shape, "peaks": int(peaks),
              "fit": result},
        figure=fig,
        note="" if model is not None else "no trustworthy fit; values are null",
    )


@tool()
def scan_map(scan: Union[int, str], value: str, normalise: str = "",
             log: bool = False, where: Optional[str] = None) -> Reply:
    """A two-motor raster as a colour map, for a grid scan.

    Returns nothing plottable when the scan swept one motor along a line
    rather than rastering two -- use `scan_curve` for those.
    """
    from matplotlib.colors import LogNorm

    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    raster = one.map(value, normalise=normalise)
    if raster is None:
        return Reply(data={"map": None},
                     note=f"scan {one.number} is not a two-motor raster "
                          f"({', '.join(one.motors) or 'no motors'})")

    y_name, x_name = (str(d) for d in raster.dims)
    fig, ax = render.figure((6.4, 5.2))
    mesh = ax.pcolormesh(
        raster.coords[x_name].values, raster.coords[y_name].values,
        raster.values, shading="nearest", cmap="viridis",
        norm=LogNorm() if log else None)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"scan {one.number} — {value}", fontsize="medium")
    fig.colorbar(mesh, ax=ax)
    return Reply(data=raster, figure=fig)


@tool()
def scan_baseline(scan: Union[int, str], contains: Optional[str] = None,
                  where: Optional[str] = None) -> Reply:
    """What the baseline stream recorded before and after the scan.

    This is where the unscanned diffractometer angles live. There are
    typically hundreds of fields, so pass *contains* -- a case-insensitive
    substring such as ``"psic"`` or ``"temp"`` -- to cut it down.
    """
    master = resolve.scan_master(scan, where)
    frame = cache.scan(master).baseline()
    total = len(frame)
    if contains:
        frame = frame[frame.index.str.contains(contains, case=False)]
    return Reply(
        data={"fields": frame},
        note=f"{len(frame)} of {total} baseline fields" if contains else "",
    )


@tool()
def scan_hkl_position(scan: Union[int, str], where: Optional[str] = None) -> Reply:
    """Where in reciprocal space the scan sat: H, K and L, mean and spread.

    The spread is how far each moved during the scan, so a rocking curve
    shows one axis spread much wider than the other two.
    """
    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    position = one.hkl_position()
    if position is None:
        return Reply(data={"hkl": None},
                     note=f"scan {one.number} records no HKL")
    return Reply(data={"hkl": position})


@tool()
def scan_trend(scans: Union[str, list], y: str, against: str = "Scan number",
               follow: str = "Fit center", shape: str = "Gauss",
               peaks: int = 1, normalise: Optional[str] = None,
               where: Optional[str] = None) -> Reply:
    """Fit one channel across many scans and plot how a fitted value moved.

    *scans* takes ``"151-160"``, ``"151,153,158"``, ``"151-160:2"`` or a
    list. *against* is ``"Scan number"`` or any baseline field that was
    stepped across the selection -- call this once and the note lists the
    fields that actually moved. *follow* is a label from `trend_choices`:
    ``"Fit center"``, ``"Fit FWHM"``, ``"Fit height"``, ``"Fit area"``, or
    ``"Peak 2 center"`` and ``"Total area"`` once *peaks* is above one.

    A scan that would not fit leaves a gap rather than a wrong point.
    """
    from da6idb import trends

    masters = resolve.scan_masters(scans, where)
    if len(masters) < 2:
        raise ValueError("a trend needs at least two scans")

    mixed = shape == "Pseudo-Voigt"
    choices = trends.trend_choices(peaks=int(peaks), mixed=mixed)
    if follow not in choices:
        raise ValueError(
            f"follow must be one of: {', '.join(choices)}")
    choice = choices[follow]

    axis = trends.trend_axis(masters, against)
    moved = trends.baseline_trend_fields(masters)

    rows = []
    for master in masters:
        one = cache.scan(master)
        try:
            curve = one.curve(y=y, normalise=normalise)
        except (KeyError, ValueError) as problem:
            rows.append({"scan": int(one.number), "x": axis.get(str(master)),
                         "value": None, "note": str(problem)})
            continue
        column = curve.columns[0]
        stats = one.statistics(y=y, normalise=normalise)
        seed = float(stats.loc[column, "fwhm"]) if "fwhm" in stats else None
        result = _fit_one(curve, column, shape, peaks, seed)
        rows.append({"scan": int(one.number), "x": axis.get(str(master)),
                     "value": trends.trend_value(result, choice), "note": ""})

    import pandas as pd

    frame = pd.DataFrame(rows).set_index("scan")
    drawn = frame.dropna(subset=["x", "value"])

    fig, ax = render.figure()
    ax.plot(drawn["x"], drawn["value"], marker="o", lw=1.2)
    ax.set_xlabel(against)
    ax.set_ylabel(follow)
    ax.set_title(f"{follow} of {y} over {len(masters)} scans", fontsize="medium")
    ax.grid(alpha=0.3)

    missing = len(frame) - len(drawn)
    note = f"{len(drawn)} of {len(masters)} scans fitted"
    if missing:
        note += f"; {missing} left a gap"
    if moved:
        note += f". Fields stepped across this selection: {', '.join(moved[:6])}"
    return Reply(data={"trend": frame, "follow": follow, "against": against},
                 figure=fig, note=note)


@tool()
def scan_export(scan: Union[int, str], y: Fields, path: Optional[str] = None,
                x: Optional[str] = None, normalise: Optional[str] = None,
                where: Optional[str] = None) -> Reply:
    """Write a scan's curves to a file: .txt, .dat, .csv, .npz, .h5.

    The extension chooses the format. With no *path* it goes to the
    server's artefact folder and the reply says where.
    """
    from da6idb.export import table_columns, write_columns

    master = resolve.scan_master(scan, where)
    one = cache.scan(master)
    curve = one.curve(x=x, y=_wanted(y), normalise=normalise)
    label = str(curve.index.name or "x")
    xs = curve.index.to_numpy(dtype=float)
    columns = table_columns(
        [(str(c), xs, curve[c].to_numpy(dtype=float)) for c in curve.columns],
        x_label=label)

    out = resolve.artefact(path, suffix=".txt", stem=f"scan{one.number}")
    description = [f"scan {one.number} from {master.name}",
                   f"plan: {one.plan}  motors: {', '.join(one.motors)}"]
    write_columns(out, [(f"scan{one.number}", label, columns)],
                  description=description)
    return Reply(data={"columns": list(columns), "rows": len(xs)},
                 artefacts=[out])
