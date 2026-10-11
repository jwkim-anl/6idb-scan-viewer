"""Telling repeats of one measurement apart, and putting them together.

Ported from the 6-ID-C fork of ``scan_viewer``. The clustering, the
step-relative tolerance and the interpolation are unchanged; what had to
change is :data:`SAMPLE_STAGES`, which names a beamline's own sample
stages and is the one thing in here that cannot be written once for every
endstation.

numpy only -- no Qt, no pandas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from da6idb.scan import read_baseline, read_scan_info

# How far apart two readings of the same thing may sit and still count as
# one setting. Degrees and millimetres want different numbers, so each
# stage carries its own.
TEMPERATURE_FIELD = "temp340_sample"
TEMPERATURE_ATOL = 2.0
# Where the sample sits, for telling repeats of one measurement from a
# series that stepped something between them. One set per endstation, and
# a scan has to carry a whole set before it can be grouped at all: a stage
# nobody listed here would otherwise leave every scan standing on it
# looking alike, and a series would average into one clean wrong curve.
SAMPLE_STAGES = (
    {"cryo_x": 0.01, "cryo_y": 0.01, "cryo_z": 0.01},
)
# How far two scans' X may drift apart and still be the same points
# measured twice, as a fraction of one step. Past half a step, point i of
# one scan is nearer point i + 1 of the other, and averaging them row by
# row would smear the curve rather than clean it.
SAME_X_STEPS = 0.5


def baseline_values(path, fields: tuple[str, ...]) -> dict[str, float]:
    """The baseline's own reading of each of *fields* that it carries.

    Before the scan, not after: where the sample sat while it ran.
    """
    recorded = read_baseline(path)
    out: dict[str, float] = {}
    for name in fields:
        if name not in recorded:
            continue
        try:
            value = float(np.asarray(recorded[name][0]).ravel()[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(value):
            out[name] = value
    return out


def collect_uniques(values: np.ndarray, atol: float) -> np.ndarray:
    """The distinct settings in *values*, counting any two within *atol* as one.

    ``process_data.collect_uniques`` verbatim. A readback never repeats
    exactly, so the settings have to be found by distance; the first value of
    each cluster is the one kept, which is why the result follows the sorted
    order rather than the order measured.
    """
    unique = np.unique(values)
    kept = [unique[0]]
    for value in unique[1:]:
        if not np.any(np.isclose(value, kept, atol=atol)):
            kept.append(value)
    return np.array(kept)


def scan_signature(path: Path) -> tuple | None:
    """What a scan has to share with another before it can be a repeat of it.

    The plan and the motors it stepped, taken from the metadata rather than
    the columns: two scans that moved different things are not repeats of one
    another however alike their numbers look. ``None`` where the scan stepped
    nothing -- a ``count`` has no shape to repeat, and the 32 of them in one
    directory here would otherwise read as a single measurement taken 32
    times, their row counts running from 1 to 600.
    """
    info = read_scan_info(path)
    motors = tuple(name for name, *_ in info["ranges"])
    return (info["plan"], motors) if motors else None


def sample_atol(name: str) -> float:
    """How far *name* may move with the sample still counting as unmoved."""
    for stage in SAMPLE_STAGES:
        if name in stage:
            return stage[name]
    return TEMPERATURE_ATOL


def scan_groups(paths: list[Path]) -> list[list[Path]]:
    """Gather the selection into the sets that are one measurement repeated.

    Three things have to agree. The plan and the motors, from
    :func:`scan_signature`. Where the sample sat and how cold it was, read
    from the baseline through :data:`SAMPLE_STAGES` -- which is what tells
    repeats from a series, 174, 175 and 176 being the same energy scan with
    the sample stepped 0.2 mm between them. And the points themselves, which
    only the curves can answer and :func:`same_grid` settles later.

    Each quantity is clustered across the selection with
    :func:`collect_uniques` at its own tolerance and every scan then assigned
    to the nearest surviving value, so the grouping does not depend on which
    scan happened to be looked at first. Groups come back in the order their
    first scan appears, which is the table's own order.

    A scan whose baseline carries no whole stage is left on its own rather
    than pooled on the temperature alone. That is the conservative way round:
    an endstation nobody has listed here draws its scans separately, which is
    what the viewer does today, instead of averaging a series into one curve
    that looks clean and is not one measurement.
    """
    readings: dict[str, dict[str, float]] = {}
    for path in paths:
        for stage in SAMPLE_STAGES:
            reading = baseline_values(path, tuple(stage))
            if len(reading) == len(stage):
                reading.update(baseline_values(path, (TEMPERATURE_FIELD,)))
                readings[str(path)] = reading
                break
    settings: dict[str, np.ndarray] = {}
    for name in {name for reading in readings.values() for name in reading}:
        measured = [
            reading[name] for reading in readings.values() if name in reading
        ]
        settings[name] = collect_uniques(np.array(measured), sample_atol(name))

    groups: dict[object, list[Path]] = {}
    for path in paths:
        signature = scan_signature(path)
        reading = readings.get(str(path))
        if signature is None or reading is None:
            # Nothing to repeat, or nothing saying where it was measured.
            key: object = ("alone", str(path))
        else:
            key = (
                signature,
                tuple(
                    (name, int(np.argmin(np.abs(settings[name] - reading[name]))))
                    for name in sorted(reading)
                ),
            )
        groups.setdefault(key, []).append(path)
    return list(groups.values())


def curve_step(x: np.ndarray) -> float:
    """The typical distance between one point of *x* and the next.

    The median rather than the span over the count: a list scan need not
    step evenly, and one long jump back to the start would make every real
    step look small beside it. Zero where nothing moved.
    """
    gaps = np.abs(np.diff(np.asarray(x, dtype=float)))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    return float(np.median(gaps)) if gaps.size else 0.0


def same_grid(x_a: np.ndarray, x_b: np.ndarray) -> bool:
    """Whether two scans measured the same points, to within half a step.

    Compared row by row over the overlap, so a scan stopped part way is
    still the same grid as far as it got. The tolerance is a fraction of the
    scan's own step rather than an absolute distance, there being no one
    figure that suits an energy in keV and an angle in degrees at once; see
    :data:`SAME_X_STEPS` for where the half comes from.
    """
    rows = min(np.size(x_a), np.size(x_b))
    if rows < 2:
        return False
    first = np.asarray(x_a, dtype=float)[:rows]
    second = np.asarray(x_b, dtype=float)[:rows]
    shared = np.isfinite(first) & np.isfinite(second)
    step = max(curve_step(first), curve_step(second))
    if not shared.any() or not step:
        return False
    gap = float(np.max(np.abs(first[shared] - second[shared])))
    return gap <= SAME_X_STEPS * step


def stack_mean(pieces: list[np.ndarray], rows: int) -> np.ndarray:
    """The mean of *pieces* row by row over *rows*, counting only the finite.

    A measurement stopped part way leaves its last rows short, so a row only
    some of the pieces reached is the mean of those and a row none of them
    reached comes back NaN -- never a short piece quietly read as zero.
    Shared by the repeats within one scan and the repeats across several,
    which are the same reduction asked for twice.
    """
    stack = np.full((len(pieces), rows), np.nan)
    for row, piece in enumerate(pieces):
        taken = np.asarray(piece, dtype=float)
        stack[row, : taken.size] = taken
    counted = np.count_nonzero(np.isfinite(stack), axis=0)
    return np.where(
        counted > 0, np.nansum(stack, axis=0) / np.maximum(counted, 1), np.nan
    )


def combine_curves(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    subtract: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """``A + B``, or ``A - B``, with B read onto A's X by interpolation.

    Two scans of one thing do not step the same points. Scans 171 and 173
    are 109 energies each and differ by up to 4e-05 keV -- the readback,
    sampled at two different moments -- so adding them point by point would
    be adding readings taken at different energies.

    Only across the overlap. Left to itself :func:`numpy.interp` holds B's
    first and last readings flat right across A, drawing a sum out where
    there is no second scan at all; ``left`` and ``right`` make that stretch
    NaN and the line simply stops. On 171 and 173 that is the first point
    and the last, 171 starting above 173 and ending below it.

    The sort is what makes this correct rather than merely tidy.
    :func:`numpy.interp` documents ``xp`` as increasing and says nothing
    whatever when it is not -- and both of those scans run *downwards* in
    energy, as does any scan stepped from the top of an edge. Handed 173 as
    it comes off disk it returns 0 finite points out of 109, so the sum
    would be a blank curve on exactly the pair this was written for; without
    the NaN guards it is instead wrong by 64452 on values around 4e5, in
    silence. An averaged curve's X can also carry the NaN pad
    :func:`group_curve` leaves, which the same clean removes.
    """
    # The reader does not truncate X against Y, so neither pair may be taken
    # at face value; the shortest wins, as reduce_dichro does.
    n_a, n_b = min(x_a.size, y_a.size), min(x_b.size, y_b.size)
    x_a = np.asarray(x_a[:n_a], dtype=float)
    y_a = np.asarray(y_a[:n_a], dtype=float)
    x_b = np.asarray(x_b[:n_b], dtype=float)
    y_b = np.asarray(y_b[:n_b], dtype=float)
    good = np.isfinite(x_b) & np.isfinite(y_b)
    order = np.argsort(x_b[good], kind="stable")
    x_b, y_b = x_b[good][order], y_b[good][order]
    if x_b.size == 0:
        return x_a, np.full(x_a.shape, np.nan)
    onto = np.interp(x_a, x_b, y_b, left=np.nan, right=np.nan)
    return x_a, (y_a - onto if subtract else y_a + onto)


def combine_name(first: str, second: str, subtract: bool) -> str:
    """How a combination reads, in the menus and on the plot alike."""
    return f"{first} {'-' if subtract else '+'} {second}"
