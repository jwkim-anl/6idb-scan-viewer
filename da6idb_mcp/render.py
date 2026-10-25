"""Turning results into something a model can read: tables and pictures.

Two jobs. :func:`to_jsonable` flattens pandas, xarray and numpy into plain
data, truncating anything that would be absurd to send -- a model asking
for a 2048x2048 frame wants to know its shape and range, not four million
numbers. :func:`png` renders a matplotlib figure under a byte budget,
because every picture returned is spent out of a context window.

Nothing here imports ``mcp``. A :class:`Reply` is a plain object; turning
it into MCP content happens in :mod:`da6idb_mcp.tools`, which is the only
place that needs the SDK -- so these can be tested without it.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from da6idb_mcp import config

# Set before pyplot is imported anywhere: these tools render to a buffer and
# must never try to open a window, least of all from a worker thread.
os.environ.setdefault("MPLBACKEND", "agg")

#: Rows of a table sent inline before it is summarised instead.
MAX_ROWS = 200
#: Elements of an array sent in full.
MAX_ELEMENTS = 1000


@dataclass
class Reply:
    """What every tool returns: numbers, optionally a picture, a word.

    *data* is anything :func:`to_jsonable` understands. *figure* is a
    matplotlib figure the decorator will render and close. *note* is the
    one line of prose worth saying -- what was assumed, what was clipped,
    where the file went -- and *artefacts* the paths written, if any.

    *picture* is PNG bytes that are already PNG bytes: a screenshot of a
    live window comes back from the host encoded, having never been a
    matplotlib figure and having no figure to be. Either field puts an
    image in the reply; a tool fills one of them, not both.
    """

    data: object = None
    figure: object = None
    note: str = ""
    artefacts: list = field(default_factory=list)
    picture: bytes = b""

    def payload(self) -> dict:
        out = {}
        if self.data is not None:
            value = to_jsonable(self.data)
            # A dict at the top level reads better flattened than nested
            # under "data", and that is what most tools return.
            out.update(value) if isinstance(value, dict) else out.update(
                {"result": value})
        if self.note:
            out["note"] = self.note
        if self.artefacts:
            out["written"] = [str(p) for p in self.artefacts]
        return out


# -- flattening --------------------------------------------------------------


def _clean_float(value):
    value = float(value)
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _array(value):
    import numpy as np

    array = np.asarray(value)
    if array.size <= MAX_ELEMENTS:
        return [to_jsonable(v) for v in array.tolist()]
    finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": _clean_float(finite.min()) if finite.size else None,
        "max": _clean_float(finite.max()) if finite.size else None,
        "mean": _clean_float(finite.mean()) if finite.size else None,
        "nan": int(np.isnan(array).sum()) if array.dtype.kind == "f" else 0,
        "preview": [to_jsonable(v) for v in array.ravel()[:50].tolist()],
        "truncated": True,
    }


def _frame(value):
    """A DataFrame as records, with its index kept as a column."""
    import numpy as np

    rows = len(value)
    body = value.head(MAX_ROWS)
    index_name = value.index.name or "index"
    out = {
        "columns": [str(c) for c in value.columns],
        "rows": rows,
        "records": [
            {index_name: to_jsonable(key),
             **{str(c): to_jsonable(v) for c, v in record.items()}}
            for key, record in zip(body.index, body.to_dict("records"))
        ],
    }
    if rows > MAX_ROWS:
        out["truncated"] = f"first {MAX_ROWS} of {rows} rows"
        numeric = value.select_dtypes("number")
        if not numeric.empty:
            out["summary"] = {
                str(c): {"min": _clean_float(numeric[c].min()),
                         "max": _clean_float(numeric[c].max()),
                         "mean": _clean_float(numeric[c].mean())}
                for c in numeric.columns
                if np.isfinite(numeric[c].to_numpy(dtype=float)).any()
            }
    return out


def to_jsonable(value):
    """Plain data, whatever came in."""
    import numpy as np

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _clean_float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _array(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    name = type(value).__name__
    if name == "DataFrame":
        return _frame(value)
    if name == "Series":
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if name == "DataArray":
        return {
            "dims": [str(d) for d in value.dims],
            "shape": list(value.shape),
            "coords": {str(k): _array(c.values)
                       for k, c in value.coords.items()},
            "values": _array(value.values),
            "attrs": to_jsonable(dict(value.attrs)),
        }
    if name in {"ImageData", "UniformGrid", "StructuredGrid"}:
        return {
            "dimensions": list(value.dimensions),
            "spacing": [float(s) for s in value.spacing],
            "origin": [float(o) for o in value.origin],
            "bounds": [float(b) for b in value.bounds],
            "arrays": list(value.array_names),
        }

    text = repr(value)
    return text if len(text) <= 4000 else text[:4000] + " …(truncated)"


# -- pictures ----------------------------------------------------------------


def figure(size=None):
    """A bare Agg figure and its axes, styled the same way every time."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=size or config.FIGURE_SIZE, dpi=config.FIGURE_DPI,
                 layout="constrained")
    return fig, fig.add_subplot(111)


def plane(image, xs, ys, x_label="x", y_label="y", title="", log=False,
          cmap="viridis", size=(6.2, 5.2)):
    """A resampled plane as a picture, with the unmeasured cells left blank.

    Both the reciprocal-space slices and the gridded maps arrive as a 2D
    array on a regular lattice with NaN wherever the scan never reached,
    so they are drawn the same way: ``imshow`` with the real HKL extent,
    so a feature's coordinates can be read straight off the axes.
    """
    import numpy as np
    from matplotlib.colors import LogNorm

    values = np.asarray(image, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    lit = values[np.isfinite(values) & (values > 0)]

    norm = None
    if log and lit.size:
        norm = LogNorm(vmin=float(lit.min()), vmax=float(lit.max()))

    fig, ax = figure(size)
    shown = ax.imshow(
        values, origin="lower", aspect="auto", cmap=cmap, norm=norm,
        extent=(float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])),
        interpolation="nearest")
    fig.colorbar(shown, ax=ax, label="intensity")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, fontsize="medium")
    return fig


def png(fig, budget: int | None = None) -> bytes:
    """A figure as PNG bytes, shrunk until it fits the budget.

    Dropping the dpi rather than the figure's size keeps the layout and the
    text proportions the model is reading; it just arrives softer. A frame
    from a 2k detector is the case that needs this.
    """
    budget = config.MAX_PNG_BYTES if budget is None else budget
    dpi = float(config.FIGURE_DPI)
    data = b""
    for _ in range(5):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        data = buffer.getvalue()
        if len(data) <= budget or dpi <= 40:
            return data
        # Bytes go roughly with the pixel count, so with the square root of
        # this ratio; the 0.9 keeps it from overshooting on a noisy image.
        dpi = max(40.0, dpi * 0.9 * math.sqrt(budget / len(data)))
    return data


def close(fig) -> None:
    try:
        fig.clear()
    except Exception:
        pass


def save(fig, path) -> Path:
    """Write a figure at full resolution for the caller to open later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(config.FIGURE_DPI, 150))
    return path
