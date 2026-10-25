"""What the image viewer does: frames, regions of interest, angles.

A region is wherever a model says it is. Everything that takes a *roi*
here takes an index into the ones the file recorded, **or** a rectangle
``[x, y, w, h]`` (``[x, y, w, h, angle]`` to turn it about its corner) --
:meth:`da6idb.session.Images.rect` makes no distinction between the two,
so a box invented in conversation is worth exactly as much as one the
detector was configured with. That is why there is no tool for creating
one: there is nothing to create.
"""

from typing import Optional, Union

from da6idb_mcp import cache, render, resolve
from da6idb_mcp.render import Reply
from da6idb_mcp.tools import tool

#: An ROI index, or a rectangle in detector pixels.
Roi = Union[int, list]

METRICS = ("sum", "mean", "min", "max", "comx", "comy")


def _images(scan, where):
    return cache.images(resolve.scan_master(scan, where))


def _draw_rects(ax, rects, labels):
    from matplotlib.patches import Polygon

    for rect, label in zip(rects, labels):
        ax.add_patch(Polygon(rect.corners(), closed=True, fill=False,
                             lw=1.2, edgecolor="white"))
        ax.text(rect.x, rect.y - 4, str(label), color="white", fontsize=7)


@tool()
def image_info(scan: Union[int, str], where: Optional[str] = None) -> Reply:
    """The detector stack for a scan: frame count, shape, and recorded ROIs.

    Start here before any other `image_*` tool -- it says how many frames
    there are to index and which ROIs the file already carries.
    """
    master = resolve.scan_master(scan, where)
    images = _images(scan, where)
    return Reply(data={
        "file": str(images.path),
        "master": str(master),
        "frames": images.n_frames,
        "frame_shape": list(images.frame_shape),
        "rois": images.rois,
        "channels": list(images.channels().columns),
    })


@tool()
def image_frame(scan: Union[int, str], frame: int = 0, log: bool = False,
                rois: Optional[list] = None, cmap: str = "viridis",
                where: Optional[str] = None) -> Reply:
    """One detector frame as a picture, with its ROIs drawn on it.

    Axes are detector pixels in the viewer's orientation, so a rectangle
    read off this picture goes straight back into any *roi* argument.
    *log* is usually what you want: diffraction frames have a few bright
    pixels and a very dark floor. *rois* defaults to the recorded ones;
    pass your own rectangles, or ``[]`` for a bare frame.
    """
    import numpy as np

    images = _images(scan, where)
    if not 0 <= int(frame) < images.n_frames:
        raise IndexError(
            f"frame must be 0..{images.n_frames - 1}, not {frame}")

    fig, ax = render.figure((5.8, 5.4))
    images.show(frame=int(frame), rois=(), log=log, ax=ax, cmap=cmap)

    chosen = images.rois.to_dict("records") if rois is None else rois
    rects = [images.rect(r) for r in chosen]
    labels = ([r.get("name", i) for i, r in enumerate(chosen)]
              if rois is None else range(len(rects)))
    _draw_rects(ax, rects, labels)

    data = np.asarray(images.frame(int(frame)), dtype=float)
    return Reply(data={
        "frame": int(frame),
        "shape": list(data.shape),
        "min": float(data.min()), "max": float(data.max()),
        "total": float(data.sum()),
    }, figure=fig)


@tool()
def image_channels(scan: Union[int, str], where: Optional[str] = None) -> Reply:
    """Everything recorded once per frame: monitors, timestamps, counters.

    These are the names a *normalise* argument takes in `image_roi_curve`.
    """
    images = _images(scan, where)
    return Reply(data={"channels": images.channels()})


@tool()
def image_rois(scan: Union[int, str], where: Optional[str] = None) -> Reply:
    """The rectangles the detector itself recorded, by index."""
    images = _images(scan, where)
    frame = images.rois
    return Reply(data={"rois": frame},
                 note="" if len(frame) else
                      "this file records none; pass [x, y, w, h] instead")


@tool()
def image_roi_rect(scan: Union[int, str], roi: Roi = 0,
                   where: Optional[str] = None) -> Reply:
    """What a region actually covers, before you reduce anything with it.

    Resolves an index or a rectangle the same way every other tool does
    and reports its corners, its clipped pixel bounds and how many pixels
    it holds -- which is the N in a ``background x N`` correction. Use it
    to check a box you invented lands on the detector and on the feature.
    """
    images = _images(scan, where)
    rect = images.rect(roi)
    shape = images.frame_shape
    return Reply(data={
        "x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height,
        "angle": rect.angle, "rotated": rect.rotated,
        "corners": rect.corners(),
        "bbox": list(rect.bbox(shape)),
        "pixels": rect.pixel_count(shape, frame=images.frame(0)),
        "frame_shape": list(shape),
    })


@tool()
def image_roi_curve(scan: Union[int, str], roi: Roi = 0, metric: str = "sum",
                    normalise: Optional[str] = None,
                    background: Optional[Roi] = None, log: bool = False,
                    where: Optional[str] = None) -> Reply:
    """One region's metric against frame number, plotted -- a rocking curve.

    *metric* is one of sum, mean, min, max, comx, comy. *background* is a
    second region whose mean per pixel is taken off first; on `sum` that
    is its mean times this region's pixel count, as the viewer does. The
    centres of mass are deliberately left on raw pixels, because a
    centroid weighted by background-subtracted counts goes unstable
    wherever the corrected sum crosses zero.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    images = _images(scan, where)
    series = images.roi_curve(roi=roi, metric=metric, normalise=normalise,
                              background=background)

    fig, ax = render.figure()
    ax.plot(series.index, series.to_numpy(dtype=float), marker=".", lw=1.2)
    ax.set_xlabel("frame")
    ax.set_ylabel(str(series.name))
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_title(f"{images.path.name}", fontsize="medium")
    return Reply(data={"label": str(series.name), "curve": series}, figure=fig)


@tool()
def image_roi_series(scan: Union[int, str], roi: Roi = 0,
                     where: Optional[str] = None) -> Reply:
    """Every metric for one region at once, frame by frame.

    The same arithmetic the image viewer runs, rotation included. Use
    this instead of six `image_roi_curve` calls when you want to see how
    the centre of mass moved alongside the intensity.
    """
    images = _images(scan, where)
    return Reply(data={"series": images.roi_series(roi=roi)})


@tool()
def image_projection(scan: Union[int, str], mode: str = "max",
                     first: int = 0, last: Optional[int] = None,
                     log: bool = True, where: Optional[str] = None) -> Reply:
    """Every frame collapsed into one picture, to find where the signal is.

    *mode* is max, sum or mean over frames `first`..`last` inclusive. The
    maximum is the one to look at first: a feature that moved across the
    detector during the rocking curve shows as a streak, and the box you
    read off it is the box to hand back as a *roi*.
    """
    import numpy as np
    from matplotlib.colors import LogNorm

    images = _images(scan, where)
    stop = images.n_frames - 1 if last is None else int(last)
    first = max(0, int(first))
    stop = min(images.n_frames - 1, stop)
    if stop < first:
        raise ValueError(f"last ({stop}) is before first ({first})")

    block = np.asarray(images.stack[first:stop + 1], dtype=float)
    if mode == "max":
        flat = block.max(axis=0)
    elif mode == "sum":
        flat = block.sum(axis=0)
    elif mode == "mean":
        flat = block.mean(axis=0)
    else:
        raise ValueError("mode must be max, sum or mean")

    norm = None
    if log:
        lit = flat[flat > 0]
        norm = LogNorm(vmin=float(lit.min()) if lit.size else 1.0,
                       vmax=float(flat.max()) if lit.size else 1.0)
    fig, ax = render.figure((5.8, 5.4))
    # .T to match Images.show: the file stores [x, y], imshow reads [row, col].
    ax.imshow(flat.T, origin="upper", cmap="viridis", norm=norm)
    ax.set_xlabel("X [pixels]")
    ax.set_ylabel("Y [pixels]")
    ax.set_title(f"{images.path.name} — {mode} of frames {first}–{stop}",
                 fontsize="medium")

    peak = np.unravel_index(int(flat.argmax()), flat.shape)
    return Reply(data={
        "mode": mode, "first": first, "last": stop,
        "brightest_pixel": {"x": int(peak[0]), "y": int(peak[1]),
                            "value": float(flat[peak])},
        "min": float(flat.min()), "max": float(flat.max()),
    }, figure=fig, note="x and y are detector pixels, ready to use as an roi")


@tool()
def image_two_theta(scan: Union[int, str], roi: Roi = 0,
                    background: Optional[Roi] = None, weight: str = "sum",
                    per_frame: bool = False,
                    where: Optional[str] = None) -> Reply:
    """The scattering angle of whatever a region held, in degrees.

    Not the detector circle's own reading: this asks xrayutilities for the
    angle of every pixel in the region, through the same geometry the HKL
    conversion uses, and averages them weighted by counts. A feature away
    from the centre pixel is at a different two theta, and this is the
    number that says by how much.

    *background* matters here more than elsewhere -- background spread
    evenly over the box pulls the answer towards the middle of the box
    rather than towards the feature. *per_frame* adds the angle frame by
    frame, and a plot of it.
    """
    images = _images(scan, where)
    angle = images.two_theta(roi=roi, background=background,
                             weight=None if weight in ("", "none") else weight)
    data = {"two_theta_deg": angle, "weight": weight}
    fig = None
    if per_frame:
        series = images.two_theta_angles(roi=roi, background=background)
        data["per_frame"] = series
        fig, ax = render.figure()
        ax.plot(series.index, series.to_numpy(dtype=float), marker=".", lw=1.2)
        ax.axhline(angle, color="crimson", lw=1.0, ls="--",
                   label=f"weighted mean {angle:.4f}°")
        ax.set_xlabel("frame")
        ax.set_ylabel("2θ [deg]")
        ax.legend(fontsize="small")
        ax.grid(alpha=0.3)
    return Reply(data=data, figure=fig)


@tool()
def image_geometry(scan: Union[int, str], where: Optional[str] = None) -> Reply:
    """The instrument description behind the angle and HKL conversions.

    Distance, pixel sizes, detector centre, energy and the circle names --
    everything `image_two_theta` and the whole `hkl_*` namespace depend on.
    Check it when a conversion gives an answer that cannot be right.
    """
    images = _images(scan, where)
    geometry = images.geometry()
    data = (geometry if isinstance(geometry, dict)
            else getattr(geometry, "__dict__", geometry))
    return Reply(data={"geometry": data})
