"""The objects a notebook starts from.

A thin layer over the reading functions: nothing here computes anything the
rest of the package does not already compute, it only gives the results names
and shapes a session can work with -- pandas for tables, xarray for the
gridded maps, plain numpy where that is what the data is.

    from da6idb import open_folder

    folder = open_folder("/data/beamtime")
    folder.summary()
    scan = folder[152]
    scan.curve("psic_eta", "lambda250k_stats1_total")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from da6idb import rois as _rois
from da6idb import scan as _scan
from da6idb.rois import Rect
from da6idb.files import list_hdf_files, resolve_image_file, scan_number
from da6idb.hkl import convert as _convert
from da6idb.hkl import detector as _detector
from da6idb.hkl import grid as _grid
from da6idb.hkl import slice as _slice
from da6idb.hkl.geometry import read_geometry
from da6idb.images import ImageData

__all__ = ["Scan", "ScanSet", "Images", "HklMap", "open_folder", "open_scan"]


def open_folder(directory) -> "ScanSet":
    """Every scan in *directory*, ready to be listed or picked from."""
    return ScanSet(directory)


def open_scan(master) -> "Scan":
    """One scan, by the path of its master file."""
    return Scan(master)


class ScanSet:
    """The scans in one directory, indexed by scan number."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._scans = {}
        for path in list_hdf_files(self.directory):
            if not _scan.read_scan_info(path).get("is_scan"):
                continue
            try:
                number = int(scan_number(path))
            except ValueError:
                continue
            self._scans[number] = path

    @property
    def numbers(self) -> list[int]:
        return sorted(self._scans)

    def __len__(self):
        return len(self._scans)

    def __iter__(self):
        for number in self.numbers:
            yield self[number]

    def __getitem__(self, number) -> "Scan":
        if number not in self._scans:
            raise KeyError(
                f"no scan {number} in {self.directory} "
                f"(have {self.numbers[:8]}{'…' if len(self._scans) > 8 else ''})"
            )
        return Scan(self._scans[number])

    def summary(self) -> pd.DataFrame:
        """One row per scan: what it was, what moved, how far, how many points."""
        rows = []
        for number in self.numbers:
            path = self._scans[number]
            info = _scan.read_scan_info(path)
            rows.append({
                "scan": number,
                "plan": info.get("plan"),
                "motors": ", ".join(info.get("motors") or []),
                "range": "; ".join(
                    f"{name} {start:g}→{stop:g}" if n is None
                    else f"{name} {start:g}→{stop:g} ({n})"
                    for name, start, stop, n in (info.get("ranges") or [])
                ),
                "points": info.get("points"),
                "count_time": info.get("count_time"),
                "image": resolve_image_file(path) is not None,
                "file": path.name,
            })
        return pd.DataFrame(rows).set_index("scan")

    def __repr__(self):
        return f"<ScanSet {self.directory} — {len(self)} scan(s)>"


class Scan:
    """One scan: its master file, and everything reachable from it."""

    def __init__(self, master):
        self.path = Path(master)
        self._info = _scan.read_scan_info(self.path)

    # -- what it was ------------------------------------------------------

    @property
    def number(self) -> str:
        return scan_number(self.path)

    @property
    def plan(self):
        return self._info.get("plan")

    @property
    def motors(self) -> list[str]:
        return list(self._info.get("motors") or [])

    @property
    def points(self):
        return self._info.get("points")

    @property
    def info(self) -> dict:
        return dict(self._info)

    def fields(self, stream: str = "primary") -> pd.DataFrame:
        """Every field you can plot, with what it is and the range it covered.

        Indexed by field name, so ``.fields().index`` is the plain list.
        *role* says whether a field is the motor the scan moved, a detector
        channel or a positioner read along the way; *hinted* marks the
        detector the plan nominated; *varies* with *min* and *max* shows at a
        glance which fields are worth an axis and which sat still -- a
        near-constant positive channel like an ion chamber is your monitor.
        The scanned motor sorts first, the fields that never moved last.
        """
        rows = _scan.describe_fields(self.path, stream)
        frame = pd.DataFrame(
            rows,
            columns=["field", "role", "hinted", "units", "varies",
                     "min", "max", "source"],
        )
        return frame.set_index("field")

    def field_names(self, stream: str = "primary") -> list[str]:
        """Just the names, in file order, including EPOCH and INDEX."""
        return _scan.stream_fields(self.path, stream)

    # -- the numbers ------------------------------------------------------

    def primary(self, fields=None) -> pd.DataFrame:
        """The primary stream, one column per field, one row per scan point."""
        wanted = list(fields) if fields is not None else self.field_names("primary")
        data = _scan.read_primary(self.path, wanted)
        frame = pd.DataFrame({k: v for k, v in data.items() if v.ndim == 1})
        frame.index.name = "point"
        return frame

    def baseline(self) -> pd.DataFrame:
        """What the baseline recorded before and after the scan."""
        rows = [{"field": k, "before": v[0], "after": v[1]}
                for k, v in _scan.read_baseline(self.path).items()]
        return pd.DataFrame(rows).set_index("field").sort_index()

    def curve(self, x=None, y=None, normalise=None) -> pd.DataFrame:
        """One x column and one or more y columns, normalised if asked.

        With no *x* the scan's own motor is used, falling back to the point
        index for a plan that moved nothing -- the same choice the browser
        makes.
        """
        fields = self.field_names("primary")
        if x is None:
            x = _scan.scan_x_field(self.path, fields) or _scan.default_x_field(fields)
        wanted = [y] if isinstance(y, str) else list(y or [])
        if not wanted:
            raise ValueError("choose at least one y field; see .fields()")
        needed = {x, *wanted} | ({normalise} if normalise else set())
        data = _scan.read_primary(self.path, list(needed))
        if x not in data:
            raise KeyError(f"{x!r} is not in the primary stream; see .fields()")
        frame = pd.DataFrame({x: data[x]})
        monitor = data.get(normalise) if normalise else None
        for name in wanted:
            if name not in data:
                continue
            values = data[name]
            if monitor is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.where(monitor != 0, values / monitor, np.nan)
            frame[name if monitor is None else f"{name}/{normalise}"] = values
        return frame.set_index(x)

    def statistics(self, y, x=None, normalise=None) -> pd.DataFrame:
        """Peak statistics for each y: maximum, minimum, centre and width."""
        curve = self.curve(x=x, y=y, normalise=normalise)
        rows = {}
        for column in curve.columns:
            rows[column] = _scan.curve_stats(
                curve.index.to_numpy(dtype=float), curve[column].to_numpy(dtype=float)
            )
        return pd.DataFrame(rows).T

    def hkl_position(self) -> pd.DataFrame | None:
        """Where the diffractometer was, averaged over the scan."""
        values = _scan.scan_hkl(self.path)
        if values is None:
            return None
        return pd.DataFrame(
            [{"axis": a, "mean": m, "spread": s}
             for a, (m, s) in zip("HKL", values)]
        ).set_index("axis")

    def map(self, value, normalise=""):
        """A two-motor raster as a labelled 2D array, or None if it is not one."""
        import xarray as xr

        raster = _scan.read_grid(self.path, value, normalise)
        if raster is None:
            return None
        return xr.DataArray(
            raster["z"],
            dims=(raster["y_name"], raster["x_name"]),
            coords={raster["y_name"]: raster["y"], raster["x_name"]: raster["x"]},
            name=value,
        )

    # -- what hangs off it ------------------------------------------------

    def images(self) -> "Images":
        """The detector frames for this scan, found as the viewers find them."""
        image = resolve_image_file(self.path)
        if image is None:
            raise FileNotFoundError(f"no detector file found for {self.path.name}")
        return Images(image, master=self.path)

    def hkl(self, **kwargs) -> "HklMap":
        """This scan in reciprocal space, geometry and all."""
        image = resolve_image_file(self.path)
        if image is None:
            raise FileNotFoundError(f"no detector file found for {self.path.name}")
        return HklMap(image, master=self.path, **kwargs)

    def __repr__(self):
        return (f"<Scan {self.number} {self.plan!r} "
                f"{', '.join(self.motors) or 'no motor'} — {self.points} points>")


class Images:
    """The detector frames for one scan, and what was recorded with them."""

    def __init__(self, path, master=None):
        self.data = ImageData(Path(path), master=master)
        self.path = self.data.path
        self._geometry = None
        self._converter = None

    @property
    def stack(self) -> np.ndarray:
        return self.data.stack

    @property
    def n_frames(self) -> int:
        return self.data.n_frames

    @property
    def frame_shape(self):
        return self.data.frame_shape

    def frame(self, index: int) -> np.ndarray:
        """One frame, indexed ``[x, y]`` as the file stores it."""
        return self.data.stack[index]

    def channels(self) -> pd.DataFrame:
        """Everything recorded per frame, including the master's own channels."""
        frame = pd.DataFrame({k: v for k, v in self.data.channels.items()
                              if np.ndim(v) == 1 and len(v) == self.n_frames})
        frame.index.name = "frame"
        return frame.reindex(sorted(frame.columns), axis=1)

    @property
    def rois(self) -> pd.DataFrame:
        """The detector's own rectangles, wherever they were recorded."""
        return pd.DataFrame(self.data.stored_rois)

    def rect(self, roi=0) -> Rect:
        """One ROI as a rectangle.

        Takes an index into :attr:`rois`, one of those rows as a dict, a
        :class:`~da6idb.rois.Rect`, or a plain ``(x, y, w, h)`` --
        ``(x, y, w, h, angle)`` to turn it about its corner. Everything that
        reduces an ROI goes through here, so a rectangle you make up is worth
        exactly as much as one the detector recorded.
        """
        if isinstance(roi, Rect):
            return roi
        if isinstance(roi, dict):
            return Rect.from_stored(roi)
        if isinstance(roi, (int, np.integer)):
            stored = self.data.stored_rois
            if not -len(stored) <= int(roi) < len(stored):
                raise IndexError(
                    f"{self.path.name} records {len(stored)} ROI(s), so {roi} "
                    "is out of range; pass (x, y, w, h) to make your own"
                )
            return Rect.from_stored(stored[int(roi)])
        values = [float(v) for v in roi]
        if len(values) not in (4, 5):
            raise ValueError(
                "a rectangle is (x, y, w, h) or (x, y, w, h, angle), "
                f"not {len(values)} value(s)"
            )
        return Rect(*values)

    def show(self, frame=0, rois=None, log=False, ax=None, cmap="viridis",
             **kwargs):
        """One frame with its ROIs drawn on it, so you can place a new one.

        Oriented like the image viewer -- x across, y down -- and labelled in
        detector pixels, so a rectangle read off these axes can go straight
        back in as ``(x, y, w, h)``. *rois* defaults to the ones the file
        records; pass a list of your own, or ``()`` for a bare frame.
        Returns the matplotlib axes, to draw on further or save.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        data = np.asarray(self.frame(int(frame)), dtype=float)
        if log:
            from matplotlib.colors import LogNorm
            lit = data[data > 0]
            kwargs.setdefault("norm", LogNorm(
                vmin=float(lit.min()) if lit.size else 1.0,
                vmax=float(data.max()) if lit.size else 1.0))
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.0))
        # .T because the file stores the frame [x, y] and imshow reads
        # [row, column]; origin="upper" to match the viewer's inverted y.
        ax.imshow(data.T, origin="upper", cmap=cmap, **kwargs)

        stored = self.data.stored_rois
        if rois is None:
            specs = list(range(len(stored)))
        elif isinstance(rois, (Rect, dict, int, np.integer)):
            specs = [rois]
        else:
            specs = list(rois)
        # The recorded ones have names; number the rest, so a signal box and
        # a background box do not both come out labelled "new".
        made_up = [i for i, spec in enumerate(specs)
                   if not isinstance(spec, (int, np.integer))]
        for position, spec in enumerate(specs):
            rect = self.rect(spec)
            if isinstance(spec, (int, np.integer)):
                name = stored[int(spec)].get("name") or f"roi{spec}"
            elif len(made_up) == 1:
                name = "new"
            else:
                name = f"new{made_up.index(position) + 1}"
            ax.add_patch(Polygon(rect.corners(), closed=True, fill=False,
                                 edgecolor="red", linewidth=1.2))
            ax.annotate(name, rect.corners()[0], color="red", fontsize=8,
                        xytext=(2, -2), textcoords="offset points",
                        va="top", ha="left")
        ax.set_xlabel("X [pixels]")
        ax.set_ylabel("Y [pixels]")
        ax.set_title(f"{self.path.name} — frame {frame}", fontsize=9)
        return ax

    def roi_series(self, roi=0) -> pd.DataFrame:
        """Every metric for one ROI, frame by frame.

        The same arithmetic the image viewer runs, rotation included, so a
        number read here and a number read there are the same number.
        """
        values = _rois.series(self.data.stack, self.rect(roi))
        frame = pd.DataFrame(values)
        frame.index = pd.RangeIndex(self.n_frames, name="frame")
        return frame

    def roi_curve(self, roi=0, metric="sum", normalise=None,
                  background=None) -> pd.Series:
        """One ROI's metric against frame number.

        With *background* -- another ROI -- its mean per pixel is taken off
        the pixel-value metrics, the sum losing it times the pixel count, as
        the viewer does. The centre of mass is deliberately left on the raw
        pixels: weighting a centroid by background-subtracted values goes
        unstable wherever the corrected sum passes through zero.
        """
        if metric not in _rois.METRICS:
            raise ValueError(f"metric must be one of {list(_rois.METRICS)}")
        rect = self.rect(roi)
        values = _rois.series(self.data.stack, rect)[metric].astype(float)
        label = (self.data.stored_rois[int(roi)]["name"]
                 if isinstance(roi, (int, np.integer)) else "roi")

        if background is not None and metric not in ("comx", "comy"):
            per_pixel = _rois.series(
                self.data.stack, self.rect(background))["mean"].astype(float)
            if metric == "sum":
                values = values - per_pixel * rect.pixel_count(
                    self.frame_shape, frame=self.data.stack[0])
            else:
                values = values - per_pixel
            label += "-bg"

        label = f"{label}.{metric}"
        if normalise:
            monitor = self.data.channels.get(normalise)
            if monitor is None:
                raise KeyError(f"{normalise!r} is not a per-frame channel; "
                               "see .channels().columns")
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(np.asarray(monitor) != 0,
                                  values / np.asarray(monitor), np.nan)
            label += f"/{normalise}"
        return pd.Series(values, name=label,
                         index=pd.RangeIndex(self.n_frames, name="frame"))

    # -- where a feature sat, in angle ------------------------------------

    def geometry(self):
        """The instrument description for this scan, read once and kept."""
        if self._geometry is None:
            self._geometry = read_geometry(self.path,
                                           master=self.data.master_path)
        return self._geometry

    def converter(self):
        """The xrayutilities converter for this geometry, built once."""
        if self._converter is None:
            self._converter = _convert.build_converter(self.geometry())
        return self._converter

    def _rect_or_frame(self, roi) -> Rect:
        """``None`` means the whole detector, which is just a big rectangle."""
        if roi is None:
            return Rect(0.0, 0.0, float(self.frame_shape[0]),
                        float(self.frame_shape[1]))
        return self.rect(roi)

    def _two_theta_pass(self, roi, background):
        """Per-frame mean scattering angle over *roi*, and its weight.

        The angle is averaged *over* the region's pixels, weighted by what
        each one counted -- not taken at the centre of mass. Two theta is
        convex in pixel position, least at the centre pixel and rising away
        from it in every direction, so the angle of the average pixel is
        systematically lower than the average of the pixels' angles. On a
        wide region that bias is real and always the same way.

        *background*, another ROI, has its mean per pixel taken off before
        anything is weighted: left in, it drags each frame's answer towards
        the middle of the box and flattens the weighting between frames
        that hold the feature and frames that hold nothing.
        """
        rect = self._rect_or_frame(roi)
        x, y = rect.sample_points(self.frame_shape)
        maps, which = _detector.two_theta_maps(self.geometry(), x, y,
                                               hxrd=self.converter())
        if background is None:
            level = np.zeros(self.n_frames)
        else:
            level = _rois.series(self.data.stack,
                                 self.rect(background))["mean"].astype(float)

        angles = np.full(self.n_frames, np.nan)
        counted = np.zeros(self.n_frames)
        for frame, sub in enumerate(_rois.sub_frames(self.data.stack, rect)):
            if sub is None:
                continue
            lit = np.asarray(sub, dtype=float) - level[frame]
            np.clip(lit, 0.0, None, out=lit)
            total = float(lit.sum())
            counted[frame] = total
            if total > 0.0:
                angles[frame] = float((lit * maps[which[frame]]).sum() / total)
        return angles, counted

    def two_theta_angles(self, roi=0, background=None) -> pd.Series:
        """The scattering angle of what *roi* held, frame by frame.

        The detector circle reads the angle of the centre pixel, so a
        feature anywhere else is at a different two theta. This asks
        xrayutilities for the angle of every pixel in the region, through
        the same geometry the HKL conversion uses, and averages them
        weighted by the counts in each -- both detector circles included,
        not just delta.

        *background* is another ROI whose mean per pixel is taken off first,
        which matters: background spread evenly over the box pulls the
        answer towards the middle of the box rather than towards the
        feature. A frame left with nothing above background gets NaN.

        *roi* is anything :meth:`rect` takes, or ``None`` for the whole
        frame. Returns one angle in degrees per frame.
        """
        angles, _ = self._two_theta_pass(roi, background)
        return pd.Series(angles, name="2theta [deg]",
                         index=pd.RangeIndex(self.n_frames, name="frame"))

    def two_theta(self, roi=0, background=None, weight="sum") -> float:
        """One two theta for the whole scan: :meth:`two_theta_angles` averaged.

        Weighted by *weight*, an ROI metric, so the frames where the feature
        was lit count for more than the ones where the box held nothing. The
        default weighs by the counts actually used -- background-subtracted,
        when *background* is given, which is what makes the weighting mean
        anything on a region whose background is a large part of its sum.
        Pass ``weight=None`` for a plain mean over the frames.

        The average is taken over the per-frame angles rather than over one
        summed image, so it stays right if the detector moved during the
        scan.
        """
        angles, counted = self._two_theta_pass(roi, background)
        if weight is None:
            return float(np.nanmean(angles))
        if weight == "sum":
            weights = counted
        elif weight in _rois.METRICS:
            weights = _rois.series(
                self.data.stack, self._rect_or_frame(roi))[weight].astype(float)
        else:
            raise ValueError(f"weight must be None or one of {list(_rois.METRICS)}")
        good = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
        if not good.any():
            raise ValueError(f"no frame has a positive {weight} to weight by")
        return float(np.average(angles[good], weights=weights[good]))

    def __repr__(self):
        return (f"<Images {self.path.name} — {self.n_frames} x "
                f"{self.frame_shape[0]}x{self.frame_shape[1]}>")


class HklMap:
    """One scan in reciprocal space: the geometry, the cloud, the grid."""

    def __init__(self, image, master=None, **kwargs):
        self.path = Path(image)
        self.master = Path(master) if master is not None else None
        self.geometry = read_geometry(self.path, self.master, **kwargs)
        # What grid() was last asked for, so slice() cuts the grid you made
        # rather than quietly building a second one at full resolution.
        self._lattice: dict = {}

    def sources(self) -> pd.Series:
        """Where each piece of the geometry came from."""
        return pd.Series(self.geometry.sources, name="source").sort_index()

    def cloud(self, threshold=0.0, max_points=2_000_000, **kwargs) -> pd.DataFrame:
        """The surviving pixels as H, K, L and intensity."""
        found = _convert.convert(self.geometry, threshold=threshold,
                                 max_points=max_points, **kwargs)
        return pd.DataFrame({
            "H": found.points[:, 0], "K": found.points[:, 1],
            "L": found.points[:, 2], "intensity": found.intensity,
        })

    def resolution(self) -> pd.Series:
        """The cell the sampling supports, and what it was measured from."""
        volume, gap, in_plane = _grid.sampling_cell(self.geometry)
        step = _grid.natural_step(self.geometry)
        footprint = _grid.sampling_footprint(self.geometry)
        return pd.Series({
            "volume_per_sample": volume, "frame_gap": gap, "in_plane": in_plane,
            "cell": float(step[0]),
            "footprint_H": footprint[0], "footprint_K": footprint[1],
            "footprint_L": footprint[2],
        })

    def extent(self) -> pd.DataFrame:
        """The H, K, L box the scan actually reaches."""
        low, high = _grid.measured_range(self.geometry)
        return pd.DataFrame({"low": low, "high": high}, index=list("HKL"))

    def grid(self, hkl_range=None, oversample=1.0, output=None, reuse=True, **kwargs):
        """Bin onto a regular lattice and hand back a labelled cube.

        The same ``.vti`` the viewers cache, so a grid made here opens there
        and the other way round.
        """
        self._lattice = dict(hkl_range=hkl_range, oversample=oversample,
                             output=output, **kwargs)
        cube, path, _made = _grid.ensure_grid(
            self.path, self.master, hkl_range=hkl_range, oversample=oversample,
            output=output, reuse=reuse, geometry=self.geometry, **kwargs)
        array = _grid.to_xarray(cube)
        array.attrs["path"] = str(path)
        return array

    def slice(self, normal=(0, 0, 1), offset=0.0, point=None, layers=1,
              **kwargs):
        """A plane through the grid, as a labelled 2D array.

        The plane is placed either by *offset* -- how far along the normal
        from the centre of the map -- or by *point*, an H, K, L the plane
        must pass through, which is usually what you actually know. The
        result records both, and with a *point* the plane's own coordinates
        are measured from it.

        Cuts the lattice :meth:`grid` last built unless told otherwise, so
        asking for a coarse grid and then slicing it does not quietly produce
        a second one at full resolution.
        """
        import xarray as xr

        settings = {**self._lattice, **kwargs}
        cube, _path, _made = _grid.ensure_grid(
            self.path, self.master, geometry=self.geometry, **settings)
        _grid.make_drawable(cube)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        origin, offset = _slice.plane_origin(
            cube.center, normal, offset=offset, point=point)
        image, us, vs, u, v, corner = _slice.sample_plane(
            cube, normal, origin, layers=layers)

        def axis(direction, samples):
            index = _slice._axis_index(direction)
            if index is None:
                return ("along [%.3f %.3f %.3f]" % tuple(direction), samples)
            return ("HKL"[index], corner[index] + samples * float(direction[index]))

        x_name, x_values = axis(u, us)
        y_name, y_values = axis(v, vs)
        return xr.DataArray(image, dims=(y_name, x_name),
                            coords={y_name: y_values, x_name: x_values},
                            name="intensity",
                            attrs={"normal": list(normal), "offset": float(offset),
                                   "point": [float(c) for c in origin],
                                   "layers": int(layers)})

    def __repr__(self):
        return (f"<HklMap {self.path.name} — {self.geometry.n_frames} frames, "
                f"angles from {self.geometry.sources.get('angles')}>")
