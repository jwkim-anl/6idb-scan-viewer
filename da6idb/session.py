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

from da6idb import scan as _scan
from da6idb.files import list_hdf_files, resolve_image_file, scan_number
from da6idb.hkl import convert as _convert
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

    def fields(self, stream: str = "primary") -> list[str]:
        return _scan.stream_fields(self.path, stream)

    # -- the numbers ------------------------------------------------------

    def primary(self, fields=None) -> pd.DataFrame:
        """The primary stream, one column per field, one row per scan point."""
        wanted = list(fields) if fields is not None else self.fields("primary")
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
        fields = self.fields("primary")
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
        image = resolve_image_file(self.path)
        if image is None:
            raise FileNotFoundError(f"no detector file found for {self.path.name}")
        return Images(image, master=self.path)

    def hkl(self, **kwargs) -> "HklMap":
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

    def roi_curve(self, roi=0, metric="sum", normalise=None) -> pd.Series:
        """One ROI's metric against frame number.

        *roi* is an index into :attr:`rois` or an explicit
        ``(x, y, width, height)``.
        """
        if isinstance(roi, (int, np.integer)):
            spec = self.data.stored_rois[int(roi)]
            x, y, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
            label = spec["name"]
        else:
            x, y, w, h = roi
            label = f"roi({x:g},{y:g},{w:g},{h:g})"
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = x0 + int(round(w)), y0 + int(round(h))
        box = self.data.stack[:, x0:x1, y0:y1]
        if box.size == 0:
            raise ValueError(f"{label} does not overlap the detector")
        flat = box.reshape(box.shape[0], -1)
        values = {
            "sum": flat.sum(axis=1), "mean": flat.mean(axis=1),
            "min": flat.min(axis=1), "max": flat.max(axis=1),
        }
        if metric not in values:
            raise ValueError(f"metric must be one of {sorted(values)}")
        series = values[metric].astype(float)
        if normalise:
            monitor = self.data.channels.get(normalise)
            if monitor is None:
                raise KeyError(f"{normalise!r} is not a per-frame channel; "
                               "see .channels().columns")
            with np.errstate(divide="ignore", invalid="ignore"):
                series = np.where(np.asarray(monitor) != 0,
                                  series / np.asarray(monitor), np.nan)
            label = f"{label}.{metric}/{normalise}"
        else:
            label = f"{label}.{metric}"
        return pd.Series(series, name=label,
                         index=pd.RangeIndex(self.n_frames, name="frame"))

    def __repr__(self):
        return (f"<Images {self.path.name} — {self.n_frames} x "
                f"{self.frame_shape[0]}x{self.frame_shape[1]}>")


class HklMap:
    """One scan in reciprocal space: the geometry, the cloud, the grid."""

    def __init__(self, image, master=None, **kwargs):
        self.path = Path(image)
        self.master = Path(master) if master is not None else None
        self.geometry = read_geometry(self.path, self.master, **kwargs)

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
        low, high = _grid.measured_range(self.geometry)
        return pd.DataFrame({"low": low, "high": high}, index=list("HKL"))

    def grid(self, hkl_range=None, oversample=1.0, output=None, reuse=True, **kwargs):
        """Bin onto a regular lattice and hand back a labelled cube.

        The same ``.vti`` the viewers cache, so a grid made here opens there
        and the other way round.
        """
        cube, path, _made = _grid.ensure_grid(
            self.path, self.master, hkl_range=hkl_range, oversample=oversample,
            output=output, reuse=reuse, geometry=self.geometry, **kwargs)
        array = _grid.to_xarray(cube)
        array.attrs["path"] = str(path)
        return array

    def slice(self, normal=(0, 0, 1), offset=0.0, layers=1, **kwargs):
        """A plane through the grid, as a labelled 2D array."""
        import xarray as xr

        cube, _path, _made = _grid.ensure_grid(
            self.path, self.master, geometry=self.geometry, **kwargs)
        _grid.make_drawable(cube)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        origin = np.asarray(cube.center, dtype=float) + normal * offset
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
                                   "layers": int(layers)})

    def __repr__(self):
        return (f"<HklMap {self.path.name} — {self.geometry.n_frames} frames, "
                f"angles from {self.geometry.sources.get('angles')}>")
