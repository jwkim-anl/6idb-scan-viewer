# `da6idb` — an analysis package for the 6-ID-B data files

The five windows in this repository already read and process everything the
beamline writes, but all of it is reachable only through Qt. This is the plan
for lifting the data layer out into a package that a Jupyter notebook can
import, with the windows then built on top of it rather than beside it.

## Phase 0 — decisions

| | |
|---|---|
| Package name | **`da6idb`** |
| Tabular data | **pandas** DataFrames |
| Gridded maps | **xarray**, with real H/K/L coordinates |
| Notebook needs Qt | **no** — the package must import without PyQt5, pyqtgraph or pyvistaqt |
| Deployment | local, JupyterLab |

Still open, and not blocking Phases 1–2:

- one repository or two, once the package exists
- whether the windows are refactored onto the package (strongly preferred: two
  copies of this logic would drift) or left as they are

## Where the code stands

Measured, not estimated:

| | |
|---|---|
| Qt-free functions | 2,066 lines, 64 public |
| GUI classes | 6,527 lines |

So the separation already exists *by function*; what is missing is the module
boundary. Three things block a notebook today:

1. **Every module imports Qt at module level.** `import hkl_grid` costs 2.8 s
   and pulls in PyQt5, pyqtgraph, matplotlib and xrayutilities to reach one
   pure function.
2. **The data layer depends upward on the GUI.** `hkl_grid` imports
   `HklGeometry`, `read_geometry` and `frame_hkl` from `hkl_viewer`, a
   PyQt5/pyvistaqt module, which lazily imports `hkl_grid` back.
3. **No tests and no packaging** — addressed by Phase 1 and Phase 5.

## Sketch

```
da6idb/
  files.py       find and pair master <-> image files
  scan.py        master: metadata, primary/baseline, rasters, curve statistics
  images.py      image stack, per-frame channels, stored ROIs
  rois.py        ROI metrics, background subtraction, normalisation
  hkl/
    geometry.py  HklGeometry, read_geometry, hkl_setup.json, transmission
    convert.py   point cloud
    grid.py      .vti gridding, automatic resolution, caching
    slice.py     plane sampling, slab averaging, line cuts
  batch.py       scan-number expansion, batch gridding
  export.py      ascii / csv / npz / hdf5

gui/             the five windows, importing da6idb and holding no logic
tests/
pyproject.toml   extras: [hkl] xrayutilities+pyvista, [gui] pyqt+pyqtgraph, [dev] pytest
```

The notebook surface is a thin object layer over those functions, so a session
does not begin by calling sixty-four of them by hand:

```python
from da6idb import open_folder

folder = open_folder("/data/beamtime")
folder.summary()                              # DataFrame: scan, plan, motor, range, points
scan = folder[152]
scan.primary()                                # DataFrame
scan.hkl_position()                           # psic h/k/l, mean and spread

images = scan.images()                        # the detector file, found as the GUI finds it
images.rois                                   # image file's own, or the master's baseline
images.roi_curve(0, "sum", normalise="Ion_Chamber_1")

hkl = scan.hkl()                              # master + image + hkl_setup.json
hkl.cloud(threshold=100)
grid = hkl.grid(hkl_range=..., oversample=1)  # xarray, cached as the same .vti the GUI uses
grid.slice(normal=(1, 1, 0), offset=0.01, average=5)
```

## Phases

| | | status |
|---|---|---|
| 0 | decisions | **done** |
| 1 | characterisation tests, before anything moves | **done** |
| 2 | lift the data layer out unchanged; cut the `hkl_grid -> hkl_viewer` cycle | **done** |
| 3 | the notebook API | **done** |
| 4 | the windows import the package, one at a time | **done** |
| 5 | packaging, docs, example notebooks | **done** |

Phase 2 is a move, not a rewrite: the tests from Phase 1 have to keep passing
unchanged throughout, and any that do not are either a bug being introduced or
a deliberate change that should be argued for.

## Phase 1 — what the tests pin

45 tests, about 7 seconds, in `tests/`. They need neither pytest nor xarray:
plain `test_*` functions that pytest will collect later, and `tests/run.py`
runs them today. `DA6IDB_RECORD=1` rewrites the recorded values, which is the
only way they should ever change.

They are about **values**, not about whether a call raises. A refactor that
moves a function unchanged has to reproduce the recorded numbers exactly.

| file | pins |
|---|---|
| `test_files.py` | scan numbering, master <-> image pairing by convention, by declared path, by name beside the master; a same-named file holding no images is refused; ROIs out of the baseline |
| `test_scan.py` | stream fields and the EPOCH/INDEX pseudo-fields, X-field choice for a motor scan and a count, scan metadata, curve statistics including degenerate input, cell edges, raster detection and statistics |
| `test_images.py` | stack and channels, stored ROIs from either source, scan channels matched by timestamp *and* by position, a mismatched master refused, index arithmetic at the edges |
| `test_hkl.py` | geometry and its sources, **centre-pixel H/K/L against the diffractometer's own psic_h/k/l**, frame corners, transmission, setup-file overrides and the fingerprint, cloud reduction |
| `test_grid.py` | sampling measurements and cell size, the lattice, **VTK point ordering**, the cache, cancellation leaving nothing, plane sampling and slab averaging |
| `test_batch.py` | scan-number expansion and parsing, discovery, one lattice across a batch |

Only `tise2_2_00152` ships with the repository, so the rest is built from
small deterministic fixtures shaped like the real files -- which also covers
the awkward cases the example pair cannot: no frame timestamp, a count plan, a
two-motor raster, a master naming its detector file elsewhere, ROIs in the
baseline.

Two checks are deliberately not golden comparisons, because they assert
something physical rather than a number: the centre-pixel H/K/L must agree
with the diffractometer to better than 5e-3, and a thicker slab must not be
noisier than a thin one.

### Does the suite actually bite?

Multiplying the computed cell size by 1.05 -- the kind of thing a tidy-up
might do -- fails three tests across two modules, including the slice shape.
A suite that passes whatever happens would be worse than none.

## Phase 2 — what moved, and what it cost

Every function moved **verbatim**, by line range, rather than being retyped:
the 45 tests from phase 1 were run after each module and had to keep passing
untouched, which is the only evidence that a move changed nothing.

The old modules now re-export what they used to define, so the windows, the
tests and anything already importing them carry on unchanged. Nothing had to
be edited to follow the move.

```
da6idb/files.py        333 lines   paths, pairing, ROI baseline
da6idb/scan.py         477 lines   master metadata, streams, rasters, statistics
da6idb/images.py                   stack, channels, stored ROIs
da6idb/hkl/geometry.py             instrument description, setup file, transmission
da6idb/hkl/convert.py              point cloud
da6idb/hkl/grid.py                 .vti gridding, resolution, cache
da6idb/hkl/slice.py                plane sampling, slab averaging
da6idb/batch.py                    scan ranges, batch gridding
```

The import graph is now a DAG with no edge back into a window module:

```
files <- images <- hkl.geometry <- hkl.convert <- hkl.grid <- hkl.slice
                                                        ^- batch
```

Checked by making `PyQt5`, `pyqtgraph` and `pyvistaqt` unimportable and loading
the whole package anyway: 0.85 s, and it still reads geometry, frames, streams
and converts. `import hkl_grid` went from 2.8 s dragging in four GUI libraries
to 0.8 s dragging in none -- matplotlib still arrives, but through
`xrayutilities`, not through us.

`make_drawable` moved out of `volume_style` into `da6idb.hkl.grid`: it prepares
a grid for drawing rather than styling a window, and leaving it behind would
have made the package depend on a Qt module for it.

## Phase 3 — the notebook layer

`da6idb.session` holds four objects, each a thin wrapper that computes nothing
the modules beneath it do not already compute:

```python
from da6idb import open_folder

folder = open_folder("/data/beamtime")   # ScanSet
folder.summary()                         # DataFrame: plan, motors, range, points, image
scan = folder[152]                       # Scan

scan.curve("psic_eta", "lambda250k_stats1_total", normalise="Ion_Chamber_1")
scan.statistics("lambda250k_stats1_total")   # max, min, centre, FWHM
scan.hkl_position()                          # psic H/K/L, mean and spread
scan.map("lambda250k_stats1_total")          # a two-motor raster, labelled

images = scan.images()
images.rois                                  # the detector's own rectangles
images.roi_curve(0, "sum", normalise="scan/Ion_Chamber_1")

m = scan.hkl()
m.resolution()                               # what the sampling supports
m.cloud(threshold=5000)                      # H, K, L, intensity
cube = m.grid(oversample=1)                  # xarray, dims (H, K, L)
cube.sel(L=7.52, method="nearest").plot()    # and it plots itself
m.slice(normal=(1, 1, 0), offset=0.01, layers=5)
```

With no `x` given, `curve` takes the scan's own motor and falls back to the
point index for a plan that moved nothing -- the same choice the browser makes,
so a notebook and a window agree about what a count plan is plotted against.

The gridded map is an `xarray.DataArray` with real H/K/L coordinates, which is
what makes `sel(L=7.52, method="nearest")` and a bare `.plot()` work. It is
backed by the same cached `.vti` the viewers use, so a grid made in a notebook
opens in the viewer and the other way round. No plotting helpers were needed in
the end: xarray and pandas already draw themselves.

`import da6idb` imports nothing until something is asked for, so the reading
modules stay usable without pandas or xarray.

### Checked

53 tests now, under both `pytest` and `tests/run.py`. One of them starts a
subprocess with `PyQt5`, `pyqtgraph` and `pyvistaqt` made unimportable and
imports the whole package, so the claim that a notebook needs no Qt is tested
rather than asserted.

## Phase 4 — the package becomes the only copy

Three jobs, of which only the first was interesting.

**The ROI arithmetic moved** into `da6idb/rois.py`, against a plain
`Rect(x, y, w, h, angle)` rather than a pyqtgraph object. It had been left in
`image_viewer.py` during phase 2, and the cost of that showed up immediately:
`Images.roi_curve`, written for the notebook in phase 3, had to reimplement
it, and did so without rotation, without background subtraction and without
centre of mass. Both now call the same function. The notebook gained all
three; the viewer gained nothing and lost nothing.

Except that it did lose something, and this is worth recording. The viewer's
rotated ROIs went through pyqtgraph's `getArrayRegion`, whose sample count is
the ceiling of a size its own arithmetic computes -- and that size lands a few
parts in 1e16 above the integer at some angles and exactly on it at others. An
8x6 rectangle was therefore sampled 8x6 at 0, 30, 60 and 90 degrees and 9x7 at
15 and 45, over the same pixels, its sum jumping about 30% with nothing
changing but the angle. On a sweep of sizes and angles it happened on a third
of the edges tested, inflating the sampled area by 14 to 56%. The package
takes the count from the geometry, so it is now 8x6 at every angle. Rotated
ROI numbers have changed accordingly; unrotated ones are bit-for-bit what they
were.

**The export writers moved** into `da6idb/export.py`. They were methods on a
window, so a notebook could not write a slice to a file at all.
`export.write_dataarray` takes an xarray slice straight from `HklMap.slice`.

**The forwarding imports are gone.** The windows had been importing 112 names
from the package, 75 of them only to pass on to each other -- `image_viewer`
imported 33 and used 2. Each window now imports exactly what it uses, and the
tests import from the package, since that is what they are testing. One
window-to-window import remains and belongs: the scan browser opens an image
viewer.

The goldens did not move, which is the point: the numbers are the same, the
code is in one place.

## Phase 5 — packaging and documentation

`pyproject.toml` installs `da6idb` with the dependencies split by what they are
for: numpy, h5py, scipy at the core, `[notebook]` for pandas and xarray,
`[hkl]` for xrayutilities and pyvista, `[gui]` for the windows, `[dev]` for
pytest. A notebook that only wants the scalar channels does not have to
install a renderer. The windows get console entry points
(`da6idb-scans`, `da6idb-hkl`, and so on) and `da6idb-grid` for the command
line gridder.

`docs/api.md` lists all 96 public callables and 9 classes with their
signatures and one-line summaries, generated by `docs/make_api.py` from the
source so it cannot drift. Writing it found ten functions with no docstring,
which now have one.

`examples/getting-started.ipynb` goes from a folder to a curve, an ROI, a
gridded map and a slice written to disk, against the scan that ships with the
repository. Every code cell is executed as part of checking it in -- a
notebook that does not run is worse than none.

Running it found a real flaw in the phase 3 API: `HklMap.slice` ignored the
grid `HklMap.grid` had just built and silently made a second one at full
resolution, so the example produced a 94 MB file nobody asked for alongside
the 2 MB one it meant to. `slice` now cuts the lattice `grid` last used unless
told otherwise.
