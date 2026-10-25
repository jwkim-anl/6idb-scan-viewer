# 6-ID-B scan viewers

Three Qt viewers for the HDF5 files written at APS 6-ID-B by a Bluesky plan and
the pvapy streaming consumer:

| | |
|---|---|
| **`scan_viewer.py`** | browse a directory of scans, plot the scalar channels, read the baseline |
| **`image_viewer.py`** | the detector frames, with ROIs and ROI plots |
| **`hkl_viewer.py`** | the frames mapped into reciprocal space, in 3D |
| **`hkl_grid.py`** | bins that map onto a regular HKL grid and saves it as `.vti` |
| **`batch_grid.py`** | grids a run of scans onto one common lattice |
| **`vti_viewer.py`** | opens a saved `.vti` on its own, with no beamline data |
| **`slice_plot.py`** | the slice plane as a flat image, with H/K/L axes |
| **`da6idb/`** | the package underneath all of them, usable from a notebook |

They chain together: the scan browser opens the image viewer for whatever scan you
have selected, and the image viewer opens the HKL map for the file it is showing.
Each also runs on its own from the command line.

`volume_style.py` holds the colours, number format and spin box the two 3D windows
share, so they cannot drift apart.

---

## Installation

```bash
conda env create -f environment.yml
conda activate bluesky-scan-viewer
```

`environment.yml` lists why each package is there. Two are worth calling out:

- **`hdf5plugin` looks optional and is not.** It is imported inside a
  `try/except`, but the pva image stacks are blosc-compressed, so without it the
  frames cannot be decoded and the image viewer opens on an error message instead
  of the data.
- **`pyvista`, `pyvistaqt`, `vtk` and `xrayutilities` are only for the HKL map.**
  They are imported when that window is opened, so the other two viewers work
  without them. The 3D view also needs a real OpenGL context; it will not render
  over a plain SSH session without X11 GLX or VirtualGL, and says so rather than
  crashing.

At the beamline, run with the environment that has these installed, e.g.

```bash
DISPLAY=localhost:10.0 \
  /path/to/6idb-bits/bin/python scan_viewer.py [directory]
```

---

## The files it expects

Each scan is a pair, in the same directory:

```
tise2_2_00152_master.hdf     the Bluesky record: metadata, primary stream, baseline
pva_tise2_2_00152.h5         the detector frames written by the pvapy consumer
```

The image file's name is taken from the master's
`entry/instrument/bluesky/metadata/pva_streaming`, falling back to deriving
`pva_<stem without "_master">.h5`. Only the master's own `file_name` is used — the
`file_path` recorded beside it is the beamline's path and means nothing elsewhere.

**The two files rarely have the same number of points.** The detector free-runs, so
a 101-point scan is typically 101–103 frames. Wherever the two must be combined,
each frame is matched to the scan point nearest it *in time*, using the epoch
timestamps both files record.

---

## `da6idb` — the same thing without a window

The windows are a view onto a package. Everything they read and compute lives in
`da6idb`, which imports without Qt and works in a notebook or a script:

```python
from da6idb import open_folder

folder = open_folder("/data/beamtime")
folder.summary()                               # DataFrame: plan, motors, range, points
scan = folder[152]

scan.curve("psic_eta", "lambda250k_stats1_total", normalise="Ion_Chamber_1")
scan.statistics("lambda250k_stats1_total")     # max, min, centre, FWHM
scan.hkl_position()                            # psic H/K/L, mean and spread

images = scan.images()                         # the detector file, found as the GUI finds it
images.rois                                    # the rectangles the beamline recorded
images.roi_curve(0, "sum", background=2, normalise="scan/Ion_Chamber_1")

m = scan.hkl()
cube = m.grid(oversample=1)                    # xarray with real H/K/L coordinates
cube.sel(L=7.52, method="nearest").plot()
m.slice(normal=(1, 1, 0), offset=0.01, layers=5)
```

Tables come back as pandas DataFrames and gridded maps as `xarray.DataArray`,
so selecting a plane is `sel(L=7.52, method="nearest")` rather than an index, and
both draw themselves. A grid made in a notebook is the same cached `.vti` the
viewers read, so it opens in either.

`examples/getting-started.ipynb` walks through all of it against the scan that
ships with this repository. It adds the repository root to `sys.path` if the
package is not installed, so it runs from a fresh clone either way. **[`docs/api.md`](docs/api.md)** lists every public
function, generated from the source.

### Installing

```bash
pip install -e . --no-deps       # just the package, touching nothing else
pip install -e .                 # reading and reducing: numpy, h5py, scipy
pip install -e ".[notebook]"     # + pandas and xarray, for da6idb.session
pip install -e ".[hkl]"          # + xrayutilities and pyvista, for reciprocal space
pip install -e ".[gui]"          # + the windows
pip install -e ".[mcp]"          # + the MCP server, below
pip install -e ".[all]"
```

The split is the point: a notebook that only wants the scalar channels does not
have to install a renderer. `environment.yml` still builds the whole thing at once
if that is easier.

### Tests

```bash
pytest tests          # or: python tests/run.py, which needs no pytest
```

150 of them, pinning values rather than asking whether a call raises — a function
that moves has to come back with the same numbers. One starts a subprocess with
PyQt5, pyqtgraph and pyvistaqt made unimportable and loads the package anyway, so
"no Qt" is tested rather than claimed.

---

## `da6idb_mcp` — the same thing from a conversation

The five windows, and the package under them, as 52 tools an LLM can call. Ask
for scan 152's rocking curve and the peak statistics come back with the plot;
ask for an L-slice of a gridded map and the image comes back; ask it to open the
real `image_viewer` on scan 152 and the window appears on your screen, still
steerable from the chat while you watch it.

```bash
pip install -e ".[notebook,hkl,mcp]"           # headless tools
pip install -e ".[all]"                        # + the real windows
python -m da6idb_mcp --list                    # the tool names; needs no mcp
python -m da6idb_mcp --folder data             # the server itself, on stdio
```

`mcp` is the one new dependency. `.mcp.json` at the root registers it for Claude
Code; set `DA6IDB_PYTHON` to the interpreter that has the package if it is not
the one on your path.

Two layers. The headless tools — `scan_*`, `image_*`, `hkl_*`, `vti_*`,
`batch_*` — compute through `da6idb` and draw with Agg, so they need no window
and no desktop session. The `viewer_*` tools drive the actual Qt windows from a
separate process, because a VTK fault is a segfault rather than an exception and
must not be able to take the rest down with it; on a machine with no display
they say so in one sentence and nothing else is affected. `da6idb_eval` runs a
snippet against the very objects the other tools built, for the question nobody
wrote a tool for.

**[`docs/mcp.md`](docs/mcp.md)** has the tool list, the flags, and why it is
built in two processes.

---

## `scan_viewer.py` — the scan browser

```bash
python scan_viewer.py [directory]
```

The directory is remembered between sessions; **Open Directory…** changes it and
**Refresh** re-reads it without losing your selection.

### The listing (left, top)

One row per scan, filled in as the directory is read — no clicking needed. Only
files carrying Bluesky scan metadata are listed; the `pva*.h5` siblings are not.

| Column | |
|---|---|
| **Scan** | the number from the file name |
| **Img** | `✓` if the detector images are on disk beside it, `–` if not |
| **Plan** | `scan`, `rel_scan`, `grid_scan`, `count`… |
| **Motor** | what the plan moved; fixed width, elided when long — drag it wider |
| **Range** | `start → stop` for each motor, so the direction shows |
| **Pts** | points, or `11x21` for a grid scan |
| **Count [s]** | count time per point |

Every column can be dragged to a different width. Click a row, or ctrl/shift-click
several to overlay them. The selected file's name appears above the plot, shortened
to the part before the scan number and given once however many of a series you pick.

### Choosing what to plot (left, bottom)

One table with three tick columns:

- **X** — one field only; defaults to the motor the scan moved, taken from the
  metadata. Your own pick sticks across scans until you untick it or open a new
  directory.
- **Y** — any number of fields, none by default.
- **Norm** — one field; every Y is divided by it. No tick means no normalization.

An **EPOCH** entry joins the real fields wherever the file records it, so you can
plot against absolute time.

### The plot

Above it, when the master records them, sit the diffractometer's own
**psic_h/k/l averaged over the scan** — where the scan was pointing in reciprocal
space, without opening the HKL viewer. A **± spread** is shown for any axis that
moved, because on a rocking curve one of the three sweeps across the peak and a bare
mean would read like a fixed position. With several scans selected each gets a line
prefixed by its scan number; scans whose master predates these fields say so rather
than showing nothing. Older masters here do not record them — of the files to hand,
only scan 151 onward do.


Curves for each Y field, overlaid across every selected file, with peak statistics
below (max, min, half-max centre and FWHM). **Show dY/dX** overlays each curve's
derivative on a second axis. Zoom with the toolbar's rectangle tool; the mouse
position is read out beside the toolbar.

**A two-motor grid scan is drawn as a colour map** over the two motors, with a
colour scale, and the readout gives the value under the pointer as well as the
position. **2D map** turns that off to plot the raster as an ordinary X-Y curve.
A grid is recognised from the data — the two motors' levels must multiply to the
point count — so a two-motor `inner_product` scan, where both move together, is
correctly left as curves.

**Baseline…** opens the before/after table for one selected file in its own window.

**Load Image…** opens the detector frames; several selected scans open side by side.

---

## `image_viewer.py` — the detector frames

**Finding the image file.** Newer masters name the detector's own file in
`instrument/bluesky/metadata/detectors_file_relative_path`, which is how a file
that follows no naming convention gets found at all — `tise2_2_00353.h5` beside
`tise2_2_00353_master.hdf`, with no `pva` about it. It is looked for where the
master says, resolved against the experiment root when the recorded path is
relative, and then by its own name beside the master, which is where it lands once
a directory has been copied off the beamline.

Failing all that it falls back on naming: the older `pva_streaming` `file_name`, then
`pva_<stem>.h5`, and finally the plain `<stem>.h5` — a file need not start with `pva`
to be found. Because those last guesses are only names, a candidate is opened and
checked for an image stack before it is accepted, so a file that merely shares a name
is passed over rather than handed to the viewer. Nothing that resolved before
resolves differently.


```bash
python image_viewer.py pva_tise2_2_00152.h5 [more.h5 ...]
```

Several files appear side by side in one window, sharing one set of ROIs, one
colour scale and one frame control, with their views linked so they stay on the
same region. The frame control spans the longest file; a file that has run out
keeps showing its last frame and says so.

**Display**: colormap, **Log scale** (`log1p`, with the level boxes staying in raw
counts), **Auto levels** or typed **Color min/max**, **Full range**, **Aspect 1:1**,
**Transpose**, **Reset view**. **Panels ▾** brings back any closed dock.

**Frames**: ◀ ▶, a frame box, **Play**/**Pause**, FPS and **Auto replay**.

### ROIs

The four rectangles the detector was using are recreated on opening, **locked**, so
that nudging one by accident cannot silently stop it matching what the beamline
recorded. They come from the image file's own `entry/data/rois` where it has one,
and otherwise from the master's baseline (`<detector>_rois_roi1_min_x` and friends),
which is the only record newer scans keep. Unlock one in the ROI table to move it.



The rectangles stored in the file (`entry/data/rois`) are recreated on open and
start **locked** — they describe the detector's own ROIs, and nudging one would
silently stop it matching what the beamline recorded. The table has a tick for
**Vis** and **Lock** per ROI, numeric **Geometry** (X, Y, W, H, angle — editable
when unlocked), and buttons to add, rename, delete, lock all and plot. Right-click
an ROI for the same actions.

- **Plot** — one window per ROI holding **one curve per file**: any metric against
  any other across frames (sum, mean, min, max, CoM, plus the file's own CA
  channels). **Offset** stacks the curves — curve *i* at `y + i × offset`, with
  **Auto** picking a step; `0` overlays them again. *Single frame* mode plots the
  ROI's X and Y projections instead. A frame slider is present in both modes.
- **2D plot** — one window per file: a scatter point per frame, X/Y/Z chosen from
  the same metrics, Z driving the colour.
- **Norm** — divides the value axes by any of those metrics, after any background
  subtraction.

**Background ROI**: tick **Bkg** on one ROI and its mean intensity per pixel is
subtracted from every other ROI — `sum` loses `b × (pixel count)`, and mean/min/max
each lose `b`, per frame and per file. CoM is deliberately left on the raw pixels,
because a centroid weighted by background-subtracted values goes unstable wherever
the corrected sum passes through zero. Moving or resizing the background updates
the other ROIs' plots live.

**`scan/` channels**: where the master sits beside the image file, its primary
stream is offered alongside the image file's own channels under a `scan/` prefix,
each frame taking the scan point nearest it in time.

---

## `hkl_viewer.py` — the 3D reciprocal-space map

```bash
python hkl_viewer.py pva_tise2_2_00152.h5 [master.hdf]
```

or the **HKL 3D…** button in the image viewer.

Every pixel of every frame is a point in (H, K, L), fixed by the diffractometer
angles at that frame, the detector geometry and the sample's UB matrix. The
conversion is `xrayutilities`' area-detector Q conversion.

### Conversion setup (its own window)

Everything that decides *where a pixel lands*. Each value is tagged with where it
came from — `[master]` or `[image]`:

- **from the master's baseline**: energy, detector distance, centre pixel, pixel
  directions, UB matrix
- **from the image file**: the circle directions (Mu `z+`, Eta `y-`, Chi `x+`,
  Phi `y-`; Nu `z+`, Delta `y-`), the beam / in-plane / surface reference vectors,
  and the detector's physical size — the master has no equivalent
- **frames**: first, last, step, an intensity threshold and a cap on converted
  points

All of it is editable. **Apply and recompute** rebuilds the map; **Reset to file**
restores what was read.

**Where each parameter comes from.** Three sources, in this order — the first that
has a value wins:

| | supplies |
|---|---|
| **`hkl_setup.json`** in the folder | whatever you put in it, and nothing else |
| **the master** | energy, detector distance, centre pixel, pixel directions, UB |
| **the image file** | detector size, the six circle directions, the beam, in-plane and surface vectors — and the five above, when no master is found |

Both files carry those five, and the master wins; the image file is the fallback. The
panel tags every value with where it came from, so this is never guesswork.

Two per-frame quantities take the *image* file in preference to the master, and are
not overridable: the **angles**, because the master's read-back clock lags the
exposure and shifted every frame of a test scan by a full step, and the **filter
transmission**, because the master samples at scan points and misses the transient
the free-running detector catches. **Angles from master** switches the first if you
want it.

**Correcting the geometry for a whole folder.** Everything in this window is
editable, and **Save to folder** writes what you changed to `hkl_setup.json` beside
the data. Every scan in that folder then opens with it — this viewer, `hkl_grid.py`
and `batch_grid.py` alike — and each corrected value is tagged `setup` so it is clear
where it came from. With no such file the files decide everything, as before.

Only the values that *differ* from what the HDF5 files say are written, so saving a
corrected detector distance does not quietly pin the energy and the UB as well. **Delete
setup** removes the file and goes back to the files. A setup file that will not parse
stops the scan opening with the reason rather than being ignored.

The override is part of the grid's fingerprint, so a `.vti` made before a correction
is re-gridded after it rather than being handed back stale.

**Attenuator.** Some scans change their filter part way through, to hold a strong
peak and a weak tail in range at once. Those frames are not on the same scale, so the
map steps wherever the filter moved, and the detector often catches the change: in
`sio_sao_sto_00007` five angles were measured twice, once through each filter, one
reading up to **44× brighter** than the other.

**Divide by filter transmission** puts them all on the unattenuated beam, and the
repeated readings then agree to a few percent. It is ticked automatically when the
transmission actually varies and left alone when it does not, and the note beside it
says what the file records — how many filter settings, and the factor between the
weakest and the strongest. The per-frame value from the image file is used in
preference to the master's, which samples at scan points and misses the transients
the free-running detector catches. The threshold is applied to the counts as
recorded, so it keeps meaning the same thing whatever the attenuator is doing.

**Angles.** By default the per-frame angles stored with each frame in the image
file are used, *not* the master's. The master records one angle per scan point on a
read-back clock that lags the exposure; matching those onto frames shifts every
frame of a rocking curve by a whole step, which moves the entire map. **Angles from
master** switches to the resampled ones if you want them; the status bar always
says which is in force.

### Display (left panel)

- **Visibility** — colormap, **Background**, **Show points**, **Show slice**, point size
- **Intensity range** — min/max, or **Auto**
- **HKL range** — per-axis limits, or **Whole map**
- **Downsampling** — **On** plus **Max drawn**; takes every Nth point so the cloud
  thins evenly rather than losing a corner of reciprocal space
- **Gridded volume** — **Grid & load (.vti)** bins the pixels onto a regular grid
  over the HKL range set above and draws that instead; see below

The cloud is coloured by intensity with **opacity rising along the scale**, so faint
points fade out instead of filling the view with haze.

**Background** switches between black and white. Black shows a faint cloud best;
white is what a paper wants. Everything drawn over it follows — the H/K/L letters,
the tick numbers, the scalar bar and the slice handle — because pyvista takes each
label's colour from its own theme font, which is black, and would otherwise leave
them invisible on one background or the other. The same control is in the `.vti`
viewer.

### The slice

An arbitrary plane, not just axis-aligned: set the **Plane normal** (three
components) and an **Offset along normal** from the centre of the map, or drag the
yellow plane in the view — dragging writes the new normal and offset back into the
boxes. **⊥H ⊥K ⊥L** snap to the axis-aligned cases. **Lock slice (hide handles)**
freezes the plane and takes the arrow and outline out of the view while the slab
keeps drawing.

The H/K/L axes stay pinned to the whole map whatever is drawn, so hiding the points
or showing only a slice does not rescale them.

---

## `hkl_grid.py` — the gridded map

The 3D viewer draws a point cloud, which keeps every pixel's exact position but is
expensive to rebuild. Gridding bins the same pixels onto a regular lattice — mean
intensity per cell, unmeasured cells left as `NaN` — and writes a **VTK ImageData
(`.vti`)** file that ParaView, pyvista or any analysis script can open.

**From the viewer.** Set the **HKL range** (or leave **Whole map** ticked) and press
**Grid & load (.vti)**. The file is written beside the image file; pressing it again
reloads it rather than redoing the work. **Show volume** switches between the grid
and the cloud, and **Cell size ×** scales the automatic resolution.

What the cached file was made for is recorded inside it and checked, so it is only
reused when the **HKL range**, the **frame range**, the **cell size** and the
**geometry** all still match. Edit a limit, or the UB, energy, distance or angles in
the Conversion setup window, and it grids afresh rather than handing back a stale
map. The grid always uses whatever the setup window currently holds, edits included.

**From analysis code.**

```python
from hkl_grid import grid_to_vti, load_vti

path = grid_to_vti("pva_tise2_2_00152.h5",
                   hkl_range=(0.93, 1.11, -0.60, -0.36, 7.39, 7.66))
grid = load_vti(path)
volume = grid["intensity"].reshape(grid.dimensions, order="F")   # NaN where unmeasured
```

or on the command line:

```bash
python hkl_grid.py pva_tise2_2_00152.h5
python hkl_grid.py pva_tise2_2_00152.h5 --range 0.93 1.11 -0.60 -0.36 7.39 7.66
python hkl_grid.py pva_tise2_2_00152.h5 --oversample 0.75     # finer
```

The master file is found beside the image file unless `--master` says otherwise.

### Two different numbers, and why it matters

At every pixel the three sampling directions — along a detector row, along a
column, and from one frame to the next — span a small parallelepiped in reciprocal
space. Two things can be measured from it, and using one for the other is the easy
mistake:

| | on the example scan | what it is |
|---|---|---|
| **volume** `\|(drow × dcol) · dframe\|` | cube of side **6.7e-4** | the HKL volume one sample speaks for → the **cell size** |
| **axis-aligned extent** | **3.6e-4, 1.3e-3, 5.6e-4** | how far one sample reaches per axis → the **fuzzy footprint** |

They differ a lot because the parallelepiped is strongly skewed. Consecutive frames
are 1.2e-3 apart measured *perpendicular to the detector surface*, while pixels
within a frame are 4.9e-4 apart — the frames are about 2.5× further apart than the
detector samples. Most of the frame-to-frame motion slides *along* the surface
rather than away from it, so reading the per-axis extent as a cell size makes the
rocking axis look fourteen times coarser than it is.

### Why the samples are spread, not just binned

That 2.5× gap is why plain nearest-cell binning cannot be made fine. At a 6.7e-4
cell it leaves roughly one empty layer between every pair of frame surfaces — 83% as
many empty cells inside the shell as filled ones, which reads as stripes. Spreading
each sample over its footprint instead (fuzzy gridding, via
`xrayutilities.FuzzyGridder3D`) drops that to **3.3%** at the same cell size, and is
several times faster. So the grid is fine *and* continuous.

`Cell size ×` scales the automatic value: below 1 is finer at several times the
memory, above 1 lighter. Total cells are capped at 100 M (`--max-cells`), about
1.6 GB while gridding; past that the cell is coarsened rather than the machine
swapped.

### Scans that change their attenuator

Frames taken through different filters are divided by their transmission before they
are binned, so a scan whose attenuator moved does not leave a step in the map. This
happens by default where the transmission varies and not at all where it does not;
`--no-transmission` turns it off. Whether it was applied is recorded in the `.vti`,
and it is part of the cache key, so switching it re-grids rather than handing back
the other version. See the HKL viewer's **Attenuator** note above for what the
correction is worth on real data.

### Unmeasured cells

About a third of the bounding box holds data. The rest was never visited: each frame
maps to a *curved* surface, so the swept volume is a skewed shell and an axis-aligned
box around it is mostly outside. That is geometry, not a resolution problem, and
coarsening does not fix it.

Those cells are written as `NaN` so analysis can mask them. Anything **drawing** the
grid must replace them first — NaN handed to a GPU volume mapper does not come out
transparent, it lights the whole empty region up. `hkl_grid.make_drawable()` does
this, and the viewer calls it on load.

---

## `batch_grid.py` — a run of scans, gridded the same way

One scan at a time is fine for looking; comparing a series means gridding them all
identically. Point it at a folder of `pva*.h5` files, give it scan numbers and an HKL
range, and it writes one `.vti` per scan into an `output` folder beside them.

```bash
python batch_grid.py                 # opens empty
python batch_grid.py /data/beamtime  # opens on that folder
```

**Scan numbers** can be typed (`152, 154 157-160`) or filled from **first / last /
every n**. Numbers with no matching file are listed and skipped rather than stopping
the run. **From the first scan** fills the HKL box with the extent that scan reaches,
as a starting point.

### One lattice for the whole set

**The cell size is measured once, from the first scan in the list, and reused for
every scan after it.** With the same HKL range as well, every grid comes out with the
same origin, spacing and shape, so they can be subtracted, stacked or stepped through
cell for cell. Measuring each scan separately would give each its own lattice and
quietly make them incomparable — which is the whole point of the batch.

The **fuzzy footprint is not shared**. It describes how far one sample of *that* scan
reaches, so each scan keeps its own. This matters more than it sounds: on one test
set, scan 153's footprint along K was 2.0e-4 where its neighbours' was 1.3e-3,
because it was a different kind of scan. Forcing the first scan's footprint on it
would have smeared it about sevenfold.

If the folder holds an `hkl_setup.json`, the panel and the log say what it overrides,
and every scan in the batch is converted with it.

**Divide by filter transmission where it varies** applies the attenuator correction,
decided per scan — a series can mix scans that moved their filter with scans that did
not, and the log says which was which and by what factor.

Re-running skips scans already gridded for the same range (their file is untouched,
which is how the log tells *reused* from *written*), so an interrupted batch can be
resumed by running it again. **Stop** finishes the current frame and leaves no
half-written file.

From a script, without the window:

```python
from batch_grid import expand_scans, grid_scans

grid_scans("/data/beamtime", expand_scans(152, 160, 2),
           hkl_range=(0.93, 1.11, -0.60, -0.36, 7.39, 7.66))
```

`expand_scans(first, last, every)` is the same numbering the **Fill list** button
uses, and `parse_scans("152, 154 157-160")` reads the typed form.

---

## `vti_viewer.py` — the saved grid, on its own

The same scene as the HKL viewer — a volume on black with H/K/L axes, an arbitrary
slice plane and the opacity ramp — but it reads a finished `.vti` and nothing else.

```bash
python vti_viewer.py pva_tise2_2_00152_hkl_grid.vti
python vti_viewer.py                 # asks for a file
```

**File ▸ Open folder…** (`Ctrl+O`) lists every `.vti` in a folder down the left;
click one to open it, or walk the list with the arrow keys. Opening a single file
lists its neighbours too. **Keep the current set-up** carries everything over to the
next file you click — colormap, intensity range, HKL range, the slice with its
normal, offset and averaging, and the camera — so a set of scans can be compared in
the same corner of reciprocal space from the same angle, and the 2D window and its
cuts follow along. Turn it off and each file opens on its own full extent.

The file list, the controls and the 3D view are separated by splitters, so any of
them can be given more room; the controls start wide enough to show every box whole.

Two reasons to use it rather than the HKL viewer:

- **It opens in about a second.** There is no conversion: the grid is already the
  answer. The HKL viewer has to convert every pixel before it can draw anything.
- **It needs neither `xrayutilities` nor `h5py`**, only `numpy`, `pyqt` and
  `pyvista`. A grid can be looked at, sliced and cropped on a machine that could not
  have produced it — a laptop, a collaborator's desktop — with no beamline data
  present.

The panel is the HKL viewer's, minus what has no meaning without the detector data:
no conversion setup, no point cloud, no downsampling. **Colormap**, **Show volume**,
**Show slice** with the same arbitrary normal, **⊥H ⊥K ⊥L**, offset and
**Lock slice**, **Intensity range**, and an **HKL range** that cuts the volume down
to a box.

**Average over** gives the slice a thickness. A plane has none, so a one-cell slice
is as noisy as the data gets; averaging a few cells either side along the normal
trades a little resolution across the plane for much less noise in it. On the example
grid the point-to-point scatter falls from 5.4 at one cell to 3.5 at nine and 2.3 at
twenty-five. The 2D window follows the same setting, so both show the same slab.

**Offset step** sets how far one click of the offset arrows (or one arrow key) moves
the plane. Opening a file sets it to the grid's cell size, so a click steps one cell
— the smallest move that can show anything new — and you can type anything else. **File ▸ Open…** (`Ctrl+O`) loads another grid.

One deliberate difference: the slice is a **true plane through the volume**, cut by
pyvista's own `add_mesh_slice`, where the HKL viewer draws a slab of nearby points.
On a regular grid that comes out as a clean image rather than a scatter, so there is
no thickness to set.

That is also why it stays responsive on a 77-million-cell grid: `add_mesh_slice`
drives a `vtkCutter` straight from the plane widget, so dragging recuts inside the
VTK pipeline rather than re-slicing from Python on every event. The normal and offset
boxes are filled in from the widget when you let go of it. **Lock slice** has no
widget to drive the cutter, so there the plane is cut once and drawn as a plain mesh.

Any VTK ImageData opens, not only the grids from `hkl_grid.py`; the axes are
labelled H/K/L because that is what these files hold, and an array under another
name is drawn as-is.

### The slice as a flat image

**Slice as 2D image…** opens the plane on its own, drawn with matplotlib: axis
labels, tick numbers, a colour bar, the matplotlib toolbar for zoom and save, and a
cursor readout giving the full **H, K, L** and the intensity under the pointer. It
follows the plane, the crop and the colour settings as you change them in the 3D
window.

What the axes are labelled with depends on the plane, because only some planes have
HKL axes lying in them:

| slice | horizontal | vertical | above the plot |
|---|---|---|---|
| ⊥L | `H` | `K` | `L = 7.5215` |
| ⊥H | `K` | `L` | `H = 1.0390` |
| normal [1 1 0] | `L` | `along [0.707 -0.707 0.000]` | the normal and a point on it |
| normal [1 2 3] | `along […]` | `along […]` | the normal and a point on it |

Each axis is judged on its own: a plane can have one image axis running exactly along
an HKL axis and the other not, and then only that one carries real H/K/L tick
numbers. The other is labelled with the direction it runs in and its ticks are a
distance from the plane's centre — but the cursor readout always gives the true
H, K, L, whatever the plane.

**Log intensity** colours by the logarithm, which is usually the only way to see a
weak feature beside a Bragg peak. Zero and unmeasured cells stay blank — a log has
nothing to say about them. The scale runs **5 decades below the top** unless you set
a positive **Min** in the 3D window: fuzzy gridding leaves cells holding a millionth
of a count, and a floor at the smallest positive value would spread the colours over
twenty decades and squeeze everything real into the top fifth of the map.

**Cuts** offers two kinds, one at a time so each gets the width to be read:

- **Along the axes** — a profile down each axis of the image, taken through a
  crosshair. Click anywhere on the image to move it. The profiles are drawn against
  the same H/K/L axes as the image and stay lined up with it.
- **Arbitrary line** — drag a line anywhere on the image and read the profile along
  it, against distance in HKL from the start of the line. Useful for cutting along a
  streak or a rod that does not run down an axis. It is sampled at the image's own
  pixel pitch and starts as a line across the middle, before you drag one.

**Cut width** averages that many rows, columns, or parallel offsets into the cut: one
pixel is the true profile, a few average the noise down. That is averaging *within*
the plane; **Average over** in the 3D window averages *through* it, and the two
compose. With **Log intensity** on the
profiles' intensity axes go logarithmic too. The status line gives the H, K, L the
cut passes through — for a line, both endpoints and its length.

### Getting the numbers out

Two buttons, and the format follows the extension you type (or the filter you pick):
**`.txt`/`.dat`** plain text, **`.csv`**, **`.npz`** for a Python session, **`.h5`**.
Every file carries a header naming the source `.vti`, the plane, its normal and the
slab thickness, so it can be read months later.

**Export image…** writes the slice as a 2D array with its axes: the ASCII form is a
plain matrix, rows along y and columns along x, under a header giving each axis's
label, range, point count and step; the CSV puts the x values in the first row and
the y values in the first column so a spreadsheet shows the map; `.npz` and `.h5`
hold `intensity`, `x` and `y` as separate arrays. Unmeasured cells are `nan`.

**Export cut…** writes the cut as columns — **position, H, K, L, intensity** for
every point along it. For the arbitrary line, position is the distance from its
start, and the header records both endpoints in H, K, L. For the axis cuts you get
one file per axis in text form (`…_along_H.txt`, `…_along_K.txt`), or both in one
`.npz`/`.h5`. The two endpoints are also shown under the toolbar while the line is
active, so they can be read off without exporting.

The plane is resampled onto a regular lattice at the grid's own cell size
(trilinearly, via `scipy.ndimage`), so the image resolution matches the data rather
than being interpolated up. Cells the scan never reached are left blank instead of
being painted the bottom colour of the map. HKL is treated as one Cartesian frame, as
it is in the 3D view, so the image is drawn at equal aspect.

### What costs time

Everything scales with the cell count, close to linearly — a 2.9 M-cell grid does in
0.05 s what a 77 M-cell one takes 1.3 s to do. Two things are worth knowing:

- **Recolouring does not rebuild.** Changing the colormap or the intensity limits
  moves no cells, so the lookup table is swapped on the actors already in the scene:
  about 17 ms rather than 1.3 s. Only a change of *what* is drawn — the crop, the
  slice plane, show/hide — rebuilds.
- **Moving the slice does not rebuild either.** The normal, the offset and the
  ⊥H ⊥K ⊥L presets drive the cutter that is already in the scene, so the plane moves
  in about 30 ms and the axes are left alone — they used to be torn down and redrawn
  on every step, which is what made them flicker. Locking only switches the handles
  off; the slice stays put.
- **Dragging cuts once, on release.** The handle moves freely and the slice follows
  when you let go.
- **`generate_triangles=True` on the cutter.** pyvista defaults it off, which puts
  `vtkCutter` on an unthreaded path that builds general polygons: 1.18 s against
  0.028 s here, for output identical to the point. This one flag was most of the
  sluggishness.
- **The volume coarsens while the camera moves.** `AutoAdjustSampleDistances` is off
  by default, so every frame of a rotation was rendered at full quality; on, VTK
  drops the ray sampling during the move and sharpens again when it stops.

What still rebuilds is loading, cropping and showing/hiding the volume — around a
second at 77 M cells, since the volume has to be handed to VTK again. If that is too
slow for the way you are working, the lever is **Cell size ×** when the grid is made:
at 2 the grid is eight times smaller and those operations are roughly eight times
quicker.

---

## Scale

Conversion is fast; rendering is the limit. A 101-frame rocking curve is 26.9 M
pixels and converts in about 0.2 s; a 601-frame scan is 160 M pixels, or 1.9 GB of
coordinates alone. So each frame is reduced as it is converted — pixels at or below
the threshold dropped, and the brightest kept if a cap is still exceeded — and the
display thins further. On one test scan a threshold of 100 counts left 84,419 points
of 26.9 M.

## Example data

One scan is included so the viewers can be tried without beamline data:

```
tise2_2_00152_master.hdf     1.5 MB
pva_tise2_2_00152.h5        20.0 MB
```

Scan 152 is a `rel_scan` rocking `psic_eta` from 20.692° to 22.692° over 101
points, 101 frames of 516×516. Because the angles actually move, it gives the HKL
map a real reciprocal-space volume — H 0.920…1.160, K −0.634…−0.353, L 7.337…7.705
— rather than the single detector surface a stationary `count` would give. It also
carries the full geometry in its baseline, so the HKL conversion works out of the
box.

From a clone:

```bash
conda env create -f environment.yml && conda activate bluesky-scan-viewer
python scan_viewer.py .          # scan 152 appears in the listing, Img ✓
```

Tick a Y field to plot it, then **Load Image…** for the frames and **HKL 3D…** for
the reciprocal-space map.

## The rest of the data is not in this repository

Apart from that pair, `.gitignore` keeps every `*.h5` / `*.hdf` out: the image
stacks run to 110 MB each, over GitHub's hard 100 MB-per-file limit, and a working
directory of them holds ~558 MB. Point the viewers at wherever the data lives.
