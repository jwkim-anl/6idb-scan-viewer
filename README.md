# 6-ID-B scan viewers

Three Qt viewers for the HDF5 files written at APS 6-ID-B by a Bluesky plan and
the pvapy streaming consumer:

| | |
|---|---|
| **`scan_viewer.py`** | browse a directory of scans, plot the scalar channels, read the baseline |
| **`image_viewer.py`** | the detector frames, with ROIs and ROI plots |
| **`hkl_viewer.py`** | the frames mapped into reciprocal space, in 3D |

They chain together: the scan browser opens the image viewer for whatever scan you
have selected, and the image viewer opens the HKL map for the file it is showing.
Each also runs on its own from the command line.

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
  /home/beams18/USER6IDB/.conda/envs/6idb-bits/bin/python scan_viewer.py [directory]
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

**Angles.** By default the per-frame angles stored with each frame in the image
file are used, *not* the master's. The master records one angle per scan point on a
read-back clock that lags the exposure; matching those onto frames shifts every
frame of a rocking curve by a whole step, which moves the entire map. **Angles from
master** switches to the resampled ones if you want them; the status bar always
says which is in force.

### Display (left panel)

- **Visibility** — colormap, **Show points**, **Show slice**, point size
- **Intensity range** — min/max, or **Auto**
- **HKL range** — per-axis limits, or **Whole map**
- **Downsampling** — **On** plus **Max drawn**; takes every Nth point so the cloud
  thins evenly rather than losing a corner of reciprocal space

The cloud is coloured by intensity with **opacity rising along the scale**, so faint
points fade out instead of filling the view with haze.

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
