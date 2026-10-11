# `da6idb` — every public function

Generated from the source: `python docs/make_api.py > docs/api.md` for this
list, `python docs/make_api_html.py` for `api.html`, which is the same thing
with every docstring in full. Regenerate rather than edit, so neither can
drift from the code.

Nothing here needs Qt. The reading modules need only numpy, h5py and scipy;
`da6idb.session` adds pandas and xarray; `da6idb.hkl` adds xrayutilities and
pyvista.

| module | what it is for |
|---|---|
| `da6idb.files` | finding a scan's files and pairing master with detector |
| `da6idb.scan` | the master: metadata, streams, rasters, peak statistics |
| `da6idb.images` | the detector stack, its channels and its stored ROIs |
| `da6idb.rois` | rectangles, and what they reduce each frame to |
| `da6idb.hkl.geometry` | the instrument description, overrides, attenuator |
| `da6idb.hkl.convert` | frames to a point cloud in reciprocal space |
| `da6idb.hkl.detector` | the scattering angle a detector pixel sees |
| `da6idb.hkl.grid` | binning onto a regular lattice, cached as `.vti` |
| `da6idb.hkl.slice` | planes and slabs through a grid |
| `da6idb.batch` | gridding a run of scans onto one lattice |
| `da6idb.export` | writing slices and cuts to text, CSV, npz or HDF5 |
| `da6idb.session` | the objects a notebook starts from |

### `da6idb.batch`

Gridding a run of scans onto one common lattice.

- `expand_scans(start, end, step=1)` — The scan numbers from *start* to *end* inclusive, every *step*.
- `parse_scans(text)` — Scan numbers from what someone typed: "152, 154 157-160".
- `find_scans(folder)` — ``{scan number: image path}`` for the pva files in *folder*.
- `grid_scans(folder, numbers, hkl_range=None, oversample=1.0, output=None, step=None, reuse=True, normalize_transmission=True, on_start=None, on_geometry=None, on_done=None, progress=None)` — Grid several scans onto one lattice. Returns ``[(number, path, note)]``.

### `da6idb.export`

Writing a slice or a cut out to a file.

- `slug(text)` — A label that is safe in a file name or an HDF5 key.
- `suffix_for(path, selected='')` — Which writer to use: the extension if it says, else the chosen filter.
- `write_matrix(path, image, x, y, x_label='x', y_label='y', description=(), suffix=None)` — A 2D array with its two axes. Unmeasured cells are written as NaN.
- `write_columns(path, datasets, description=(), suffix=None, note_width=None)` — Cuts as columns. Returns the names written.
- `write_dataarray(array, path, description=())` — Write a 2D :class:`xarray.DataArray`, taking its axes from the coords.

### `da6idb.files`

Finding the files a scan is made of, and pairing them up.

- `detector_files_from_metadata(master)` — The detector file paths the master records, in the order it lists them.
- `image_path_for_master(master)` — The image file belonging to *master*, or ``None`` if none is next to it.
- `master_path_for_image(image)` — The master file this image file belongs to.
- `read_master_rois(master)` — The detector's ROIs as recorded in the master's baseline.
- `is_image_file(path)` — Whether *path* is itself an image file (i.e. it holds an image stack).
- `resolve_image_file(path)` — The image file to open for *path*, which may be a master or an image file.
- `scan_number(path)` — The scan number in a file name: ``00008`` from ``tise2_2_00008_master.hdf``.
- `file_prefix(path)` — The part of a file name in front of the scan number: ``tise2_2``.
- `list_hdf_files(directory)` — Return the HDF5 files directly inside *directory*, sorted by name.

### `da6idb.hkl.convert`

Turning detector frames into a cloud of points in reciprocal space.

- `build_converter(geometry)` — The xrayutilities area-detector converter for this geometry.
- `frame_hkl(geometry, hxrd, frame)` — (H, K, L) for every pixel of one frame, each shaped like the detector.
- **`HklCloud`** — The reduced point cloud handed to the plotter.
  - `.bounds(self)` — ``((h0, h1), (k0, k1), (l0, l1))`` of the points kept.
- `convert(geometry, first=0, last=None, step=1, threshold=0.0, max_points=2000000, progress=None)` — Map the selected frames into (H, K, L), keeping the interesting pixels.

### `da6idb.hkl.detector`

The scattering angle a detector pixel sees, from xrayutilities' geometry.

- `circle_delta(geometry)` — What the delta circle read on each frame.
- `detector_basis(hxrd, detector_angles)` — Pixel (0, 0) and one step along each pixel axis, in lab coordinates.
- `two_theta_maps(geometry, x, y, hxrd=None)` — Scattering angles at pixel coordinates *x*, *y*, for every frame.
- `two_theta_at(geometry, x, y, hxrd=None)` — The scattering angle at pixel ``(x, y)`` on each frame, in degrees.

### `da6idb.hkl.geometry`

The instrument description that turns a detector pixel into H, K, L.

- **`HklGeometry`** — Everything the conversion needs, and where each part came from.
  - `.transmission_varies(self)` — Whether the attenuator moved during the scan.
  - `.frame_scale(self, frame)` — What to multiply a frame's counts by to undo the attenuator.
  - `.energy_ev(self)` — The beam energy in eV, which is what xrayutilities wants.
  - `.pixel_widths(self)` — One pixel's size, from the detector's own size and frame shape.
- `setup_path(folder)` — Where the per-folder setup lives.
- `read_setup(folder)` — The overrides for *folder*, as ``{field: raw value}``. Empty if none.
- `apply_setup(geometry, folder=None)` — Lay the folder's setup over *geometry*. Returns the fields it changed.
- `write_setup(folder, values)` — Write *values* as the folder's setup, or remove it when empty.
- `setup_differences(geometry)` — What in *geometry* differs from what its files alone say.
- `read_transmission(image, master, n_frames, frame_epoch=None)` — Attenuator transmission for each frame, or None if nothing records it.
- `read_geometry(image, master=None, angles_from_master=False, use_setup=True)` — The conversion parameters for *image*, master first where it has them.

### `da6idb.hkl.grid`

Bin a pva detector stack onto a regular HKL grid and save it as ``.vti``.

- **`GridCancelled`** — Raised when a progress callback asked to stop, so nothing is written.
- `geometry_fingerprint(geometry)` — A short digest of everything that decides where a pixel lands.
- `sampling_cell(geometry, frame=None, stride=8)` — What the sampling supports, in HKL units.
- `sampling_footprint(geometry, frame=None, stride=8)` — How far one sample reaches along each axis -- the fuzzy spread width.
- `natural_step(geometry, oversample=1.0, frame=None)` — The cell size the sampling supports: a cube holding about one sample.
- `measured_range(geometry, sample_frames=8)` — The HKL box the scan actually reaches, as ``(low, high)`` triples.
- `default_output(image, hkl_range=None, oversample=1.0)` — Where the grid for this file, range and detail level is kept.
- `matches_range(path, hkl_range, tolerance=1e-06)` — Whether an existing .vti was made for this HKL range.
- `grid_to_vti(image, master=None, hkl_range=None, output=None, step=None, width=None, oversample=1.0, first=0, last=None, frame_step=1, max_cells=MAX_CELLS, reuse=True, progress=None, geometry=None, normalize_transmission=None)` — Bin *image* onto a regular HKL grid and write it as a ``.vti``.
- `ensure_grid(image, master=None, hkl_range=None, output=None, reuse=True, geometry=None, oversample=1.0, **kwargs)` — Return ``(grid, path, regridded)`` for this file and HKL range.
- **`LoadedGrids`** — A few gridded maps held in memory, keyed by the file they came from.
  - `.stamp(path)` — What says a file on disk is still the one that was read.
  - `.get(self, path)` — What was held for *path*, or None if nothing current is.
  - `.put(self, path, value, cells)` — Hold *value* for *path*, dropping the oldest to stay in budget.
  - `.clear(self)` — Let go of everything held, for a caller watching its memory.
- `make_drawable(grid, fill=0.0)` — Replace NaN in a loaded grid so a volume mapper can draw it.
- `to_xarray(grid, name='intensity')` — A gridded map as an :class:`xarray.DataArray` with real H/K/L coordinates.
- `load_vti(path)` — Read a grid written by :func:`grid_to_vti`.
- `main(argv=None)` — The command line: grid one file and say what came out.

### `da6idb.hkl.slice`

Cutting a plane through a gridded map.

- `plane_basis(normal)` — Two in-plane directions for *normal*, preferring the HKL axes.
- `plane_origin(centre, normal, offset=0.0, point=None)` — Where the cutting plane sits, from an offset or from a point on it.
- `slab_offsets(layers)` — Offsets, in cells, of a slab of *layers* planes centred on zero.
- `sample_volume(grid, points)` — Trilinear values of *grid* at arbitrary HKL points; NaN off the grid.
- `average_along_normal(grid, points, normal, layers=1, step=None)` — Average *grid* over a slab of *layers* cells centred on *points*.
- `line_cut(grid, start, end, samples=None, width=1, normal=None, step=None)` — Intensity along the segment from *start* to *end*, both in HKL.
- `sample_plane(grid, normal, origin, pixels=None, layers=1)` — Resample *grid* on the plane. Returns (image, us, vs, u, v, origin).

### `da6idb.images`

The detector's image stack, and everything recorded alongside it.

- `nearest_index(frame_epoch, scan_epoch)` — For each frame, the scan point closest to it in time.
- `evenly_spaced_index(n_frames, n_scan_points)` — Map each frame onto a scan point by position, not by time.
- `read_scan_channels(image, frame_epoch, n_frames=None, master=None)` — The master file's primary fields, resampled onto the image's frames.
- **`ImageData`** — The whole contents of one ``pva*.h5`` file, held in memory.
  - properties: `n_frames`, `frame_shape`

### `da6idb.rois`

Rectangles on the detector, and what they reduce each frame to.

- **`Rect`** — A region of interest: corner, size and angle, all in detector pixels.
  - properties: `rotated`
  - `.from_roi(cls, roi)` — From a pyqtgraph ROI, so the windows can hand one straight over.
  - `.from_stored(cls, spec)` — From the dictionaries the files record.
  - `.corners(self)` — The four corners ``(x, y)``, going round, for drawing the rectangle.
  - `.sample_shape(self, shape)` — How many samples :meth:`sub_frame` takes along each axis.
  - `.sample_points(self, shape)` — Frame coordinates of every sample, shaped like the sub-image.
  - `.to_frame(self, u, v, shape)` — Where ROI coordinates land on the frame, as ``(x, y)`` pixels.
  - `.bbox(self, shape)` — Axis-aligned pixel bounds ``(x0, x1, y0, y1)``, clipped to *shape*.
  - `.pixel_count(self, shape, frame=None)` — How many pixels it covers -- the N in a ``background x N`` correction.
  - `.sub_frame(self, frame)` — The pixels of one frame inside the rectangle.
- `reduce_frame(sub)` — ``(sum, mean, min, max, comx, comy)`` of one sub-image indexed ``[x, y]``.
- `empty_series(n_frames)` — Every metric, all zero -- what a rectangle off the detector reduces to.
- `sub_frames(stack, rect)` — Each frame's pixels inside *rect*, laid out like :meth:`Rect.sample_points`.
- `series(stack, rect)` — Every metric, for every frame.

### `da6idb.scan`

Reading a master file: its metadata, its streams and the numbers in them.

- `resolve_hint_field(name, fields)` — The recorded field a hinted name stands for, or None for none.
- `scan_y_fields(path, fields)` — The detectors the scan itself nominates to plot, from its ``hints``.
- `stream_fields(path, stream)` — Field names under *stream* that carry a ``value`` dataset.
- `read_primary(path, fields)` — Read the ``value`` array of each of *fields* from the primary stream.
- `read_baseline(path)` — Return ``{field: (before, after)}`` from the baseline stream.
- `read_scan_info(path)` — What the scan was: plan, points, motors and the range they covered.
- `describe_fields(path, stream='primary')` — Every field in *stream*, with what it is and the range it covered.
- `default_x_field(fields)` — The last field, skipping trailing time-like ones (e.g. ``scaler_time``).
- `scan_x_field(path, fields)` — The field to plot against: the axis this scan actually scanned.
- `scan_hkl(path)` — Mean and spread of psic_h/k/l over the scan, or None if not recorded.
- `format_scan_hkl(values, decimals=4)` — ``H 1.0135 K -0.4926 ± 0.0386 L 7.4840`` for one scan.
- `read_grid(path, value_field, monitor_field='')` — *value_field* laid out over a two-motor grid scan, or ``None``.
- `axis_edges(values)` — Cell boundaries around the grid's sample positions.
- `curve_stats(x, y)` — Max/min (with position), and FWHM center/width, for one plotted curve.
- `grid_stats(grid)` — Max/min of a grid and the (x, y) they sit at.

### `da6idb.session`

The objects a notebook starts from.

- `open_folder(directory)` — Every scan in *directory*, ready to be listed or picked from.
- `open_scan(master)` — One scan, by the path of its master file.
- **`ScanSet`** — The scans in one directory, indexed by scan number.
  - properties: `numbers`
  - `.summary(self)` — One row per scan: what it was, what moved, how far, how many points.
- **`Scan`** — One scan: its master file, and everything reachable from it.
  - properties: `number`, `plan`, `motors`, `points`, `info`
  - `.fields(self, stream='primary')` — Every field you can plot, with what it is and the range it covered.
  - `.field_names(self, stream='primary')` — Just the names, in file order, including EPOCH and INDEX.
  - `.primary(self, fields=None)` — The primary stream, one column per field, one row per scan point.
  - `.baseline(self)` — What the baseline recorded before and after the scan.
  - `.curve(self, x=None, y=None, normalise=None)` — One x column and one or more y columns, normalised if asked.
  - `.statistics(self, y, x=None, normalise=None)` — Peak statistics for each y: maximum, minimum, centre and width.
  - `.hkl_position(self)` — Where the diffractometer was, averaged over the scan.
  - `.map(self, value, normalise='')` — A two-motor raster as a labelled 2D array, or None if it is not one.
  - `.images(self)` — The detector frames for this scan, found as the viewers find them.
  - `.hkl(self, **kwargs)` — This scan in reciprocal space, geometry and all.
- **`Images`** — The detector frames for one scan, and what was recorded with them.
  - properties: `stack`, `n_frames`, `frame_shape`, `rois`
  - `.frame(self, index)` — One frame, indexed ``[x, y]`` as the file stores it.
  - `.channels(self)` — Everything recorded per frame, including the master's own channels.
  - `.rect(self, roi=0)` — One ROI as a rectangle.
  - `.show(self, frame=0, rois=None, log=False, ax=None, cmap='viridis', **kwargs)` — One frame with its ROIs drawn on it, so you can place a new one.
  - `.roi_series(self, roi=0)` — Every metric for one ROI, frame by frame.
  - `.roi_curve(self, roi=0, metric='sum', normalise=None, background=None)` — One ROI's metric against frame number.
  - `.geometry(self)` — The instrument description for this scan, read once and kept.
  - `.converter(self)` — The xrayutilities converter for this geometry, built once.
  - `.two_theta_angles(self, roi=0, background=None)` — The scattering angle of what *roi* held, frame by frame.
  - `.two_theta(self, roi=0, background=None, weight='sum')` — One two theta for the whole scan: :meth:`two_theta_angles` averaged.
- **`HklMap`** — One scan in reciprocal space: the geometry, the cloud, the grid.
  - `.sources(self)` — Where each piece of the geometry came from.
  - `.cloud(self, threshold=0.0, max_points=2000000, **kwargs)` — The surviving pixels as H, K, L and intensity.
  - `.resolution(self)` — The cell the sampling supports, and what it was measured from.
  - `.extent(self)` — The H, K, L box the scan actually reaches.
  - `.grid(self, hkl_range=None, oversample=1.0, output=None, reuse=True, **kwargs)` — Bin onto a regular lattice and hand back a labelled cube.
  - `.slice(self, normal=(0, 0, 1), offset=0.0, point=None, layers=1, **kwargs)` — A plane through the grid, as a labelled 2D array.
  - `.cut(self, start, end, samples=None, width=1, normal=None, **kwargs)` — Intensity along the line between two H, K, L points.

<!-- 122 callables, 10 classes -->
