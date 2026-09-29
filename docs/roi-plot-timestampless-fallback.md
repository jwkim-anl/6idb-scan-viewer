# ROI plot parameters for timestamp-less PVA image files

## Problem

When a `pva*.h5` image file contained **only** image data (`entry/data/data`)
and no `entry/data/frame_timestamp`, the ROI plot's X/Y/Norm parameter
dropdowns in `image_viewer.py` never offered the matching `*_master.hdf`
file's scan parameters (motor positions, etc.) — even when that master file
sat right next to the image file on disk. Only the built-in ROI metrics
(sum, mean, min, max, comx, comy, frame) were selectable.

## Root cause

The dropdowns are built by `ImageViewerWindow.metric_keys()` from
`ImageData.channels`, which is populated by `read_scan_channels()`. That
function resamples the master file's `primary` stream onto the image's
frames by matching each frame's timestamp to the nearest scan point in time
(`nearest_index()`), because the detector free-runs and rarely has the same
number of points as the scan.

That time-based match requires a per-frame clock. Without one
(`frame_epoch is None`), `read_scan_channels()` returned `{}` immediately —
before it even looked up the master file — so no scan parameters were ever
offered, regardless of whether a master file existed.

## Fix

`read_scan_channels()` now accepts an optional `n_frames` argument and
branches on whether a usable frame timestamp is present:

- **With a timestamp** (unchanged): match each frame to the nearest scan
  point in time via `nearest_index()`, and discard the whole channel set if
  the two files' clocks don't line up (the existing `5 * spacing` sanity
  check).
- **Without a timestamp** (new): fall back to `_evenly_spaced_index()`,
  which spreads frames evenly across the master's scan points by position —
  frame 0 maps to scan point 0, the last frame maps to the last scan point,
  and everything in between is linearly interpolated and rounded to the
  nearest index. The scan point count itself comes from
  `_scan_point_count()`, which takes the most common length among the
  master's own `primary` stream fields (a few fields, like read-once
  presets, can have a different length than the scan).

`ImageData.__init__` now passes the frame count through:

```python
self.channels.update(read_scan_channels(self.path, self.epoch, n_frames))
```

## Behavior change

A PVA file with only `entry/data/data` will now pick up its matching
`*_master.hdf` file's parameters in the ROI plot dropdowns (prefixed
`scan/`), using positional alignment instead of time alignment.

## Caveat

Positional alignment has no timing information to sanity-check against, so
there is no way to detect a mismatched or unrelated master file the way the
timestamp path can (via the `5 * spacing` check). It also assumes the camera
ran for the same duration as the scan with no significant lead/lag; if the
camera started well before or after the scan, frames near the start/end can
be off by more than one scan point. This is a best-effort fallback for files
that carry no timing data at all, not a replacement for the timestamp-based
match.

## Files changed

- `image_viewer.py`
  - Added `_evenly_spaced_index()`
  - Added `_scan_point_count()`
  - `read_scan_channels()`: new `n_frames` parameter and no-timestamp branch
  - `ImageData.__init__()`: passes `n_frames` to `read_scan_channels()`
