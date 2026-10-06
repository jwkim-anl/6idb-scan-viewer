"""Pinning the reciprocal-space conversion: geometry in, H/K/L out."""

import dataclasses

import numpy as np
from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example

from da6idb.hkl import convert as _c, geometry as hv


def test_geometry_sources_and_values():
    if not have_example():
        return
    g = hv.read_geometry(EXAMPLE_IMAGE)
    check("hkl/sources", g.sources)
    check("hkl/scalars", {
        "energy_kev": g.energy_kev, "distance": g.distance,
        "center_pixel": list(g.center_pixel), "detector_size": list(g.detector_size),
        "pixel_direction": list(g.pixel_direction),
        "sample_dirs": list(g.sample_dirs), "detector_dirs": list(g.detector_dirs),
        "primary_beam": list(g.primary_beam), "inplane": list(g.inplane),
        "surface": list(g.surface), "n_frames": g.n_frames,
        "frame_shape": list(g.frame_shape),
    })
    check("hkl/ub", g.ub)


def test_centre_pixel_matches_the_diffractometer():
    """The strongest check there is: our H/K/L against the instrument's own."""
    if not have_example():
        return
    import h5py
    g = hv.read_geometry(EXAMPLE_IMAGE)
    hxrd = _c.build_converter(g)
    row, column = (int(round(v)) for v in g.center_pixel)
    with h5py.File(EXAMPLE_MASTER) as f:
        pri = f["entry/instrument/bluesky/streams/primary"]
        theirs = {k: np.asarray(pri[f"psic_{k}"]["value"]) for k in "hkl"}
    worst = 0.0
    for frame in (0, 25, 50, 75, 100):
        ours = [np.asarray(a)[row, column] for a in _c.frame_hkl(g, hxrd, frame)]
        mine = np.array(ours, dtype=float)
        theirs_frame = np.array([theirs[k][frame] for k in "hkl"])
        worst = max(worst, float(np.max(np.abs(mine - theirs_frame))))
    # pinned loosely on purpose: this is a physics agreement, not a checksum
    assert worst < 5e-3, f"centre-pixel HKL drifted from the instrument: {worst}"
    check("hkl/centre_pixel_agreement_decimals", int(-np.floor(np.log10(worst))))


def test_frame_hkl_corners():
    if not have_example():
        return
    g = hv.read_geometry(EXAMPLE_IMAGE)
    hxrd = _c.build_converter(g)
    h, k, l = (np.asarray(a) for a in _c.frame_hkl(g, hxrd, 50))
    corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
    check("hkl/frame50_corners",
          [[float(h[r, c]), float(k[r, c]), float(l[r, c])] for r, c in corners])


def test_transmission_is_read_and_only_corrects_when_it_moves():
    if not have_example():
        return
    g = hv.read_geometry(EXAMPLE_IMAGE)
    check("hkl/transmission_source", g.sources.get("transmission"))
    check("hkl/transmission_varies", g.transmission_varies())
    check("hkl/transmission_normalise", g.normalize_transmission)
    check("hkl/frame_scale_constant_filter", [g.frame_scale(i) for i in (0, 10, 50)])


def test_setup_file_overrides_and_only_records_changes():
    if not have_example():
        return
    import shutil
    with Sandbox() as d:
        shutil.copy(EXAMPLE_MASTER, d); shutil.copy(EXAMPLE_IMAGE, d)
        image = d / EXAMPLE_IMAGE.name
        plain = hv.read_geometry(image)
        assert hv.read_setup(d) == {}
        # nothing changed -> nothing to write
        assert hv.setup_differences(plain) == {}
        plain.distance = 845.5
        changed = hv.setup_differences(plain)
        assert set(changed) == {"distance"}
        hv.write_setup(d, changed)
        again = hv.read_geometry(image)
        assert again.distance == 845.5
        assert again.sources["distance"] == "setup"
        assert again.sources["ub"] == "master"       # untouched by the override
        # and the fingerprint must notice
        from da6idb.hkl import grid as hg
        assert hg.geometry_fingerprint(again) != hg.geometry_fingerprint(
            hv.read_geometry(image, use_setup=False))


def test_conversion_reduces_to_the_brightest():
    if not have_example():
        return
    g = hv.read_geometry(EXAMPLE_IMAGE)
    cloud = _c.convert(g, first=0, last=4, step=2, threshold=100, max_points=5000)
    assert cloud.points.shape[0] <= 5000
    assert cloud.points.shape[0] == cloud.intensity.size
    assert float(cloud.intensity.min()) > 100
    check("hkl/cloud_shape", [int(cloud.points.shape[0]), int(cloud.frames_used)])
    check("hkl/cloud_bounds", [list(b) for b in cloud.bounds()])


def _bare_geometry(**changes):
    """A geometry with no file behind it, to pin the arithmetic alone."""
    from pathlib import Path
    g = hv.HklGeometry(image_path=Path("none.h5"), master_path=None,
                       frame_shape=(100, 100), n_frames=3,
                       distance=1000.0, detector_size=(100.0, 100.0),
                       center_pixel=(50.0, 50.0))
    return dataclasses.replace(g, **changes) if changes else g


def test_delta_axis_and_sign_come_from_the_recorded_directions():
    from da6idb.hkl import detector as dt
    # as 6-ID-B records it: pixels run y- then z-, delta turns about y-
    assert dt.delta_pixel_axis(_bare_geometry()) == (1, -1.0)
    # hang the detector the other way up and the sign goes with it
    assert dt.delta_pixel_axis(
        _bare_geometry(pixel_direction=("y-", "z+"))) == (1, 1.0)
    # swap the pixel axes and it is comx, not comy, that follows delta
    assert dt.delta_pixel_axis(
        _bare_geometry(pixel_direction=("z-", "y-"))) == (0, -1.0)
    # a detector whose pixels do not follow delta at all is refused
    try:
        dt.delta_pixel_axis(_bare_geometry(pixel_direction=("y-", "x+")))
    except ValueError as error:
        assert "delta circle" in str(error)
    else:
        raise AssertionError("should have refused")


def test_pixel_offset_is_the_angle_the_pixel_subtends():
    from da6idb.hkl import detector as dt
    g = _bare_geometry()                       # 1 unit pixels, 1000 away
    assert dt.pixel_offset(g, 50.0) == 0.0     # the centre reads the circle
    assert np.isclose(dt.pixel_offset(g, 51.0),
                      -np.degrees(np.arctan2(1.0, 1000.0)))
    # an arctangent, not a small-angle approximation: 1000 pixels out is 45
    assert np.isclose(dt.pixel_offset(g, 1050.0), -45.0)
    assert np.isclose(dt.pixel_offset(g, -950.0), 45.0)


def test_delta_angles_add_the_circle_reading_of_each_frame():
    from da6idb.hkl import detector as dt
    g = _bare_geometry(detector_angles=np.array([[0.0, 0.0, 0.0],
                                                 [10.0, 20.0, 30.0]]))
    assert np.allclose(dt.delta_angles(g, 50.0), [10.0, 20.0, 30.0])
    # one pixel per frame is taken frame by frame
    assert np.allclose(dt.delta_angles(g, [50.0, 1050.0, 50.0]),
                       [10.0, -25.0, 30.0])


def test_delta_is_the_circle_reading_that_would_centre_the_feature():
    """Checked against xrayutilities, which knows nothing of this module."""
    if not have_example():
        return
    g = hv.read_geometry(EXAMPLE_IMAGE)
    from da6idb.hkl import detector as dt
    hxrd = _c.build_converter(g)
    frame = 0
    sample = [float(row[frame]) for row in g.sample_angles]
    nu, delta = (float(row[frame]) for row in g.detector_angles)
    cx, cy = g.center_pixel

    def hkl_at(delta_value, i, j):
        q = hxrd.Ang2Q.area(*sample, nu, delta_value, UB=g.ub)
        return np.array([c[int(i), int(j)] for c in q])

    for pixel in (260.0, 338.0, 430.0):
        moved = float(dt.delta_angles(g, pixel)[frame])
        # the feature's own pixel at the real delta, against the centre pixel
        # at the delta we derive: the same place in reciprocal space
        assert np.allclose(hkl_at(delta, cx, pixel), hkl_at(moved, cx, cy),
                           atol=1e-9)
    # and it is not a no-op: off-centre really does move the angle
    assert abs(float(dt.delta_angles(g, 430.0)[frame]) - delta) > 0.3
