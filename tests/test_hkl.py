"""Pinning the reciprocal-space conversion: geometry in, H/K/L out."""

import dataclasses

import numpy as np
from pathlib import Path

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



def test_circle_delta_is_the_motor_reading_per_frame():
    from da6idb.hkl import detector as dt
    g = hv.HklGeometry(image_path=Path("none.h5"), master_path=None,
                       frame_shape=(8, 8), n_frames=3,
                       detector_angles=np.array([[1.0, 2.0, 3.0],
                                                 [10.0, 20.0, 30.0]]))
    assert np.allclose(dt.circle_delta(g), [10.0, 20.0, 30.0])
    # a still detector is recorded once and broadcast over the frames
    g = dataclasses.replace(g, detector_angles=np.array([[1.0], [10.0]]))
    assert np.allclose(dt.circle_delta(g), [10.0, 10.0, 10.0])
    try:
        dt.circle_delta(dataclasses.replace(g, detector_angles=np.zeros((3, 3))))
    except ValueError as error:
        assert "detector circles" in str(error)
    else:
        raise AssertionError("should have refused")


def test_two_theta_matches_the_bragg_condition():
    """Against |Q| from the conversion, which gets there another way."""
    if not have_example():
        return
    from da6idb.hkl import detector as dt
    g = hv.read_geometry(EXAMPLE_IMAGE)
    hxrd = _c.build_converter(g)
    frame = 0
    sample = [float(row[frame]) for row in g.sample_angles]
    nu, delta = (float(row[frame]) for row in g.detector_angles)

    for x, y in ((311.0, 338.0), (120.0, 260.0), (470.0, 430.0)):
        q = hxrd.Ang2Q.area(*sample, nu, delta, UB=np.eye(3))
        magnitude = np.sqrt(sum(c[int(x), int(y)] ** 2 for c in q))
        bragg = 2 * np.degrees(np.arcsin(
            magnitude * hxrd.wavelength / (4 * np.pi)))
        assert np.isclose(dt.two_theta_at(g, x, y, hxrd=hxrd)[frame], bragg,
                          atol=1e-9)
    # the centre pixel is what the circle reads, this scan having nu = 0
    assert np.isclose(dt.two_theta_at(g, *g.center_pixel, hxrd=hxrd)[frame],
                      delta, atol=1e-9)


def test_two_theta_is_exact_at_a_fractional_pixel():
    """The basis is affine, so a centre of mass is not rounded to a pixel."""
    if not have_example():
        return
    from da6idb.hkl import detector as dt
    g = hv.read_geometry(EXAMPLE_IMAGE)
    hxrd = _c.build_converter(g)
    angles = [float(row[0]) for row in g.detector_angles]
    origin, step_x, step_y = dt.detector_basis(hxrd, angles)
    full = np.asarray(hxrd.Ang2Q.getDetectorPos(*angles, dim=2))
    for i, j in ((0, 0), (311, 338), (515, 515), (120, 430)):
        assert np.allclose(origin + i * step_x + j * step_y, full[:, i, j],
                           atol=1e-9)
    # and between pixels it lands between their angles, not on one of them
    low, high = (float(dt.two_theta_at(g, 311.0, y, hxrd=hxrd)[0])
                 for y in (400.0, 401.0))
    middle = float(dt.two_theta_at(g, 311.0, 400.5, hxrd=hxrd)[0])
    assert min(low, high) < middle < max(low, high)
    assert np.isclose(middle, (low + high) / 2, atol=1e-6)


def test_two_theta_follows_both_circles_not_just_delta():
    """The point of asking xrayutilities rather than doing it by hand."""
    if not have_example():
        return
    from da6idb.hkl import detector as dt
    g = hv.read_geometry(EXAMPLE_IMAGE)
    off_centre = (470.0, 430.0)
    swung = dataclasses.replace(
        g, detector_angles=np.array([[3.0] * g.n_frames,
                                     list(g.detector_angles[1])]))
    straight = float(dt.two_theta_at(g, *off_centre)[0])
    turned = float(dt.two_theta_at(swung, *off_centre)[0])
    # 0.031 deg at this pixel: small, but it is the whole difference
    # between the angle the detector read and the angle the spot was at
    assert abs(turned - straight) > 0.02, (straight, turned)
    # a pixel off-centre horizontally is at a larger angle than its vertical
    # offset alone would say -- which is what delta on its own would miss
    vertical_only = float(dt.two_theta_at(g, *g.center_pixel, hxrd=None)[0])
    assert float(dt.two_theta_at(g, 470.0, g.center_pixel[1])[0]) > vertical_only
