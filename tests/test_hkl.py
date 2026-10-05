"""Pinning the reciprocal-space conversion: geometry in, H/K/L out."""

import numpy as np
from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example

import hkl_viewer as hv


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
    hxrd = hv.build_converter(g)
    row, column = (int(round(v)) for v in g.center_pixel)
    with h5py.File(EXAMPLE_MASTER) as f:
        pri = f["entry/instrument/bluesky/streams/primary"]
        theirs = {k: np.asarray(pri[f"psic_{k}"]["value"]) for k in "hkl"}
    worst = 0.0
    for frame in (0, 25, 50, 75, 100):
        ours = [np.asarray(a)[row, column] for a in hv.frame_hkl(g, hxrd, frame)]
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
    hxrd = hv.build_converter(g)
    h, k, l = (np.asarray(a) for a in hv.frame_hkl(g, hxrd, 50))
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
        import hkl_grid as hg
        assert hg.geometry_fingerprint(again) != hg.geometry_fingerprint(
            hv.read_geometry(image, use_setup=False))


def test_conversion_reduces_to_the_brightest():
    if not have_example():
        return
    g = hv.read_geometry(EXAMPLE_IMAGE)
    cloud = hv.convert(g, first=0, last=4, step=2, threshold=100, max_points=5000)
    assert cloud.points.shape[0] <= 5000
    assert cloud.points.shape[0] == cloud.intensity.size
    assert float(cloud.intensity.min()) > 100
    check("hkl/cloud_shape", [int(cloud.points.shape[0]), int(cloud.frames_used)])
    check("hkl/cloud_bounds", [list(b) for b in cloud.bounds()])
