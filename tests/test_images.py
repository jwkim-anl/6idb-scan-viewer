"""Pinning the image stack, its per-frame channels and its ROIs."""

import numpy as np
from helpers import EXAMPLE_IMAGE, Sandbox, check, have_example, write_image, write_master

from da6idb import files, images as iv
from da6idb.rois import Rect


def test_nearest_index_matching():
    scan = np.array([0.0, 1.0, 2.0, 3.0])
    frames = np.array([-0.4, 0.1, 0.6, 1.49, 1.51, 3.9])
    check("images/nearest_index", iv.nearest_index(frames, scan))


def test_stack_and_channels():
    with Sandbox() as d:
        image = write_image(d / "pva_s_00001.h5", frames=13, shape=(16, 16))
        data = iv.ImageData(image)
        assert data.stack.shape == (13, 16, 16)
        assert data.n_frames == 13 and data.frame_shape == (16, 16)
        assert data.epoch is not None and data.epoch.size == 13
        check("images/channel_names", sorted(k for k in data.channels))


def test_stored_rois_from_the_image_file():
    with Sandbox() as d:
        image = write_image(d / "pva_s_00002.h5")
        check("images/stored_rois", iv.ImageData(image).stored_rois)


def test_rois_fall_back_to_the_master_baseline():
    with Sandbox() as d:
        write_master(d / "s_00003_master.hdf", baseline={
            "lambda250k_rois_roi1_min_x": 2, "lambda250k_rois_roi1_min_y": 3,
            "lambda250k_rois_roi1_size_x": 5, "lambda250k_rois_roi1_size_y": 4,
        })
        image = write_image(d / "pva_s_00003.h5", rois=False)
        data = iv.ImageData(image)
        check("images/rois_from_master", data.stored_rois)


def test_scan_channels_matched_by_timestamp():
    with Sandbox() as d:
        write_master(d / "s_00004_master.hdf", points=11, epoch0=1.79e9)
        image = write_image(d / "pva_s_00004.h5", frames=13, epoch0=1.79e9)
        data = iv.ImageData(image)
        scan = {k: v for k, v in data.channels.items() if k.startswith("scan/")}
        assert scan, "the master's fields should be offered"
        check("images/scan_channels_by_time",
              {k: v for k, v in sorted(scan.items()) if "psic_eta" in k})


def test_scan_channels_without_a_timestamp():
    """No clock to match on, so frames spread evenly over the scan points."""
    with Sandbox() as d:
        write_master(d / "s_00005_master.hdf", points=11)
        image = write_image(d / "pva_s_00005.h5", frames=13, timestamps=False)
        data = iv.ImageData(image)
        assert data.epoch is None
        eta = data.channels["scan/psic_eta"]
        assert eta.size == 13
        # the ends land exactly on the first and last scan point
        import h5py
        with h5py.File(d / "s_00005_master.hdf") as f:
            truth = np.asarray(
                f["entry/instrument/bluesky/streams/primary/psic_eta/value"])
        assert eta[0] == truth[0] and eta[-1] == truth[-1]
        check("images/scan_channels_by_position", eta)


def test_a_mismatched_master_is_refused():
    """Clocks a decade apart describe different scans; pair nothing."""
    with Sandbox() as d:
        write_master(d / "s_00006_master.hdf", points=11, epoch0=1.70e9)
        image = write_image(d / "pva_s_00006.h5", frames=13, epoch0=1.79e9)
        data = iv.ImageData(image)
        assert not any(k.startswith("scan/") for k in data.channels)


def test_evenly_spaced_index_edges():
    check("images/evenly_spaced", {
        f"{n}->{p}": iv._evenly_spaced_index(n, p)
        for n, p in ((0, 11), (1, 11), (2, 11), (11, 11), (5, 3), (30, 4))
    })


def test_example_image():
    if not have_example():
        return
    data = iv.ImageData(EXAMPLE_IMAGE)
    check("images/example_shape", list(data.stack.shape))
    check("images/example_rois", data.stored_rois)
    check("images/example_channels", sorted(data.channels))
    check("images/example_epoch_span",
          [float(data.epoch[0]), float(data.epoch[-1])] if data.epoch is not None else None)


def test_rect_corners_match_the_axes_sub_frame_walks():
    """What show() draws has to be what series() sums."""
    rect = Rect(10.0, 20.0, 8.0, 6.0, 30.0)
    corners = rect.corners()
    check("images/rect_corners_30deg", np.round(corners, 6).tolist())
    # corner, then one width along, then one height across: the same two unit
    # vectors sub_frame steps along, so the drawing cannot drift from the sum
    assert np.allclose(corners[0], (10.0, 20.0))
    assert np.isclose(np.hypot(*(corners[1] - corners[0])), 8.0)
    assert np.isclose(np.hypot(*(corners[3] - corners[0])), 6.0)
    # square corners, and unrotated it is just the plain box
    assert np.isclose(np.dot(corners[1] - corners[0], corners[3] - corners[0]), 0.0)
    plain = Rect(10.0, 20.0, 8.0, 6.0).corners()
    assert np.allclose(plain, [(10, 20), (18, 20), (18, 26), (10, 26)])
