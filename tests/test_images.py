"""Pinning the image stack, its per-frame channels and its ROIs."""

import numpy as np
from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example
from helpers import write_image, write_master

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


def testevenly_spaced_index_edges():
    check("images/evenly_spaced", {
        f"{n}->{p}": iv.evenly_spaced_index(n, p)
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


def test_to_frame_puts_a_centre_of_mass_back_where_it_came_from():
    """comx/comy are in the sub-image's coordinates, not the detector's."""
    from da6idb.rois import series
    grid_x, grid_y = np.mgrid[0:60, 0:60]
    blob = 1000.0 * np.exp(-((grid_x - 33.0) ** 2 + (grid_y - 27.0) ** 2) / 8.0)
    stack = blob[None].astype(np.float32)
    width, height = 16.0, 14.0
    for angle in (0.0, 20.0, 45.0, 90.0, 135.0, 200.0):
        radians = np.deg2rad(angle)
        along = np.array([np.cos(radians), np.sin(radians)])
        across = np.array([-np.sin(radians), np.cos(radians)])
        # rotation is about the corner, so put the corner where the blob
        # lands in the middle of the box whatever the angle
        corner = np.array([33.0, 27.0]) - width / 2 * along - height / 2 * across
        rect = Rect(float(corner[0]), float(corner[1]), width, height, angle)
        metrics = series(stack, rect)
        x, y = rect.to_frame(metrics["comx"][0], metrics["comy"][0], (60, 60))
        assert np.hypot(x - 33.0, y - 27.0) < 0.05, (angle, x, y)

    # unrotated, the origin is the clipped integer corner the slice started
    # at -- not the fractional one, which would shift every result
    frame = np.zeros((1, 60, 60), dtype=np.float32)
    frame[0, 33, 27] = 1000.0
    for rect in (Rect(25, 20, 16, 14), Rect(24.6, 19.3, 16.0, 14.0)):
        metrics = series(frame, rect)
        x, y = rect.to_frame(metrics["comx"][0], metrics["comy"][0], (60, 60))
        assert (x, y) == (33.0, 27.0)


def _with_frame_positions(path, n_frames, **columns):
    """A master carrying a frame_positions group, as a fly writer leaves."""
    import h5py
    write_master(path, points=max(n_frames // 2, 2))
    with h5py.File(path, "a") as f:
        group = f.create_group(iv.FRAME_POSITIONS_PATH)
        for name, values in columns.items():
            group.create_dataset(name, data=np.asarray(values))
    return path


def test_frame_positions_are_read_per_frame():
    with Sandbox() as d:
        path = _with_frame_positions(
            d / "s_00020_master.hdf", 6,
            cryo_x=np.linspace(0.0, 1.0, 6),
            in_scan=np.array([False, True, True, True, True, False]),
            line=np.arange(6),
        )
        got = iv.read_frame_positions(path, 6)
        assert sorted(got) == ["frame/cryo_x", "frame/in_scan", "frame/line"]
        assert np.allclose(got["frame/cryo_x"], np.linspace(0.0, 1.0, 6))
        # booleans come through as 0 and 1, so they can be plotted and
        # divided by like any other column
        assert got["frame/in_scan"].dtype == float
        assert list(got["frame/in_scan"]) == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_a_column_that_is_not_per_frame_is_left_out():
    with Sandbox() as d:
        path = _with_frame_positions(
            d / "s_00020_master.hdf", 6,
            cryo_x=np.linspace(0.0, 1.0, 6),
            something_else=np.zeros(4),      # not one row per frame
        )
        got = iv.read_frame_positions(path, 6)
        assert "frame/cryo_x" in got
        assert "frame/something_else" not in got, "a short column was lined up"
        # and asked about a different frame count, none of it applies
        assert iv.read_frame_positions(path, 99) == {}


def test_a_master_without_the_group_gives_nothing():
    with Sandbox() as d:
        plain = write_master(d / "s_00001_master.hdf", points=5)
        assert iv.read_frame_positions(plain, 5) == {}
        assert iv.read_frame_positions(None, 5) == {}
        assert iv.read_frame_positions(d / "missing.hdf", 5) == {}


def test_the_fly_scan_prefers_its_recorded_positions():
    """The resampled channel and the recorded one must not be confused."""
    folder = EXAMPLE_MASTER.parent / "test_26_3"
    image = EXAMPLE_MASTER.parent / "lambda" / "test_26_3" / "test_00020.h5"
    if not (folder.is_dir() and image.is_file()):
        return
    data = iv.ImageData(image)
    recorded = data.channels["frame/cryo_x"]
    resampled = data.channels["scan/cryo_x"]
    assert recorded.size == data.n_frames == 170
    # the stream holds 120 points against 170 frames, so carrying them
    # across lands a long way out -- which is why the writer records them
    assert np.abs(recorded - resampled).max() > 0.3
    check("images/frame_positions", {
        "frames": int(data.n_frames),
        "in_scan": int(np.count_nonzero(data.channels["frame/in_scan"])),
        "columns": sorted(k for k in data.channels if k.startswith("frame/")),
    })
