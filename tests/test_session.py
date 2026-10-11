"""Pinning the notebook layer: the shapes and labels a session actually sees."""

import numpy as np
from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example
from helpers import write_image, write_master

import da6idb
from da6idb.hkl import detector as _detector


def test_the_package_imports_without_qt():
    """The whole point: a notebook must not need a window toolkit."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in ('PyQt5','pyqtgraph','pyvistaqt'): return self\n"
        "    def load_module(self, name): raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import da6idb\n"
        "from da6idb import files, scan, images, batch, session\n"
        "from da6idb.hkl import geometry, convert, grid, slice\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-600:]


def test_scanset_lists_and_indexes():
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=9)
        write_master(d / "s_00008_master.hdf", points=5)
        folder = da6idb.open_folder(d)
        assert folder.numbers == [7, 8]
        summary = folder.summary()
        check("session/summary_columns", list(summary.columns))
        check("session/summary_index", list(summary.index))
        check("session/summary_row7", {k: summary.loc[7, k] for k in
                                        ("plan", "motors", "points", "image")})
        assert folder[7].number == "00007"
        try:
            folder[99]
        except KeyError:
            pass
        else:
            raise AssertionError("an absent scan should raise")


def test_curve_and_statistics():
    with Sandbox() as d:
        master = write_master(d / "s_00009_master.hdf", points=21)
        scan = da6idb.open_scan(master)
        curve = scan.curve(y="lambda250k_stats1_total")
        assert curve.index.name == "psic_eta" and len(curve) == 21
        check("session/curve_columns", list(curve.columns))
        normalised = scan.curve(y="lambda250k_stats1_total",
                                normalise="Ion_Chamber_1")
        check("session/curve_normalised_columns", list(normalised.columns))
        # the monitor is 2.0 throughout, so normalising must halve it
        assert np.allclose(normalised.iloc[:, 0].to_numpy(),
                           curve.iloc[:, 0].to_numpy() / 2.0)
        stats = scan.statistics("lambda250k_stats1_total")
        check("session/statistics_columns", sorted(stats.columns))
        check("session/statistics", stats.iloc[0].to_dict())


def test_curve_picks_the_index_for_a_count_plan():
    with Sandbox() as d:
        master = write_master(d / "s_00010_master.hdf", plan="count", motors=())
        curve = da6idb.open_scan(master).curve(y="lambda250k_stats1_total")
        assert curve.index.name == "INDEX"


def test_images_and_roi_curve():
    with Sandbox() as d:
        write_master(d / "s_00011_master.hdf", points=11)
        write_image(d / "pva_s_00011.h5", frames=13, shape=(16, 16))
        images = da6idb.open_scan(d / "s_00011_master.hdf").images()
        assert images.n_frames == 13 and images.frame_shape == (16, 16)
        check("session/roi_table", images.rois.to_dict("list"))
        channels = images.channels()
        assert len(channels) == 13
        curve = images.roi_curve(0, "sum")
        assert len(curve) == 13 and curve.name == "ROI1.sum"
        check("session/roi_curve_head", curve.head(4).tolist())
        # a mean over the same box cannot exceed its maximum
        assert (images.roi_curve(0, "mean") <= images.roi_curve(0, "max")).all()


def test_hkl_tables():
    if not have_example():
        return
    m = da6idb.open_scan(EXAMPLE_MASTER).hkl()
    check("session/hkl_sources", m.sources().to_dict())
    check("session/hkl_resolution", m.resolution().to_dict())
    check("session/hkl_extent", m.extent().to_dict())
    cloud = m.cloud(threshold=8000, max_points=500)
    check("session/cloud_columns", list(cloud.columns))
    assert (cloud["intensity"] > 8000).all()


def test_grid_is_a_labelled_cube():
    if not have_example():
        return
    with Sandbox() as d:
        m = da6idb.open_scan(EXAMPLE_MASTER).hkl()
        cube = m.grid(oversample=10.0, output=d / "g.vti")
        assert cube.dims == ("H", "K", "L")
        check("session/cube_shape", list(cube.shape))
        # coordinates are real HKL, so selecting by value works
        plane = cube.sel(L=float(cube.L[len(cube.L) // 2]), method="nearest")
        assert plane.dims == ("H", "K")
        check("session/cube_attrs", {k: cube.attrs[k] for k in
                                      ("frames", "oversample", "transmission_corrected")})
        # the same file the viewers cache
        assert (d / "g.vti").is_file()


def test_slice_is_labelled_by_axis_where_it_can_be():
    if not have_example():
        return
    with Sandbox() as d:
        m = da6idb.open_scan(EXAMPLE_MASTER).hkl()
        m.grid(oversample=10.0, output=d / "g.vti")
        flat = m.slice(normal=(0, 0, 1), output=d / "g.vti")
        check("session/slice_dims_axis_aligned", list(flat.dims))
        tilted = m.slice(normal=(1, 1, 0), output=d / "g.vti")
        check("session/slice_dims_tilted", list(tilted.dims))
        # averaging over a slab cannot make it noisier
        thin = m.slice(normal=(0, 0, 1), layers=1, output=d / "g.vti")
        thick = m.slice(normal=(0, 0, 1), layers=5, output=d / "g.vti")
        assert float(np.nanstd(np.diff(thick.values, axis=1))) <= \
               float(np.nanstd(np.diff(thin.values, axis=1)))


def test_fields_is_a_table_you_can_choose_axes_from():
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=9)
        scan = da6idb.open_folder(d)[7]
        table = scan.fields()
        check("session/fields_columns", list(table.columns))
        check("session/fields_index", list(table.index))
        assert table.loc["psic_eta", "role"] == "scanned"
        assert bool(table.loc["lambda250k_stats1_total", "hinted"])
        # the plain list is still one call away, and in file order
        names = scan.field_names()
        assert sorted(names) == sorted(table.index)
        assert names[-1] == "INDEX"
        # curve() still works off the names, not the table
        assert not scan.curve(y="lambda250k_stats1_total").empty


def test_a_made_up_roi_reduces_like_a_recorded_one():
    """The point of the whole thing: your rectangle is worth the file's."""
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=13)
        write_image(d / "pva_s_00007.h5", frames=13, shape=(16, 16))
        images = da6idb.open_folder(d)[7].images()
        stored = images.rois.iloc[0]
        same = (stored["x"], stored["y"], stored["w"], stored["h"])
        assert np.array_equal(images.roi_curve(0, "sum").values,
                              images.roi_curve(same, "sum").values)
        # tuple, Rect and the stored row as a dict are all the same rectangle
        assert images.rect(same) == images.rect(dict(stored)) == images.rect(0)
        mine = (2, 3, 5, 4)
        check("session/made_up_roi", {
            k: float(v) for k, v in images.roi_series(mine).iloc[0].items()})
        assert images.roi_curve(mine, "sum").name == "roi.sum"


def test_rect_rejects_what_is_not_a_rectangle():
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=5)
        write_image(d / "pva_s_00007.h5", frames=5, shape=(16, 16))
        images = da6idb.open_folder(d)[7].images()
        for bad, kind in (((1, 2, 3), ValueError), (99, IndexError)):
            try:
                images.rect(bad)
            except kind as error:
                assert str(error)          # says what to do instead
            else:
                raise AssertionError(f"{bad!r} should have raised {kind}")


def test_show_draws_the_frame_and_its_rois():
    import matplotlib
    matplotlib.use("Agg")
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=7)
        write_image(d / "pva_s_00007.h5", frames=7, shape=(16, 12))
        images = da6idb.open_folder(d)[7].images()
        ax = images.show(0)                       # the recorded ones
        assert len(ax.patches) == len(images.rois)
        ax = images.show(1, rois=[(2, 3, 5, 4)], log=True)
        assert len(ax.patches) == 1
        assert len(images.show(0, rois=()).patches) == 0
        # drawn the way the viewer shows it: (x, y) on the axes is frame[x, y]
        shown = ax.images[0].get_array()
        frame = images.frame(1)
        assert all(shown[y, x] == frame[x, y]
                   for x in (0, 7, 15) for y in (0, 5, 11))
        assert ax.get_ylim()[0] > ax.get_ylim()[1]    # y runs downwards


def test_show_names_the_rois_it_draws():
    """A signal box and a background box must not both read "new"."""
    import matplotlib
    matplotlib.use("Agg")
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=5)
        write_image(d / "pva_s_00007.h5", frames=5, shape=(16, 12))
        images = da6idb.open_folder(d)[7].images()
        labels = lambda ax: [t.get_text() for t in ax.texts]
        assert labels(images.show(0, rois=[(1, 1, 4, 4)])) == ["new"]
        assert labels(images.show(0, rois=[(1, 1, 4, 4), (1, 6, 4, 4)])) == [
            "new1", "new2"]
        # the recorded ones keep the names the file gave them
        recorded = labels(images.show(0))
        assert recorded == list(images.rois["name"])
        assert labels(images.show(0, rois=[0, (1, 1, 4, 4)])) == [
            recorded[0], "new"]



def test_two_theta_per_frame_and_averaged():
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=9)
        write_image(d / "pva_s_00007.h5", frames=9, shape=(16, 16))
        images = da6idb.open_folder(d)[7].images()
        spot = (2, 3, 6, 6)
        per_frame = images.two_theta_angles(spot)
        assert len(per_frame) == images.n_frames
        assert per_frame.index.name == "frame"
        # the plain mean is the mean; the weighted one is a weighted mean
        assert np.isclose(images.two_theta(spot, weight=None), per_frame.mean())
        weights = images.roi_series(spot)["sum"].to_numpy(dtype=float)
        assert np.isclose(images.two_theta(spot),
                          np.average(per_frame.to_numpy(), weights=weights))
        # None is the whole detector, which is just a big rectangle
        assert len(images.two_theta_angles(None)) == images.n_frames
        try:
            images.two_theta(spot, weight="nonsense")
        except ValueError as error:
            assert "weight must be" in str(error)
        else:
            raise AssertionError("should have refused")


def test_two_theta_of_the_example_spot():
    if not have_example():
        return
    images = da6idb.open_scan(EXAMPLE_MASTER).images()
    spot = (440, 360, 20, 20)
    circle = float(images.geometry().detector_angles[1][0])
    check("session/two_theta_spot", {
        "circle_delta": round(circle, 6),
        "at_peak": round(float(images.two_theta_angles(spot).max()), 6),
        "weighted": round(images.two_theta(spot), 6),
        "plain": round(images.two_theta(spot, weight=None), 6),
    })
    # the box sits below the centre pixel, so it is a smaller angle than the
    # circle reads; a stored ROI straddling the centre is not
    assert images.two_theta(spot) < circle
    assert abs(images.two_theta(0) - circle) < 0.01


def test_two_theta_averages_the_angles_not_the_pixels():
    """Two theta is convex across the detector, so the two differ."""
    if not have_example():
        return
    images = da6idb.open_scan(EXAMPLE_MASTER).images()
    g = images.geometry()
    wide = (111, 330, 400, 20)          # a strip right across the detector

    # the angle at the centre pixel is the least anywhere on the detector --
    # a geometry statement, so asked of the geometry, not of the counts
    centre = float(_detector.two_theta_at(g, *g.center_pixel,
                                          hxrd=images.converter())[0])
    for x in (111.0, 211.0, 411.0, 511.0):
        assert float(_detector.two_theta_at(g, x, g.center_pixel[1],
                                            hxrd=images.converter())[0]) > centre

    # so averaging the angles over a wide strip must exceed the angle at its
    # centre of mass, which is what evaluating at the centroid would give
    rect = images.rect(wide)
    metrics = images.roi_series(wide)
    at_centroid = _detector.two_theta_at(
        g, *rect.to_frame(metrics["comx"], metrics["comy"], images.frame_shape),
        hxrd=images.converter())
    averaged = images.two_theta_angles(wide).to_numpy()
    assert (averaged > at_centroid).all()
    check("session/two_theta_jensen_gap",
          round(float((averaged - at_centroid).mean()) * 1000, 3))


def test_two_theta_background_moves_it_towards_the_feature():
    if not have_example():
        return
    images = da6idb.open_scan(EXAMPLE_MASTER).images()
    spot, empty = (440, 360, 20, 20), (440, 390, 20, 20)
    raw = images.two_theta(spot)
    clean = images.two_theta(spot, background=empty)
    check("session/two_theta_background", {
        "raw": round(raw, 6), "with_background": round(clean, 6)})
    assert raw != clean
    # with the background gone, the frames holding the spot carry the
    # average: it moves towards the angle at the peak of the rocking curve
    peak = float(images.two_theta_angles(spot, background=empty).iloc[
        int(images.roi_curve(spot, "sum", background=empty).idxmax())])
    assert abs(clean - peak) < abs(raw - peak)


def test_background_can_flip_which_side_of_the_circle_the_angle_lands():
    """The trap: a raw centre of mass follows the box, not the feature.

    Background spread evenly over a region has its centre of mass at the
    region's own centre. When that centre is on the opposite side of the
    detector's centre pixel from the feature, and the background is most of
    what the box holds, the reported angle comes out the wrong side of the
    circle reading -- above it while the spot is plainly below.
    """
    from da6idb.hkl.geometry import HklGeometry
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=1)
        write_image(d / "pva_s_00007.h5", frames=1, shape=(16, 16))
        images = da6idb.open_folder(d)[7].images()
        stack = np.full((1, 16, 16), 20.0, dtype=np.float32)
        stack[0, 7, 8] += 80.0                 # a weak feature just below centre
        images.data.stack = stack
        images._geometry = HklGeometry(
            image_path=images.path, master_path=None, frame_shape=(16, 16),
            n_frames=1, distance=1000.0, detector_size=(16.0, 16.0),
            center_pixel=(8.0, 7.0),
            detector_angles=np.array([[0.0], [10.0]]))
        images._converter = None

        box = (5, 4, 6, 6)        # y 4-10, centre 6.5: above the centre pixel
        empty = (5, 11, 6, 4)     # pure background
        patch = stack[0, 5:11, 4:10]
        assert 20 * patch.size / patch.sum() > 0.85      # background dominates

        raw = float(images.two_theta_angles(box)[0])
        clean = float(images.two_theta_angles(box, background=empty)[0])
        assert raw > 10.0, raw          # follows the box, the wrong side
        assert clean < 10.0, clean      # follows the feature, which is below
        check("session/two_theta_background_flip",
              {"raw": round(raw, 6), "corrected": round(clean, 6)})


def test_slice_can_be_placed_by_a_point_on_it():
    if not have_example():
        return
    hkl = da6idb.open_scan(EXAMPLE_MASTER).hkl()
    coarse = dict(oversample=6.0)
    by_offset = hkl.slice(normal=(0, 0, 1), offset=0.0, **coarse)
    centre_l = float(by_offset.attrs["point"][2])

    point = (1.04, -0.49, centre_l - 0.02)
    by_point = hkl.slice(normal=(0, 0, 1), point=point, **coarse)
    assert np.allclose(by_point.attrs["point"], point)
    assert np.isclose(by_point.attrs["offset"], -0.02)
    # the same plane asked for the other way round must give the same image
    same = hkl.slice(normal=(0, 0, 1), offset=-0.02, **coarse)
    assert np.isclose(float(same.attrs["point"][2]), point[2])
    a = np.asarray(by_point); b = np.asarray(same)
    assert a.shape == b.shape
    assert np.allclose(a, b, equal_nan=True)

    # tilted: the plane's own coordinates are measured from the point given
    tilted = hkl.slice(normal=(1, 1, 2), point=point, **coarse)
    assert np.allclose(tilted.attrs["point"], point)
    assert all(name.startswith("along [") for name in tilted.dims)


def test_slice_leaves_unmeasured_cells_blank():
    """A slice has to scale its colours on measured data, not on fill.

    ``grid`` already returns NaN where the scan never reached; a slice that
    turned those into zeros would paint most of the plane the bottom colour
    and drag every automatic colour range down with it.
    """
    if not have_example():
        return
    hkl = da6idb.open_scan(EXAMPLE_MASTER).hkl()
    cube = hkl.grid(oversample=6.0)
    assert np.isnan(np.asarray(cube)).any()          # the lattice has holes

    sl = hkl.slice(normal=(0, 0, 1), offset=0.0)
    values = np.asarray(sl, dtype=float)
    empty = np.isnan(values)
    assert empty.any(), "nothing came back unmeasured"
    # whatever is left must be real data, not a floor of zeros
    assert (values[~empty] == 0).mean() < 0.02
    check("session/slice_blank_fraction", round(float(empty.mean()), 3))

    # and a slab average must not let one NaN layer swallow the rest
    thick = np.asarray(hkl.slice(normal=(0, 0, 1), offset=0.0, layers=5),
                       dtype=float)
    assert np.isnan(thick).mean() <= empty.mean() + 0.01
    assert np.nanmax(thick) > 0


def test_cut_between_two_hkl_points():
    if not have_example():
        return
    hkl = da6idb.open_scan(EXAMPLE_MASTER).hkl()
    hkl.grid(oversample=6.0)
    start, end = (0.96, -0.50, 7.52), (1.14, -0.50, 7.52)
    cut = hkl.cut(start, end)
    assert list(cut.columns) == ["distance", "H", "K", "L", "intensity"]
    assert np.isclose(cut["distance"].iloc[0], 0.0)
    assert np.isclose(cut["distance"].iloc[-1], np.linalg.norm(
        np.array(end) - np.array(start)))
    # the line was asked for along H at fixed K and L, so that is where it ran
    assert np.allclose(cut["K"], -0.50) and np.allclose(cut["L"], 7.52)
    assert cut.attrs["start"] == list(start) and cut.attrs["end"] == list(end)
    check("session/cut_shape", {"rows": len(cut),
                                "measured": int(cut["intensity"].notna().sum())})

    # width averages, and keeps the cut on the plane when given its normal
    wide = hkl.cut(start, end, width=5, normal=(0, 0, 1))
    assert len(wide) == len(cut)
    assert np.allclose(wide["L"], 7.52)        # the widening stayed in-plane
    assert wide["intensity"].std() < cut["intensity"].std()   # and smoothed it

    # a cut that leaves the measured volume is NaN there, not an error
    away = hkl.cut((5.0, 5.0, 5.0), (5.1, 5.0, 5.0))
    assert away["intensity"].isna().all()


def test_the_lattice_is_held_between_slices():
    """Sweeping planes must not re-read the whole .vti for each one."""
    if not have_example():
        return
    import time
    hkl = da6idb.open_scan(EXAMPLE_MASTER).hkl()
    hkl.grid(oversample=6.0)
    held = hkl._cube_cache
    assert held is not None

    start = time.perf_counter()
    for offset in np.linspace(-0.02, 0.02, 8):
        hkl.slice(normal=(0, 0, 1), offset=float(offset))
    swept = time.perf_counter() - start
    # the same object served every plane, and nothing was read again
    assert hkl._cube_cache[0] == held[0]
    assert hkl._cube_cache[3] is held[3]
    assert swept < 8 * 0.5, f"a sweep of 8 planes took {swept:.2f}s"


def test_a_held_lattice_is_dropped_when_it_should_be():
    if not have_example():
        return
    from da6idb.hkl import grid as _g
    with Sandbox() as d:
        hkl = da6idb.open_scan(EXAMPLE_MASTER).hkl()
        out = d / "held.vti"
        box = (1.00, 1.08, -0.55, -0.45, 7.45, 7.60)
        hkl.grid(hkl_range=box, step=0.003, output=out)
        first = float(hkl.slice(normal=(0, 0, 1), offset=0.0).max())

        # another process regrids the same file: the next slice must see it
        cube = _g.load_vti(out)
        cube.point_data["intensity"] = np.asarray(
            cube.point_data["intensity"]) * 2.0
        cube.save(str(out))
        assert np.isclose(
            float(hkl.slice(normal=(0, 0, 1), offset=0.0).max()), 2 * first)

        # and reuse=False goes back to the data, not to either of those
        assert np.isclose(
            float(hkl.slice(normal=(0, 0, 1), offset=0.0, reuse=False).max()),
            first)

        # a different lattice is not served from the held one
        other = hkl.grid(hkl_range=box, step=0.006, output=d / "other.vti")
        assert other.shape != hkl.grid(hkl_range=box, step=0.003,
                                       output=out).shape


def test_a_returned_grid_belongs_to_the_caller():
    """It used to be a view on the file's buffer; now the cube is held."""
    if not have_example():
        return
    hkl = da6idb.open_scan(EXAMPLE_MASTER).hkl()
    cube = hkl.grid(oversample=6.0)
    before = float(hkl.slice(normal=(0, 0, 1), offset=0.0).max())
    cube[:] = -999.0
    assert np.isclose(float(hkl.slice(normal=(0, 0, 1), offset=0.0).max()),
                      before)
