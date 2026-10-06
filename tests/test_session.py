"""Pinning the notebook layer: the shapes and labels a session actually sees."""

import numpy as np
from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example
from helpers import write_image, write_master

import da6idb


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


def test_delta_angles_per_frame_and_averaged():
    with Sandbox() as d:
        write_master(d / "s_00007_master.hdf", points=9)
        write_image(d / "pva_s_00007.h5", frames=9, shape=(16, 16))
        images = da6idb.open_folder(d)[7].images()
        spot = (2, 3, 6, 6)
        per_frame = images.delta_angles(spot)
        assert len(per_frame) == images.n_frames
        assert per_frame.index.name == "frame"
        # the plain mean is the mean; the weighted one is a weighted mean
        assert np.isclose(images.delta_angle(spot, weight=None),
                          per_frame.mean())
        weights = images.roi_series(spot)["sum"].to_numpy(dtype=float)
        assert np.isclose(images.delta_angle(spot),
                          np.average(per_frame.to_numpy(), weights=weights))
        # None is the whole detector, which is just a big rectangle
        assert len(images.delta_angles(None)) == images.n_frames
        try:
            images.delta_angle(spot, weight="nonsense")
        except ValueError as error:
            assert "weight must be" in str(error)
        else:
            raise AssertionError("should have refused")


def test_delta_angle_of_the_example_spot():
    if not have_example():
        return
    images = da6idb.open_scan(EXAMPLE_MASTER).images()
    spot = (440, 360, 20, 20)
    check("session/delta_angle_spot", {
        "circle": round(float(images.geometry().detector_angles[1][0]), 6),
        "at_peak": round(float(images.delta_angles(spot).max()), 6),
        "weighted": round(images.delta_angle(spot), 6),
        "plain": round(images.delta_angle(spot, weight=None), 6),
    })
    # the box sits below the centre pixel, so it is a lower angle than the
    # circle reads -- and the stored ROI straddling the centre is not
    assert images.delta_angle(spot) < float(images.geometry().detector_angles[1][0])
