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
