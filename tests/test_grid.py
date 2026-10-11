"""Pinning the gridding: resolution, the lattice, the cache, the slice."""

import numpy as np
from helpers import EXAMPLE_IMAGE, Sandbox, check, have_example

from da6idb.hkl import geometry as hv, grid as hg, slice as sp

RANGE = (0.95, 1.10, -0.60, -0.40, 7.40, 7.65)


def _geometry():
    return hv.read_geometry(EXAMPLE_IMAGE)


def test_resolution_comes_from_the_sampling():
    if not have_example():
        return
    g = _geometry()
    volume, gap, in_plane = hg.sampling_cell(g)
    check("grid/sampling_cell", {"volume": volume, "gap": gap, "in_plane": in_plane})
    check("grid/natural_step", hg.natural_step(g))
    check("grid/footprint", hg.sampling_footprint(g))
    # oversampling scales the cell, nothing else
    assert np.allclose(hg.natural_step(g, oversample=2.0), 2.0 * hg.natural_step(g))


def test_measured_range():
    if not have_example():
        return
    low, high = hg.measured_range(_geometry())
    check("grid/measured_range", [list(low), list(high)])


def test_grid_writes_a_lattice_and_records_what_made_it():
    if not have_example():
        return
    with Sandbox() as d:
        out = hg.grid_to_vti(EXAMPLE_IMAGE, hkl_range=RANGE, oversample=4.0,
                             output=d / "g.vti", reuse=False)
        grid = hg.load_vti(out)
        check("grid/dimensions", list(grid.dimensions))
        check("grid/spacing", [round(float(v), 9) for v in grid.spacing])
        check("grid/origin", [round(float(v), 9) for v in grid.origin])
        check("grid/filled", [int(v) for v in grid.field_data["filled_cells"]])
        check("grid/recorded_range", [float(v) for v in grid.field_data["hkl_range"]])
        values = np.asarray(grid.point_data["intensity"])
        check("grid/intensity_summary", {
            "finite": int(np.isfinite(values).sum()),
            "max": float(np.nanmax(values)),
            "nan_fraction": round(float(np.isnan(values).mean()), 6),
        })


def test_vtk_point_order_is_x_fastest():
    """A transposed grid would still look plausible; this catches it."""
    if not have_example():
        return
    with Sandbox() as d:
        out = hg.grid_to_vti(EXAMPLE_IMAGE, hkl_range=RANGE, oversample=4.0,
                             output=d / "g.vti", reuse=False)
        grid = hg.load_vti(out)
        values = np.asarray(grid.point_data["intensity"])
        index = int(np.nanargmax(values))
        ih, ik, il = np.unravel_index(index, grid.dimensions, order="F")
        expected = np.asarray(grid.origin) + np.array([ih, ik, il]) * np.asarray(grid.spacing)
        assert np.allclose(expected, grid.points[index]), "point ordering is wrong"
        volume = values.reshape(grid.dimensions, order="F")
        assert volume[ih, ik, il] == values[index]


def test_cache_reuses_only_for_the_same_request():
    if not have_example():
        return
    with Sandbox() as d:
        g = _geometry()
        _grid, path, made = hg.ensure_grid(EXAMPLE_IMAGE, hkl_range=RANGE,
                                           oversample=4.0, output=d / "g.vti",
                                           geometry=g)
        assert made is True
        _grid, _path, made = hg.ensure_grid(EXAMPLE_IMAGE, hkl_range=RANGE,
                                            oversample=4.0, output=d / "g.vti",
                                            geometry=g)
        assert made is False, "an identical request should reuse the file"
        g.distance = g.distance + 5.0
        _grid, _path, made = hg.ensure_grid(EXAMPLE_IMAGE, hkl_range=RANGE,
                                            oversample=4.0, output=d / "g.vti",
                                            geometry=g)
        assert made is True, "edited geometry must not reuse the old grid"


def test_cancelling_writes_nothing():
    if not have_example():
        return
    with Sandbox() as d:
        try:
            hg.grid_to_vti(EXAMPLE_IMAGE, hkl_range=RANGE, oversample=4.0,
                           output=d / "g.vti", reuse=False,
                           progress=lambda done, total: done < 3)
        except hg.GridCancelled:
            pass
        else:
            raise AssertionError("cancelling should raise")
        assert not (d / "g.vti").is_file()


def test_plane_sampling_and_cuts():
    if not have_example():
        return
    with Sandbox() as d:
        out = hg.grid_to_vti(EXAMPLE_IMAGE, hkl_range=RANGE, oversample=4.0,
                             output=d / "g.vti", reuse=False)
        grid = hg.load_vti(out)
        hg.make_drawable(grid)
        centre = np.asarray(grid.center)
        image, us, vs, u, v, origin = sp.sample_plane(grid, (0, 0, 1), centre)
        check("grid/slice_shape", [int(image.shape[0]), int(image.shape[1])])
        check("grid/slice_basis", [list(u), list(v)])
        check("grid/slice_finite", int(np.isfinite(image).sum()))
        # a thicker slab averages, so it cannot be noisier
        thin = sp.sample_plane(grid, (0, 0, 1), centre, layers=1)[0]
        thick = sp.sample_plane(grid, (0, 0, 1), centre, layers=5)[0]
        thin_scatter = float(np.nanstd(np.diff(thin, axis=1)))
        thick_scatter = float(np.nanstd(np.diff(thick, axis=1)))
        assert thick_scatter <= thin_scatter


def test_plane_basis_prefers_the_axes():
    check("grid/basis_perp_l", [list(a) for a in sp.plane_basis((0, 0, 1))])
    check("grid/basis_perp_h", [list(a) for a in sp.plane_basis((1, 0, 0))])
    u, v = sp.plane_basis((1, 1, 0))
    assert abs(float(np.dot(u, v))) < 1e-12, "the basis must stay orthogonal"
    check("grid/slab_offsets", {n: sp.slab_offsets(n) for n in (1, 2, 5)})


def test_plane_origin_from_an_offset_or_from_a_point():
    from da6idb.hkl.slice import plane_origin
    centre = np.array([1.0, -0.5, 7.5])
    normal = np.array([0.0, 0.0, 2.0])          # not a unit vector on purpose
    origin, offset = plane_origin(centre, normal, offset=0.02)
    assert np.allclose(origin, [1.0, -0.5, 7.52]) and offset == 0.02
    # a point names the plane directly, and reports the offset it works out to
    origin, offset = plane_origin(centre, normal, point=[1.04, -0.49, 7.45])
    assert np.allclose(origin, [1.04, -0.49, 7.45])
    assert np.isclose(offset, -0.05)
    # only the component along the normal counts towards the offset
    _origin, offset = plane_origin(centre, normal, point=[9.0, 9.0, 7.5])
    assert np.isclose(offset, 0.0)
    # the two are inverses: feed the offset back and get the same plane
    point = [1.04, -0.49, 7.45]
    _o, off = plane_origin(centre, normal, point=point)
    back, _ = plane_origin(centre, normal, offset=off)
    assert np.isclose(np.dot(np.array(point) - back, [0, 0, 1]), 0.0)
    try:
        plane_origin(centre, [0, 0, 0], offset=1.0)
    except ValueError as error:
        assert "zero" in str(error)
    else:
        raise AssertionError("a zero normal should have been refused")


def _toy_cube(values, origin=(0.0, 0.0, 0.0), spacing=(0.1, 0.1, 0.1)):
    """A small lattice with values we chose, to cut through."""
    import pyvista as pv
    cube = pv.ImageData(dimensions=values.shape, origin=origin, spacing=spacing)
    cube.point_data["intensity"] = values.ravel(order="F")
    return cube


def test_line_cut_between_two_points():
    from da6idb.hkl.slice import line_cut
    # a ramp along H, flat in K and L, so the cut's values are predictable
    h = np.arange(11, dtype=float)[:, None, None] * np.ones((11, 5, 5))
    cube = _toy_cube(h.astype(np.float32))

    distance, points, values = line_cut(cube, (0.0, 0.2, 0.2), (1.0, 0.2, 0.2))
    assert np.isclose(distance[0], 0.0) and np.isclose(distance[-1], 1.0)
    assert np.allclose(points[:, 1], 0.2) and np.allclose(points[:, 2], 0.2)
    # H runs 0 to 1 over 10 cells of 0.1, and the value is the cell index
    assert np.allclose(values, points[:, 0] / 0.1, atol=1e-5)

    # the ends name the cut: reversing them reverses the profile
    _d, _p, back = line_cut(cube, (1.0, 0.2, 0.2), (0.0, 0.2, 0.2))
    assert np.allclose(back, values[::-1], atol=1e-5)

    # off the lattice is NaN, not an error or a clamp
    _d, _p, outside = line_cut(cube, (0.0, 0.2, 9.0), (1.0, 0.2, 9.0))
    assert np.isnan(outside).all()

    try:
        line_cut(cube, (0.5, 0.2, 0.2), (0.5, 0.2, 0.2))
    except ValueError as error:
        assert "same point" in str(error)
    else:
        raise AssertionError("a zero-length cut should have been refused")


def test_line_cut_width_averages_across_the_line():
    from da6idb.hkl.slice import line_cut
    # flat everywhere except one bright plane at K = index 2
    values = np.zeros((11, 5, 5), dtype=np.float32)
    values[:, 2, :] = 90.0
    cube = _toy_cube(values)

    start, end = (0.2, 0.2, 0.2), (0.8, 0.2, 0.2)
    _d, _p, narrow = line_cut(cube, start, end, width=1)
    assert np.allclose(narrow, 90.0)
    # widening across K (the slice normal being L) pulls in the dark rows
    _d, _p, wide = line_cut(cube, start, end, width=3, normal=(0.0, 0.0, 1.0))
    assert np.allclose(wide, 30.0), wide[:3]
    # with the normal along K instead, the widening stays on the bright plane
    _d, _p, inplane = line_cut(cube, start, end, width=3, normal=(0.0, 1.0, 0.0))
    assert np.allclose(inplane, 90.0)


def test_loaded_grids_holds_and_drops_by_cells():
    from da6idb.hkl.grid import LoadedGrids
    with Sandbox() as d:
        held = LoadedGrids(max_cells=25)
        paths = []
        for n in range(3):
            path = d / f"g{n}.vti"
            path.write_text(f"pretend grid {n}")
            paths.append(path)
            held.put(path, f"grid {n}", cells=10)

        # 30 cells against a budget of 25, so the oldest went
        assert len(held) == 2
        assert held.get(paths[0]) is None
        assert held.get(paths[1]) == "grid 1"
        assert held.get(paths[2]) == "grid 2"

        # reading one makes it the newest, so the other goes first next time
        held.get(paths[1])
        fourth = d / "g3.vti"
        fourth.write_text("pretend grid 3")
        held.put(fourth, "grid 3", cells=10)
        assert held.get(paths[2]) is None, "the least recently used should have gone"
        assert held.get(paths[1]) == "grid 1"
        assert held.get(fourth) == "grid 3"

        # one grid over budget is still kept: it is the one on screen
        big = LoadedGrids(max_cells=5)
        big.put(paths[0], "huge", cells=1000)
        assert big.get(paths[0]) == "huge" and len(big) == 1


def test_loaded_grids_notices_a_file_that_changed():
    import time
    from da6idb.hkl.grid import LoadedGrids
    with Sandbox() as d:
        path = d / "g.vti"
        path.write_text("first")
        held = LoadedGrids()
        held.put(path, "the first read", cells=1)
        assert held.get(path) == "the first read"

        time.sleep(0.01)
        path.write_text("second, and longer")      # a regrid from elsewhere
        assert held.get(path) is None, "a changed file must not be served"

        held.put(path, "the second read", cells=1)
        assert held.get(path) == "the second read"

        # a path that is not there at all is simply a miss, not an error
        assert held.get(d / "gone.vti") is None
        held.put(d / "gone.vti", "nothing", cells=1)
        assert held.get(d / "gone.vti") is None
