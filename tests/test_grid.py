"""Pinning the gridding: resolution, the lattice, the cache, the slice."""

import numpy as np
from helpers import EXAMPLE_IMAGE, Sandbox, check, have_example

import hkl_grid as hg
import hkl_viewer as hv
import slice_plot as sp

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
