"""Pinning how a master and its image file find each other."""

from helpers import EXAMPLE_IMAGE, EXAMPLE_MASTER, Sandbox, check, have_example
from helpers import write_image, write_master

from da6idb import files


def test_scan_number_and_prefix():
    cases = ["tise2_2_00152_master.hdf", "pva_tise2_2_00152.h5",
             "test_00001_master.hdf", "lineup_00034_master.hdf", "noDigits.hdf"]
    from pathlib import Path
    check("files/scan_number", {c: files.scan_number(Path(c)) for c in cases})
    check("files/file_prefix", {c: files.file_prefix(Path(c)) for c in cases})


def test_pairing_by_convention():
    with Sandbox() as d:
        write_master(d / "s_00001_master.hdf")
        write_image(d / "pva_s_00001.h5")
        assert files.image_path_for_master(d / "s_00001_master.hdf").name == "pva_s_00001.h5"
        assert files.master_path_for_image(d / "pva_s_00001.h5").name == "s_00001_master.hdf"


def test_pairing_without_the_pva_prefix():
    """Newer writers drop the prefix; the plain name must still be found."""
    with Sandbox() as d:
        write_master(d / "s_00002_master.hdf")
        write_image(d / "s_00002.h5")
        found = files.image_path_for_master(d / "s_00002_master.hdf")
        assert found is not None and found.name == "s_00002.h5"


def test_a_file_sharing_the_name_but_holding_no_images_is_refused():
    import h5py
    with Sandbox() as d:
        write_master(d / "s_00003_master.hdf")
        with h5py.File(d / "s_00003.h5", "w") as f:
            f.create_dataset("something/else", data=[1, 2, 3])
        assert files.image_path_for_master(d / "s_00003_master.hdf") is None


def test_declared_path_is_preferred_and_falls_back():
    """The master names the file; found there, else by name beside the master."""
    with Sandbox() as d:
        master = d / "exp" / "s_00004_master.hdf"
        write_master(
            master,
            metadata={
                "master_file_path": "exp/s_00004_master.hdf",
                "detectors_file_relative_path": "detector/s_00004.h5\n",
            },
        )
        assert files.detector_files_from_metadata(master) == ["detector/s_00004.h5"]
        # where it says
        write_image(d / "detector" / "s_00004.h5")
        files._RESOLVE_CACHE.clear()
        assert files.image_path_for_master(master).samefile(d / "detector" / "s_00004.h5")
        # and when that is gone, beside the master
        (d / "detector" / "s_00004.h5").unlink()
        write_image(master.parent / "s_00004.h5")
        files._RESOLVE_CACHE.clear()
        assert files.image_path_for_master(master).samefile(master.parent / "s_00004.h5")


def test_master_found_from_an_image_in_a_parallel_tree():
    with Sandbox() as d:
        master = write_master(d / "exp" / "s_00005_master.hdf")
        image = write_image(d / "lambda" / "exp" / "s_00005.h5")
        assert files.master_path_for_image(image).samefile(master)


def test_rois_from_the_master_baseline():
    with Sandbox() as d:
        master = write_master(d / "s_00006_master.hdf", baseline={
            "lambda250k_rois_roi1_min_x": 286, "lambda250k_rois_roi1_min_y": 335,
            "lambda250k_rois_roi1_size_x": 50, "lambda250k_rois_roi1_size_y": 5,
            "lambda250k_rois_roi2_min_x": 111, "lambda250k_rois_roi2_min_y": 333,
            "lambda250k_rois_roi2_size_x": 400, "lambda250k_rois_roi2_size_y": 10,
        })
        check("files/master_rois", files.read_master_rois(master))


def test_example_pair_resolves():
    if not have_example():
        return
    files._RESOLVE_CACHE.clear()
    assert files.image_path_for_master(EXAMPLE_MASTER).name == EXAMPLE_IMAGE.name
    assert files.master_path_for_image(EXAMPLE_IMAGE).name == EXAMPLE_MASTER.name


def test_scan_tags_gain_their_series_only_when_numbers_collide():
    from pathlib import Path
    alone = [Path("tise2_2_00006_master.hdf"), Path("tise2_2_00007_master.hdf")]
    tags = files.scan_tags(alone)
    assert sorted(tags.values()) == ["00006", "00007"]   # the number is enough
    # two series in one directory sharing a number: both gain the series
    clash = alone + [Path("sio_sao_sto_00007_master.hdf")]
    tags = files.scan_tags(clash)
    assert tags[str(Path("tise2_2_00007_master.hdf"))] == "tise2_2 00007"
    assert tags[str(Path("sio_sao_sto_00007_master.hdf"))] == "sio_sao_sto 00007"
