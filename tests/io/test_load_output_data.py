import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from susi.io.load_output_data import (
    list_stand_folders,
    list_subdirectories_sorted,
    netcdf_filepath_of_first_scenario,
)


class TestListSubdirectoriesSorted:
    def test_sorts_unpadded_numeric_suffixes_naturally(self):
        # Lexicographic sort would put "stand_10"/"stand_11" before "stand_2"
        # (see #217): a project with more than 9 stands must still list them
        # in numeric order.
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            names = ["stand_1", "stand_2", "stand_10", "stand_11", "stand_21"]
            for name in names:
                (output_dir / name).mkdir()

            result = list_subdirectories_sorted(output_dir)

            assert [p.name for p in result] == [
                "stand_1",
                "stand_2",
                "stand_10",
                "stand_11",
                "stand_21",
            ]

    def test_ignores_files(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "stand_1").mkdir()
            (output_dir / "not_a_stand.txt").touch()

            result = list_subdirectories_sorted(output_dir)

            assert [p.name for p in result] == ["stand_1"]


class TestListStandFolders:
    def test_lists_a_runs_stand_folders_in_natural_order(self, tmp_path):
        # Bare numeric stand IDs, created out of order: the listing must come
        # back 1, 2, 10 whatever order the filesystem hands the folders over
        # in, and whatever a lexicographic sort would make of "10" and "2".
        for stand_id in ["10", "2", "1"]:
            (tmp_path / stand_id).mkdir()

        result = list_stand_folders(run_dirpath=tmp_path)

        assert [p.name for p in result] == ["1", "2", "10"]

    def test_every_folder_of_a_run_is_a_stand(self, tmp_path):
        # Nothing is filtered out by name: a stray folder is listed like any
        # other, so that it is reported by whatever reads it as a stand rather
        # than silently dropped. Plain files are not folders, so not stands.
        (tmp_path / "1").mkdir()
        (tmp_path / "figures").mkdir()
        (tmp_path / "notes.txt").touch()

        result = list_stand_folders(run_dirpath=tmp_path)

        assert [p.name for p in result] == ["1", "figures"]


def _write_scenario(scenario_dirpath: Path, netcdf_filepath: Path) -> None:
    """A scenario folder with the two jsons a simulation writes."""
    scenario_dirpath.mkdir(parents=True)
    (scenario_dirpath / "metadata.json").write_text(
        json.dumps({"netcdf_output_filepath": str(netcdf_filepath)})
    )
    (scenario_dirpath / "params.json").write_text(json.dumps({}))


class TestNetcdfFilepathOfFirstScenario:
    def test_returns_the_netcdf_of_the_first_scenario_of_the_first_stand(
        self, tmp_path
    ):
        # Stands and scenarios created out of order: "first" is the natural
        # sort of the names, so stand "2" before "10", scenario "1" before "3".
        for stand_id in ["10", "2"]:
            for scenario_id in ["3", "1"]:
                _write_scenario(
                    scenario_dirpath=tmp_path / stand_id / scenario_id,
                    netcdf_filepath=Path(f"/somewhere/{stand_id}_{scenario_id}.nc"),
                )

        result = netcdf_filepath_of_first_scenario(run_dirpath=tmp_path)

        assert result == Path("/somewhere/2_1.nc")

    def test_a_run_with_no_stands_is_reported_by_name(self, tmp_path):
        # An empty run folder (e.g. a run whose simulations were never
        # written) must say which folder it is, not raise a bare IndexError.
        with pytest.raises(ValueError, match=str(tmp_path)):
            netcdf_filepath_of_first_scenario(run_dirpath=tmp_path)

    def test_a_stand_with_no_scenarios_is_reported_by_name(self, tmp_path):
        (tmp_path / "1").mkdir()

        with pytest.raises(ValueError, match=str(tmp_path / "1")):
            netcdf_filepath_of_first_scenario(run_dirpath=tmp_path)
