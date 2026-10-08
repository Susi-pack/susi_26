from pathlib import Path
from tempfile import TemporaryDirectory

from susi.io.load_output_data import list_stand_folders, list_subdirectories_sorted


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
