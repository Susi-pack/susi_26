from pathlib import Path
from tempfile import TemporaryDirectory

from susi.io.load_output_data import list_subdirectories_sorted


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
