from pydantic import ValidationError
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory

from susi.io.app_settings import AppSettings
from susi.io.metadata_model import SimulationMetaData


def test_parent_output_folder_exists():
    with pytest.raises(ValidationError):
        SimulationMetaData(
            experiment_id="lala", parent_output_folder=Path("non-existent-folder-path")
        )


def test_full_path_of_new_experiment_does_not_exist():
    # Attempts to use the folder /susi_26/tests (which obviously exists) as a
    # new folder for a Susi experiment output.
    # This is obviously a very bad idea and should fail.
    with pytest.raises(ValidationError):
        SimulationMetaData(
            parent_output_folder=AppSettings().project_root_path, experiment_id="tests"
        )


def test_single_run_folder_path():
    with TemporaryDirectory() as tmpdir:
        metadata = SimulationMetaData(
            experiment_id="my_experiment",
            parent_output_folder=Path(tmpdir),
        )
        assert metadata.experiment_folder_path == Path(tmpdir) / "my_experiment"


def test_batch_run_folder_path():
    with TemporaryDirectory() as tmpdir:
        metadata = SimulationMetaData(
            experiment_id="my_experiment",
            parent_output_folder=Path(tmpdir),
            stand_id="stand_A",
            scenario_id="scenario_1",
        )
        assert (
            metadata.experiment_folder_path
            == Path(tmpdir) / "my_experiment" / "stand_A" / "scenario_1"
        )


def test_only_stand_id_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="stand_id and scenario_id"):
            SimulationMetaData(
                experiment_id="my_experiment",
                parent_output_folder=Path(tmpdir),
                stand_id="stand_A",
            )


def test_only_scenario_id_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="stand_id and scenario_id"):
            SimulationMetaData(
                experiment_id="my_experiment",
                parent_output_folder=Path(tmpdir),
                scenario_id="scenario_1",
            )


def test_stand_id_with_slash_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="path separators"):
            SimulationMetaData(
                experiment_id="my_experiment",
                parent_output_folder=Path(tmpdir),
                stand_id="stand/A",
                scenario_id="scenario_1",
            )


def test_scenario_id_with_backslash_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="path separators"):
            SimulationMetaData(
                experiment_id="my_experiment",
                parent_output_folder=Path(tmpdir),
                stand_id="stand_A",
                scenario_id="scenario\\1",
            )
