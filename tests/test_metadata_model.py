from pydantic import ValidationError
from pathlib import Path
import pytest

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
