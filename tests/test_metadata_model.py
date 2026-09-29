import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from susi.io.app_settings import PROJECTS_ROOT_ENV_VAR
from susi.io.metadata_model import SimulationMetaData


def test_project_dir_must_exist(tmp_path):
    with pytest.raises(ValidationError):
        SimulationMetaData(
            project_dir=tmp_path / "non-existent-project",
            run_id="lala",
        )


def test_full_path_of_new_simulation_does_not_exist(tmp_project):
    # A run whose folder is already there would overwrite a previous run's
    # results, so it must be refused.
    (tmp_project / "outputs" / "my_run").mkdir()

    with pytest.raises(ValidationError, match="already exists"):
        SimulationMetaData(project_dir=tmp_project, run_id="my_run")


def test_single_run_folder_path(tmp_project):
    metadata = SimulationMetaData(project_dir=tmp_project, run_id="my_run")

    assert metadata.simulation_folder_path == tmp_project / "outputs" / "my_run"


def test_batch_run_folder_path(tmp_project):
    metadata = SimulationMetaData(
        project_dir=tmp_project,
        run_id="my_run",
        stand_id="stand_A",
        scenario_id="scenario_1",
    )

    assert (
        metadata.simulation_folder_path
        == tmp_project / "outputs" / "my_run" / "stand_A" / "scenario_1"
    )


def test_only_stand_id_raises(tmp_project):
    with pytest.raises(ValidationError, match="stand_id and scenario_id"):
        SimulationMetaData(
            project_dir=tmp_project,
            run_id="my_run",
            stand_id="stand_A",
        )


def test_only_scenario_id_raises(tmp_project):
    with pytest.raises(ValidationError, match="stand_id and scenario_id"):
        SimulationMetaData(
            project_dir=tmp_project,
            run_id="my_run",
            scenario_id="scenario_1",
        )


def test_stand_id_with_slash_raises(tmp_project):
    with pytest.raises(ValidationError, match="path separators"):
        SimulationMetaData(
            project_dir=tmp_project,
            run_id="my_run",
            stand_id="stand/A",
            scenario_id="scenario_1",
        )


def test_scenario_id_with_backslash_raises(tmp_project):
    with pytest.raises(ValidationError, match="path separators"):
        SimulationMetaData(
            project_dir=tmp_project,
            run_id="my_run",
            stand_id="stand_A",
            scenario_id="scenario\\1",
        )


# %% A project is identified by its folder


def test_project_id_is_the_folder_name(tmp_project):
    metadata = SimulationMetaData(project_dir=tmp_project, run_id="my_run")

    assert metadata.project_id == "my_project"


def test_project_outside_the_projects_root_writes_into_its_own_outputs(
    tmp_path, monkeypatch
):
    # The point of naming the folder rather than a name to look up: an
    # example or testing project in the checkout runs where it sits, and the
    # projects root has no say in where its outputs go.
    monkeypatch.setenv(PROJECTS_ROOT_ENV_VAR, str(tmp_path / "does_not_exist"))
    project_dir = tmp_path / "example_projects" / "paroninkorpi"
    (project_dir / "outputs").mkdir(parents=True)

    metadata = SimulationMetaData(project_dir=project_dir, run_id="my_run")

    assert metadata.simulation_folder_path == project_dir / "outputs" / "my_run"


def test_project_id_and_project_dir_are_written_to_metadata_json(tmp_project):
    # The analysis side reads metadata.json as a plain dict and groups by
    # `project_id`, so it must stay in the output even though it is no
    # longer an input.
    metadata = SimulationMetaData(project_dir=tmp_project, run_id="my_run")
    metadata.simulation_folder_path.mkdir()

    metadata.dump_json_to_file()
    written = json.loads(metadata.metadata_output_filepath.read_text())

    assert written["project_id"] == "my_project"
    assert Path(written["project_dir"]) == tmp_project
    assert Path(written["project_dir"]).is_absolute()
    assert "parent_output_folder" not in written


def test_project_dir_is_made_absolute(tmp_project, monkeypatch):
    # Serialized next to the absolute *_filepath fields, and a relative path
    # in metadata.json would mean nothing once read from elsewhere.
    monkeypatch.chdir(tmp_project.parent)

    metadata = SimulationMetaData(project_dir=Path("my_project"), run_id="my_run")

    assert metadata.project_dir == tmp_project
    assert metadata.project_id == "my_project"


def test_project_without_outputs_folder_raises(tmp_path):
    # A folder that is not a project at all: no inputs/, no outputs/.
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()

    with pytest.raises(ValidationError, match="outputs/"):
        SimulationMetaData(project_dir=project_dir, run_id="my_run")


def test_project_id_is_not_an_input(tmp_project):
    with pytest.raises(ValidationError):
        SimulationMetaData(
            project_dir=tmp_project,
            project_id="my_project",  # ty: ignore[unknown-argument]
            run_id="my_run",
        )


def test_parent_output_folder_is_gone(tmp_project, tmp_path):
    # There is one way to say where outputs go: the project's folder.
    with pytest.raises(ValidationError, match="parent_output_folder"):
        SimulationMetaData(
            project_dir=tmp_project,
            run_id="my_run",
            parent_output_folder=tmp_path,  # ty: ignore[unknown-argument]
        )


# %% A bad SUSI_PROJECTS_ROOT never breaks the import


def test_bad_projects_root_does_not_break_the_import(tmp_path):
    """
    Importing the module must not touch the filesystem.

    Run in a subprocess rather than with `importlib.reload`, because a reload
    would leave every other module in this test session holding a stale
    `SimulationMetaData` class.
    """
    env = dict(os.environ, **{PROJECTS_ROOT_ENV_VAR: str(tmp_path / "does_not_exist")})

    result = subprocess.run(
        [sys.executable, "-c", "import susi.io.metadata_model"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
