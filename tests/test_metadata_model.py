import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

import susi.io.utils as io_utils
from susi.io.app_settings import PROJECTS_ROOT_ENV_VAR
from susi.io.metadata_model import SimulationMetaData


@pytest.fixture
def projects_root(tmp_path, monkeypatch) -> Path:
    """
    An empty projects root, with `SUSI_PROJECTS_ROOT` pointed at it.

    Tests that let `parent_output_folder` be derived need a projects root they
    can actually create folders in; the repo's own `projects/` is user data.
    """
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setenv(PROJECTS_ROOT_ENV_VAR, str(root))
    return root


def make_project(projects_root: Path, project_id: str) -> Path:
    """Create `<projects_root>/<project_id>/{inputs,outputs}` and return it."""
    project_dir = projects_root / project_id
    (project_dir / "inputs").mkdir(parents=True)
    (project_dir / "outputs").mkdir(parents=True)
    return project_dir


def test_parent_output_folder_exists():
    with pytest.raises(ValidationError):
        SimulationMetaData(
            project_id="lala",
            run_id="lala",
            parent_output_folder=Path("non-existent-folder-path"),
        )


def test_full_path_of_new_simulation_does_not_exist():
    # Attempts to use the folder /susi_26/tests (which obviously exists) as a
    # new folder for a Susi simulation output.
    # This is obviously a very bad idea and should fail.
    with pytest.raises(ValidationError):
        SimulationMetaData(
            parent_output_folder=io_utils.repo_root(),
            project_id="my_project",
            run_id="tests",
        )


def test_single_run_folder_path():
    with TemporaryDirectory() as tmpdir:
        metadata = SimulationMetaData(
            project_id="my_project",
            run_id="my_run",
            parent_output_folder=Path(tmpdir),
        )
        assert metadata.simulation_folder_path == Path(tmpdir) / "my_run"


def test_batch_run_folder_path():
    with TemporaryDirectory() as tmpdir:
        metadata = SimulationMetaData(
            project_id="my_project",
            run_id="my_run",
            parent_output_folder=Path(tmpdir),
            stand_id="stand_A",
            scenario_id="scenario_1",
        )
        assert (
            metadata.simulation_folder_path
            == Path(tmpdir) / "my_run" / "stand_A" / "scenario_1"
        )


def test_only_stand_id_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="stand_id and scenario_id"):
            SimulationMetaData(
                project_id="my_project",
                run_id="my_run",
                parent_output_folder=Path(tmpdir),
                stand_id="stand_A",
            )


def test_only_scenario_id_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="stand_id and scenario_id"):
            SimulationMetaData(
                project_id="my_project",
                run_id="my_run",
                parent_output_folder=Path(tmpdir),
                scenario_id="scenario_1",
            )


def test_stand_id_with_slash_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="path separators"):
            SimulationMetaData(
                project_id="my_project",
                run_id="my_run",
                parent_output_folder=Path(tmpdir),
                stand_id="stand/A",
                scenario_id="scenario_1",
            )


def test_scenario_id_with_backslash_raises():
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="path separators"):
            SimulationMetaData(
                project_id="my_project",
                run_id="my_run",
                parent_output_folder=Path(tmpdir),
                stand_id="stand_A",
                scenario_id="scenario\\1",
            )


# %% Deriving parent_output_folder from project_id


def test_parent_output_folder_derived_from_project_id(projects_root):
    project_dir = make_project(projects_root, "my_project")

    metadata = SimulationMetaData(project_id="my_project", run_id="my_run")

    assert metadata.parent_output_folder == project_dir / "outputs"
    assert metadata.simulation_folder_path == project_dir / "outputs" / "my_run"


def test_derived_parent_output_folder_keeps_stand_and_scenario_levels(projects_root):
    project_dir = make_project(projects_root, "my_project")

    metadata = SimulationMetaData(
        project_id="my_project",
        run_id="my_run",
        stand_id="stand_A",
        scenario_id="scenario_1",
    )

    assert (
        metadata.simulation_folder_path
        == project_dir / "outputs" / "my_run" / "stand_A" / "scenario_1"
    )


def test_explicit_parent_output_folder_wins_over_project_id(projects_root, tmp_path):
    make_project(projects_root, "my_project")
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()

    metadata = SimulationMetaData(
        project_id="my_project",
        run_id="my_run",
        parent_output_folder=elsewhere,
    )

    assert metadata.parent_output_folder == elsewhere
    assert metadata.simulation_folder_path == elsewhere / "my_run"


def test_project_without_outputs_folder_raises(projects_root):
    # A folder that is not a project at all: no inputs/, no outputs/.
    (projects_root / "my_project").mkdir()

    with pytest.raises(ValidationError, match="outputs/"):
        SimulationMetaData(project_id="my_project", run_id="my_run")


def test_unknown_project_raises(projects_root):
    with pytest.raises(ValidationError, match="outputs/"):
        SimulationMetaData(project_id="no_such_project", run_id="my_run")


# %% A bad SUSI_PROJECTS_ROOT fails at construction, not at import


def test_bad_projects_root_raises_at_construction(tmp_path, monkeypatch):
    monkeypatch.setenv(PROJECTS_ROOT_ENV_VAR, str(tmp_path / "does_not_exist"))

    with pytest.raises(ValueError, match=PROJECTS_ROOT_ENV_VAR):
        SimulationMetaData(project_id="my_project", run_id="my_run")


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
    )

    assert result.returncode == 0, result.stderr
