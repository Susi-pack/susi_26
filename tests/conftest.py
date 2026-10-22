from pathlib import Path

import pytest

from susi.io.project_layout import inputs_dir_for_project, outputs_dir_for_project

# %% Test groups (docs/internal/running_tests.md)


def pytest_addoption(parser):
    parser.addoption(
        "--all",
        action="store_true",
        help="Run every test group: fast, slow and golden.",
    )


def pytest_configure(config):
    # pyproject's addopts deselects the slow and golden groups with
    # `-m "not slow and not golden"`. --all undoes that.
    if config.getoption("--all"):
        config.option.markexpr = ""


@pytest.fixture
def tmp_project(tmp_path) -> Path:
    """
    A throwaway project folder, `tmp_path/my_project/{inputs,outputs}`.

    What a test hands `SimulationMetaData(project_dir=...)` so the run writes
    through the same code path a real one does -- into the project's own
    `outputs/` -- rather than into the repo or a user's projects root. There
    is no way to point a run anywhere that is not a project, so tests build
    one.
    """
    project_dir = tmp_path / "my_project"
    inputs_dir_for_project(project_dir).mkdir(parents=True)
    outputs_dir_for_project(project_dir).mkdir()
    return project_dir
