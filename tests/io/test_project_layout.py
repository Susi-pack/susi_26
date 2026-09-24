from pathlib import Path

import pytest

from susi.io.app_settings import PROJECTS_ROOT_ENV_VAR
from susi.io import project_layout


# %% project_dir


def test_project_dir_hangs_the_project_off_the_projects_root(monkeypatch, tmp_path):
    monkeypatch.setenv(PROJECTS_ROOT_ENV_VAR, str(tmp_path))

    assert project_layout.project_dir("paroninkorpi") == tmp_path / "paroninkorpi"


# %% Pure path composition
#
# These take a project dir rather than reading settings, so they need no
# filesystem and no env var at all.


def test_inputs_dir_for_project():
    assert project_layout.inputs_dir_for_project(Path("/p/paroninkorpi")) == Path(
        "/p/paroninkorpi/inputs"
    )


def test_outputs_dir_for_project():
    assert project_layout.outputs_dir_for_project(Path("/p/paroninkorpi")) == Path(
        "/p/paroninkorpi/outputs"
    )


def test_allometry_dir_sits_under_inputs():
    # Allometry CSVs are something the project is simulated *from*, so they
    # belong in inputs/, not next to the runs' results.
    assert project_layout.allometry_dir_for_project(Path("/p/paroninkorpi")) == Path(
        "/p/paroninkorpi/inputs/allometry"
    )


def test_new_growth_allometry_dir_sits_inside_the_allometry_dir():
    # A subfolder, so a listing of allometry/ shows the stand set and the
    # new-growth files apart.
    assert project_layout.new_growth_allometry_dir_for_project(
        Path("/p/paroninkorpi")
    ) == Path("/p/paroninkorpi/inputs/allometry/new_growth")


def test_stand_data_path_sits_beside_the_other_inputs():
    assert project_layout.stand_data_path_for_project(Path("/p/paroninkorpi")) == Path(
        "/p/paroninkorpi/inputs/stand_data.json"
    )


def test_run_dir_nests_under_outputs():
    assert project_layout.run_dir(Path("/p/paroninkorpi"), run_id="2026-09") == Path(
        "/p/paroninkorpi/outputs/2026-09"
    )


# %% require_project_dir


def test_require_project_dir_accepts_a_project_with_inputs(tmp_path):
    (tmp_path / "inputs").mkdir()

    project_layout.require_project_dir(tmp_path)


def test_missing_project_folder_names_the_env_var(tmp_path):
    # The usual cause is a typo or a SUSI_PROJECTS_ROOT pointing at the
    # wrong filesystem, so the error has to name the knob.
    with pytest.raises(FileNotFoundError, match=PROJECTS_ROOT_ENV_VAR):
        project_layout.require_project_dir(tmp_path / "no_such_project")


def test_project_folder_without_inputs_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="inputs/"):
        project_layout.require_project_dir(tmp_path)
