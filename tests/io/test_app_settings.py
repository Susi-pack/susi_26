import pytest

from susi.io.app_settings import PROJECTS_ROOT_ENV_VAR, AppSettings
import susi.io.utils as io_utils


class TestProjectsRoot:
    def test_defaults_to_projects_folder_under_the_repo_root(self, monkeypatch):
        # projects/.gitkeep is tracked, so this folder exists on a fresh
        # clone -- which is what makes validate_default=True safe to turn on.
        monkeypatch.delenv(PROJECTS_ROOT_ENV_VAR, raising=False)

        assert AppSettings().projects_root == io_utils.get_project_root() / "projects"

    def test_env_var_overrides_the_default(self, monkeypatch, tmp_path):
        # The override that makes running on a cluster possible: the data
        # lives on a different filesystem from the code.
        elsewhere = tmp_path / "somewhere_else"
        elsewhere.mkdir()
        monkeypatch.setenv(PROJECTS_ROOT_ENV_VAR, str(elsewhere))

        assert AppSettings().projects_root == elsewhere

    def test_nonexistent_env_var_path_fails_loudly_and_by_name(
        self, monkeypatch, tmp_path
    ):
        # A typo'd env var must not degrade into "no projects found later";
        # the error has to name the variable that caused it.
        monkeypatch.setenv(PROJECTS_ROOT_ENV_VAR, str(tmp_path / "does_not_exist"))

        with pytest.raises(ValueError, match=PROJECTS_ROOT_ENV_VAR):
            AppSettings()
