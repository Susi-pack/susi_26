from pathlib import Path

from pydantic import DirectoryPath, TypeAdapter

from susi.io.app_settings import AppSettings


class TestUserInputFolder:
    def test_resolves_to_repo_root_inputs_folder(self):
        app_settings = AppSettings()

        assert app_settings.user_input_folder == (
            app_settings.project_root_path / Path("inputs/")
        )

    def test_validates_as_directory_path(self):
        app_settings = AppSettings()

        # Round-tripping through DirectoryPath validation confirms the
        # folder actually exists on disk, not just that the field is typed
        # as one.
        validated = TypeAdapter(DirectoryPath).validate_python(
            app_settings.user_input_folder
        )

        assert validated == app_settings.user_input_folder
