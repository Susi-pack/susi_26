from pathlib import Path

from pydantic import BaseModel, DirectoryPath, Field

import susi.io.utils as io_utils


class AppSettings(BaseModel):
    project_root_path: DirectoryPath = io_utils.get_project_root()
    input_folder: DirectoryPath = project_root_path / Path("src/inputs/")
    user_input_folder: DirectoryPath = Field(
        default=(project_root_path / Path("inputs/")),
        description=(
            "Root folder for user-authored, untracked datasets "
            "(e.g. site-specific parameter models, weather files) "
            "used to build SusiParams. Distinct from input_folder, "
            "which holds the tracked, repo-shipped system inputs."
        ),
    )
    output_folder: DirectoryPath = Field(
        default=(project_root_path / Path("outputs/")),
        description="Root folder where all outputs go.",
    )
