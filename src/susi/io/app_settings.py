from pathlib import Path

from pydantic import BaseModel, DirectoryPath, Field

import susi.io.utils as io_utils


class AppSettings(BaseModel):
    project_root_path: DirectoryPath = io_utils.get_project_root()
    input_folder: DirectoryPath = project_root_path / Path("src/inputs/")
    output_folder: DirectoryPath = Field(
        default=(project_root_path / Path("outputs/")),
        description="Root folder where all outputs go.",
    )

    metadata_store_filename: str = "metadata.json"
    parameter_store_filename: str = "params.json"
