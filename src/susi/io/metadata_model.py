import platform
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

import susi.io.utils as io_utils
from susi.io.app_settings import AppSettings

_app_settings = AppSettings()


def does_filename_have_extension(filename: str, extension: str) -> None:
    if not filename.endswith(extension):
        raise ValueError(f"File {filename} needs to have extension {extension}.")
    return None


class SimulationMetaData(BaseModel):
    model_config = ConfigDict(
        validate_default=True,  # validate default values
    )

    metadata_schema_version: int = 1

    metadata_output_filename: str = Field(
        default="metadata.json",
        description="Name of the output metadata file. Needs to be JSON. It will be stored inside the `experiment_folder_path`.",
    )
    parameter_output_filename: str = Field(
        default="params.json",
        description="Name of the output parameters file. Needs to be JSON. It will be stored inside the `experiment_folder_path`.",
    )
    netcdf_output_filename: str = Field(
        default="susi.nc",
        description="Name of the output netcdf file. Needs to have extension '.nc'. It will be stored inside the `experiment_folder_path`.",
    )

    timestamp_start: str = Field(
        default_factory=io_utils.generate_current_datetime_stamp,
        description="Initial timestamp. (Technically, it takes the timestamp at the time the current class is created).",
    )

    timestamp_end: str = Field(
        init=False,
        default=str(),
        description="Final timestamp, recorded when the metadata dumping is done.",
    )

    git_commit_hash: str = Field(
        init=False,
        default_factory=io_utils.get_git_revision_short_hash,
        description="Git commit identifier.",
    )

    host_info: str = Field(
        init=False,
        default=str(platform.uname()),
        description="Info about who ran the simulations.",
    )

    experiment_id: str = Field(
        default="",
        description="Uniquely identifies each simulation experiment.",
    )

    @field_validator("metadata_output_filename", "parameter_output_filename")
    @classmethod
    def check_json_extension(cls, value: str) -> str:
        does_filename_have_extension(filename=value, extension=".json")
        return value

    @field_validator("netcdf_output_filename")
    @classmethod
    def check_netcdf_extension(cls, value: str) -> str:
        does_filename_have_extension(filename=value, extension=".nc")
        return value

    @model_validator(mode="after")
    def set_experiment_id(self) -> "SimulationMetaData":
        """Generate experiment_id from timestamp_start if not already set."""
        if not self.experiment_id:
            self.experiment_id = io_utils.generate_experiment_ID(
                datetime_stamp=self.timestamp_start
            )
        return self

    @computed_field
    @property
    def experiment_folder_path(self) -> Path:
        """
        Directory Path for all simulation results: metadata, parameters, netcdf file.
        The name of the folder is the experiment_id.
        """
        return _app_settings.output_folder.joinpath(self.experiment_id)

    @computed_field
    @property
    def metadata_output_filepath(self) -> Path:
        return self.experiment_folder_path.joinpath(self.metadata_output_filename)

    @computed_field
    @property
    def parameter_output_filepath(self) -> Path:
        return self.experiment_folder_path.joinpath(self.parameter_output_filename)

    @computed_field
    @property
    def netcdf_output_filepath(self) -> Path:
        return self.experiment_folder_path.joinpath(self.netcdf_output_filename)

    def create_output_folder(self) -> None:
        io_utils.create_folder(path=self.experiment_folder_path)
        return None

    def record_end_timestamp(self) -> None:
        self.timestamp_end = io_utils.generate_current_datetime_stamp()
        return None

    def dump_json_to_file(self) -> None:
        with open(self.metadata_output_filepath, "w") as f:
            f.write(self.model_dump_json())
