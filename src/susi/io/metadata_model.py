from pyexpat import model
import platform
from pathlib import Path

from functools import cached_property
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
    NewPath,
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
        frozen=True,
        default="metadata.json",
        description="Name of the output metadata file. Needs to be JSON. It will be stored inside the `experiment_folder_path`.",
    )
    parameter_output_filename: str = Field(
        frozen=True,
        default="params.json",
        description="Name of the output parameters file. Needs to be JSON. It will be stored inside the `experiment_folder_path`.",
    )
    netcdf_output_filename: str = Field(
        frozen=True,
        default="susi.nc",
        description="Name of the output netcdf file. Needs to have extension '.nc'. It will be stored inside the `experiment_folder_path`.",
    )

    timestamp_start: str = Field(
        frozen=True,
        default_factory=io_utils.generate_current_datetime_stamp,
        description="Initial timestamp. (Technically, it takes the timestamp at the time the current class is created).",
    )

    timestamp_end: str = Field(
        init=False,
        default=str(),
        description="Final timestamp, recorded when the metadata dumping is done.",
    )

    git_commit_hash: str = Field(
        frozen=True,
        init=False,
        default_factory=io_utils.get_git_revision_short_hash,
        description="Git commit identifier.",
    )

    host_info: str = Field(
        frozen=True,
        init=False,
        default=str(platform.uname()),
        description="Info about who ran the simulations.",
    )

    experiment_name: str = Field(
        default="",
        description="A name for the experiment. It defaults to the empty string. This field will be used to create the folder where Susi outputs are stored. Susi automatically attaches the starting date/time and a random string to each folder name to avoid name collisions.",
    )

    experiment_folder_path: Path | None = Field(
        default=None,
        description="Directory Path for all simulation results: metadata, parameters, and netcdf file. If None (default), the folder name is the same as experiment_id.",
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

    @computed_field
    @cached_property  # computes experiment_id only once. Otherwise, we get different random strings in each call!
    def experiment_id(self) -> str:
        """
        A string that uniquely identifies each simulation experiment. If None, it is computed as a combination of the starting timestamp and a random number to avoid naame collisions.
        """
        return io_utils.generate_experiment_ID(
            experiment_name=self.experiment_name, datetime_stamp=self.timestamp_start
        )

    @model_validator(mode="after")
    def set_experiment_folder_path(self):
        if self.experiment_folder_path is None:
            self.experiment_folder_path = _app_settings.output_folder.joinpath(
                self.experiment_id
            )
        return self

    @computed_field
    @property
    def metadata_output_filepath(self) -> Path:
        assert self.experiment_folder_path is not None
        return self.experiment_folder_path.joinpath(self.metadata_output_filename)

    @computed_field
    @property
    def parameter_output_filepath(self) -> Path:
        assert self.experiment_folder_path is not None
        return self.experiment_folder_path.joinpath(self.parameter_output_filename)

    @computed_field
    @property
    def netcdf_output_filepath(self) -> Path:
        assert self.experiment_folder_path is not None
        return self.experiment_folder_path.joinpath(self.netcdf_output_filename)

    def record_end_timestamp(self) -> None:
        self.timestamp_end = io_utils.generate_current_datetime_stamp()
        return None

    def dump_json_to_file(self) -> None:
        with open(self.metadata_output_filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
