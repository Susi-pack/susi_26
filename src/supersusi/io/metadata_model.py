import platform
from pathlib import Path
from typing_extensions import Self
import datetime
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
    NewPath,
    DirectoryPath,
)

import supersusi.io.utils as io_utils
from supersusi.io.app_settings import AppSettings

_app_settings = AppSettings()


def does_filename_have_extension(filename: str, extension: str) -> None:
    if not filename.endswith(extension):
        raise ValueError(f"File {filename} needs to have extension {extension}.")
    return None


class SimulationMetaData(BaseModel):
    model_config = ConfigDict(
        validate_default=True,  # validate default values
        extra="forbid",
    )

    experiment_id: str = Field(
        description="A name that identifies the simulation experiment. This will be the name of the folder where Susi outputs are stored. In single Susi runs `experiment_id` becomes the root folder of the experiment. The full path of `parent_output_folder` / `experiment_id` must not exist: otherwise we would be overwriting previous simulations' results.",
    )

    stand_id: str | None = Field(
        description="Required when running multiple SUSI simulations with `MultipleSusi`, it gives the name to the first hierarchy of SUSI output folders (`scenario_id` is the second one). Not needed for single simulations.",
        default=None,
    )

    scenario_id: str | None = Field(
        description="Required when running multiple SUSI simulations with `MultipleSusi`, it gives the name to the second hierarchy of SUSI output folders (`stand_id` is the first one). Not needed for single simulations.",
        default=None,
    )

    parent_output_folder: DirectoryPath = Field(
        default=_app_settings.output_folder,
        description="The directory where the output of the Susi simulation will be stored.",
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

    timestamp_start: datetime.datetime = Field(
        frozen=True,
        default_factory=datetime.datetime.now,
        description="Initial timestamp. (Technically, it takes the timestamp at the time the current class is created).",
    )

    timestamp_end: datetime.datetime = Field(
        init=False,
        default_factory=datetime.datetime.now,
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
    @property
    def experiment_folder_path(self) -> NewPath:
        """
        Directory Path for Susi simulation results: metadata, parameters, and netcdf file.
        If stand_id and scenario_id are not given it results in `parent_output_folder`/`experiment_id`.
        If stand_id and scenario_id are give it results in `parent_output_folder/experiment_id/stand_id/scenario_id`
        The resulting path must not exist.
        """

        base = self.parent_output_folder / self.experiment_id
        if self.stand_id and self.scenario_id:
            return base / self.stand_id / self.scenario_id
        return base

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

    @field_validator("stand_id", "scenario_id")
    def validate_folder_names(cls, v):
        if v is None:
            return v

        if "/" in v or "\\" in v:
            raise ValueError("Folder names must not contain path separators '/' or ''.")

        return v

    @model_validator(mode="after")
    def check_experiment_folder_path_does_not_exist(self) -> Self:
        if self.experiment_folder_path.exists():
            raise ValueError(
                "A file or a directory with the same path as the new Susi experiment folder already exists. The new path must not exist."
            )
        return self

    @model_validator(mode="after")
    def validate_stand_scenario_pair(self):
        if (self.stand_id is None) != (self.scenario_id is None):
            raise ValueError(
                "stand_id and scenario_id must either both be set or both be None."
            )
        return self

    def record_end_timestamp(self) -> None:
        self.timestamp_end = datetime.datetime.now()
        return None

    def dump_json_to_file(self) -> None:
        with open(self.metadata_output_filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
