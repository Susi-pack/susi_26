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

import susi.io.utils as io_utils
from susi.io.project_layout import outputs_dir_for_project, project_dir


def does_filename_have_extension(filename: str, extension: str) -> None:
    if not filename.endswith(extension):
        raise ValueError(f"File {filename} needs to have extension {extension}.")
    return None


class SimulationMetaData(BaseModel):
    model_config = ConfigDict(
        validate_default=True,  # validate default values
        extra="forbid",
    )

    project_id: str = Field(
        description="The project this run belongs to: the user-chosen name grouping one simulation study, e.g. 'paroninkorpi'. It names the folder `projects/<project_id>/` under the projects root, and it is what `parent_output_folder` is derived from when the caller does not supply one.",
    )

    run_id: str = Field(
        description="A name that identifies this particular run of the project. It becomes the first output folder level under the project's `outputs/`, so that two runs of the same project -- before and after a parameter change -- sit side by side instead of colliding. The full `experiment_folder_path` must not exist: otherwise we would be overwriting a previous run's results.",
    )

    stand_id: str | None = Field(
        description="Required when running multiple SUSI simulations with `MultipleSusi`, it gives the name to the first hierarchy of SUSI output folders (`scenario_id` is the second one). Not needed for single simulations.",
        default=None,
    )

    scenario_id: str | None = Field(
        description="Required when running multiple SUSI simulations with `MultipleSusi`, it gives the name to the second hierarchy of SUSI output folders (`stand_id` is the first one). Not needed for single simulations.",
        default=None,
    )

    parent_output_folder: DirectoryPath | None = Field(
        default=None,
        description="The directory the run's output folders are created under. Leave it out and `_derive_parent_output_folder` fills it in from `project_id`, as the project's `outputs/` folder -- which is where a run's results belong. Set it explicitly only to write somewhere that is not a project at all, as the golden-file test does.",
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
        If stand_id and scenario_id are not given it results in `parent_output_folder`/`run_id`.
        If stand_id and scenario_id are give it results in `parent_output_folder/run_id/stand_id/scenario_id`
        The resulting path must not exist.
        """

        # `_derive_parent_output_folder` runs before any computed field can be
        # read, so the field is never still None here.
        assert self.parent_output_folder is not None
        base = self.parent_output_folder / self.run_id
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

    # Must stay defined before `check_experiment_folder_path_does_not_exist`:
    # pydantic runs "after" validators in class-body order, and that one reads
    # `experiment_folder_path`, which needs `parent_output_folder` filled in.
    @model_validator(mode="after")
    def _derive_parent_output_folder(self) -> Self:
        """
        Default `parent_output_folder` to the project's own `outputs/` folder.

        This cannot be a field default: pydantic has no way to default one
        field from the value of another, and the folder depends on
        `project_id`.

        `AppSettings()` is constructed here, inside the validator (via
        `project_dir`), rather than once at module level. That way importing
        this module never touches the filesystem, and a mistyped
        `SUSI_PROJECTS_ROOT` fails when you build a simulation -- with a
        message naming the variable -- instead of failing the import of
        `susi.io.metadata_model`.
        """
        if self.parent_output_folder is not None:
            return self

        outputs_dir = outputs_dir_for_project(project_dir(self.project_id))
        if not outputs_dir.is_dir():
            raise ValueError(
                f"Project {self.project_id!r} has no outputs/ folder (expected "
                f"at {outputs_dir}). A project keeps everything its runs "
                "produce in outputs/; create the folder, or pass "
                "`parent_output_folder` to write somewhere outside a project."
            )
        self.parent_output_folder = outputs_dir
        return self

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
