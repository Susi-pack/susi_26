from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import netCDF4

import susi.io.utils as io_utils


# %% dataclasses
@dataclass
class NetcdfVariable:
    path: str  # Example: "/balance/K/fertilization_release"
    name: str  # Example: "fertilization_release"
    dimension_names: tuple[str]
    shape: tuple[int]
    units: str  # Unit description


# %% Functions
def list_subdirectories(path: Path):
    return (x for x in path.iterdir() if x.is_dir())


def list_variable_absolute_paths(group: netCDF4.Dataset, path: str = "/") -> list[str]:
    vars_with_paths = []
    for v in group.variables:
        vars_with_paths.append(f"{path}/{v}".replace("//", "/"))

    for name, subgroup in group.groups.items():
        subpath = f"{path}/{name}".replace("//", "/")
        vars_with_paths.extend(list_variable_absolute_paths(subgroup, subpath))

    return vars_with_paths


def get_var_by_path(group: netCDF4.Dataset, var_path: str) -> dict:
    parts = var_path.strip("/").split("/")
    group = group
    for p in parts[:-1]:  # navigate to the group
        group = group.groups[p]
    return group.variables[parts[-1]][:]  # read the data


def _load_single_experiment_metadatas(
    experiment_folderpath: Path,
    metadata_filename: str = "metadata.json",
    params_filename: str = "params.json",
) -> pd.DataFrame:
    """
    Reads metadata and parameter info from json files.
    Returns dict of all json values.
    """
    metadata_filepath = experiment_folderpath.joinpath(metadata_filename)
    params_filepath = experiment_folderpath.joinpath(params_filename)

    metadata, params = map(
        io_utils.read_json_file, [metadata_filepath, params_filepath]
    )
    return pd.json_normalize(metadata | params)


def coerce_datetime_format(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = ["timestamp_start", "timestamp_end"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
    return df


def modify_after_load(
    df: pd.DataFrame, set_experiment_id_as_index: bool = False
) -> pd.DataFrame:
    df = df.copy()

    # set datetime formats
    df = coerce_datetime_format(df)

    # sort by starting date first
    df = df.sort_values(by="timestamp_start", ignore_index=True, ascending=False)

    # set experiment_id as index
    if set_experiment_id_as_index:
        df = df.set_index(keys="experiment_id")

    return df


def load_all_metadatas_from_single_folder(folder: Path) -> pd.DataFrame:
    experiment_folderpaths = list_subdirectories(folder)
    df = pd.concat(
        [
            _load_single_experiment_metadatas(exp_fpath)
            for exp_fpath in experiment_folderpaths
        ]
    )

    return modify_after_load(df)


def load_all_metadatas_from_folders(folders: list[Path]) -> list[pd.DataFrame]:
    all_metadatas = []
    for folder in folders:
        all_metadatas.append(load_all_metadatas_from_single_folder(folder))
    return all_metadatas


def list_all_netcdf_variables(netcdf_filepath: Path) -> list[NetcdfVariable]:
    """
    Explore the structure of a NetCDF file and return info on all variables.
    """

    def _recursive_group(group, prefix=""):
        variables: list[NetcdfVariable] = []

        for var_name, var in group.variables.items():
            variable = NetcdfVariable(
                name=var_name,
                path=f"{prefix}/{var_name}" if prefix else f"/{var_name}",
                dimension_names=var.dimensions,
                shape=var.shape,
                units=getattr(var, "units"),
            )
            variables.append(variable)

        # Recurse into subgroups
        for sub_name, sub_group in group.groups.items():
            sub_prefix = f"{prefix}/{sub_name}" if prefix else f"/{sub_name}"
            variables.extend(_recursive_group(sub_group, sub_prefix))

        return variables

    with netCDF4.Dataset(netcdf_filepath, "r") as nc:
        return _recursive_group(nc)


def _get_variable_by_path(group, path_parts):
    """Recursively navigate groups to find the variable."""
    if len(path_parts) == 1:
        return group.variables[path_parts[0]]
    else:
        sub_group_name = path_parts[0]
        return _get_variable_by_path(group.groups[sub_group_name], path_parts[1:])


def read_value_several_variables_from_single_file(
    netcdf_filepath: Path, variables: list[NetcdfVariable]
) -> list[np.ndarray]:
    """
    Read the values of a list of NetcdfVariables from a NetCDF file.
    """
    variables_values: list[np.ndarray] = []

    with netCDF4.Dataset(netcdf_filepath, "r") as nc:
        for variable in variables:
            # split path and remove empty strings
            var_path_parts = [p for p in variable.path.split("/") if p]

            var = _get_variable_by_path(nc, var_path_parts)

            variables_values.append(var[:])

    return variables_values
