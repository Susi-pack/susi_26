from typing import NewType
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
import netCDF4


import susi.io.utils as io_utils


# %% dataclasses

NetcdfVariablePath = NewType(
    "NetcdfVariablePath", str
)  # Example: "/balance/K/fertilization_release"
ScenarioName = NewType("ScenarioName", str)  # Examples: "DNM", "default", "fertilized"
TargetVariableDict = NewType("TargetVariableDict", dict[NetcdfVariablePath, float])


@dataclass
class NetcdfVariableInfo:
    """
    Contains all information of a variable from the Susi netcdf file,
    except its values
    """

    path: NetcdfVariablePath
    name: str  # Example: "fertilization_release"
    dimension_names: tuple[str]
    shape: tuple[int]
    units: str  # Unit description


@dataclass
class NetcdfVariableValue:
    """
    Contains the id for the netcdf variable (in path),
    and its value.
    """

    path: NetcdfVariablePath
    value: np.ndarray


@dataclass
class ScenarioArrayData:
    """Container class to preserve both arrays and names"""

    arrays: list[np.ndarray]  # One array per stand
    scenario_names: list[list[str]]  # Names for each scenario in each stand


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


def list_all_netcdf_variables(netcdf_filepath: Path) -> list[NetcdfVariableInfo]:
    """
    Explore the structure of a NetCDF file and return info on all variables.
    """

    def _recursive_group(group, prefix=""):
        variables: list[NetcdfVariableInfo] = []

        for var_name, var in group.variables.items():
            variable = NetcdfVariableInfo(
                name=var_name,
                path=NetcdfVariablePath(
                    f"{prefix}/{var_name}" if prefix else f"/{var_name}"
                ),
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
    netcdf_filepath: Path, variables: list[NetcdfVariableInfo]
) -> list[NetcdfVariableValue]:
    """
    Read the values of a list of NetcdfVariables from a NetCDF file.
    """
    variables_values: list[NetcdfVariableValue] = []

    with netCDF4.Dataset(netcdf_filepath, "r") as nc:
        for variable in variables:
            # split path and remove empty strings
            var_path_parts = [p for p in variable.path.split("/") if p]

            var = _get_variable_by_path(nc, var_path_parts)

            variables_values.append(
                NetcdfVariableValue(path=variable.path, value=var[:])
            )

    return variables_values


def choose_netcdf_vars_by_path(
    paths: tuple[NetcdfVariablePath],
    all_variables: list[NetcdfVariableInfo],
) -> list[NetcdfVariableInfo]:
    """
    Filters variables by path.
    I tried to filtering by name, but the variable names are not unique: they depend on the path
    E.g., the name "volume" has several different variables
    """
    variables = []
    for var_info in all_variables:
        if var_info.path in paths:
            variables.append(var_info)

    # if len(variables) != len(names):
    #     raise ValueError("Could not locate all variables by name")

    return variables


def read_netcdf_variable_values_for_stand(
    chosen_variables: list[NetcdfVariableInfo],
    metadata_df: pd.DataFrame,
) -> dict[ScenarioName, NetcdfVariableValue]:
    # Netcdf data is saved in a dictionary where the experiment folder path is the key.
    data = {}
    for _, experiment_info in metadata_df.iterrows():
        data[experiment_info["experiment_id"]] = (
            read_value_several_variables_from_single_file(
                netcdf_filepath=experiment_info["netcdf_output_filepath"],
                variables=chosen_variables,
            )
        )

    return data


def read_chosen_variables_from_netcdf_by_stands_and_scenarios(
    chosen_vars: list[NetcdfVariableInfo],
    metadata_by_stand: list[pd.DataFrame],
) -> list[dict[ScenarioName, NetcdfVariableValue]]:
    """
    Return nested structure:
        list <- dimension of number of stands
            dict <- dimension of scenarios for each stand
    """
    chosen_variables_by_stand_and_scenario = []

    for n_stand, metadata_df in enumerate(metadata_by_stand):
        chosen_variables_by_stand_and_scenario.append(
            read_netcdf_variable_values_for_stand(
                metadata_df=metadata_df, chosen_variables=chosen_vars
            )
        )
    return chosen_variables_by_stand_and_scenario
