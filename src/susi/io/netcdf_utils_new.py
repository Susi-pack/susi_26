from typing import NewType, Sequence
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import netCDF4

from susi.io.output_data_structure import (
    StandID,
    ScenarioID,
    NetcdfVariableInfo,
    NetcdfVariablePath,
)
import susi.io.utils as io_utils


# %% dataclasses

TargetVariableDict = NewType("TargetVariableDict", dict[NetcdfVariablePath, float])


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
    scenario_names: list[list[ScenarioID]]  # Names for each scenario in each stand


# %% Functions
def list_subdirectories(path: Path) -> list[Path]:
    return [x for x in path.iterdir() if x.is_dir()]


def list_variable_absolute_paths(group: netCDF4.Dataset, path: str = "/") -> list[str]:
    vars_with_paths = []
    for v in group.variables:
        vars_with_paths.append(f"{path}/{v}".replace("//", "/"))

    for name, subgroup in group.groups.items():
        subpath = f"{path}/{name}".replace("//", "/")
        vars_with_paths.extend(list_variable_absolute_paths(subgroup, subpath))

    return vars_with_paths


def read_json_metadatas(
    experiment_folderpath: Path,
    metadata_filename: str = "metadata.json",
    params_filename: str = "params.json",
) -> tuple[dict, dict]:
    metadata_filepath = experiment_folderpath.joinpath(metadata_filename)
    params_filepath = experiment_folderpath.joinpath(params_filename)

    metadata, params = map(
        io_utils.read_json_file, [metadata_filepath, params_filepath]
    )
    return metadata, params


def load_single_experiment_metadatas(
    experiment_folderpath: Path,
    metadata_filename: str = "metadata.json",
    params_filename: str = "params.json",
) -> pd.DataFrame:
    """
    Reads metadata and parameter info from json files.
    Returns dict of all json values.
    """
    metadata, params = read_json_metadatas(
        experiment_folderpath, metadata_filename, params_filename
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


def load_all_metadatas_from_single_stand(folder: Path) -> pd.DataFrame:
    experiment_folderpaths = list_subdirectories(folder)
    df = pd.concat(
        [
            load_single_experiment_metadatas(exp_fpath)
            for exp_fpath in experiment_folderpaths
        ]
    )

    return modify_after_load(df)


def load_all_metadatas_from_stands(folders: list[Path]) -> dict[StandID, pd.DataFrame]:
    all_metadatas = {}
    for folder in folders:
        all_metadatas[StandID(folder.name)] = load_all_metadatas_from_single_stand(
            folder
        )
    return all_metadatas


def list_all_netcdf_variables(
    netcdf_filepath: Path,
) -> dict[NetcdfVariablePath, NetcdfVariableInfo]:
    """
    Explore the structure of a NetCDF file and return info on all variables.
    """

    def _recursive_group(group, prefix=""):
        variables: dict[NetcdfVariablePath, NetcdfVariableInfo] = {}

        for var_name, var in group.variables.items():
            path = NetcdfVariablePath(
                f"{prefix}/{var_name}" if prefix else f"/{var_name}"
            )
            variable = NetcdfVariableInfo(
                name=var_name,
                dimension_names=var.dimensions,
                shape=var.shape,
                units=getattr(var, "units"),
            )
            variables[path] = variable

        # Recurse into subgroups
        for sub_name, sub_group in group.groups.items():
            sub_prefix = f"{prefix}/{sub_name}" if prefix else f"/{sub_name}"
            variables.update(_recursive_group(sub_group, sub_prefix))

        return variables

    with netCDF4.Dataset(netcdf_filepath, "r") as nc:
        return _recursive_group(nc)


def _get_variable_by_path(
    group: netCDF4.Dataset, path_parts: list[str]
) -> netCDF4.Variable:
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
    paths: Sequence[NetcdfVariablePath],
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
) -> dict[ScenarioID, NetcdfVariableValue]:
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
) -> list[dict[ScenarioID, list[NetcdfVariableValue]]]:
    """
    Return nested structure:
        list <- dimension of number of stands
            dict <- dimension of scenarios for each stand
                list <- dimension number of vars
    """
    chosen_variables_by_stand_and_scenario = []

    for n_stand, metadata_df in enumerate(metadata_by_stand):
        chosen_variables_by_stand_and_scenario.append(
            read_netcdf_variable_values_for_stand(
                metadata_df=metadata_df, chosen_variables=chosen_vars
            )
        )
    return chosen_variables_by_stand_and_scenario


def transform_list_of_scenarios_to_optimization_array_structure(
    vars_of_interest_by_stand: list[dict[ScenarioID, TargetVariableDict]],
    n_stands: int,
    target_variable_paths: Sequence[NetcdfVariablePath],
) -> ScenarioArrayData:
    target_variable_arrays: list[np.ndarray] = []

    scenario_names: list[list[ScenarioID]] = []

    n_target_variables = len(target_variable_paths)

    for n_stand in range(n_stands):
        vars_of_interest_single_stand = vars_of_interest_by_stand[n_stand]
        number_of_scenarios = len(vars_of_interest_single_stand)

        # Initialize the array that holds the variables' values
        target_var = np.ones((number_of_scenarios, n_target_variables)) * np.nan
        names_for_stand = []

        for row, (scenario_name, target_variable_dict) in enumerate(
            vars_of_interest_single_stand.items()
        ):
            names_for_stand.append(scenario_name)

            for col, var_path in enumerate(target_variable_paths):
                target_var[row, col] = target_variable_dict[var_path]

        # Check no field without filling
        # assert not np.isnan(target_var).any()

        target_variable_arrays.append(target_var)
        scenario_names.append(names_for_stand)

    return ScenarioArrayData(
        arrays=target_variable_arrays, scenario_names=scenario_names
    )


def transform_array_data_to_list_of_scenarios(
    array_data: ScenarioArrayData,
    n_stands: int,
    target_variable_paths: Sequence[NetcdfVariablePath],
) -> list[dict[ScenarioID, TargetVariableDict]]:
    vars_of_interest_by_stand: list[dict[ScenarioID, TargetVariableDict]] = []

    for n_stand in range(n_stands):
        scenarios_for_stand = {}

        for row_idx, scenarios_data in enumerate(array_data.arrays[n_stand]):
            scenario_name = array_data.scenario_names[n_stand][row_idx]
            target_variable_values = {
                target_var_path: scenarios_data[col_idx]
                for (col_idx, target_var_path) in enumerate(target_variable_paths)
            }

            scenarios_for_stand[scenario_name] = target_variable_values

        vars_of_interest_by_stand.append(scenarios_for_stand)

    return vars_of_interest_by_stand
