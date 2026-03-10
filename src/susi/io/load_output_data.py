# Read netcdf files and load variables into and OutputDataStore
from functools import cached_property

from typing import NewType, Sequence
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import netCDF4

import susi.io.utils as io_utils


# %% dataclasses
# Strings by other names
# Example: "/balance/K/fertilization_release"
NetcdfVariablePath = NewType("NetcdfVariablePath", str)
StandID = NewType("StandID", str)  # same as stand folder name
ScenarioID = NewType("ScenarioID", str)  # same as scenario folder name

TargetVariableDict = NewType("TargetVariableDict", dict[NetcdfVariablePath, float])


@dataclass(frozen=True)
class NetcdfVariableInfo:
    """
    Contains all information of a variable from the Susi netcdf file,
    except its values
    """

    name: str  # Example: "fertilization_release"
    dimension_names: tuple[str]
    shape: tuple[int]
    units: str  # Unit description


class NetcdfVariableArray:
    """
    Wraps a raw NetCDF numpy array and exposes clean spatial/temporal aggregations.

    Raw arrays may be 1-D, 2-D (usually time × location), or 3-D
    (scenario × time × location).
    In the 3-D case the first and last location columns are ditch columns and
    are excluded from all calculations.
    Only single-scenario (shape[0] == 1) 3-D arrays are currently supported.
    The rest of the arrays (1-D, and 2-D) are only stored, not modified.
    """

    def __init__(self, raw: np.ndarray) -> None:
        self._raw = raw
        self._validate()

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self._raw.ndim == 3 and self._raw.shape[0] != 1:
            raise NotImplementedError(
                f"Only single-scenario 3-D arrays are supported "
                f"(shape[0] must be 1, got {self._raw.shape[0]})."
            )

    # ------------------------------------------------------------------
    # Core processed view
    # ------------------------------------------------------------------

    @cached_property
    def processed(self) -> np.ndarray:
        """
        Return the array with implementation quirks removed:
        - 1-D and 2-D arrays are returned as-is.
        - 3-D arrays are reduced to 2-D by dropping the single scenario axis.
        """

        n_dims = self._raw.ndim
        match n_dims:
            case 1 | 2:
                return self._raw
            case 3:
                # shape[0] == 1 is guaranteed by _validate
                assert self._raw.shape[0] == 1
                # Remove the scenario column
                return self._strip_ditch_columns(self._raw[0, :, :])
            case _:
                raise ValueError(f"Expected 1-, 2-, or 3-D array; got {n_dims}-D.")

    @staticmethod
    def _strip_ditch_columns(array: np.ndarray) -> np.ndarray:
        """Remove the first and last columns (ditch columns) from a 2-D array."""
        assert array.ndim == 2
        return array[:, 1:-1]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _require_2D(array: np.ndarray) -> None:
        if array.ndim != 2:
            raise ValueError(
                f"This function required a 2-D array. Got {array.ndim} instead."
            )

    # ------------------------------------------------------------------
    # Time slicing
    # ------------------------------------------------------------------

    def last_timestep(self) -> np.ndarray:
        return self.processed[-1, :]

    # ------------------------------------------------------------------
    # Aggregators
    # ------------------------------------------------------------------

    def spatial_mean_at_last_timestep(self) -> float:
        """Mean over all locations at the final timestep."""
        self._require_2D(self.processed)
        return float(self.last_timestep().mean())

    def spatial_sum_at_last_timestep(self) -> float:
        """Sum over all locations at the final timestep."""
        self._require_2D(self.processed)
        return float(self.last_timestep().sum())

    def mean_over_space(self) -> np.ndarray:
        """Time-series of spatial means; one value per timestep (1-D)."""
        self._require_2D(self.processed)
        return np.mean(self.processed, axis=1)

    def mean_over_time(self) -> np.ndarray:
        """Spatial profile of temporal means; one value per location (1-D)."""
        self._require_2D(self.processed)
        return np.mean(self.processed, axis=0)

    def mean_of_all_values(self) -> float:
        """
        Mean of every value in the processed array
        Works for any dimensionality).
        """
        return float(self.processed.mean())


@dataclass
class OutputDataStore:
    """
    The main data structure.
    SoA (struct of arrays), keyed by variable name.
    """

    stands: list[StandID]  # ["stand_A", "stand_B", ...]
    scenarios: dict[StandID, list[ScenarioID]]  # {"stand_A": ["scen_1", "scen_2"], ...}
    variables: Sequence[NetcdfVariablePath]

    # Core data structure: Struct of Arrays, keyed by variable
    # Example: {var1: {(standA, scenA): [...]} }
    data: dict[
        NetcdfVariablePath, dict[tuple[StandID, ScenarioID], NetcdfVariableArray]
    ]

    def get_variable_values_all_scenarios(
        self, variable_path: NetcdfVariablePath
    ) -> dict[tuple[StandID, ScenarioID], NetcdfVariableArray]:
        return self.data[variable_path]

    def get_variable_value_for_scenario_and_stand(
        self,
        variable_path: NetcdfVariablePath,
        stand_id: StandID,
        scenario_id: ScenarioID,
    ) -> NetcdfVariableArray:
        return self.data[variable_path][(stand_id, scenario_id)]


# %% Functions
def list_subdirectories(path: Path) -> list[Path]:
    return [x for x in path.iterdir() if x.is_dir()]


# def list_variable_absolute_paths(group: netCDF4.Dataset, path: str = "/") -> list[str]:
#     vars_with_paths = []
#     for v in group.variables:
#         vars_with_paths.append(f"{path}/{v}".replace("//", "/"))
#
#     for name, subgroup in group.groups.items():
#         subpath = f"{path}/{name}".replace("//", "/")
#         vars_with_paths.extend(list_variable_absolute_paths(subgroup, subpath))
#
#     return vars_with_paths
#
#
def _read_json_metadatas(
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


def _load_single_experiment_metadatas(
    experiment_folderpath: Path,
    metadata_filename: str = "metadata.json",
    params_filename: str = "params.json",
) -> pd.DataFrame:
    """
    Reads metadata and parameter info from json files.
    Returns dict of all json values.
    """
    metadata, params = _read_json_metadatas(
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


def _load_all_metadatas_from_single_stand(folder: Path) -> pd.DataFrame:
    experiment_folderpaths = list_subdirectories(folder)
    df = pd.concat(
        [
            _load_single_experiment_metadatas(exp_fpath)
            for exp_fpath in experiment_folderpaths
        ]
    )

    return modify_after_load(df)


def load_all_metadatas_from_stands(folders: list[Path]) -> dict[StandID, pd.DataFrame]:
    all_metadatas = {}
    for folder in folders:
        all_metadatas[StandID(folder.name)] = _load_all_metadatas_from_single_stand(
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


def _read_value_several_variables_from_single_file(
    netcdf_filepath: Path, variable_paths: Sequence[NetcdfVariablePath]
) -> dict[NetcdfVariablePath, np.ndarray]:
    """
    Read the values of a list of NetcdfVariables from a NetCDF file.
    """
    data: dict[NetcdfVariablePath, np.ndarray] = {}

    with netCDF4.Dataset(netcdf_filepath, "r") as nc:
        for variable_path in variable_paths:
            # split path and remove empty strings
            var_path_parts = [p for p in variable_path.split("/") if p]

            values = _get_variable_by_path(nc, var_path_parts)

            data[variable_path] = values[:]

    return data


# def choose_netcdf_vars_by_path(
#     paths: Sequence[NetcdfVariablePath],
#     all_variables: list[NetcdfVariableInfo],
# ) -> list[NetcdfVariableInfo]:
#     """
#     Filters variables by path.
#     I tried to filtering by name, but the variable names are not unique: they depend on the path
#     E.g., the name "volume" has several different variables
#     """
#     variables = []
#     for var_info in all_variables:
#         if var_info.path in paths:
#             variables.append(var_info)
#
#     # if len(variables) != len(names):
#     #     raise ValueError("Could not locate all variables by name")
#
#     return variables
#


def _get_scenarios_for_stand(metadata_df: pd.DataFrame) -> list[ScenarioID]:
    # TODO: change key "experiment_id" to the new scenario key.
    return list(metadata_df["experiment_id"])


def _get_netcdf_filepaths_for_stand(metadata_df: pd.DataFrame) -> list[Path]:
    return list(metadata_df["netcdf_output_filepath"])


def read_netcdf_files_for_selected_variables(
    selected_variables: Sequence[NetcdfVariablePath],
    metadata_by_stand: dict[StandID, pd.DataFrame],
) -> OutputDataStore:
    stands: list[StandID] = []
    scenarios_by_stand: dict[StandID, list[ScenarioID]] = {}
    data: dict[
        NetcdfVariablePath, dict[tuple[StandID, ScenarioID], NetcdfVariableArray]
    ] = {var_path: {} for var_path in selected_variables}

    for stand_id, metadata_df in metadata_by_stand.items():
        stands.append(stand_id)

        # We rely on pandas dataframe ordering for the two orders from
        # scenarios and netcdf_filepaths to match
        scenarios = _get_scenarios_for_stand(metadata_df)
        netcdf_filepaths = _get_netcdf_filepaths_for_stand(metadata_df)

        scenarios_by_stand[stand_id] = scenarios

        for scenario_id, netcdf_filepath in zip(scenarios, netcdf_filepaths):
            variable_values = _read_value_several_variables_from_single_file(
                netcdf_filepath=netcdf_filepath, variable_paths=selected_variables
            )

            for var_path, var_value in variable_values.items():
                data[var_path][(stand_id, scenario_id)] = NetcdfVariableArray(var_value)
    return OutputDataStore(
        stands=stands,
        scenarios=scenarios_by_stand,
        variables=selected_variables,
        data=data,
    )


# def transform_list_of_scenarios_to_optimization_array_structure(
#     vars_of_interest_by_stand: list[dict[ScenarioID, TargetVariableDict]],
#     n_stands: int,
#     target_variable_paths: Sequence[NetcdfVariablePath],
# ) -> ScenarioArrayData:
#     target_variable_arrays: list[np.ndarray] = []
#
#     scenario_names: list[list[ScenarioID]] = []
#
#     n_target_variables = len(target_variable_paths)
#
#     for n_stand in range(n_stands):
#         vars_of_interest_single_stand = vars_of_interest_by_stand[n_stand]
#         number_of_scenarios = len(vars_of_interest_single_stand)
#
#         # Initialize the array that holds the variables' values
#         target_var = np.ones((number_of_scenarios, n_target_variables)) * np.nan
#         names_for_stand = []
#
#         for row, (scenario_name, target_variable_dict) in enumerate(
#             vars_of_interest_single_stand.items()
#         ):
#             names_for_stand.append(scenario_name)
#
#             for col, var_path in enumerate(target_variable_paths):
#                 target_var[row, col] = target_variable_dict[var_path]
#
#         # Check no field without filling
#         # assert not np.isnan(target_var).any()
#
#         target_variable_arrays.append(target_var)
#         scenario_names.append(names_for_stand)
#
#     return ScenarioArrayData(
#         arrays=target_variable_arrays, scenario_names=scenario_names
#     )
#
#
# def transform_array_data_to_list_of_scenarios(
#     array_data: ScenarioArrayData,
#     n_stands: int,
#     target_variable_paths: Sequence[NetcdfVariablePath],
# ) -> list[dict[ScenarioID, TargetVariableDict]]:
#     vars_of_interest_by_stand: list[dict[ScenarioID, TargetVariableDict]] = []
#
#     for n_stand in range(n_stands):
#         scenarios_for_stand = {}
#
#         for row_idx, scenarios_data in enumerate(array_data.arrays[n_stand]):
#             scenario_name = array_data.scenario_names[n_stand][row_idx]
#             target_variable_values = {
#                 target_var_path: scenarios_data[col_idx]
#                 for (col_idx, target_var_path) in enumerate(target_variable_paths)
#             }
#
#             scenarios_for_stand[scenario_name] = target_variable_values
#
#         vars_of_interest_by_stand.append(scenarios_for_stand)
#
#     return vars_of_interest_by_stand
