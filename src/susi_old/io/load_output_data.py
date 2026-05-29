# Read netcdf files and load variables into and OutputDataStore
from functools import cached_property

from typing import NewType, Sequence, Callable
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


@dataclass
class SimulationParamsFromJSON:
    susi_params: dict
    metadata: dict


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

    def initial_timestep(self) -> np.ndarray:
        return self.processed[0, :]

    # ------------------------------------------------------------------
    # Aggregators
    # ------------------------------------------------------------------

    def spatial_mean_at_last_timestep(self) -> float:
        """Mean over all locations at the final timestep."""
        self._require_2D(self.processed)
        return float(self.last_timestep().mean())

    def spatial_mean_at_initial_timestep(self) -> float:
        """Mean over all locations at the initialization of the simulation."""
        self._require_2D(self.processed)
        return float(self.initial_timestep().mean())

    def spatial_sum_at_last_timestep(self) -> float:
        """Sum over all locations at the final timestep."""
        self._require_2D(self.processed)
        return float(self.last_timestep().sum())

    def mean_over_space(self) -> np.ndarray:
        """Time-series of spatial means; one value per timestep (1-D)."""
        self._require_2D(self.processed)
        return np.mean(self.processed, axis=1)

    def mean_over_space_sum_over_time(self) -> float:
        """Sum of the time series and spatial mean;  (scalar)."""
        self._require_2D(self.processed)
        return np.sum(np.mean(self.processed, axis=1))

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


# New type for calling the methods in other parts of the code
NetcdfAggregationFn = Callable[[NetcdfVariableArray], float]


@dataclass
class OutputDataStore:
    """
    The main data structure.
    SoA (struct of arrays), keyed by variable name.
    """

    stands: list[StandID]  # ["stand_A", "stand_B", ...]
    scenarios: dict[
        StandID, Sequence[ScenarioID]
    ]  # {"stand_A": ["scen_1", "scen_2"], ...}
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


def read_params_from_jsons(
    experiment_folderpath: Path,
    metadata_filename: str = "metadata.json",
    params_filename: str = "params.json",
) -> SimulationParamsFromJSON:
    metadata_filepath = experiment_folderpath.joinpath(metadata_filename)
    params_filepath = experiment_folderpath.joinpath(params_filename)

    metadata, params = map(
        io_utils.read_json_file, [metadata_filepath, params_filepath]
    )
    return SimulationParamsFromJSON(metadata=metadata, susi_params=params)


def _load_single_experiment_metadatas(
    experiment_folderpath: Path,
    metadata_filename: str = "metadata.json",
    params_filename: str = "params.json",
) -> pd.DataFrame:
    """
    Reads metadata and parameter info from json files.
    Returns dict of all json values.
    """
    params_from_json = read_params_from_jsons(
        experiment_folderpath, metadata_filename, params_filename
    )

    return pd.json_normalize(params_from_json.metadata | params_from_json.susi_params)


def coerce_datetime_format(df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = ["timestamp_start", "timestamp_end"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
    return df


def modify_after_load(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # set datetime formats
    df = coerce_datetime_format(df)

    # sort by starting date first
    df = df.sort_values(by="timestamp_start", ignore_index=True, ascending=False)

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


def read_value_several_variables_from_single_file(
    netcdf_filepath: Path, variable_paths: Sequence[NetcdfVariablePath]
) -> dict[NetcdfVariablePath, NetcdfVariableArray]:
    """
    Read the values of a list of NetcdfVariables from a NetCDF file.
    """
    data: dict[NetcdfVariablePath, NetcdfVariableArray] = {}

    with netCDF4.Dataset(netcdf_filepath, "r") as nc:
        for variable_path in variable_paths:
            # split path and remove empty strings
            var_path_parts = [p for p in variable_path.split("/") if p]

            values = _get_variable_by_path(nc, var_path_parts)

            data[variable_path] = NetcdfVariableArray(raw=values[:])

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


def get_scenarios_for_stand(metadata_df: pd.DataFrame) -> list[ScenarioID]:
    return [ScenarioID(scen) for scen in metadata_df["scenario_id"]]


def get_netcdf_filepaths_for_stand(metadata_df: pd.DataFrame) -> list[Path]:
    return [Path(p) for p in metadata_df["netcdf_output_filepath"]]


def read_netcdf_files_for_selected_variables(
    selected_variables: Sequence[NetcdfVariablePath],
    scenarios_by_stand: dict[StandID, Sequence[ScenarioID]],
    netcdf_filepaths_by_stand: dict[StandID, tuple[Path, ...]],
) -> OutputDataStore:
    stands: list[StandID] = []
    data: dict[
        NetcdfVariablePath, dict[tuple[StandID, ScenarioID], NetcdfVariableArray]
    ] = {var_path: {} for var_path in selected_variables}

    for stand_id, scenarios in scenarios_by_stand.items():
        stands.append(stand_id)

        for scenario_id, netcdf_filepath in zip(
            scenarios, netcdf_filepaths_by_stand[stand_id]
        ):
            variable_values = read_value_several_variables_from_single_file(
                netcdf_filepath=netcdf_filepath, variable_paths=selected_variables
            )

            for var_path, var_value in variable_values.items():
                data[var_path][(stand_id, scenario_id)] = var_value
    return OutputDataStore(
        stands=stands,
        scenarios=scenarios_by_stand,
        variables=selected_variables,
        data=data,
    )


def read_netcdf_files_for_selected_variables_from_metadatas(
    selected_variables: Sequence[NetcdfVariablePath],
    metadata_by_stand: dict[StandID, pd.DataFrame],
) -> OutputDataStore:
    """
    A simpler API for the previous function
    if we have metadata_by_stand
    """

    scenarios_by_stand = {}
    netcdf_filepaths_by_stand = {}

    for stand_id, metadata_df in metadata_by_stand.items():
        # We rely on pandas dataframe ordering for the two orders from
        # scenarios and netcdf_filepaths to match
        scenarios = get_scenarios_for_stand(metadata_df)
        netcdf_filepaths = get_netcdf_filepaths_for_stand(metadata_df)

        scenarios_by_stand[StandID(stand_id)] = scenarios
        netcdf_filepaths_by_stand[StandID(stand_id)] = netcdf_filepaths

    return read_netcdf_files_for_selected_variables(
        selected_variables=selected_variables,
        scenarios_by_stand=scenarios_by_stand,
        netcdf_filepaths_by_stand=netcdf_filepaths_by_stand,
    )
