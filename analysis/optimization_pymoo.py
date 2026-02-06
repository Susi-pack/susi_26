# %%
from dataclasses import dataclass
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pathlib import Path
import netCDF4
import pandas as pd

import susi.io.utils as io_utils
from susi.io.app_settings import AppSettings

# %% Get data


app_settings = AppSettings()


def list_subdirectories(path: Path):
    return (x for x in path.iterdir() if x.is_dir())


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


def load_all_metadatas_from_folder(
    folder: Path = app_settings.output_folder,
) -> pd.DataFrame:
    experiment_folderpaths = list_subdirectories(folder)
    df = pd.concat(
        [
            _load_single_experiment_metadatas(exp_fpath)
            for exp_fpath in experiment_folderpaths
        ]
    )

    return modify_after_load(df)


# %% Load paroninkorpi metadata
N_STANDS = 21

metadata_by_stand = [0] * N_STANDS

for stand_n in range(0, N_STANDS):
    stand_foldername = f"paroninkorpi/stand_{stand_n + 1:02d}"

    output_folder = app_settings.output_folder / stand_foldername

    df = load_all_metadatas_from_folder(folder=output_folder)

    metadata_by_stand[stand_n] = df

# %% compute total number of combinations

scenarios_cardinality = []
n_total_combinations = 1
for stand_n in range(0, N_STANDS):
    n_scenarios = metadata_by_stand[stand_n].shape[0]
    scenarios_cardinality.append(n_scenarios)

    n_total_combinations *= n_scenarios

print("Number of scenarios for each stand:")
print(scenarios_cardinality)
print(f"Number of total combinations: {n_total_combinations:.2e}")


# %% Read ncdf data into python dictionary with netcdf


def read_netcdf_files_for_stand(
    metadata_df: pd.DataFrame,
) -> dict[str : netCDF4.Dataset]:
    # Netcdf data is saved in a dictionary where the experiment folder path is the key.
    data = {}
    for _, experiment_info in metadata_df.iterrows():
        data[experiment_info["experiment_id"]] = netCDF4.Dataset(
            experiment_info["netcdf_output_filepath"], "r"
        )

    return data


all_variables_by_stand = [0] * N_STANDS

for n_stand, metadata_df in enumerate(metadata_by_stand):
    all_variables_by_stand[n_stand] = read_netcdf_files_for_stand(metadata_df)


# %% interesting_variables by stand
def get_last_year_values(var: float) -> np.ndarray:
    return var[:, -1, :]


def get_scenario_name_from_path(path: str) -> str:
    return path.split("_")[-1]


@dataclass
class TargetVariables:
    volume: float
    soil_c_balance: float


vars_of_interest_by_stand = [{} for _ in range(N_STANDS)]

for n_stand in range(N_STANDS):
    for scenario_name, netcdf_data in all_variables_by_stand[n_stand].items():
        # Get variable values
        volume = netcdf_data.groups["stand"].variables["volume"][:]

        # Negative sign because we actually want to maximize volume,
        # but it is a minimization problem
        total_last_year_volume = -np.sum(get_last_year_values(volume))

        soil_c_balance_co2eq = (
            netcdf_data.groups["balance"].groups["C"].variables["soil_c_balance_co2eq"]
        )
        total_last_year_soil_c_balance = np.sum(
            get_last_year_values(soil_c_balance_co2eq)
        )

        # Add to list
        vars_of_interest_by_stand[n_stand][scenario_name] = TargetVariables(
            volume=total_last_year_volume,
            soil_c_balance=total_last_year_soil_c_balance,
        )

# TODO: read each interesting variable from netcdf file in the same loop.
# Do so using a context manager.
# Otherwise, the netcdf files stay open and consume too much memory.
# So: with open file:
#       get necessary vars into list of dicts (above called vars_of_interest_by_stand)


# %% pymoo (not started yet)
