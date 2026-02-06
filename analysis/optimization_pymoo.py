# %%
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
        df[col] = pd.to_datetime(df[col], format=io_utils.datetime_format())
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


# %% Load parameter metadata
OUTPUT_FOLDER = app_settings.output_folder / "paroninkorpi"


# TODO: thinning.nc have not been added yet. Maybe put them in a separate folder?
df = load_all_metadatas_from_folder(folder=OUTPUT_FOLDER)

# %% Read ncdf data into python dictionary with netcdf

# Netcdf data is saved in a dictionary where the experiment folder path is the key.
data = {}

for _, experiment_info in df.iterrows():
    netcdf_filepath = Path(experiment_info["netcdf_output_filepath"])
    data[experiment_info["experiment_folder_path"]] = netCDF4.Dataset(
        netcdf_filepath, "r"
    )


# %% read interesting variables
def get_last_year_values(var: float) -> np.ndarray:
    return var[:, -1, :]


def get_scenario_name_from_path(path: str) -> str:
    return path.split("_")[-1]


N_STANDS = 21

data_per_stand = [{}] * N_STANDS
for n_stand in range(0, N_STANDS):
    keys_for_given_stand = [p for p in data.keys() if f"scenario_{n_stand + 1}" in p]
    breakpoint()
    data_per_stand[n_stand] = {
        k: v for k, v in data.items() if k in keys_for_given_stand
    }

d = data["/home/txart/projects/hiket/susi_26/outputs/paroninkorpi/base_scenario_17_DNM"]

stand_group = d.groups["stand"]
volume = stand_group.variables["volume"][:]
total_last_year_volume = np.sum(get_last_year_values(volume))

soil_c_balance_co2eq = d.groups["balance"].groups["C"].variables["soil_c_balance_co2eq"]
total_last_yeaar_soil_c_balance = np.sum(get_last_year_values(soil_c_balance_co2eq))


# %% pymoo (not started yet)
