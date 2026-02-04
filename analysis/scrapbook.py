# %%
from pathlib import Path
import datetime
import netCDF4
import xarray as xr
import pandas as pd
from typing import NewType

import susi.io.utils as io_utils
from susi.io import netcdf_utils
from susi.io.app_settings import AppSettings

# %%

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

df = load_all_metadatas_from_folder(folder=OUTPUT_FOLDER)

# %% Read netcdf data with xarray into single array
# Example: get all _partialblocking scenarios

partialblocking_paths = [
    p.joinpath("susi.nc")
    for p in OUTPUT_FOLDER.glob("*")
    if "partialblocking" in str(p)
]

xr.open_mfdataset(paths=partialblocking_paths, decode_times=False)

# %% Query and filter as desired
# Example: get all _partialblocking
df = df[df["experiment_id"].str.contains("partialblocking")]


# %% Read ncdf data into python dictionary with netcdf

# Netcdf data is saved in a dictionary where the experimentID is the key.
data = {}

for _, experiment_info in df.iterrows():
    netcdf_filepath = Path(experiment_info["netcdf_output_filepath"])
    data[experiment_info["experiment_id"]] = netCDF4.Dataset(netcdf_filepath, "r")


# %% Experimental widgets


import ipywidgets
from IPython.display import display

output = ipywidgets.Output()

experiment_ID_dropdown = ipywidgets.Dropdown(
    options=sorted(list(data.keys())), description="Experiment ID"
)


display(experiment_ID_dropdown)
