# %%
from pathlib import Path
import netCDF4
import xarray as xr
import pandas as pd

import susi.io.utils as io_utils
from susi.io.project_layout import project_dir, run_dir

# %%

# Which run to poke at. A scenario's metadata lives at
# projects/<project>/outputs/<run_id>/<stand_id>/<scenario_id>/, so there is
# no single global outputs folder to default to any more -- name the run.
PROJECT_ID = "paroninkorpi"
RUN_ID = "paroninkorpi"

RUN_DIR = run_dir(project_dir(PROJECT_ID), run_id=RUN_ID)


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
    df: pd.DataFrame, set_run_id_as_index: bool = False
) -> pd.DataFrame:
    df = df.copy()

    # set datetime formats
    df = coerce_datetime_format(df)

    # sort by starting date first
    df = df.sort_values(by="timestamp_start", ignore_index=True, ascending=False)

    # set run_id as index
    if set_run_id_as_index:
        df = df.set_index(keys="run_id")

    return df


def load_all_metadatas_from_folder(folder: Path) -> pd.DataFrame:
    experiment_folderpaths = list_subdirectories(folder)
    df = pd.concat(
        [
            _load_single_experiment_metadatas(exp_fpath)
            for exp_fpath in experiment_folderpaths
        ]
    )

    return modify_after_load(df)


# %% Load parameter metadata
for stand_n in range(1, 22):
    stand_foldername = f"stand_{stand_n:02d}"

    df = load_all_metadatas_from_folder(folder=RUN_DIR / stand_foldername)

# %% Read netcdf data with xarray into single array (Not complete yet)
# Example: get all _partialblocking scenarios

partialblocking_paths = [
    p.joinpath("susi.nc")
    for p in OUTPUT_FOLDER.glob("*")
    if "partialblocking" in str(p)
]

xr.open_mfdataset(paths=partialblocking_paths, decode_times=False)

# %% Query and filter as desired
# Example: get all _partialblocking
df = df[df["run_id"].str.contains("partialblocking")]


# %% Read ncdf data into python dictionary with netcdf

# Netcdf data is saved in a dictionary where the run_id is the key.
data = {}

for _, experiment_info in df.iterrows():
    netcdf_filepath = Path(experiment_info["netcdf_output_filepath"])
    data[experiment_info["run_id"]] = netCDF4.Dataset(netcdf_filepath, "r")


# %% Experimental widgets


import ipywidgets
from IPython.display import display

output = ipywidgets.Output()

experiment_ID_dropdown = ipywidgets.Dropdown(
    options=sorted(list(data.keys())), description="Experiment ID"
)


display(experiment_ID_dropdown)
