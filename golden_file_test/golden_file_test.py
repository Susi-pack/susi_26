# Checks whether the results of executing susi_calls.py
# change over time or not.
# Useful to know if my changes are changing the code in any way.

# PROCEDURE
# the True or Golden results of calling susi_calls.py are stored in golden_susi.nc
# We compute the sha256 hash of that file.
# Then we call the modified susi_calls.py, store it in susi.nc, and compute the sha256 hash.
# The hashes of the 2 files should be identical.


from pathlib import Path

import netCDF4
import numpy as np

from susi.io.execution_config import SimulationParams
from susi.io import netcdf_utils
from susi.core.susi_utils import read_FMI_weather
from susi.core.susi_main import Susi
from inputs.parameters import golden_test
from susi.io.metadata_model import SimulationMetaData, _app_settings


def masked_arrays_equal(a, b, rtol=1e-5, atol=1e-5):
    # Convert to masked arrays if needed
    a = np.ma.array(a, copy=False)
    b = np.ma.array(b, copy=False)

    # Shape check
    if a.shape != b.shape:
        return False

    # Masks must match exactly
    if not np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b)):
        return False

    # Compare data, ignoring masked values and treating NaNs as equal
    return np.allclose(
        a.data,
        b.data,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def match_netcdf_files(new_netcdf_filepath: Path, golden_netcdf_filepath: Path):
    new_ds = netCDF4.Dataset(new_netcdf_filepath, "r")
    golden_ds = netCDF4.Dataset(golden_netcdf_filepath, "r")

    new_var_paths = netcdf_utils.list_variable_absolute_paths(group=new_ds)
    golden_var_paths = netcdf_utils.list_variable_absolute_paths(group=golden_ds)

    # Compare variable names first
    assert golden_var_paths == new_var_paths

    for var in new_var_paths:
        new = netcdf_utils.get_var_by_path(new_ds, var)
        golden = netcdf_utils.get_var_by_path(golden_ds, var)

        if not masked_arrays_equal(a=new, b=golden):
            raise ValueError(f" There were differences in variable {var}")
    return True


# %% Run SUSI
forc = read_FMI_weather(
    ID=0,
    start_date=golden_test.PARAMETERS.simulation_config.start_date,
    end_date=golden_test.PARAMETERS.simulation_config.end_date,
    sourcefile=golden_test.PARAMETERS.weather_parameters.FMI_weather_filepath,
)

project_root_path = _app_settings.project_root_path
GOLDEN_NETCDF_FILE_PATH = project_root_path / Path("golden_file_test/golden_susi.nc")
NEW_SUSI_EXPERIMENT_FOLDER_PATH = project_root_path / Path("golden_file_test")

# Initiate susi parameters
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(
        parent_output_folder=NEW_SUSI_EXPERIMENT_FOLDER_PATH,
        experiment_id="golden_file_new_experiment",
    ),
    susi_params=golden_test.PARAMETERS,
)

susi = Susi(
    simulation_parameters=simulation_parameters,
)

# Run susi
susi.run()

# %% Check test


test_passes = match_netcdf_files(
    new_netcdf_filepath=simulation_parameters.metadata.netcdf_output_filepath,
    golden_netcdf_filepath=GOLDEN_NETCDF_FILE_PATH,
)

if not test_passes:
    raise ValueError("ERROR in the golden test!")
else:
    print("---------------------")
    print("Golden test passed!")
    print("---------------------")
