# Checks whether the results of executing susi_calls.py
# change over time or not.
# Useful to know if my changes are changing the code in any way.

# PROCEDURE
# the True or Golden results of calling susi_calls.py are stored in golden_susi.nc
# We compute the sha256 hash of that file.
# Then we call the modified susi_calls.py, store it in susi.nc, and compute the sha256 hash.
# The hashes of the 2 files should be identical.


from pathlib import Path

import numpy as np
import shutil


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
    new_variables = netcdf_utils.list_all_netcdf_variables(new_netcdf_filepath)
    golden_variables = netcdf_utils.list_all_netcdf_variables(golden_netcdf_filepath)

    # Compare variable attributes first
    assert new_variables == golden_variables

    new_vars_values = netcdf_utils.read_value_several_variables_from_single_file(
        netcdf_filepath=new_netcdf_filepath, variables=new_variables
    )
    golden_vars_values = netcdf_utils.read_value_several_variables_from_single_file(
        netcdf_filepath=golden_netcdf_filepath, variables=golden_variables
    )

    for new_var, golden_var in zip(new_vars_values, golden_vars_values):
        if not masked_arrays_equal(a=new_var.value, b=golden_var.value):
            raise ValueError(" There were differences in some variable.")
    return True


# %% Run SUSI


def test_golden_susi():
    project_root_path = _app_settings.project_root_path
    GOLDEN_NETCDF_FILE_PATH = project_root_path / Path(
        "tests/golden_file_test/golden_susi.nc"
    )
    NEW_SUSI_EXPERIMENT_FOLDER_PATH = project_root_path / Path("tests/golden_file_test")

    new_golden_output_folder_name = "golden_file_new_experiment"

    # Remove previous golden test output folder if exists
    new_golden_output_folderpath = (
        NEW_SUSI_EXPERIMENT_FOLDER_PATH / new_golden_output_folder_name
    )
    if new_golden_output_folderpath.is_dir():
        shutil.rmtree(new_golden_output_folderpath)

    # Initiate susi parameters
    simulation_parameters = SimulationParams(
        metadata=SimulationMetaData(
            parent_output_folder=NEW_SUSI_EXPERIMENT_FOLDER_PATH,
            experiment_id=new_golden_output_folder_name,
        ),
        susi_params=golden_test.PARAMETERS,
    )

    susi = Susi(
        simulation_parameters=simulation_parameters,
    )

    # Run susi
    susi.run()

    test_passes = match_netcdf_files(
        new_netcdf_filepath=simulation_parameters.metadata.netcdf_output_filepath,
        golden_netcdf_filepath=GOLDEN_NETCDF_FILE_PATH,
    )

    assert test_passes
