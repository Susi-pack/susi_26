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
from susi.io import load_output_data
from susi.core.susi_main import Susi
from system_inputs.parameters import golden_test
from susi.io.metadata_model import SimulationMetaData
from susi.io.project_layout import outputs_dir_for_project, run_dir
import susi.io.utils as io_utils


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
    new_variables = load_output_data.list_all_netcdf_variables(new_netcdf_filepath)
    golden_variables = load_output_data.list_all_netcdf_variables(
        golden_netcdf_filepath
    )

    # Compare variable attributes first
    assert new_variables.keys() == golden_variables.keys()

    new_vars_values = load_output_data.read_value_several_variables_from_single_file(
        netcdf_filepath=new_netcdf_filepath, variable_paths=list(new_variables.keys())
    )
    golden_vars_values = load_output_data.read_value_several_variables_from_single_file(
        netcdf_filepath=golden_netcdf_filepath,
        variable_paths=list(golden_variables.keys()),
    )

    for key in new_vars_values.keys():
        golden_var_value = golden_vars_values[key]._raw
        new_var_value = new_vars_values[key]._raw
        if not masked_arrays_equal(a=golden_var_value, b=new_var_value):
            raise ValueError(
                f"There were differences in variable {key}.\n Golden value: {golden_var_value}\n New value: {new_var_value}"
            )
    return True


# %% Run SUSI


def test_golden_susi():
    repo_root_path = io_utils.repo_root()
    GOLDEN_NETCDF_FILE_PATH = repo_root_path / Path(
        "tests/golden_file_test/golden_susi.nc"
    )
    # The golden test folder doubles as a throwaway project, so the new run
    # lands at tests/golden_file_test/outputs/golden_file_new_run/, next to
    # the golden netcdf it is compared with. outputs/ is gitignored and not
    # in the checkout, so it is created here: a run never creates it itself.
    GOLDEN_TEST_PROJECT_DIR = repo_root_path / Path("tests/golden_file_test")
    outputs_dir_for_project(GOLDEN_TEST_PROJECT_DIR).mkdir(exist_ok=True)

    new_golden_run_id = "golden_file_new_run"

    # Remove previous golden test output folder if exists
    new_golden_output_folderpath = run_dir(GOLDEN_TEST_PROJECT_DIR, new_golden_run_id)
    if new_golden_output_folderpath.is_dir():
        shutil.rmtree(new_golden_output_folderpath)

    # Initiate susi parameters
    simulation_parameters = SimulationParams(
        metadata=SimulationMetaData(
            project_dir=GOLDEN_TEST_PROJECT_DIR,
            run_id=new_golden_run_id,
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
