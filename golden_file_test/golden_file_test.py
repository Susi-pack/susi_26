# Checks whether the results of executing susi_calls.py
# change over time or not.
# Useful to know if my changes are changing the code in any way.

# PROCEDURE
# the True or Golden results of calling susi_calls.py are stored in golden_susi.nc
# We compute the sha256 hash of that file.
# Then we call the modified susi_calls.py, store it in susi.nc, and compute the sha256 hash.
# The hashes of the 2 files should be identical.


import subprocess
from pathlib import Path

import netCDF4
import numpy as np

from susi.io import netcdf_utils
from susi.io.app_settings import AppSettings
from susi.io.utils import get_project_root


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


project_root_path = get_project_root()

settings = AppSettings()

CURRENT_SUSI_CALLS_PATH = project_root_path / Path("src/scripts/susi_calls.py")
GOLDEN_NETCDF_FILE_PATH = project_root_path / Path("golden_file_test/golden_susi.nc")
NEW_SUSI_NETCDF_FILE_PATH = settings.output_folder / Path("susi.nc")

print("Executing susi_calls.py...")
subprocess.run(
    ["python", str(CURRENT_SUSI_CALLS_PATH)],
    check=True,
)

test_passes = match_netcdf_files(
    new_netcdf_filepath=NEW_SUSI_NETCDF_FILE_PATH,
    golden_netcdf_filepath=GOLDEN_NETCDF_FILE_PATH,
)

if not test_passes:
    raise ValueError("ERROR in the golden test!")
else:
    print("---------------------")
    print("Golden test passed!")
    print("---------------------")
