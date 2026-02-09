import pytest
import numpy as np
from pathlib import Path
import netCDF4
from tempfile import TemporaryDirectory

from susi.io.netcdf_utils import (
    list_all_netcdf_variables,
    read_value_netcdf_variable,
)


def create_mock_netcdf_file(filepath: Path):
    """
    Create a mock NetCDF file with some variables and groups.
    """
    with netCDF4.Dataset(filepath, "w") as nc:
        # Global dimensions
        nc.createDimension("time", 10)
        nc.createDimension("lat", 5)
        nc.createDimension("lon", 5)

        # Root-level variable
        var1 = nc.createVariable("temperature", "f4", ("time", "lat", "lon"))
        var1.units = "K"
        var1[:] = np.arange(250, 250 + 10 * 5 * 5, dtype=np.float32).reshape(10, 5, 5)

        # Group
        group = nc.createGroup("balance")
        group.createDimension("x", 3)
        group.createDimension("y", 2)

        var2 = group.createVariable("fertilization_release", "i4", ("x", "y"))
        var2.units = "kg/m2"
        var2[:] = np.arange(6).reshape(3, 2)

        # Nested subgroup
        subgroup = group.createGroup("nested")
        subgroup.createDimension("z", 4)
        var3 = subgroup.createVariable("nested_var", "f8", ("z",))
        var3.units = "m/s"
        var3[:] = np.linspace(0, 1, 4)


@pytest.fixture
def mock_netcdf_file():
    with TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "mock.nc"
        create_mock_netcdf_file(filepath)
        yield filepath


def test_explore_netcdf_structure(mock_netcdf_file):
    variables = list_all_netcdf_variables(mock_netcdf_file)

    # Check we found the expected number of variables
    paths = [var.path for var in variables]
    expected_paths = [
        "/temperature",
        "/balance/fertilization_release",
        "/balance/nested/nested_var",
    ]
    assert set(paths) == set(expected_paths)

    # Check attributes of one variable
    temp_var = next(var for var in variables if var.name == "temperature")
    assert temp_var.shape == (10, 5, 5)
    assert temp_var.dimension_names == ("time", "lat", "lon")
    assert temp_var.units == "K"


def test_read_netcdf_variable(mock_netcdf_file):
    variables = list_all_netcdf_variables(mock_netcdf_file)

    # Pick the nested variable
    nested_var = next(var for var in variables if var.name == "nested_var")

    data = read_value_netcdf_variable(mock_netcdf_file, nested_var)
    assert isinstance(data, np.ndarray)
    assert data.shape == (4,)
    np.testing.assert_allclose(data, np.linspace(0, 1, 4))

    # Pick root-level variable
    temp_var = next(var for var in variables if var.name == "temperature")
    data = read_value_netcdf_variable(mock_netcdf_file, temp_var)
    assert data.shape == (10, 5, 5)
    assert data[0, 0, 0] == 250


def test_read_and_write_consistency(mock_netcdf_file):
    """
    Ensure that reading a variable returns exactly the same data as originally written.
    """
    variables = list_all_netcdf_variables(mock_netcdf_file)

    for var in variables:
        data = read_value_netcdf_variable(mock_netcdf_file, var)
        # Reopen file and read directly from netCDF4 for comparison
        with netCDF4.Dataset(mock_netcdf_file, "r") as nc:
            path_parts = [p for p in var.path.split("/") if p]
            group = nc
            for part in path_parts[:-1]:
                group = group.groups[part]
            expected_data = group.variables[path_parts[-1]][:]
            np.testing.assert_array_equal(data, expected_data)
