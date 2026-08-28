from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from susi.io.load_output_data import NetcdfVariablePath
from analysis.shared_reporting_utils.single_scenario_dashboard import (
    load_report_data,
)


def create_mock_netcdf_file(filepath: Path) -> None:
    """
    Create a small NetCDF file with a handful of variables at paths matching
    a few real Single Scenario Dashboard VARIABLE_PATHS entries.
    """
    with netCDF4.Dataset(filepath, "w") as nc:
        nc.createDimension("time", 3)
        nc.createDimension("space", 4)

        stand_group = nc.createGroup("stand")
        volume_var = stand_group.createVariable("volume", "f8", ("time", "space"))
        volume_var[:] = np.arange(12, dtype=np.float64).reshape(3, 4)

        strip_group = nc.createGroup("strip")
        dwt_var = strip_group.createVariable("dwt", "f8", ("time", "space"))
        dwt_var[:] = -1.0 * np.arange(12, dtype=np.float64).reshape(3, 4)

        cpy_group = nc.createGroup("cpy")
        et_var = cpy_group.createVariable("ET_yr", "f8", ("time", "space"))
        et_var[:] = np.arange(100, 112, dtype=np.float64).reshape(3, 4)


@pytest.fixture
def mock_netcdf_file():
    with TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "mock.nc"
        create_mock_netcdf_file(filepath)
        yield filepath


class TestLoadReportData:
    def test_returned_keys_match_requested_variable_paths_exactly(
        self, mock_netcdf_file
    ):
        variable_paths = (
            NetcdfVariablePath("/stand/volume"),
            NetcdfVariablePath("/strip/dwt"),
            NetcdfVariablePath("/cpy/ET_yr"),
        )

        data = load_report_data(
            netcdf_filepath=mock_netcdf_file, variable_paths=variable_paths
        )

        assert set(data.keys()) == set(variable_paths)

    def test_values_round_trip_correctly(self, mock_netcdf_file):
        variable_paths = (NetcdfVariablePath("/stand/volume"),)

        data = load_report_data(
            netcdf_filepath=mock_netcdf_file, variable_paths=variable_paths
        )

        expected = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = data[NetcdfVariablePath("/stand/volume")]
        assert result.raw_shape == (3, 4)
        np.testing.assert_array_equal(result.processed, expected)

    def test_subset_of_variable_paths_omits_others(self, mock_netcdf_file):
        variable_paths = (NetcdfVariablePath("/strip/dwt"),)

        data = load_report_data(
            netcdf_filepath=mock_netcdf_file, variable_paths=variable_paths
        )

        assert NetcdfVariablePath("/stand/volume") not in data
        assert NetcdfVariablePath("/cpy/ET_yr") not in data
