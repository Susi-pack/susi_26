from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from analysis.shared_reporting_utils import plots
from analysis.shared_reporting_utils.single_scenario_dashboard import (
    SECTIONS,
    load_report_data,
)
from susi.io.load_output_data import NetcdfVariablePath


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


class TestSections:
    """
    Wiring-only tests for SECTIONS: which title/plot_fn each key maps to.
    Doesn't call the plot_fns themselves -- `plots.py`'s figure builders are
    deliberately untested (per #227), since exercising them needs a full
    ~100-variable netcdf fixture for no extra coverage.
    """

    def test_keys_cover_all_seven_sections(self):
        assert set(SECTIONS.keys()) == {
            "stand",
            "hydrology",
            "mass",
            "carbon",
            "N",
            "P",
            "K",
        }

    def test_titles(self):
        assert SECTIONS["stand"][0] == "Stand"
        assert SECTIONS["hydrology"][0] == "Hydrology"
        assert SECTIONS["mass"][0] == "Mass"
        assert SECTIONS["carbon"][0] == "Carbon"
        assert SECTIONS["N"][0] == "Nitrogen Balance"
        assert SECTIONS["P"][0] == "Phosphorus Balance"
        assert SECTIONS["K"][0] == "Potassium Balance"

    def test_non_substance_sections_wire_to_matching_plots_function(self):
        assert SECTIONS["stand"][1] is plots.stand
        assert SECTIONS["hydrology"][1] is plots.hydrology
        assert SECTIONS["mass"][1] is plots.mass
        assert SECTIONS["carbon"][1] is plots.carbon

    def test_nutrient_balance_sections_wire_to_matching_substance(self):
        for substance in ("N", "P", "K"):
            _, plot_fn = SECTIONS[substance]
            assert isinstance(plot_fn, partial)
            assert plot_fn.func is plots.nutrient_balance
            assert plot_fn.keywords == {"substance": substance}
