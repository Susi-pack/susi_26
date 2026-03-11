import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import netCDF4
from tempfile import TemporaryDirectory
import json

from susi.io.load_output_data import (
    list_all_netcdf_variables,
    _get_variable_by_path,
    read_value_several_variables_from_single_file,
    coerce_datetime_format,
    modify_after_load,
    list_subdirectories,
    _load_single_experiment_metadatas,
    _load_all_metadatas_from_single_stand,
    load_all_metadatas_from_stands,
    NetcdfVariableInfo,
    NetcdfVariablePath,
    ScenarioID,
    StandID,
    TargetVariableDict,
    OutputDataStore,
    read_netcdf_files_for_selected_variables,
)

from susi.io.load_output_data import NetcdfVariableArray


def create_mock_netcdf_file(filepath: Path):
    """
    Create a mock NetCDF file with some variables and groups.
    """
    with netCDF4.Dataset(filepath, "w") as nc:
        nc.createDimension("time", 10)
        nc.createDimension("lat", 5)
        nc.createDimension("lon", 5)

        var1 = nc.createVariable("temperature", "f4", ("time", "lat", "lon"))
        var1.units = "K"
        var1[:] = np.arange(250, 250 + 10 * 5 * 5, dtype=np.float32).reshape(10, 5, 5)

        group = nc.createGroup("balance")
        group.createDimension("x", 3)
        group.createDimension("y", 2)

        var2 = group.createVariable("fertilization_release", "i4", ("x", "y"))
        var2.units = "kg/m2"
        var2[:] = np.arange(6).reshape(3, 2)

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

    assert isinstance(variables, dict)
    expected_paths = [
        NetcdfVariablePath("/temperature"),
        NetcdfVariablePath("/balance/fertilization_release"),
        NetcdfVariablePath("/balance/nested/nested_var"),
    ]
    assert set(variables.keys()) == set(expected_paths)

    temp_var = variables[NetcdfVariablePath("/temperature")]
    assert temp_var.shape == (10, 5, 5)
    assert temp_var.dimension_names == ("time", "lat", "lon")
    assert temp_var.units == "K"
    assert temp_var.name == "temperature"


def test_get_variable_by_path_internal(mock_netcdf_file):
    with netCDF4.Dataset(mock_netcdf_file, "r") as nc:
        var = _get_variable_by_path(nc, ["temperature"])
        assert var.shape == (10, 5, 5)

        var = _get_variable_by_path(nc, ["balance", "fertilization_release"])
        assert var.shape == (3, 2)

        var = _get_variable_by_path(nc, ["balance", "nested", "nested_var"])
        assert var.shape == (4,)


def test_read_value_several_variables_from_single_file(mock_netcdf_file):
    selected_vars = [
        NetcdfVariablePath("/balance/fertilization_release"),
        NetcdfVariablePath("/balance/nested/nested_var"),
    ]

    values = read_value_several_variables_from_single_file(
        mock_netcdf_file, selected_vars
    )

    assert isinstance(values, dict)
    assert set(values.keys()) == set(selected_vars)

    fert_val = values[NetcdfVariablePath("/balance/fertilization_release")]
    assert isinstance(fert_val, NetcdfVariableArray)
    assert fert_val.processed.shape == (3, 2)

    nested_val = values[NetcdfVariablePath("/balance/nested/nested_var")]
    assert isinstance(nested_val, NetcdfVariableArray)
    assert nested_val.processed.shape == (4,)


def test_coerce_datetime_format():
    df = pd.DataFrame(
        {
            "timestamp_start": ["2020-01-01", "2020-01-02"],
            "timestamp_end": ["2020-12-31", "2020-12-30"],
            "value": [1, 2],
        }
    )

    result = coerce_datetime_format(df)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp_start"])
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp_end"])


def test_modify_after_load():
    df = pd.DataFrame(
        {
            "timestamp_start": ["2020-01-01", "2020-01-02"],
            "timestamp_end": ["2020-12-31", "2020-12-30"],
            "experiment_id": ["exp1", "exp2"],
            "value": [1, 2],
        }
    )

    result = modify_after_load(df, set_experiment_id_as_index=False)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp_start"])
    assert result.iloc[0]["timestamp_start"] > result.iloc[1]["timestamp_start"]


def test_modify_after_load_with_index():
    df = pd.DataFrame(
        {
            "timestamp_start": ["2020-01-01", "2020-01-02"],
            "timestamp_end": ["2020-12-31", "2020-12-30"],
            "experiment_id": ["exp1", "exp2"],
            "value": [1, 2],
        }
    )

    result = modify_after_load(df, set_experiment_id_as_index=True)

    assert "experiment_id" not in result.columns
    assert "exp1" in result.index
    assert "exp2" in result.index


def test_list_subdirectories():
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "dir1").mkdir()
        (base / "dir2").mkdir()
        (base / "file.txt").touch()

        result = list(list_subdirectories(base))
        result_names = {p.name for p in result}

        assert result_names == {"dir1", "dir2"}


def create_mock_experiment_folder(base_path: Path, experiment_id: str):
    exp_dir = base_path / experiment_id
    exp_dir.mkdir()

    metadata = {
        "experiment_id": experiment_id,
        "timestamp_start": "2020-01-01T00:00:00",
        "timestamp_end": "2020-12-31T00:00:00",
    }
    params = {
        "param1": 1.0,
        "param2": "value",
    }

    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with open(exp_dir / "params.json", "w") as f:
        json.dump(params, f)


@pytest.fixture
def mock_experiment_folders():
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        create_mock_experiment_folder(base, "exp1")
        create_mock_experiment_folder(base, "exp2")
        yield base


def test_load_single_experiment_metadatas(mock_experiment_folders):
    exp_folder = mock_experiment_folders / "exp1"

    result = _load_single_experiment_metadatas(exp_folder)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["experiment_id"] == "exp1"
    assert result.iloc[0]["param1"] == 1.0


def test_load_all_metadatas_from_single_folder(mock_experiment_folders):
    result = _load_all_metadatas_from_single_stand(mock_experiment_folders)

    assert len(result) == 2
    assert set(result["experiment_id"]) == {"exp1", "exp2"}


def test_load_all_metadatas_from_stands(mock_experiment_folders):
    folder1 = mock_experiment_folders / "stand_A"
    folder2 = mock_experiment_folders / "stand_B"
    folder1.mkdir()
    folder2.mkdir()
    create_mock_experiment_folder(folder1, "exp3")
    create_mock_experiment_folder(folder2, "exp4")

    result = load_all_metadatas_from_stands([folder1, folder2])

    assert isinstance(result, dict)
    assert set(result.keys()) == {StandID("stand_A"), StandID("stand_B")}
    assert len(result[StandID("stand_A")]) == 1
    assert len(result[StandID("stand_B")]) == 1


class TestNetcdfVariableArray:
    """Tests for NetcdfVariableArray class."""

    def test_construction_1d_array(self):
        raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = NetcdfVariableArray(raw)
        assert arr._raw is raw

    def test_construction_2d_array(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        arr = NetcdfVariableArray(raw)
        assert arr._raw is raw

    def test_construction_3d_array_single_scenario(self):
        raw = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        arr = NetcdfVariableArray(raw)
        assert arr._raw is raw

    def test_construction_3d_array_multiple_scenarios_raises(self):
        raw = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        with pytest.raises(NotImplementedError):
            NetcdfVariableArray(raw)

    def test_construction_invalid_dimensions_raises(self):
        raw = np.array([[[[1.0]]]])
        arr = NetcdfVariableArray(raw)
        with pytest.raises(ValueError):
            _ = arr.processed

    def test_last_timestep_3d(self):
        raw = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        arr = NetcdfVariableArray(raw)
        result = arr.last_timestep()
        np.testing.assert_array_equal(result, np.array([5.0]))

    def test_last_timestep_fails_for_1d(self):
        raw = np.array([1.0, 2.0, 3.0])
        arr = NetcdfVariableArray(raw)
        with pytest.raises(IndexError):
            arr.last_timestep()

    def test_spatial_mean_at_last_timestep(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        arr = NetcdfVariableArray(raw)
        assert arr.spatial_mean_at_last_timestep() == 5.5

    def test_spatial_sum_at_last_timestep(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        arr = NetcdfVariableArray(raw)
        assert arr.spatial_sum_at_last_timestep() == 11.0

    def test_mean_over_space(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        arr = NetcdfVariableArray(raw)
        expected = np.array([1.5, 3.5, 5.5])
        np.testing.assert_array_equal(arr.mean_over_space(), expected)

    def test_mean_over_time(self):
        raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        arr = NetcdfVariableArray(raw)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_equal(arr.mean_over_time(), expected)

    def test_mean_of_all_values_1d(self):
        raw = np.array([1.0, 2.0, 3.0])
        arr = NetcdfVariableArray(raw)
        assert arr.mean_of_all_values() == 2.0

    def test_mean_of_all_values_2d(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr = NetcdfVariableArray(raw)
        assert arr.mean_of_all_values() == 2.5

    def test_mean_of_all_values_3d(self):
        raw = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        arr = NetcdfVariableArray(raw)
        assert arr.mean_of_all_values() == 3.5

    def test_with_nan_values(self):
        raw = np.array([[1.0, np.nan], [3.0, 4.0]])
        arr = NetcdfVariableArray(raw)
        result = arr.mean_of_all_values()
        assert np.isnan(result)

    def test_different_dtypes_float32(self):
        raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr = NetcdfVariableArray(raw)
        assert arr.processed.dtype == np.float32

    def test_different_dtypes_int(self):
        raw = np.array([[1, 2], [3, 4]], dtype=np.int32)
        arr = NetcdfVariableArray(raw)
        result = arr.mean_of_all_values()
        assert result == 2.5


class TestOutputDataStore:
    """Tests for OutputDataStore class."""

    @pytest.fixture
    def mock_output_datastore(self, mock_netcdf_file):
        selected_vars = [NetcdfVariablePath("/balance/fertilization_release")]
        metadata_by_stand = {
            StandID("stand_A"): pd.DataFrame(
                {
                    "experiment_id": ["scenario_1", "scenario_2"],
                    "netcdf_output_filepath": [mock_netcdf_file, mock_netcdf_file],
                }
            )
        }
        return read_netcdf_files_for_selected_variables(
            selected_variables=selected_vars,
            metadata_by_stand=metadata_by_stand,
        )

    def test_output_datastore_creation(self, mock_output_datastore):
        assert isinstance(mock_output_datastore, OutputDataStore)
        assert mock_output_datastore.stands == [StandID("stand_A")]
        assert mock_output_datastore.scenarios == {
            StandID("stand_A"): [ScenarioID("scenario_1"), ScenarioID("scenario_2")]
        }
        assert mock_output_datastore.variables == [
            NetcdfVariablePath("/balance/fertilization_release")
        ]

    def test_get_variable_values_all_scenarios(self, mock_output_datastore):
        result = mock_output_datastore.get_variable_values_all_scenarios(
            NetcdfVariablePath("/balance/fertilization_release")
        )
        assert isinstance(result, dict)
        assert (StandID("stand_A"), ScenarioID("scenario_1")) in result
        assert (StandID("stand_A"), ScenarioID("scenario_2")) in result

    def test_get_variable_value_for_scenario_and_stand(self, mock_output_datastore):
        result = mock_output_datastore.get_variable_value_for_scenario_and_stand(
            variable_path=NetcdfVariablePath("/balance/fertilization_release"),
            stand_id=StandID("stand_A"),
            scenario_id=ScenarioID("scenario_1"),
        )
        assert isinstance(result, NetcdfVariableArray)

    def test_output_datastore_with_multiple_stands(self, mock_netcdf_file):
        selected_vars = [NetcdfVariablePath("/balance/fertilization_release")]
        metadata_by_stand = {
            StandID("stand_A"): pd.DataFrame(
                {
                    "experiment_id": ["scen_A1"],
                    "netcdf_output_filepath": [mock_netcdf_file],
                }
            ),
            StandID("stand_B"): pd.DataFrame(
                {
                    "experiment_id": ["scen_B1", "scen_B2"],
                    "netcdf_output_filepath": [mock_netcdf_file, mock_netcdf_file],
                }
            ),
        }
        store = read_netcdf_files_for_selected_variables(
            selected_variables=selected_vars,
            metadata_by_stand=metadata_by_stand,
        )

        assert store.stands == [StandID("stand_A"), StandID("stand_B")]
        assert store.scenarios[StandID("stand_A")] == [ScenarioID("scen_A1")]
        assert store.scenarios[StandID("stand_B")] == [
            ScenarioID("scen_B1"),
            ScenarioID("scen_B2"),
        ]
