import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import netCDF4
from tempfile import TemporaryDirectory
import json

from susi.io.netcdf_utils import (
    list_all_netcdf_variables,
    list_variable_absolute_paths,
    _get_variable_by_path,
    read_value_several_variables_from_single_file,
    choose_netcdf_vars_by_path,
    coerce_datetime_format,
    modify_after_load,
    list_subdirectories,
    load_single_experiment_metadatas,
    load_all_metadatas_from_single_folder,
    load_all_metadatas_from_folders,
    transform_list_of_scenarios_to_optimization_array_structure,
    transform_array_data_to_list_of_scenarios,
    NetcdfVariableInfo,
    ScenarioArrayData,
    NetcdfVariablePath,
    ScenarioName,
    TargetVariableDict,
)

from susi.io.load_output_data import NetcdfVariableArray


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


def test_list_variable_absolute_paths(mock_netcdf_file):
    with netCDF4.Dataset(mock_netcdf_file, "r") as nc:
        paths = list_variable_absolute_paths(nc)
    expected = [
        "/temperature",
        "/balance/fertilization_release",
        "/balance/nested/nested_var",
    ]
    assert set(paths) == set(expected)


def test_get_variable_by_path_internal(mock_netcdf_file):
    with netCDF4.Dataset(mock_netcdf_file, "r") as nc:
        var = _get_variable_by_path(nc, ["temperature"])
        assert var.shape == (10, 5, 5)

        var = _get_variable_by_path(nc, ["balance", "fertilization_release"])
        assert var.shape == (3, 2)

        var = _get_variable_by_path(nc, ["balance", "nested", "nested_var"])
        assert var.shape == (4,)


def test_read_value_several_variables_from_single_file(mock_netcdf_file):
    variables = list_all_netcdf_variables(mock_netcdf_file)
    chosen = choose_netcdf_vars_by_path(
        [
            NetcdfVariablePath("/temperature"),
            NetcdfVariablePath("/balance/fertilization_release"),
        ],
        variables,
    )

    values = read_value_several_variables_from_single_file(mock_netcdf_file, chosen)

    assert len(values) == 2
    paths = {v.path for v in values}
    assert paths == {"/temperature", "/balance/fertilization_release"}

    temp_val = next(v for v in values if v.path == "/temperature")
    assert temp_val.value.shape == (10, 5, 5)

    fert_val = next(v for v in values if v.path == "/balance/fertilization_release")
    assert fert_val.value.shape == (3, 2)


def test_choose_netcdf_vars_by_path():
    variables = [
        NetcdfVariableInfo(
            path=NetcdfVariablePath("/temp"),
            name="temp",
            dimension_names=("x",),
            shape=(5,),
            units="K",
        ),
        NetcdfVariableInfo(
            path=NetcdfVariablePath("/humidity"),
            name="humidity",
            dimension_names=("x",),
            shape=(5,),
            units="%",
        ),
        NetcdfVariableInfo(
            path=NetcdfVariablePath("/pressure"),
            name="pressure",
            dimension_names=("x",),
            shape=(5,),
            units="hPa",
        ),
    ]

    chosen = choose_netcdf_vars_by_path(
        [NetcdfVariablePath("/temp"), NetcdfVariablePath("/pressure")], variables
    )

    assert len(chosen) == 2
    assert {v.path for v in chosen} == {"/temp", "/pressure"}


def test_choose_netcdf_vars_by_path_empty():
    variables = [
        NetcdfVariableInfo(
            path=NetcdfVariablePath("/temp"),
            name="temp",
            dimension_names=("x",),
            shape=(5,),
            units="K",
        ),
    ]

    chosen = choose_netcdf_vars_by_path(
        (NetcdfVariablePath("/nonexistent"),), variables
    )

    assert len(chosen) == 0


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

    result = load_single_experiment_metadatas(exp_folder)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["experiment_id"] == "exp1"
    assert result.iloc[0]["param1"] == 1.0


def test_load_all_metadatas_from_single_folder(mock_experiment_folders):
    result = load_all_metadatas_from_single_folder(mock_experiment_folders)

    assert len(result) == 2
    assert set(result["experiment_id"]) == {"exp1", "exp2"}


def test_load_all_metadatas_from_folders(mock_experiment_folders):
    folder1 = mock_experiment_folders / "sub1"
    folder2 = mock_experiment_folders / "sub2"
    folder1.mkdir()
    folder2.mkdir()
    create_mock_experiment_folder(folder1, "exp3")
    create_mock_experiment_folder(folder2, "exp4")

    result = load_all_metadatas_from_folders([folder1, folder2])

    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.DataFrame)


def test_transform_list_of_scenarios_to_optimization_array_structure():
    vars_of_interest_by_stand = [
        {
            ScenarioName("scenario_A"): TargetVariableDict(
                {NetcdfVariablePath("/var1"): 1.0, NetcdfVariablePath("/var2"): 2.0}
            ),
            ScenarioName("scenario_B"): TargetVariableDict(
                {NetcdfVariablePath("/var1"): 3.0, NetcdfVariablePath("/var2"): 4.0}
            ),
        },
        {
            ScenarioName("scenario_C"): TargetVariableDict(
                {NetcdfVariablePath("/var1"): 5.0, NetcdfVariablePath("/var2"): 6.0}
            ),
        },
    ]

    result = transform_list_of_scenarios_to_optimization_array_structure(
        vars_of_interest_by_stand,
        n_stands=2,
        target_variable_paths=[
            NetcdfVariablePath("/var1"),
            NetcdfVariablePath("/var2"),
        ],
    )

    assert isinstance(result, ScenarioArrayData)
    assert len(result.arrays) == 2
    assert len(result.scenario_names) == 2

    assert result.arrays[0].shape == (2, 2)
    assert result.arrays[1].shape == (1, 2)

    assert result.scenario_names[0] == ["scenario_A", "scenario_B"]
    assert result.scenario_names[1] == ["scenario_C"]

    np.testing.assert_array_equal(result.arrays[0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(result.arrays[1], [[5.0, 6.0]])


def test_transform_array_data_to_list_of_scenarios():
    array_data = ScenarioArrayData(
        arrays=[np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0]])],
        scenario_names=[
            [ScenarioName("scenario_A"), ScenarioName("scenario_B")],
            [ScenarioName("scenario_C")],
        ],
    )

    result = transform_array_data_to_list_of_scenarios(
        array_data,
        n_stands=2,
        target_variable_paths=[
            NetcdfVariablePath("/var1"),
            NetcdfVariablePath("/var2"),
        ],
    )

    assert len(result) == 2

    assert result[0][ScenarioName("scenario_A")] == TargetVariableDict(
        {
            NetcdfVariablePath("/var1"): 1.0,
            NetcdfVariablePath("/var2"): 2.0,
        }
    )
    assert result[0][ScenarioName("scenario_B")] == TargetVariableDict(
        {
            NetcdfVariablePath("/var1"): 3.0,
            NetcdfVariablePath("/var2"): 4.0,
        }
    )
    assert result[1][ScenarioName("scenario_C")] == TargetVariableDict(
        {
            NetcdfVariablePath("/var1"): 5.0,
            NetcdfVariablePath("/var2"): 6.0,
        }
    )


def test_transform_roundtrip():
    original = [
        {
            ScenarioName("A"): TargetVariableDict(
                {NetcdfVariablePath("/v1"): 1.5, NetcdfVariablePath("/v2"): 2.5}
            ),
            ScenarioName("B"): TargetVariableDict(
                {NetcdfVariablePath("/v1"): 3.5, NetcdfVariablePath("/v2"): 4.5}
            ),
        },
        {
            ScenarioName("C"): TargetVariableDict(
                {NetcdfVariablePath("/v1"): 5.5, NetcdfVariablePath("/v2"): 6.5}
            ),
        },
    ]

    array_data = transform_list_of_scenarios_to_optimization_array_structure(
        original,
        n_stands=2,
        target_variable_paths=[NetcdfVariablePath("/v1"), NetcdfVariablePath("/v2")],
    )

    result = transform_array_data_to_list_of_scenarios(
        array_data,
        n_stands=2,
        target_variable_paths=[NetcdfVariablePath("/v1"), NetcdfVariablePath("/v2")],
    )

    assert result == original


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
        with pytest.raises(AssertionError):
            arr.last_timestep()

    def test_last_timestep_fails_for_1d(self):
        raw = np.array([1.0, 2.0, 3.0])
        arr = NetcdfVariableArray(raw)
        with pytest.raises(AssertionError):
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
