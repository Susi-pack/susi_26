import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from analysis.core.parse_outputs import (
    ParamName,
    find_differing_params,
    find_unique_params,
    retrieve_parameters_for_stand,
    retrieve_scenarios_for_stand,
)
from susi.io.load_output_data import ScenarioID, SimulationParamsFromJSON, StandID


@pytest.fixture
def mock_run_dirpath():
    with TemporaryDirectory() as tmp:
        run_dirpath = Path(tmp)
        stand_dir = run_dirpath / "stand_01"
        stand_dir.mkdir()

        # Create 2 scenarios with differing site_parameters/ditch_depth_west
        scenarios = [("scenario_a", [-0.2]), ("scenario_b", [-0.7])]
        for scen_id, ditch_val in scenarios:
            scen_dir = stand_dir / scen_id
            scen_dir.mkdir()
            susi_params = {
                "site_parameters": {
                    "ditch_depth_west": ditch_val,
                    "ditch_depth_east": [-0.2],
                },
                "weather_parameters": {"temp": 20},
            }
            # Write params.json (required by read_params_from_jsons)
            with open(scen_dir / "params.json", "w") as f:
                json.dump(susi_params, f)
            # Write metadata.json (required by read_params_from_jsons)
            with open(scen_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "stand_id": "stand_01",
                        "scenario_id": scen_id,
                        "simulation_folder_path": str(scen_dir),
                    },
                    f,
                )

        yield run_dirpath


class TestRetrieveScenariosForStand:
    def test_returns_empty_for_missing_stand(self, mock_run_dirpath):
        result = retrieve_scenarios_for_stand(StandID("stand_99"), mock_run_dirpath)
        assert result == {}

    def test_returns_all_scenario_folders(self, mock_run_dirpath):
        result = retrieve_scenarios_for_stand(StandID("stand_01"), mock_run_dirpath)
        assert ScenarioID("scenario_a") in result
        assert ScenarioID("scenario_b") in result
        assert (
            result[ScenarioID("scenario_a")]
            == mock_run_dirpath / "stand_01" / "scenario_a"
        )
        assert (
            result[ScenarioID("scenario_b")]
            == mock_run_dirpath / "stand_01" / "scenario_b"
        )

    def test_ignores_files_in_stand_dir(self, mock_run_dirpath):
        # Create a file (not directory) in stand dir
        (mock_run_dirpath / "stand_01" / "not_a_scenario.txt").touch()
        result = retrieve_scenarios_for_stand(StandID("stand_01"), mock_run_dirpath)
        assert len(result) == 2


class TestRetrieveParametersForStand:
    def test_returns_params_for_all_scenarios(self, mock_run_dirpath):
        result = retrieve_parameters_for_stand(StandID("stand_01"), mock_run_dirpath)
        assert ScenarioID("scenario_a") in result
        assert ScenarioID("scenario_b") in result
        assert isinstance(result[ScenarioID("scenario_a")], SimulationParamsFromJSON)

    def test_returns_correct_susi_params(self, mock_run_dirpath):
        result = retrieve_parameters_for_stand(StandID("stand_01"), mock_run_dirpath)
        scen_a_params = result[ScenarioID("scenario_a")].susi_params
        assert scen_a_params["site_parameters"]["ditch_depth_west"] == [-0.2]


class TestFindDifferingParams:
    def test_returns_only_differing_params(self, mock_run_dirpath):
        result = find_differing_params(StandID("stand_01"), mock_run_dirpath)
        # Only ditch_depth_west differs between scenarios
        assert ParamName("site_parameters/ditch_depth_west") in result
        # ditch_depth_east is identical across scenarios, should not be present
        assert ParamName("site_parameters/ditch_depth_east") not in result
        # weather_parameters/temp is identical, should not be present
        assert ParamName("weather_parameters/temp") not in result

    def test_maps_values_to_correct_scenarios(self, mock_run_dirpath):
        result = find_differing_params(StandID("stand_01"), mock_run_dirpath)
        param = ParamName("site_parameters/ditch_depth_west")
        # Convert list values to tuples for hashability as dict keys
        assert result[param][(-0.2,)] == [ScenarioID("scenario_a")]
        assert result[param][(-0.7,)] == [ScenarioID("scenario_b")]

    def test_returns_empty_when_all_params_identical(self):
        with TemporaryDirectory() as tmp:
            run_dirpath = Path(tmp)
            stand_dir = run_dirpath / "stand_01"
            stand_dir.mkdir()
            # Create 2 scenarios with identical params
            for scen_id in ["scenario_a", "scenario_b"]:
                scen_dir = stand_dir / scen_id
                scen_dir.mkdir()
                susi_params = {"site_parameters": {"ditch_depth_west": [-0.2]}}
                with open(scen_dir / "params.json", "w") as f:
                    json.dump(susi_params, f)
                with open(scen_dir / "metadata.json", "w") as f:
                    json.dump({"stand_id": "stand_01", "scenario_id": scen_id}, f)

            result = find_differing_params(StandID("stand_01"), run_dirpath)
            assert result == {}

    def test_handles_three_scenarios(self):
        with TemporaryDirectory() as tmp:
            run_dirpath = Path(tmp)
            stand_dir = run_dirpath / "stand_01"
            stand_dir.mkdir()
            # 3 scenarios with different values for one param
            scenarios = [
                ("scenario_a", [-0.2]),
                ("scenario_b", [-0.7]),
                ("scenario_c", [-0.2]),  # Same as scenario_a
            ]
            for scen_id, ditch_val in scenarios:
                scen_dir = stand_dir / scen_id
                scen_dir.mkdir()
                susi_params = {"site_parameters": {"ditch_depth_west": ditch_val}}
                with open(scen_dir / "params.json", "w") as f:
                    json.dump(susi_params, f)
                with open(scen_dir / "metadata.json", "w") as f:
                    json.dump({"stand_id": "stand_01", "scenario_id": scen_id}, f)

            result = find_differing_params(StandID("stand_01"), run_dirpath)
            param = ParamName("site_parameters/ditch_depth_west")
            assert result[param][(-0.2,)] == [
                ScenarioID("scenario_a"),
                ScenarioID("scenario_c"),
            ]
            assert result[param][(-0.7,)] == [ScenarioID("scenario_b")]

    def test_excludes_scenario_name_even_though_it_differs(self):
        with TemporaryDirectory() as tmp:
            run_dirpath = Path(tmp)
            stand_dir = run_dirpath / "stand_01"
            stand_dir.mkdir()
            # 2 scenarios that differ on both scenario_name (trivially, by
            # definition) and a real parameter (ditch_depth_west).
            scenarios = [
                ("scenario_a", [-0.2], "scenario_a"),
                ("scenario_b", [-0.7], "scenario_b"),
            ]
            for scen_id, ditch_val, scenario_name in scenarios:
                scen_dir = stand_dir / scen_id
                scen_dir.mkdir()
                susi_params = {
                    "site_parameters": {
                        "ditch_depth_west": ditch_val,
                        "scenario_name": scenario_name,
                    },
                }
                with open(scen_dir / "params.json", "w") as f:
                    json.dump(susi_params, f)
                with open(scen_dir / "metadata.json", "w") as f:
                    json.dump({"stand_id": "stand_01", "scenario_id": scen_id}, f)

            result = find_differing_params(StandID("stand_01"), run_dirpath)
            # Real differing parameter is present.
            assert ParamName("site_parameters/ditch_depth_west") in result
            # scenario_name differs across scenarios too, but must never be
            # surfaced as a differing parameter.
            assert ParamName("site_parameters/scenario_name") not in result


class TestFindUniqueParams:
    def test_returns_only_unique_params(self, mock_run_dirpath):
        result = find_unique_params(StandID("stand_01"), mock_run_dirpath)
        # ditch_depth_east is identical across scenarios (-0.2)
        assert ParamName("site_parameters/ditch_depth_east") in result
        assert result[ParamName("site_parameters/ditch_depth_east")] == (-0.2,)
        # weather_parameters/temp is identical (20)
        assert ParamName("weather_parameters/temp") in result
        assert result[ParamName("weather_parameters/temp")] == 20
        # ditch_depth_west differs, should not be present
        assert ParamName("site_parameters/ditch_depth_west") not in result

    def test_returns_empty_when_all_params_differ(self):
        with TemporaryDirectory() as tmp:
            run_dirpath = Path(tmp)
            stand_dir = run_dirpath / "stand_01"
            stand_dir.mkdir()
            # 2 scenarios with different params for all parameters
            scenarios = [("scenario_a", [-0.2], 20), ("scenario_b", [-0.7], 25)]
            for scen_id, ditch_val, temp in scenarios:
                scen_dir = stand_dir / scen_id
                scen_dir.mkdir()
                susi_params = {
                    "site_parameters": {"ditch_depth_west": ditch_val},
                    "weather_parameters": {"temp": temp},
                }
                with open(scen_dir / "params.json", "w") as f:
                    json.dump(susi_params, f)
                with open(scen_dir / "metadata.json", "w") as f:
                    json.dump({"stand_id": "stand_01", "scenario_id": scen_id}, f)

            result = find_unique_params(StandID("stand_01"), run_dirpath)
            assert result == {}

    def test_returns_empty_for_missing_stand(self, mock_run_dirpath):
        result = find_unique_params(StandID("stand_99"), mock_run_dirpath)
        assert result == {}

    def test_handles_three_scenarios_unique(self):
        with TemporaryDirectory() as tmp:
            run_dirpath = Path(tmp)
            stand_dir = run_dirpath / "stand_01"
            stand_dir.mkdir()
            # 3 scenarios, one param unique across all
            scenarios = [
                ("scenario_a", [-0.2], 20),
                ("scenario_b", [-0.7], 20),
                ("scenario_c", [-0.5], 20),
            ]
            for scen_id, ditch_val, temp in scenarios:
                scen_dir = stand_dir / scen_id
                scen_dir.mkdir()
                susi_params = {
                    "site_parameters": {"ditch_depth_west": ditch_val},
                    "weather_parameters": {"temp": temp},
                }
                with open(scen_dir / "params.json", "w") as f:
                    json.dump(susi_params, f)
                with open(scen_dir / "metadata.json", "w") as f:
                    json.dump({"stand_id": "stand_01", "scenario_id": scen_id}, f)

            result = find_unique_params(StandID("stand_01"), run_dirpath)
            # temp is unique (20) across all 3 scenarios
            assert ParamName("weather_parameters/temp") in result
            assert result[ParamName("weather_parameters/temp")] == 20
            # ditch_depth_west differs, should not be present
            assert ParamName("site_parameters/ditch_depth_west") not in result
