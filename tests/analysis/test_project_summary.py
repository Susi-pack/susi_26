import numpy as np
import pandas as pd

from analysis.shared_reporting_utils.project_summary import (
    SUMMARY_VARIABLES,
    aggregate_var,
    build_summary_dataframe,
)
from susi.io.load_output_data import (
    NetcdfVariableArray,
    NetcdfVariablePath,
    OutputDataStore,
    ScenarioID,
    StandID,
)


def _make_data_store() -> OutputDataStore:
    """
    A small hand-built OutputDataStore with 2 stands (one with 2 scenarios,
    one with 1) and 2 variables, deliberately listed out of
    stand/scenario-sorted order so build_summary_dataframe's own sort can be
    exercised/asserted on.
    """
    path_one = NetcdfVariablePath("/var/one")
    path_two = NetcdfVariablePath("/var/two")

    stand_a = StandID("stand_A")
    stand_b = StandID("stand_B")
    scen_1 = ScenarioID("scen_1")
    scen_2 = ScenarioID("scen_2")

    # 2-D (time x space) raw arrays -- required by
    # spatial_mean_at_last_timestep, exercised via path_two below.
    arrays = {
        (stand_a, scen_1): {
            path_one: NetcdfVariableArray(raw=np.array([[1.0, 2.0], [3.0, 4.0]])),
            path_two: NetcdfVariableArray(raw=np.array([[10.0, 20.0], [30.0, 40.0]])),
        },
        (stand_a, scen_2): {
            path_one: NetcdfVariableArray(raw=np.array([[5.0, 6.0], [7.0, 8.0]])),
            path_two: NetcdfVariableArray(raw=np.array([[50.0, 60.0], [70.0, 80.0]])),
        },
        (stand_b, scen_1): {
            path_one: NetcdfVariableArray(raw=np.array([[9.0, 10.0], [11.0, 12.0]])),
            path_two: NetcdfVariableArray(
                raw=np.array([[90.0, 100.0], [110.0, 120.0]])
            ),
        },
    }

    data = {
        path_one: {key: arrs[path_one] for key, arrs in arrays.items()},
        path_two: {key: arrs[path_two] for key, arrs in arrays.items()},
    }

    return OutputDataStore(
        # Deliberately out of alphabetical order.
        stands=[stand_b, stand_a],
        scenarios={
            stand_a: [scen_2, scen_1],  # also out of order
            stand_b: [scen_1],
        },
        variables=[path_one, path_two],
        data=data,
    )


class TestBuildSummaryDataframeShape:
    def test_one_row_per_stand_scenario_pair(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var("/var/one", NetcdfVariableArray.mean_of_all_values),
        )

        df = build_summary_dataframe(data_store, chosen_variables)

        # stand_A has 2 scenarios, stand_B has 1 -> 3 rows total
        assert df.shape[0] == 3

    def test_columns_are_stand_scenario_plus_one_per_variable(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var("/var/one", NetcdfVariableArray.mean_of_all_values),
            aggregate_var(
                "/var/two",
                NetcdfVariableArray.spatial_mean_at_last_timestep,
                label="Var Two End",
            ),
        )

        df = build_summary_dataframe(data_store, chosen_variables)

        assert list(df.columns) == ["stand", "scenario", "/var/one", "Var Two End"]


class TestBuildSummaryDataframeLabeling:
    def test_default_label_is_str_of_netcdf_path(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var("/var/one", NetcdfVariableArray.mean_of_all_values),
        )

        df = build_summary_dataframe(data_store, chosen_variables)

        assert "/var/one" in df.columns

    def test_explicit_label_overrides_default(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var(
                "/var/two",
                NetcdfVariableArray.spatial_mean_at_last_timestep,
                label="Var Two End",
            ),
        )

        df = build_summary_dataframe(data_store, chosen_variables)

        assert "Var Two End" in df.columns
        assert "/var/two" not in df.columns


class TestBuildSummaryDataframeValuesAndSortOrder:
    def test_rows_sorted_by_stand_then_scenario(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var("/var/one", NetcdfVariableArray.mean_of_all_values),
        )

        df = build_summary_dataframe(data_store, chosen_variables)

        assert list(zip(df["stand"], df["scenario"])) == [
            ("stand_A", "scen_1"),
            ("stand_A", "scen_2"),
            ("stand_B", "scen_1"),
        ]

    def test_aggregated_values_are_correct(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var("/var/one", NetcdfVariableArray.mean_of_all_values),
            aggregate_var(
                "/var/two",
                NetcdfVariableArray.spatial_mean_at_last_timestep,
                label="Var Two End",
            ),
        )

        df = build_summary_dataframe(data_store, chosen_variables)
        df = df.set_index(["stand", "scenario"])

        # mean_of_all_values over [[1,2],[3,4]] == 2.5
        assert df.loc[("stand_A", "scen_1"), "/var/one"] == 2.5
        # mean_of_all_values over [[5,6],[7,8]] == 6.5
        assert df.loc[("stand_A", "scen_2"), "/var/one"] == 6.5
        # mean_of_all_values over [[9,10],[11,12]] == 10.5
        assert df.loc[("stand_B", "scen_1"), "/var/one"] == 10.5

        # spatial_mean_at_last_timestep over [[10,20],[30,40]] -> mean(30,40) == 35.0
        assert df.loc[("stand_A", "scen_1"), "Var Two End"] == 35.0
        # over [[50,60],[70,80]] -> mean(70,80) == 75.0
        assert df.loc[("stand_A", "scen_2"), "Var Two End"] == 75.0
        # over [[90,100],[110,120]] -> mean(110,120) == 115.0
        assert df.loc[("stand_B", "scen_1"), "Var Two End"] == 115.0

    def test_returns_a_dataframe(self):
        data_store = _make_data_store()
        chosen_variables = (
            aggregate_var("/var/one", NetcdfVariableArray.mean_of_all_values),
        )

        df = build_summary_dataframe(data_store, chosen_variables)

        assert isinstance(df, pd.DataFrame)


class TestSummaryVariables:
    def test_has_twenty_entries(self):
        # Regression check: SUMMARY_VARIABLES must stay identical to the
        # previously-duplicated CHOSEN_VARIABLES tuples (20 entries).
        assert len(SUMMARY_VARIABLES) == 23
