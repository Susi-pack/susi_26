from susi.io.load_output_data import ScenarioID
from analysis.core.parse_outputs import ParamName
from analysis.shared_reporting_utils.param_comparison import shape_differing_params


class TestShapeDifferingParams:
    def test_shapes_one_param_with_multiple_values_into_rows(self):
        differing = {
            ParamName("site_parameters/some_param"): {
                1: [ScenarioID("scenario_a"), ScenarioID("scenario_b")],
                2: [ScenarioID("scenario_c")],
            },
        }

        result = shape_differing_params(differing)

        assert result == {
            ParamName("site_parameters/some_param"): [
                {"value": 1, "scenarios": "scenario_a, scenario_b"},
                {"value": 2, "scenarios": "scenario_c"},
            ],
        }

    def test_shapes_multiple_params_independently(self):
        differing = {
            ParamName("param_one"): {
                "x": [ScenarioID("scenario_a")],
            },
            ParamName("param_two"): {
                "y": [ScenarioID("scenario_b"), ScenarioID("scenario_c")],
            },
        }

        result = shape_differing_params(differing)

        assert result == {
            ParamName("param_one"): [{"value": "x", "scenarios": "scenario_a"}],
            ParamName("param_two"): [
                {"value": "y", "scenarios": "scenario_b, scenario_c"}
            ],
        }

    def test_empty_input_yields_empty_output(self):
        assert shape_differing_params({}) == {}

    def test_single_scenario_value_has_no_trailing_separator(self):
        differing = {
            ParamName("param"): {"only_value": [ScenarioID("only_scenario")]},
        }

        result = shape_differing_params(differing)

        assert result[ParamName("param")][0]["scenarios"] == "only_scenario"
