"""
Tests for the optimization notebook's printed configuration.

The notebook prints the target specs chosen with its widgets as Python, so a
user can paste them into the notebook's in-code cell instead of clicking
through the widgets again. That only works if the printed code, run on its
own, rebuilds exactly the same target specs.
"""

from typing import Any

from analysis.notebooks.components import optimization
from analysis.optimization.core import Direction, TargetSpec
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath


def test_printed_target_specs_rebuild_the_same_specs_in_the_same_order():
    # Not in alphabetical order, and mixing directions and aggregations, so
    # that neither a sorted nor a uniform printout would pass by accident.
    target_specs = {
        NetcdfVariablePath("/stand/volume"): TargetSpec(
            aggregation=NetcdfVariableArray.spatial_mean_at_last_timestep,
            direction=Direction.MAXIMIZE,
        ),
        NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"): TargetSpec(
            aggregation=NetcdfVariableArray.mean_over_space_sum_over_time,
            direction=Direction.MINIMIZE,
        ),
        NetcdfVariablePath("/balance/N/to_water"): TargetSpec(
            aggregation=NetcdfVariableArray.mean_of_all_values,
            direction=Direction.MINIMIZE,
        ),
    }

    code = optimization.format_target_specs_as_code(target_specs)

    # Empty, so the printed code must bring its own imports, and a wrong
    # import path fails here.
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102 -- running the printed code is the test

    assert namespace["target_specs"] == target_specs
    assert list(namespace["target_specs"].keys()) == list(target_specs.keys())
