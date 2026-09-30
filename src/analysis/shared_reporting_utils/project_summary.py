"""
Shared aggregation logic for the "project summary" table, reused by both
the Streamlit page (`analysis.streamlit.pages.project_summary`) and the
notebook renderer (`analysis.notebooks.components.summary_table`).

Extracted per #230 to eliminate the duplicated fixed-variable-list tuple and
row-building loop that used to live independently in each frontend.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from susi.io.load_output_data import (
    NetcdfAggregationFn,
    NetcdfVariableArray,
    NetcdfVariablePath,
    OutputDataStore,
)


@dataclass(frozen=True)
class AggregatedNetcdfVar:
    netcdf_path: NetcdfVariablePath
    aggregation_method: NetcdfAggregationFn
    display_label: str | None = (
        None  # Label to be displayed. If None, the netcdf_path is used.
    )


def aggregate_var(
    # Helper to reduce verbosity
    path: str,
    fn: NetcdfAggregationFn,
    label: str | None = None,
) -> AggregatedNetcdfVar:
    return AggregatedNetcdfVar(
        netcdf_path=NetcdfVariablePath(path),
        aggregation_method=fn,
        display_label=label,
    )


# Aggregation function shorthands, used below to keep SUMMARY_VARIABLES concise.
_mean = NetcdfVariableArray.mean_of_all_values
_end = NetcdfVariableArray.spatial_mean_at_last_timestep
_sum = NetcdfVariableArray.mean_over_space_sum_over_time
_initial = NetcdfVariableArray.spatial_mean_at_initial_timestep


# Fixed variable list shown in the project summary table, shared by both the
# Streamlit page and the notebook renderer.
SUMMARY_VARIABLES = (
    aggregate_var("strip/dwtyr_growingseason", _mean),
    aggregate_var("/strip/dwtyr_latesummer", _mean),
    aggregate_var("/stand/volumegrowth", _mean),
    aggregate_var("/stand/volume", _end, label="/stand/volume END"),
    aggregate_var("/stand/volume", _initial, label="/stand/volume INITIAL"),
    aggregate_var("/stand/logvolume", _end),
    aggregate_var("/stand/pulpvolume", _end),
    aggregate_var("/stand/harvested_volume", _sum),
    aggregate_var("/stand/harvested_log_volume", _sum),
    aggregate_var("/stand/harvested_pulp_volume", _sum),
    aggregate_var("/export/hmwtoditch", _mean),
    aggregate_var("/export/lmwtoditch", _mean),
    aggregate_var("/balance/C/stand_c_balance_co2eq", _mean),
    aggregate_var("/balance/C/soil_c_balance_co2eq", _mean),
    aggregate_var("/balance/N/balance_root_lyr", _mean),
    aggregate_var("/balance/P/balance_root_lyr", _mean),
    aggregate_var("/balance/K/balance_root_lyr", _mean),
    aggregate_var("/balance/N/to_water", _mean),
    aggregate_var("/balance/P/to_water", _mean),
    aggregate_var("/balance/K/to_water", _mean),
)


def build_summary_dataframe(
    data_store: OutputDataStore,
    chosen_variables: Sequence[AggregatedNetcdfVar],
) -> pd.DataFrame:
    """
    Build the per-stand/scenario aggregated summary table.

    One row per (stand, scenario) pair in `data_store`, with "stand" and
    "scenario" string columns plus one column per `chosen_variables` entry
    (keyed by its `display_label` if set, else `str(netcdf_path)`), holding
    the result of applying its `aggregation_method` to the corresponding
    variable array.

    Returns the DataFrame sorted by ["stand", "scenario"].
    """
    rows = []
    for stand_id in data_store.stands:
        for scenario_id in data_store.scenarios[stand_id]:
            row: dict[str, str | int | float] = {
                "stand": str(stand_id),
                "scenario": str(scenario_id),
            }
            for agg_var in chosen_variables:
                var_array = data_store.get_variable_value_for_scenario_and_stand(
                    agg_var.netcdf_path, stand_id, scenario_id
                )
                label = (
                    agg_var.display_label
                    if agg_var.display_label is not None
                    else str(agg_var.netcdf_path)
                )
                row[label] = agg_var.aggregation_method(var_array)

            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(by=["stand", "scenario"])
