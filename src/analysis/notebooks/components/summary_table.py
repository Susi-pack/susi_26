"""
Notebook equivalent of `analysis.gui.pages.project_summary`.

Per #215's picker/renderer split, this is a renderer element: it computes
and displays the per-stand/scenario aggregated summary table, and also
returns the underlying DataFrame so a later cell can reuse it without
recomputing.

Reuses the aggregation logic (`netcdf_reader.aggregate_var`, `OutputDataStore`)
unchanged, per #219. Drops the Streamlit page's `@st.cache_data` layer: a
notebook's natural cell-reuse -- rerunning this cell only when its inputs
change -- serves as the equivalent, so
`read_netcdf_files_for_selected_variables_from_metadatas` is called directly
instead of through `netcdf_reader.cached_read_netcdf_files`.

Typical usage:

    df_summary = summary_table.display_summary_table(project_dropdown.value)
"""

from pathlib import Path

import pandas as pd
from IPython.display import display

import susi.io.load_output_data as load_output
from analysis.gui.components import netcdf_reader

_mean = load_output.NetcdfVariableArray.mean_of_all_values
_end = load_output.NetcdfVariableArray.spatial_mean_at_last_timestep
_sum = load_output.NetcdfVariableArray.mean_over_space_sum_over_time
_initial = load_output.NetcdfVariableArray.spatial_mean_at_initial_timestep

# Same fixed variable list as the Streamlit page
# (analysis.gui.pages.project_summary). Copied rather than imported since
# it's defined inline there, not exported for reuse.
CHOSEN_VARIABLES = (
    netcdf_reader.aggregate_var("strip/dwtyr_growingseason", _mean),
    netcdf_reader.aggregate_var("/strip/dwtyr_latesummer", _mean),
    netcdf_reader.aggregate_var("/stand/volumegrowth", _mean),
    netcdf_reader.aggregate_var("/stand/volume", _end, label="/stand/volume END"),
    netcdf_reader.aggregate_var(
        "/stand/volume", _initial, label="/stand/volume INITIAL"
    ),
    netcdf_reader.aggregate_var("/stand/logvolume", _end),
    netcdf_reader.aggregate_var("/stand/pulpvolume", _end),
    netcdf_reader.aggregate_var("/stand/harvested_volume", _sum),
    netcdf_reader.aggregate_var("/stand/harvested_log_volume", _sum),
    netcdf_reader.aggregate_var("/stand/harvested_pulp_volume", _sum),
    netcdf_reader.aggregate_var("/export/hmwtoditch", _mean),
    netcdf_reader.aggregate_var("/export/lmwtoditch", _mean),
    netcdf_reader.aggregate_var("/balance/C/stand_c_balance_co2eq", _mean),
    netcdf_reader.aggregate_var("/balance/C/soil_c_balance_co2eq", _mean),
    netcdf_reader.aggregate_var("/balance/N/balance_root_lyr", _mean),
    netcdf_reader.aggregate_var("/balance/P/balance_root_lyr", _mean),
    netcdf_reader.aggregate_var("/balance/K/balance_root_lyr", _mean),
    netcdf_reader.aggregate_var("/balance/N/to_water", _mean),
    netcdf_reader.aggregate_var("/balance/P/to_water", _mean),
    netcdf_reader.aggregate_var("/balance/K/to_water", _mean),
)


def display_summary_table(dir_path: Path) -> pd.DataFrame:
    """
    Compute and display the per-stand/scenario summary table for dir_path.

    dir_path is a project folder containing one subfolder per stand, each
    holding one subfolder per scenario -- e.g. the `.value` of a project-level
    `folder_selection.build_dropdown()`.

    Returns the aggregated DataFrame (sorted by stand, then scenario) so a
    later cell can reuse it without recomputing.
    """
    stand_folderpaths = load_output.list_subdirectories(path=dir_path)
    metadata_by_stand = load_output.load_all_metadatas_from_stands(
        folders=stand_folderpaths
    )

    data_store = load_output.read_netcdf_files_for_selected_variables_from_metadatas(
        selected_variables=[var.netcdf_path for var in CHOSEN_VARIABLES],
        metadata_by_stand=metadata_by_stand,
    )

    rows = []
    for stand_id in data_store.stands:
        for scenario_id in data_store.scenarios[stand_id]:
            row: dict[str, str | int | float] = {
                "stand": str(stand_id),
                "scenario": str(scenario_id),
            }
            for agg_var in CHOSEN_VARIABLES:
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

    df_summary = pd.DataFrame(rows).sort_values(by=["stand", "scenario"])
    display(df_summary)
    return df_summary
