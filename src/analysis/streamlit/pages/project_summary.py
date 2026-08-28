import streamlit as st

import pandas as pd

from analysis.streamlit.components import folder_selection, netcdf_reader
import susi.io.load_output_data as load_output

st.header("Project summary")

dir_path = folder_selection.build_folder_selection_widget(
    dir_path=st.session_state.settings["data_folder"], label="project"
)

stand_folderpaths = load_output.list_subdirectories(path=dir_path)

metadata_by_stand = load_output.load_all_metadatas_from_stands(
    folders=stand_folderpaths
)


mean = load_output.NetcdfVariableArray.mean_of_all_values
end = load_output.NetcdfVariableArray.spatial_mean_at_last_timestep
sum = load_output.NetcdfVariableArray.mean_over_space_sum_over_time
initial = load_output.NetcdfVariableArray.spatial_mean_at_initial_timestep


CHOSEN_VARIABLES = (
    netcdf_reader.aggregate_var("strip/dwtyr_growingseason", mean),
    netcdf_reader.aggregate_var("/strip/dwtyr_latesummer", mean),
    netcdf_reader.aggregate_var("/stand/volumegrowth", mean),
    netcdf_reader.aggregate_var("/stand/volume", end, label="/stand/volume END"),
    netcdf_reader.aggregate_var(
        "/stand/volume", initial, label="/stand/volume INITIAL"
    ),
    netcdf_reader.aggregate_var("/stand/logvolume", end),
    netcdf_reader.aggregate_var("/stand/pulpvolume", end),
    netcdf_reader.aggregate_var("/stand/harvested_volume", sum),
    netcdf_reader.aggregate_var("/stand/harvested_log_volume", sum),
    netcdf_reader.aggregate_var("/stand/harvested_pulp_volume", sum),
    netcdf_reader.aggregate_var("/export/hmwtoditch", mean),
    netcdf_reader.aggregate_var("/export/lmwtoditch", mean),
    netcdf_reader.aggregate_var("/balance/C/stand_c_balance_co2eq", mean),
    netcdf_reader.aggregate_var("/balance/C/soil_c_balance_co2eq", mean),
    netcdf_reader.aggregate_var("/balance/N/balance_root_lyr", mean),
    netcdf_reader.aggregate_var("/balance/P/balance_root_lyr", mean),
    netcdf_reader.aggregate_var("/balance/K/balance_root_lyr", mean),
    netcdf_reader.aggregate_var("/balance/N/to_water", mean),
    netcdf_reader.aggregate_var("/balance/P/to_water", mean),
    netcdf_reader.aggregate_var("/balance/K/to_water", mean),
)

# We don't use the function
# read_netcdf_files_for_selected_variables_from_metadatas()
# here because I want to cache to make UI faster
scenarios_by_stand = {}
netcdf_filepaths_by_stand = {}

for stand_id, metadata_df in metadata_by_stand.items():
    # We rely on pandas dataframe ordering for the two orders from
    # scenarios and netcdf_filepaths to match
    scenarios = load_output.get_scenarios_for_stand(metadata_df)
    netcdf_filepaths = load_output.get_netcdf_filepaths_for_stand(metadata_df)

    scenarios_by_stand[load_output.StandID(stand_id)] = scenarios
    netcdf_filepaths_by_stand[load_output.StandID(stand_id)] = netcdf_filepaths

data_store: load_output.OutputDataStore = netcdf_reader.cached_read_netcdf_files(
    selected_variables=CHOSEN_VARIABLES,
    scenarios_by_stand=scenarios_by_stand,
    netcdf_filepaths_by_stand=netcdf_filepaths_by_stand,
)

rows = []
for stand_id in data_store.stands:
    for scenario_id in data_store.scenarios[stand_id]:
        row: dict[str, str | int | float] = {
            "stand": str(stand_id),
            "scenario": str(scenario_id),
        }
        for agg_netcdf_var in CHOSEN_VARIABLES:
            var_array = data_store.get_variable_value_for_scenario_and_stand(
                agg_netcdf_var.netcdf_path, stand_id, scenario_id
            )
            label = (
                agg_netcdf_var.display_label
                if agg_netcdf_var.display_label is not None
                else str(agg_netcdf_var.netcdf_path)
            )
            row[label] = agg_netcdf_var.aggregation_method(var_array)

        rows.append(row)

df_agg = pd.DataFrame(rows)

df_agg = df_agg.sort_values(by=["stand", "scenario"])

st.dataframe(df_agg)
