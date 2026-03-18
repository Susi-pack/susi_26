import streamlit as st

import pandas as pd

from analysis.gui.components import folder_selection
import susi.io.load_output_data as load_output

st.header("Project summary")

dir_path = folder_selection.build_folder_selection_widget(
    dir_path=st.session_state.settings["data_folder"], label="project"
)

stand_folderpaths = load_output.list_subdirectories(path=dir_path)

metadata_by_stand = load_output.load_all_metadatas_from_stands(
    folders=stand_folderpaths
)


sample_netcdf_filepath = metadata_by_stand[
    load_output.StandID(stand_folderpaths[0].name)
].iloc[0]["netcdf_output_filepath"]


# all_variables = load_output.list_all_netcdf_variables(sample_netcdf_filepath)

CHOSEN_VARIABLES = (
    [load_output.NetcdfVariablePath("/strip/dwtyr_growingseason"), "mean"],             # mean
    [load_output.NetcdfVariablePath("/strip/dwtyr_latesummer"), "mean"],                # mean
    [load_output.NetcdfVariablePath("/stand/volumegrowth"), "mean"],                    # mean
    [load_output.NetcdfVariablePath("/stand/volume"),  "end"],                           # end
    [load_output.NetcdfVariablePath("/stand/volume"),  "initial"],                           # end
    [load_output.NetcdfVariablePath("/stand/logvolume"), "end"],                       # end   
    [load_output.NetcdfVariablePath("/stand/pulpvolume"), "end"],                      # end    
    [load_output.NetcdfVariablePath("/stand/harvested_volume"), "sum"],                # sum
    [load_output.NetcdfVariablePath("/stand/harvested_log_volume"), "sum"],             # sum
    [load_output.NetcdfVariablePath("/stand/harvested_pulp_volume"), "sum"],            # sum
    [load_output.NetcdfVariablePath("/export/hmwtoditch"), "mean"],                      # mean 
    [load_output.NetcdfVariablePath("/export/lmwtoditch"),"mean"],                      # mean
    [load_output.NetcdfVariablePath("/balance/C/stand_c_balance_co2eq"),"mean"],        # mean
    [load_output.NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"), "mean"],        # mean
    [load_output.NetcdfVariablePath("/balance/N/balance_root_lyr"), "mean"],            # mean
    [load_output.NetcdfVariablePath("/balance/P/balance_root_lyr"),"mean"],             # mean
    [load_output.NetcdfVariablePath("/balance/K/balance_root_lyr"),"mean"],             # mean
    [load_output.NetcdfVariablePath("/balance/N/to_water"),"mean"],             # mean
    [load_output.NetcdfVariablePath("/balance/P/to_water"),"mean"],             # mean
    [load_output.NetcdfVariablePath("/balance/K/to_water"),"mean"],             # mean
)


# TODO: (Later) Try to cache this
# cached_read_of_all_variables = lru_cache(
#     nc_utils.read_chosen_variables_from_netcdf_by_stands_and_scenarios
# )
# chosen_variables_by_stand_and_scenario = cached_read_of_all_variables(
#     chosen_vars=tuple(chosen_vars), metadata_by_stand=tuple(metadata_by_stand)
# )
# The issue with the approach above is that the dataframe object of metadata_by_stand is not cacheable.
# Maybe we can cache it with Streamlit directly?

data_store: load_output.OutputDataStore = (
    load_output.read_netcdf_files_for_selected_variables(
        selected_variables= [var[0] for var in CHOSEN_VARIABLES], metadata_by_stand=metadata_by_stand
    )
)

rows = []
for stand_id in data_store.stands:
    for scenario_id in data_store.scenarios[stand_id]:
        row: dict[str, str | int | float] = {
            "stand": str(stand_id),
            "scenario": str(scenario_id),
        }
        for var_path, method_name in CHOSEN_VARIABLES:
            var_array = data_store.get_variable_value_for_scenario_and_stand(
                var_path, stand_id, scenario_id
            )
            if method_name =="mean":
                row[str(var_path)] = var_array.mean_of_all_values()
            elif method_name == "end":
                row[str(var_path)] = var_array.spatial_mean_at_last_timestep()
            elif method_name == "sum":
                row[str(var_path)] = var_array.mean_over_space_sum_over_time()
            elif method_name == "initial":
                row[str(var_path + 'initial')] = var_array.spatial_mean_at_initial_timestep()
                       
        rows.append(row)

df_means = pd.DataFrame(rows)

df_means = df_means.sort_values(by=["stand", "scenario"])

st.dataframe(df_means, height=800)
