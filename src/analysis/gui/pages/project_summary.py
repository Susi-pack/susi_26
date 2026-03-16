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
    load_output.NetcdfVariablePath("/strip/dwtyr"),
    load_output.NetcdfVariablePath("/stand/volumegrowth"),
    load_output.NetcdfVariablePath("/export/hmwtoditch"),
    load_output.NetcdfVariablePath("/export/lmwtoditch"),
    load_output.NetcdfVariablePath("/groundvegetation/ds_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/h_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/s_litterfall"),
    load_output.NetcdfVariablePath("/stand/nonwoodylitter"),
    load_output.NetcdfVariablePath("/stand/woodylitter"),
    load_output.NetcdfVariablePath("/esom/Mass/out"),
    load_output.NetcdfVariablePath("/groundvegetation/gv_tot"),
    load_output.NetcdfVariablePath("/stand/biomass"),
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
        selected_variables=CHOSEN_VARIABLES, metadata_by_stand=metadata_by_stand
    )
)

rows = []
for stand_id in data_store.stands:
    for scenario_id in data_store.scenarios[stand_id]:
        row: dict[str, str | int | float] = {
            "stand": str(stand_id),
            "scenario": str(scenario_id),
        }
        for var_path in data_store.variables:
            var_array = data_store.get_variable_value_for_scenario_and_stand(
                var_path, stand_id, scenario_id
            )
            row[str(var_path)] = var_array.mean_of_all_values()
        rows.append(row)

df_means = pd.DataFrame(rows)

df_means = df_means.sort_values(by=["stand", "scenario"])

st.dataframe(df_means, height=800)
