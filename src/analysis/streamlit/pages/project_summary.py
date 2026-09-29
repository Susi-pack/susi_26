import streamlit as st

import susi.io.load_output_data as load_output
from analysis.shared_reporting_utils.project_summary import (
    SUMMARY_VARIABLES,
    build_summary_dataframe,
)
from analysis.streamlit.components import folder_selection, netcdf_reader

st.header("Project summary")

# The stands are one level below a run, not below the project itself.
dir_path = folder_selection.build_project_and_run_selection_widget(
    projects_root=st.session_state.settings["projects_root"]
).run_dir

stand_folderpaths = load_output.list_subdirectories(path=dir_path)

metadata_by_stand = load_output.load_all_metadatas_from_stands(
    folders=stand_folderpaths
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
    selected_variables=SUMMARY_VARIABLES,
    scenarios_by_stand=scenarios_by_stand,
    netcdf_filepaths_by_stand=netcdf_filepaths_by_stand,
)

df_agg = build_summary_dataframe(data_store, SUMMARY_VARIABLES)

st.dataframe(df_agg)
