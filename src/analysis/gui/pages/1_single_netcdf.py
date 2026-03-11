import streamlit as st
from pathlib import Path

import susi.io.load_output_data as load_output
from susi.io.app_settings import AppSettings

from analysis.gui.components import (
    metadata_expander,
    netcdf_variable_explorer,
    netcdf_variable_plots_ui,
    folder_selection,
)

# %% Choose folder
st.header("Choose project folder")

col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.markdown("**Data folder**")

with col2:
    st.write(st.session_state.settings["data_folder"])

with col3:
    if st.button("Browse…", use_container_width=True):
        result = folder_selection.pick_folder_popup()
        if result:
            st.session_state.settings["data_folder"] = result
            st.rerun()


chosen_scenario_folder = folder_selection.build_folder_selection_widget(
    dir_path=folder_selection.build_folder_selection_widget(
        dir_path=folder_selection.build_folder_selection_widget(
            dir_path=st.session_state.settings["data_folder"], label="project"
        ),
        label="stand",
    ),
    label="scenario",
)


# %% Metadata expander
metadata, susi_params = load_output.read_json_metadatas(
    experiment_folderpath=chosen_scenario_folder
)

metadata_expander.build(metadata=metadata, susi_params=susi_params)

# %% Show summary table
st.markdown("---")
st.subheader("Summary")

st.write("summary table will go here")

# %% Read Netcdf variabales
# The golden test netcdf is used to read the variable structure  of the netcdf file
sample_netcdf_filepath = (
    AppSettings().project_root_path / "tests/golden_file_test/golden_susi.nc"
)
all_variables = load_output.list_all_netcdf_variables(sample_netcdf_filepath)

# Actual netcdf file path for reading values (not only structure of the file)
chosen_netcdf_filepath = Path(metadata["netcdf_output_filepath"])

chosen_netcdf_variables = netcdf_variable_explorer.build(netcdf_variables=all_variables)


# %% Display selection summary and plots
st.markdown("---")
st.subheader("Plots")
if not chosen_netcdf_variables:
    st.write("No variables chosen")
else:
    variables_values = load_output.read_value_several_variables_from_single_file(
        netcdf_filepath=chosen_netcdf_filepath,
        variable_paths=list(chosen_netcdf_variables.keys()),
    )
    netcdf_variable_plots_ui.build(variables_values)
