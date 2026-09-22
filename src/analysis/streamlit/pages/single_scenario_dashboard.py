import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

import susi.io.load_output_data as load_output

from analysis.shared_reporting_utils.single_scenario_dashboard import (
    SECTIONS,
    VARIABLE_PATHS,
    load_report_data,
)
from analysis.streamlit.components import folder_selection

st.header("Choose project folder")

col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.markdown("**Projects root**")

with col2:
    st.write(st.session_state.settings["projects_root"])

with col3:
    if st.button("Browse…", use_container_width=True):
        result = folder_selection.pick_folder_popup()
        if result:
            st.session_state.settings["projects_root"] = result
            st.rerun()


chosen_scenario_folder = folder_selection.build_folder_selection_widget(
    dir_path=folder_selection.build_folder_selection_widget(
        dir_path=folder_selection.build_project_and_run_selection_widget(
            projects_root=st.session_state.settings["projects_root"]
        ),
        label="stand",
    ),
    label="scenario",
)


params = load_output.read_params_from_jsons(
    experiment_folderpath=chosen_scenario_folder
)

chosen_netcdf_filepath = Path(params.metadata["netcdf_output_filepath"])

data: dict[load_output.NetcdfVariablePath, load_output.NetcdfVariableArray] = (
    load_report_data(
        netcdf_filepath=chosen_netcdf_filepath, variable_paths=VARIABLE_PATHS
    )
)

for title, plot_fn in SECTIONS.values():
    st.markdown(f"## {title}")
    fig = plot_fn(data=data)
    st.pyplot(fig)
    plt.close(fig)
