import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

import susi.io.load_output_data as load_output

from analysis.shared_reporting_utils import plots
from analysis.shared_reporting_utils.single_scenario_dashboard import (
    VARIABLE_PATHS,
    load_report_data,
)
from analysis.streamlit.components import folder_selection

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


params = load_output.read_params_from_jsons(
    experiment_folderpath=chosen_scenario_folder
)

chosen_netcdf_filepath = Path(params.metadata["netcdf_output_filepath"])
all_variables = load_output.list_all_netcdf_variables(chosen_netcdf_filepath)

data: dict[load_output.NetcdfVariablePath, load_output.NetcdfVariableArray] = (
    load_report_data(
        netcdf_filepath=chosen_netcdf_filepath, variable_paths=VARIABLE_PATHS
    )
)

st.markdown("## Stand")
fig_stand = plots.stand(data=data)
st.pyplot(fig_stand)
plt.close(fig_stand)

st.markdown("## Hydrology")
fig_hydro = plots.hydrology(data=data)
st.pyplot(fig_hydro)
plt.close(fig_hydro)

st.markdown("## Mass")
fig_mass = plots.mass(data=data)
st.pyplot(fig_mass)
plt.close(fig_mass)

st.markdown("## Carbon")
fig_carbon = plots.carbon(data=data)
st.pyplot(fig_carbon)
plt.close(fig_carbon)

st.markdown("## Nitrogen Balance")
fig_n = plots.nutrient_balance(data=data, substance="N")
st.pyplot(fig_n)
plt.close(fig_n)

st.markdown("## Phosphorus Balance")
fig_p = plots.nutrient_balance(data=data, substance="P")
st.pyplot(fig_p)
plt.close(fig_p)

st.markdown("## Potassium Balance")
fig_k = plots.nutrient_balance(data=data, substance="K")
st.pyplot(fig_k)
plt.close(fig_k)
