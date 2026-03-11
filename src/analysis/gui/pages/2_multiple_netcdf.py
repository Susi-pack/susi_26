import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

import susi.io.load_output_data as load_output
from susi.io.load_output_data import NetcdfVariablePath

from analysis.gui.components import plots, folder_selection

chosen_scenario_folder = folder_selection.build_folder_selection_widget(
    dir_path=folder_selection.build_folder_selection_widget(
        dir_path=st.session_state.settings["data_folder"], label="project"
    ),
    label="stand",
)


chosen_susi_folders = st.multiselect(
    label="Choose 2 SUSI netcdf files to compare",
    options=load_output.list_subdirectories(chosen_scenario_folder),
    max_selections=2,
)

if len(chosen_susi_folders) == 2:
    metadata_0, susi_params_0 = load_output.read_json_metadatas(
        experiment_folderpath=chosen_susi_folders[0]
    )
    metadata_1, susi_params_1 = load_output.read_json_metadatas(
        experiment_folderpath=chosen_susi_folders[1]
    )

    chosen_netcdf_filepath_0 = Path(metadata_0["netcdf_output_filepath"])
    chosen_netcdf_filepath_1 = Path(metadata_1["netcdf_output_filepath"])

    VARIABLE_PATHS = (
        NetcdfVariablePath("/strip/dwtyr"),
        NetcdfVariablePath("/stand/volumegrowth"),
        NetcdfVariablePath("/export/hmwtoditch"),
        NetcdfVariablePath("/export/lmwtoditch"),
        NetcdfVariablePath("/groundvegetation/ds_litterfall"),
        NetcdfVariablePath("/groundvegetation/h_litterfall"),
        NetcdfVariablePath("/groundvegetation/s_litterfall"),
        NetcdfVariablePath("/stand/nonwoodylitter"),
        NetcdfVariablePath("/stand/woodylitter"),
        NetcdfVariablePath("/esom/Mass/out"),
        NetcdfVariablePath("/groundvegetation/gv_tot"),
        NetcdfVariablePath("/stand/biomass"),
    )

    all_variables = load_output.list_all_netcdf_variables(chosen_netcdf_filepath_0)

    variables_values_0 = load_output.read_value_several_variables_from_single_file(
        netcdf_filepath=chosen_netcdf_filepath_0,
        variable_paths=VARIABLE_PATHS,
    )

    variables_values_1 = load_output.read_value_several_variables_from_single_file(
        netcdf_filepath=chosen_netcdf_filepath_1,
        variable_paths=VARIABLE_PATHS,
    )

    st.markdown("## Comparison")
    fig = plots.compare_runs(variables_values_0, variables_values_1)
    st.pyplot(fig)
    plt.close(fig)
elif len(chosen_susi_folders) == 1:
    st.info("Please select one more folder to compare.")
else:
    st.info("Please select exactly 2 folders to compare.")
