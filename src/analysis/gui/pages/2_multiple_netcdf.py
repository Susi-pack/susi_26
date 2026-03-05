import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

import susi.io.netcdf_utils as nc_utils

from analysis.gui.components import plots, folder_selection

chosen_scenario_folder = folder_selection.build_folder_selection_widget(
    dir_path=folder_selection.build_folder_selection_widget(
        dir_path=st.session_state.settings["data_folder"], label="project"
    ),
    label="stand",
)


chosen_susi_folders = st.multiselect(
    label="Choose 2 SUSI netcdf files to compare",
    options=nc_utils.list_subdirectories(chosen_scenario_folder),
    max_selections=2,
)

if len(chosen_susi_folders) == 2:
    metadata_0, susi_params_0 = nc_utils.read_json_metadatas(
        experiment_folderpath=chosen_susi_folders[0]
    )
    metadata_1, susi_params_1 = nc_utils.read_json_metadatas(
        experiment_folderpath=chosen_susi_folders[1]
    )

    chosen_netcdf_filepath_0 = Path(metadata_0["netcdf_output_filepath"])
    chosen_netcdf_filepath_1 = Path(metadata_1["netcdf_output_filepath"])

    VARIABLE_PATHS = (
        nc_utils.NetcdfVariablePath("/strip/dwtyr"),
        nc_utils.NetcdfVariablePath("/stand/volumegrowth"),
        nc_utils.NetcdfVariablePath("/export/hmwtoditch"),
        nc_utils.NetcdfVariablePath("/export/lmwtoditch"),
        nc_utils.NetcdfVariablePath("/groundvegetation/ds_litterfall"),
        nc_utils.NetcdfVariablePath("/groundvegetation/h_litterfall"),
        nc_utils.NetcdfVariablePath("/groundvegetation/s_litterfall"),
        nc_utils.NetcdfVariablePath("/stand/nonwoodylitter"),
        nc_utils.NetcdfVariablePath("/stand/woodylitter"),
        nc_utils.NetcdfVariablePath("/esom/Mass/out"),
        nc_utils.NetcdfVariablePath("/groundvegetation/gv_tot"),
        nc_utils.NetcdfVariablePath("/stand/biomass"),
    )

    all_variables = nc_utils.list_all_netcdf_variables(chosen_netcdf_filepath_0)

    chosen_vars = nc_utils.choose_netcdf_vars_by_path(
        paths=VARIABLE_PATHS, all_variables=all_variables
    )

    variables_values_0 = nc_utils.read_value_several_variables_from_single_file(
        netcdf_filepath=chosen_netcdf_filepath_0,
        variables=chosen_vars,
    )

    variables_values_1 = nc_utils.read_value_several_variables_from_single_file(
        netcdf_filepath=chosen_netcdf_filepath_1,
        variables=chosen_vars,
    )

    scen = 0

    st.markdown("## Comparison")
    fig = plots.compare_runs(variables_values_0, variables_values_1, scen=scen)
    st.pyplot(fig)
    plt.close(fig)
elif len(chosen_susi_folders) == 1:
    st.info("Please select one more folder to compare.")
else:
    st.info("Please select exactly 2 folders to compare.")
