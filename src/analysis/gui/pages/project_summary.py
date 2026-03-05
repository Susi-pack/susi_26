import streamlit as st
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import colorsys

from analysis.gui.components import folder_selection
import susi.io.netcdf_utils as nc_utils

st.header("Project summary")

dir_path = folder_selection.build_folder_selection_widget(
    dir_path=st.session_state.settings["data_folder"], label="project"
)

stand_folderpaths = nc_utils.list_subdirectories(path=dir_path)

metadata_by_stand = nc_utils.load_all_metadatas_from_folders(folders=stand_folderpaths)

sample_netcdf_file_path = Path(metadata_by_stand[0]["netcdf_output_filepath"][0])

st.write(sample_netcdf_file_path)


CHOSEN_VARIABLES = (
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

all_variables = nc_utils.list_all_netcdf_variables(sample_netcdf_file_path)

chosen_vars = nc_utils.choose_netcdf_vars_by_path(
    paths=CHOSEN_VARIABLES, all_variables=all_variables
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

chosen_variables_by_stand_and_scenario = (
    nc_utils.read_chosen_variables_from_netcdf_by_stands_and_scenarios(
        chosen_vars=chosen_vars, metadata_by_stand=metadata_by_stand
    )
)

rows = []
for stand_idx, scenarios_dict in enumerate(chosen_variables_by_stand_and_scenario):
    stand_name = stand_folderpaths[stand_idx].name
    for scenario_name, var_values in scenarios_dict.items():
        row = {"stand": stand_name, "scenario": scenario_name}
        for var_val in var_values:
            var_path = str(var_val.path)
            row[var_path] = np.mean(var_val.value)
        rows.append(row)

df_means = pd.DataFrame(rows)

df_means = df_means.sort_values(by=["stand", "scenario"])

st.dataframe(df_means, height=800)
