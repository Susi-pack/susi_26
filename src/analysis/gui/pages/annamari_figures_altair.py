import streamlit as st
from pathlib import Path

import susi.io.netcdf_utils as nc_utils
from susi.io.app_settings import AppSettings

from analysis.gui.components import plots_altair, folder_selection

chosen_scenario_folder = folder_selection.build_folder_selection_widget(
    dir_path=folder_selection.build_folder_selection_widget(
        dir_path=folder_selection.build_folder_selection_widget(
            dir_path=st.session_state.settings["data_folder"], label="project"
        ),
        label="stand",
    ),
    label="scenario",
)

metadata, susi_params = nc_utils.read_params_from_jsons(
    experiment_folderpath=chosen_scenario_folder
)

sample_netcdf_filepath = (
    AppSettings().project_root_path / "tests/golden_file_test/golden_susi.nc"
)
all_variables = nc_utils.list_all_netcdf_variables(sample_netcdf_filepath)

chosen_netcdf_filepath = Path(metadata["netcdf_output_filepath"])

VARIABLE_PATHS = (
    nc_utils.NetcdfVariablePath("/strip/dwtyr"),
    nc_utils.NetcdfVariablePath("/strip/dwtyr_growingseason"),
    nc_utils.NetcdfVariablePath("/strip/dwtyr_latesummer"),
    nc_utils.NetcdfVariablePath("/strip/dwt"),
    nc_utils.NetcdfVariablePath("/strip/roff"),
    nc_utils.NetcdfVariablePath("/strip/roffwest"),
    nc_utils.NetcdfVariablePath("/strip/roffeast"),
    nc_utils.NetcdfVariablePath("/strip/surfacerunoff"),
    nc_utils.NetcdfVariablePath("/strip/deltas"),
    nc_utils.NetcdfVariablePath("/strip/elevation"),
    nc_utils.NetcdfVariablePath("/cpy/ET_yr"),
    nc_utils.NetcdfVariablePath("/cpy/transpi_yr"),
    nc_utils.NetcdfVariablePath("/cpy/efloor_yr"),
    nc_utils.NetcdfVariablePath("/cpy/SWEmax"),
    nc_utils.NetcdfVariablePath("/cpy/interc_yr"),
    nc_utils.NetcdfVariablePath("/stand/volume"),
    nc_utils.NetcdfVariablePath("/stand/dominant/volume"),
    nc_utils.NetcdfVariablePath("/stand/subdominant/volume"),
    nc_utils.NetcdfVariablePath("/stand/under/volume"),
    nc_utils.NetcdfVariablePath("/stand/logvolume"),
    nc_utils.NetcdfVariablePath("/stand/pulpvolume"),
    nc_utils.NetcdfVariablePath("/stand/leafmass"),
    nc_utils.NetcdfVariablePath("/stand/dominant/leafmass"),
    nc_utils.NetcdfVariablePath("/stand/dominant/leafmax"),
    nc_utils.NetcdfVariablePath("/stand/dominant/leafmin"),
    nc_utils.NetcdfVariablePath("/stand/subdominant/leafmass"),
    nc_utils.NetcdfVariablePath("/stand/subdominant/leafmax"),
    nc_utils.NetcdfVariablePath("/stand/subdominant/leafmin"),
    nc_utils.NetcdfVariablePath("/stand/under/leafmass"),
    nc_utils.NetcdfVariablePath("/stand/under/leafmax"),
    nc_utils.NetcdfVariablePath("/stand/under/leafmin"),
    nc_utils.NetcdfVariablePath("/stand/nut_stat"),
    nc_utils.NetcdfVariablePath("/stand/dominant/NPP"),
    nc_utils.NetcdfVariablePath("/stand/dominant/NPP_pot"),
    nc_utils.NetcdfVariablePath("/stand/n_demand"),
    nc_utils.NetcdfVariablePath("/stand/p_demand"),
    nc_utils.NetcdfVariablePath("/stand/k_demand"),
    nc_utils.NetcdfVariablePath("/stand/nonwoodylitter"),
    nc_utils.NetcdfVariablePath("/stand/woodylitter"),
    nc_utils.NetcdfVariablePath("/stand/finerootlitter"),
    nc_utils.NetcdfVariablePath("/stand/biomass"),
    nc_utils.NetcdfVariablePath("/stand/volumegrowth"),
    nc_utils.NetcdfVariablePath("/groundvegetation/ds_litterfall"),
    nc_utils.NetcdfVariablePath("/groundvegetation/h_litterfall"),
    nc_utils.NetcdfVariablePath("/groundvegetation/s_litterfall"),
    nc_utils.NetcdfVariablePath("/groundvegetation/gv_tot"),
    nc_utils.NetcdfVariablePath("/esom/Mass/out"),
    nc_utils.NetcdfVariablePath("/esom/Mass/L0L"),
    nc_utils.NetcdfVariablePath("/esom/Mass/L0W"),
    nc_utils.NetcdfVariablePath("/esom/Mass/LL"),
    nc_utils.NetcdfVariablePath("/esom/Mass/LW"),
    nc_utils.NetcdfVariablePath("/esom/Mass/FL"),
    nc_utils.NetcdfVariablePath("/esom/Mass/FW"),
    nc_utils.NetcdfVariablePath("/esom/Mass/H"),
    nc_utils.NetcdfVariablePath("/esom/Mass/P1"),
    nc_utils.NetcdfVariablePath("/esom/Mass/P2"),
    nc_utils.NetcdfVariablePath("/esom/Mass/P3"),
    nc_utils.NetcdfVariablePath("/balance/C/LMWdoc_to_water"),
    nc_utils.NetcdfVariablePath("/balance/C/HMW_to_water"),
    nc_utils.NetcdfVariablePath("/balance/C/LMWdoc_to_atm"),
    nc_utils.NetcdfVariablePath("/balance/C/HMW_to_atm"),
    nc_utils.NetcdfVariablePath("/balance/C/co2c_release"),
    nc_utils.NetcdfVariablePath("/balance/C/ch4c_release"),
    nc_utils.NetcdfVariablePath("/balance/C/stand_litter_in"),
    nc_utils.NetcdfVariablePath("/balance/C/gv_litter_in"),
    nc_utils.NetcdfVariablePath("/balance/C/soil_c_balance_c"),
    nc_utils.NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
    nc_utils.NetcdfVariablePath("/balance/C/stand_c_balance_c"),
    nc_utils.NetcdfVariablePath("/balance/C/stand_c_balance_co2eq"),
    nc_utils.NetcdfVariablePath("/balance/N/to_water"),
    nc_utils.NetcdfVariablePath("/balance/N/decomposition_below_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/N/decomposition_tot"),
    nc_utils.NetcdfVariablePath("/balance/N/decomposition_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/N/deposition"),
    nc_utils.NetcdfVariablePath("/balance/N/fertilization_release"),
    nc_utils.NetcdfVariablePath("/balance/N/stand_demand"),
    nc_utils.NetcdfVariablePath("/balance/N/gv_demand"),
    nc_utils.NetcdfVariablePath("/balance/N/balance_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/P/to_water"),
    nc_utils.NetcdfVariablePath("/balance/P/decomposition_below_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/P/decomposition_tot"),
    nc_utils.NetcdfVariablePath("/balance/P/decomposition_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/P/deposition"),
    nc_utils.NetcdfVariablePath("/balance/P/fertilization_release"),
    nc_utils.NetcdfVariablePath("/balance/P/stand_demand"),
    nc_utils.NetcdfVariablePath("/balance/P/gv_demand"),
    nc_utils.NetcdfVariablePath("/balance/P/balance_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/K/to_water"),
    nc_utils.NetcdfVariablePath("/balance/K/decomposition_below_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/K/decomposition_tot"),
    nc_utils.NetcdfVariablePath("/balance/K/decomposition_root_lyr"),
    nc_utils.NetcdfVariablePath("/balance/K/deposition"),
    nc_utils.NetcdfVariablePath("/balance/K/fertilization_release"),
    nc_utils.NetcdfVariablePath("/balance/K/stand_demand"),
    nc_utils.NetcdfVariablePath("/balance/K/gv_demand"),
    nc_utils.NetcdfVariablePath("/balance/K/balance_root_lyr"),
)

chosen_vars = nc_utils.choose_netcdf_vars_by_path(
    paths=VARIABLE_PATHS, all_variables=all_variables
)

variables_values = nc_utils.read_value_several_variables_from_single_file(
    netcdf_filepath=chosen_netcdf_filepath,
    variables=chosen_vars,
)

scen = 0

st.markdown("## Stand")
charts = plots_altair.stand(variables_values, scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[3], width="stretch")
with col2:
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[6], width="stretch")
    st.altair_chart(charts[7], width="stretch")

st.altair_chart(charts[8], width="stretch")

col3, col4 = st.columns(2)
with col3:
    st.altair_chart(charts[9], width="stretch")
    st.altair_chart(charts[10], width="stretch")
with col4:
    st.altair_chart(charts[11], width="stretch")

col5, col6, col7 = st.columns(3)
with col5:
    st.altair_chart(charts[12], width="stretch")
    st.altair_chart(charts[15], width="stretch")
with col6:
    st.altair_chart(charts[13], width="stretch")
    st.altair_chart(charts[16], width="stretch")
with col7:
    st.altair_chart(charts[14], width="stretch")
    st.altair_chart(charts[17], width="stretch")

st.markdown("## Hydrology")
charts = plots_altair.hydrology(variables_values, scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[3], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[7], width="stretch")
    st.altair_chart(charts[9], width="stretch")
with col2:
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[6], width="stretch")
    st.altair_chart(charts[8], width="stretch")
    st.altair_chart(charts[10], width="stretch")

st.markdown("## Mass")
charts = plots_altair.mass(variables_values, scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[3], width="stretch")
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[6], width="stretch")
with col2:
    st.altair_chart(charts[7], width="stretch")
    st.altair_chart(charts[8], width="stretch")
    st.altair_chart(charts[9], width="stretch")
    st.altair_chart(charts[10], width="stretch")
    st.altair_chart(charts[11], width="stretch")
    st.altair_chart(charts[12], width="stretch")

st.altair_chart(charts[13], width="stretch")
st.altair_chart(charts[14], width="stretch")

st.markdown("## Carbon")
charts = plots_altair.carbon(variables_values, scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[6], width="stretch")
    st.altair_chart(charts[8], width="stretch")
    st.altair_chart(charts[10], width="stretch")
    st.altair_chart(charts[12], width="stretch")
with col2:
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[3], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[7], width="stretch")
    st.altair_chart(charts[9], width="stretch")
    st.altair_chart(charts[11], width="stretch")
    st.altair_chart(charts[13], width="stretch")

st.markdown("## Nitrogen Balance")
charts = plots_altair.nutrient_balance(variables_values, substance="N", scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[6], width="stretch")
    st.altair_chart(charts[8], width="stretch")
    st.altair_chart(charts[10], width="stretch")
with col2:
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[3], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[7], width="stretch")
    st.altair_chart(charts[9], width="stretch")
    st.altair_chart(charts[11], width="stretch")

st.markdown("## Phosphorus Balance")
charts = plots_altair.nutrient_balance(variables_values, substance="P", scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[6], width="stretch")
    st.altair_chart(charts[8], width="stretch")
    st.altair_chart(charts[10], width="stretch")
with col2:
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[3], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[7], width="stretch")
    st.altair_chart(charts[9], width="stretch")
    st.altair_chart(charts[11], width="stretch")

st.markdown("## Potassium Balance")
charts = plots_altair.nutrient_balance(variables_values, substance="K", scen=scen)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(charts[0], width="stretch")
    st.altair_chart(charts[2], width="stretch")
    st.altair_chart(charts[4], width="stretch")
    st.altair_chart(charts[6], width="stretch")
    st.altair_chart(charts[8], width="stretch")
    st.altair_chart(charts[10], width="stretch")
with col2:
    st.altair_chart(charts[1], width="stretch")
    st.altair_chart(charts[3], width="stretch")
    st.altair_chart(charts[5], width="stretch")
    st.altair_chart(charts[7], width="stretch")
    st.altair_chart(charts[9], width="stretch")
    st.altair_chart(charts[11], width="stretch")
