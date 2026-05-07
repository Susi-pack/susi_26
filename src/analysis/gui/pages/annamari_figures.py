import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

import susi.io.load_output_data as load_output
from susi.io.app_settings import AppSettings

from analysis.gui.components import plots, folder_selection

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


metadata, susi_params = load_output.read_params_from_jsons(
    experiment_folderpath=chosen_scenario_folder
)

chosen_netcdf_filepath = Path(metadata["netcdf_output_filepath"])
all_variables = load_output.list_all_netcdf_variables(chosen_netcdf_filepath)

VARIABLE_PATHS = (
    load_output.NetcdfVariablePath("/strip/dwtyr"),
    load_output.NetcdfVariablePath("/strip/dwtyr_growingseason"),
    load_output.NetcdfVariablePath("/strip/dwtyr_latesummer"),
    load_output.NetcdfVariablePath("/strip/dwt"),
    load_output.NetcdfVariablePath("/strip/roff"),
    load_output.NetcdfVariablePath("/strip/roffwest"),
    load_output.NetcdfVariablePath("/strip/roffeast"),
    load_output.NetcdfVariablePath("/strip/surfacerunoff"),
    load_output.NetcdfVariablePath("/strip/deltas"),
    load_output.NetcdfVariablePath("/strip/elevation"),
    load_output.NetcdfVariablePath("/cpy/ET_yr"),
    load_output.NetcdfVariablePath("/cpy/transpi_yr"),
    load_output.NetcdfVariablePath("/cpy/efloor_yr"),
    load_output.NetcdfVariablePath("/cpy/SWEmax"),
    load_output.NetcdfVariablePath("/cpy/interc_yr"),
    load_output.NetcdfVariablePath("/stand/volume"),
    load_output.NetcdfVariablePath("/stand/dominant/volume"),
    load_output.NetcdfVariablePath("/stand/subdominant/volume"),
    load_output.NetcdfVariablePath("/stand/under/volume"),
    load_output.NetcdfVariablePath("/stand/logvolume"),
    load_output.NetcdfVariablePath("/stand/pulpvolume"),
    load_output.NetcdfVariablePath("/stand/leafmass"),
    load_output.NetcdfVariablePath("/stand/dominant/leafmass"),
    load_output.NetcdfVariablePath("/stand/dominant/leafmax"),
    load_output.NetcdfVariablePath("/stand/dominant/leafmin"),
    load_output.NetcdfVariablePath("/stand/subdominant/leafmass"),
    load_output.NetcdfVariablePath("/stand/subdominant/leafmax"),
    load_output.NetcdfVariablePath("/stand/subdominant/leafmin"),
    load_output.NetcdfVariablePath("/stand/under/leafmass"),
    load_output.NetcdfVariablePath("/stand/under/leafmax"),
    load_output.NetcdfVariablePath("/stand/under/leafmin"),
    load_output.NetcdfVariablePath("/stand/nut_stat"),
    load_output.NetcdfVariablePath("/stand/dominant/NPP"),
    load_output.NetcdfVariablePath("/stand/dominant/NPP_pot"),
    load_output.NetcdfVariablePath("/stand/n_demand"),
    load_output.NetcdfVariablePath("/stand/p_demand"),
    load_output.NetcdfVariablePath("/stand/k_demand"),
    load_output.NetcdfVariablePath("/stand/nonwoodylitter"),
    load_output.NetcdfVariablePath("/stand/woodylitter"),
    load_output.NetcdfVariablePath("/stand/finerootlitter"),
    load_output.NetcdfVariablePath("/stand/biomass"),
    load_output.NetcdfVariablePath("/stand/volumegrowth"),
    load_output.NetcdfVariablePath("/groundvegetation/ds_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/h_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/s_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/gv_tot"),
    load_output.NetcdfVariablePath("/esom/Mass/out"),
    load_output.NetcdfVariablePath("/esom/Mass/L0L"),
    load_output.NetcdfVariablePath("/esom/Mass/L0W"),
    load_output.NetcdfVariablePath("/esom/Mass/LL"),
    load_output.NetcdfVariablePath("/esom/Mass/LW"),
    load_output.NetcdfVariablePath("/esom/Mass/FL"),
    load_output.NetcdfVariablePath("/esom/Mass/FW"),
    load_output.NetcdfVariablePath("/esom/Mass/H"),
    load_output.NetcdfVariablePath("/esom/Mass/P1"),
    load_output.NetcdfVariablePath("/esom/Mass/P2"),
    load_output.NetcdfVariablePath("/esom/Mass/P3"),
    load_output.NetcdfVariablePath("/balance/C/LMWdoc_to_water"),
    load_output.NetcdfVariablePath("/balance/C/HMW_to_water"),
    load_output.NetcdfVariablePath("/balance/C/LMWdoc_to_atm"),
    load_output.NetcdfVariablePath("/balance/C/HMW_to_atm"),
    load_output.NetcdfVariablePath("/balance/C/co2c_release"),
    load_output.NetcdfVariablePath("/balance/C/ch4c_release"),
    load_output.NetcdfVariablePath("/balance/C/stand_litter_in"),
    load_output.NetcdfVariablePath("/balance/C/gv_litter_in"),
    load_output.NetcdfVariablePath("/balance/C/soil_c_balance_c"),
    load_output.NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
    load_output.NetcdfVariablePath("/balance/C/stand_c_balance_c"),
    load_output.NetcdfVariablePath("/balance/C/stand_c_balance_co2eq"),
    load_output.NetcdfVariablePath("/balance/N/to_water"),
    load_output.NetcdfVariablePath("/balance/N/decomposition_below_root_lyr"),
    load_output.NetcdfVariablePath("/balance/N/decomposition_tot"),
    load_output.NetcdfVariablePath("/balance/N/decomposition_root_lyr"),
    load_output.NetcdfVariablePath("/balance/N/deposition"),
    load_output.NetcdfVariablePath("/balance/N/fertilization_release"),
    load_output.NetcdfVariablePath("/balance/N/stand_demand"),
    load_output.NetcdfVariablePath("/balance/N/gv_demand"),
    load_output.NetcdfVariablePath("/balance/N/balance_root_lyr"),
    load_output.NetcdfVariablePath("/balance/P/to_water"),
    load_output.NetcdfVariablePath("/balance/P/decomposition_below_root_lyr"),
    load_output.NetcdfVariablePath("/balance/P/decomposition_tot"),
    load_output.NetcdfVariablePath("/balance/P/decomposition_root_lyr"),
    load_output.NetcdfVariablePath("/balance/P/deposition"),
    load_output.NetcdfVariablePath("/balance/P/fertilization_release"),
    load_output.NetcdfVariablePath("/balance/P/stand_demand"),
    load_output.NetcdfVariablePath("/balance/P/gv_demand"),
    load_output.NetcdfVariablePath("/balance/P/balance_root_lyr"),
    load_output.NetcdfVariablePath("/balance/K/to_water"),
    load_output.NetcdfVariablePath("/balance/K/decomposition_below_root_lyr"),
    load_output.NetcdfVariablePath("/balance/K/decomposition_tot"),
    load_output.NetcdfVariablePath("/balance/K/decomposition_root_lyr"),
    load_output.NetcdfVariablePath("/balance/K/deposition"),
    load_output.NetcdfVariablePath("/balance/K/fertilization_release"),
    load_output.NetcdfVariablePath("/balance/K/stand_demand"),
    load_output.NetcdfVariablePath("/balance/K/gv_demand"),
    load_output.NetcdfVariablePath("/balance/K/balance_root_lyr"),
)

data: dict[load_output.NetcdfVariablePath, load_output.NetcdfVariableArray] = (
    load_output.read_value_several_variables_from_single_file(
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
