"""
Shared data-loading logic for the "Single Scenario Dashboard" -- the fixed,
~100-variable set of 7 figures (stand, hydrology, mass, carbon, nutrient
balance x3) built from one scenario's netCDF output.

Reused by both the Streamlit page
(`analysis.streamlit.pages.single_scenario_dashboard`) and the notebook
component (`analysis.notebooks.components.single_scenario_dashboard`).

Extracted per #231 to eliminate the duplicated fixed-variable-path tuple
that used to be hand-copied independently in each frontend.
"""

from pathlib import Path
from typing import Sequence

import susi.io.load_output_data as load_output
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath

# The fixed variable-path list behind the Single Scenario Dashboard's 7
# figures (stand, hydrology, mass, carbon, nutrient balance x3). Order
# matters only in that it must stay stable/complete -- consumers look
# variables up by path, not by position.
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


def load_report_data(
    netcdf_filepath: Path, variable_paths: Sequence[NetcdfVariablePath]
) -> dict[NetcdfVariablePath, NetcdfVariableArray]:
    """Read the given variables from a single netcdf file."""
    return load_output.read_value_several_variables_from_single_file(
        netcdf_filepath=netcdf_filepath, variable_paths=variable_paths
    )
