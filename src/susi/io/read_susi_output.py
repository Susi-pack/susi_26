from susi.io.output_data_structure import (
    StandID,
    NetcdfVariablePath,
    NetcdfVariableInfo,
)
import susi.io.netcdf_utils_new as nc_utils_new
from susi.io.app_settings import AppSettings

# %%

PROJECT_ID = "paroninkorpi"
stand_folderpaths = nc_utils_new.list_subdirectories(
    AppSettings().output_folder / PROJECT_ID
)
# stand_ids: list[StandID] = [StandID(path.name) for path in stand_folderpaths]

metadata_by_stand = nc_utils_new.load_all_metadatas_from_stands(
    folders=stand_folderpaths
)


# All Susi netcdf files have the same structure.
# Pick one and get the group/variables hierarchical structure.
# This will be useful in choosing what variables to read later.
sample_netcdf_filepath = metadata_by_stand[StandID(stand_folderpaths[0].name)].iloc[0][
    "netcdf_output_filepath"
]

all_variables: dict[NetcdfVariablePath, NetcdfVariableInfo] = (
    nc_utils_new.list_all_netcdf_variables(sample_netcdf_filepath)
)

# Choose the variables of interest
VARS_OF_INTEREST = (
    nc_utils_new.NetcdfVariablePath("/stand/volume"),
    nc_utils_new.NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
)

# %% CONTINUE HERE
# list (dimension n_sites) of dicts (scenarios for each site).
# Dicts hold arrays with the value of variables
chosen_variables_by_stand_and_scenario = (
    nc_utils_new.read_chosen_variables_from_netcdf_by_stands_and_scenarios(
        chosen_vars=chosen_vars, metadata_by_stand=metadata_by_stand
    )
)
