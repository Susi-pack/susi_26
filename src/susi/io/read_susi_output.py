from susi.io.load_output_data import (
    NetcdfVariablePath,
    StandID,
    ScenarioID,
)
import susi.io.load_output_data as load_output
from susi.io.app_settings import AppSettings

# %%

PROJECT_ID = "paroninkorpi"
stand_folderpaths = load_output.list_subdirectories(
    AppSettings().output_folder / PROJECT_ID
)
# stand_ids: list[StandID] = [StandID(path.name) for path in stand_folderpaths]

metadata_by_stand = load_output.load_all_metadatas_from_stands(
    folders=stand_folderpaths
)


# All Susi netcdf files have the same structure.
# Pick one to get the group/variables hierarchical structure.
# This will be useful in choosing what variables to read later.
sample_netcdf_filepath = metadata_by_stand[
    load_output.StandID(stand_folderpaths[0].name)
].iloc[0]["netcdf_output_filepath"]

all_variables: dict[load_output.NetcdfVariablePath, load_output.NetcdfVariableInfo] = (
    load_output.list_all_netcdf_variables(sample_netcdf_filepath)
)

VARS_OF_INTEREST = (
    NetcdfVariablePath("/stand/volume"),
    NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
)

data_store: load_output.OutputDataStore = (
    load_output.read_netcdf_files_for_selected_variables(
        selected_variables=VARS_OF_INTEREST, metadata_by_stand=metadata_by_stand
    )
)

# %% Trials

v = data_store.get_variable_value_for_scenario_and_stand(
    variable_path=NetcdfVariablePath("/time"),
    stand_id=StandID("stand_01"),
    scenario_id=ScenarioID("default"),
)

v.last_timestep()

for _, info in all_variables.items():
    print(info.shape)
