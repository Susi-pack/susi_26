import streamlit as st
import numpy as np

from analysis.gui.components import (
    folder_selection,
    netcdf_variable_explorer,
)
from analysis.optimization.pareto_corner_plot import pareto_corner_plot
from susi.io.app_settings import AppSettings
import susi.io.load_output_data as load_output

import analysis.optimization.core as opti_core
from susi.io.utils import read_json_file

st.header("Optimization")

# %% Choose project

dir_path = folder_selection.build_folder_selection_widget(
    dir_path=st.session_state.settings["data_folder"], label="project"
)

# %% Specify stand areas
if "paroninkorpi" in str(dir_path):
    JSON_FROM_XML_PATH = (
        AppSettings().project_root_path
        / "xmltoallometry_with_areas/extra_XML_info.json"
    )
    j = read_json_file(path=JSON_FROM_XML_PATH)

    def _get_stand_area_from_json_file(json: dict, stand_number: int) -> float:
        return json["stand_datas"][str(stand_number)]["area"]

    def _stand_id_to_stand_number(stand_id: load_output.StandID) -> int:
        return int(str(stand_id).split("_")[-1])

    stand_ids = [load_output.StandID(f"stand_{i}") for i in range(1, 22)]

    stand_areas_ha: dict[load_output.StandID, float] = {
        stand_id: _get_stand_area_from_json_file(
            json=j, stand_number=_stand_id_to_stand_number(stand_id)
        )
        for stand_id in stand_ids
    }

else:
    raise ValueError(
        "Missing method to get stand areas for any area except Paroninkorpi"
    )

st.subheader("Stand areas")
with st.expander("View stand areas", expanded=False):
    st.write(stand_areas_ha)

# %% Choose Netcdf variabales
# The golden test netcdf is used to read the variable structure  of the netcdf file
sample_netcdf_filepath = (
    AppSettings().project_root_path / "tests/golden_file_test/golden_susi.nc"
)
all_variables = load_output.list_all_netcdf_variables(sample_netcdf_filepath)

_DEFAULT_VARIABLES = [
    "/stand/volume",
    "/balance/C/soil_c_balance_co2eq",
    "/balance/N/to_water",
]
# %% Optimization configuration form
with st.form(key="optimization_config"):
    chosen_netcdf_variables = netcdf_variable_explorer.build(
        netcdf_variables=all_variables, preselected=_DEFAULT_VARIABLES
    )

    st.subheader("Choose aggregation method per variable")

    _AGG_METHODS: dict[str, load_output.NetcdfAggregationFn] = {
        "Mean of all values": load_output.NetcdfVariableArray.mean_of_all_values,
        "Spatial mean at last timestep": load_output.NetcdfVariableArray.spatial_mean_at_last_timestep,
        "Mean over space, sum over time": load_output.NetcdfVariableArray.mean_over_space_sum_over_time,
        "Spatial mean at initial timestep": load_output.NetcdfVariableArray.spatial_mean_at_initial_timestep,
    }

    chosen_var_properties: dict[
        load_output.NetcdfVariablePath, opti_core.TargetVariableProperties
    ] = {}
    for var_path in chosen_netcdf_variables:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{var_path}**")
        with col2:
            chosen_aggregation_label = st.selectbox(
                label="Aggregation method",
                options=list(_AGG_METHODS.keys()),
                key=f"agg_{var_path}",
                label_visibility="collapsed",
            )
        with col3:
            invert_optimization = st.checkbox(
                label="Invert sign?",
                value=False,
                help="If selected, adds a negative sign to the data for the optimization algorithm, which always tries to minimize. This should be selected if you want to a) minimize a variable with negative values, or b) maximize a variable with positive values.",
                key=f"invert_{var_path}",
            )
        chosen_var_properties[var_path] = opti_core.TargetVariableProperties(
            aggregation_function=_AGG_METHODS[chosen_aggregation_label],
            invert_optimization=invert_optimization,
        )

    epsilon = st.number_input(
        label="Epsilon (Pareto front precision)",
        min_value=1e-9,
        max_value=1e4,
        value=1e-7,
        format="%.1e",
        help="Controls the granularity of the Pareto front. Smaller values give more precise results but require more computation.",
    )

    submitted = st.form_submit_button("Run Optimization", type="primary")

if submitted:
    with st.spinner("Running optimization..."):
        results = opti_core.run_optimization(
            variable_info=chosen_var_properties,
            project_dirpath=dir_path,
            stand_areas=stand_areas_ha,
            epsilon=epsilon,
            n_random_points=10000,
        )
        st.session_state["optimization_results"] = results
        st.session_state["optimization_var_paths"] = list(chosen_var_properties.keys())
    st.success("Optimization complete")

# %% Visualize solutions

if "optimization_results" in st.session_state:
    if st.button("Plot Results", type="secondary"):
        results: opti_core.OptimizationResults = st.session_state[
            "optimization_results"
        ]
        var_paths = st.session_state.get("optimization_var_paths", [])

        fig = pareto_corner_plot(
            data=results.pareto_front.target_vectors,
            random_points=results.random_points.target_vectors,
            labels=var_paths,
            show_diagonal=False,
            label_fontsize=8,
        )

        st.pyplot(fig, width="content")
