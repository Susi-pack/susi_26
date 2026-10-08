import streamlit as st

import analysis.optimization.core as opti_core
import susi.io.load_output_data as load_output
import susi.io.utils as io_utils
from analysis.optimization.pareto_corner_plot import pareto_corner_plot
from analysis.optimization.stand_areas import stand_areas_for_run
from analysis.streamlit.components import (
    folder_selection,
    netcdf_variable_explorer,
)

st.header("Optimization")

# %% Choose project and run

# Both halves of the selection are used below: the run's folder holds one
# subfolder per stand, while the project is where those stands' areas come
# from.
selection = folder_selection.build_project_and_run_selection_widget(
    projects_root=st.session_state.settings["projects_root"]
)
run_dirpath = selection.run_dir

# %% Specify stand areas
# Read from the project's own inputs/stand_data.json. Shared with the notebook
# port so both frontends read the same areas the same way.
stand_areas_ha: dict[load_output.StandID, float] = stand_areas_for_run(run_dirpath)

st.subheader("Stand areas")
with st.expander("View stand areas", expanded=False):
    st.write(stand_areas_ha)

# %% Choose Netcdf variabales
# The golden test netcdf is used to read the variable structure  of the netcdf file
sample_netcdf_filepath = io_utils.repo_root() / "tests/golden_file_test/golden_susi.nc"
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

    st.subheader("Choose aggregation method and direction per variable")

    target_specs: dict[load_output.NetcdfVariablePath, opti_core.TargetSpec] = {}
    for var_path in chosen_netcdf_variables:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**{var_path}**")
        with col2:
            chosen_aggregation_label = st.selectbox(
                label="Aggregation method",
                options=list(opti_core.AGGREGATION_METHODS_BY_LABEL.keys()),
                key=f"agg_{var_path}",
                label_visibility="collapsed",
            )
        with col3:
            direction = st.selectbox(
                label="Direction",
                options=list(opti_core.Direction),
                format_func=lambda direction: direction.value.capitalize(),
                help=opti_core.DIRECTION_HELP,
                key=f"direction_{var_path}",
                label_visibility="collapsed",
            )
        target_specs[var_path] = opti_core.TargetSpec(
            aggregation=opti_core.AGGREGATION_METHODS_BY_LABEL[
                chosen_aggregation_label
            ],
            direction=direction,
        )

    epsilon = st.number_input(
        label="Epsilon (Pareto front precision)",
        min_value=opti_core.EPSILON_MIN,
        max_value=opti_core.EPSILON_MAX,
        value=1e-7,
        format="%.1e",
        help="Controls the granularity of the Pareto front. Smaller values give more precise results but require more computation.",
    )

    submitted = st.form_submit_button("Run Optimization", type="primary")

if submitted:
    with st.spinner("Running optimization..."):
        results = opti_core.run_optimization(
            target_specs=target_specs,
            run_dirpath=run_dirpath,
            stand_areas=stand_areas_ha,
            epsilon=epsilon,
            n_random_points=10000,
        )
        st.session_state["optimization_results"] = results
        st.session_state["optimization_var_paths"] = list(target_specs.keys())
    st.success("Optimization complete")

# %% Visualize solutions

if "optimization_results" in st.session_state and st.button(
    "Plot Results", type="secondary"
):
    results: opti_core.OptimizationResults = st.session_state["optimization_results"]
    var_paths = st.session_state.get("optimization_var_paths", [])

    fig = pareto_corner_plot(
        data=results.pareto_front.target_vectors,
        random_points=results.random_points.target_vectors,
        labels=var_paths,
        show_diagonal=False,
        label_fontsize=8,
    )

    st.pyplot(fig, width="content")
