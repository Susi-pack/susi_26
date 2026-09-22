import streamlit as st

from analysis.streamlit.components import folder_selection
from pathlib import Path
from susi.io.load_output_data import StandID
from analysis.core.parse_outputs import find_differing_params, find_unique_params
from analysis.shared_reporting_utils.param_comparison import shape_differing_params


# %% Choose stand folder
st.header("Choose stand folder")

col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.markdown("**Projects root**")

with col2:
    st.write(st.session_state.settings["projects_root"])

with col3:
    if st.button("Browse…", use_container_width=True):
        result = folder_selection.pick_folder_popup()
        if result:
            st.session_state.settings["projects_root"] = result
            st.rerun()

dir_path = folder_selection.build_folder_selection_widget(
    dir_path=folder_selection.build_project_and_run_selection_widget(
        projects_root=st.session_state.settings["projects_root"]
    ).run_dir,
    label="stand",
)

# Extract stand_id and output_dir from selection
stand_id = None
output_dir = None
if dir_path:
    stand_path = Path(dir_path)
    if stand_path.exists() and stand_path.is_dir():
        stand_id = StandID(stand_path.name)
        output_dir = stand_path.parent

# %% Show variables per scenario
if stand_id and output_dir:
    st.header("Parameter Comparison Across Scenarios")

    # Get differing and unique parameters
    differing = find_differing_params(stand_id, output_dir)
    unique = find_unique_params(stand_id, output_dir)
    shaped_differing = shape_differing_params(differing)

    # Display differing parameters (table-like, no expanders)
    st.subheader("Differing Parameters")
    if shaped_differing:
        for param_name, rows in shaped_differing.items():
            # Parameter name as section title
            st.markdown(f"**{param_name}**")
            # Two-column grid: left = value, right = scenarios
            for row in rows:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.code(f"{row['value']}")
                with col2:
                    st.write(row["scenarios"])
            st.divider()  # Separator between parameters
    else:
        st.info("No differing parameters found across scenarios.")

    # Display unique parameters (inside single expander)
    with st.expander("Unique Parameters (Same Across All Scenarios)"):
        if unique:
            for param_name, value in unique.items():
                st.markdown(f"**{param_name}**: `{value}`")
        else:
            st.write("No unique parameters found (all parameters differ).")
else:
    if dir_path:
        st.warning("Invalid stand folder selected.")
    else:
        st.info("Please select a stand folder to compare scenarios.")
