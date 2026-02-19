import streamlit as st
from streamlit_tree_select import tree_select
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import susi.io.netcdf_utils as nc_utils
from susi.io.app_settings import AppSettings

# %% Choose folder

susi_folders = nc_utils.list_subdirectories(
    path=st.session_state.settings["data_folder"]
)

chosen_susi_folder = st.selectbox(label="Choose SUSI folder", options=susi_folders)

# %% Metadata expander
metadata, susi_params = nc_utils.read_json_metadatas(
    experiment_folderpath=chosen_susi_folder
)

with st.expander("see metadata", expanded=False):
    st.write("metadata")
    st.json(metadata)
    st.write("Susi params")
    st.json(susi_params)

# %% Read Netcdf variabales
# The golden test netcdf is used to read the variable structure  of the netcdf file
sample_netcdf_filepath = (
    AppSettings().project_root_path / "tests/golden_file_test/golden_susi.nc"
)

all_variables = nc_utils.list_all_netcdf_variables(sample_netcdf_filepath)

# Actual data file path for reading values
chosen_netcdf_filepath = Path(metadata["netcdf_output_filepath"])


def build_tree_nodes(variables):
    """Build tree-select nodes from variable paths.

    All node values must be unique, so we use full paths for both
    group nodes and leaf nodes.
    """
    # Build nested dictionary structure with full paths
    tree_dict = {}

    for var in variables:
        parts = [p for p in var.path.split("/") if p]
        current = tree_dict
        current_path = ""

        # Navigate/create path
        for i, part in enumerate(parts[:-1]):
            current_path = f"{current_path}/{part}" if current_path else f"/{part}"
            if part not in current:
                current[part] = {"children": {}, "full_path": current_path}
            current = current[part]["children"]

        # Add variable as leaf node
        var_name = parts[-1] if parts else var.name
        current[var_name] = {"variable": var}

    # Convert to tree-select format
    def dict_to_nodes(d, parent_path=""):
        nodes = []
        for key, value in d.items():
            if "variable" in value:
                # Leaf node (actual variable)
                var = value["variable"]
                nodes.append(
                    {
                        "label": f"{var.name}",
                        "value": var.path,
                        "title": f"Path: {var.path}\nShape: {var.shape}\nInfo: {var.units or 'N/A'}",
                    }
                )
            else:
                # Group node - use prefixed path to avoid conflicts with variables
                full_path = value.get("full_path", key)
                children = dict_to_nodes(value.get("children", {}), full_path)
                nodes.append(
                    {
                        "label": key,
                        "value": f"__group__{full_path}",  # Prefix to ensure uniqueness
                        "children": children,
                    }
                )
        return nodes

    return dict_to_nodes(tree_dict)


# Build tree and render
tree_nodes = build_tree_nodes(all_variables)

st.subheader("NetCDF Variables")
st.markdown("Select variables from the tree:")

# Use st.container with height parameter for scrollable area
tree_container = st.container(height=400, border=True)
with tree_container:
    result = tree_select(
        tree_nodes,
        check_model="leaf",
        show_expand_all=True,
    )

# Map selected paths back to NetcdfVariableInfo objects
selected_paths = result.get("checked", [])
chosen_netcdf_variables = [var for var in all_variables if var.path in selected_paths]

# Display selection summary and plots
if chosen_netcdf_variables:
    st.markdown("---")
    st.subheader(f"Selected Variables ({len(chosen_netcdf_variables)})")
    for var in chosen_netcdf_variables:
        st.write(f"- `{var.path}` ({var.shape}, {var.units})")

    # Read actual values from the NetCDF file
    st.markdown("---")
    st.subheader("Variable Plots")

    if chosen_netcdf_filepath.exists():
        variables_values = nc_utils.read_value_several_variables_from_single_file(
            netcdf_filepath=chosen_netcdf_filepath,
            variables=chosen_netcdf_variables,
        )

        for var_value in variables_values:
            st.markdown(f"**{var_value.path}**")

            if len(var_value.value.shape) == 3:
                # 3D variable: (scenario, time, space)
                data = var_value.value[0, :, :]  # Take first scenario
                n_time, n_space = data.shape

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("*Time-Space Waterfall (Grouped Bars)*")
                    fig1, ax1 = plt.subplots(figsize=(8, 5))

                    # Grouped bars - each time step has bars for each space point
                    x = np.arange(n_time)
                    width = 0.8 / n_space

                    for space_idx in range(n_space):
                        offset = (space_idx - n_space / 2 + 0.5) * width
                        ax1.bar(
                            x + offset,
                            data[:, space_idx],
                            width,
                            label=f"Space {space_idx}" if n_space <= 10 else None,
                        )

                    ax1.set_xlabel("Time Step")
                    ax1.set_ylabel("Value")
                    ax1.set_title(f"Temporal Evolution")
                    if n_space <= 10:
                        ax1.legend(
                            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small"
                        )
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.markdown("*Spatial Statistics (Mean ± Std)*")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))

                    # Aggregate over space
                    mean_over_space = np.mean(data, axis=1)
                    std_over_space = np.std(data, axis=1)

                    x_time = np.arange(n_time)
                    ax2.plot(x_time, mean_over_space, "b-", label="Mean", linewidth=2)
                    ax2.fill_between(
                        x_time,
                        mean_over_space - std_over_space,
                        mean_over_space + std_over_space,
                        alpha=0.3,
                        color="blue",
                        label="±1 Std",
                    )

                    ax2.set_xlabel("Time Step")
                    ax2.set_ylabel("Value")
                    ax2.set_title("Spatial Mean and Variability")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    plt.close(fig2)
            else:
                st.info(
                    f"Variable has shape {var_value.value.shape}. Only 3D variables (scenario, time, space) are currently plotted."
                )
    else:
        st.error(f"NetCDF file not found: {chosen_netcdf_filepath}")
