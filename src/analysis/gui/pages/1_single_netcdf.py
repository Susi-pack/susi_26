import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import susi.io.netcdf_utils as nc_utils
from susi.io.app_settings import AppSettings

from analysis.gui.components import metadata_expander, netcdf_variable_explorer

# %% Choose folder

susi_folders = nc_utils.list_subdirectories(
    path=st.session_state.settings["data_folder"]
)

chosen_susi_folder = st.selectbox(label="Choose SUSI folder", options=susi_folders)

# %% Metadata expander
metadata, susi_params = nc_utils.read_json_metadatas(
    experiment_folderpath=chosen_susi_folder
)

metadata_expander.build(metadata=metadata, susi_params=susi_params)

# %% Read Netcdf variabales
# The golden test netcdf is used to read the variable structure  of the netcdf file
sample_netcdf_filepath = (
    AppSettings().project_root_path / "tests/golden_file_test/golden_susi.nc"
)
all_variables = nc_utils.list_all_netcdf_variables(sample_netcdf_filepath)

# Actual netcdf file path for reading values (not only structure of the file)
chosen_netcdf_filepath = Path(metadata["netcdf_output_filepath"])


st.subheader("NetCDF Variables")
st.markdown("Select variables from the tree:")

chosen_netcdf_variables = netcdf_variable_explorer.build(netcdf_variables=all_variables)


# %% Display selection summary and plots
if chosen_netcdf_variables:
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
                    ax1.set_title("Temporal Evolution")
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
