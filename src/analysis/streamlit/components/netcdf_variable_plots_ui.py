import streamlit as st
import matplotlib.pyplot as plt

from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath

from analysis.shared_reporting_utils import plots


def build(variables_values: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    for var_path, var_value in variables_values.items():
        st.markdown(f"**{var_path}**")

        if len(var_value.raw_shape) != 3:
            st.info(
                f"Variable has shape {var_value.raw_shape}. Only 3D variables (scenario, time, space) are currently plotted."
            )
        else:
            data = var_value.processed
            n_time, n_space = data.shape

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("*Time-Space Waterfall (Grouped Bars)*")
                fig = plots.spatial_bars(data=data)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.markdown("*Spatial Statistics (Mean ± Std)*")
                fig = plots.temporal_stats(data=data)
                st.pyplot(fig)
                plt.close(fig)
