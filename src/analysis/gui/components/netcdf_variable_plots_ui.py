import streamlit as st
import matplotlib.pyplot as plt

from susi.io.netcdf_utils import NetcdfVariableValue

from analysis.gui.components import plots


def build(variables_values: list[NetcdfVariableValue]) -> None:
    for var_value in variables_values:
        st.markdown(f"**{var_value.path}**")

        if len(var_value.value.shape) != 3:
            st.info(
                f"Variable has shape {var_value.value.shape}. Only 3D variables (scenario, time, space) are currently plotted."
            )
        else:
            # 3D variable: (scenario, time, space)
            data = var_value.value[0, :, :]  # Take first scenario
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
