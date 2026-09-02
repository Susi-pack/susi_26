import streamlit as st
import matplotlib.pyplot as plt

from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath

from analysis.shared_reporting_utils.quick_look_plots import quick_look_sections


def build(variables_values: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    for var_path, message, figures in quick_look_sections(variables_values):
        st.markdown(f"**{var_path}**")

        if message is not None:
            st.info(message)
            continue

        columns = st.columns(len(figures))
        for column, (title, fig) in zip(columns, figures):
            with column:
                st.markdown(f"*{title}*")
                st.pyplot(fig)
                plt.close(fig)
