"""
Notebook equivalent of `analysis.streamlit.components.netcdf_variable_plots_ui`.

Reuses the quick-look plotting logic (`plots.spatial_bars`,
`plots.temporal_stats`) unchanged -- it's already Streamlit-independent,
pure matplotlib. Replaces the Streamlit display wrapper (`st.pyplot` +
`plt.close`, laid out in two `st.columns`) with plain inline matplotlib
display, one plot after another (see #215: functional, not aesthetic,
equivalence).

Typical usage:

    quick_look_plots.display_quick_look_plots(selected_variable_values)
"""

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from analysis.shared_reporting_utils import plots
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath


def display_quick_look_plots(
    variables_values: dict[NetcdfVariablePath, NetcdfVariableArray],
) -> None:
    """
    Display a waterfall bar chart and a spatial mean/std plot per variable.

    Only 3-D (scenario, time, space) variables can be plotted this way,
    matching the Streamlit version's restriction; other-shaped variables get
    a one-line notice instead of a plot.
    """
    for var_path, var_value in variables_values.items():
        display(Markdown(f"**{var_path}**"))

        if len(var_value.raw_shape) != 3:
            display(
                Markdown(
                    f"*Variable has shape {var_value.raw_shape}. Only 3D "
                    "variables (scenario, time, space) are currently plotted.*"
                )
            )
            continue

        data = var_value.processed

        display(Markdown("*Time-Space Waterfall (Grouped Bars)*"))
        fig = plots.spatial_bars(data=data)
        display(fig)
        plt.close(fig)

        display(Markdown("*Spatial Statistics (Mean ± Std)*"))
        fig = plots.temporal_stats(data=data)
        display(fig)
        plt.close(fig)
