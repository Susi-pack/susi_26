"""
Notebook equivalent of `analysis.streamlit.components.netcdf_variable_plots_ui`.

Reuses `shared_reporting_utils/quick_look_plots.py`'s `quick_look_sections`
for the 3D-shape guard and per-variable (title, fig) generation (per #234) --
`plots.spatial_bars`/`plots.temporal_stats` are reached only through it, not
imported directly. Replaces the Streamlit display wrapper (`st.pyplot` +
`plt.close`, laid out in two `st.columns`) with plain inline matplotlib
display, one plot after another (see #215: functional, not aesthetic,
equivalence).

Typical usage:

    quick_look_plots.display_quick_look_plots(selected_variable_values)
"""

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from analysis.shared_reporting_utils.quick_look_plots import quick_look_sections
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
    for var_path, message, figures in quick_look_sections(variables_values):
        display(Markdown(f"**{var_path}**"))

        if message is not None:
            display(Markdown(f"*{message}*"))
            continue

        for title, fig in figures:
            display(Markdown(f"*{title}*"))
            display(fig)
            plt.close(fig)
