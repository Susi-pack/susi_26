"""
Shared per-variable logic behind the "quick look" plots: a waterfall bar
chart and a spatial mean/std plot for each 3D (scenario, time, space)
netCDF variable, plus the "not plottable" guard/message for any other shape.

Reused by both the Streamlit component
(`analysis.streamlit.components.netcdf_variable_plots_ui`) and the notebook
component (`analysis.notebooks.components.quick_look_plots`).

Extracted per #234 to eliminate the duplicated 3D-shape guard and
title/figure generation that used to be hand-repeated independently in each
frontend, keeping the guard condition and message strings defined once.
"""

from collections.abc import Iterator

from matplotlib.figure import Figure

from analysis.shared_reporting_utils import plots
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath

# The two (title, fig) entries shown for a plottable (3D) variable, in
# display order.
_FIGURE_TITLES = (
    "Time-Space Waterfall (Grouped Bars)",
    "Spatial Statistics (Mean ± Std)",
)


def unsupported_shape_message(raw_shape: tuple[int, ...]) -> str:
    """The notice shown in place of plots for a non-3D variable."""
    return (
        f"Variable has shape {raw_shape}. Only 3D variables "
        "(scenario, time, space) are currently plotted."
    )


def quick_look_sections(
    variables_values: dict[NetcdfVariablePath, NetcdfVariableArray],
) -> Iterator[tuple[NetcdfVariablePath, str | None, list[tuple[str, Figure]]]]:
    """
    Yield, per variable in `variables_values` (in order):
    `(var_path, unsupported_shape_message, figures)`.

    For a 3D (scenario, time, space) variable, `unsupported_shape_message`
    is None and `figures` holds its (title, fig) pairs -- waterfall bars
    then spatial mean/std. For any other shape, `unsupported_shape_message`
    is a one-line notice and `figures` is empty.
    """
    for var_path, var_value in variables_values.items():
        if len(var_value.raw_shape) != 3:
            yield var_path, unsupported_shape_message(var_value.raw_shape), []
            continue

        data = var_value.processed
        figures = [
            (_FIGURE_TITLES[0], plots.spatial_bars(data=data)),
            (_FIGURE_TITLES[1], plots.temporal_stats(data=data)),
        ]
        yield var_path, None, figures
