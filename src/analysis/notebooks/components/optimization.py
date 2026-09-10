"""
Notebook equivalent of `analysis.streamlit.pages.optimization`.

Covers the three parts of the Streamlit page that aren't already ported
elsewhere: showing a project's stand areas, configuring how each chosen
netcdf variable becomes an optimization target, and plotting the resulting
Pareto front. Project selection reuses `folder_selection`, and variable
selection reuses `variable_selection`.

Per #215's picker/renderer split, `build_target_config` is a picker (it
displays its widgets and returns them, to be read in a later cell by
`target_variable_properties`), while `display_stand_areas` and
`display_pareto_corner_plot` are renderers (they display their output and
also return the underlying data).

The Streamlit page wraps the configuration in an `st.form` with a "Run
Optimization" submit button. A notebook needs neither: running the next cell
is the submit, and there is no re-run-on-every-widget-change to guard
against.

Stand areas are still Paroninkorpi-only, exactly as in the Streamlit page --
they come from the shared `analysis.optimization.stand_areas`, whose
generalization is issue #216.

Typical usage, one call per cell:

    stand_areas_ha = optimization.display_stand_areas(project_dropdown.value)

    # -- next cell, after ticking variables in a variable_selection selector --
    config = optimization.build_target_config(chosen_netcdf_variables)

    # -- next cell, after choosing aggregations --
    variable_info = optimization.target_variable_properties(config)
"""

from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import analysis.optimization.core as opti_core
from analysis.optimization.pareto_corner_plot import pareto_corner_plot
from analysis.optimization.stand_areas import stand_areas_for_project
from susi.io.load_output_data import NetcdfVariablePath, StandID

_INVERT_SIGN_TOOLTIP = (
    "If selected, adds a negative sign to the data for the optimization "
    "algorithm, which always tries to minimize. This should be selected if "
    "you want to a) minimize a variable with negative values, or b) maximize "
    "a variable with positive values."
)


def display_stand_areas(project_dirpath: Path) -> dict[StandID, float]:
    """
    Display the area in hectares of every stand in a project, and return them.

    project_dirpath is a project output folder -- e.g. the `.value` of a
    project-level `folder_selection.build_dropdown()`. The returned dict is
    what `core.prepare_optimization_data()` expects as `stand_areas`.

    Raises for any project other than Paroninkorpi (#216); see
    `analysis.optimization.stand_areas`.
    """
    stand_areas_ha = stand_areas_for_project(project_dirpath=project_dirpath)

    areas_dataframe = pd.DataFrame(
        {
            "stand": list(stand_areas_ha.keys()),
            "area_ha": list(stand_areas_ha.values()),
        }
    )
    # Replaces the Streamlit page's collapsed "View stand areas" expander:
    # a notebook has no equivalent fold, so the table is simply shown, with
    # every row visible (pandas would otherwise elide the middle of a long
    # stand list).
    with pd.option_context("display.max_rows", None):
        display(areas_dataframe)
    print(f"Total area: {sum(stand_areas_ha.values()):.1f} ha")

    return stand_areas_ha


@dataclass(frozen=True)
class TargetConfig:
    """
    The widgets configuring how chosen variables become optimization targets.

    One aggregation Dropdown and one invert Checkbox per variable, plus the
    single epsilon input for the whole run. Returned by build_target_config()
    so a later cell can read the user's choices via
    target_variable_properties() and `config.epsilon.value`.

    container is exposed mainly for completeness/debugging -- reading the
    configuration needs only the dropdowns, checkboxes and epsilon.
    """

    aggregation_dropdowns: dict[NetcdfVariablePath, widgets.Dropdown]
    invert_checkboxes: dict[NetcdfVariablePath, widgets.Checkbox]
    epsilon: widgets.BoundedFloatText
    container: widgets.VBox


def build_target_config(
    variable_paths: list[NetcdfVariablePath],
    default_aggregation_label: str | None = None,
) -> TargetConfig:
    """
    Build, display, and return one configuration row per target variable.

    Each row is the notebook counterpart of the Streamlit page's three
    columns: the variable's path, the aggregation method that collapses its
    array to a single number, and whether to flip its sign (the optimizer
    only ever minimizes). Aggregation options come from
    `core.AGGREGATION_METHODS_BY_LABEL`, shared with the Streamlit page.

    default_aggregation_label preselects an aggregation for every row;
    it defaults to the first entry of that shared mapping, matching the
    Streamlit selectbox's own default.
    """
    aggregation_labels = list(opti_core.AGGREGATION_METHODS_BY_LABEL.keys())
    if default_aggregation_label is None:
        default_aggregation_label = aggregation_labels[0]
    if default_aggregation_label not in aggregation_labels:
        raise ValueError(
            f"Unknown aggregation method '{default_aggregation_label}'. "
            f"Known methods: {aggregation_labels}"
        )

    aggregation_dropdowns: dict[NetcdfVariablePath, widgets.Dropdown] = {}
    invert_checkboxes: dict[NetcdfVariablePath, widgets.Checkbox] = {}
    rows: list[widgets.HBox] = []

    for var_path in variable_paths:
        aggregation_dropdowns[var_path] = widgets.Dropdown(
            options=aggregation_labels,
            value=default_aggregation_label,
            layout=widgets.Layout(width="320px"),
        )
        invert_checkboxes[var_path] = widgets.Checkbox(
            value=False,
            description="Invert sign?",
            tooltip=_INVERT_SIGN_TOOLTIP,
            indent=False,
            layout=widgets.Layout(width="150px"),
        )
        rows.append(
            widgets.HBox(
                [
                    widgets.HTML(
                        value=f"<b>{var_path}</b>",
                        layout=widgets.Layout(width="320px"),
                    ),
                    aggregation_dropdowns[var_path],
                    invert_checkboxes[var_path],
                ]
            )
        )

    epsilon = widgets.BoundedFloatText(
        value=1e-7,
        min=1e-9,
        max=1e4,
        description="Epsilon:",
        style={"description_width": "initial"},
    )
    epsilon_row = widgets.HBox(
        [
            epsilon,
            widgets.HTML(
                value=(
                    "<i>Pareto front precision. Smaller values give more "
                    "precise results but require more computation.</i>"
                )
            ),
        ]
    )

    container = widgets.VBox([*rows, epsilon_row])
    display(container)

    return TargetConfig(
        aggregation_dropdowns=aggregation_dropdowns,
        invert_checkboxes=invert_checkboxes,
        epsilon=epsilon,
        container=container,
    )


def target_variable_properties(
    config: TargetConfig,
) -> dict[NetcdfVariablePath, opti_core.TargetVariableProperties]:
    """
    Read a TargetConfig's widgets into the optimization's input API.

    The result is what `core.prepare_optimization_data()` takes as
    `variable_info`. Its insertion order sets the column order of the
    optimization's target vectors, so it also fixes the axis order of
    display_pareto_corner_plot()'s labels.
    """
    return {
        var_path: opti_core.TargetVariableProperties(
            aggregation_function=opti_core.AGGREGATION_METHODS_BY_LABEL[
                dropdown.value
            ],
            invert_optimization=config.invert_checkboxes[var_path].value,
        )
        for var_path, dropdown in config.aggregation_dropdowns.items()
    }


def display_pareto_corner_plot(
    results: opti_core.OptimizationResults,
    labels: list[NetcdfVariablePath],
) -> plt.Figure:
    """
    Display the Pareto front as a corner plot, against the random points.

    labels names one target variable per axis, in the same order as the
    variable_info passed to the optimization -- e.g.
    `list(variable_info.keys())`.

    Returns the figure so a later cell can adjust or save it.
    """
    figure = pareto_corner_plot(
        data=results.pareto_front.target_vectors,
        random_points=results.random_points.target_vectors,
        labels=list(labels),
        show_diagonal=False,
        label_fontsize=8,
    )
    display(figure)
    plt.close(figure)
    return figure
