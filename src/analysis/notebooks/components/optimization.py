"""
Notebook equivalent of `analysis.streamlit.pages.optimization`.

Typical usage, one call per cell:

    optimization.display_stand_areas(stand_areas_ha)

    # -- next cell, after ticking variables in a variable_selection selector --
    widgets = optimization.build_target_spec_widgets(chosen_netcdf_variables)

    # -- next cell, after choosing aggregations and directions --
    target_specs = optimization.target_specs_from_widgets(widgets)
    print(optimization.format_target_specs_as_code(target_specs))

The widgets are only one way to produce `target_specs`: the printed code can
replace that last cell, so a notebook can fix its configuration in code.
"""

from dataclasses import dataclass

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import analysis.optimization.core as opti_core
from analysis.optimization.pareto_corner_plot import pareto_corner_plot
from susi.io.load_output_data import NetcdfVariablePath, StandID


def display_stand_areas(stand_areas: dict[StandID, float]) -> None:
    """
    Display the area in hectares of every stand in a run, and return them.

    project_dir is the project's own folder -- the `.value` of the project
    dropdown -- and run_id names one of its runs, i.e. the `.name` of the
    `folder_selection.build_run_dropdown()` selection. The returned dict is
    what `core.prepare_optimization_data()` expects as `stand_areas`.

    Raises if the project's `stand_data.json` carries no area for one of the
    run's stands; see `analysis.optimization.stand_areas`.
    """
    areas_dataframe = pd.DataFrame(
        {
            "stand": list(stand_areas.keys()),
            "area_ha": list(stand_areas.values()),
        }
    )
    # Replaces the Streamlit page's collapsed "View stand areas" expander:
    # a notebook has no equivalent fold, so the table is simply shown, with
    # every row visible (pandas would otherwise elide the middle of a long
    # stand list).
    with pd.option_context("display.max_rows", None):
        display(areas_dataframe)
    print(f"Total area: {sum(stand_areas.values()):.1f} ha")


@dataclass(frozen=True)
class TargetSpecWidgets:
    """
    The widgets choosing how each chosen variable becomes a target spec.

    One aggregation Dropdown and one direction Dropdown per variable.
    Returned by build_target_spec_widgets() so that a later cell, run after
    the user has made their choices, can read them with
    target_specs_from_widgets().
    """

    aggregation_dropdowns: dict[NetcdfVariablePath, widgets.Dropdown]
    direction_dropdowns: dict[NetcdfVariablePath, widgets.Dropdown]


def build_target_spec_widgets(
    variable_paths: list[NetcdfVariablePath],
    default_aggregation_label: str | None = None,
) -> TargetSpecWidgets:
    """
    Build, display, and return one configuration row per target variable.

    Each row is the notebook counterpart of the Streamlit page's three
    columns: the variable's path, the aggregation method that collapses its
    array to a single number, and the direction (minimize or maximize).
    Aggregation options and the direction's help text come from `core`,
    shared with the Streamlit page.

    default_aggregation_label preselects an aggregation for every row;
    it defaults to the first entry of `core.AGGREGATION_METHODS_BY_LABEL`,
    matching the Streamlit selectbox's own default. Every row starts out
    minimizing, as the Streamlit page's do.
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
    direction_dropdowns: dict[NetcdfVariablePath, widgets.Dropdown] = {}
    rows: list[widgets.HBox] = []

    for var_path in variable_paths:
        aggregation_dropdowns[var_path] = widgets.Dropdown(
            options=aggregation_labels,
            value=default_aggregation_label,
            layout=widgets.Layout(width="320px"),
        )
        direction_dropdowns[var_path] = widgets.Dropdown(
            # (label, value) pairs: the dropdown's value is the Direction
            # itself, so reading it back needs no lookup.
            options=[
                (direction.value.capitalize(), direction)
                for direction in opti_core.Direction
            ],
            value=opti_core.Direction.MINIMIZE,
            tooltip=opti_core.DIRECTION_HELP,
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
                    direction_dropdowns[var_path],
                ]
            )
        )

    display(widgets.VBox(rows))

    return TargetSpecWidgets(
        aggregation_dropdowns=aggregation_dropdowns,
        direction_dropdowns=direction_dropdowns,
    )


def target_specs_from_widgets(
    target_spec_widgets: TargetSpecWidgets,
) -> dict[NetcdfVariablePath, opti_core.TargetSpec]:
    """
    Read the widgets' current choices into target specs.

    The result is what `core.prepare_optimization_data()` takes as
    `target_specs`. Its insertion order sets the column order of the
    optimization's target vectors, so it also fixes the axis order of
    display_pareto_corner_plot()'s labels.
    """
    return {
        var_path: opti_core.TargetSpec(
            aggregation=opti_core.AGGREGATION_METHODS_BY_LABEL[dropdown.value],
            direction=target_spec_widgets.direction_dropdowns[var_path].value,
        )
        for var_path, dropdown in target_spec_widgets.aggregation_dropdowns.items()
    }


def format_target_specs_as_code(
    target_specs: dict[NetcdfVariablePath, opti_core.TargetSpec],
) -> str:
    """
    Write target specs out as Python that rebuilds them.

    The returned code assigns `target_specs`, and runs as-is in a notebook
    that imports `TargetSpec`, `Direction`, `NetcdfVariableArray` and
    `NetcdfVariablePath`. Printing it lets a user pick their targets with the
    widgets once, then paste the result in place of the widget cells.

    An aggregation is written by its `__qualname__`, e.g.
    `NetcdfVariableArray.mean_of_all_values`. That works for every method of
    `NetcdfVariableArray`, which covers all the aggregations the widgets
    offer. Anything else, such as a hand-written lambda, raises instead of
    printing code that would not run.
    """
    lines = ["target_specs = {"]
    for var_path, target_spec in target_specs.items():
        # getattr: the aggregation's type is a bare Callable, which need not
        # have a __qualname__.
        aggregation_name = getattr(target_spec.aggregation, "__qualname__", "")
        if not aggregation_name.startswith("NetcdfVariableArray."):
            raise ValueError(
                f"Cannot print the aggregation of '{var_path}' as code: "
                f"{target_spec.aggregation!r} is not a method of "
                "NetcdfVariableArray."
            )
        lines += [
            f'    NetcdfVariablePath("{var_path}"): TargetSpec(',
            f"        aggregation={aggregation_name},",
            f"        direction=Direction.{target_spec.direction.name},",
            "    ),",
        ]
    lines.append("}")
    return "\n".join(lines)


def display_pareto_corner_plot(
    results: opti_core.OptimizationResults,
    labels: list[NetcdfVariablePath],
) -> plt.Figure:
    """
    Display the Pareto front as a corner plot, against the random points.

    labels names one target variable per axis, in the same order as the
    target_specs passed to the optimization -- e.g.
    `list(target_specs.keys())`.

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
