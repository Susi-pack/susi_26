"""
Notebook equivalent of `analysis.gui.components.netcdf_variable_explorer`.

Flattens the Streamlit tree-select (~90 hierarchical netcdf variable paths,
via `streamlit_tree_select`) to a filterable, checkbox-based multi-select
list, built from core `ipywidgets` only -- no `ipytree` dependency, since
the tree's grouping is presentational and isn't needed for functional
equivalence (see #215, #220).

Per #215's picker/renderer split, this is a picker element: it displays its
widget as a side effect and returns it, so the caller reads the checked
variables from it in a later cell, once the user has ticked some boxes.

Typical usage, one call per cell:

    selector = variable_selection.build_selector(all_variables)

    # -- next cell, run after checking some boxes above --
    chosen_variables = variable_selection.checked_variables(selector, all_variables)
"""

from dataclasses import dataclass

import ipywidgets as widgets
from IPython.display import display

from susi.io.load_output_data import NetcdfVariableInfo, NetcdfVariablePath


@dataclass(frozen=True)
class VariableSelector:
    """
    The filter Text widget plus one Checkbox per variable, keyed by path.

    Returned by build_selector() so a later cell can read which boxes are
    checked via checked_variables(). filter_text and container are exposed
    mainly for completeness/debugging -- reading selections only needs
    checkboxes (via checked_variables()).
    """

    filter_text: widgets.Text
    checkboxes: dict[NetcdfVariablePath, widgets.Checkbox]
    container: widgets.VBox


def build_selector(
    variables: dict[NetcdfVariablePath, NetcdfVariableInfo],
    preselected: list[NetcdfVariablePath] | None = None,
) -> VariableSelector:
    """
    Build, display, and return a filterable checkbox list over variables.

    Each checkbox is labeled with the variable's full path, with a tooltip
    showing its shape and units (replacing the Streamlit tree node's `title`
    tooltip). Typing in the filter box hides checkboxes whose path doesn't
    contain the typed text (case-insensitive substring match); filtering
    only shows/hides checkboxes, it never unchecks them, so a selection
    survives changing or clearing the filter.
    """
    preselected_set = set(preselected or ())

    checkboxes: dict[NetcdfVariablePath, widgets.Checkbox] = {}
    for var_path, var_info in variables.items():
        checkboxes[var_path] = widgets.Checkbox(
            value=var_path in preselected_set,
            description=var_path,
            tooltip=f"Shape: {var_info.shape}\nUnits: {var_info.units or 'N/A'}",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "initial"},
        )

    filter_text = widgets.Text(
        placeholder="Filter variables by path…", description="Filter:"
    )

    def _apply_filter(change) -> None:
        needle = change["new"].strip().lower()
        for var_path, checkbox in checkboxes.items():
            checkbox.layout.display = "" if needle in var_path.lower() else "none"

    filter_text.observe(_apply_filter, names="value")

    checklist_box = widgets.VBox(
        list(checkboxes.values()),
        layout=widgets.Layout(
            height="400px", overflow_y="auto", border="1px solid lightgray"
        ),
    )
    container = widgets.VBox([filter_text, checklist_box])
    display(container)

    return VariableSelector(
        filter_text=filter_text, checkboxes=checkboxes, container=container
    )


def checked_variables(
    selector: VariableSelector,
    variables: dict[NetcdfVariablePath, NetcdfVariableInfo],
) -> dict[NetcdfVariablePath, NetcdfVariableInfo]:
    """Return the subset of variables whose checkbox is currently checked."""
    return {
        var_path: var_info
        for var_path, var_info in variables.items()
        if selector.checkboxes[var_path].value
    }
