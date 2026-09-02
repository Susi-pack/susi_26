"""
Shared row-shaping logic for the "Differing Parameters" section of the
Parameter Comparison view.

Reused by both the Streamlit page
(`analysis.streamlit.pages.compare_scenarios_for_stand`) and the notebook
component (`analysis.notebooks.components.param_comparison`).

Extracted per #235 to eliminate the duplicated "value -> comma-joined
scenario list" row-shaping step that used to be independently reimplemented
in each frontend.
"""

from typing import Any

from analysis.core.parse_outputs import ParamName


def shape_differing_params(
    differing: dict[ParamName, dict[Any, list]],
) -> dict[ParamName, list[dict[str, Any]]]:
    """
    Shape `find_differing_params`'s output into per-parameter display rows.

    For each differing parameter, turns its {value: [scenario, ...]}
    mapping into an ordered list of {"value": value, "scenarios": "a, b, c"}
    rows -- one row per distinct value, with its scenarios comma-joined.
    """
    return {
        param_name: [
            {"value": value, "scenarios": ", ".join(scenarios)}
            for value, scenarios in value_map.items()
        ]
        for param_name, value_map in differing.items()
    }
