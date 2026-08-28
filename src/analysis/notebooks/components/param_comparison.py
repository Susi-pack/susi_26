"""
Notebook equivalent of the parameter-comparison display in
`analysis.streamlit.pages.compare_scenarios_for_stand`.

Reuses `find_differing_params`/`find_unique_params` from the shared backend
unchanged (they're already Streamlit-independent). Per #215's picker/renderer
split, this is a renderer element: it displays its output and also returns
the underlying (differing, unique) dicts, so a later cell can reuse them
without recomputing.

Typical usage:

    differing, unique = param_comparison.display_param_comparison(stand_id, output_dir)
"""

from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import display, Markdown

from susi.io.load_output_data import StandID
from analysis.core.parse_outputs import (
    ParamName,
    find_differing_params,
    find_unique_params,
)

# Scenario name is how scenarios are told apart in the first place, so it
# always "differs" across them — showing it as a differing parameter would
# just be noise. Matches the Streamlit version's skip of the same param.
_SCENARIO_NAME_PARAM = "site_parameters/scenario_name"


def display_param_comparison(
    stand_id: StandID, output_dir: Path
) -> tuple[dict[ParamName, dict[Any, list]], dict[ParamName, Any]]:
    """
    Compute and display differing/unique parameters across a stand's scenarios.

    Displays:
    - "Differing Parameters": one small DataFrame per differing parameter,
      mapping each value to the scenarios that have it.
    - "Unique Parameters": a Markdown bullet list of parameters with the
      same value across every scenario.

    Returns the (differing, unique) dicts exactly as produced by
    `find_differing_params`/`find_unique_params` (unfiltered — the
    scenario-name skip above is a display-only concern), so callers can
    reuse them without recomputing.
    """
    differing = find_differing_params(stand_id, output_dir)
    unique = find_unique_params(stand_id, output_dir)

    display(Markdown("## Differing Parameters"))
    displayable = {
        name: value_map
        for name, value_map in differing.items()
        if name != _SCENARIO_NAME_PARAM
    }
    if displayable:
        for param_name, value_map in displayable.items():
            display(Markdown(f"**{param_name}**"))
            rows = [
                {"value": value, "scenarios": ", ".join(scenarios)}
                for value, scenarios in value_map.items()
            ]
            display(pd.DataFrame(rows))
    else:
        display(Markdown("*No differing parameters found across scenarios.*"))

    display(Markdown("## Unique Parameters (Same Across All Scenarios)"))
    if unique:
        display(
            Markdown(
                "\n".join(f"- **{name}**: `{value}`" for name, value in unique.items())
            )
        )
    else:
        display(Markdown("*No unique parameters found (all parameters differ).*"))

    return differing, unique
