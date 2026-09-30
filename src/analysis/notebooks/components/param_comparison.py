"""
Notebook equivalent of the parameter-comparison display in
`analysis.streamlit.pages.compare_scenarios_for_stand`.

Reuses `find_differing_params`/`find_unique_params` from the shared backend
unchanged (they're already Streamlit-independent), and
`shared_reporting_utils/param_comparison.py`'s `shape_differing_params` for
the value -> comma-joined-scenarios row-shaping step (per #235). Per #215's
picker/renderer split, this is a renderer element: it displays its output and
also returns the underlying (differing, unique) dicts, so a later cell can
reuse them without recomputing.

Typical usage:

    differing, unique = param_comparison.display_param_comparison(stand_id, output_dir)
"""

from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import Markdown, display

from analysis.core.parse_outputs import (
    ParamName,
    find_differing_params,
    find_unique_params,
)
from analysis.shared_reporting_utils.param_comparison import shape_differing_params
from susi.io.load_output_data import StandID


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
    `find_differing_params`/`find_unique_params`, so callers can reuse them
    without recomputing.
    """
    differing = find_differing_params(stand_id, output_dir)
    unique = find_unique_params(stand_id, output_dir)
    shaped_differing = shape_differing_params(differing)

    display(Markdown("## Differing Parameters"))
    if shaped_differing:
        for param_name, rows in shaped_differing.items():
            display(Markdown(f"**{param_name}**"))
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
