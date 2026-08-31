"""
Notebook equivalent of `analysis.streamlit.pages.single_scenario_dashboard`.

Reuses `shared_reporting_utils/plots.py`'s `stand`/`hydrology`/`mass`/
`carbon`/`nutrient_balance` functions unchanged -- they're already
Streamlit-independent, pure matplotlib -- and
`shared_reporting_utils/single_scenario_dashboard.py`'s `VARIABLE_PATHS`/
`load_report_data` for the fixed variable-path list and data loading.
Replaces `st.pyplot`/`plt.close` display wrapping with plain inline
matplotlib display.

Per #221, one figure/section per cell: rather than one function that dumps
all 7 dashboards at once, each section gets its own `display_*` function, so
the eventual report notebook can call one per cell.

Typical usage, one call per cell:

    report_data = single_scenario_dashboard.load_report_data(netcdf_filepath)

    # -- next cell --
    single_scenario_dashboard.display_stand(report_data)

    # -- next cell --
    single_scenario_dashboard.display_hydrology(report_data)

    # -- next cell --
    single_scenario_dashboard.display_mass(report_data)

    # -- next cell --
    single_scenario_dashboard.display_carbon(report_data)

    # -- next cell --
    single_scenario_dashboard.display_nutrient_balance(report_data, substance="N")

    # -- next cell --
    single_scenario_dashboard.display_nutrient_balance(report_data, substance="P")

    # -- next cell --
    single_scenario_dashboard.display_nutrient_balance(report_data, substance="K")
"""

from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from analysis.shared_reporting_utils import plots
from analysis.shared_reporting_utils import (
    single_scenario_dashboard as shared_single_scenario_dashboard,
)
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath


def load_report_data(
    netcdf_filepath: Path,
) -> dict[NetcdfVariablePath, NetcdfVariableArray]:
    """Read every VARIABLE_PATHS variable from a single netcdf file."""
    return shared_single_scenario_dashboard.load_report_data(
        netcdf_filepath=netcdf_filepath,
        variable_paths=shared_single_scenario_dashboard.VARIABLE_PATHS,
    )


def _display_figure(title: str, fig) -> None:
    display(Markdown(f"## {title}"))
    display(fig)
    plt.close(fig)


def display_stand(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    _display_figure("Stand", plots.stand(data=data))


def display_hydrology(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    _display_figure("Hydrology", plots.hydrology(data=data))


def display_mass(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    _display_figure("Mass", plots.mass(data=data))


def display_carbon(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    _display_figure("Carbon", plots.carbon(data=data))


_NUTRIENT_NAMES = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}


def display_nutrient_balance(
    data: dict[NetcdfVariablePath, NetcdfVariableArray], substance: str
) -> None:
    """substance is one of "N", "P", "K"."""
    title = f"{_NUTRIENT_NAMES.get(substance, substance)} Balance"
    _display_figure(title, plots.nutrient_balance(data=data, substance=substance))
