"""
Notebook equivalent of `analysis.streamlit.pages.single_scenario_dashboard`.

Reuses `shared_reporting_utils/single_scenario_dashboard.py`'s
`VARIABLE_PATHS`/`load_report_data` for the fixed variable-path list and
data loading, and its `SECTIONS` for the per-section (title, plot_fn)
pairing (per #233) -- `plots.py`'s `stand`/`hydrology`/`mass`/`carbon`/
`nutrient_balance` functions are reached only through `SECTIONS`, not
imported directly. Replaces `st.pyplot`/`plt.close` display wrapping with
plain inline matplotlib display.

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
    title, plot_fn = shared_single_scenario_dashboard.SECTIONS["stand"]
    _display_figure(title, plot_fn(data=data))


def display_hydrology(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    title, plot_fn = shared_single_scenario_dashboard.SECTIONS["hydrology"]
    _display_figure(title, plot_fn(data=data))


def display_mass(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    title, plot_fn = shared_single_scenario_dashboard.SECTIONS["mass"]
    _display_figure(title, plot_fn(data=data))


def display_carbon(data: dict[NetcdfVariablePath, NetcdfVariableArray]) -> None:
    title, plot_fn = shared_single_scenario_dashboard.SECTIONS["carbon"]
    _display_figure(title, plot_fn(data=data))


def display_nutrient_balance(
    data: dict[NetcdfVariablePath, NetcdfVariableArray], substance: str
) -> None:
    """substance is one of "N", "P", "K"."""
    title, plot_fn = shared_single_scenario_dashboard.SECTIONS[substance]
    _display_figure(title, plot_fn(data=data))
