"""
Notebook equivalent of `analysis.streamlit.pages.annamari_figures`.

Reuses `components/plots.py`'s `stand`/`hydrology`/`mass`/`carbon`/
`nutrient_balance` functions unchanged -- they're already
Streamlit-independent, pure matplotlib -- and the fixed variable-path list
unchanged (copied below, since it's defined inline in the Streamlit page,
not exported for reuse). Replaces `st.pyplot`/`plt.close` display wrapping
with plain inline matplotlib display.

Per #221, one figure/section per cell: rather than one function that dumps
all 7 dashboards at once, each section gets its own `display_*` function, so
the eventual report notebook can call one per cell.

Typical usage, one call per cell:

    report_data = canned_report.load_report_data(netcdf_filepath)

    # -- next cell --
    canned_report.display_stand(report_data)

    # -- next cell --
    canned_report.display_hydrology(report_data)

    # -- next cell --
    canned_report.display_mass(report_data)

    # -- next cell --
    canned_report.display_carbon(report_data)

    # -- next cell --
    canned_report.display_nutrient_balance(report_data, substance="N")

    # -- next cell --
    canned_report.display_nutrient_balance(report_data, substance="P")

    # -- next cell --
    canned_report.display_nutrient_balance(report_data, substance="K")
"""

from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import susi.io.load_output_data as load_output
from analysis.streamlit.components import plots
from susi.io.load_output_data import NetcdfVariableArray, NetcdfVariablePath

# Same fixed variable list as the Streamlit page
# (analysis.streamlit.pages.annamari_figures). Copied rather than imported since
# it's defined inline there, not exported for reuse.
VARIABLE_PATHS = (
    load_output.NetcdfVariablePath("/strip/dwtyr"),
    load_output.NetcdfVariablePath("/strip/dwtyr_growingseason"),
    load_output.NetcdfVariablePath("/strip/dwtyr_latesummer"),
    load_output.NetcdfVariablePath("/strip/dwt"),
    load_output.NetcdfVariablePath("/strip/roff"),
    load_output.NetcdfVariablePath("/strip/roffwest"),
    load_output.NetcdfVariablePath("/strip/roffeast"),
    load_output.NetcdfVariablePath("/strip/surfacerunoff"),
    load_output.NetcdfVariablePath("/strip/deltas"),
    load_output.NetcdfVariablePath("/strip/elevation"),
    load_output.NetcdfVariablePath("/cpy/ET_yr"),
    load_output.NetcdfVariablePath("/cpy/transpi_yr"),
    load_output.NetcdfVariablePath("/cpy/efloor_yr"),
    load_output.NetcdfVariablePath("/cpy/SWEmax"),
    load_output.NetcdfVariablePath("/cpy/interc_yr"),
    load_output.NetcdfVariablePath("/stand/volume"),
    load_output.NetcdfVariablePath("/stand/dominant/volume"),
    load_output.NetcdfVariablePath("/stand/subdominant/volume"),
    load_output.NetcdfVariablePath("/stand/under/volume"),
    load_output.NetcdfVariablePath("/stand/logvolume"),
    load_output.NetcdfVariablePath("/stand/pulpvolume"),
    load_output.NetcdfVariablePath("/stand/leafmass"),
    load_output.NetcdfVariablePath("/stand/dominant/leafmass"),
    load_output.NetcdfVariablePath("/stand/dominant/leafmax"),
    load_output.NetcdfVariablePath("/stand/dominant/leafmin"),
    load_output.NetcdfVariablePath("/stand/subdominant/leafmass"),
    load_output.NetcdfVariablePath("/stand/subdominant/leafmax"),
    load_output.NetcdfVariablePath("/stand/subdominant/leafmin"),
    load_output.NetcdfVariablePath("/stand/under/leafmass"),
    load_output.NetcdfVariablePath("/stand/under/leafmax"),
    load_output.NetcdfVariablePath("/stand/under/leafmin"),
    load_output.NetcdfVariablePath("/stand/nut_stat"),
    load_output.NetcdfVariablePath("/stand/dominant/NPP"),
    load_output.NetcdfVariablePath("/stand/dominant/NPP_pot"),
    load_output.NetcdfVariablePath("/stand/n_demand"),
    load_output.NetcdfVariablePath("/stand/p_demand"),
    load_output.NetcdfVariablePath("/stand/k_demand"),
    load_output.NetcdfVariablePath("/stand/nonwoodylitter"),
    load_output.NetcdfVariablePath("/stand/woodylitter"),
    load_output.NetcdfVariablePath("/stand/finerootlitter"),
    load_output.NetcdfVariablePath("/stand/biomass"),
    load_output.NetcdfVariablePath("/stand/volumegrowth"),
    load_output.NetcdfVariablePath("/groundvegetation/ds_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/h_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/s_litterfall"),
    load_output.NetcdfVariablePath("/groundvegetation/gv_tot"),
    load_output.NetcdfVariablePath("/esom/Mass/out"),
    load_output.NetcdfVariablePath("/esom/Mass/L0L"),
    load_output.NetcdfVariablePath("/esom/Mass/L0W"),
    load_output.NetcdfVariablePath("/esom/Mass/LL"),
    load_output.NetcdfVariablePath("/esom/Mass/LW"),
    load_output.NetcdfVariablePath("/esom/Mass/FL"),
    load_output.NetcdfVariablePath("/esom/Mass/FW"),
    load_output.NetcdfVariablePath("/esom/Mass/H"),
    load_output.NetcdfVariablePath("/esom/Mass/P1"),
    load_output.NetcdfVariablePath("/esom/Mass/P2"),
    load_output.NetcdfVariablePath("/esom/Mass/P3"),
    load_output.NetcdfVariablePath("/balance/C/LMWdoc_to_water"),
    load_output.NetcdfVariablePath("/balance/C/HMW_to_water"),
    load_output.NetcdfVariablePath("/balance/C/LMWdoc_to_atm"),
    load_output.NetcdfVariablePath("/balance/C/HMW_to_atm"),
    load_output.NetcdfVariablePath("/balance/C/co2c_release"),
    load_output.NetcdfVariablePath("/balance/C/ch4c_release"),
    load_output.NetcdfVariablePath("/balance/C/stand_litter_in"),
    load_output.NetcdfVariablePath("/balance/C/gv_litter_in"),
    load_output.NetcdfVariablePath("/balance/C/soil_c_balance_c"),
    load_output.NetcdfVariablePath("/balance/C/soil_c_balance_co2eq"),
    load_output.NetcdfVariablePath("/balance/C/stand_c_balance_c"),
    load_output.NetcdfVariablePath("/balance/C/stand_c_balance_co2eq"),
    load_output.NetcdfVariablePath("/balance/N/to_water"),
    load_output.NetcdfVariablePath("/balance/N/decomposition_below_root_lyr"),
    load_output.NetcdfVariablePath("/balance/N/decomposition_tot"),
    load_output.NetcdfVariablePath("/balance/N/decomposition_root_lyr"),
    load_output.NetcdfVariablePath("/balance/N/deposition"),
    load_output.NetcdfVariablePath("/balance/N/fertilization_release"),
    load_output.NetcdfVariablePath("/balance/N/stand_demand"),
    load_output.NetcdfVariablePath("/balance/N/gv_demand"),
    load_output.NetcdfVariablePath("/balance/N/balance_root_lyr"),
    load_output.NetcdfVariablePath("/balance/P/to_water"),
    load_output.NetcdfVariablePath("/balance/P/decomposition_below_root_lyr"),
    load_output.NetcdfVariablePath("/balance/P/decomposition_tot"),
    load_output.NetcdfVariablePath("/balance/P/decomposition_root_lyr"),
    load_output.NetcdfVariablePath("/balance/P/deposition"),
    load_output.NetcdfVariablePath("/balance/P/fertilization_release"),
    load_output.NetcdfVariablePath("/balance/P/stand_demand"),
    load_output.NetcdfVariablePath("/balance/P/gv_demand"),
    load_output.NetcdfVariablePath("/balance/P/balance_root_lyr"),
    load_output.NetcdfVariablePath("/balance/K/to_water"),
    load_output.NetcdfVariablePath("/balance/K/decomposition_below_root_lyr"),
    load_output.NetcdfVariablePath("/balance/K/decomposition_tot"),
    load_output.NetcdfVariablePath("/balance/K/decomposition_root_lyr"),
    load_output.NetcdfVariablePath("/balance/K/deposition"),
    load_output.NetcdfVariablePath("/balance/K/fertilization_release"),
    load_output.NetcdfVariablePath("/balance/K/stand_demand"),
    load_output.NetcdfVariablePath("/balance/K/gv_demand"),
    load_output.NetcdfVariablePath("/balance/K/balance_root_lyr"),
)


def load_report_data(
    netcdf_filepath: Path,
) -> dict[NetcdfVariablePath, NetcdfVariableArray]:
    """Read every VARIABLE_PATHS variable from a single netcdf file."""
    return load_output.read_value_several_variables_from_single_file(
        netcdf_filepath=netcdf_filepath, variable_paths=VARIABLE_PATHS
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
