from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from analysis.shared_reporting_utils.project_summary import AggregatedNetcdfVar
from susi.io.load_output_data import (
    OutputDataStore,
    ScenarioID,
    StandID,
    read_netcdf_files_for_selected_variables,
)


# Cacheing to not re-read the same variables twice
@st.cache_data(
    hash_funcs={
        AggregatedNetcdfVar: lambda v: hash(
            (v.netcdf_path, v.aggregation_method.__name__, v.display_label)
        )
    }
)
def cached_read_netcdf_files(
    selected_variables: tuple[AggregatedNetcdfVar, ...],
    scenarios_by_stand: dict[StandID, Sequence[ScenarioID]],
    netcdf_filepaths_by_stand: dict[StandID, Sequence[Path]],
) -> OutputDataStore:
    return read_netcdf_files_for_selected_variables(
        selected_variables=[var.netcdf_path for var in selected_variables],
        scenarios_by_stand=scenarios_by_stand,
        netcdf_filepaths_by_stand=netcdf_filepaths_by_stand,
    )
