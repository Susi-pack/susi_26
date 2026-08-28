import streamlit as st
from typing import Sequence
from dataclasses import dataclass
from pathlib import Path

from susi.io.load_output_data import (
    read_netcdf_files_for_selected_variables,
    NetcdfVariablePath,
    NetcdfAggregationFn,
    StandID,
    ScenarioID,
    OutputDataStore,
)


@dataclass(frozen=True)
class AggregatedNetcdfVar:
    netcdf_path: NetcdfVariablePath
    aggregation_method: NetcdfAggregationFn
    display_label: str | None = (
        None  # Label to be displayed. If None, the netcdf_path is used.
    )


def aggregate_var(
    # Helper to reduce verbosity
    path: str,
    fn: NetcdfAggregationFn,
    label: str | None = None,
) -> AggregatedNetcdfVar:
    return AggregatedNetcdfVar(
        netcdf_path=NetcdfVariablePath(path),
        aggregation_method=fn,
        display_label=label,
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
    netcdf_filepaths_by_stand: dict[StandID, tuple[Path, ...]],
) -> OutputDataStore:
    return read_netcdf_files_for_selected_variables(
        selected_variables=[var.netcdf_path for var in selected_variables],
        scenarios_by_stand=scenarios_by_stand,
        netcdf_filepaths_by_stand=netcdf_filepaths_by_stand,
    )
