"""
Notebook equivalent of `analysis.streamlit.pages.project_summary`.

Per #215's picker/renderer split, this is a renderer element: it computes
and displays the per-stand/scenario aggregated summary table, and also
returns the underlying DataFrame so a later cell can reuse it without
recomputing.

Reuses the shared aggregation logic (`SUMMARY_VARIABLES`,
`build_summary_dataframe`) from `analysis.shared_reporting_utils.project_summary`,
per #230. Drops the Streamlit page's `@st.cache_data` layer: a notebook's
natural cell-reuse -- rerunning this cell only when its inputs change --
serves as the equivalent, so
`read_netcdf_files_for_selected_variables_from_metadatas` is called directly
instead of through `netcdf_reader.cached_read_netcdf_files`.

Typical usage:

    df_summary = summary_table.display_summary_table(run_dropdown.value)
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from IPython.display import display

import susi.io.load_output_data as load_output
from analysis.shared_reporting_utils.project_summary import (
    SUMMARY_VARIABLES,
    AggregatedNetcdfVar,
    build_summary_dataframe,
)


def display_summary_table(
    run_dirpath: Path, variables: Sequence[AggregatedNetcdfVar] = SUMMARY_VARIABLES
) -> pd.DataFrame:
    """
    Compute and display the per-stand/scenario summary table for run_dirpath.

    run_dirpath is one run of a project -- `projects/<project>/outputs/<run_id>/`,
    the `.value` of `folder_selection.build_run_dropdown()`. It contains one
    subfolder per stand, each holding one subfolder per scenario.

    Returns the aggregated DataFrame (sorted by stand, then scenario) so a
    later cell can reuse it without recomputing.
    """
    stand_folderpaths = load_output.list_stand_folders(run_dirpath=run_dirpath)
    metadata_by_stand = load_output.load_all_metadatas_from_stands(
        folders=stand_folderpaths
    )

    data_store = load_output.read_netcdf_files_for_selected_variables_from_metadatas(
        selected_variables=[var.netcdf_path for var in variables],
        metadata_by_stand=metadata_by_stand,
    )

    df_summary = build_summary_dataframe(data_store, variables)
    # Show every row and column -- pandas' defaults for `display.max_rows`
    # (60) and `display.max_columns` (20) would otherwise collapse the
    # middle rows/columns into an ellipsis for this table.
    with pd.option_context(
        "display.max_columns", None, "display.max_rows", None, "display.width", None
    ):
        display(df_summary)
    return df_summary
