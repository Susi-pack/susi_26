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

    df_summary = summary_table.display_summary_table(project_dropdown.value)
"""

from pathlib import Path

import pandas as pd
from IPython.display import display

import susi.io.load_output_data as load_output
from analysis.shared_reporting_utils.project_summary import (
    SUMMARY_VARIABLES,
    build_summary_dataframe,
)


def display_summary_table(dir_path: Path) -> pd.DataFrame:
    """
    Compute and display the per-stand/scenario summary table for dir_path.

    dir_path is a project folder containing one subfolder per stand, each
    holding one subfolder per scenario -- e.g. the `.value` of a project-level
    `folder_selection.build_dropdown()`.

    Returns the aggregated DataFrame (sorted by stand, then scenario) so a
    later cell can reuse it without recomputing.
    """
    stand_folderpaths = load_output.list_subdirectories(path=dir_path)
    metadata_by_stand = load_output.load_all_metadatas_from_stands(
        folders=stand_folderpaths
    )

    data_store = load_output.read_netcdf_files_for_selected_variables_from_metadatas(
        selected_variables=[var.netcdf_path for var in SUMMARY_VARIABLES],
        metadata_by_stand=metadata_by_stand,
    )

    df_summary = build_summary_dataframe(data_store, SUMMARY_VARIABLES)
    display(df_summary)
    return df_summary
