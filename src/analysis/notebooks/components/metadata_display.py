"""
Notebook equivalent of `analysis.gui.components.metadata_expander`.

Metadata display is not itself a picker or a renderer over freshly-computed
data (see #215's picker/renderer split) — the caller already holds
`metadata`/`susi_params` (e.g. from `load_output.read_params_from_jsons`), so
this function's only job is to display them; there is nothing new to hand
back.

Typical usage:

    params = load_output.read_params_from_jsons(experiment_folderpath=chosen_scenario_folder)
    metadata_display.display_metadata(params.metadata, params.susi_params)
"""

from IPython.display import display, JSON, Markdown


def display_metadata(metadata: dict, susi_params: dict) -> None:
    """
    Display metadata and susi_params as collapsible JSON tree views.

    Replaces the Streamlit version's `st.expander` + `st.json` with
    `IPython.display.JSON`, which natively renders a collapsible tree in
    Jupyter — no extra dependency needed. Both trees start collapsed
    (`expanded=False`) so they don't clutter the notebook, matching the
    Streamlit expander's default `expanded=False`.
    """
    display(Markdown("**metadata**"))
    display(JSON(metadata, expanded=False))
    display(Markdown("**Susi params**"))
    display(JSON(susi_params, expanded=False))
