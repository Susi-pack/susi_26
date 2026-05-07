import streamlit as st
from pathlib import Path

from susi.io.app_settings import AppSettings


def load_default_settings_into_session_state() -> None:
    # guard ensuring initialization only happens once, not on every rerender
    if "settings" not in st.session_state:
        st.session_state.settings: Path = {
            "data_folder": AppSettings().output_folder,
        }

    return None


load_default_settings_into_session_state()

st.set_page_config(
    page_title="SUSI results", layout="wide", initial_sidebar_state="expanded"
)

pages = [
    st.Page("pages/project_summary.py", title="Project summary"),
    st.Page("pages/1_single_netcdf.py", title="Single"),
    st.Page("pages/2_multiple_netcdf.py", title="Multiple"),
    st.Page(
        "pages/compare_scenarios_for_stand.py", title="Compare scenarios single stand"
    ),
    st.Page("pages/annamari_figures.py", title="Annamari"),
    st.Page("pages/annamari_figures_altair.py", title="Annamari (Altair)"),
    st.Page("pages/9_settings.py", title="Settings"),
]

pg = st.navigation(pages)
pg.run()
