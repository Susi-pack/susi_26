import streamlit as st
from susi.io.app_settings import AppSettings


def load_default_settings_into_session_state() -> None:
    # guard ensuring initialization only happens once, not on every rerender
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "data_folder": AppSettings().output_folder,
        }

    return None


load_default_settings_into_session_state()

pages = [
    st.Page("pages/1_single_netcdf.py", title="Single"),
    st.Page("pages/2_multiple_netcdf.py", title="Multiple"),
    st.Page("pages/9_settings.py", title="Settings"),
]

pg = st.navigation(pages)
pg.run()
