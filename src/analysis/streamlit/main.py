import streamlit as st

from susi.io.app_settings import AppSettings


def load_default_settings_into_session_state() -> None:
    # guard ensuring initialization only happens once, not on every rerender
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "projects_root": AppSettings().projects_root,
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
    st.Page("pages/single_scenario_dashboard.py", title="Single scenario dashboard"),
    st.Page("pages/optimization.py", title="Optimization"),
    st.Page("pages/9_settings.py", title="Settings"),
]

# ty resolves st.navigation to the streamlit.navigation submodule rather than
# the navigation() function streamlit re-exports under the same name.
pg = st.navigation(pages)  # ty: ignore[call-non-callable]
pg.run()
