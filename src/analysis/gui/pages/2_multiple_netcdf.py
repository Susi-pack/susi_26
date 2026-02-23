import streamlit as st
import susi.io.netcdf_utils as nc_utils


metadata_by_stand = nc_utils.load_all_metadatas_from_folders(
    folders=nc_utils.list_subdirectories(path=st.session_state.settings["data_folder"])
)

st.write(metadata_by_stand)
