import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import streamlit as st


def pick_folder() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return Path(folder)


st.header("Settings")

col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.markdown("**Projects root**")

with col2:
    st.write(st.session_state.settings["projects_root"])

with col3:
    if st.button("Browse…", use_container_width=True):
        result = pick_folder()
        if result:
            st.session_state.settings["projects_root"] = result
            st.rerun()

st.write()
