import streamlit as st
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import susi.io.netcdf_utils as nc_utils


def pick_folder_popup() -> Path:
    """
    Uses TK
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    folder = filedialog.askdirectory()
    root.destroy()
    return Path(folder)


def build_folder_selection_widget(dir_path: Path, label: str) -> Path:
    subdirs = nc_utils.list_subdirectories(dir_path)
    dir_names = [dir.name for dir in subdirs]

    selected_dir_name = st.selectbox(
        label=f"Choose {label} folder",
        options=dir_names,
    )
    return dir_path / selected_dir_name
