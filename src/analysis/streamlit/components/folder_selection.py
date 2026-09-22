import streamlit as st
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import susi.io.load_output_data as load_output
from susi.io.project_layout import outputs_dir_for_project


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
    subdirs = load_output.list_subdirectories_sorted(dir_path)
    dir_names = [dir.name for dir in subdirs]

    selected_dir_name = st.selectbox(
        label=f"Choose {label} folder",
        options=dir_names,
    )
    return dir_path / selected_dir_name


def build_run_selection_widget(projects_root: Path) -> Path:
    """
    Pick a project and then one of its runs, returning the run's folder.

    The full output path is
    `<projects_root>/<project>/outputs/<run_id>/<stand_id>/<scenario_id>`, so
    a page that wants stands has to get past two levels first. `outputs/` is
    not one of them: a project has exactly one, and offering it as a dropdown
    choice is what makes the next dropdown list `inputs` and `outputs` as if
    they were stands. So only the project and the run are chosen; the
    `outputs/` hop is composed in.

    Stand and scenario dropdowns chain off the returned path the usual way.
    """
    project_dirpath = build_folder_selection_widget(
        dir_path=projects_root, label="project"
    )

    outputs_dirpath = outputs_dir_for_project(project_dirpath)
    if not outputs_dirpath.is_dir():
        # A project that has inputs but has never been run. Say so here,
        # rather than letting the run dropdown fail on a missing directory.
        raise ValueError(
            f"Project {project_dirpath.name} has no outputs/ folder (expected "
            f"at {outputs_dirpath}). Nothing has been run for it yet."
        )

    return build_folder_selection_widget(dir_path=outputs_dirpath, label="run")
