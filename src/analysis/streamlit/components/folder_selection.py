import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog

import streamlit as st

import susi.io.load_output_data as load_output
from susi.io import project_layout
from susi.io.project_layout import require_outputs_dir


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


@dataclass(frozen=True)
class ProjectAndRunSelection:
    """
    What the project and run dropdowns together picked.

    Both halves are kept, not just the run folder they compose into: the
    stand-area lookup needs the *project* (its `inputs/stand_data.json`) as
    well as the run (its stand folders), and recovering the project by walking
    `run_dir.parent.parent` back up would re-derive by convention something the
    picker already knows.
    """

    project_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        """`<project_dir>/outputs/<run_id>` -- what the stand dropdown chains off."""
        return project_layout.run_dir(project_dir=self.project_dir, run_id=self.run_id)


def build_project_and_run_selection_widget(
    projects_root: Path,
) -> ProjectAndRunSelection:
    """
    Pick a project and then one of its runs.

    The full output path is
    `<projects_root>/<project>/outputs/<run_id>/<stand_id>/<scenario_id>`, so
    a page that wants stands has to get past two levels first. `outputs/` is
    not one of them: a project has exactly one, and offering it as a dropdown
    choice is what makes the next dropdown list `inputs` and `outputs` as if
    they were stands. So only the project and the run are chosen; the
    `outputs/` hop is composed in.

    Stand and scenario dropdowns chain off the returned `.run_dir` the usual
    way.
    """
    project_dirpath = build_folder_selection_widget(
        dir_path=projects_root, label="project"
    )

    # `require_outputs_dir` rather than plain path composition: it reports a
    # project that has never been run, instead of leaving the run dropdown to
    # fail on a missing directory.
    outputs_dirpath = require_outputs_dir(project_dirpath)

    run_dirpath = build_folder_selection_widget(dir_path=outputs_dirpath, label="run")

    return ProjectAndRunSelection(project_dir=project_dirpath, run_id=run_dirpath.name)
