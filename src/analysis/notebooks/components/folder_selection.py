"""
Notebook equivalent of `analysis.gui.components.folder_selection`.

Reactivity model (see #215, #217): one selection step per notebook cell,
sequential re-run. No `.observe()`-based reactive cascading between dependent
widgets. Each builder function displays its widget as a side effect and also
returns it, so the caller can read the user's choice from the widget's native
attribute (`Dropdown.value`, `FileChooser.selected`) in a later cell, once the
notebook has been re-run up to that point and the user has made a selection.

Typical chained usage, one call per cell:

    project_dropdown = folder_selection.build_dropdown(DATA_FOLDER, label="project")
    project_dropdown

    # -- next cell, run after picking a value above --
    stand_dropdown = folder_selection.build_dropdown(
        project_dropdown.value, label="stand"
    )
    stand_dropdown

    # -- next cell --
    scenario_dropdown = folder_selection.build_dropdown(
        stand_dropdown.value, label="scenario"
    )
    scenario_dropdown

    # -- next cell --
    chosen_scenario_folder = scenario_dropdown.value
"""

from pathlib import Path

import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser

import susi.io.load_output_data as load_output


def build_dropdown(dir_path: Path, label: str) -> widgets.Dropdown:
    """
    Build, display, and return a Dropdown listing the subdirectories of dir_path.

    Dropdown options are (name, Path) pairs, so `.value` on the returned
    widget is already a Path, ready to pass straight into the next
    build_dropdown() call in the chain.

    Raises ValueError, labeled with which selection step failed, if dir_path
    doesn't exist (e.g. a stale value from an earlier `.value` in the chain)
    or has no subdirectories: an empty Dropdown's `.value` is silently None,
    which would otherwise crash confusingly deep inside the next chained
    call instead of at the point of the actual problem.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"{label} folder not found: {dir_path}")

    subdirs = load_output.list_subdirectories_sorted(dir_path)
    if not subdirs:
        raise ValueError(f"No {label} subfolders found in {dir_path}")

    dropdown = widgets.Dropdown(
        options=[(subdir.name, subdir) for subdir in subdirs],
        description=f"{label}:",
    )
    display(dropdown)
    return dropdown


def build_folder_browser(start_path: Path) -> FileChooser:
    """
    Build, display, and return a directory-only file chooser rooted at start_path.

    Replaces the Streamlit version's native OS folder-picker popup (tkinter).
    That popup can't be ported as-is: in a CSC JupyterLab session the notebook
    kernel runs on a remote compute node, so a tkinter dialog would try to open
    a GUI window on that node rather than on the user's laptop. This widget's
    directory listing instead runs in the kernel process itself, so it
    correctly browses the compute node's filesystem.

    Navigating to a directory alone does not set `.selected` — the user must
    also click the widget's "Select" button to commit the choice. Once they've
    done that, read `.selected` (a str path) from the returned widget in a
    later cell.
    """
    chooser = FileChooser(str(start_path), show_only_dirs=True)
    display(chooser)
    return chooser
