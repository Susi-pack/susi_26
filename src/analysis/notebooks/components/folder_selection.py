"""
Notebook equivalent of `analysis.streamlit.components.folder_selection`.

Reactivity model (see #215, #217): one selection step per notebook cell,
sequential re-run. No `.observe()`-based reactive cascading between dependent
widgets. Each builder function displays its widget as a side effect and also
returns it, so the caller can read the user's choice from the widget's native
attribute (`Dropdown.value`, `FileChooser.selected`) in a later cell, once the
notebook has been re-run up to that point and the user has made a selection.

Typical chained usage, one call per cell:

    project_dropdown = folder_selection.build_dropdown(PROJECTS_ROOT, label="project")
    project_dropdown

    # -- next cell, run after picking a value above --
    run_dropdown = folder_selection.build_run_dropdown(project_dropdown.value)
    run_dropdown

    # -- next cell --
    stand_dropdown = folder_selection.build_dropdown(
        run_dropdown.value, label="stand"
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
from susi.io.project_layout import outputs_dir_for_project


def build_dropdown(dir_path: Path | str, label: str) -> widgets.Dropdown:
    """
    Build, display, and return a Dropdown listing the subdirectories of dir_path.

    Dropdown options are (name, Path) pairs, so `.value` on the returned
    widget is already a Path, ready to pass straight into the next
    build_dropdown() call in the chain. dir_path also accepts a str, so a
    build_folder_browser() FileChooser's `.selected` (a str, not a Path) can
    be chained straight in too.

    Raises ValueError, labeled with which selection step failed, if dir_path
    doesn't exist (e.g. a stale value from an earlier `.value` in the chain)
    or has no subdirectories: an empty Dropdown's `.value` is silently None,
    which would otherwise crash confusingly deep inside the next chained
    call instead of at the point of the actual problem.
    """
    dir_path = Path(dir_path)
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


def build_run_dropdown(project_dirpath: Path | str) -> widgets.Dropdown:
    """
    Build, display, and return the run dropdown for an already-chosen project.

    A project's results live at
    `<project>/outputs/<run_id>/<stand_id>/<scenario_id>`, so the chain from a
    project down to a stand has two steps in it, not one. Only one of them is
    a choice: `outputs/` gets no dropdown, because a project has exactly one
    and offering it as an option is what makes the next dropdown in the chain
    list a project's `inputs` and `outputs` as if they were stands.

    The stand dropdown chains off the returned widget's `.value` as usual.
    """
    outputs_dirpath = outputs_dir_for_project(Path(project_dirpath))
    if not outputs_dirpath.is_dir():
        # A project that has inputs but has never been run. Said here rather
        # than left to build_dropdown, which would report the project's
        # missing outputs/ folder as a missing "run" folder.
        raise ValueError(
            f"Project {Path(project_dirpath).name} has no outputs/ folder "
            f"(expected at {outputs_dirpath}). Nothing has been run for it yet."
        )

    return build_dropdown(outputs_dirpath, label="run")


def resolve_start_folder(chooser: FileChooser, default: Path) -> Path:
    """
    Return chooser's committed selection, or default if none has been made yet.

    Navigating a build_folder_browser() FileChooser doesn't set `.selected`
    until the user clicks its "Select" button (see that function's
    docstring), so a page notebook that offers browsing as an alternative to
    a default starting folder needs this "has the user picked something yet"
    check before it can build its first build_dropdown() chain link.
    """
    return Path(chooser.selected) if chooser.selected else default


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
