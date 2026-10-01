"""Where a project's files live.

A project is identified by its folder, wherever that is (ADR 0006): user
projects live under `AppSettings.projects_root`, tracked example and testing
projects in the checkout. Each holds its own `inputs/` and `outputs/`, so a
project's data and its results are co-located by construction:

    projects/
    └── paroninkorpi/
        ├── inputs/
        │   ├── allometry/
        │   │   └── new_growth/
        │   ├── stand_data.json
        │   └── config.toml
        ├── data/
        │   └── other: weather.csv, export.xml, ditch_depth.tif, ...
        └── outputs/
            └── <run_id>/<stand_id>/<scenario_id>/

Everything here is pure path composition -- no argparse, no I/O -- except
`require_project_dir`, which exists precisely to hit the filesystem. The
argparse-flavoured helpers that wrap these paths live in
`tools/shared_allometry_tool_utils/cli_paths.py` instead.
"""

from pathlib import Path

from susi.io.app_settings import AppSettings

# The per-project document describing every stand the project simulates.
STAND_DATA_FILENAME = "stand_data.json"

# The config file that xml_to_allometry.py and metsakeskus_to_allometry.py
# both look up in a project's inputs
CONFIG_FILENAME = "config.toml"


def project_dir(project_id: str) -> Path:
    """The folder of the *user* project named `project_id`, under the projects root.

    How a user's run script names its project --
    `SimulationMetaData(project_dir=project_dir("paroninkorpi"), ...)` -- and
    where `init_project` creates a new one. Nothing derives a project's
    folder from its name behind the caller's back: `SimulationMetaData` is
    handed the folder, and example and testing projects, which live in the
    checkout rather than under the projects root, are named from
    `susi.io.utils.repo_root()` instead (ADR 0006).

    `AppSettings()` is constructed per call rather than held at module level
    so that importing this module never touches the filesystem, and so a
    mistyped `SUSI_PROJECTS_ROOT` fails where a path is actually asked for.
    """
    return AppSettings().projects_root / project_id


def inputs_dir_for_project(project_dir: Path) -> Path:
    """Everything the project is simulated *from*."""
    return project_dir / "inputs"


def outputs_dir_for_project(project_dir: Path) -> Path:
    """Everything the project's runs produce."""
    return project_dir / "outputs"


def data_dir_for_project(project_dir: Path) -> Path:
    """
    The preferred home for a project's raw data: weather, XML exports, rasters.

    Preferred, not enforced: nothing requires it, and a script names its own
    data paths and may put them anywhere. It is created with every new
    project, and this is the one place its name is spelled.
    """
    return project_dir / "data"


def allometry_dir_for_project(project_dir: Path) -> Path:
    """
    Where the allometry tools write a project's allometry CSVs.
    """
    return inputs_dir_for_project(project_dir) / "allometry"


def new_growth_allometry_dir_for_project(project_dir: Path) -> Path:
    """
    Where new_growth_allometry.py writes a project's new-growth allometry CSVs.

    A subfolder of the allometry dir rather than a filename prefix beside the
    stand set, so a listing of allometry/ shows the stand tools' complete set
    and the per-species new-growth files apart. This is the one place the
    subfolder is named.
    """
    return allometry_dir_for_project(project_dir) / "new_growth"


def stand_data_path_for_project(project_dir: Path) -> Path:
    """The project's `stand_data.json`, alongside its other inputs."""
    return inputs_dir_for_project(project_dir) / STAND_DATA_FILENAME


def run_dir(project_dir: Path, run_id: str) -> Path:
    """One run of a project. Stand and scenario folders nest below this."""
    return outputs_dir_for_project(project_dir) / run_id

def project_dir_from_run_dir(run_dir: Path)->Path:
    """
    Return the project where a run is located.
    The folder structure is always project/outputs/run,
    so the project directory is two folders higher than the run directory.
    """
    return run_dir.parent.parent


def require_outputs_dir(project_dir: Path) -> Path:
    """
    Return a project's `outputs/`, failing with a message that says what is missing.

    Its callers -- `SimulationMetaData`'s check on its `project_dir` and
    both analysis frontends' run pickers -- all hit the same case: a project
    that exists but has never been run. One message for it, next to the path
    composition it is about.
    """
    outputs_dir = outputs_dir_for_project(project_dir)
    if not outputs_dir.is_dir():
        raise FileNotFoundError(
            f"Project {project_dir.name} has no outputs/ folder (expected at "
            f"{outputs_dir}). A project keeps everything its runs produce "
            "in outputs/; create the folder if nothing has been run for this "
            "project yet."
        )
    return outputs_dir


def require_project_dir(project_dir: Path) -> None:
    """
    Fail with a message that says what is missing.
    """
    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"Project folder not found: {project_dir}. A project is identified "
            "by its folder: check the path you passed, or create the folder "
            "with its inputs/ and outputs/ (init_project does both)."
        )

    inputs_dir = inputs_dir_for_project(project_dir)
    if not inputs_dir.is_dir():
        raise FileNotFoundError(
            f"Project {project_dir.name} has no inputs/ folder (expected at "
            f"{inputs_dir}). A project keeps everything it is simulated from "
            "in inputs/ and everything its runs produce in outputs/."
        )

