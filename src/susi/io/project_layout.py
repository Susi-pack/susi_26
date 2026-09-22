"""Where a project's files live.

One folder per project under `AppSettings.projects_root`, each holding its
own `inputs/` and `outputs/`, so a project's data and its results are
co-located by construction:

    projects/
    └── paroninkorpi/
        ├── inputs/
        │   ├── allometry/
        │   ├── stand_data.json
        │   ├── config.toml
        │   └── other: weather.csv, export.xml, ditch_depth.tif, ...
        └── outputs/
            └── <run_id>/<stand_id>/<scenario_id>/

Everything here is pure path composition -- no argparse, no I/O -- except
`require_project_dir`, which exists precisely to hit the filesystem. The
argparse-flavoured helpers that wrap these paths live in
`tools/shared_allometry_tool_utils/project_layout.py` instead.
"""

from pathlib import Path

from susi.io.app_settings import PROJECTS_ROOT_ENV_VAR, AppSettings

# The per-project document describing every stand the project simulates.
STAND_DATA_FILENAME = "stand_data.json"


def project_dir(project_id: str) -> Path:
    """The folder holding everything about one project.

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


def allometry_dir_for_project(project_dir: Path) -> Path:
    """
    Where the allometry tools write a project's allometry CSVs.
    """
    return inputs_dir_for_project(project_dir) / "allometry"


def stand_data_path_for_project(project_dir: Path) -> Path:
    """The project's `stand_data.json`, alongside its other inputs."""
    return inputs_dir_for_project(project_dir) / STAND_DATA_FILENAME


def run_dir(project_dir: Path, run_id: str) -> Path:
    """One run of a project. Stand and scenario folders nest below this."""
    return outputs_dir_for_project(project_dir) / run_id


def require_project_dir(project_dir: Path) -> None:
    """
    Fail with a message that says what is missing.
    """
    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"Project folder not found: {project_dir}. Projects are looked up "
            f"under the projects root, which {PROJECTS_ROOT_ENV_VAR} overrides "
            "when set. Check the project name, or create the folder with its "
            "inputs/ and outputs/."
        )

    inputs_dir = inputs_dir_for_project(project_dir)
    if not inputs_dir.is_dir():
        raise FileNotFoundError(
            f"Project {project_dir.name} has no inputs/ folder (expected at "
            f"{inputs_dir}). A project keeps everything it is simulated from "
            "in inputs/ and everything its runs produce in outputs/."
        )
