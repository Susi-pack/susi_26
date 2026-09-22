import json
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root")


def create_folder(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"A folder with the same path, i.e., {path}, already exists."
        )


def get_git_revision_short_hash() -> str:
    """The SUSI checkout's current commit, for `SimulationMetaData`'s
    provenance record.

    `cwd` is the checkout, not the process's working directory. Without it
    git answers about whatever repository it finds by searching upward from
    wherever the run was started -- so a run launched from inside a project
    folder that is its own git repository recorded *that* repository's
    commit, and a run launched from a project outside any repository (what
    SUSI_PROJECTS_ROOT is for) crashed before the simulation began.
    `get_project_root()` walks up from this file, so it names the code that
    is actually running whatever the caller's cwd is."""
    return (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=get_project_root()
        )
        .decode("ascii")
        .strip()
    )


def read_json_file(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
