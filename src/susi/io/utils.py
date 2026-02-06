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
        path.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"A folder with the same path, i.e., {path}, already exists."
        )


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def read_json_file(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
