import json
import random
import string
import subprocess
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root")


def datetime_format() -> str:
    return "%Y-%m-%d_%H:%M:%S"


def generate_current_datetime_stamp() -> str:
    return datetime.now().strftime(datetime_format())


def random_id_generator(size) -> str:
    characters = string.ascii_uppercase + string.digits
    return "".join(random.choice(characters) for _ in range(size))


def generate_experiment_ID(datetime_stamp: str) -> str:
    return datetime_stamp + "_" + random_id_generator(size=8)


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
