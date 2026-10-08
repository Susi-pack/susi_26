import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field

import susi.io.utils as io_utils

# Overrides where `projects/` is looked for.
# This is what makes running on CSC work: CSC wants data on a different filesystem from the code.
PROJECTS_ROOT_ENV_VAR = "SUSI_PROJECTS_ROOT"


def _default_projects_root() -> Path:
    """
    `$SUSI_PROJECTS_ROOT` when set, else `<repo root>/projects`.

    A set-but-wrong env var fails here.
    """
    env_value = os.environ.get(PROJECTS_ROOT_ENV_VAR)
    if env_value is None:
        return io_utils.repo_root() / "projects"

    projects_root = Path(env_value)
    if not projects_root.is_dir():
        raise ValueError(
            f"{PROJECTS_ROOT_ENV_VAR} is set to {env_value!r}, which is not an "
            "existing directory. Point it at the folder holding your project "
            f"folders, or unset {PROJECTS_ROOT_ENV_VAR} to fall back to "
            "<repo root>/projects."
        )
    return projects_root


class AppSettings(BaseModel):
    # Pydantic does not validate defaults unless asked to, so without this a
    # nonexistent default would be accepted silently despite the
    # DirectoryPath annotation. Safe to switch on because `projects/.gitkeep`
    # is tracked: the default always exists on a fresh clone, so this check
    # fires only when someone pointed the settings somewhere wrong.
    model_config = ConfigDict(validate_default=True)

    projects_root: DirectoryPath = Field(
        default_factory=_default_projects_root,
        description=(
            "Root folder holding one folder per project, each with its own "
            "inputs/ and outputs/. Defaults to <repo root>/projects, "
            f"overridable with the {PROJECTS_ROOT_ENV_VAR} environment "
            "variable."
        ),
    )
