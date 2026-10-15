import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, DirectoryPath, Field

import susi.io.utils as io_utils

# Overrides where `projects/` is looked for.
# This is what makes running on CSC work: CSC wants data on a different filesystem from the code.
PROJECTS_ROOT_ENV_VAR = "SUSI_PROJECTS_ROOT"


def _default_projects_root() -> Path:
    """
    `$SUSI_PROJECTS_ROOT` when set, else `<repo root>/projects` in a checkout.

    Fails here, naming the env var, when it is set but wrong, or when it is
    unset outside a checkout.
    """
    env_value = os.environ.get(PROJECTS_ROOT_ENV_VAR)
    if env_value is None:
        repo_root = io_utils.repo_root()
        if io_utils.is_checkout(repo_root):
            return repo_root / "projects"
        # An installed copy has no `<repo root>/projects` to fall back to.
        raise ValueError(
            f"{PROJECTS_ROOT_ENV_VAR} is not set. SUSI is not running from a "
            "checkout, so there is no default projects root: set "
            f"{PROJECTS_ROOT_ENV_VAR} to the folder holding your project "
            "folders."
        )

    projects_root = Path(env_value)
    if not projects_root.is_dir():
        raise ValueError(
            f"{PROJECTS_ROOT_ENV_VAR} is set to {env_value!r}, which is not an "
            "existing directory. Point it at the folder holding your project "
            f"folders, or, in a SUSI checkout, unset {PROJECTS_ROOT_ENV_VAR} "
            "to fall back to <repo root>/projects."
        )
    return projects_root


class AppSettings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    projects_root: DirectoryPath = Field(
        default_factory=_default_projects_root,
        description=(
            "Root folder holding one folder per project, each with its own "
            "inputs/ and outputs/. Set with the "
            f"{PROJECTS_ROOT_ENV_VAR} environment variable; in a SUSI "
            "checkout it defaults to <repo root>/projects."
        ),
    )
