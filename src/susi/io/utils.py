import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

# The SUSI checkout, and its source tree: this file is
# src/susi/io/utils.py, so the repo root is three levels up.
#
# This is the one definition of where the code lives. Anything that needs a
# path *into* the checkout composes it from here rather than counting
# `parents[...]` of its own -- a second anchor is a second thing to fix when
# a file moves, and the two drift silently because neither is wrong on its
# own.
#
# Anchored on `__file__` rather than by searching upwards for pyproject.toml:
# the search has a failure mode (no pyproject.toml above us, so it raised
# FileNotFoundError) and a worse success mode (a stray pyproject.toml in a
# parent directory answers instead). Neither is hypothetical now that
# SUSI_PROJECTS_ROOT exists precisely so a user's data can live outside the
# checkout.
#
# Not an `AppSettings` field: where the running code lives is a fact, not a
# setting, and nothing a user may point elsewhere -- the same reasoning that
# turned `AppSettings.input_folder` into `system_inputs.SYSTEM_INPUTS_DIR`.
SUSI_REPO_ROOT = Path(__file__).resolve().parents[3]

# `src/`: the package itself, plus its siblings `tools/`, `analysis/` and
# `system_inputs/`. Only meaningful in a checkout -- an
# installed copy of the package has no src/, and pyproject.toml installs
# `susi*` and `analysis*` only -- which is fine for its callers, all of
# which are scripts and tools run from the checkout.
SRC_DIR = SUSI_REPO_ROOT / "src"


def repo_root() -> Path:
    """The SUSI checkout's root directory, i.e. `SUSI_REPO_ROOT`.

    A function over the constant, deliberately: it is the seam tests
    redirect at a tmp_path (monkeypatching a module-level constant that
    callers have already read does nothing). Prefer `SUSI_REPO_ROOT` where
    no such redirection is wanted.

    Named for the *repository*, not the "project": a project in this
    codebase is one study's folder under the projects root
    (`project_dir`, `project_id`, `require_project_dir`), which is exactly
    what this is not -- and the old name `get_project_root` is why
    init_project.py grew a `parents[2]` of its own rather than calling it.
    """
    return SUSI_REPO_ROOT


def create_folder(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"A folder with the same path, i.e., {path}, already exists."
        )


GIT_HASH_FALLBACK_PREFIX = "could not get git commit hash. The package version was: "


def get_git_revision_short_hash() -> str:
    """The SUSI checkout's current commit, for `SimulationMetaData`'s
    provenance record.
    When there is no commit to be had, the sentence
    `"could not get git commit hash. The package version was: <version>"`,
    with a one-line warning on stderr.
    """
    checkout = repo_root()
    if not (checkout / "pyproject.toml").exists():
        return _git_hash_fallback(f"{checkout} is not a SUSI checkout")

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=checkout,
                stderr=subprocess.PIPE,
            )
            .decode("ascii")
            .strip()
        )
    except FileNotFoundError:
        return _git_hash_fallback("git is not installed")
    except subprocess.CalledProcessError as error:
        git_says = error.stderr.decode(errors="replace").strip().splitlines()
        return _git_hash_fallback(git_says[0] if git_says else str(error))


def _git_hash_fallback(reason: str) -> str:
    """What `get_git_revision_short_hash` records when git cannot be asked.

    A sentence rather than the bare version, so that the value in
    metadata.json reads as a fallback and not as a commit hash."""
    print(
        f"Warning: could not get git commit hash ({reason}). "
        "Recording the package version instead.",
        file=sys.stderr,
    )
    return GIT_HASH_FALLBACK_PREFIX + importlib.metadata.version("susi-26")


def read_json_file(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
