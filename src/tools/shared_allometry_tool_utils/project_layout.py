"""The argparse-flavoured half of the `--project-dir` convention shared by
xml_to_allometry.py, metsakeskus_to_allometry.py and new_growth_allometry.py:
the refuse-if-it-already-exists check on the output folder, and the
`--config` defaults-to-`<project-dir>/inputs/<name>` resolution.

`--project-dir` is the project root -- `projects/<project>/` -- so every
path under it is composed by `susi.io.project_layout`, which owns the
layout. Nothing here joins `"inputs"` or `"allometry"` onto a path itself.
"""

import argparse
from pathlib import Path
from typing import Optional

from susi.io.project_layout import inputs_dir_for_project


def check_output_dir_available(
    output_dir: Path, parser: argparse.ArgumentParser
) -> None:
    """Refuse to reuse an existing folder rather than silently overwriting
    (or, previously, deleting) whatever a prior run left there -- the user
    must pick a different --project-dir instead. Meant to be called during
    argument parsing, including in --dry-run mode: "would this run even
    start?" is exactly what a dry run is for. Creating the folder itself is
    each tool's main()'s job, and only on a real run."""
    if output_dir.exists():
        parser.error(
            f"Output folder already exists: {output_dir}. Refusing to run into "
            "an existing folder. Pass a different --project-dir instead."
        )


def resolve_config_path(
    explicit_config: Optional[Path],
    project_dir: Path,
    parser: argparse.ArgumentParser,
    config_filename: str,
) -> Path:
    """--config defaults to `config_filename` inside the project's inputs/
    folder -- a tool's config is something the project is run *from*, so it
    lives with the project's other inputs. An explicit --config always wins
    over that default lookup.

    Each tool passes its own filename rather than sharing one default:
    several tools' configs can sit in the same project's inputs/ folder, so
    a single `config.toml` would collide. Required, not defaulted, so no
    tool picks up a neighbour's filename by accident."""
    if explicit_config is not None:
        return explicit_config

    config_path = inputs_dir_for_project(project_dir) / config_filename
    if not config_path.exists() or not config_path.is_file():
        parser.error(
            f"No config file found at the default location: {config_path}. "
            "Pass --config to use a different name or location."
        )
    if config_path.suffix.lower() != ".toml":
        parser.error(f"Default config path is not a .toml file: {config_path}")
    return config_path
