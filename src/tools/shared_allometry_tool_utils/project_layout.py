"""The `--project-dir` / `<project-dir>/allometry` output convention shared
by xml_to_allometry.py and metsakeskus_to_allometry.py: where output goes,
the refuse-if-it-already-exists check, and the `--config` defaults-to-
`<project-dir>/config.toml` resolution.
"""

import argparse
from pathlib import Path
from typing import Optional


def output_dir_for_project(project_dir: Path) -> Path:
    """Where a project's allometry files go. There is no flag to point the
    output somewhere else: appending allometry/ onto --project-dir is an
    invariant of both tools rather than a default."""
    return project_dir / "allometry"


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
) -> Path:
    """--config defaults to config.toml directly inside --project-dir -- the
    layout the docs have the user set up beforehand. An explicit --config
    always wins over that default lookup."""
    if explicit_config is not None:
        return explicit_config

    config_path = project_dir / "config.toml"
    if not config_path.exists() or not config_path.is_file():
        parser.error(
            f"No config file found at the default location: {config_path}. "
            "Pass --config to use a different name or location."
        )
    if config_path.suffix.lower() != ".toml":
        parser.error(f"Default config path is not a .toml file: {config_path}")
    return config_path
