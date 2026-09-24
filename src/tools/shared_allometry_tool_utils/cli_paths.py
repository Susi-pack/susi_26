"""The argparse-flavoured half of the `--project-dir` convention shared by
xml_to_allometry.py, metsakeskus_to_allometry.py and new_growth_allometry.py:
the refuse-if-it-already-exists checks on the output folder (the stand tools)
and the output file (new_growth_allometry.py), the `--config`
defaults-to-`<project-dir>/inputs/<name>` resolution, and finalize_cli_config,
the validate-then-resolve-output-dir tail the two stand tools'
parse_CLI_arguments share.

Named cli_paths, not project_layout: `--project-dir` is the project root --
`projects/<project>/` -- so every path under it is actually composed by
`susi.io.project_layout`, which owns the layout. Nothing here joins
`"inputs"` or `"allometry"` onto a path itself; this module only wraps that
composition in argparse-flavoured validation.
"""

import argparse
from pathlib import Path
from typing import Optional

from susi.io.project_layout import allometry_dir_for_project, inputs_dir_for_project
from tools.shared_allometry_tool_utils.input_validation import validate_altitude_ddy


def check_output_dir_available(
    output_dir: Path, parser: argparse.ArgumentParser
) -> None:
    """Refuse to reuse an existing folder rather than silently overwriting
    (or, previously, deleting) whatever a prior run left there. The stand
    tools write a complete set -- one CSV per stand plus stand_data.json --
    and mixing a new set into an old one would be wrong. Meant to be called
    during argument parsing, including in --dry-run mode: "would this run
    even start?" is exactly what a dry run is for. Creating the folder
    itself is each tool's main()'s job, and only on a real run.

    The advice is to move the old folder aside, not to pass a different
    --project-dir: a project is one folder (ADR 0003), so a second
    --project-dir would split it in two."""
    if output_dir.exists():
        parser.error(
            f"Output folder already exists: {output_dir}. Refusing to run into "
            "an existing folder. To regenerate it, rename or move the existing "
            "folder first."
        )


def check_output_file_available(
    output_path: Path, parser: argparse.ArgumentParser
) -> None:
    """The per-file counterpart of check_output_dir_available, for
    new_growth_allometry.py: it writes one file per species into a shared
    folder, so only its own target file is a reason to refuse -- the folder
    existing is the normal case (a stand tool made allometry/, an earlier
    species made new_growth/). Same timing as the folder check: during
    argument parsing, dry run included."""
    if output_path.exists():
        parser.error(
            f"Output file already exists: {output_path}. Refusing to overwrite "
            "it. Delete it first to regenerate it."
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


def finalize_cli_config(
    parser: argparse.ArgumentParser,
    altitude: float,
    ddy: float,
    project_dir: Path,
    allow_out_of_range_values: bool,
) -> Path:
    """The validate-then-resolve-output-dir tail the two stand tools'
    parse_CLI_arguments repeat once their own config is loaded:
    validate_altitude_ddy, then refuse an already-existing output folder.
    Returns the validated, available allometry output dir.

    Takes altitude/ddy as plain floats rather than a whole config object, so
    it stays independent of the fact that ExtractionConfig and XmlConfig are
    unrelated types.

    new_growth_allometry.py does not use this: it writes one file per species
    into a folder that normally already exists, so it calls
    validate_altitude_ddy itself and refuses only its own output file
    (check_output_file_available)."""
    validate_altitude_ddy(parser, altitude, ddy, allow_out_of_range_values)

    output_dir = allometry_dir_for_project(project_dir)
    check_output_dir_available(output_dir, parser)
    return output_dir
