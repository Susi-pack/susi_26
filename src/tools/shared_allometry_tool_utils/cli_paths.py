"""The argparse-flavoured half of the `--project-dir` convention shared by
xml_to_allometry.py, metsakeskus_to_allometry.py and new_growth_allometry.py:
the refuse-if-it-already-exists checks on the allometry set (the stand
tools) and the output file (new_growth_allometry.py), the `--config`
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

from susi.io.project_layout import (
    allometry_dir_for_project,
    inputs_dir_for_project,
    stand_data_path_for_project,
)
from tools.shared_allometry_tool_utils.input_validation import validate_altitude_ddy


def check_allometry_set_available(
    output_dir: Path, json_path: Path, parser: argparse.ArgumentParser
) -> None:
    """Refuse to write an allometry set over an existing one, rather than
    silently overwriting (or, previously, deleting) whatever a prior run left
    there. The stand tools write a complete set -- one CSV per stand plus the
    stand-data document referencing them -- and mixing a new set into an old
    one would be wrong. Either half existing is reason enough: a document
    whose folder was moved aside alone would otherwise be overwritten, leaving
    the moved folder referenced by nothing. Meant to be called during
    argument parsing, including in --dry-run mode: "would this run even
    start?" is exactly what a dry run is for. Creating the folder itself is
    each tool's main()'s job, and only on a real run (create_allometry_set_dir).

    The advice is to move the old set aside, not to pass a different
    --project-dir: a project is one folder, so a second
    --project-dir would split it in two."""
    existing = [path for path in (output_dir, json_path) if path.exists()]
    if existing:
        parser.error(
            "Allometry set already exists: "
            + " and ".join(str(path) for path in existing)
            + ". Refusing to write a set over an existing one. To regenerate "
            "it, rename or move what exists first."
        )


def create_allometry_set_dir(output_dir: Path) -> None:
    """Creates a set's allometry folder, at the start of a real run's writing
    -- not at argument-parsing time, so a run that fails while reading or
    filtering leaves no empty folder behind to block the next attempt.

    check_allometry_set_available has already refused an existing folder, so
    what can still go wrong here is the OS refusing it (no write permission
    on inputs/, a read-only file system, ...). That ends the run with the
    folder and the OS's own reason, not a traceback."""
    try:
        output_dir.mkdir(parents=True)
    except OSError as error:
        raise SystemExit(
            f"Could not create the allometry folder {output_dir}: "
            f"{error.strerror or error}."
        ) from error


def check_output_file_available(
    output_path: Path, parser: argparse.ArgumentParser
) -> None:
    """The per-file counterpart of check_allometry_set_available, for
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
    explicit_config: Path | None,
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
    set_name: str | None,
    allow_out_of_range_values: bool,
) -> Path:
    """The validate-then-resolve-output-dir tail the two stand tools'
    parse_CLI_arguments repeat once their own config is loaded:
    validate_altitude_ddy, then refuse an allometry set that already exists
    (either half of it). Returns the validated, available allometry output
    dir of the set `set_name` names -- None for the project's default set.

    Takes altitude/ddy as plain floats rather than a whole config object, so
    it stays independent of the fact that ExtractionConfig and XmlConfig are
    unrelated types.

    new_growth_allometry.py does not use this: it writes one file per species
    into a folder that normally already exists, so it calls
    validate_altitude_ddy itself and refuses only its own output file
    (check_output_file_available)."""
    validate_altitude_ddy(parser, altitude, ddy, allow_out_of_range_values)

    output_dir = allometry_dir_for_project(project_dir, set_name)
    check_allometry_set_available(
        output_dir, stand_data_path_for_project(project_dir, set_name), parser
    )
    return output_dir
