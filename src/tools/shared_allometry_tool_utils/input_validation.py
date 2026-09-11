"""
CLI-argument and config validation shared by xml_to_allometry.py and
metsakeskus_to_allometry.py: file/directory validator factories for
argparse, the altitude/ddy enforced-range checks, and TOML-config
presence/unknown-field checking.

Per-field type coercion and defaulting for a config file (e.g. `int(raw.get(
"n_trees", 20))`) stays with each tool -- only the shape checking that is
genuinely identical between them lives here.
"""

import argparse
import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Typical value ranges for Finnish forest land, used to warn/block on likely
# mistyped altitude/ddy input.
ALTITUDE_MIN = 0.0  # metres above sea level
ALTITUDE_MAX = 1000.0
DDY_MIN = 500.0  # temperature sum, degree days per year
DDY_MAX = 2000.0


def make_existing_file_validator(suffix: str) -> Callable[[str], Path]:
    """Builds an argparse `type=` callable that accepts an existing file with
    the given suffix (e.g. ".xml", ".gpkg", ".toml") -- replaces several
    near-identical hand-written validators (valid_xml_path, valid_gpkg_path,
    valid_config_path, ...) that only differed in which suffix they checked."""

    def validator(value: str) -> Path:
        path = Path(value)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"File does not exist: {value}")
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"Not a file: {value}")
        if path.suffix.lower() != suffix.lower():
            raise argparse.ArgumentTypeError(f"File must have a {suffix} extension")
        return path

    return validator


def valid_existing_directory(value: str) -> Path:
    """An argparse `type=` callable for a directory that must already exist
    -- used for --project-dir in both tools: the folder the docs have the
    user set up beforehand, with the input file and config.toml colocated
    inside it, not a bare name the tool creates on the fly."""
    if not value.strip():
        raise argparse.ArgumentTypeError("Directory must not be empty")
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Not a directory: {value}")
    return path


def out_of_range_message(
    name: str, value: float, min_value: float, max_value: float
) -> Optional[str]:
    """Return a human-readable message if value falls outside [min_value, max_value],
    or None if it is within range (bounds are inclusive)."""
    if value < min_value or value > max_value:
        return (
            f"{name}={value} is outside the enforced range [{min_value}, {max_value}]"
        )
    return None


def validate_altitude_ddy(
    parser: argparse.ArgumentParser,
    altitude: float,
    ddy: float,
    allow_out_of_range_values: bool,
) -> None:
    """The NaN-check-then-range-check-then-warn/block flow both tools apply
    to their altitude/ddy values (CLI flags for xml_to_allometry.py,
    config-file fields for metsakeskus_to_allometry.py -- identical either
    way once the two floats are in hand). Calls parser.error() (which prints
    usage and raises SystemExit) on a violation that isn't allowed through."""

    # NaN is not a physically meaningful altitude/DDY value under any
    # circumstances (unlike an out-of-range-but-real number), so it is
    # rejected outright -- allow_out_of_range_values does not apply.
    nan_names = [
        name
        for name, value in (("altitude", altitude), ("ddy", ddy))
        if math.isnan(value)
    ]
    if nan_names:
        parser.error(f"{', '.join(nan_names)} must be a real number, not NaN.")

    # Collect every out-of-range violation before reporting, so the user
    # learns about all of them in one run instead of fixing them one at a
    # time across repeated invocations.
    out_of_range_messages = [
        message
        for message in (
            out_of_range_message(name, value, min_value, max_value)
            for name, value, min_value, max_value in (
                ("altitude", altitude, ALTITUDE_MIN, ALTITUDE_MAX),
                ("ddy", ddy, DDY_MIN, DDY_MAX),
            )
        )
        if message is not None
    ]

    if out_of_range_messages:
        if allow_out_of_range_values:
            for message in out_of_range_messages:
                print(
                    f"Warning: {message}; proceeding due to --allow-out-of-range-values"
                )
        else:
            parser.error(
                "; ".join(out_of_range_messages)
                + ". Pass --allow-out-of-range-values to override."
            )


def check_config_fields(
    raw: dict, config_dataclass: type, required_fields: tuple[str, ...]
) -> None:
    """Presence/unknown-field checking for a TOML-derived dict, parameterized
    by a config dataclass type (whose field names are the allowed set) and a
    tuple of required-field names. Raises ValueError, collecting every
    violation of one kind before reporting, so the user learns about all
    missing fields (or all unknown ones) in a single run."""
    missing = [name for name in required_fields if name not in raw]
    if missing:
        raise ValueError(
            f"Config file is missing required field(s): {', '.join(missing)}"
        )

    known_fields = {f.name for f in dataclasses.fields(config_dataclass)}
    unexpected = sorted(set(raw) - known_fields)
    if unexpected:
        raise ValueError(f"Config file has unknown field(s): {', '.join(unexpected)}")


def load_toml_config(config_path: Path, parse: Callable[[dict], T]) -> T:
    """Reads a TOML config file and hands the raw dict to `parse` (e.g.
    xml_to_allometry.py's parse_xml_config or metsakeskus_to_allometry.py's
    parse_extraction_config) -- the file-reading boilerplate both tools'
    config loaders were otherwise duplicating byte-for-byte."""
    with open(config_path, "rb") as config_file:
        raw = tomllib.load(config_file)
    return parse(raw)
