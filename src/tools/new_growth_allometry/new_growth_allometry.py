# Generates the allometry file for a new growth stand, i.e.,
# the age-1 stand that starts growing immediately after a clear cut,
#
# Started by Mikko Niemi, finished by Iñaki Urzainki.

# %% Imports
import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from enum import Enum

from pydantic import field_validator

from susi.core.allometric_road_map import Growth_and_Yield_Table
from susi.io.extra_pydantic_types import PositiveInt, StrictFrozenModel
from tools.shared_allometry_tool_utils.input_validation import (
    load_toml_config,
    make_existing_file_validator,
    valid_existing_directory,
    validate_altitude_ddy,
)
from tools.shared_allometry_tool_utils.print_formatting import print_section
from tools.shared_allometry_tool_utils.project_layout import (
    check_output_dir_available,
    output_dir_for_project,
    resolve_config_path,
)
from tools.shared_allometry_tool_utils.shared_utils import point_to_ykj
from tools.shared_allometry_tool_utils.tree_stratum import TreeStratum, ZERO_STRATUM

# %% Constants -- hard-coded, non-negotiable


# A freshly regenerated stand is, by definition, age 1 with no diameter or basal area yet.
AGE = 1
BASAL_AREA = 0.0
MEAN_DIAMETER = 0.0


class Species(Enum):
    PINE = "pine"
    SPRUCE = "spruce"
    BIRCH = "birch"


SPECIES_CODE = {
    Species.PINE: 1,
    Species.SPRUCE: 2,
    Species.BIRCH: 3,
}

# Starting height (m) immediately after a clear cut (Mikko Niemi's magic numbers).
# This is a genuine model parameter.
# Leave alone unless you know what you are doing.
STARTING_HEIGHT = {
    Species.PINE: 0.1,
    Species.SPRUCE: 0.2,
    Species.BIRCH: 0.2,
}

OUTPUT_FILENAME_PREFIX = "new_growth_"


# %% Config


class NewGrowthConfig(StrictFrozenModel):
    """
    Defaulted/required parameters, loaded from a TOML file.
    """

    # Required, no defaults
    altitude: float
    ddy: float
    fertility_class: int
    x: float  # ETRS-TM35FIN (EPSG:3067) metres, converted to YKJ before use
    y: float  # ETRS-TM35FIN (EPSG:3067) metres, converted to YKJ before use
    species: Species  # the only species growing here
    stems_count: PositiveInt  # units: stems/ha

    # Defaulted -- the same values xml_to_allometry.py's XmlConfig already uses.
    n_trees: int = 20
    start_year: int = 5
    end_year: int = 80
    step_years: int = 5

    @field_validator("species", mode="before")
    @classmethod
    def _normalize_species(cls, raw_species: object) -> Species:
        """
        Lets the config file spell "PINE" or " pine "
        """
        if isinstance(raw_species, Species):
            return raw_species
        valid_values = [s.value for s in Species]
        if not isinstance(raw_species, str):
            raise ValueError(
                f"species must be a string, one of {valid_values}; got {raw_species!r}"
            )
        try:
            return Species(raw_species.strip().lower())
        except ValueError:
            raise ValueError(
                f"species must be one of {valid_values}; got {raw_species!r}"
            )


def load_new_growth_config(config_path: Path) -> NewGrowthConfig:
    return load_toml_config(config_path, NewGrowthConfig.model_validate)


# %% CLI args


@dataclass(frozen=True)
class CLIArguments:
    config: NewGrowthConfig
    config_path: Path
    project_dir: Path
    allow_out_of_range_values: bool
    dry_run: bool


# %% Functions


def build_strata(
    species: Species, stems_count: int
) -> tuple[TreeStratum, TreeStratum, TreeStratum]:
    """
    Places the one configured species into its own fixed slot
    (pine -> index 0, spruce -> index 1, birch -> index 2
    """
    stratum = TreeStratum(
        age=AGE,
        basal_area=BASAL_AREA,
        stem_count=stems_count,
        mean_diameter=MEAN_DIAMETER,
        mean_height=STARTING_HEIGHT[species],
    )
    strata = [ZERO_STRATUM, ZERO_STRATUM, ZERO_STRATUM]
    slot = {Species.PINE: 0, Species.SPRUCE: 1, Species.BIRCH: 2}[species]
    strata[slot] = stratum
    return (strata[0], strata[1], strata[2])


def plan_output(config: NewGrowthConfig, output_dir: Path) -> Path:
    """
    Where the single output CSV would land.
    Knowable before any growth table is computed, so both a dry run and the realrun name it the same way.
    """
    return output_dir / f"{OUTPUT_FILENAME_PREFIX}{config.species.value}.csv"


def build_growth_and_yield_table(
    config: NewGrowthConfig, x_ykj: int, y_ykj: int
) -> pd.DataFrame:
    strata = build_strata(config.species, config.stems_count)

    gy = Growth_and_Yield_Table(
        age_1=strata[0].age,
        G_1=strata[0].basal_area,
        N_1=strata[0].stem_count,
        Dg_1=strata[0].mean_diameter,
        Hg_1=strata[0].mean_height,
        age_2=strata[1].age,
        G_2=strata[1].basal_area,
        N_2=strata[1].stem_count,
        Dg_2=strata[1].mean_diameter,
        Hg_2=strata[1].mean_height,
        age_3=strata[2].age,
        G_3=strata[2].basal_area,
        N_3=strata[2].stem_count,
        Dg_3=strata[2].mean_diameter,
        Hg_3=strata[2].mean_height,
        DDY=config.ddy,
        fertility_class=config.fertility_class,
        peat=1,
        y=y_ykj,
        x=x_ykj,
        altitude=config.altitude,
        n_trees=config.n_trees,
    )
    table = gy.get_table(
        start_year=config.start_year,
        end_year=config.end_year,
        step_years=config.step_years,
    )
    return table


# %% CLI


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the new-growth allometry file for one freshly "
            "regenerated (post-clearcut) stand"
        )
    )
    parser.add_argument(
        "--config",
        type=make_existing_file_validator(".toml"),
        default=None,
        help=(
            "Path to the TOML config file. Defaults to config.toml directly "
            "inside --project-dir."
        ),
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=valid_existing_directory,
        help=(
            "Path to the project's folder. Decides the output directory, "
            "<project-dir>/allometry/, and -- unless --config is given -- "
            "where the config file is looked up: <project-dir>/config.toml."
        ),
    )
    parser.add_argument(
        "--allow-out-of-range-values",
        action="store_true",
        help=(
            "Allow the config file's altitude/ddy values outside their enforced range "
            "instead of blocking. Out-of-range values are still printed as a warning."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what the run would produce, then exit having written "
            "nothing at all: no CSV, not even the output folder."
        ),
    )

    args = parser.parse_args()

    # --config defaults to config.toml directly inside --project-dir -- the
    # layout the docs have the user set up beforehand.
    config_path = resolve_config_path(args.config, args.project_dir, parser)
    config = load_new_growth_config(config_path)

    validate_altitude_ddy(
        parser, config.altitude, config.ddy, args.allow_out_of_range_values
    )

    # Refuse to reuse an existing folder rather than silently overwriting
    # whatever a prior run left there. This check runs in dry-run mode too:
    # "would this run even start?" is exactly what a dry run is for.
    # Creating the folder is main()'s job, and only on a real run.
    output_dir = output_dir_for_project(args.project_dir)
    check_output_dir_available(output_dir, parser)

    return CLIArguments(
        config=config,
        config_path=config_path,
        project_dir=args.project_dir,
        allow_out_of_range_values=args.allow_out_of_range_values,
        dry_run=args.dry_run,
    )


# %% main


def main():
    cli_args = parse_CLI_arguments()
    config = cli_args.config
    output_dir = output_dir_for_project(cli_args.project_dir)

    # Computed once here (not inside build_growth_and_yield_table) so the
    # Reading section below can print them: with no informational JSON dump
    # for this tool (there is no extraction step to audit -- the config
    # already is the full input), the console output is the only place these
    # resolved values are ever shown.
    x_ykj, y_ykj = point_to_ykj(config.x, config.y)
    starting_height = STARTING_HEIGHT[config.species]

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - project_dir     = {cli_args.project_dir.resolve()}")
    print(f"    - config          = {cli_args.config_path.resolve()}")
    print(f"    - output_dir      = {output_dir.resolve()}")
    print(f"    - species         = {config.species.value}")
    print(f"    - stems_count     = {config.stems_count} stems/ha")
    print(f"    - fertility_class = {config.fertility_class}")
    print(f"    - altitude        = {config.altitude}")
    print(f"    - ddy             = {config.ddy}")
    print(f"    - x, y (input)    = {config.x}, {config.y} (ETRS-TM35FIN metres)")
    print(f"    - x, y (YKJ)      = {x_ykj}, {y_ykj}")
    print(f"    - starting_height = {starting_height} m (fixed model parameter)")
    if cli_args.dry_run:
        print("    - DRY RUN -- nothing will be written")
    print()

    output_path = plan_output(config, output_dir)

    # Everything above this point is identical in a dry run: there is no
    # filtering stage for a single hand-specified stand, so Reading is all a
    # dry run has to show before the planned output.
    if cli_args.dry_run:
        print_section("Writing (dry run -- nothing is written)")
        print(f"Destination folder: {output_dir.resolve()} (not created)")
        print()
        print(f"Would write: {output_path}")
        return

    print_section("Writing")
    print(f"Destination folder: {output_dir.resolve()}")
    # Created here rather than at argument-parsing time, so a run that fails
    # while reading leaves no empty folder behind to block the next attempt.
    output_dir.mkdir(parents=True)
    print()

    table = build_growth_and_yield_table(config, x_ykj, y_ykj)
    table.to_csv(output_path, index=False)

    print(f"New-growth allometry file written: {output_path}")


if __name__ == "__main__":
    main()
