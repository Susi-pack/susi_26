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

from pydantic import ValidationError, field_validator

from susi.io.extra_pydantic_types import PositiveInt
from susi.io.load_output_data import StandID
from susi.io.project_layout import allometry_dir_for_project
from tools.shared_allometry_tool_utils.allometry_generation_defaults import (
    AllometryGenerationDefaults,
)
from tools.shared_allometry_tool_utils.growth_and_yield_table import (
    build_growth_and_yield_table as build_shared_growth_and_yield_table,
)
from tools.shared_allometry_tool_utils.input_validation import (
    load_toml_config,
    make_existing_file_validator,
    valid_existing_directory,
    validate_x_y_ykj,
)
from tools.shared_allometry_tool_utils.print_formatting import print_section
from tools.shared_allometry_tool_utils.cli_paths import (
    finalize_cli_config,
    resolve_config_path,
)
from tools.shared_allometry_tool_utils.shared_utils import point_to_ykj
from tools.shared_allometry_tool_utils.stand_data import (
    StandDataDocument,
    load_stand_data_document_from_json,
)
from tools.shared_allometry_tool_utils.tree_stratum import (
    PerSpecies,
    TreeStratum,
    ZERO_STRATUM,
)

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

# What --config falls back to inside the project's inputs/ folder.
# Tool-specific on purpose: a project can hold several tools' configs at
# once, so a bare "config.toml" would collide with xml_to_allometry.py's and
# metsakeskus_to_allometry.py's.
DEFAULT_CONFIG_FILENAME = "new_growth_config.toml"


# %% Config


class NewGrowthSourcedConfig(AllometryGenerationDefaults):
    """
    The config for a *sourced-mode* run (--stand-data/--stand-id): only the
    parameters that a StandDataDocument cannot supply.

    Everything a sourced run still has to be told by hand lives here --
    which species is regenerating, at what density, and how far forward to
    project it. The five site facts a sourced run reads from the document
    instead (altitude, ddy, fertility_class, x, y) are declared on
    NewGrowthConfig below, not here, so StrictFrozenModel's (via
    AllometryGenerationDefaults) extra="forbid" rejects a sourced-mode
    config that still hand-types one of them -- silently ignoring a stale
    `altitude = 100` while actually using the document's value would be the
    worst of both worlds.

    Subclasses AllometryGenerationDefaults for the shared n_trees/
    start_year/end_year/step_years defaults, the same ones
    xml_to_allometry.py's XmlConfig and metsakeskus_to_allometry.py's
    ExtractionConfig use.
    """

    # Required, no defaults
    species: Species  # the only species growing here
    stems_count: PositiveInt  # units: stems/ha

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


class NewGrowthConfig(NewGrowthSourcedConfig):
    """
    The config for a *standalone* run (no --stand-data/--stand-id): the
    sourced-mode config plus the five site facts that, with no stand-data
    document to read them from, have to be typed out by hand.

    A standalone config is a sourced config with more in it, which is why
    this subclasses rather than repeats it -- the species/stems_count/
    projection fields and their validation are identical in both modes.
    """

    # Required, no defaults
    altitude: float
    ddy: float
    fertility_class: int
    x: float  # ETRS-TM35FIN (EPSG:3067) metres, converted to YKJ before use
    y: float  # ETRS-TM35FIN (EPSG:3067) metres, converted to YKJ before use


def load_new_growth_config(config_path: Path) -> NewGrowthConfig:
    return load_toml_config(config_path, NewGrowthConfig.model_validate)


def load_new_growth_sourced_config(config_path: Path) -> NewGrowthSourcedConfig:
    return load_toml_config(config_path, NewGrowthSourcedConfig.model_validate)


# %% Site inputs -- the five values the two modes resolve differently


@dataclass(frozen=True)
class SiteInputs:
    """
    The site facts Growth_and_Yield_Table needs about *where* the stand is,
    once whichever mode is in play has resolved them: hand-typed in the
    config (standalone) or read off a StandDataDocument (sourced).

    Coordinates are YKJ grid units here, never ETRS-TM35FIN metres -- each
    mode does whatever conversion it needs before building one of these, so
    everything downstream is conversion-agnostic.
    """

    altitude: float
    ddy: float
    # site_fertility_class, not fertility_class: CONTEXT.md's glossary puts
    # the shorter name on its Avoid list (it is this field's pre-landing
    # name). The config *key* stays `fertility_class` -- that one is
    # pre-existing and user-facing -- but nothing new is named after it.
    site_fertility_class: int
    x_ykj: int
    y_ykj: int


def resolve_standalone_site_inputs(config: NewGrowthConfig) -> SiteInputs:
    """Standalone mode: every value comes from the config file, with the
    config's ETRS-TM35FIN (EPSG:3067) x/y converted to the YKJ grid units
    Growth_and_Yield_Table expects."""
    x_ykj, y_ykj = point_to_ykj(config.x, config.y)
    return SiteInputs(
        altitude=config.altitude,
        ddy=config.ddy,
        site_fertility_class=config.fertility_class,
        x_ykj=x_ykj,
        y_ykj=y_ykj,
    )


def resolve_sourced_site_inputs(
    stand_data_document: StandDataDocument, stand_id: StandID
) -> SiteInputs:
    """Sourced mode: altitude/ddy come from the document's root (they are
    project-global, one value for every stand in it), the rest from the one
    named stand.

    No point_to_ykj call here, unlike standalone mode: StandData.x_ykj/
    .y_ykj are already YKJ grid units, converted once by whichever tool
    generated the document. Nor is there any build_stand_params/`n`
    involved -- site_fertility_class is a plain StandData field, and a
    number of soil columns has nothing to do with an allometry table.

    Raises KeyError if stand_id is not in the document; the CLI turns that
    into a parser.error listing what the document does contain.
    """
    stand_data = stand_data_document.stands[stand_id]
    return SiteInputs(
        altitude=stand_data_document.altitude,
        ddy=stand_data_document.ddy,
        site_fertility_class=stand_data.site_fertility_class,
        x_ykj=stand_data.x_ykj,
        y_ykj=stand_data.y_ykj,
    )


# %% CLI args


@dataclass(frozen=True)
class StandaloneOrigin:
    """Where a standalone run's SiteInputs came from: the config file
    itself. Carries the raw ETRS-TM35FIN pair purely so the Reading section
    can print it next to the YKJ values it was converted into."""

    x: float
    y: float


@dataclass(frozen=True)
class SourcedOrigin:
    """Where a sourced run's SiteInputs came from: one named stand of one
    stand-data document. The two never travel apart -- the CLI rejects
    either flag without the other -- so they live in one object rather than
    as two separately-nullable fields."""

    stand_data_path: Path
    stand_id: StandID


# Which mode a run is in, asked once. Everything downstream branches on this
# single value instead of re-deriving the mode from a null check here and an
# isinstance there.
SiteInputsOrigin = StandaloneOrigin | SourcedOrigin


@dataclass(frozen=True)
class CLIArguments:
    # NewGrowthSourcedConfig, not NewGrowthConfig: a standalone run's config
    # is one of these too (NewGrowthConfig subclasses it), and nothing past
    # this point reads the five standalone-only fields -- site_inputs below
    # already holds their resolved values.
    config: NewGrowthSourcedConfig
    config_path: Path
    site_inputs: SiteInputs
    origin: SiteInputsOrigin
    project_dir: Path
    allow_out_of_range_values: bool
    dry_run: bool


# %% Functions


def build_strata(species: Species, stems_count: int) -> PerSpecies[TreeStratum]:
    """
    Places the one configured species into its own PerSpecies slot
    (pine -> pine, spruce -> spruce, birch -> deciduous), leaving the other
    two at ZERO_STRATUM.
    """
    stratum = TreeStratum(
        age=AGE,
        basal_area=BASAL_AREA,
        stem_count=stems_count,
        mean_diameter=MEAN_DIAMETER,
        mean_height=STARTING_HEIGHT[species],
    )
    return PerSpecies(
        pine=stratum if species == Species.PINE else ZERO_STRATUM,
        spruce=stratum if species == Species.SPRUCE else ZERO_STRATUM,
        deciduous=stratum if species == Species.BIRCH else ZERO_STRATUM,
    )


def plan_output(config: NewGrowthSourcedConfig, output_dir: Path) -> Path:
    """
    Where the single output CSV would land.
    Knowable before any growth table is computed, so both a dry run and the realrun name it the same way.
    """
    return output_dir / f"{OUTPUT_FILENAME_PREFIX}{config.species.value}.csv"


def build_growth_and_yield_table(
    config: NewGrowthSourcedConfig, site_inputs: SiteInputs
) -> pd.DataFrame:
    """The one growth table this tool produces. Mode-agnostic by
    construction: it sees the species/density/projection settings from the
    config and the already-resolved SiteInputs, never the raw config fields
    a standalone run would have had to be told. Delegates the actual
    Growth_and_Yield_Table construction to the shared helper, passing the
    full (never-isolated) PerSpecies straight through -- there is only ever
    one live species here, the other two already ZERO_STRATUM."""
    strata = build_strata(config.species, config.stems_count)
    return build_shared_growth_and_yield_table(
        strata=strata,
        fertility_class=site_inputs.site_fertility_class,
        x_ykj=site_inputs.x_ykj,
        y_ykj=site_inputs.y_ykj,
        altitude=site_inputs.altitude,
        ddy=site_inputs.ddy,
        n_trees=config.n_trees,
        start_year=config.start_year,
        end_year=config.end_year,
        step_years=config.step_years,
        peat=1,
    )


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
            f"Path to the TOML config file. Defaults to "
            f"{DEFAULT_CONFIG_FILENAME} inside the project's inputs/ folder."
        ),
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=valid_existing_directory,
        help=(
            "Path to the project's folder -- the project root, the folder "
            "holding its inputs/ and outputs/. Decides the output directory, "
            "<project-dir>/inputs/allometry/, and -- unless --config is "
            "given -- where the config file is looked up: "
            f"<project-dir>/inputs/{DEFAULT_CONFIG_FILENAME}."
        ),
    )
    parser.add_argument(
        "--stand-data",
        type=make_existing_file_validator(".json"),
        default=None,
        help=(
            "Path to a project's stand_data.json -- normally "
            "<project-dir>/inputs/stand_data.json, as written there by "
            "xml_to_allometry.py/metsakeskus_to_allometry.py, though any "
            "project's document may be pointed at. Selects sourced mode: "
            "altitude, ddy, fertility_class and the YKJ coordinates are read "
            "from the document instead of the config file. Must be given "
            "together with --stand-id."
        ),
    )
    parser.add_argument(
        "--stand-id",
        default=None,
        help=(
            "Which stand of --stand-data to read the site values from. Must be "
            "given together with --stand-data."
        ),
    )
    parser.add_argument(
        "--allow-out-of-range-values",
        action="store_true",
        help=(
            "Allow altitude/ddy and the YKJ x/y coordinates outside their "
            "enforced ranges instead of blocking. Out-of-range values are "
            "still printed as a warning."
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

    # --stand-data and --stand-id are meaningless apart: a document with no
    # stand named, or a stand name with no document to look it up in. Reject
    # either alone rather than half-entering sourced mode.
    if (args.stand_data is None) != (args.stand_id is None):
        parser.error(
            "--stand-data and --stand-id must be given together: pass both to "
            "run in sourced mode, or neither to run in standalone mode."
        )

    # --config defaults to DEFAULT_CONFIG_FILENAME inside the project's
    # inputs/ folder -- the layout the docs have the user set up beforehand.
    config_path = resolve_config_path(
        args.config, args.project_dir, parser, DEFAULT_CONFIG_FILENAME
    )

    config: NewGrowthSourcedConfig
    origin: SiteInputsOrigin
    if args.stand_data is None:
        standalone_config = load_new_growth_config(config_path)
        config = standalone_config
        site_inputs = resolve_standalone_site_inputs(standalone_config)
        origin = StandaloneOrigin(x=standalone_config.x, y=standalone_config.y)
    else:
        config = load_new_growth_sourced_config(config_path)
        origin = SourcedOrigin(
            stand_data_path=args.stand_data, stand_id=StandID(args.stand_id)
        )
        # A malformed or out-of-bounds document is a bad input like any
        # other, so it exits through parser.error like every other bad input
        # these tools take -- not as a raw Pydantic traceback. Note that
        # StandData's own x_ykj/y_ykj bounds are narrower than
        # input_validation's X/Y_YKJ_MIN/MAX, so a wildly wrong coordinate
        # is caught here rather than by validate_x_y_ykj below, and
        # --allow-out-of-range-values cannot wave it through.
        try:
            stand_data_document = load_stand_data_document_from_json(args.stand_data)
        except ValidationError as invalid_document:
            parser.error(f"Could not read {args.stand_data}: {invalid_document}")
        try:
            site_inputs = resolve_sourced_site_inputs(
                stand_data_document=stand_data_document, stand_id=origin.stand_id
            )
        except KeyError:
            # Name what the document does hold: a mistyped --stand-id is far
            # likelier than a genuinely absent stand, and the user has no
            # other way to see the document's keys from here.
            parser.error(
                f"--stand-id {args.stand_id!r} is not in {args.stand_data}. "
                f"That document holds: {sorted(stand_data_document.stands)}"
            )

    # Applied to the resolved values, not to the config file's own fields, so
    # both modes are held to exactly the same ranges. validate_x_y_ykj has no
    # equivalent in the other two tools (they never resolve a coordinate),
    # so it stays here rather than moving into finalize_cli_config, which
    # covers the altitude/ddy-then-output-dir tail every tool does share.
    # Called after finalize_cli_config (not before) so that when both an
    # altitude/ddy value and a coordinate are out of range at once,
    # altitude/ddy is still reported first, same as before this tail moved
    # into the shared helper.
    finalize_cli_config(
        parser,
        site_inputs.altitude,
        site_inputs.ddy,
        args.project_dir,
        args.allow_out_of_range_values,
    )
    validate_x_y_ykj(
        parser, site_inputs.x_ykj, site_inputs.y_ykj, args.allow_out_of_range_values
    )

    return CLIArguments(
        config=config,
        config_path=config_path,
        site_inputs=site_inputs,
        origin=origin,
        project_dir=args.project_dir,
        allow_out_of_range_values=args.allow_out_of_range_values,
        dry_run=args.dry_run,
    )


# %% main


def main():
    cli_args = parse_CLI_arguments()
    config = cli_args.config
    output_dir = allometry_dir_for_project(cli_args.project_dir)

    # Resolved during argument parsing, printed here: with no informational
    # JSON dump for this tool (there is no extraction step to audit), the
    # console output is the only place these resolved values are ever shown
    # -- and in sourced mode it is also the only place the user can see
    # which document and stand they came out of.
    site_inputs = cli_args.site_inputs
    starting_height = STARTING_HEIGHT[config.species]

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - project_dir     = {cli_args.project_dir.resolve()}")
    print(f"    - config          = {cli_args.config_path.resolve()}")
    print(f"    - output_dir      = {output_dir.resolve()}")
    print(f"    - species         = {config.species.value}")
    print(f"    - stems_count     = {config.stems_count} stems/ha")
    origin = cli_args.origin
    if isinstance(origin, SourcedOrigin):
        # Sourced mode: name the source of every value that did not come
        # from the config file, so a wrong --stand-id shows up here.
        print(f"    - stand_data      = {origin.stand_data_path.resolve()}")
        print(f"    - stand_id        = {origin.stand_id}")
        print(
            f"    - fertility_class = {site_inputs.site_fertility_class} (from stand)"
        )
        print(f"    - altitude        = {site_inputs.altitude} (from document root)")
        print(f"    - ddy             = {site_inputs.ddy} (from document root)")
        print(
            f"    - x, y (YKJ)      = {site_inputs.x_ykj}, {site_inputs.y_ykj} "
            "(from stand, already YKJ -- no conversion)"
        )
    else:
        print(f"    - fertility_class = {site_inputs.site_fertility_class}")
        print(f"    - altitude        = {site_inputs.altitude}")
        print(f"    - ddy             = {site_inputs.ddy}")
        print(f"    - x, y (input)    = {origin.x}, {origin.y} (ETRS-TM35FIN metres)")
        print(f"    - x, y (YKJ)      = {site_inputs.x_ykj}, {site_inputs.y_ykj}")
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

    table = build_growth_and_yield_table(config, site_inputs)
    table.to_csv(output_path, index=False)

    print(f"New-growth allometry file written: {output_path}")


if __name__ == "__main__":
    main()
