# Gets XML stand data to the allometry file format required by SUSI
# Adapted by Iñaki Urzainki from Mikko Niemi's original code.

# %% Imports
from susi.io.load_output_data import StandID
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerName,
)
from typing import Optional
import pandas as pd
import xmltodict
import argparse
from dataclasses import dataclass
from pathlib import Path
from susi.core.allometric_road_map import Growth_and_Yield_Table
from susi.io.extra_pydantic_types import StrictFrozenModel
from tools.shared_allometry_tool_utils.input_validation import (
    load_toml_config,
    make_existing_file_validator,
    valid_existing_directory,
    validate_altitude_ddy,
)
from tools.shared_allometry_tool_utils.print_formatting import (
    StandSkipped,
    print_section,
    print_skips,
)
from tools.shared_allometry_tool_utils.project_layout import (
    check_output_dir_available,
    output_dir_for_project,
    resolve_config_path,
)
from tools.shared_allometry_tool_utils.shared_utils import point_to_ykj
from tools.shared_allometry_tool_utils.tree_stratum import TreeStratum, ZERO_STRATUM
from tools.shared_allometry_tool_utils.stand_data import (
    STAND_DATA_FILENAME,
    StandData,
    StandDataDocument,
)


# %% Config


class XmlConfig(StrictFrozenModel):
    """Defaulted/required parameters, loaded from a TOML file.

    StrictFrozenModel (susi.io.extra_pydantic_types) gives us presence
    checking for the required fields below (no default -> required),
    rejection of unknown fields (extra="forbid"), and immutability
    (frozen=True) for free -- replacing check_config_fields,
    REQUIRED_CONFIG_FIELDS, and the dataclass's own frozen=True."""

    # Required, no defaults
    altitude: float
    ddy: float

    # Defaulted -- the same values metsakeskus_to_allometry.py's
    # ExtractionConfig already uses.
    n_trees: int = 20
    start_year: int = 5
    end_year: int = 80
    step_years: int = 5


def load_xml_config(config_path: Path) -> XmlConfig:
    return load_toml_config(config_path, XmlConfig.model_validate)


# %% dataclasses
@dataclass(frozen=True)
class CLIArguments:
    xml_filepath: Path
    config: XmlConfig
    config_path: Path
    project_dir: Path
    allow_out_of_range_values: bool
    dry_run: bool


class NoTreeStrataError(ValueError):
    """
    Raised by get_stand_data_from_xml when a stand has no ts:TreeStrata
    container at all.
    """


@dataclass(frozen=True)
class ParsedStand:
    """One stand as parsed from the XML, before its allometry CSVs are
    written.

    Carries tree_strata -- the raw per-species growth-model inputs (age,
    basal_area, stem_count, mean_diameter, mean_height) -- which has no
    field on the shared StandData (see CONTEXT.md's StandData entry: once a
    stand's CSVs exist, only their paths and species ids matter downstream,
    not the raw numbers that produced them). process_stand consumes
    tree_strata to build the dominant/subdominant Growth_and_Yield_Table
    calls and returns the final StandData, so nothing outside this module
    needs a ParsedStand.
    """

    id: StandID
    fertility_class: int
    polygon: str
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum]
    dominant_species: int
    subdominant_species: int
    x_ykj: int
    y_ykj: int
    stem_count: int  # units: trees/ha
    mean_diameter: float  # cm

    # Optional parameters, only used for information in the stand_data.json dump
    main_group: Optional[int] = None
    sub_group: Optional[int] = None
    basal_area: Optional[float] = None  # m2/ha
    mean_height: Optional[float] = None  # m
    total_volume: Optional[float] = None  # m3/ha
    area: Optional[float] = None  # ha


@dataclass(frozen=True)
class StandPlanned:
    """Which allometry CSV(s) one stand would produce, and where -- knowable
    before any growth table is computed. Mirrors
    metsakeskus_to_allometry.py's type of the same name.

    The subdominant file exists exactly when the second-ranked species
    carries basal area of its own; see docs/adr/0002 for why a
    monoculture's subdominant layer is not written at all rather than
    duplicating the dominant one."""

    stand_id: StandID
    dominant_csv: Path
    dominant_species: int
    subdominant_csv: Optional[Path]
    subdominant_species: int


# %% Functions


def read_stands_from_xml_file(xml_file_path: Path) -> list:
    with open(xml_file_path, encoding="utf8") as fd:
        forestdata = xmltodict.parse(fd.read())

    try:
        stands = forestdata["ForestPropertyData"]["st:Stands"]["st:Stand"]
    except KeyError as error:
        raise KeyError(
            f"The XML file needs to have the keys ['ForestPropertyData']['st:Stands']['st:Stand']: {error}"
        )

    # xmltodict collapses a single repeated element to a bare dict instead of
    # a one-item list -- the same quirk get_tree_strata_data already guards
    # against for a stand's own TreeStratum children. Without this, an XML
    # file with exactly one <st:Stand> would make build_stands iterate the
    # dict's string keys (e.g. "st:StandBasicData") instead of stand records
    # (#281).
    if isinstance(stands, dict):
        stands = [stands]

    return stands


def parse_polygon_to_coords(polygon_string: str) -> tuple[tuple[float, float], ...]:
    """
    Example:
    input (str):  "377727.33186172193,6767404.564623927 377711.6057512127,6767410.3891093"

    output tuple of tuples:
    ((377727.33186172193, 6767404.564623927),
    (377711.6057512127, 6767410.3891093))

    """
    coords = []
    for pair in polygon_string.strip().split(" "):
        if pair.strip() == "":
            continue
        x, y = pair.split(",")
        coords.append((float(x), float(y)))
    return tuple(coords)


def get_tree_strata_data(
    tree_strata_xml_data: list,
) -> tuple[TreeStratum, TreeStratum, TreeStratum]:
    """
    Map strata into fixed species-code slots:

    index 0 -> TreeSpecies 1
    index 1 -> TreeSpecies 2
    index 2 -> TreeSpecies >= 3

    Missing species are represented by the shared ZERO_STRATUM -- TreeStratum
    is immutable, so there is nothing a shared instance in all three slots
    risks (no defensive per-slot copies needed, unlike when this used to be
    built from a local, mutable-in-principle BaseModel).
    """

    if isinstance(tree_strata_xml_data, dict):
        tree_strata_xml_data = [tree_strata_xml_data]

    strata = [ZERO_STRATUM, ZERO_STRATUM, ZERO_STRATUM]

    for stratum in tree_strata_xml_data:
        tree_species = int(stratum["tst:TreeSpecies"])

        tree_stratum = TreeStratum(
            age=float(stratum["tst:Age"]),
            basal_area=float(stratum["tst:BasalArea"]),
            stem_count=float(stratum["tst:StemCount"]),
            mean_diameter=float(stratum["tst:MeanDiameter"]),
            mean_height=float(stratum["tst:MeanHeight"]),
        )

        if tree_species == 1:
            strata[0] = tree_stratum
        elif tree_species == 2:
            strata[1] = tree_stratum
        else:
            strata[2] = tree_stratum

    return (strata[0], strata[1], strata[2])


def determine_dominant_and_subdominant_species(
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum],
) -> tuple[int, int]:
    """Species codes (1, 2, 3 -- see get_tree_strata_data for what each slot
    means) ranked by basal area, richest first. Mirrors
    metsakeskus_to_allometry.py's function of the same name: the subdominant
    is always the second-ranked species' own stratum, even when its basal
    area is zero -- see docs/adr/0002 for why this deliberately does not
    duplicate the dominant species for a monoculture stand."""
    basal_areas = {
        1: tree_strata[0].basal_area,
        2: tree_strata[1].basal_area,
        3: tree_strata[2].basal_area,
    }
    ranked = sorted(
        basal_areas, key=lambda species_code: basal_areas[species_code], reverse=True
    )
    return ranked[0], ranked[1]


def species_stratum(
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum], species_code: int
) -> TreeStratum:
    """The one TreeStratum a species code (1, 2, 3) refers to."""
    return {1: tree_strata[0], 2: tree_strata[1], 3: tree_strata[2]}[species_code]


def isolate_species_layer(
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum], active_species: int
) -> tuple[TreeStratum, TreeStratum, TreeStratum]:
    """Zeroes every species slot except active_species -- this is how a
    single canopy layer (dominant or subdominant) is modeled as that one
    species growing alone (docs/adr/0002)."""
    if active_species == 1:
        return (tree_strata[0], ZERO_STRATUM, ZERO_STRATUM)
    if active_species == 2:
        return (ZERO_STRATUM, tree_strata[1], ZERO_STRATUM)
    return (ZERO_STRATUM, ZERO_STRATUM, tree_strata[2])


def get_stand_data_from_xml(stand: dict) -> ParsedStand:
    stand_basic_data = stand["st:StandBasicData"]

    tree_stand_data = stand.get("ts:TreeStandData")
    if tree_stand_data is None:
        raise ValueError(f"Stand {stand['@id']} has no ts:TreeStandData")

    tree_stand_data_date = tree_stand_data.get("ts:TreeStandDataDate")
    if tree_stand_data_date is None:
        raise ValueError(f"Stand {stand['@id']} has no ts:TreeStandDataDate")

    tree_stand_summary = tree_stand_data_date.get("tss:TreeStandSummary")
    if tree_stand_summary is None:
        raise ValueError(f"Stand {stand['@id']} has no tss:TreeStandSummary")

    tree_strata_container = tree_stand_data_date.get("tst:TreeStrata")

    # Stands with no tree strata information at all are not malformed -- see
    # NoTreeStrataError -- so this is reported as a skip, not a batch-aborting
    # raise like the other missing pieces above/below.
    if tree_strata_container is None:
        raise NoTreeStrataError(f"Stand {stand['@id']} has no TreeStrata")

    tree_strata_xml_data = tree_strata_container.get("tst:TreeStratum")

    if tree_strata_xml_data is None:
        raise ValueError(f"Stand {stand['@id']} has no TreeStratum")

    tree_strata = get_tree_strata_data(tree_strata_xml_data)

    dominant_species, subdominant_species = determine_dominant_and_subdominant_species(
        tree_strata
    )

    strata_stem_counts_per_stratum = [stratum.stem_count for stratum in tree_strata]

    polygon = stand_basic_data["gdt:PolygonGeometry"]["gml:polygonProperty"][
        "gml:Polygon"
    ]["gml:exterior"]["gml:LinearRing"]["gml:coordinates"]

    # Location in YKJ coordinates -- input to Growth_and_Yield_Table and, per
    # ticket 06, a required (stored, not lazily-derived) StandData field.
    # Coordinate transformer ETRS-TM35FIN (EPSG:3067) -> YKJ (EPSG:2393)
    first_vertex = parse_polygon_to_coords(polygon)[0]
    x_ykj, y_ykj = point_to_ykj(*first_vertex)

    # Parse and validate all necessary XML data
    return ParsedStand(
        id=StandID(stand["@id"]),
        fertility_class=int(float(stand_basic_data["st:FertilityClass"])),
        polygon=polygon,
        tree_strata=tree_strata,
        dominant_species=dominant_species,
        subdominant_species=subdominant_species,
        x_ykj=x_ykj,
        y_ykj=y_ykj,
        stem_count=round(sum(strata_stem_counts_per_stratum)),
        mean_diameter=float(tree_stand_summary["tss:MeanDiameter"]),
        # Optionals
        main_group=int(stand_basic_data["st:MainGroup"]),
        sub_group=int(stand_basic_data["st:SubGroup"]),
        basal_area=float(tree_stand_summary["tss:BasalArea"]),
        mean_height=float(tree_stand_summary["tss:MeanHeight"]),
        total_volume=float(tree_stand_summary["tss:Volume"]),
        area=float(stand_basic_data["st:Area"]),
    )


def build_stands(stands) -> tuple[list[ParsedStand], list[StandSkipped]]:
    """Batch get_stand_data_from_xml, isolating the one skippable failure
    (NoTreeStrataError -- see get_stand_data_from_xml) as a StandSkipped
    instead of a discarded None sentinel, so the Filtering section can
    report which stands were dropped and why."""
    parsed_stands: list[ParsedStand] = []
    skipped: list[StandSkipped] = []
    for stand in stands:
        try:
            parsed_stands.append(get_stand_data_from_xml(stand))
        except NoTreeStrataError as error:
            skipped.append(
                StandSkipped(stand_id=StandID(stand["@id"]), reason=str(error))
            )
    return parsed_stands, skipped


def dump_stand_data_document(output_dir: Path, document: StandDataDocument) -> None:
    json_output = output_dir / STAND_DATA_FILENAME
    json_output.write_text(document.model_dump_json())
    return None


def plan_stand_output(parsed_stand: ParsedStand, output_dir: Path) -> StandPlanned:
    """Which allometry CSV(s) one stand would produce, and where -- knowable
    before any growth table is computed, so both a dry run and the real run
    name them the same way."""
    subdominant_stratum = species_stratum(
        parsed_stand.tree_strata, parsed_stand.subdominant_species
    )
    return StandPlanned(
        stand_id=parsed_stand.id,
        dominant_csv=output_dir / f"{parsed_stand.id}_dominant.csv",
        dominant_species=parsed_stand.dominant_species,
        subdominant_csv=(
            output_dir / f"{parsed_stand.id}_subdominant.csv"
            if subdominant_stratum.basal_area > 0
            else None
        ),
        subdominant_species=parsed_stand.subdominant_species,
    )


def build_growth_and_yield_table(
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum],
    active_species: int,
    fertility_class: int,
    x_ykj: int,
    y_ykj: int,
    altitude: float,
    ddy: float,
    n_trees: int,
    start_year: int,
    end_year: int,
    step_years: int,
    PEAT: int,
) -> pd.DataFrame:
    """One canopy layer's allometric growth trajectory (get_table's
    age-indexed rows from start_year to end_year), modeled as active_species
    growing alone. Mirrors metsakeskus_to_allometry.py's function of the
    same name."""
    layer = isolate_species_layer(tree_strata, active_species)

    gy = Growth_and_Yield_Table(
        age_1=layer[0].age,
        G_1=layer[0].basal_area,
        N_1=layer[0].stem_count,
        Dg_1=layer[0].mean_diameter,
        Hg_1=layer[0].mean_height,
        age_2=layer[1].age,
        G_2=layer[1].basal_area,
        N_2=layer[1].stem_count,
        Dg_2=layer[1].mean_diameter,
        Hg_2=layer[1].mean_height,
        age_3=layer[2].age,
        G_3=layer[2].basal_area,
        N_3=layer[2].stem_count,
        Dg_3=layer[2].mean_diameter,
        Hg_3=layer[2].mean_height,
        DDY=ddy,  # Temperature sum, degree days
        fertility_class=fertility_class,
        peat=PEAT,
        y=y_ykj,
        x=x_ykj,
        altitude=altitude,  # Altitude above the sea level
        n_trees=n_trees,
    )
    return gy.get_table(start_year=start_year, end_year=end_year, step_years=step_years)


def process_stand(
    config: XmlConfig, parsed_stand: ParsedStand, PEAT: int, output_dir: Path
) -> StandData:
    """Builds the dominant canopy layer's allometry CSV (always) and the
    subdominant's (only when a genuine second species is present -- see
    plan_stand_output), then returns the finished StandData record for this
    stand, canopy_layer_files populated with what was actually written.

    Both growth tables are computed in full BEFORE either is written to
    disk, mirroring metsakeskus_to_allometry.py's process_stand: a failure
    partway through must not leave a stray, unreferenced CSV behind."""
    plan = plan_stand_output(parsed_stand, output_dir)

    dominant_table = build_growth_and_yield_table(
        parsed_stand.tree_strata,
        plan.dominant_species,
        parsed_stand.fertility_class,
        parsed_stand.x_ykj,
        parsed_stand.y_ykj,
        config.altitude,
        config.ddy,
        config.n_trees,
        config.start_year,
        config.end_year,
        config.step_years,
        PEAT,
    )

    subdominant_table: Optional[pd.DataFrame] = None
    if plan.subdominant_csv is not None:
        subdominant_table = build_growth_and_yield_table(
            parsed_stand.tree_strata,
            plan.subdominant_species,
            parsed_stand.fertility_class,
            parsed_stand.x_ykj,
            parsed_stand.y_ykj,
            config.altitude,
            config.ddy,
            config.n_trees,
            config.start_year,
            config.end_year,
            config.step_years,
            PEAT,
        )

    # Both tables computed successfully (or there is no subdominant layer to
    # compute) -- only now do we write anything to disk.
    dominant_table.to_csv(plan.dominant_csv, index=False)
    canopy_layer_files = {
        CanopyLayerName.dominant: AllometryFileAndSpecies(
            file_path=plan.dominant_csv, species_id=plan.dominant_species
        ),
    }

    if subdominant_table is not None:
        assert plan.subdominant_csv is not None
        subdominant_table.to_csv(plan.subdominant_csv, index=False)
        canopy_layer_files[CanopyLayerName.subdominant] = AllometryFileAndSpecies(
            file_path=plan.subdominant_csv, species_id=plan.subdominant_species
        )

    print(f"Allometric road map successfully generated for stand {parsed_stand.id}")

    return StandData(
        site_fertility_class=parsed_stand.fertility_class,
        canopy_layer_files=canopy_layer_files,
        x_ykj=parsed_stand.x_ykj,
        y_ykj=parsed_stand.y_ykj,
        polygon=parsed_stand.polygon,
        stand_area=parsed_stand.area,
        main_group=parsed_stand.main_group,
        sub_group=parsed_stand.sub_group,
        basal_area=parsed_stand.basal_area,
        mean_height=parsed_stand.mean_height,
        mean_diameter=parsed_stand.mean_diameter,
        total_volume=parsed_stand.total_volume,
        stem_count=parsed_stand.stem_count,
    )


# %% CLI


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description="Convert XML stand data to allometry file format"
    )
    parser.add_argument(
        "xml_file",
        type=make_existing_file_validator(".xml"),
        help="Path to the XML file",
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
            "nothing at all: no CSVs, no JSON, not even the output folder."
        ),
    )

    args = parser.parse_args()

    # --config defaults to config.toml directly inside --project-dir -- the
    # layout the docs have the user set up beforehand.
    config_path = resolve_config_path(args.config, args.project_dir, parser)
    config = load_xml_config(config_path)

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
        xml_filepath=args.xml_file,
        config=config,
        config_path=config_path,
        project_dir=args.project_dir,
        allow_out_of_range_values=args.allow_out_of_range_values,
        dry_run=args.dry_run,
    )


# %% Progress-report printing (side-effecting; kept out of the pure layer above)


def csv_counts(plans: list[StandPlanned]) -> tuple[int, int]:
    """(dominant, subdominant) CSV counts -- every planned stand has a
    dominant file, only some have a subdominant one."""
    dominant = len(plans)
    subdominant = sum(1 for plan in plans if plan.subdominant_csv is not None)
    return dominant, subdominant


def print_dry_run_plan(parsed_stands: list[ParsedStand], output_dir: Path) -> None:
    """The dry run's stand-in for the real run's writing report: the same
    counts and file names, with nothing on disk. plan_stand_output is pure
    and cheap (no growth-model math), so the dry run can report the real
    dominant/subdominant split, not just a stand count."""
    plans = [plan_stand_output(stand, output_dir) for stand in parsed_stands]
    dominant, subdominant = csv_counts(plans)
    json_path = output_dir / STAND_DATA_FILENAME

    print(f"Destination folder: {output_dir.resolve()} (not created)")
    print()
    print(
        f"Would write: {len(plans):,} stand(s) -- {dominant:,} dominant + "
        f"{subdominant:,} subdominant = {dominant + subdominant:,} CSV(s)"
    )
    print(f"Would write informational JSON: {json_path}")


# %% main


def main():
    cli_args = parse_CLI_arguments()
    output_dir = output_dir_for_project(cli_args.project_dir)

    print_section("Reading")
    print("Tool initialized with:")
    print(f"    - xml_file    = {cli_args.xml_filepath.resolve()}")
    print(f"    - project_dir = {cli_args.project_dir.resolve()}")
    print(f"    - config      = {cli_args.config_path.resolve()}")
    print(f"    - output_dir  = {output_dir.resolve()}")
    print(f"    - altitude    = {cli_args.config.altitude}")
    print(f"    - ddy         = {cli_args.config.ddy}")
    if cli_args.dry_run:
        print("    - DRY RUN -- nothing will be written")
    print()

    stands = read_stands_from_xml_file(cli_args.xml_filepath)

    print_section("Filtering")
    print("1. TreeStrata check -- stands with no recorded tree strata are skipped:")
    parsed_stands, skipped = build_stands(stands)
    total_stands = len(parsed_stands) + len(skipped)
    print(f"   -> {len(parsed_stands):,} / {total_stands:,} stands kept")
    print_skips(skipped, "   Skipped (no TreeStrata)")
    print()
    print(f"Stands ready for allometry: {len(parsed_stands):,}")

    # Everything above this point is identical in a dry run: reading and
    # filtering are precisely what a dry run exists to show. What it skips is
    # everything below -- the growth model and every write, the output
    # folder included.
    if cli_args.dry_run:
        print_section("Writing (dry run -- nothing is written)")
        print_dry_run_plan(parsed_stands, output_dir)
        return

    print_section("Writing")
    print(f"Destination folder: {output_dir.resolve()}")
    # Created here rather than at argument-parsing time, so a run that fails
    # while reading or filtering leaves no empty folder behind to block the
    # next attempt.
    output_dir.mkdir(parents=True)
    print()

    # PEAT=1 assumes all sites are peatland sites.
    print("Assuming all sites are peatland sites!")
    final_stands: dict[StandID, StandData] = {
        parsed_stand.id: process_stand(
            cli_args.config, parsed_stand, PEAT=1, output_dir=output_dir
        )
        for parsed_stand in parsed_stands
    }

    # Derived from what process_stand actually wrote, not recomputed via a
    # second plan_stand_output pass over every stand.
    dominant = len(final_stands)
    subdominant = sum(
        1
        for stand_data in final_stands.values()
        if CanopyLayerName.subdominant in stand_data.canopy_layer_files
    )
    print(
        f"Allometry files written: {len(final_stands):,} stand(s) -- {dominant:,} "
        f"dominant + {subdominant:,} subdominant = "
        f"{dominant + subdominant:,} CSV(s)"
    )

    json_path = output_dir / STAND_DATA_FILENAME
    dump_stand_data_document(
        output_dir=output_dir,
        document=StandDataDocument(
            altitude=cli_args.config.altitude,
            ddy=cli_args.config.ddy,
            stands=final_stands,
        ),
    )
    print(f"Informational JSON written: {json_path}")


if __name__ == "__main__":
    main()
