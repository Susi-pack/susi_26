# Gets XML stand data to the allometry file format required by SUSI
# Adapted by Iñaki Urzainki from Mikko Niemi's original code.

# %% Imports
from susi.io.load_output_data import StandID
from susi.io.susi_parameter_model import (
    AllometryFileAndSpecies,
    CanopyLayerName,
)
from typing import Optional
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
    """
    Defaulted and required parameters, loaded from a TOML file.
    """

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
    """
    One stand as parsed from the XML, before its allometry CSV is written.

    Difference with StandData: StandData is what finally gets written to the JSON file.
    ParsedStand is temporary, and never exists outside this file.
    """

    id: StandID
    fertility_class: int
    polygon: str
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum]
    main_species: int
    x_ykj: int
    y_ykj: int
    stem_count: int  # units: trees/ha
    mean_diameter: float  # cm

    # Optional parameters, only used for information in the stand_data.json dump
    main_group: Optional[int] = None
    sub_group: Optional[int] = None
    soil_type: Optional[int] = None
    mean_age: Optional[int] = None  # years
    basal_area: Optional[float] = None  # m2/ha
    mean_height: Optional[float] = None  # m
    total_volume: Optional[float] = None  # m3/ha
    area: Optional[float] = None  # ha


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
    Map strata into fixed species slots:

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

    strata_basal_areas_per_stratum = []
    strata_stem_counts_per_stratum = []
    for tree_stratum in tree_strata:
        strata_basal_areas_per_stratum.append(tree_stratum.basal_area)
        strata_stem_counts_per_stratum.append(tree_stratum.stem_count)

    # The main species is the one with the largest basal area
    #           index 0 -> TreeSpecies 1 (Pine)
    #           index 1 -> TreeSpecies 2 (Spruce)
    #           index 2 -> TreeSpecies >= 3 (Deciduous trees)
    # The +1 is there to convert index number to tree species code
    main_species = (
        strata_basal_areas_per_stratum.index(max(strata_basal_areas_per_stratum)) + 1
    )

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
        main_species=main_species,
        x_ykj=x_ykj,
        y_ykj=y_ykj,
        stem_count=round(sum(strata_stem_counts_per_stratum)),
        mean_diameter=float(tree_stand_summary["tss:MeanDiameter"]),
        # Optionals
        main_group=int(stand_basic_data["st:MainGroup"]),
        sub_group=int(stand_basic_data["st:SubGroup"]),
        # soil_type is genuinely Optional (unlike the other "Optionals"
        # here): a writer with no recorded soil type (see
        # metsakeskus_to_allometry.py's StandSiteAttributes) omits the
        # <st:SoilType> tag entirely rather than inventing a value, so this
        # must tolerate that -- xmltodict's .get() returns None for a
        # missing tag, same as the model's own soil_type: Optional[int].
        soil_type=(
            int(soil_type_xml)
            if (soil_type_xml := stand_basic_data.get("st:SoilType")) is not None
            else None
        ),
        mean_age=int(tree_stand_summary["tss:MeanAge"]),
        basal_area=float(tree_stand_summary["tss:BasalArea"]),
        mean_height=float(tree_stand_summary["tss:MeanHeight"]),
        total_volume=float(tree_stand_summary["tss:Volume"]),
        area=float(stand_basic_data["st:Area"]),
    )


def build_stands(stands) -> tuple[list[ParsedStand], list[StandSkipped]]:
    """
    Batch get_stand_data_from_xml, isolating the one skippable failure
    (NoTreeStrataError -- see get_stand_data_from_xml) as a StandSkipped
    instead of a discarded None sentinel, so the Filtering section can
    report which stands were dropped and why.
    """
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


def plan_stand_output(parsed_stand: ParsedStand, output_dir: Path) -> Path:
    """
    Where one stand's allometry CSV would land -- knowable before any
    growth table is computed, so both a dry run and the real run name it the
    same way.
    """
    return output_dir / f"{parsed_stand.id}.csv"


def process_stand(
    config: XmlConfig, parsed_stand: ParsedStand, PEAT: int, output_dir: Path
) -> StandData:
    """
    Builds the single allometry CSV for this stand
    """
    strata_basal_areas_per_stratum = [
        stratum.basal_area for stratum in parsed_stand.tree_strata
    ]
    strata_stem_counts_per_stratum = [
        stratum.stem_count for stratum in parsed_stand.tree_strata
    ]

    gy = Growth_and_Yield_Table(
        age_1=parsed_stand.tree_strata[0].age,
        G_1=strata_basal_areas_per_stratum[0],
        N_1=strata_stem_counts_per_stratum[0],
        Dg_1=parsed_stand.tree_strata[0].mean_diameter,
        Hg_1=parsed_stand.tree_strata[0].mean_height,
        age_2=parsed_stand.tree_strata[1].age,
        G_2=strata_basal_areas_per_stratum[1],
        N_2=strata_stem_counts_per_stratum[1],
        Dg_2=parsed_stand.tree_strata[1].mean_diameter,
        Hg_2=parsed_stand.tree_strata[1].mean_height,
        age_3=parsed_stand.tree_strata[2].age,
        G_3=strata_basal_areas_per_stratum[2],
        N_3=strata_stem_counts_per_stratum[2],
        Dg_3=parsed_stand.tree_strata[2].mean_diameter,
        Hg_3=parsed_stand.tree_strata[2].mean_height,
        DDY=config.ddy,  # Temperature sum, degree days
        fertility_class=parsed_stand.fertility_class,
        peat=PEAT,
        y=parsed_stand.y_ykj,
        x=parsed_stand.x_ykj,
        altitude=config.altitude,  # Altitude above the sea level
        n_trees=config.n_trees,
    )
    page_1 = gy.get_table(
        start_year=config.start_year,
        end_year=config.end_year,
        step_years=config.step_years,
    )

    output_path = plan_stand_output(parsed_stand, output_dir)
    page_1.to_csv(output_path, index=False)

    print(f"Allometric road map successfully generated for stand {parsed_stand.id}")

    return StandData(
        site_fertility_class=parsed_stand.fertility_class,
        canopy_layer_files={
            CanopyLayerName.dominant: AllometryFileAndSpecies(
                file_path=output_path, species_id=parsed_stand.main_species
            ),
        },
        x_ykj=parsed_stand.x_ykj,
        y_ykj=parsed_stand.y_ykj,
        polygon=parsed_stand.polygon,
        stand_area=parsed_stand.area,
        main_group=parsed_stand.main_group,
        sub_group=parsed_stand.sub_group,
        soil_type=parsed_stand.soil_type,
        mean_age=parsed_stand.mean_age,
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
    config_path = resolve_config_path(
        args.config, args.project_dir, parser, "config.toml"
    )
    config = load_xml_config(config_path)

    validate_altitude_ddy(
        parser, config.altitude, config.ddy, args.allow_out_of_range_values
    )

    # Refuse to reuse an existing folder rather than silently overwriting
    # whatever a prior run left there. This check runs in dry-run mode too:
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


def print_dry_run_plan(parsed_stands: list[ParsedStand], output_dir: Path) -> None:
    """The dry run's stand-in for the real run's writing report: the same
    counts and file names, with nothing on disk."""
    json_path = output_dir / STAND_DATA_FILENAME

    print(f"Destination folder: {output_dir.resolve()} (not created)")
    print()
    print(
        f"Would write: {len(parsed_stands):,} stand(s) -- {len(parsed_stands):,} CSV(s)"
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

    print(
        f"Allometry files written: {len(final_stands):,} stand(s) -- "
        f"{len(final_stands):,} CSV(s)"
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
