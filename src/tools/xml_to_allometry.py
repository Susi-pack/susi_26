# Gets XML stand data to the allometry file format required by SUSI
# Adapted by Iñaki Urzainki from Mikko Niemi's original code.

# %% Imports
from susi.io.load_output_data import StandID
from typing import Optional
import math
import xmltodict
from pydantic import BaseModel, computed_field, Field
import argparse
from dataclasses import dataclass
from pathlib import Path
from pyproj import Transformer
from susi.core.allometric_road_map import Growth_and_Yield_Table


# %% Constants

# Typical value ranges for Finnish forest land, used to warn/block on likely
# mistyped --altitude / --ddy input. See issue #194 / PR #174 discussion.
ALTITUDE_MIN = 0.0  # metres above sea level
ALTITUDE_MAX = 1000.0
DDY_MIN = 500.0  # temperature sum, degree days per year
DDY_MAX = 2000.0


# %% dataclasses
@dataclass
class CLIArguments:
    xml_filepath: Path
    output_folder: Path
    altitude: float
    ddy: float


class TreeStratum(BaseModel):
    age: int
    basal_area: float
    stem_count: int
    mean_diameter: float
    mean_height: float


class StandData(BaseModel):
    """
    Stand data  from the XML file
    """

    id: StandID
    fertility_class: int
    polygon: str
    tree_strata: tuple[TreeStratum, TreeStratum, TreeStratum]
    main_species: int
    stem_count: int = Field(description="units: trees/ha")
    mean_diameter: float = Field(description="cm")

    # Optional parameters, only used for information in json dump
    main_group: Optional[int] = None
    sub_group: Optional[int] = None
    soil_type: Optional[int] = None
    mean_age: Optional[int] = Field(default=None, description="years")
    basal_area: Optional[float] = Field(default=None, description="m2/ha")
    mean_height: Optional[float] = Field(default=None, description="m")
    total_volume: Optional[float] = Field(default=None, description="m3/ha")
    area: Optional[float] = Field(default=None, description="ha")

    @computed_field
    @property
    def coords(self) -> tuple[tuple[float, float], ...]:
        return parse_polygon_to_coords(self.polygon)


class ManyStandDatas(BaseModel):
    """
    Only used to serialize everything to the same .json
    """

    stand_datas: dict[StandID, StandData]


def _create_many_stand_datas(stand_datas: list[StandData]) -> ManyStandDatas:
    many_stand_datas = {}
    for stand_data in stand_datas:
        many_stand_datas[stand_data.id] = stand_data

    return ManyStandDatas(stand_datas=many_stand_datas)


# %% Functions
def valid_xml_path(value: str) -> Path:
    path = Path(value)

    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {value}")

    if path.suffix.lower() != ".xml":
        raise argparse.ArgumentTypeError("File must have .xml extension")

    return path


def out_of_range_message(
    name: str, value: float, min_value: float, max_value: float
) -> Optional[str]:
    """
    Return a human-readable message if value falls outside [min_value, max_value],
    or None if it is within range (bounds are inclusive).
    """
    if value < min_value or value > max_value:
        return (
            f"{name}={value} is outside the enforced range [{min_value}, {max_value}]"
        )
    return None


def valid_directory(value: str) -> Path:
    path = Path(value)

    if not path.exists():
        raise argparse.ArgumentTypeError("Specified output path does not exist.")

    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path exists but is not a directory: {value}")

    return path


def parse_CLI_arguments() -> CLIArguments:
    parser = argparse.ArgumentParser(
        description="Convert XML stand data to allometry file format"
    )
    parser.add_argument(
        "xml_file",
        type=valid_xml_path,
        help="Path to the XML file",
    )
    parser.add_argument(
        "output_dir",
        type=valid_directory,
        help="Output folder for generated allometry files.",
    )

    parser.add_argument(
        "--altitude",
        type=float,
        required=True,
        help=(
            "Altitude above sea level, in metres, applied to every stand in this run "
            "(example: --altitude=150 -- Finnish forest land altitude is typically "
            "sea level to 700 m). "
            f"Blocked if outside the enforced range [{ALTITUDE_MIN}, {ALTITUDE_MAX}] "
            "unless --allow-out-of-range-values is given."
        ),
    )

    parser.add_argument(
        "--ddy",
        type=float,
        required=True,
        help=(
            "Temperature sum (degree days per year, DDY), applied to every stand in "
            "this run (example: --ddy=1200 -- Finnish DDY is typically ~600 in "
            "Lapland to ~1500 in southern Finland). "
            f"Blocked if outside the enforced range [{DDY_MIN}, {DDY_MAX}] unless "
            "--allow-out-of-range-values is given."
        ),
    )

    parser.add_argument(
        "--allow-out-of-range-values",
        action="store_true",
        help=(
            "Allow --altitude/--ddy values outside their enforced range instead of "
            "blocking. Out-of-range values are still printed as a warning."
        ),
    )

    args = parser.parse_args()

    # NaN is not a physically meaningful altitude/DDY value under any
    # circumstances (unlike an out-of-range-but-real number), so it is
    # rejected outright -- --allow-out-of-range-values does not apply.
    nan_names = [
        name
        for name, value in (("altitude", args.altitude), ("ddy", args.ddy))
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
                ("altitude", args.altitude, ALTITUDE_MIN, ALTITUDE_MAX),
                ("ddy", args.ddy, DDY_MIN, DDY_MAX),
            )
        )
        if message is not None
    ]

    if out_of_range_messages:
        if args.allow_out_of_range_values:
            for message in out_of_range_messages:
                print(
                    f"Warning: {message}; proceeding due to --allow-out-of-range-values"
                )
        else:
            parser.error(
                "; ".join(out_of_range_messages)
                + ". Pass --allow-out-of-range-values to override."
            )

    return CLIArguments(
        xml_filepath=args.xml_file,
        output_folder=args.output_dir,
        altitude=args.altitude,
        ddy=args.ddy,
    )


def read_stands_from_xml_file(xml_file_path: Path) -> dict:
    with open(xml_file_path, encoding="utf8") as fd:
        forestdata = xmltodict.parse(fd.read())

    try:
        stands = forestdata["ForestPropertyData"]["st:Stands"]["st:Stand"]
    except KeyError as error:
        raise KeyError(
            f"The XML file needs to have the keys ['ForestPropertyData']['st:Stands']['st:Stand']: {error}"
        )

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

    Missing species are represented by empty strata.
    """

    if isinstance(tree_strata_xml_data, dict):
        tree_strata_xml_data = [tree_strata_xml_data]

    empty_stratum = TreeStratum(
        age=0,
        basal_area=0,
        stem_count=0,
        mean_diameter=0,
        mean_height=0,
    )

    strata = [
        empty_stratum.model_copy(),
        empty_stratum.model_copy(),
        empty_stratum.model_copy(),
    ]

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

    return tuple(strata)


def get_stand_data_from_xml(stand: dict) -> StandData | None:
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

    # Skip stands if tree strata information is missing.
    if tree_strata_container is None:
        f"Skipping stand {stand['@id']}: no TreeStrata"
        return None

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

    # Parse and validate all necessary XML data
    return StandData(
        id=StandID(stand["@id"]),
        fertility_class=int(float(stand_basic_data["st:FertilityClass"])),
        polygon=stand_basic_data["gdt:PolygonGeometry"]["gml:polygonProperty"][
            "gml:Polygon"
        ]["gml:exterior"]["gml:LinearRing"]["gml:coordinates"],
        tree_strata=tree_strata,
        stem_count=round(sum(strata_stem_counts_per_stratum)),
        main_species=main_species,
        mean_diameter=float(tree_stand_summary["tss:MeanDiameter"]),
        # Optionals
        main_group=int(stand_basic_data["st:MainGroup"]),
        sub_group=int(stand_basic_data["st:SubGroup"]),
        soil_type=int(stand_basic_data["st:SoilType"]),
        mean_age=int(tree_stand_summary["tss:MeanAge"]),
        basal_area=float(tree_stand_summary["tss:BasalArea"]),
        mean_height=float(tree_stand_summary["tss:MeanHeight"]),
        total_volume=float(tree_stand_summary["tss:Volume"]),
        area=float(stand_basic_data["st:Area"]),
    )


def dump_json_info_to_file(
    output_folder: Path, many_stand_datas: ManyStandDatas
) -> None:
    json_output = output_folder / "extra_XML_info.json"
    json_output.write_text(many_stand_datas.model_dump_json())
    return None


def get_ykj_coordinates(coords: tuple[float, float]) -> tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:3067", "EPSG:2393", always_xy=True)
    ykj_e, ykj_n = transformer.transform(coords[0], coords[1])
    y = round(ykj_n / 1000)
    x = round(ykj_e / 10000)
    return x, y


def process_stand(cli_args: CLIArguments, stand_data: StandData, PEAT: int):
    strata_basal_areas_per_stratum = [
        stratum.basal_area for stratum in stand_data.tree_strata
    ]
    strata_stem_counts_per_stratum = [
        stratum.stem_count for stratum in stand_data.tree_strata
    ]

    # Location in YKJ coordinates, and input variables x & y to sawlog reduction model
    # Coordinate transformer ETRS-TM35FIN (EPSG:3067) -> YKJ (EPSG:2393)

    coords = (stand_data.coords[0][0], stand_data.coords[0][1])
    x, y = get_ykj_coordinates(coords)

    # Generate stand allometry
    gy = Growth_and_Yield_Table(
        age_1=stand_data.tree_strata[0].age,
        G_1=strata_basal_areas_per_stratum[0],
        N_1=strata_stem_counts_per_stratum[0],
        Dg_1=stand_data.tree_strata[0].mean_diameter,
        Hg_1=stand_data.tree_strata[0].mean_height,
        age_2=stand_data.tree_strata[1].age,
        G_2=strata_basal_areas_per_stratum[1],
        N_2=strata_stem_counts_per_stratum[1],
        Dg_2=stand_data.tree_strata[1].mean_diameter,
        Hg_2=stand_data.tree_strata[1].mean_height,
        age_3=stand_data.tree_strata[2].age,
        G_3=strata_basal_areas_per_stratum[2],
        N_3=strata_stem_counts_per_stratum[2],
        Dg_3=stand_data.tree_strata[2].mean_diameter,
        Hg_3=stand_data.tree_strata[2].mean_height,
        DDY=cli_args.ddy,  # Temperature sum, degree days
        fertility_class=stand_data.fertility_class,
        peat=PEAT,
        y=y,
        x=x,
        altitude=cli_args.altitude,  # Altitude above the sea level
        n_trees=20,  # Number of reference trees per stratum
    )
    page_1 = gy.get_table(start_year=5, end_year=80, step_years=5)
    page_1.insert(0, "Species_ID", stand_data.main_species)

    page_1.to_csv(
        cli_args.output_folder / f"susi_input_{stand_data.id}.csv",
        index=False,
    )

    print(f"Allometric road map successfully generated for stand {stand_data.id}")


# %% main


def main():
    cli_args = parse_CLI_arguments()

    print("Tool initialized with:")
    print(f"    - altitude = {cli_args.altitude}")
    print(f"    - ddy = {cli_args.ddy}")

    stands = read_stands_from_xml_file(cli_args.xml_filepath)

    stand_datas = [
        stand_data
        for stand_data in (get_stand_data_from_xml(stand) for stand in stands)
        if stand_data is not None
    ]

    # PEAT=1 assumes all sites are peatland sites.
    print("Assuming all sites are peatland sites!")
    for stand_data in stand_datas:
        process_stand(cli_args, stand_data, PEAT=1)

    dump_json_info_to_file(
        output_folder=cli_args.output_folder,
        many_stand_datas=_create_many_stand_datas(stand_datas=stand_datas),
    )


if __name__ == "__main__":
    main()
