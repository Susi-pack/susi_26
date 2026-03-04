# Gets XML stand data to the allometry file format required by SUSI
# Adapted by Iñaki Urzainki from Mikko Niemi's original code.

# %% Imports
from typing import Optional
import pandas as pd
import xmltodict
from pydantic import BaseModel, computed_field, Field
import argparse
from dataclasses import dataclass
from pathlib import Path
from pyproj import Transformer
from susi.core.allometric_road_map import Growth_and_Yield_Table


# %% dataclasses
@dataclass
class CLIArguments:
    xml_filepath: Path
    output_folder: Path
    do_thinning: bool


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

    id: int
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

    @computed_field
    @property
    def coords(self) -> tuple[tuple[float, float], ...]:
        return parse_polygon_to_coords(self.polygon)


class ManyStandDatas(BaseModel):
    """
    Only used to serialize everything to the same .json
    """

    stand_datas: list[StandData]


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
        "--do-thinning", action="store_true", help="Do thinning. Default: False"
    )

    args = parser.parse_args()

    return CLIArguments(
        xml_filepath=args.xml_file,
        output_folder=args.output_dir,
        do_thinning=args.do_thinning,
    )


def sampling_stand_thinning_rate(species_id, stem_count):
    target_N = 1800 if species_id == 2 else 2000
    return target_N / stem_count


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
    tree_strata_list = []
    for stratum in tree_strata_xml_data:
        tree_strata_list.append(
            TreeStratum(
                age=int(stratum["tst:Age"]),
                basal_area=float(stratum["tst:BasalArea"]),
                stem_count=int(stratum["tst:StemCount"]),
                mean_diameter=float(stratum["tst:MeanDiameter"]),
                mean_height=float(stratum["tst:MeanHeight"]),
            )
        )
    return tuple(tree_strata_list)


def get_stand_data_from_xml(stand: dict) -> StandData:
    stand_basic_data = stand["st:StandBasicData"]
    tree_stand_summary = stand["ts:TreeStandData"]["ts:TreeStandDataDate"][
        "tss:TreeStandSummary"
    ]
    tree_strata_xml_data = stand["ts:TreeStandData"]["ts:TreeStandDataDate"][
        "tst:TreeStrata"
    ]["tst:TreeStratum"]
    tree_strata = get_tree_strata_data(tree_strata_xml_data)

    strata_basal_areas_per_stratum = []
    strata_stem_counts_per_stratum = []
    for tree_stratum in tree_strata:
        strata_basal_areas_per_stratum.append(tree_stratum.basal_area)
        strata_stem_counts_per_stratum.append(tree_stratum.stem_count)

    # The main species is the one with the largest basal area
    # THe +1 is there to agree with Mikko's nomenclature starting at 1.
    # I (Iñaki) don't know if that is more fundamental than that, i.e.,
    # I do not know if it is a choice by Mikko or if the number of the
    # species actually means something else.
    # NOTE: in Mikko's script the species are {1,2,4}. Here they are {1,2,3}
    main_species = (
        strata_basal_areas_per_stratum.index(max(strata_basal_areas_per_stratum)) + 1
    )

    # Parse and validate all necessary XML data
    return StandData(
        id=int(stand["@id"]),
        fertility_class=int(stand_basic_data["st:FertilityClass"]),
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
    )


def compute_thinning_rate(stand_data: StandData) -> float:
    # Check the need of sapling stand thinning
    threshold = 2200 if stand_data.main_species == 2 else 2500

    thinning_needed: bool = (stand_data.mean_diameter < 8) & (
        stand_data.stem_count > threshold
    )

    if thinning_needed:
        thinning_rate = sampling_stand_thinning_rate(
            stand_data.main_species, stand_data.stem_count
        )

        print("Sampling stand thinning is necessary!")
        print(
            f"--- Stem count decreased from {stand_data.stem_count} to {round(thinning_rate * stand_data.stem_count)}"
        )
        print()

    else:  # NO thinning
        thinning_rate = 1.0
    return thinning_rate


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


def process_stand(cli_args: CLIArguments, stand_data: StandData):
    if cli_args.do_thinning:
        thinning_rate = compute_thinning_rate(stand_data)
    else:
        thinning_rate = 1.0

    # Apply thinning:
    strata_basal_areas_per_stratum = [
        stratum.basal_area * thinning_rate for stratum in stand_data.tree_strata
    ]
    strata_stem_counts_per_stratum = [
        stratum.stem_count * thinning_rate for stratum in stand_data.tree_strata
    ]

    # Location in YKJ coordinates, and input variables x & y to sawlog reduction model
    # Coordinate transformer ETRS-TM35FIN (EPSG:3067) -> YKJ (EPSG:2393)

    coords = (stand_data.coords[0][0], stand_data.coords[0][1])
    x, y = get_ykj_coordinates(coords)

    print("Assuming all sites are peatland sites")
    PEAT = 1

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
        DDY=1300,  # Temperature sum, degree days
        fertility_class=stand_data.fertility_class,
        peat=PEAT,
        y=y,
        x=x,
        altitude=123,  # Altitude above the sea level
        n_trees=20,  # Number of reference trees per stratum
    )
    page_1 = gy.get_table(start_year=5, end_year=60, step_years=5)

    # Write to Excel with two sheets
    page2 = pd.DataFrame(
        {
            "StandID": [1],
            "Schedule": [1],
            "Year": [0],
            "HarvestType": ["no_loggings"],
            "Species_id": [stand_data.main_species],
        }
    )

    with pd.ExcelWriter(
        cli_args.output_folder / f"susi_input_{stand_data.id}.xlsx",
        # engine="xlsxwriter",
    ) as writer:
        page_1.to_excel(writer, sheet_name="StandData", index=False)
        page2.to_excel(writer, sheet_name="Loggings", index=False)

    print(f"Allometric road map successfully generated for stand {stand_data.id}")
    print()


# %% main


def main():
    cli_args = parse_CLI_arguments()

    print("Tool initialized with:")
    print(f"    - thinning = {cli_args.do_thinning}")

    stands = read_stands_from_xml_file(cli_args.xml_filepath)

    stand_datas = [get_stand_data_from_xml(stand) for stand in stands]

    for stand_data in stand_datas:
        process_stand(cli_args, stand_data)

    dump_json_info_to_file(
        output_folder=cli_args.output_folder,
        many_stand_datas=ManyStandDatas(stand_datas=stand_datas),
    )


if __name__ == "__main__":
    main()
    import sys

    sys.exit()
