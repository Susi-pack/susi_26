# Adaptation by Iñaki Urzainki to theoriginal script created by Mikko Niemi
# Allometry files are read from a directory
# One SUSI simulation is run per allometry file

# %% imports
import numpy as np
import pandas as pd
import rasterio
import datetime
import json
from dataclasses import dataclass

from multiprocessing import Pool
from netCDF4 import Dataset
from pathlib import Path
from os import listdir
from os.path import isfile, join
from susi.core.susi_main import Susi
from susi.core.thinning_models import (
    calculate_thinning_recommendation,
    ThinningRecommendation,
)
from scipy.optimize import root_scalar
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
import xmltodict

from susi.io.app_settings import AppSettings

from susi.io.susi_parameter_model import (
    PeatTypes,
    TreeSpecies,
    StandardNPKFertilizationParameters,
    NutrientFertilizationParameters,
    SiteParams,
    WeatherParams,
    SimulationConfig,
    SusiParams,
    AllometryParams,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
    CanopyLayerAllometryPointers,
)

from susi.io.execution_config import SimulationParams, MultipleSusis
from susi.io.metadata_model import SimulationMetaData


# %% External data files


def load_file_pointers() -> dict:
    """Load file paths from the external configuration file."""
    config_path = (
        AppSettings().project_root_path / "inputs/Paroninkorpi/file_pointers.json"
    )

    if not config_path.exists():
        print("ERROR: Configuration file not found!")
        print(f"Expected location: {config_path}")
        print()
        print(
            "This script expects a 'file_pointers.json' file to live in 'inputs/paroninkorpi/'."
        )
        print("Please create this file with the following structure:")
        print(
            """
{
    "weather_file": "path/from/project/root/to/weather.csv",
    "allometry_directory": "path/from/project/root/to/allometry/folder",
    "forest_data_xml": "path/from/project/root/to/forest.xml",
    "ditch_depth_raster": "path/from/project/root/to/ditch.tif"
}
"""
        )
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


FILE_POINTERS = load_file_pointers()


# %% Functions


def lidar_ditch_depth(ditch_depth_raster, coords, buffer_m=10):
    # Build Shapely polygon
    polygon = Polygon(coords)

    # Add buffer (10 meters by default)
    polygon_buffered = polygon.buffer(buffer_m)
    geojson_polygon = [mapping(polygon_buffered)]

    # Open raster and mask
    with rasterio.open(ditch_depth_raster) as src:
        out_image, _out_transform = mask(src, geojson_polygon, crop=True)
        data = out_image[0].astype(float).flatten()

        # Replace nodata with NaN
        nodata = src.nodata
        data[data == nodata] = np.nan

        # Drop NaNs
        data = data[~np.isnan(data)]

        mean_value = data.mean()

    return (-1) * np.round(mean_value, decimals=2)


def predict_ditch_depth(drainage_age, peat_thickness=0.61, ditch_bed_slope=0.62):
    """
    Predict ditch shallowing by Hökkä et al. 2020, Baltic Forestry 26(2), article id 453. https://doi.org/10.46490/BF453
    """
    base = (
        49.139
        + 23.981 * np.power(drainage_age, -0.2)
        - 0.343 * drainage_age
        - 0.404 * drainage_age * peat_thickness
        + 17.961 * peat_thickness
        + 3.224 * ditch_bed_slope
    )
    if drainage_age > 35:
        base += 0.00134 * np.power(drainage_age, 2)
    return base


def estimate_drainage_age(ditch_depth_cm, peat_thickness=0.61, ditch_bed_slope=0.62):
    # Ditch depth difference at a given age
    def ditch_depth_difference(age):
        return (
            predict_ditch_depth(age, peat_thickness, ditch_bed_slope) - ditch_depth_cm
        )

    # Try solving in a reasonable range of drainage ages
    solution = root_scalar(ditch_depth_difference, bracket=[1, 100], method="brentq")

    if solution.converged:
        return np.round(solution.root)
    else:
        return None


def get_ditch_shallowing(ditch_depth, time=20):
    if ditch_depth <= -0.27:
        est_drainage_age = estimate_drainage_age((-1) * 100 * ditch_depth)
        ditch_shallowing = predict_ditch_depth(est_drainage_age + time)
        return np.round((-1) * 0.01 * ditch_shallowing, 2)
    else:
        return round(ditch_depth * 0.8, 2)


def get_ncf_outputs(file):
    # Output NetCDF4 files
    ncf = Dataset(file, mode="r")

    hdom = list(np.mean(ncf["stand"]["hdom"][:, :, 1:-1], axis=2)[0])
    ba = list(np.mean(ncf["stand"]["basalarea"][:, :, 1:-1], axis=2)[0])
    vol = list(np.mean(ncf["stand"]["volume"][:, :, 1:-1], axis=2)[0])
    log_vol = list(np.mean(ncf["stand"]["logvolume"][:, :, 1:-1], axis=2)[0])
    pulp_vol = list(np.mean(ncf["stand"]["pulpvolume"][:, :, 1:-1], axis=2)[0])

    dwtyr_latesummer = list(
        np.mean(ncf["strip"]["dwtyr_latesummer"][:, 1:, 1:-1], axis=2)[0]
    )

    stand_litter = list(np.mean(ncf["balance"]["C"]["stand_litter_in"], axis=2)[0])
    gv_litter = list(
        np.mean(ncf["balance"]["C"]["gv_litter_in"][:, 1:, 1:-1], axis=2)[0]
    )
    co2c_release = list(
        np.mean(ncf["balance"]["C"]["co2c_release"][:, 1:, 1:-1], axis=2)[0]
    )
    ch4c_release = list(
        np.mean(ncf["balance"]["C"]["ch4c_release"][:, 1:, 1:-1], axis=2)[0]
    )
    LMW_to_water = list(np.mean(ncf["balance"]["C"]["LMWdoc_to_water"], axis=2)[0])
    LMW_to_atm = list(
        np.mean(ncf["balance"]["C"]["LMWdoc_to_atm"][:, 1:, 1:-1], axis=2)[0]
    )
    HMW_to_water = list(
        np.mean(ncf["balance"]["C"]["HMW_to_water"][:, 1:, 1:-1], axis=2)[0]
    )
    HMW_to_atm = list(
        np.mean(ncf["balance"]["C"]["HMW_to_atm"][:, 1:, 1:-1], axis=2)[0]
    )
    soil_C = list(
        np.mean(ncf["balance"]["C"]["soil_c_balance_c"][:, 1:, 1:-1], axis=2)[0]
    )

    stand_change = list(np.mean(ncf["balance"]["C"]["stand_change"], axis=2)[0])
    gv_change = list(np.mean(ncf["balance"]["C"]["gv_change"][:, 1:, 1:-1], axis=2)[0])
    ecosystem_C = list(
        np.mean(ncf["balance"]["C"]["stand_c_balance_c"][:, 1:, 1:-1], axis=2)[0]
    )

    soil_CO2eq = list(
        np.mean(ncf["balance"]["C"]["soil_c_balance_co2eq"][:, 1:, 1:-1], axis=2)[0]
    )
    ecosystem_CO2eq = list(
        np.mean(ncf["balance"]["C"]["stand_c_balance_co2eq"][:, 1:, 1:-1], axis=2)[0]
    )

    N_to_water = list(np.mean(ncf["balance"]["N"]["to_water"][:, 1:, 1:-1], axis=2)[0])
    P_to_water = list(np.mean(ncf["balance"]["P"]["to_water"][:, 1:, 1:-1], axis=2)[0])

    ncf.close()

    return {
        "hdom": hdom,
        "ba": ba,
        "vol": vol,
        "log_vol": log_vol,
        "pulp_vol": pulp_vol,
        "dwtyr_latesummer": dwtyr_latesummer,
        "stand_litter": stand_litter,
        "gv_litter": gv_litter,
        "co2c_release": co2c_release,
        "ch4c_release": ch4c_release,
        "LMW_to_water": LMW_to_water,
        "LMW_to_atm": LMW_to_atm,
        "HMW_to_water": HMW_to_water,
        "HMW_to_atm": HMW_to_atm,
        "stand_change": stand_change,
        "gv_change": gv_change,
        "soil_C": soil_C,
        "ecosystem_C": ecosystem_C,
        "soil_CO2eq": soil_CO2eq,
        "ecosystem_CO2eq": ecosystem_CO2eq,
        "N_to_water": N_to_water,
        "P_to_water": P_to_water,
    }


def allometry_filename_from_stand_number(stand_number: int) -> str:
    return f"susi_input_{stand_number}.xlsx"


def read_initial_dominant_stand_age_from_allometry_file(
    stand_number: int, allometry_files_folder: Path
) -> float:
    allometry_filepath = allometry_files_folder / allometry_filename_from_stand_number(
        stand_number
    )
    return float(pd.read_excel(allometry_filepath)["Age"][0])


def should_implement_thinning(
    base_scenario_results, susi_params: SusiParams, G_1, G_2, yr
) -> float | None:
    species = "pine" if G_1 >= G_2 else "spruce"
    if susi_params.site_parameters.site_fertility_class <= 2:
        species = "spruce"
    if susi_params.site_parameters.site_fertility_class >= 4:
        species = "pine"

    thinning_guidelines = calculate_thinning_recommendation(
        region="Southern_Finland",
        soil="Organic_soil",
        fertility_class=susi_params.site_parameters.site_fertility_class,
        main_sp=species,
        H_dom=base_scenario_results["hdom"][yr],
    )
    match thinning_guidelines:
        case None:
            print(
                f"No thinning at year {yr}, as dominant height ({base_scenario_results['hdom'][yr]:.1f} m) outside the thinning model range."
            )
            return None
        case ThinningRecommendation():
            if base_scenario_results["ba"][yr] < thinning_guidelines.BA_recommendation:
                print(
                    f"No thinning at year {yr}, as basal area ({base_scenario_results['ba'][yr]:.1f} m2/ha) below the thinning limit ({thinning_guidelines.BA_recommendation:.1f} m2/ha)."
                )
                return None
            else:
                print(
                    f"Thinning possible at year {yr}, basal area from {base_scenario_results['ba'][yr]:.1f} m2/ha to {thinning_guidelines.BA_limit:.1f} m2/ha."
                )
                return thinning_guidelines.BA_limit


def prepare_susi_params(
    stand_number: int,
    allometry_files_directory_path: Path,
    ditch_depth,
    fertility_class: int,
    scenario: str,
) -> SimulationParams:

    weather_file_path = AppSettings().project_root_path / FILE_POINTERS["weather_file"]

    start_date = datetime.datetime(2005, 1, 1)
    # Fertilized at the start year if scen == fertilization.
    # Else, not fertilized (out of the simulation period)
    fertilization_application_year = (
        start_date.year if scenario == "fertilized" else 2200
    )

    # Partial blocking
    if scenario == "partialblocking":
        ditch_depth_east = -0.10
        ditch_depth_20y_east = -0.10
    else:
        ditch_depth_east = ditch_depth
        ditch_depth_20y_east = get_ditch_shallowing(ditch_depth, time=20)

    if fertility_class > 4:
        peat_types = [PeatTypes.sphagnum] * 8
        rho_mor = 80.0
    else:
        peat_types = [PeatTypes.generic] * 8
        rho_mor = 85.0

    return SimulationParams(
        metadata=SimulationMetaData(
            experiment_id="paroninkorpi",
            stand_id=f"stand_{stand_number}",
            scenario_id=scenario,
        ),
        susi_params=SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=weather_file_path,
            ),
            simulation_config=SimulationConfig(
                start_date=start_date,
                end_date=datetime.datetime(2024, 12, 31),
            ),
            allometry_parameters=AllometryParams(
                allometry_dir_path=allometry_files_directory_path,
                dominant={1: allometry_filename_from_stand_number(stand_number)},
                subdominant={0: "susi_motti_input_lyr_1.xlsx"},
                under={0: "susi_motti_input_lyr_2.xlsx"},
            ),
            canopy_parameters=CanopyParams(),
            organic_layer_parameters=OrganicLayerParams(),
            output_parameters=OutputParams(),
            photo_parameters=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            site_parameters=SiteParams(
                L=40.0,
                initial_dominant_stand_age_years=read_initial_dominant_stand_age_from_allometry_file(
                    stand_number=stand_number,
                    allometry_files_folder=allometry_files_directory_path,
                ),
                initial_subdominant_stand_age_years=0.0,
                initial_understorey_age_years=0.0,
                canopylayers=CanopyLayerAllometryPointers(
                    dominant=[1] * 20, subdominant=[0] * 20, under=[0] * 20
                ),
                site_fertility_class=fertility_class,
                sitename="susirun",
                species=TreeSpecies("Pine"),
                sfc_specification=1,
                hdom=None,
                vol=None,
                smc="Peatland",
                nLyrs=60,
                dzLyr=0.05,
                ditch_depth_west=[ditch_depth],
                ditch_depth_east=[ditch_depth_east],
                ditch_depth_20y_west=[get_ditch_shallowing(ditch_depth, time=20)],
                ditch_depth_20y_east=[ditch_depth_20y_east],
                scenario_name=[scenario],
                drain_age=30.0,
                initial_h=-0.2,
                slope=0.0,
                peat_type=peat_types,
                peat_type_bottom=[PeatTypes.generic],
                anisotropy=10.0,
                vonP=True,
                vonP_top=[2, 5, 5, 5, 6, 6, 7, 7],
                vonP_bottom=8,
                bd_top=None,
                bd_bottom=0.16,
                peatN=None,
                peatP=None,
                peatK=None,
                enable_peattop=True,
                enable_peatmiddle=True,
                enable_peatbottom=True,
                rho_mor=rho_mor,
                h_mor=h_mor_from_drainage_and_mass_mor_Pitkanen,
                cutting_yr=2200,  # out of the simulation period
                cutting_to_ba=12,
                depoN=3.5,  # Lestijärvi
                depoP=1.0,  # Lestijärvi
                depoK=0.6,  # Lestijärvi
                fertilization=StandardNPKFertilizationParameters(
                    application_year=fertilization_application_year,
                    N=NutrientFertilizationParameters(
                        dose=0.0,
                        decay_k=0.5,
                        eff=1.0,
                    ),  # fertilization dose in kg ha-1, decay_k in yr-1
                    P=NutrientFertilizationParameters(dose=45.0, decay_k=0.2, eff=1.0),
                    K=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
                    pH_increment=1.0,
                ),
                peat_temperature=PeatTemperatureParams(),
            ),
        ),
    )


def create_thinning_parameters(
    base_params: SimulationParams, cutting_yr: float, cutting_to_ba: float
) -> SimulationParams:
    """
    Create new parameter models based on another one.
    This function changes the value of the specified parameters
    and returns a fully validated model.
    """

    # Get parameters of the base model into a Python dictionary
    params = base_params.model_dump(exclude_computed_fields=True)

    base_scenario_name = params["metadata"]["scenario_id"]

    thinning_scenario_name = f"{base_scenario_name}_thinning_at_yr_{cutting_yr}"

    # Modify the Python dictionary
    params["susi_params"]["site_parameters"]["cutting_yr"] = cutting_yr
    params["susi_params"]["site_parameters"]["cutting_to_ba"] = cutting_to_ba

    params["susi_params"]["site_parameters"]["scenario_name"] = [thinning_scenario_name]

    params["metadata"]["scenario_id"] = thinning_scenario_name

    # Validate the model to check that you did not make a mistake
    return SimulationParams.model_validate(params)


# %% Run function


def run(
    simulation_parameters: SimulationParams,
    G_1: int | float,
    G_2: int | float,
) -> None:
    """
    Logic:
    1. Run Susi once for each base scenario.
    2. From the results of the simulation, evaluate if thinning is possible or not
    3. If thinning is possible, re-run Susi with thinning
    """
    # Initiate susi class
    susi = Susi(simulation_parameters)

    # Run simulation
    susi.run()

    # Save stuff
    susi.write_params_and_metadata()

    # Load results
    base_scenario_results = get_ncf_outputs(
        simulation_parameters.metadata.netcdf_output_filepath
    )

    for yr in range(0, 20, 5):
        ba_to_cut = should_implement_thinning(
            base_scenario_results=base_scenario_results,
            susi_params=simulation_parameters.susi_params,
            G_1=G_1,
            G_2=G_2,
            yr=yr,
        )
        if ba_to_cut is not None:
            thinning_parameters = create_thinning_parameters(
                base_params=simulation_parameters,
                cutting_yr=int(
                    simulation_parameters.susi_params.simulation_config.start_date.year
                    + yr
                ),
                cutting_to_ba=ba_to_cut,
            )

            susi = Susi(thinning_parameters)

            susi.run()

            susi.write_params_and_metadata()


# %% Get pre-computed allometry files from folder
ALLOMETRY_FILES_DIRECTORY_PATH: Path = (
    AppSettings().project_root_path / FILE_POINTERS["allometry_directory"]
)


def list_all_files_in_directory_with_given_extension(
    dir: Path, extension: str
) -> list[Path | str]:
    return [
        join(dir, f)
        for f in sorted(listdir(dir))
        if isfile(join(dir, f)) and f.endswith(extension)
    ]


allometry_filepaths = list_all_files_in_directory_with_given_extension(
    ALLOMETRY_FILES_DIRECTORY_PATH, extension=".xlsx"
)

# We will simulate one stand for each allometry file
N_STANDS = len(allometry_filepaths)

# %% Get additional xml data for each stand


@dataclass
class DataFromXml:
    coords: list[tuple[float, float]]
    fertility_class: int
    G_1: float
    G_2: float


def get_XML_data_for_each_stand() -> list[DataFromXml]:

    xml_path = AppSettings().project_root_path / FILE_POINTERS["forest_data_xml"]
    with open(xml_path, encoding="utf8") as fd:
        forestdata = xmltodict.parse(fd.read())

    stands = forestdata["ForestPropertyData"]["st:Stands"]

    xml_data: list[DataFromXml] = []

    for stand in stands["st:Stand"]:
        StandBasicData = stand["st:StandBasicData"]
        polygon_str = StandBasicData["gdt:PolygonGeometry"]["gml:polygonProperty"][
            "gml:Polygon"
        ]["gml:exterior"]["gml:LinearRing"]["gml:coordinates"]

        # coords
        coords = []
        for pair in polygon_str.strip().split(" "):
            if pair.strip() == "":
                continue
            x, y = pair.split(",")
            coords.append((float(x), float(y)))

        # G_1, G_2
        G_1 = G_2 = 0
        N_1 = N_2 = 0

        TreeStandData = stand["ts:TreeStandData"]
        TreeStandSummary = TreeStandData["ts:TreeStandDataDate"]["tss:TreeStandSummary"]
        try:
            TreeStratum = TreeStandData["ts:TreeStandDataDate"]["tst:TreeStrata"][
                "tst:TreeStratum"
            ]
        except Exception:
            continue

        # Extract stratum attributes
        for stratum in TreeStratum:
            species = int(stratum["tst:TreeSpecies"])
            if species == 1:
                G_1 = float(stratum["tst:BasalArea"])
                N_1 = int(stratum["tst:StemCount"])
            elif species == 2:
                G_2 = float(stratum["tst:BasalArea"])
                N_2 = int(stratum["tst:StemCount"])
            elif species >= 3:
                G_3 = float(stratum["tst:BasalArea"])
                N_3 = int(stratum["tst:StemCount"])

        G_values = {1: G_1, 2: G_2, 4: G_3}
        main_sp = max(G_values.keys(), key=lambda k: G_values[k]) if G_values else None
        N_total = N_1 + N_2 + N_3

        # Check the need of sapling stand thinning

        def sampling_stand_thinning_rate(species_id, stem_count):
            target_N = 1800 if species_id == 2 else 2000
            return target_N / stem_count

        Dg_total = float(TreeStandSummary["tss:MeanDiameter"])
        if ((main_sp == 2) & (N_total > 2200) & (Dg_total < 8)) | (
            (main_sp != 2) & (N_total > 2500) & (Dg_total < 8)
        ):
            thinning_rate = sampling_stand_thinning_rate(main_sp, N_total)

            G_1 *= thinning_rate
            G_2 *= thinning_rate
            G_3 *= thinning_rate
            N_1 *= thinning_rate
            N_2 *= thinning_rate
            N_3 *= thinning_rate

        # FertilityClass
        FertilityClass = int(StandBasicData["st:FertilityClass"])

        xml_data.append(
            DataFromXml(
                coords=coords,
                G_1=G_1,
                G_2=G_2,
                fertility_class=FertilityClass,
            )
        )
    return xml_data


# %% Get ditch depth for each stand from raster file
def get_ditch_depth_from_raster_by_stand(xml_data: list[DataFromXml]) -> list[float]:
    """initial ditch depth, m"""
    ditch_depth_raster_filepath = (
        AppSettings().project_root_path / FILE_POINTERS["ditch_depth_raster"]
    )

    n_stands = len(xml_data)

    ditch_depths = []
    for n_stand in range(n_stands):
        ditch_depths.append(
            lidar_ditch_depth(
                ditch_depth_raster_filepath, xml_data[n_stand].coords, buffer_m=10
            )
        )

    return ditch_depths


xml_data = get_XML_data_for_each_stand()

ditch_depth_for_each_stand = get_ditch_depth_from_raster_by_stand(xml_data)

fertility_class_for_each_stand = [i.fertility_class for i in xml_data]
G_1_for_each_stand = [i.G_1 for i in xml_data]
G_2_for_each_stand = [i.G_2 for i in xml_data]


# %% Create params for all base scenario Susi runs


# List of parameters that completely determine each Susi simulation
all_parameters: list[SimulationParams] = []

# These are necessary in order to check for thinning
G_1_per_parameter_set = []
G_2_per_parameter_set = []

stand_numbers = range(1, N_STANDS + 1)
for stand_number in stand_numbers:
    ditch_depth = ditch_depth_for_each_stand[stand_number - 1]
    fertility_class = fertility_class_for_each_stand[stand_number - 1]

    ### SET BASE SCENARIOS
    if ditch_depth > -0.40:
        # base_scenarios = ["default", "fertilized", "partialblocking", "DNM"]
        base_scenarios = ["default", "partialblocking", "DNM"]
    else:
        # base_scenarios = ["default", "fertilized", "partialblocking"]
        base_scenarios = ["default", "partialblocking"]

    for scen in base_scenarios:
        if scen == "DNM":
            ditch_depth = -0.60

        susi_params = prepare_susi_params(
            stand_number=stand_number,
            allometry_files_directory_path=ALLOMETRY_FILES_DIRECTORY_PATH,
            ditch_depth=ditch_depth,
            fertility_class=fertility_class,
            scenario=scen,
        )

        G_1_per_parameter_set.append(G_1_for_each_stand[stand_number - 1])
        G_2_per_parameter_set.append(G_2_for_each_stand[stand_number - 1])

        all_parameters.append(susi_params)


# %% Execute parallel processing

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=5,
)

# run() expects 3 arguments. We transpose or "zip" them here
multiprocessing_args = list(
    zip(
        execution_config.simulation_parameter_list,
        G_1_per_parameter_set,
        G_2_per_parameter_set,
    )
)
if __name__ == "__main__":
    with Pool(processes=execution_config.n_parallel_processes) as pool:
        pool.starmap(func=run, iterable=multiprocessing_args)
