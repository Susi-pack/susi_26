# Adaptation by Iñaki Urzainki to theoriginal script created by Mikko Niemi
# Allometry files are read from a directory
# One SUSI simulation is run per allometry file

# %% imports
import numpy as np
import pandas as pd
import rasterio
import datetime
import xmltodict

from netCDF4 import Dataset
from pathlib import Path
from os import listdir
from os.path import isfile, join
from susi.core.susi_main import Susi
from susi.core.susi_utils import read_FMI_weather
from susi.core.allometric_road_map import Growth_and_Yield_Table
from susi.core.thinning_models import calculate_thinning_recommendation
from scipy.optimize import root_scalar
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
from pyproj import Transformer

from susi.io.app_settings import AppSettings

from susi.io.susi_parameter_model import (
    PeatTypes,
    TreeSpecies,
    FertilizationParameters,
    NutrientFertilizationParameters,
    SiteParams,
    WeatherParams,
    SimulationConfig,
    SusiParams,
    MottiFileParams,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
)

# %% Functions

def lidar_ditch_depth(ditch_depth_raster, coords, buffer_m=10):
    # Build Shapely polygon
    polygon = Polygon(coords)

    # Add buffer (10 meters by default)
    polygon_buffered = polygon.buffer(buffer_m)
    geojson_polygon = [mapping(polygon_buffered)]

    # Open raster and mask
    with rasterio.open(ditch_depth_raster) as src:
        out_image, out_transform = mask(src, geojson_polygon, crop=True)
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

def get_allometry_filepath_from_stand_number(stand_number:int)->str:
    return f"susi_input_{stand_number}.xlsx"

def read_initial_dominant_stand_age_from_allometry_file(stand_number:int, allometry_files_folder:Path)->float:
    allometry_filepath = get_allometry_filepath_from_stand_number(stand_number)
    return float(pd.read_excel(allometry_filepath)["Age"][0])

def initialize_SUSI(spara, ditch_depth, scen):
    spara["drain_age"] = 30.0
    mass_mor = (
        1.616 * np.log(spara["drain_age"]) - 1.409
    )  # Pitkänen et al. 2012 Forest Ecology and Management 284 (2012) 100–106
    spara["h_mor"] = mass_mor / spara["rho_mor"]

    if np.median(sfc) > 4:
        spara["peat type"] = ["S", "S", "S", "S", "S", "S", "S", "S"]
        spara["peat type bottom"] = ["A"]
        spara["vonP top"] = [2, 5, 5, 5, 6, 6, 7, 7]
        spara["anisotropy"] = 10
        spara["rho_mor"] = 80.0
    else:
        spara["vonP top"] = [2, 5, 5, 5, 6, 6, 7, 7]
        spara["anisotropy"] = 10
        spara["rho_mor"] = 85.0

    spara["depoN"] = 3.5  # Lestijärvi
    spara["depoP"] = 1.0  # Lestijärvi
    spara["depoK"] = 0.6  # Lestijärvi

    spara["ditch depth west"] = [ditch_depth]
    spara["ditch depth east"] = [ditch_depth]
    spara["ditch depth 20y west"] = [get_ditch_shallowing(ditch_depth, 20)]
    spara["ditch depth 20y east"] = [get_ditch_shallowing(ditch_depth, 20)]

    spara["scenario name"] = [scen]
    spara["cutting_yr"] = 2200  # out of the simulation period

    # Fertilized at the start year (scen == fertilization) or out of the simulation period
    spara["fertilization"]["application year"] = (
        start_yr if scen == "_fertilized" else 2200
    )

    # Partial blocking
    if scen == "_partialblocking":
        spara["ditch depth east"] = [-0.10]
        spara["ditch depth 20y east"] = [-0.10]

    return spara

def prepare_susi_params(spara, stand_number:int, allometry_files_directory_path: Path, ditch_depth, fertility_class:int,scen)->SusiParams:
    input_folder = AppSettings().input_folder
    weather_file_path = input_folder / "paroninkorpi/weather_paroninkorpi/Weather_observations_Janakkala_1980_2024.csv"

    return SusiParams(
        weather_parameters=WeatherParams(
            FMI_weather_filepath=weather_file_path,
        ),
        simulation_config=SimulationConfig(
            start_date=datetime.datetime(2005, 1, 1),
            end_date=datetime.datetime(2024, 12, 31),
        ),
        motti_file_parameters=MottiFileParams(
            path=allometry_files_directory_path,
            dominant={1: get_allometry_filepath_from_stand_number(stand_number)},
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
            initial_dominant_stand_age_years=read_initial_dominant_stand_age_from_allometry_file(stand_number=stand_number, allometry_files_folder=allometry_files_directory_path),
            initial_subdominant_stand_age_years=0.0,
            initial_understorey_age_years=0.0,
            site_fertility_class=4,
            sitename="susirun",
            species=TreeSpecies("Pine"),
            sfc_specification=1,
            hdom=None,
            vol=None,
            smc="Peatland",
            nLyrs=60,
            dzLyr=0.05,
            ditch_depth_west=[-0.5],
            ditch_depth_east=[-0.5],
            ditch_depth_20y_west=[-0.5],
            ditch_depth_20y_east=[-0.5],
            scenario_name=["D60"],  # kasvunlisaykset
            drain_age=100.0,
            initial_h=-0.2,
            slope=0.0,
            peat_type=[PeatTypes.generic] * 8,
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
            rho_mor=90.0,
            h_mor=h_mor_from_drainage_and_mass_mor_Pitkanen,
            cutting_yr=2004,
            cutting_to_ba=12,
            depoN=4.0,
            depoP=0.1,
            depoK=1.0,
            fertilization=FertilizationParameters(
                application_year=2201,
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
    )


# %% Get pre-computed allometry files from folder
ALLOMETRY_FILES_DIRECTORY_PATH:Path = AppSettings().input_folder / "paroninkorpi/Stand_allometry"

def list_all_files_in_directory(dir:Path)->list[Path|str]:
    return [join(dir, f) for f in sorted(listdir(dir)) if isfile(join(dir, f))]

allometry_filepaths = list_all_files_in_directory(ALLOMETRY_FILES_DIRECTORY_PATH)



# %% Create params for all Susi runs

# We will simulate one stand for each allometry file
stand_numbers = range(1, len(allometry_filepaths)+1)

susi_parameters:list[SusiParams] = []

# Save parameters that define each simulation
ditch_depths = {}
fertility_classes = {}

for stand_number in stand_numbers:

    PLACEHOLDER_DITCH_DEPTH = 0.1
    ditch_depth = PLACEHOLDER_DITCH_DEPTH

    # TODO: Change the placeholder with the following when I get the lidar raster file
    # Drainage attributes:
    # ditch_depth = lidar_ditch_depth(
    #     ditch_depth_raster, coords, buffer_m=10
    # )  # initial ditch depth, m
    ditch_depths[stand_number] = ditch_depth


    # TODO: Change the placeholder when we get the XML data
    FERTILITY_CLASS_PLACEHOLDER = 4
    fertility_class = FERTILITY_CLASS_PLACEHOLDER
    fertility_classes[stand_number] = fertility_class


    ### SET BASE SCENARIOS
    if ditch_depth > -0.40:
        base_scenarios = ["", "_fertilized", "_partialblocking", "_DNM"]
    else:
        base_scenarios = ["", "_fertilized", "_partialblocking"]

    for scen in base_scenarios:
        if scen == "_DNM":
            ditch_depth = -0.60

        # wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(
        #     wlocation="undefined",
        #     peat=site,
        #     folderName=folderName,
        #     hdomSim=None,
        #     ageSim=ageSim,
        #     sarkaSim=sarkaSim,
        #     sfc=sfc,
        #     n=n,
        # )
        # spara = initialize_SUSI(spara, ditch_depth, scen)

        susi_params = prepare_susi_params(spara, stand_number=stand_number, allometry_files_directory_path=ALLOMETRY_FILES_DIRECTORY_PATH, ditch_depth=ditch_depth, fertility_class=fertility_class, scen)

        outpara["netcdf"] = f"{area_name}_StandNumber_{stand_numbers}{scen}.nc"


        print(
            f"Simulation period {start_yr}-{end_yr}. Initial ditch depth {ditch_depth} m, and after {end_yr - start_yr + 1} years {get_ditch_shallowing(ditch_depth, 20)} m."
        )
        print()

        print("#######################")
        print("###### CALL SUSI ######")
        print("#######################")
        print()


        susi = Susi()

        susi.run_susi(
            forc,
            wpara,
            cpara,
            org_para,
            spara,
            outpara,
            photopara,
            start_yr,
            end_yr,
            wlocation="undefined",
            mottifile=mottifile,
            peat="other",
            photosite="All data",
            folderName=folderName,
            ageSim=ageSim,
            sarkaSim=sarkaSim,
            sfc=sfc,
        )

        predictions = get_ncf_outputs(f"{folderName}{outpara['netcdf']}")

        hdom = predictions["hdom"]
        ba = predictions["ba"]

        # STUDY THINNING ALTERNATIVES:

        print()
        print("Studying thinning alternatives:")
        print()

        region = "Southern_Finland"
        soil = "Organic_soil"
        species = "pine" if G_1 >= G_2 else "spruce"
        if fertility_class <= 2:
            species = "spruce"
        if fertility_class >= 4:
            species = "pine"

        for yr in range(0, 20, 5):
            thinningGuidelines = calculate_thinning_recommendation(
                region, soil, fertility_class, species, hdom[yr]
            )
            if thinningGuidelines == (None, None):
                print(
                    f"No thinning at year {yr}, as dominant height ({hdom[yr]:.1f} m) outside the thinning model range."
                )
            elif ba[yr] < thinningGuidelines[1]:
                print(
                    f"No thinning at year {yr}, as basal area ({ba[yr]:.1f} m2/ha) below the thinning limit ({thinningGuidelines[1]:.1f} m2/ha)."
                )
            else:
                print(
                    f"Thinning possible at year {yr}, basal area from {ba[yr]:.1f} m2/ha to {thinningGuidelines[0]:.1f} m2/ha."
                )
                print("----- CALL SUSI! -----")
                print()

                wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(
                    wlocation="undefined",
                    peat=site,
                    folderName=folderName,
                    hdomSim=None,
                    ageSim=ageSim,
                    sarkaSim=sarkaSim,
                    sfc=sfc,
                    n=n,
                )
                spara = initialize_SUSI(spara, ditch_depth, scen)

                spara["cutting_yr"] = int(start_yr + yr)
                spara["cutting_to_ba"] = thinningGuidelines[0]
                spara["scenario name"] = [f"{scen}_thinning_at_yr_{yr}"]

                outpara["netcdf"] = (
                    f"{area_name}_StandNumber_{stand_numbers}{scen}_thinning_yr_{yr}.nc"
                )

                susi.run_susi(
                    forc,
                    wpara,
                    cpara,
                    org_para,
                    spara,
                    outpara,
                    photopara,
                    start_yr,
                    end_yr,
                    wlocation="undefined",
                    mottifile=mottifile,
                    peat="other",
                    photosite="All data",
                    folderName=folderName,
                    ageSim=ageSim,
                    sarkaSim=sarkaSim,
                    sfc=sfc,
                )

# %%

    """ Combine results to Excel from the existing ncf-files """

    from pathlib import Path

    folder_path = Path(folderName)

    results = []

# Loop stands:
    for stand in stands["st:Stand"]:
    StandNumber = int(stand["@id"])

# Save stand basic to a new variable:
    StandBasicData = stand["st:StandBasicData"]

# Loop through files containing stand number in their name:
    for file in folder_path.iterdir():
        if file.is_file() and (
            f"StandNumber_{StandNumber}_" in file.name
            or f"StandNumber_{StandNumber}." in file.name
        ):
            fertilized = "Ash" if "fertilized" in file.name else ""
            if "partialblocking" in file.name:
                ditch_management = "partial_blocking"
            elif "DNM" in file.name:
                ditch_management = "DNM_60_cm"
            else:
                ditch_management = ""
            if "thinning" in file.name:
                logging_type = "Thinning"
                logging_yr = int(file.name.split("_")[-1].split(".")[0])
            else:
                logging_type = ""
                logging_yr = ""

            mottifile = pd.read_excel(f"{allometry_files}susi_input_{StandNumber}.xlsx")

            predictions = get_ncf_outputs(f"{folderName}{file.name}")

            sim_result = {
                "Area": area_name,
                "StandNumber": StandNumber,
                "Fertilization": fertilized,
                "Ditch_management": ditch_management,
                "Logging": logging_type,
                "Logging_yr": logging_yr,
                "Initial_ditch_depth": ditch_attributes[StandNumber]["ditch_depth"],
                "Strip_width": ditch_attributes[StandNumber]["strip_width"],
                "MainGroup": int(StandBasicData["st:MainGroup"]),
                "SubGroup": int(StandBasicData["st:SubGroup"]),
                "FertilityClass": int(StandBasicData["st:FertilityClass"]),
                "SoilType": int(StandBasicData["st:SoilType"]),
                "MainSp": pd.read_excel(
                    f"{allometry_files}susi_input_{StandNumber}.xlsx",
                    sheet_name="Loggings",
                )["Species_id"][0],
                "MeanAge": mottifile["Age"][0],
                "BasalArea": np.round(mottifile["BA"][0], 1),
                "StemCount": mottifile["N"][0],
                "MeanDiameter": mottifile["Dg"][0],
                "MeanHeight": mottifile["Hg"][0],
                "Volume": np.round(mottifile["Volume"][0]),
                "Annual_vol_gr": np.round(
                    (predictions["vol"][20] - predictions["vol"][0]) / 20, 1
                ),  ## ADD HARVEST VOLUME
                "dwtyr_latesummer": np.round(
                    np.mean(predictions["dwtyr_latesummer"][1:]), 2
                ),
                "stand_litter": np.round(np.mean(predictions["stand_litter"][1:])),
                "gv_litter": np.round(np.mean(predictions["gv_litter"][1:])),
                "co2c_release": (-1)
                * np.round(np.mean(predictions["co2c_release"][1:])),
                "ch4c_release": (-1)
                * np.round(np.mean(predictions["ch4c_release"][1:]), 1),
                "LMW_to_water": (-1)
                * np.round(np.mean(predictions["LMW_to_water"][1:]), 1),
                "LMW_to_atm": (-1)
                * np.round(np.mean(predictions["LMW_to_atm"][1:]), 1),
                "HMW_to_water": (-1)
                * np.round(np.mean(predictions["HMW_to_water"][1:]), 1),
                "HMW_to_atm": (-1)
                * np.round(np.mean(predictions["HMW_to_atm"][1:]), 1),
                "soil_C_balance": np.round(np.mean(predictions["soil_C"][1:])),
                "stand_change": np.round(np.mean(predictions["stand_change"][1:])),
                "gv_change": np.round(np.mean(predictions["gv_change"][1:])),
                "ecos_C_balance": np.round(np.mean(predictions["ecosystem_C"][1:])),
                "soil_CO2eq": np.round(np.mean(predictions["soil_CO2eq"][1:])),
                "ecos_CO2eq": np.round(np.mean(predictions["ecosystem_CO2eq"][1:])),
                "N_to_water": np.round(np.mean(predictions["N_to_water"][1:]), 2),
                "P_to_water": np.round(np.mean(predictions["P_to_water"][1:]), 2),
            }
            results.append(sim_result)

# Convert list of dicts to DataFrame
    df_results = pd.DataFrame(results)

    df_results.to_excel(f"{base_folder}{area_name}_simulation_results.xlsx", index=False)
    print()
    print(f"Simulation results saved to {base_folder}{area_name}_simulation_results.xlsx")
