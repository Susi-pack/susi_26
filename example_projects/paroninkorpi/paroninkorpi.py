# Adaptation by Iñaki Urzainki to the original script created by Mikko Niemi
# Allometry files are read from a directory
# One SUSI simulation is run per allometry file

# %% imports
import datetime
from multiprocessing import Pool

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.mask import mask
from scipy.optimize import root_scalar
from shapely.geometry import Polygon, mapping

from susi.core.susi_main import Susi
from susi.core.thinning_models import (
    ThinningRecommendation,
    calculate_thinning_recommendation,
)
from susi.io.execution_config import MultipleSusis, SimulationParams
from susi.io.load_output_data import StandID
from susi.io.metadata_model import SimulationMetaData
from susi.io.project_layout import stand_data_path_for_project
from susi.io.stand_data import (
    StandDataDocument,
    build_stand_params,
    load_stand_data_document_from_json,
)
from susi.io.susi_parameter_model import (
    CanopyLayerName,
    CanopyParams,
    CuttingManagementParams,
    LocationsForPhotoParams,
    NutrientFertilizationParameters,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    PeatTypes,
    SimulationConfig,
    SiteParams,
    StandardNPKFertilizationParameters,
    StandParams,
    SusiParams,
    Thinning,
    WeatherParams,
    get_photo_parameters_by_location,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
)
from susi.io.utils import repo_root

# %% External data files

PROJECT_DIR = repo_root() / "example_projects" / "paroninkorpi"

WEATHER_FILEPATH = PROJECT_DIR / "data" / "Weather_observations_Janakkala_1980_2024.csv"
DITCH_DEPTH_RASTER_FILEPATH = PROJECT_DIR / "data" / "ditch_depth_1m.tif"
STAND_DATA_FILEPATH = stand_data_path_for_project(PROJECT_DIR)


# %% Load stand data
N_SOIL_COLS = 20

stand_data_document = load_stand_data_document_from_json(STAND_DATA_FILEPATH)

# fertility class, species id, basal areas G_1 and G_2
site_fertility_classes: dict[StandID, int] = {}
# species_ids: dict[StandID, int] = {}
G_1s: dict[StandID, float] = {}
G_2s: dict[StandID, float] = {}
stand_params: dict[StandID, StandParams] = {}


for stand_id, data in stand_data_document.stands.items():
    site_fertility_classes[stand_id] = data.site_fertility_class
    # species_ids[stand_id] = data.allometry_file_per_layer["dominant"].species_id
    assert data.basal_area_pine is not None and data.basal_area_spruce is not None, (
        f"Stand {stand_id!r} lacks basal areas, required by this script"
    )

    G_1s[stand_id] = data.basal_area_pine
    G_2s[stand_id] = data.basal_area_spruce
    stand_params[stand_id] = build_stand_params(
        stand_data_document=stand_data_document, stand_id=stand_id, n=N_SOIL_COLS
    )


# %% Functions


def lidar_ditch_depth(ditch_depth_raster, polygon: Polygon, buffer_m=10):
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


def should_implement_thinning(
    base_scenario_results, susi_params: SusiParams, G_1, G_2, yr
) -> float | None:
    species = "pine" if G_1 >= G_2 else "spruce"
    if susi_params.stand_params.site_fertility_class <= 2:
        species = "spruce"
    if susi_params.stand_params.site_fertility_class >= 4:
        species = "pine"

    thinning_guidelines = calculate_thinning_recommendation(
        region="Southern_Finland",
        soil="Organic_soil",
        fertility_class=susi_params.stand_params.site_fertility_class,
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
    stand_id: StandID,
    stand_param: StandParams,
    ditch_depth: float,
    fertility_class: int,
    scenario: str,
) -> SimulationParams:

    start_date = datetime.datetime(2005, 1, 1)
    # Fertilized at the start year if scen == fertilization.
    # Else, not fertilized (out of the simulation period)
    if scenario == "fertilized":
        fertilization_management = StandardNPKFertilizationParameters(
            application_year=start_date.year,
            N=NutrientFertilizationParameters(
                dose=0.0,
                decay_k=0.5,
                eff=1.0,
            ),  # fertilization dose in kg ha-1, decay_k in yr-1
            P=NutrientFertilizationParameters(dose=45.0, decay_k=0.2, eff=1.0),
            K=NutrientFertilizationParameters(dose=100.0, decay_k=0.3, eff=1.0),
            pH_increment=1.0,
        )
    else:
        fertilization_management = None

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
            project_dir=PROJECT_DIR,
            run_id="first_test",
            stand_id=f"stand_{stand_id}",
            scenario_id=scenario,
        ),
        susi_params=SusiParams(
            weather_parameters=WeatherParams(
                FMI_weather_filepath=WEATHER_FILEPATH,
            ),
            simulation_config=SimulationConfig(
                start_date=start_date,
                end_date=datetime.datetime(2024, 12, 31),
            ),
            stand_params=stand_param,
            canopy_parameters=CanopyParams(),
            organic_layer_parameters=OrganicLayerParams(),
            output_parameters=OutputParams(),
            photo_parameters=get_photo_parameters_by_location(
                location=LocationsForPhotoParams("All_data")
            ),
            site_parameters=SiteParams(
                L=40.0,
                n=N_SOIL_COLS,
                sitename="susirun",
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
                cutting_management=None,
                depoN=3.5,  # Lestijärvi
                depoP=1.0,  # Lestijärvi
                depoK=0.6,  # Lestijärvi
                fertilization=fertilization_management,
                peat_temperature=PeatTemperatureParams(),
            ),
        ),
    )


def create_thinning_parameters(
    base_params: SimulationParams, cutting_yr: int, cutting_to_ba: float
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
    params["susi_params"]["site_parameters"]["cutting_management"] = (
        CuttingManagementParams(
            application_yr=cutting_yr,
            management_type=Thinning(
                target_basal_area={CanopyLayerName.dominant: cutting_to_ba}
            ),
        )
    )

    params["susi_params"]["site_parameters"]["scenario_name"] = [thinning_scenario_name]

    params["metadata"]["scenario_id"] = thinning_scenario_name

    # Validate the model to check that you did not make a mistake
    return SimulationParams.model_validate(params)


# %% Run function


def run(
    simulation_parameters: SimulationParams,
    G_1: float,
    G_2: float,
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


# %% Get ditch depth for each stand from raster file
def get_ditch_depth_from_raster_by_stand(
    stand_data_document: StandDataDocument,
) -> dict[StandID, float]:
    """initial ditch depth, m"""
    ditch_depth_raster_filepath = DITCH_DEPTH_RASTER_FILEPATH

    ditch_depths = {}
    for stand_id, data in stand_data_document.stands.items():
        assert data.polygon is not None, f"Stand {stand_id!r} has no polygon"
        ditch_depths[stand_id] = lidar_ditch_depth(
            ditch_depth_raster_filepath, data.polygon, buffer_m=10
        )

    return ditch_depths


ditch_depths = get_ditch_depth_from_raster_by_stand(stand_data_document)


# %% Create params for all base scenario Susi runs


# List of parameters that completely determine each Susi simulation
all_parameters: list[SimulationParams] = []

# These are necessary in order to check for thinning
G_1_per_parameter_set = []
G_2_per_parameter_set = []

for stand_id in stand_data_document.stands:
    ### SET BASE SCENARIOS
    ditch_depth = ditch_depths[stand_id]
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
            stand_id=stand_id,
            stand_param=stand_params[stand_id],
            ditch_depth=ditch_depth,
            fertility_class=site_fertility_classes[stand_id],
            scenario=scen,
        )

        G_1_per_parameter_set.append(G_1s[stand_id])
        G_2_per_parameter_set.append(G_2s[stand_id])

        all_parameters.append(susi_params)


# %% Execute parallel processing

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=7,
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
