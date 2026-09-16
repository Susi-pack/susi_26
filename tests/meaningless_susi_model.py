# Create a meaningless but valid Susi parameter file
# to run some tests with it.

import datetime

from susi.io.app_settings import AppSettings
from susi.io.susi_parameter_model import (
    PeatTypes,
    TreeSpecies,
    SiteParams,
    WeatherParams,
    SimulationConfig,
    SusiParams,
    CanopyLayerAllometry,
    CanopyLayerName,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    PeatTemperatureParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
    h_mor_from_drainage_and_mass_mor_Pitkanen,
    Thinning,
    CuttingManagementParams,
    AllometryFileAndSpecies,
)

_app_settings = AppSettings()

_N_SOIL_COLS = 5

PARAMETERS = SusiParams(
    weather_parameters=WeatherParams(
        FMI_weather_filepath=_app_settings.input_folder.joinpath(
            "system/weather/CFw.csv"
        ),
    ),
    simulation_config=SimulationConfig(
        start_date=datetime.datetime(2004, 1, 1),
        end_date=datetime.datetime(2007, 12, 31),
    ),
    allometry_parameters=CanopyLayerAllometry(
        allometry_dir_path=_app_settings.input_folder.joinpath("system/allometry"),
        allometry_file_registry={
            1: AllometryFileAndSpecies(filename="CF_41.csv", species_id=1)
        },
        pointers={
            CanopyLayerName.dominant: [1] * _N_SOIL_COLS,
            CanopyLayerName.subdominant: None,
            CanopyLayerName.under: None,
        },
    ),
    canopy_parameters=CanopyParams(),
    organic_layer_parameters=OrganicLayerParams(),
    output_parameters=OutputParams(),
    photo_parameters=get_photo_parameters_by_location(
        location=LocationsForPhotoParams("All_data")
    ),
    # The meaninglessness is only here.
    site_parameters=SiteParams(
        L=10.0,
        n=_N_SOIL_COLS,
        initial_canopylayer_age_years={
            CanopyLayerName.dominant: 60.0,
            CanopyLayerName.subdominant: 0.0,
            CanopyLayerName.under: 0.0,
        },
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
        cutting_management=CuttingManagementParams(
            application_yr=2004, management_type=Thinning(target_basal_area={CanopyLayerName.dominant: 12})
        ),
        depoN=4.0,
        depoP=0.1,
        depoK=1.0,
        fertilization=None,
        peat_temperature=PeatTemperatureParams(),
    ),
)
