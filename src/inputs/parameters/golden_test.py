import datetime
from pathlib import Path

from susi.io.app_settings import AppSettings
from susi.io.susi_parameter_model import (
    WeatherParams,
    SimulationConfig,
    SusiParams,
    MottiFileParams,
    CanopyParams,
    OrganicLayerParams,
    OutputParams,
    get_photo_parameters_by_location,
    LocationsForPhotoParams,
)

_app_settings = AppSettings()
PARAMETERS = SusiParams(
    weather_parameters=WeatherParams(
        FMI_weather_filepath=_app_settings.input_folder.joinpath("weather/CFw.csv"),
    ),
    simulation_config=SimulationConfig(
        start_date=datetime.datetime(2004, 1, 1),
        end_date=datetime.datetime(2007, 12, 31),
        L=40.0,
        initial_dominant_stand_age_years=100.0,
        initial_subdominant_stand_age_years=0.0,
        initial_understorey_age_years=0.0,
        site_fertility_class=4,
    ),
    motti_file_parameters=MottiFileParams(
        path=_app_settings.input_folder,
        dominant={1: "CF_41.xlsx"},
        subdominant={0: "susi_motti_input_lyr_1.xlsx"},
        under={0: "susi_motti_input_lyr_2.xlsx"},
    ),
    canopy_parameters=CanopyParams(),
    organic_layer_parameters=OrganicLayerParams(),
    output_parameters=OutputParams(
        outfolder=_app_settings.project_root_path / Path("outputs/"),
        netcdf=Path("susi.nc"),
    ),
    photo_parameters=get_photo_parameters_by_location(
        location=LocationsForPhotoParams("All_data")
    ),
)
