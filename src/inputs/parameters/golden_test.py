import datetime

from susi.io.app_settings import AppSettings
from susi.io.susi_parameter_model import (
    WeatherParams,
    SimulationConfig,
    SusiParams,
    MottiFileParams,
)

_app_settings = AppSettings()
PARAMETERS = SusiParams(
    weather_parameters=WeatherParams(
        weather_filepath=_app_settings.input_folder.joinpath("weather/CFw.csv")
    ),
    simulation_config=SimulationConfig(
        start_date=datetime.datetime(2004, 1, 1),
        end_date=datetime.datetime(2007, 12, 31),
    ),
    motti_file_parameters=MottiFileParams(
        path=_app_settings.input_folder,
        dominant={1: "CF_41.xlsx"},
        subdominant={0: "susi_motti_input_lyr_1.xlsx"},
        under={0: "susi_motti_input_lyr_2.xlsx"},
    ),
)
