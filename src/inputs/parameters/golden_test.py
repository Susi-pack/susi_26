import datetime

from susi.io.app_settings import AppSettings
from susi.io.susi_parameter_model import WeatherParameters, SimulationConfig, SusiParams

_app_settings = AppSettings()
PARAMETERS = SusiParams(
    weather_parameters=WeatherParameters(
        weather_filepath=_app_settings.input_folder.joinpath("weather/CFw.csv")
    ),
    simulation_config=SimulationConfig(
        start_date=datetime.datetime(2004, 1, 1),
        end_date=datetime.datetime(2007, 12, 31),
    ),
)
