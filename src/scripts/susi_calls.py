# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""

# THIS files
from susi.core.susi_utils import read_FMI_weather
from susi.core.susi_main import Susi
from susi.io.app_settings import AppSettings
from inputs.parameters import golden_test

# ***************** local call for SUSI*****************************************************
app_settings = AppSettings()


# read weather input
forc = read_FMI_weather(
    ID=0,
    start_date=golden_test.PARAMETERS.simulation_config.start_date,
    end_date=golden_test.PARAMETERS.simulation_config.end_date,
    sourcefile=golden_test.PARAMETERS.weather_parameters.FMI_weather_filepath,
)


susi = Susi()  # Initaiate susi class

susi.run_susi(
    forc=forc,
    parameters=golden_test.PARAMETERS,
)  # Run susi
