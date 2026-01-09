# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""

# THIS files
import numpy as np
import datetime
from pathlib import Path

from scipy.optimize import golden

from susi.core.susi_utils import read_FMI_weather
from inputs.susi_para import get_susi_para
from susi.core.susi_main import Susi
from susi.io.app_settings import AppSettings
from inputs.parameters import golden_test
from susi.io.susi_parameter_model import PeatTypes

# ***************** local call for SUSI*****************************************************
app_settings = AppSettings()

wdata = golden_test.PARAMETERS.weather_parameters.FMI_weather_filepath

mottifile = golden_test.PARAMETERS.motti_file_parameters


start_date = golden_test.PARAMETERS.simulation_config.start_date
end_date = golden_test.PARAMETERS.simulation_config.end_date
start_yr = start_date.year
end_yr = end_date.year


sarkaSim = golden_test.PARAMETERS.site_parameters.L
n = golden_test.PARAMETERS.site_parameters.n


site = "develop_scens"  # name of the parameter set in get_susi_para

forc = read_FMI_weather(0, start_date, end_date, sourcefile=wdata)  # read weather input

cpara = golden_test.PARAMETERS.canopy_parameters

org_para = golden_test.PARAMETERS.organic_layer_parameters

outpara = golden_test.PARAMETERS.output_parameters

photopara = golden_test.PARAMETERS.photo_parameters

spara = golden_test.PARAMETERS.site_parameters

susi = Susi()  # Initaiate susi class

susi.run_susi(
    forc,
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
    folderName=app_settings.output_folder,
    ageSim=golden_test.PARAMETERS.site_parameters.age,
    sfc=golden_test.PARAMETERS.site_parameters.sfc,
)  # Run susi
