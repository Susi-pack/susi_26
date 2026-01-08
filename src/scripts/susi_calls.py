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

# ***************** local call for SUSI*****************************************************
app_settings = AppSettings()

wdata = golden_test.PARAMETERS.weather_parameters.FMI_weather_filepath

mottifile = golden_test.PARAMETERS.motti_file_parameters


start_date = golden_test.PARAMETERS.simulation_config.start_date
end_date = golden_test.PARAMETERS.simulation_config.end_date
start_yr = start_date.year
end_yr = end_date.year


sarkaSim = golden_test.PARAMETERS.simulation_config.L
n = golden_test.PARAMETERS.simulation_config.n


site = "develop_scens"  # name of the parameter set in get_susi_para

forc = read_FMI_weather(0, start_date, end_date, sourcefile=wdata)  # read weather input

cpara = golden_test.PARAMETERS.canopy_parameters

org_para, spara, outpara, photopara = get_susi_para(
    wlocation="undefined",
    peat=site,
    folderName=app_settings.output_folder,
    hdomSim=None,
    ageSim=golden_test.PARAMETERS.simulation_config.age,
    sarkaSim=sarkaSim,
    sfc=golden_test.PARAMETERS.simulation_config.sfc,
    n=n,
)

spara["cutting_yr"] = (
    2004  # cutting year, not used if year is outside the simulation period
)
spara["drain_age"] = 100.0  # time since drainage, yrs
mass_mor = (
    1.616 * np.log(spara["drain_age"]) - 1.409
)  # Pitkänen et al. 2012 Forest Ecology and Management 284 (2012) 100–106

if np.median(golden_test.PARAMETERS.simulation_config.sfc) > 4:
    spara["peat type"] = [
        "S",
        "S",
        "S",
        "S",
        "S",
        "S",
        "S",
        "S",
    ]  # Peat type 'S' if Sphagnum, 'A' if woody or Carex-peat
    spara["peat type bottom"] = ["A"]
    spara["vonP top"] = [2, 5, 5, 5, 6, 6, 7, 7]  # Degree of decomposition
    spara["anisotropy"] = 10  # Anisotropy of peat hydraulic conductivity
    spara["rho_mor"] = 80.0  # bulk density of mor layer kg m-3
else:
    spara["vonP top"] = [2, 5, 5, 5, 6, 6, 7, 7]
    spara["anisotropy"] = 10
    spara["rho_mor"] = 90.0

spara["h_mor"] = mass_mor / spara["rho_mor"]

spara["ditch depth west"] = [
    -0.5
]  # ditch depth at the beginning of simulation m, if given several values SUSI calculates scenarios for each ditch depth
spara["ditch depth east"] = [-0.5]
spara["ditch depth 20y west"] = [-0.5]  # Ditch depth after 20 yrs, m, negative down
spara["ditch depth 20y east"] = [-0.5]  # Ditch depth after 20 yrs, m, negative down
spara["scenario name"] = [
    "D60"
]  # Scanario names, equal nmber of names than ditch depth scenarios
# spara['enable_peatmiddle'] = False,
# spara['enable_peatbottom'] = False

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
    ageSim=golden_test.PARAMETERS.simulation_config.age,
    sfc=golden_test.PARAMETERS.simulation_config.sfc,
)  # Run susi
