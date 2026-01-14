# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""

# THIS files
from susi.io.execution_config import SimulationParams
from susi.core.susi_utils import read_FMI_weather
from susi.core.susi_main import Susi
from inputs.parameters import golden_test
from susi.io.metadata_model import SimulationMetaData

# ***************** local call for SUSI*****************************************************

# read weather input
forc = read_FMI_weather(
    ID=0,
    start_date=golden_test.PARAMETERS.simulation_config.start_date,
    end_date=golden_test.PARAMETERS.simulation_config.end_date,
    sourcefile=golden_test.PARAMETERS.weather_parameters.FMI_weather_filepath,
)
# Specifies all parameters needed for a single run
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(),
    susi_params=golden_test.PARAMETERS,
)

# Initiate susi class
susi = Susi(
    simulation_parameters=simulation_parameters,
    weather_forcing=forc,
)

# Create output folder where results go
susi.create_output_folder()

# Run simulation
susi.run()

# Save stuff
susi.write_params_and_metadata()
